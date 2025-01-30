import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import time
import inspect


from dataclasses import dataclass
# -----------------------------------------------------------------------------

@dataclass
class GPTConfig:
    n_layers : int = 12 # number of layers
    n_embd : int = 768   # number of embedding dimensions
    n_head : int = 4    # numebr of attention heads
    vocab_size : int = 8192 # number of possible encoding indices over which we spit out conditional posteriors and chain them to approximate
                            # true distribution with the chain of resulting categorical softmaxes over vocab_size
    block_size : int = 256 # 16x16x1024 vectors flattened and arranged in raster scan order



class CausalSelfAttention (nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.config = config
        # key, query, value prohjections for all heads but in a batch
        self.c_attn = nn.Linear (config.n_embd, 3*config.n_embd)
        # output projection
        self.c_proj = nn.Linear (config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1

        self.n_embd = config.n_embd
        self.n_head = config.n_head

        # not really a bias more of a mask, although following the OpenAI/HF naming
        self.register_buffer ('bias', torch.tril(torch.ones(config.block_size, config.block_size))
                                .view (1,1, config.block_size, config.block_size))

    def forward (self, idx):
        B, T, C = idx.size()
        qkv = self.c_attn(idx) # (B, T, 3C)
        q, k, v = qkv.split (self.n_embd, dim=2) # each (B, T, C)
        k = k.view (B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view (B, T, self.n_head, C//self.n_head).transpose(1,2) # (B, nh, T, hs)
        v = v.view (B, T, self.n_head, C//self.n_head).transpose(1, 2) # (B, nh, T, hs)
        # can also use permute + contiguous 
        att = q @ k.transpose(2,3)

        # attention materializes the large (T,T) matrix for all the queries and keys dotproduct
        # scale by 1/root fanin of second matrix so that softmax doesnt devolve into one hot vectors for huge variance numbers
        #att = (q @ k.transpose(-2, -1)) * (1s.0 / math.sqrt (k.size(-1))) # (B, nh, T, T)
        #att = att.masked_fill (self.bias[:,:,:T,:T] == 0, float('-inf'))
        #att = F.softmax (att, dim=-1)
        #y = att @ v # (B, nh, T, T) @ (B, nH, T, Hs) -> (B, nH, T, Hs)

        # flash attention does not materialize the large (T,T) matrix for all the queries and keys
        # evaluates softmax in a streaming manner (online normalized calc for softmax by nvidia) and fuses matmul ops in the same kernel
        y = F.scaled_dot_product_attention (q, k, v, is_causal=True) # (B, nh, T, hs)

        y = y.transpose (1, 2).contiguous ().view (B, T, C)
        y = self.c_proj (y)
        return y


class MLP (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.c_fc = nn.Linear (config.n_embd, 4*config.n_embd)
        self.gelu = nn.GELU (approximate='tanh')
        self.c_proj = nn.Linear (4 * config.n_embd, config.n_embd)
        self.c_proj.NANO_GPT_SCALE_INIT = 1
    def forward (self, x):
        x = self.c_fc (x)
        x = self.gelu (x)
        x = self.c_proj (x)
        return x

class Block (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ln_1 = nn.LayerNorm (config.n_embd) 
        self.c_attn = CausalSelfAttention (config)
        
        self.ln_2 = nn.LayerNorm (config.n_embd)
        self.mlp =  MLP (config)
    def forward (self, x):
        x = x + self.c_attn(self.ln_1 (x)) # pre normalization
        x = x + self.mlp (self.ln_2 (x))        # it is preferable to have a clean residual stream all the way from supervision to the inputs
        return x                            # unlike attention paper where layer norms are inside the residual stream
    # Attention is a aggregation function, pooling function, weighted sum function, reduce operation. This is where tokens communicate and exchange information
    # Whereas MLP happens at every single token individually, there's no info being collected or exchanged between the tokens.
    # so the attention is the reduce and the MLP is the map, so what you end up with is the transformer ends up just being repeated application
    # of map-reduce if you wanna think about it that way. 
    # MLP is where the network thinks on the information pooled from other tokens individually
    # And every one of these blocks iteratively refines, representation inside the residual stream.
    
class GPT (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict (dict (
            wte = nn.Embedding (config.vocab_size, config.n_embd), 
            wpe = nn.Embedding (config.block_size, config.n_embd),
            h = nn.ModuleList ([Block(config) for _ in range (config.n_layers)]),
            ln_f = nn.LayerNorm (config.n_embd)
        ))
        self.lm_head = nn.Linear (config.n_embd, config.vocab_size, bias=False) # n_embd to vocab_size without bias

        # weight sharing
        # Weight tying shceme between embedding and pre softmax linear layer
        # Makes training efficient, otherwise we'd have 30% (50257*768 = 40M; 40/120 ~ 30% of parameters), because we don't have to train as many parameters
        # and it improves results by putting in the iductive bias that both these embeddings should share similarities between tokens.
        # SAVED A TON OF PARAMETERS!
        self.transformer.wte.weight = self.lm_head.weight

        # write weight init later
    
    def _init_weights (self, module):
        if isinstance (module, nn.Linear):
            fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
            std = (fan_in) ** -0.5
            if hasattr (module, 'NANO_GPT_SCALE_INIT'):
                std *= (2 * self.config.n_layers) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance (module, nn.Embedding):
            std = GPTConfig.n_emd ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)


    def forward (self, idx, targets=None):
        B, T = idx.size()
        assert T <= self.config.block_size, f"Can not forward a sequence of length {T}, block size is {self.config.block_size}"
        
        pos = torch.arange (0, T, dtype=torch.long, device=idx.device)
        pos_emb = self.transformer.wpe(pos)
        tok_emb = self.transformer.wte(idx)
        x = tok_emb + pos_emb
        # forward the blocks of transfomer
        for block in self.transformer.h:
            x = block (x)
        # forward the final norm layer and the classifier
        x = self.transformer.ln_f(x)
        logits = self.lm_head (x) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            # the reduction in F.cross_entropy is mean (be careful while accumulating gradients to match larger batch sizes)
            loss = F.cross_entropy (logits.view (B*T, self.config.vocab_size), targets.view (B*T)) # stretch out logits and targets for cross entropy loss

        return logits, loss
    

    # -------------------------------------------------------------------------------------------------
    def configure_optimizers (self, weight_decay, learning_rate, device):

        param_dict = {pn:p for pn, p in self.named_parameters ()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        num_decay_params = sum (p.numel() for p in decay_params)
        num_no_decay_params = sum (p.numel() for p in no_decay_params)

        optim_groups = [
            { "params" : decay_params, "weight_decay" : weight_decay},
            { "params" : no_decay_params, "weight_decay" : 0.0}
        ]

        print (f"num decay parameter tensors:{len(decay_params)} with {num_decay_params} parameters")
        print (f"num decay parameter tensors:{len(no_decay_params)} with {num_no_decay_params} parameters")

        fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print (f"Using fused AdamW : {use_fused}")
        optimizer = torch.optim.AdamW (optim_groups, lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, fused=use_fused)

        return optimizer
    
