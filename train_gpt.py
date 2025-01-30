import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from gpt import GPT, GPTConfig

from dataclasses import dataclass
from torch.utils.tensorboard import SummaryWriter
# -------------------------------------------------------------------------------------------------------------
# Training hyperparameters and Macros

START_TOKEN = 8192
vqgan_checkpoint_path = "./vqgan checkpoints/model_100000.pt"
inference_path = "./inferences"

context_completion_shard_path = "./shardifiedLatents/context_render_shard.npy"
# path where half context completions will be saved
half_context_inference_path = "./half_context"
assert os.path.exists (vqgan_checkpoint_path)


total_batch_size = 64 
# The setting of B is purely optimization performance kind of setting, so in any case you should be getting the same answers
# upto like a floating point error, because the gradient accumulation kicks in and can handle everything serially! (as necessary)
# Because the real batch size is 2**19 either way HAHAHA!
B = 4 # micro batch size
T = 256 # sequence length

# create a directory to log and write checkpoints
log_dir = "logs"

# mutate this for loading checkpoints or starting fresh runs
fresh_run = False
gpt_checkpoint_file_path = "./logs/model_24000.pt"

steps_per_val = 50
steps_per_checkpoint = 1000
steps_per_inference = 250

# src path to shards of sequences of codebook vector indices that represent image in latent space
# (to learn to model such sequences in auto regressive fashion)
# pass image dataset through shardify.py and then through image_quantizer.py
@dataclass
class DataLoaderConfig:
    data_root : str = "./ShardifiedLatents"

# --------------------------------------------------------------------------------------------------------------
def load_tokens (filename):
    npt = np.load (filename)
    npt = npt.astype(np.int32) # Introspective change
    ptt = torch.tensor (npt, dtype=torch.long)
    return ptt

# load encoding indices from shardified latents
def load_indices (filename, for_inference=True):
    npt = np.load (filename)
    npt = npt.astype (np.int32)
    ptt = torch.tensor (npt, dtype=torch.long)
    if for_inference:
        ptt = ptt[:,1:]
    return ptt

# ----------------------------------------------------------------------------------------------------------------



class DataLoaderLite:
    def __init__(self, process_rank, num_processes, split, B, T):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        assert split in {"train", "val"}
        self.data_root = DataLoaderConfig.data_root
        
        shards = [s for s in os.listdir(self.data_root) if split in s]
        shards = sorted(shards)
        shards = [os.path.join(self.data_root, s) for s in shards]
        self.shards = shards
        assert len(shards) > 0, f"No shards found for split {split}"
        if master_process:
            print (f"found {len(shards)} for split {split}")
        self.reset()
    
    def reset (self):
        self.current_shard = 0
        
        # 800, 257
        # current row
        self.current_position = self.B * self.process_rank
        
        self.tokens = load_tokens(self.shards[self.current_shard])
    
    def next_batch (self):
        buf = self.tokens[self.current_position: (self.current_position+self.B)]

        idx = buf [:, :-1] # inputs
        targets = buf [:, 1:] # targets
        # advance the position in the shard for the current process
        self.current_position += self.B * self.num_processes
        # if loading the next batch would be out of bounds, reset
        if (self.current_position+self.B > len(self.tokens)):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.current_position = self.B * self.process_rank
            self.tokens = load_tokens (self.shards[self.current_shard])
        
        return idx, targets

# -----------------------------------------------------------------------------------------------------------------------------------------
# simple launch:
# python train_gpt2.py
# DDP launch for e.g. 8 GPUs:
# torchrun --standalone --nproc_per_node=1 train_gpt.py

# run the training loop
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import os

# setup DDP (distributed data parallel).
# torchrun command sets the env variables RANK, LOCAL_RANK, WORLD_SIZE
ddp = int (os.environ.get ('RANK', -1)) != -1 # is this a ddp run?
if ddp:
    # use of DDP atm demands cuda, we set the device appropriately according to rank
    assert torch.cuda.is_available(), "for now I think we need CUDA for DDP"
    init_process_group(backend='nccl')
    ddp_rank = int (os.environ['RANK'])
    ddp_local_rank = int (os.environ['LOCAL_RANK'])
    ddp_world_size = int (os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    torch.cuda.set_device (device)
    master_process = ddp_rank == 0 # this process will do the logging checkpoining etc
else:
    # vanilla, non-DDP run
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    # attempt to autodetect device
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    print (f"using device: {device}")
    

torch.manual_seed(1337)
if torch.cuda.is_available():
    torch.cuda.manual_seed(1337)


assert (total_batch_size % B * ddp_world_size == 0), f"Make sure total batch size is divisible by (B * ddp_world_size)"
grad_accum_steps = total_batch_size // (ddp_world_size * B)

if master_process:
    print (f"Desired total batch size : {total_batch_size}")
    print (f"Calculated gradient accumulation steps : {grad_accum_steps}")


train_loader = DataLoaderLite (ddp_rank, ddp_world_size, "train", B, T) # T is not required in dataloader since information is encapsulated in images
# it does not span across multiple images (rows of loaded tensor)
val_loader = DataLoaderLite (ddp_rank, ddp_world_size, "val", B, T)

torch.set_float32_matmul_precision ("high")

# Create model
# 8 exact same GPT models are created on 8 processes, because the seeds are fixed
# TODO: Refactor this jank.

gpt = GPT (GPTConfig(vocab_size=8200))
gpt.to(device)

# setup quantizer and decoder from VQGAN model for inference
from resnet_vqgan import VQGan
from quantizer import QuantizerConfig
from data_utils import DataUtils, Data_Utils_Config
vqgan = VQGan()
vqgan.eval()
vqgan.to(device)

if ddp:
    # forward is unchanged, backward is mostly unchanged except there is overlap between computation and communication of gradients
    # while the backward pass is still going on, to average the gradients from all processes
    # we're tacking on this average as we will see in a bit
    gpt = DDP (gpt, device_ids=[ddp_local_rank])
    vqgan = DDP (vqgan, device_ids=[ddp_local_rank])
raw_gpt = gpt.module if ddp else gpt
raw_vqgan = vqgan.module if ddp else vqgan
# load checkpoints for VQGAN model
vqgan_checkpoint = torch.load (vqgan_checkpoint_path)
raw_vqgan.load_state_dict (vqgan_checkpoint['vqgan_model'])
print (f"VQGAN checkpoint weights loaded.")

# shard util for inference
shard_util = DataUtils (Data_Utils_Config)


num_epochs = 1600
max_lr = 6e-4
min_lr = max_lr * 0.1
warmup_steps = 57
max_steps = num_epochs * 57 # 22,800 max steps
import math
def get_lr (it):
    # 1) linear warmup if step in warmup region
    if it < warmup_steps:
        return ((it+1) / warmup_steps ) * max_lr

    # 2) post cosine decay region (formality) this case will never trigger in our case since we stop at max steps
    if it > max_steps:
        return min_lr
    
    # 3) else if in cosine decay regions
    decay_ratio = (it - warmup_steps) / (max_steps - warmup_steps)
    assert 0 <= decay_ratio <= 1
    cosine_factor =  0.5 * (1.0 + math.cos(math.pi * decay_ratio))

    return min_lr + cosine_factor * (max_lr - min_lr)

# optimize!
# First try to crush and overfit a small batch

optimizer = raw_gpt.configure_optimizers (0.1, learning_rate=6e-4, device = device)


writer = SummaryWriter(log_dir=log_dir)
if master_process:
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join (log_dir, "log.txt")
    with open (log_file, "w") as f: # open for writing to clear the file
        pass


if fresh_run:
    # run
    start_step = 0    
    checkpoint = {}
    if master_process:
        print (f"Starting a fresh run")
else:
    
    assert os.path.exists (gpt_checkpoint_file_path)
    if master_process:
        print (f"Resuming from checkpoint:{gpt_checkpoint_file_path}")

    # load the checkpoint file
    checkpoint = torch.load (gpt_checkpoint_file_path)

    # load from checkpoint dictionary
    start_step = checkpoint['step']
    optimizer.load_state_dict(checkpoint['optim'])
    raw_gpt.load_state_dict(checkpoint['model'])
    
    train_loader.current_shard = checkpoint['current_shard']
    train_loader.tokens = load_tokens(train_loader.shards[train_loader.current_shard])
    train_loader.current_position = checkpoint['current_pos_gpu0'] + train_loader.B * ddp_rank

    val_loader.current_shard = checkpoint['val_current_shard']
    val_loader.tokens = load_tokens (val_loader.shards[val_loader.current_shard])
    val_loader.current_position = checkpoint ['val_current_pos_gpu0'] + val_loader.B * ddp_rank


    if ddp:
        gpt = DDP (raw_gpt, device_ids=[ddp_local_rank])
    raw_gpt = gpt.module if ddp else gpt # Same as raw_gpt if python run (not torchrun)

    if master_process:
        print (f"Resuming from position: {train_loader.current_position} in shard:{train_loader.current_shard}")
        print (f"from step:{start_step}")


use_compile = True
if use_compile:                 # turned off torch compile interferes with inference
    torch.compile (gpt)
    torch.compile (vqgan)
    print (f"using torch compile")

for step in range (start_step, max_steps):
    t0 = time.time()
    last_step = (step == max_steps - 1)

    # once in a while evaluate our validation loss
    if step % steps_per_val == 0 or last_step:
        val_loss_steps = 20
        val_loss_accum = 0.0
        gpt.eval()
        val_loader.reset()
        with torch.no_grad():
            for _ in range (val_loss_steps):
                idx, targets = val_loader.next_batch()
                idx, targets = idx.to(device), targets.to(device)

                with torch.autocast (device_type=device, dtype=torch.bfloat16):
                    # forward and get loss and logits
                    logits, loss = gpt (idx, targets)

                loss = loss / val_loss_steps
                val_loss_accum += loss.detach() # detaches tensor from computation graph returns reference to the same underlying memory
            
            if ddp:
                dist.all_reduce(val_loss_accum, op=dist.ReduceOp.AVG)
            if master_process:
                print (f"Validation loss : {val_loss_accum.item():.4f}")
                writer.add_scalars("Loss", {"val_loss": val_loss_accum.item()}, step)
                with open(log_file, "a") as f:
                    f.write (f"{step} val {val_loss_accum.item():.4f}\n")
        
        # checkpoint 
        if step > 0 and (step % steps_per_checkpoint == 0 or step == last_step):
            checkpoint_path = os.path.join (log_dir, f"model_{step:05d}.pt")
            checkpoint = {}
            checkpoint['step'] = step
            checkpoint['model'] = raw_gpt.state_dict()
            checkpoint['optim'] = optimizer.state_dict()
            checkpoint['current_shard'] = train_loader.current_shard
            checkpoint['current_pos_gpu0'] = train_loader.current_position
            checkpoint['val_current_shard'] = val_loader.current_shard
            checkpoint['val_current_pos_gpu0'] = val_loader.current_position

            torch.save (checkpoint, checkpoint_path)
            print (f"saving model checkpoint : model_{step:05d}.pt")
    
    # once in a while generate images from model, except at step 0 which is noise
    if (step % steps_per_inference == 0 or last_step):
        inference_batches = 10
        num_return_sequences = 4
        MAX_LENGTH = 257
        sample_rng = torch.Generator(device=device)
        for _ in range(inference_batches):
            # 1) blind completion
            sample_rng.manual_seed(step + _)
            
            #sample_rng = torch.Generator (device=device)
            #sample_rng.manual_seed(42 + step + _)

            tokens = torch.tensor([START_TOKEN]).unsqueeze(0).repeat(num_return_sequences, 1)
            xgen = tokens.to(device=device)


            # native resolution renders (not super resolution yet)
            while (xgen.size(1) < MAX_LENGTH):
                with torch.no_grad():

                    with torch.autocast (device_type=device, dtype=torch.bfloat16):
                        logits, loss = gpt (xgen)
                    # one forward is only one example give me what comes after it, not training so can drop the T examples packed into a single forward
                    logits = logits [:, -1, :] # all batches (B), last token, (vocab_size)
                    probs = F.softmax (logits, dim=-1) # B, VOCAB SIZE
                    #  B,T -> B, T, C -> ->LM HEAD -> B, T, VOCABSIZE

                    # do top-k sampling
                    # top-k probs here becomes (4, 50) top-k indices is becomes (4,50)
                    topk_probs, topk_indices = torch.topk (probs, k=50, dim=-1) # (B, 50)

                    # select a token from the top-k probabilities
                    # note: Multinomial does not demand the input to sum to 1, i.e it wont mutate your probability distribution of top 50 probs
                    ix = torch.multinomial (topk_probs, num_samples=1, generator=sample_rng) # (B, 1)

                    x_col = torch.gather (topk_indices, -1, ix)
                    if torch.all((x_col >= 0) & (x_col < raw_vqgan.quantizer.codebook.weight.size(0))):
                        xgen = torch.cat((xgen, x_col), dim=-1)

            # drop the start token from every example
            xgen = xgen[:, 1:] # B=num_, 256
            
            latent_vectors = raw_vqgan.quantizer.codebook(xgen).view(num_return_sequences, QuantizerConfig.latent_resolution, QuantizerConfig.latent_resolution, QuantizerConfig.n_embd) # B, 16,16, 1024
            # convs expect B C H W as input in PyTorch
            latent_vectors = latent_vectors.permute(0, 3, 1, 2).contiguous()
            # dont forget to forward exactly as defined in the model
            post_quant_activation = raw_vqgan.post_quant_conv (latent_vectors)
            image_tensor = raw_vqgan.decoder (post_quant_activation)

            current_step_path = f"{inference_path}/{step}"
            shard_util.tensor_to_image (image_tensor, current_step_path, "neural")
            del image_tensor, post_quant_activation, latent_vectors, xgen, logits, probs
            torch.cuda.empty_cache( )
        
        
        # 2) 50% context
        with torch.no_grad():
            half_context_latent_tensor = load_indices (context_completion_shard_path, for_inference=False) # dont drop the start token
            # load first 8 rows out of 16 from latent, including the start token
            tokens = half_context_latent_tensor [:12, :129] # theres just 4 images in that shard eitherway
            xgen = tokens.to(device)

            while (xgen.size(1) < MAX_LENGTH):
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, loss = gpt (xgen) # (B, T, VOCABSIZE)

                logits = logits[:,-1, :]
                probs = F.softmax (logits, dim=-1) # (B, VOCABSIZE)

                topk_probs, topk_indices = torch.topk (probs, 50, dim=-1) # (B, 50)

                ix = torch.multinomial (topk_probs, num_samples=1, generator=sample_rng)
                x_col = torch.gather (topk_indices, -1, ix)
                
                # & elementwise logical and
                if torch.all((x_col >= 0) & (x_col < raw_vqgan.quantizer.codebook.weight.size(0))):
                    xgen = torch.cat ((xgen, x_col), dim=-1)
            
            # drop the start token for inference
            xgen = xgen [:, 1:] # (B, 256)

            # B, 16, 16, 1024
            latent_vectors = raw_vqgan.quantizer.codebook (xgen).view (xgen.size(0), QuantizerConfig.latent_resolution, QuantizerConfig.latent_resolution, QuantizerConfig.n_embd)
            # prepare for passing to decoder (convs) (make it B C H W)
            latent_vectors = latent_vectors.permute (0, 3, 1, 2).contiguous()
            
            # dont forget to forward exactly as defined in the model
            post_quant_activation = raw_vqgan.post_quant_conv (latent_vectors)
            image_tensor = raw_vqgan.decoder (post_quant_activation)
            render_path = f"{inference_path}/{half_context_inference_path}/{step}"
            shard_util.tensor_to_image (image_tensor, render_path, "neural")
        
    # do one step of the optimization
    gpt.train()
    optimizer.zero_grad()
    loss_accum = 0.0
    for micro_step in range (grad_accum_steps):

        idx, targets = train_loader.next_batch()
        idx, targets = idx.to(device), targets.to(device)

        with torch.autocast (device_type=device, dtype=torch.bfloat16):
            logits, loss = gpt (idx, targets) # logits (B, T, VS)
        
        loss = loss / grad_accum_steps
        loss_accum += loss.detach()

        if ddp:
            gpt.require_backward_grad_sync = (micro_step == (grad_accum_steps - 1))

        loss.backward()

    if ddp:
        dist.all_reduce (val_loss_accum, op=dist.ReduceOp.AVG)
    
    norm = torch.nn.utils.clip_grad_norm_ (gpt.parameters(), 1.0)

    lr = get_lr (step)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    optimizer.step()
    torch.cuda.empty_cache()
    torch.cuda.synchronize ()
    t1 = time.time()
    dt = (t1 - t0) * 1000 # time in miliseconds
    tokens_per_sec = (train_loader.B * train_loader.T * ddp_world_size * grad_accum_steps) / (t1 - t0)
    if master_process:
        print (f"step: {step} | loss: {loss_accum.item():.4f} | lr: {lr:.4e} | norm : {norm:.4f} | dt: {dt:.2f}ms | tok/sec: {tokens_per_sec}")
        writer.add_scalars("Loss", {"train_loss": loss_accum.item()}, step)
        with open (log_file, "a") as f:
            f.write (f"{step} train {loss_accum.item():.6f}\n")

if ddp:
    destroy_process_group()
