import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from quantizer import Quantizer
from decoder import Decoder
from dataclasses import dataclass

import inspect


class VQGan (nn.Module):
    def __init__(self, encoder_config, quantizer_config, decoder_config):
        super().__init__()
        self.encoder = Encoder (encoder_config)
        
        # sort of a projection layer for isolation
        self.pre_quant_conv = nn.Conv2d (encoder_config.latent_dim, encoder_config.latent_dim, kernel_size=encoder_config.proj_kernel_size, stride=encoder_config.proj_stride, padding=encoder_config.proj_padding)
        
        self.quantizer = Quantizer (quantizer_config)

        # Same as the pre quantization conv
        self.post_quant_conv = nn.Conv2d (encoder_config.latent_dim, encoder_config.latent_dim, kernel_size=encoder_config.proj_kernel_size, stride=encoder_config.proj_stride, padding=encoder_config.proj_padding)

        self.decoder =  Decoder (decoder_config)

        # 1.0 because swish
        self.kaiming_init_gain = 1.0 # to compensate for non linearities during initialization of weights, refer kaiming init

        # formality
        self.encoder_config = encoder_config
        self.quantizer_config = quantizer_config
        self.decoder_config = decoder_config

        # init sorcery
        self.apply(self._init_weights)
    
    # deprecated comments, we are not hard coding or approximating it here
    # hard coding bad practice, wont scale with increasing fan_in like Xavier or Kaming init
    # but we will keep this because that is the GPT2 initialization per their source code
    # 0.02 is reasonably similar to 1/root(768 or 1024 or 1600 or 1280) for the gpt2 models
    def _init_weights (self, module):
        if isinstance (module, nn.Conv2d):            
            # u = 0, calculate sigma such that weights should be initialized from that N(u, sigma) according to Kaiming init
            fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
            std = (self.kaiming_init_gain / fan_in) ** 0.5
            
            if hasattr (module, 'SKIP_CONNECTION_SCALE_INIT'):
                n_skip_additions = self.encoder_config.n_compressions + self.encoder_config.n_post_compression_skip_connections \
                                    + self.decoder_config.n_expansions + self.decoder_config.n_pre_expansion_skip_connections
                std *= n_skip_additions ** -0.5
            nn.init.normal_ (module.weight, mean=0.0, std=std)
        elif isinstance (module, nn.Embedding):
            std = (self.quantizer_config.n_embd) ** -0.5
            nn.init.normal_ (module.weight, mean=0.0, std=std) # optional generator can be passed for debugging
    
    def forward (self, X):
        # X (B, C, H, W)

        # encode images
        ze = self.encoder (X) # (B, latent_dim=1024, H=16, W=16)

        ze = self.pre_quant_conv (ze) # (B, C, H, W)

        vq_loss, zq, encoding_indices = self.quantizer (ze) # (B, C, H, W)

        zq = self.post_quant_conv (zq) # (B, C, H, W)

        X_decoded = self.decoder (zq) # (B, C, H, W)

        return X_decoded, encoding_indices, vq_loss     

    def compute_lambda (self, perceptual_loss, gan_loss):
        lambda_factor = None

        delta = 1e-6
        last_layer = self.decoder.model[-1]
        last_layer_weight = last_layer.weight

        perceptual_loss_grad = torch.autograd.grad (perceptual_loss, last_layer_weight, retain_graph=True)[0] # to extract the value from (a,) tuple
        gan_loss_grad = torch.autograd.grad (gan_loss, last_layer_weight, retain_graph=True)[0] # to extract the value from (a,) tuple

        lambda_factor = torch.norm(perceptual_loss_grad) / (torch.norm(gan_loss_grad) + delta)
        lambda_factor = torch.clamp (lambda_factor, 0, 1e4).detach()
        return 0.8 * lambda_factor
    
    # weight adopt function
    # A function that turns off gan loss for first epoch
    # as advised in VQGAN paper


    # ---------------------------------------------------------------------------------------------------------------------------------
    # optimizer configurations
    # ---------------------------------------------------------------------------------------------------------------------------------
    # weight decay:
    # At initialization the weights are initialized so that all outcomes are equally likely,
    # now we want the weights to be optimized (pushed or pulled) so that likely outcomes are assigned higher probabilities
    # but at the same time, the we subject the weights to a force, regularizing force that kind of works against optimizing force
    # we decay the weights while stepping in the direction of gradients
    # sort of a force against the optimization that pulls down the weights and makes the network use more of the weights, distributing the task across multiple channels if you would,
    # to achieve the activations, basically regularization
    
    def configure_optimizers (self, weight_decay, learning_rate, device):
        # start with all parameters that require gradients
        param_dict = {pn:p for pn,p in self.named_parameters()}
        param_dict = {pn:p for pn,p in param_dict.items() if p.requires_grad}
        
        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all tensors in matmul + embeddings decay, all biases and layer norms don't
        decay_params = [p for pn, p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params' : decay_params, 'weight_decay' : weight_decay},
            {'params' : no_decay_params, 'weight_decay' : 0.0}
        ]

        num_decay_params = sum (p.numel () for p in decay_params)
        num_no_decay_params = sum (p.numel () for p in no_decay_params)

        print (f"optimizer_configuration for VQGAN model:\n num decayed parameter tensors:{len(decay_params)} with {num_decay_params} parameters")
        print (f"num non-decayed parameter tensors:{len(no_decay_params)} with {num_no_decay_params} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device # or device == 'cuda'
        print (f"using fused AdamW:{use_fused}")
        # kernel fusion for AdamW update instead of iterating over all the tensors to step which is a lot slower.
        optimizer = torch.optim.AdamW (optim_groups, learning_rate, betas=(0.9,0.95), eps=1e-8, use_fused=use_fused)
        return optimizer