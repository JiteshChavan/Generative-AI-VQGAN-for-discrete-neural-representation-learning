
from dataclasses import dataclass

from encoder import Encoder, EncoderConfig
from decoder import Decoder, DecoderConfig

import torch
import torch.nn as nn
import torch.nn.functional as F

import inspect



class Unet(nn.Module):
    def __init__(self, encoder_config, decoder_config):
        super().__init__()
        self.encoder = Encoder (encoder_config)
        self.decoder = Decoder (decoder_config)


        # use later when writing final module
        self.kaiming_init_gain = 1.0
        # formality
        self.encoder_config = encoder_config
        self.decoder_config = decoder_config
    
    def forward (self, X):
        # X (B, C, H, W)

        # encode images
        skips, ze = self.encoder (X) # (B, latent_dim=1024, H=16, W=16)
        images = self.decoder (ze, skips) # (B, C, H, W)
        del skips, ze
        return images
    
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
        optimizer = torch.optim.AdamW (optim_groups, learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer

