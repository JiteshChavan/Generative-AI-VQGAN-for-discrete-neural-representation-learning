import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder
from quantizer import Quantizer
from decoder import Decoder
from dataclasses import dataclass

@dataclass
class VQGanConfig:
    pass


class VQGan (nn.Module):
    def __init__(self, encoder_config, quantizer_config, decoder_config, config):
        super().__init__()
        self.encoder = Encoder (encoder_config)
        
        # sort of a projection layer for isolation
        self.pre_quant_conv = nn.Conv2d (encoder_config.latent_dim, encoder_config.latent_dim, kernel_size=encoder_config.proj_kernel_size, stride=encoder_config.proj_stride, padding=encoder_config.proj_padding)
        
        self.quantizer = Quantizer (quantizer_config)

        # Same as the pre quantization conv
        self.post_quant_conv = nn.Conv2d (encoder_config.latent_dim, encoder_config.latent_dim, kernel_size=encoder_config.proj_kernel_size, stride=encoder_config.proj_stride, padding=encoder_config.proj_padding)

        self.decoder =  Decoder (decoder_config)
    
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


