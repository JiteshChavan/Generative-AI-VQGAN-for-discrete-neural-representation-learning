

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoder import Encoder, EncoderConfig
from quantizer import Quantizer, QuantizerConfig
from decoder import Decoder, DecoderConfig

from dataclasses import dataclass

import inspect

@dataclass
class VQGanConfig:
    pass

class VQGan (nn.Module):
    def __init__(self, vqganConfig):
        super().__init__()
        self.encoder = Encoder (EncoderConfig)
        
        self.pre_quant_conv = nn.Conv2d (EncoderConfig.latent_channels, EncoderConfig.latent_channels, kernel_size=1, stride=1, padding=0)
        
        self.quantizer = Quantizer (QuantizerConfig)

        self.post_quant_conv = nn.Conv2d (2*DecoderConfig.latent_channels, DecoderConfig.latent_channels, kernel_size=1, stride=1, padding=0)

        self.decoder = Decoder (DecoderConfig)



        # delete later if not needed
        self.vqganConfig = vqganConfig
    

    def forward (self, x):
        skips, ze = self.encoder (x)

        pre_quant_activation = self.pre_quant_conv (ze)

        vq_loss, encoding_indices, zq  = self.quantizer (pre_quant_activation)

        assert (pre_quant_activation.size(1) == zq.size(1)) # assert that channels match
        zq = torch.cat ((pre_quant_activation, zq), dim=1) # concatenate along channels, classical u-net skip connection

        zq = self.post_quant_conv (zq)

        image = self.decoder (skips, zq)

        return vq_loss, encoding_indices, image


    def compute_lambda (self, perceptual_rec_loss, gan_loss, device):
        lambda_factor = None

        delta = 1e-6
        last_layer = self.decoder.to_image_conv
        last_layer_weight = last_layer.weight

        with torch.autocast(device_type=device, dtype=torch.bfloat16):
            perceptual_rec_loss_grad = torch.autograd.grad (perceptual_rec_loss, last_layer_weight, retain_graph=True)[0] # to extract the value from (a,) tuple
            gan_loss_grad = torch.autograd.grad (gan_loss, last_layer_weight, retain_graph=True)[0] # to extract the value from (a,) tuple

        lambda_factor = torch.norm(perceptual_rec_loss_grad) / (torch.norm(gan_loss_grad) + delta)
        lambda_factor = torch.clamp (lambda_factor, 0, 1e4).detach()

        return 0.8 * lambda_factor
    

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

        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}

        decay_params = [p for pn,p in param_dict.items() if p.dim() >= 2]
        no_decay_params = [p for pn,p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {"params" : decay_params, "weight_decay" : weight_decay},
            {"params" : no_decay_params, "weight_decay" : 0.0}
        ]

        num_decay_params = sum (p.numel() for p in decay_params)
        num_nodecay_params = sum (p.numel() for p in no_decay_params)

        print (f"optimizer_configuration for VQGAN model:\n num decayed parameter tensors:{len(decay_params)} with {num_decay_params} parameters")
        print (f"num non-decayed parameter tensors:{len(no_decay_params)} with {num_nodecay_params} parameters")

        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print (f"using fused AdamW:{use_fused}")

        optimizer = torch.optim.AdamW (optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer
