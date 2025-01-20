

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet_encoder import ResNetEncoder, ResNetEncoderConfig
from quantizer import Quantizer, QuantizerConfig
from resnet_decoder import ResNetDecoder, ResNetDecoderConfig

from dataclasses import dataclass

import inspect


class VQGan (nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = ResNetEncoder (ResNetEncoderConfig)
        
        self.pre_quant_conv = nn.Conv2d (ResNetEncoderConfig.latent_channels, ResNetEncoderConfig.latent_channels, kernel_size=1, stride=1, padding=0)
        
        self.quantizer = Quantizer (QuantizerConfig)

        self.post_quant_conv = nn.Conv2d (ResNetDecoderConfig.latent_channels, ResNetDecoderConfig.latent_channels, kernel_size=1, stride=1, padding=0)

        self.decoder = ResNetDecoder (ResNetDecoderConfig)

        self.kaiming_init_gain = 1.0

        self.apply (self._init_weights)

    def _init_weights (self, module):
        if isinstance (module, nn.Conv2d):
            fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
            std = (self.kaiming_init_gain / fan_in)**0.5

            if hasattr (module, "SKIP_CONNECTION_SCALE_INIT"):
                std *= (len(self.encoder.block) + 3 + (len(self.decoder.block) / 2) + 3) ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance (module, nn.Embedding):
            std = (QuantizerConfig.n_embd)**-0.5
            torch.nn.init.normal_ (module.weight, mean=0.0, std=std)
        

    def forward (self, x):
        ze = self.encoder (x)

        pre_quant_activation = self.pre_quant_conv (ze)

        vq_loss, encoding_indices, zq  = self.quantizer (pre_quant_activation)

        post_quant_activation = self.post_quant_conv (zq)

        image = self.decoder (post_quant_activation)

        return vq_loss, encoding_indices, image


    def compute_lambda (self, perceptual_rec_loss, gan_loss, device):
        lambda_factor = None

        delta = 1e-6
        last_layer = self.decoder.to_image_conv.block[-1]
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
        # start with all parameters that require gradients
        param_dict = {pn:p for pn, p in self.named_parameters()}
        param_dict = {pn:p for pn, p in param_dict.items() if p.requires_grad}


        # create optim groups. Any parameter that is 2D will be weight decayed, otherwise no.
        # i.e. all tensors in matmul + embeddings decay, all biases and layer norms don't
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

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device
        print (f"using fused AdamW:{use_fused}")

        # kernel fusion for AdamW update instead of iterating over all the tensors to step which is a lot slower.
        optimizer = torch.optim.AdamW (optim_groups, lr=learning_rate, betas=(0.9,0.95), eps=1e-8, fused=use_fused)
        return optimizer
