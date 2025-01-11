import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from modules import GroupNorm

import inspect

@dataclass
class DiscriminatorConfig:
    latent_in_channels : int = 64
    image_channels : int = 3    
    n_layers : int = 3

    disc_kernel_size : int = 4
    disc_stride : int = 2
    disc_padding : int = 1

    disc_last_layer_stride : int = 1

    kaiming_init_gain : float = 1.0


class Discriminator (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        layers = []
        self.in_conv = nn.Conv2d (config.image_channels, config.latent_in_channels, kernel_size=config.disc_kernel_size, stride=config.disc_stride, padding=config.disc_padding)
        layers.append (self.in_conv)

        # channels map
        # res map
        # 3     64  128 256 512 1  
        # 256   128 64  32  31  30
        in_channels = config.latent_in_channels
        out_channels = 2 * in_channels

        LAST_LAYER = config.n_layers - 1
        for i in range (config.n_layers):
            if i == LAST_LAYER:
                stride = 1
            else:
                stride = config.disc_stride
            layers.append (nn.Conv2d(in_channels, out_channels, kernel_size=config.disc_kernel_size, stride=stride, padding=config.disc_padding))
            layers.append (GroupNorm (out_channels))
            layers.append (nn.LeakyReLU(0.1, inplace=True))
            print (f"in:{in_channels}, out:{out_channels}, stride:{stride}")
            in_channels = out_channels
            out_channels = 2 * in_channels
        
        layers.append (nn.Conv2d(in_channels, 1, kernel_size=config.disc_kernel_size, stride=config.disc_last_layer_stride, padding=config.disc_padding))

        self.model = nn.Sequential(*layers)

        # kaiming init sorcery
        self.apply(self._init_weights)
    
    def forward (self, X):
        return self.model(X)
    
    def _init_weights (self, module):
       if isinstance (module, nn.Conv2d):            
           # u = 0, calculate sigma such that weights should be initialized from that N(u, sigma) according to Kaiming init
           fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
           std = (self.kaiming_init_gain / fan_in) ** 0.5       
       nn.init.normal_ (module.weight, mean=0.0, std=std) # optional generator can be passed for debugging


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

        print (f"optimizer_configuration for DISCRIMINATOR model:\n num decayed parameter tensors:{len(decay_params)} with {num_decay_params} parameters")
        print (f"num non-decayed parameter tensors:{len(no_decay_params)} with {num_no_decay_params} parameters")

        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and 'cuda' in device # or device == 'cuda'
        print (f"using fused AdamW:{use_fused}")
        # kernel fusion for AdamW update instead of iterating over all the tensors to step which is a lot slower.
        optimizer = torch.optim.AdamW (optim_groups, learning_rate, betas=(0.9,0.95), eps=1e-8, use_fused=use_fused)
        return optimizer
            

    


