import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass
from modules import GroupNorm

@dataclass
class DiscriminatorConfig:
    latent_in_channels : int = 64
    image_channels : int = 3    
    n_layers : int = 3

    disc_kernel_size : int = 4
    disc_stride : int = 2
    disc_padding : int = 1

    disc_last_layer_stride : int = 1


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
    
    def forward (self, X):
        return self.model(X)
            

    


