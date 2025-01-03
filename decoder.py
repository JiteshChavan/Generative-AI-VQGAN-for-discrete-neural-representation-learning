import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

from modules import ResidualBlock, Swish, GroupNorm, UpSampleBlock, SelfAttention

@dataclass
class DecoderConfig:
    # general decode config
    latent_dim : int = 1024
    latent_res : int = 16
    image_res : int = 256
    n_expansions : int = 4 # (spatial resolution is expanded by 2^4)
    image_channels : int = 3
    penultimate_channels : int = 64

    # conv kernel config
    conv_kernel_size : int = 3
    conv_stride : int = 1
    conv_padding : int = 1

    # Res block config
    res_kernel_size : int = 3
    res_stride : int = 1
    res_padding : int = 1

    # Self Attention Config
    attention_resolution : int = 16 # latent resolution
    n_head : int = 4
    att_kernel_size : int = 1
    att_stride : int = 1
    att_padding : int = 0

    # proj config
    proj_kernel_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0

    # UpSample config
    up_sample_factor : int = 2
    up_sample_kernel_size : int = 3
    up_sample_stride : int = 1
    up_sample_padding : int = 1

    # res block channel up configs
    channel_up_kernel_size : int = 1
    channel_up_stride : int = 1
    channel_up_padding : int = 0


class Decoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        layers = []
        initial_conv = nn.Conv2d (self.config.latent_dim, self.config.latent_dim, kernel_size=self.config.conv_kernel_size, stride=self.config.conv_stride, padding=self.config.conv_padding)
        layers.append(initial_conv)
        layers.append(ResidualBlock (self.config.latent_dim, self.config.latent_dim, self.config))
        layers.append (SelfAttention(self.config.latent_dim, self.config))
        layers.append(ResidualBlock (self.config.latent_dim, self.config.latent_dim, self.config))

        # init setup for channels
        in_channels = self.config.latent_dim
        out_channels = in_channels // 2
        for i in range (self.config.n_expansions):
            if i == 0:
                out_channels = in_channels
            else:
                in_channels = out_channels
                out_channels = in_channels // 2
            
            # res and channel map
            #       0       1       2       3
            # dim   1024    512     256     128 -> collapse after loop to 3
            # res   32      64      128     256
            layers.append (ResidualBlock(in_channels, out_channels, self.config))
            layers.append (UpSampleBlock(out_channels, self.config, self.config.up_sample_factor))
        
        layers.append (GroupNorm(out_channels))
        layers.append(Swish())
        # Here res = 256, channels = 128 (antisymmetric to encoder)
        layers.append(nn.Conv2d (out_channels, self.config.penultimate_channels, kernel_size=self.config.conv_kernel_size, stride=self.config.conv_stride, padding=self.config.conv_padding))
        collapse_conv = nn.Conv2d (self.config.penultimate_channels, self.config.image_channels, kernel_size=self.config.conv_kernel_size, stride=self.config.conv_stride, padding=self.config.conv_padding)
        layers.append(collapse_conv)

        self.model = nn.Sequential (*layers)

    def forward (self, X):
        image = self.model(X)
        return image
        
        

        



