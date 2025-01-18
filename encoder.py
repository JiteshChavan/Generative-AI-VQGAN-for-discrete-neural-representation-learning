import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import ConvBlock, ConvBlockConfig
from modules import SelfAttention, SelfAttentionConfig
from modules import ResidualBlock, ResBlockConfig

from dataclasses import dataclass


@dataclass
class EncoderConfig:
    
    image_channels : int = 3
    latent_channels : int = 1024

    in_resolution : int = 256

    channel_map = [64, 128, 256, 512]
    
    n_compressions : int = 4

    n_skip_additions : int = 2 # actual additions not skip concatenations

    kaiming_init_gain : float = 1.0 # not sure about gelu but GPT2 seems to have 1 gain


class Encoder (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.kaiming_init_gain = config.kaiming_init_gain
        self.block = nn.ModuleList ()
        in_channels = config.image_channels
        for stage in config.channel_map:
            self.block.append(ConvBlock(in_channels, stage, ConvBlockConfig))
            in_channels = stage
        
        self.max_pool = nn.MaxPool2d (kernel_size=2, stride=2)

        self.bottleneck = ConvBlock (config.channel_map[-1], 2*config.channel_map[-1], ConvBlockConfig)

        # gnarly, dont maintain state
        # self.skip_connections = []

        self.bottleneck_attention = SelfAttention (config.latent_channels, SelfAttentionConfig)
        self.bottleneck_feed_forward = ResidualBlock (config.latent_channels, ResBlockConfig)

        self.apply(self._init_weights)

    def _init_weights (self, module):
        if isinstance (module, nn.Conv2d):
            fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
            std = (self.kaiming_init_gain / fan_in)**-0.5

            if hasattr (module, 'SKIP_CONNECTION_SCALE_INIT'):
                std *= self.config.n_skip_additions ** -0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
            

    def forward (self, x):
        # stack these behind corressponding expansion stages (unet inductive bias)
        skip_connections = []
        for doubleConvs in self.block:
            x = doubleConvs(x)
            skip_connections.append(x)
            x = self.max_pool(x)
        x = self.bottleneck(x)
        x = self.bottleneck_attention (x)
        x = self.bottleneck_feed_forward (x)

        
        return skip_connections, x

