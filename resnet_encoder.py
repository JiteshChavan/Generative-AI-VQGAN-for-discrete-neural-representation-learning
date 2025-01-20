import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from modules import ResNetResBlock, ResNetResBlockConfig
from modules import SelfAttention, SelfAttentionConfig


@dataclass
class ResNetEncoderConfig:
    image_channels : int = 3
    # scale this to increaes input resolution compatibility
    # change channel map as well
    latent_channels : int = 1024

    in_resolution : int = 256
    n_compressions : int = 4

    # res_map
    # 256 -> 128 -> 64 -> 32 -> 16
    # 256 3
    # 256 - 64 -> 128 -> 64
    # 128- 128 -> 64 128
    # 64 256 -> 32 256
    # 32 512 -> 16 -> 512
    channel_map = [64, 128, 256, 512]

    # TODO: CALCULATE AND CHANGE LATER


    #skip_additions = len(model.encoder.block) + 1 for bottleneck + 1 for attention + 1 for bottleneckfeedforward


class ResNetEncoder (nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.config = config

        self.block = nn.ModuleList ()

        in_channels = config.image_channels
        for stage in config.channel_map:
            if stage == 64:
                num_groups = 8
            elif stage == 128:
                num_groups = 16
            else:
                num_groups = 32

            self.block.append (ResNetResBlock (in_channels, stage, ResNetResBlockConfig, num_groups=num_groups))
            in_channels = stage
        

        self.down_sample_max_pool = nn.MaxPool2d (kernel_size=2, stride=2)

        # encoder bottleneck, by default 32 groups in the group norm
        self.bottleneck = ResNetResBlock (config.channel_map[-1], 2* config.channel_map[-1], ResNetResBlockConfig)

        # bottleneck self attention
        # default groups in pre layer group norm = 32
        self.bottleneck_attention = SelfAttention (config.latent_channels, SelfAttentionConfig)

        # bottleneck feed forward
        self.bottleneck_feedforward = ResNetResBlock (config.latent_channels, config.latent_channels, ResNetResBlockConfig)

    def forward (self, x):

        activation = x
        for i in range (len(self.block)):
            activation = self.block[i](activation)
            activation = self.down_sample_max_pool (activation)
        
        # 1024 channels
        activation = self.bottleneck (activation) # 16, 1024
        activation = self.bottleneck_attention (activation)
        # conv norm gelu -> conv norm gelu
        activation = self.bottleneck_feedforward (activation) # 16 , 1024

        return activation
        
