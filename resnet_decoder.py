import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

from modules import ResNetResBlock, ResNetResBlockConfig
from modules import UpSampleSkipResNetResBlockConfig

from modules import SelfAttention, SelfAttentionConfig
from modules import GroupNorm

@dataclass
class ResNetDecoderConfig:
    latent_channels : int = 1024
    image_channels : int = 3

    channel_map = [512, 256, 128, 64]
    n_expansions : int = 4

    # calculate n_skip additions
    # skip additions = len(self.block) / 2 + 1 for bottleneck attention + 1 for bottleneck feed forward + 1 for to image conv

class ResNetDecoder (nn.Module):
    def __init__ (self, config):
        super().__init__()
        self.config = config
        
        self.bottleneck_feedforward = ResNetResBlock (config.latent_channels, config.latent_channels, ResNetResBlockConfig)

        self.bottleneck_attention =  SelfAttention (config.latent_channels, SelfAttentionConfig)

        self.block = nn.ModuleList()

        in_channels = config.latent_channels
        for stage in config.channel_map:
            if stage == 128:
                num_groups = 16
            elif stage == 64:
                num_groups = 8
            else:
                num_groups = 32
            
            self.block.append(nn.ConvTranspose2d (in_channels, stage, kernel_size=2, stride=2) )
            self.block.append(ResNetResBlock (stage, stage, ResNetResBlockConfig, num_groups=num_groups))
            in_channels = stage
        
        self.to_image_conv = ResNetResBlock (config.channel_map[-1], config.image_channels, ResNetResBlockConfig, num_groups=1)
       
        self.final_group_norm = GroupNorm (config.image_channels, num_groups=1)
        self.tanh = nn.Tanh()
    
    def forward (self, x):

        activation = x
        activation = self.bottleneck_feedforward(activation)
        activation = self.bottleneck_attention (activation)
        
        for i in range (len (self.block)):
            activation = self.block[i](activation)
        activation = self.to_image_conv (activation)
        activation = self.final_group_norm(activation)
        image = self.tanh (activation)

        return image