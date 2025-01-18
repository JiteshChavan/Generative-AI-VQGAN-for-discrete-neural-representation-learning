import torch
import torch.nn as nn
import torch.nn.functional as F


from modules import ConvBlock, ConvBlockConfig
from modules import GroupNorm

from dataclasses import dataclass

@dataclass
class DecoderConfig:
    latent_channels : int = 1024
    out_channels : int = 3 # image channels RGB

    # make sure thats exact reverse of encoder config
    # not scalable, make sure to change channel map to scale the model to higher latent_dimensions
    channel_map = [512, 256, 128, 64]
    n_expansions : int = 4

    kaiming_init_gain : float = 1.0


class Decoder (nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.kaiming_init_gain = config.kaiming_init_gain

        self.block = nn.ModuleList ()

        # deconv bvlock then match then double convblock
        for stage in config.channel_map:
            # expands spatial resolution by factor of 2 and halves the number of channels at the same time
            self.block.append (nn.ConvTranspose2d(2*stage, stage, kernel_size=2, stride=2))
            self.block.append (ConvBlock (2*stage, stage, ConvBlockConfig))

        self.to_image_conv = ConvBlock (stage, config.out_channels, ConvBlockConfig, num_groups=1)
        self.final_group_norm = GroupNorm (config.out_channels, num_groups=1)
        self.tanh = nn.Tanh()

        self.apply (self._init_weights)
    
    def _init_weights (self, module):
        if isinstance (module, nn.Conv2d):
            fan_in = nn.init._calculate_correct_fan (module.weight, mode='fan_in')
            std = (self.kaiming_init_gain / fan_in)**-0.5

            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward (self, skip_connections, x):
        # hacky be careful
        skip_connections.reverse()
        activation = x

        for i in (range (len (self.block))):

            if i % 2 == 0:
                # expand spatial, half channels with conv transpose
                upsampled = self.block[i](activation)
                assert (skip_connections[i//2].size(1) == upsampled.size(1))
                # add skip connections for inductive bias
                informed_activation = torch.cat ((skip_connections[i//2], upsampled), dim=1)
                activation = informed_activation

                del upsampled
            else:
                # send to convblock
                activation = self.block[i](activation)
        
        activation = self.to_image_conv (activation)
        activation = self.final_group_norm (activation)

        image = self.tanh (activation)

        # not maintaining state so dont need this
        #del skip_connections[:]
        #torch.cuda.empty_cache()
        return image


