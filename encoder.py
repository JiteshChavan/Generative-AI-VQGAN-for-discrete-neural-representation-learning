import torch.nn as nn
import torch.nn.functional as F
from modules import GroupNorm, Swish, ResidualBlock, DownSampleBlock, SelfAttention
from dataclasses import dataclass

from utility_modules.seed_manager import Seeds



#  = nn.ModuleList ([Block (config) for _ in range (config.n_layer)])
@dataclass
class EncoderConfig:
    # general configs
    in_resolution : int = 256
    in_channels : int = 3
    n_compressions : int = 4 # (spatial resolution is compressed by 2^4)
    n_post_compression_skip_connections : int = 3


    # latent config
    latent_in_channels : int = 128
    latent_dim : int = 1024 # TODO: check, and change later if needed

    latent_resolution : int = 16 # on which we run attention
    
    # res block configs
    res_kernel_size : int = 3
    res_stride : int = 1
    res_padding : int = 1

    # res block channel up configs
    channel_up_kernel_size : int = 1
    channel_up_stride : int = 1
    channel_up_padding : int = 0

    # attention configs
    attention_resolution : int = 16 # latent resolution
    n_head : int = 4
    att_kernel_size : int = 1
    att_stride : int = 1
    att_padding : int = 0

    # configs for different conv layers
    conv_kernel_size : int = 3
    conv_stride : int = 1
    conv_padding : int = 1

    proj_kernel_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0

    # down sample configs
    d_sample_kernel_size : int = 3
    d_sample_factor = 2 # (specifies stride)
    d_sample_padding : int = 0




class Encoder (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # explode RGB into 128 using conv2d

        self.initial_conv = nn.Conv2d (config.in_channels, config.latent_in_channels, kernel_size=config.conv_kernel_size, stride=config.conv_stride, padding=config.conv_padding)
        layers = [self.initial_conv]
        
        # init setup for channels
        in_channels = config.latent_in_channels
        out_channels = 2*in_channels

        for i in range (config.n_compressions):
            layers.append (ResidualBlock (in_channels, out_channels, config))
            layers.append (DownSampleBlock (out_channels, config))
            # channel map
            # 256, 256, 512, 1024
            # 0,    1,   2,    3
            if i == 0:
                in_channels = out_channels
            else :
                in_channels = out_channels
                out_channels = 2 * in_channels


        #  TODO: UPDATE this thing "n_post_compression_skip_connections" in config if you want to add more skip connections post compression, res has 1 skip conn, selfatt has 1 as well
        layers.append (ResidualBlock (in_channels, in_channels, config))
        layers.append (SelfAttention (in_channels, config))
        layers.append (ResidualBlock (in_channels, in_channels, config))
        
        layers.append (GroupNorm (in_channels))
        layers.append (Swish())
        layers.append (nn.Conv2d(in_channels, config.latent_dim, kernel_size=config.conv_kernel_size, stride=config.conv_stride, padding=config.conv_padding))
        
        self.model = nn.Sequential(*layers)

    # don't need to explicitly sample ze from N(u,sigma), quantizer already snaps similar zes to same zq
    # which sort of already organizes the latent space via quantization and commitment loss gradients
    # so if we sample zes from posterior ze|x we get zq and reconstructing this zq gives meaningful variations from p(x)
    #def reparametrize (self, mu, log_var):
    #    eps = torch.randn_like (mu) # shape should be same as mu and log_var (B, latent_dim, latent_resolution, latent_resolution)
    #    std = torch.exp (0.5* F.softplus(log_var))
    #    out = mu + eps * std
    #    return out
    
    def forward (self, X):
        ze = self.model(X)
        return ze

