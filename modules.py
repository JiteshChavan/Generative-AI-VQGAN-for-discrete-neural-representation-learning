import torch
import torch.nn as nn
import torch.nn.functional as F

from dataclasses import dataclass

@dataclass
class ConvBlockConfig:
    kernel_size : int = 3
    stride : int = 1
    padding : int = 1

@dataclass
class SelfAttentionConfig:
    # attention configs
    attention_resolution : int = 16 # latent resolution
    n_head : int = 4
    att_kernel_size : int = 1
    att_stride : int = 1
    att_padding : int = 0

    proj_kernel_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0

@dataclass
class ResBlockConfig:
    # post attention resblock (feedforward) pre norm style just as in GPT2 paper
    res_kernel_size : int = 1
    res_stride : int = 1
    res_padding : int = 0

@dataclass
class ResNetResBlockConfig:

    kernel_size : int = 3
    stride : int = 1
    padding : int = 1

    proj_k_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0

@dataclass
class MaxPoolSkipResNetResBlockConfig:

    kernel_size : int = 1
    stride : int = 1
    padding : int = 0

    proj_k_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0

@dataclass
class UpSampleSkipResNetResBlockConfig:

    kernel_size : int = 1
    stride : int = 1
    padding : int = 0

    proj_k_size : int = 1
    proj_stride : int = 1
    proj_padding : int = 0


class ConvBlock (nn.Module):

    def __init__ (self, in_channels, out_channels, conv_block_config, num_groups=32):
            super().__init__()
            self.block= nn.Sequential(
            nn.Conv2d (in_channels, out_channels, kernel_size=conv_block_config.kernel_size, stride=conv_block_config.stride, padding=conv_block_config.padding),
            GroupNorm (out_channels, num_groups),
            nn.GELU (),
            nn.Conv2d (out_channels, out_channels, kernel_size=conv_block_config.kernel_size, stride=conv_block_config.stride, padding=conv_block_config.padding),
            GroupNorm(out_channels, num_groups),
            nn.GELU()
        )

    def forward (self, x):
        return self.block(x)


class GroupNorm (nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        assert channels % num_groups == 0, f"num_groups: {num_groups} must be divisible by channels {channels}"
        self.group_norm = nn.GroupNorm (num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
    
    def forward (self, x):
        return self.group_norm(x)
    
# TODO:  incorporate pos embedding in the network itself 
class SelfAttention (nn.Module):
    def __init__(self, channels, config):

        super().__init__()
        
        att_kernel_size=config.att_kernel_size
        att_stride = config.att_stride
        att_padding = config.att_padding
        proj_kernel_size = config.proj_kernel_size
        proj_stride = config.proj_stride
        proj_padding = config.proj_padding

        self.n_head = config.n_head
        self.channels = channels

        assert channels % self.n_head == 0, f"Specified channels:{channels} are not divisible by number of attention heads{self.n_head}"
        # norm
        self.group_norm = GroupNorm (channels)
        # attention
        self.conv_attention = nn.Conv2d (channels, 3 * channels, kernel_size=att_kernel_size, stride=att_stride, padding=att_padding)
        self.conv_projection = nn.Conv2d (channels, channels, kernel_size=proj_kernel_size, stride=proj_stride, padding=proj_padding)
        # scale init shenanigans
        self.conv_projection.SKIP_CONNECTION_SCALE_INIT = 1
    
    def forward (self, X):
        B, C, H, W = X.size()
        # normalize X
        x_normalized = self.group_norm (X)

        # emit kqv
        # X (B, C, H, W)
        kqv = self.conv_attention (x_normalized) # (B, 3C, H, W)
        q, k, v = kqv.split (self.channels ,dim=1) #(B,C,H,W) x3

        # Required shape for F.scaled_dot_product_attention (B, nh, T, hs)

        head_size = C//self.n_head

        k=k.view (B, self.n_head, head_size, H*W).transpose(-2,-1) # (B, nh, HW, hs)
        q = q.view (B, self.n_head, head_size, H*W).transpose(-2,-1) # (B, nh, HW, hs)  # this head tends to all the pixels for these particular channels
        v=v.view (B, self.n_head, head_size, H*W).transpose(-2,-1) # (B, nh, HW, hs)

        #(HW HW) @ (HW, hs)
        #att = k @ q * (1.0/math.sqrt(head_size))
        #att = F.softmax (att, dim=-1) # (B, nh, HW, HW)
        #y = att @ v.transpose (-1,-2) # (B, nh, HW, HW) @ (B, nh, HW, hs) -> (B, nh, HW, hs) same as B T T @ B T C
        
        y = F.scaled_dot_product_attention (q, k, v, is_causal=False) # returns aggregated info : batched as a specific head for entire pixel space for particular channels (B, nh, HW(or T), hs)
        y = y.permute (0, 1, 3, 2).contiguous ().view (B, C, H, W)

        y = self.conv_projection(y)

        del kqv, q, k, v, x_normalized
        return X + y
    
class ResidualBlock (nn.Module):
    def __init__(self, in_channels, config):
        super().__init__()
        self.in_channels = in_channels

        kernel_size = config.res_kernel_size
        stride = config.res_stride
        padding = config.res_padding

        layers = [
            GroupNorm (in_channels),
            nn.Conv2d (in_channels, in_channels, kernel_size=1, stride=1, padding=0),
        ]
        self.conv_projection = nn.Conv2d (in_channels, in_channels, kernel_size=kernel_size, stride=stride, padding=padding) # Project back into residual pathway
        # conv_proj is the final projection layer only IF in_channels = out_channels
        self.conv_projection.SKIP_CONNECTION_SCALE_INIT = 1
        layers.append (self.conv_projection)
        self.block = nn.Sequential (*layers)


    def forward (self, x):
        return x + self.block(x)
    
class ResNetResBlock (nn.Module):
    def __init__ (self, in_channels, out_channels, config, num_groups = 8):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = config.kernel_size
        stride =config.stride
        padding = config.padding


        layers = [
            nn.Conv2d (in_channels, out_channels, kernel_size, stride, padding),
            GroupNorm (out_channels, num_groups),
            nn.GELU(),
            nn.Conv2d (out_channels, out_channels, kernel_size, stride, padding),
            GroupNorm (out_channels, num_groups),
            nn.GELU()
        ]

        self.conv_projection = nn.Conv2d (out_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv_projection.SKIP_CONNECTION_SCALE_INIT = 1
        layers.append (self.conv_projection)

        self.block = nn.Sequential (*layers)

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d (in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        
    def forward (self, x):
        if (self.in_channels == self.out_channels):
            return x + self.block(x)
        else:
            return self.channel_up(x) + self.block(x)


@dataclass
class ResNetResBlockCustomConfig:
    kernel_size : int = 3
    stride : int = 1
    padding : int = 1



class ResNetResBlockCustom (nn.Module):
    def __init__ (self, in_channels, intermediate_channels, config, num_groups=8):
        super().__init__()
        self.config = config

        layers = [
            nn.Conv2d (in_channels, intermediate_channels, config.kernel_size, config.stride, config.padding),
            GroupNorm (intermediate_channels, num_groups),
            nn.GELU(),
            nn.Conv2d (intermediate_channels, in_channels, config.kernel_size, config.stride, config.padding),
            GroupNorm (in_channels, num_groups),
            nn.GELU()
        ]

        self.conv_projection = nn.Conv2d (in_channels, in_channels, kernel_size=1, stride=1, padding=0) # projection
        self.conv_projection.SKIP_CONNECTION_SCALE_INIT = 1
        layers.append (self.conv_projection)

        self.block = nn.Sequential (*layers)


        # no need for channel up since in channels is adding to in channels

    def forward (self, x):
        return x + self.block(x)

        


