import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# TODO: try setting up a config class later to generalize the code better

class GroupNorm (nn.Module):
    def __init__(self, channels, num_groups=32):
        super().__init__()
        assert channels % num_groups == 0, f"num_groups: {num_groups} must be divisible by channels {channels}"
        self.group_norm = nn.GroupNorm (num_groups=num_groups, num_channels=channels, eps=1e-6, affine=True)
    
    def forward (self, X):
        return self.group_norm(X)

class Swish (nn.Module):
    def forward (self, X):
        return X * torch.sigmoid(X)

class ResidualBlock (nn.Module):
    def __init__(self, in_channels, out_channels, config):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        kernel_size = config.res_kernel_size
        stride = config.res_stride
        padding = config.res_padding

        channel_up_kernel_size=config.channel_up_kernel_size
        channel_up_stride = config.channel_up_stride
        channel_up_padding = config.channel_up_padding
        
        self.block = nn.Sequential (
            GroupNorm (in_channels),
            Swish (),
            nn.Conv2d (in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            GroupNorm (out_channels),
            Swish (),
            nn.Conv2d (out_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding) # Project back into residual pathway
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d (in_channels, out_channels, kernel_size=channel_up_kernel_size, stride=channel_up_stride, padding=channel_up_padding)
    
    def forward (self, X):

        if self.in_channels != self.out_channels:
            return self.channel_up(X) + self.block(X)
        else:
            return X + self.block(X)

class UpSampleBlock (nn.Module):
    def __init__(self, channels, factor=2.0):
        super().__init__()
        self.conv = nn.Conv2d (channels, channels, kernel_size=3, stride=1, padding=1)
        self.factor = factor
    def forward (self, X):
        X = F.interpolate (X, scale_factor=self.factor)
        return self.conv(X)

class DownSampleBlock (nn.Module):
    def __init__(self, channels, config):
        super().__init__()
        kernel_size = config.d_sample_kernel_size
        d_sample_factor = config.d_sample_factor
        padding = config.d_sample_padding
        
        self.conv = nn.Conv2d (channels, channels, kernel_size=kernel_size, stride=d_sample_factor, padding=padding)
    
    def forward (self, X):
        X = F.pad (X, (0, 1, 0, 1), mode='constant', value=0) # Kernel size 3 makes it so that stride = 2 gives you one less than half the size so one extra row and column to fix it
        return self.conv (X)


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
        return X + y

