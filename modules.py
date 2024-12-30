import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.block = nn.Sequential (
            GroupNorm (in_channels),
            Swish (),
            nn.Conv2d (in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            GroupNorm (out_channels),
            Swish (),
            nn.Conv2d (out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d (in_channels, out_channels, kernel_size=1, stride=1, padding=0)
    
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
    def __init__(self, channels, factor=2):
        super().__init__()
        self.conv = nn.Conv2d (channels, channels, kernel_size=3, stride=factor, padding=0)
    
    def forward (self, X):
        X = F.pad (X, (0, 1, 0, 1), mode='constant', value=0) # Kernel size 3 makes it so that stride = 2 gives you one less than half the size so one extra row and column to fix it
        return self.conv (X)

class NonLocalBlock (nn.Module):
    pass