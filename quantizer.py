import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class QuantizerConfig:
    vocab_size : int = 256
    # n_embd has to be same as encoder output latent_dim since we find difference
    n_embd : int = 256
    commitment_cost : int = 0.25

class Quantizer (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # codebook
        self.codebook = nn.Embedding (config.vocab_size, config.n_embd)

    # X is encoder output # (B, latent_channels, latent_res, latent_res) -> (B, C, H, W)
    def forward (self, ze):
        # this is ze
        ze = ze.permute (0, 2, 3, 1).contiguous() # (B, H, W, C)
        B, H, W, C = ze.shape
        # unroll every vector from 16x16 representation into a column and introduce a spurious dimension for broadcasting
        ze_flat = ze.view (B*H*W, C)
        
        # codebook shape (VS, C) for now C = n_embd
        # (a-b)^2 = a^2 -2ab + b^2
        # BHW,C - VS,C = BHW C + VS C - BHW VS

        # find difference for each latent for each codebook vector so you have BHW, VS differences
        distances = torch.sum(ze_flat**2, dim=1, keepdim=True) + torch.sum(self.codebook.weight**2, dim=1) - 2*torch.matmul(ze_flat, self.codebook.weight.t())

        encoding_indices = torch.argmin (distances, dim=1) # (BHW)

        # this is zq
        zq = self.codebook(encoding_indices).view(B, H, W, C) # (B, H, W, n_embd or C)

        # calculate losses
        # X (B, H, W, C)
        quantization_loss = F.mse_loss (ze.detach(), zq)
        commitment_loss = F.mse_loss (ze, zq.detach())

        vq_loss = quantization_loss + self.config.commitment_cost * commitment_loss

        # if we are trainin, we need gradients to flow back to encoder from decoder
        if self.training:
            zq = ze + (zq - ze).detach() # (B, H, W, C)

        # arrange zq to be processed by decoder (B, C, H, W)        
        zq = zq.permute(0, 3, 1, 2).contiguous()
        # arrage latent representation (16x16) according to images in batch
        encoding_indices = encoding_indices.view(B, H*W) # (B, HW) or (B, 256)
        return vq_loss, zq, encoding_indices


