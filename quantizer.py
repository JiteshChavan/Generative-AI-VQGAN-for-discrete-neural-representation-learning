import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class QuantizerConfig:
    vocab_size : int = 256
    # n_embd has to be same as encoder output latent_dim since we find difference
    n_embd : int = 1024
    commitment_cost : int = 0.25

class Quantizer (nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # codebook
        self.codebook = nn.Embedding (config.vocab_size, config.n_embd)

    # X is encoder output # (B, latent_channels, latent_res, latent_res) -> (B, C, H, W)
    def forward (self, X):
        # this is ze
        X = X.permute (0, 2, 3, 1).contiguous() # (B, H, W, C)
        B, H, W, C = X.shape

        # unroll every vector from 16x16 representation into a column and introduce a spurious dimension for broadcasting
        flat_X = X.view (B*H*W, C) # (BHW, 1, C)
        flat_X = flat_X.unsqueeze(1)
        # codebook shape (VS, C) for now C = n_embd

        # find difference for each latent for each codebook vector so you have BHW, VS differences
        # (BHW, 1, C) - (VS, C) -> (BHW VS C)
        # reduce the differences to estimated euclidean distance (mean(squared sum) instead of root(squared sum))
        distances = (flat_X - self.codebook.weight).pow(2).mean(2) # (BHW, VS)

        encoding_indices = torch.argmin (distances, dim=1) # (BHW)

        # this is zq
        quantized = self.codebook(encoding_indices).view(B, H, W, C) # (B, H, W, n_embd or C)

        # calculate losses
        # X (B, H, W, C)
        quantization_loss = F.mse_loss (X.detach(), quantized)
        commitment_loss = F.mse_loss (X, quantized.detach())

        vq_loss = quantization_loss + self.config.commitment_cost * commitment_loss

        # if we are trainin, we need gradients to flow back to encoder from decoder
        if self.training:
            quantized = X + (quantized - X).detach() # (B, H, W, C)

        # arrange zq to be processed by decoder (B, C, H, W)        
        quantized = quantized.permute(0, 3, 1, 2).contiguous()
        # arrage latent representation (16x16) according to images in batch
        encoding_indices = encoding_indices.view(B, H*W) # (B, HW) or (B, 256)
        return vq_loss, quantized, encoding_indices


