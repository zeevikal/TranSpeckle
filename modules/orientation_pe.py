"""
Orientation-Aware Positional Embeddings

Extends standard learned positional embeddings with explicit angular encoding
relative to the image center. Captures radial symmetry patterns in speckle data.

Google Colab Setup:
    !pip install torch

"""

import math
import torch
import torch.nn as nn


class OrientationPositionalEmbedding(nn.Module):
    """
    Positional embedding with orientation encoding
    
    Args:
        base_embedding: Learned base positional embedding
        emb_dim: Embedding dimension
        grid_size: Patch grid size (e.g., 8 for 8x8 patches)
        add_theta: Whether to add angular encoding
    """
    
    def __init__(self, base_embedding: nn.Parameter, emb_dim: int, grid_size: int = 16, add_theta: bool = True):
        super().__init__()
        self.emb_dim = emb_dim
        self.add_theta = add_theta
        self.base_embedding = base_embedding

        if add_theta:
            self.theta_proj = nn.Linear(2, emb_dim)
            self.register_buffer("theta_encoding", self._create_theta_encoding(grid_size))

    def _create_theta_encoding(self, grid_size: int):
        """Create sinusoidal angular encoding for each patch"""
        encoding = torch.zeros(grid_size * grid_size, 2)
        center = grid_size / 2 - 0.5
        for i in range(grid_size):
            for j in range(grid_size):
                dx = j - center
                dy = i - center
                theta = math.atan2(dy, dx)
                idx = i * grid_size + j
                encoding[idx, 0] = math.sin(theta)
                encoding[idx, 1] = math.cos(theta)
        return encoding

    def forward(self, x):
        B, N, D = x.shape
        pos = self.base_embedding.repeat(B, 1, 1)

        if self.add_theta:
            theta_feats = self.theta_proj(self.theta_encoding)
            theta_feats = theta_feats.unsqueeze(0).repeat(B, 1, 1)
            pos = pos + theta_feats

        return x + pos
