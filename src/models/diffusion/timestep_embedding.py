"""Timestep embedding for diffusion models."""

import torch
import torch.nn as nn
import math


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal positional embedding for timesteps."""

    def __init__(self, dim: int, max_period: int = 10000):
        """Initialize sinusoidal embedding.

        Args:
            dim: Embedding dimension
            max_period: Maximum period for sinusoidal functions
        """
        super().__init__()
        self.dim = dim
        self.max_period = max_period

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Create sinusoidal embeddings for timesteps.

        Args:
            timesteps: Timesteps, shape (B,)

        Returns:
            Embeddings, shape (B, dim)
        """
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=timesteps.device) * -embeddings)
        embeddings = timesteps[:, None] * embeddings[None, :]
        embeddings = torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

        # Handle odd dimensions
        if self.dim % 2 == 1:
            embeddings = torch.nn.functional.pad(embeddings, (0, 1))

        return embeddings


class TimestepEmbedding(nn.Module):
    """Complete timestep embedding with MLP projection."""

    def __init__(
        self,
        timestep_dim: int = 128,
        embedding_dim: int = 512,
        activation: str = 'silu'
    ):
        """Initialize timestep embedding.

        Args:
            timestep_dim: Dimension of sinusoidal embedding
            embedding_dim: Dimension of output embedding
            activation: Activation function ('silu', 'relu', 'gelu')
        """
        super().__init__()

        self.sinusoidal_embedding = SinusoidalPositionEmbedding(timestep_dim)

        # MLP projection
        self.mlp = nn.Sequential(
            nn.Linear(timestep_dim, embedding_dim),
            self._get_activation(activation),
            nn.Linear(embedding_dim, embedding_dim)
        )

    def _get_activation(self, name: str) -> nn.Module:
        """Get activation function by name."""
        activations = {
            'silu': nn.SiLU(),
            'relu': nn.ReLU(),
            'gelu': nn.GELU()
        }
        return activations.get(name, nn.SiLU())

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Embed timesteps.

        Args:
            timesteps: Timesteps, shape (B,)

        Returns:
            Embeddings, shape (B, embedding_dim)
        """
        # Sinusoidal embedding
        emb = self.sinusoidal_embedding(timesteps)

        # MLP projection
        emb = self.mlp(emb)

        return emb
