"""3D self-attention module for diffusion models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class SelfAttention3D(nn.Module):
    """Multi-head self-attention for 3D feature maps."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 32
    ):
        """Initialize 3D self-attention.

        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            head_dim: Dimension per head
        """
        super().__init__()

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        # Project to Q, K, V
        inner_dim = num_heads * head_dim
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv3d(channels, inner_dim * 3, kernel_size=1, bias=False)
        self.proj_out = nn.Conv3d(inner_dim, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor, shape (B, C, D, H, W)

        Returns:
            Output tensor, shape (B, C, D, H, W)
        """
        residual = x
        B, C, D, H, W = x.shape

        # Normalize
        x = self.norm(x)

        # Compute Q, K, V
        qkv = self.qkv(x)  # (B, inner_dim*3, D, H, W)

        # Reshape for multi-head attention
        # Split into Q, K, V and reshape to (B, num_heads, D*H*W, head_dim)
        q, k, v = rearrange(
            qkv,
            'b (three heads dim) d h w -> three b heads (d h w) dim',
            three=3,
            heads=self.num_heads,
            dim=self.head_dim
        )

        # Attention scores
        attn = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale  # (B, heads, DHW, DHW)
        attn = F.softmax(attn, dim=-1)

        # Apply attention to values
        out = torch.einsum('bhij,bhjd->bhid', attn, v)  # (B, heads, DHW, dim)

        # Reshape back to spatial dimensions
        out = rearrange(
            out,
            'b heads (d h w) dim -> b (heads dim) d h w',
            d=D,
            h=H,
            w=W,
            heads=self.num_heads
        )

        # Project output
        out = self.proj_out(out)

        # Residual connection
        return out + residual


class AttentionBlock3D(nn.Module):
    """Attention block wrapper with optional skip."""

    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        head_dim: int = 32,
        use_checkpoint: bool = False
    ):
        """Initialize attention block.

        Args:
            channels: Number of input channels
            num_heads: Number of attention heads
            head_dim: Dimension per head
            use_checkpoint: Use gradient checkpointing to save memory
        """
        super().__init__()

        self.attention = SelfAttention3D(channels, num_heads, head_dim)
        self.use_checkpoint = use_checkpoint

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply attention with optional checkpointing.

        Args:
            x: Input tensor, shape (B, C, D, H, W)

        Returns:
            Output tensor, shape (B, C, D, H, W)
        """
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self.attention, x, use_reentrant=False)
        else:
            return self.attention(x)
