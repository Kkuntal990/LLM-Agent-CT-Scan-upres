"""3D ResNet blocks with FiLM conditioning for diffusion models."""

import torch
import torch.nn as nn


class ResNetBlock3D(nn.Module):
    """3D ResNet block with timestep conditioning via FiLM (Feature-wise Linear Modulation)."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        timestep_emb_dim: int,
        groups: int = 8,
        dropout: float = 0.0
    ):
        """Initialize 3D ResNet block.

        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            timestep_emb_dim: Dimension of timestep embedding
            groups: Number of groups for GroupNorm
            dropout: Dropout probability
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # First conv block
        self.norm1 = nn.GroupNorm(groups, in_channels)
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.activation = nn.SiLU()

        # Timestep embedding projection (for FiLM conditioning)
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(timestep_emb_dim, out_channels * 2)  # Scale and shift parameters
        )

        # Second conv block
        self.norm2 = nn.GroupNorm(groups, out_channels)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1)

        # Residual connection
        if in_channels != out_channels:
            self.residual_conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual_conv = nn.Identity()

    def forward(self, x: torch.Tensor, time_emb: torch.Tensor) -> torch.Tensor:
        """Forward pass with timestep conditioning.

        Args:
            x: Input tensor, shape (B, C_in, D, H, W)
            time_emb: Timestep embedding, shape (B, timestep_emb_dim)

        Returns:
            Output tensor, shape (B, C_out, D, H, W)
        """
        residual = x

        # First conv block
        h = self.norm1(x)
        h = self.activation(h)
        h = self.conv1(h)

        # Apply FiLM conditioning from timestep
        # Project timestep embedding to scale and shift parameters
        time_emb = self.time_emb_proj(time_emb)
        # Split into scale and shift
        scale, shift = torch.chunk(time_emb, 2, dim=1)
        # Reshape for broadcasting: (B, C, 1, 1, 1)
        scale = scale[:, :, None, None, None]
        shift = shift[:, :, None, None, None]
        # Apply FiLM: h = scale * h + shift
        h = scale * h + shift

        # Second conv block
        h = self.norm2(h)
        h = self.activation(h)
        h = self.dropout(h)
        h = self.conv2(h)

        # Residual connection
        return h + self.residual_conv(residual)


class DownsampleBlock3D(nn.Module):
    """Downsampling block using strided convolution."""

    def __init__(self, channels: int):
        """Initialize downsample block.

        Args:
            channels: Number of channels
        """
        super().__init__()
        self.conv = nn.Conv3d(channels, channels, kernel_size=3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Downsample input.

        Args:
            x: Input tensor, shape (B, C, D, H, W)

        Returns:
            Downsampled tensor, shape (B, C, D//2, H//2, W//2)
        """
        return self.conv(x)


class UpsampleBlock3D(nn.Module):
    """Upsampling block using transposed convolution."""

    def __init__(self, channels: int):
        """Initialize upsample block.

        Args:
            channels: Number of channels
        """
        super().__init__()
        self.conv = nn.ConvTranspose3d(channels, channels, kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Upsample input.

        Args:
            x: Input tensor, shape (B, C, D, H, W)

        Returns:
            Upsampled tensor, shape (B, C, D*2, H*2, W*2)
        """
        return self.conv(x)
