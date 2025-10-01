"""3D Residual U-Net with z-axis upsampling for through-plane SR.

Lightweight architecture that upsamples only along z-axis while preserving
in-plane resolution, optimized for MPS backend on Apple Silicon.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class ResidualBlock3D(nn.Module):
    """3D Residual block with two convolutions and skip connection."""

    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm3d(channels)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        out = self.relu(out)
        return out


class DownBlock(nn.Module):
    """Encoder block with residual convolutions and downsampling."""

    def __init__(self, in_channels: int, out_channels: int, downsample_z: bool = True):
        super().__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock3D(out_channels)

        # Downsample along z-axis only or all axes
        if downsample_z:
            self.pool = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2))
        else:
            self.pool = None

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.relu(self.bn(self.conv(x)))
        x = self.residual(x)
        skip = x
        if self.pool is not None:
            x = self.pool(x)
        return x, skip


class UpBlock(nn.Module):
    """Decoder block with z-axis upsampling and skip connections."""

    def __init__(self, in_channels: int, out_channels: int, upsample_z_only: bool = False):
        super().__init__()
        self.upsample_z_only = upsample_z_only

        if upsample_z_only:
            # Transposed conv only along z-axis: kernel (2,1,1), stride (2,1,1)
            self.up = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=(2, 1, 1),
                stride=(2, 1, 1)
            )
        else:
            # Standard 3D upsampling
            self.up = nn.ConvTranspose3d(
                in_channels, out_channels,
                kernel_size=2,
                stride=2
            )

        self.conv = nn.Conv3d(out_channels * 2, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.residual = ResidualBlock3D(out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.up(x)

        # Handle size mismatch from pooling
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)

        x = torch.cat([x, skip], dim=1)
        x = self.relu(self.bn(self.conv(x)))
        x = self.residual(x)
        return x


class ResidualUNet3D(nn.Module):
    """Lightweight 3D Residual U-Net for through-plane super-resolution.

    Architecture designed for z-axis upsampling while preserving in-plane
    resolution. Optimized for Apple Silicon MPS backend.

    Args:
        in_channels: Number of input channels (default 1 for CT)
        out_channels: Number of output channels (default 1)
        base_channels: Base number of feature channels (default 32)
        depth: Number of encoder/decoder levels (default 3)
        upscale_factor: Z-axis upsampling factor (default 2)
        z_only_upsample: If True, upsample only z-axis in decoder
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        depth: int = 3,
        upscale_factor: int = 2,
        z_only_upsample: bool = True
    ):
        super().__init__()
        self.depth = depth
        self.upscale_factor = upscale_factor
        self.z_only_upsample = z_only_upsample

        # Initial convolution
        self.initial_conv = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(base_channels),
            nn.ReLU(inplace=True)
        )

        # Encoder
        self.encoders = nn.ModuleList()
        channels = base_channels
        for i in range(depth):
            next_channels = channels * 2
            self.encoders.append(DownBlock(channels, next_channels, downsample_z=True))
            channels = next_channels

        # Bottleneck
        self.bottleneck = ResidualBlock3D(channels)

        # Decoder
        self.decoders = nn.ModuleList()
        for i in range(depth):
            next_channels = channels // 2
            self.decoders.append(
                UpBlock(channels, next_channels, upsample_z_only=z_only_upsample)
            )
            channels = next_channels

        # Additional z-upsampling if needed
        if upscale_factor > 2 ** depth:
            # Add extra upsampling layers
            extra_factor = upscale_factor // (2 ** depth)
            self.extra_upsample = nn.ConvTranspose3d(
                channels, channels,
                kernel_size=(extra_factor, 1, 1),
                stride=(extra_factor, 1, 1)
            )
        else:
            self.extra_upsample = None

        # Final output convolution
        self.output_conv = nn.Conv3d(channels, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor, shape (B, C, D, H, W)

        Returns:
            Output tensor with upsampled z-dimension
        """
        # Initial conv
        x = self.initial_conv(x)

        # Encoder with skip connections
        skips = []
        for encoder in self.encoders:
            x, skip = encoder(x)
            skips.append(skip)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder with skip connections
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)

        # Extra upsampling if needed
        if self.extra_upsample is not None:
            x = self.extra_upsample(x)

        # Output
        x = self.output_conv(x)

        return x

    def get_num_parameters(self) -> int:
        """Get total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_model(
    upscale_factor: int = 2,
    base_channels: int = 32,
    depth: int = 3,
    device: str = 'mps'
) -> ResidualUNet3D:
    """Factory function to create and initialize model.

    Args:
        upscale_factor: Z-axis upsampling factor
        base_channels: Base feature channels
        depth: U-Net depth
        device: Device to place model on

    Returns:
        Initialized model on specified device
    """
    model = ResidualUNet3D(
        in_channels=1,
        out_channels=1,
        base_channels=base_channels,
        depth=depth,
        upscale_factor=upscale_factor,
        z_only_upsample=True
    )

    model = model.to(device)

    print(f"Model created with {model.get_num_parameters():,} parameters")
    print(f"Device: {device}")

    return model
