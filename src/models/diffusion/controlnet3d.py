"""3D ControlNet for conditioning latent diffusion on LR inputs.

Adapted from DiffBIR's IRControlNet for medical imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class IRControlNet3D(nn.Module):
    """Image Restoration ControlNet for 3D medical images.

    Extracts conditioning features from LR input to guide diffusion.
    Based on DiffBIR (ECCV 2024) adapted for 3D volumes.
    """

    def __init__(
        self,
        in_channels: int = 4,  # Latent channels
        control_channels: int = 4,  # Output control features
        base_channels: int = 64
    ):
        super().__init__()

        self.in_channels = in_channels
        self.control_channels = control_channels

        # Feature extraction from LR latent
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # Degradation-aware feature blocks
        self.block1 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )

        self.block2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU(),
            nn.Conv3d(base_channels * 2, base_channels * 2, 3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )

        # Output projection to control features
        self.conv_out = nn.Conv3d(base_channels * 2, control_channels, 1)

        # Zero initialization for gradual learning
        nn.init.zeros_(self.conv_out.weight)
        nn.init.zeros_(self.conv_out.bias)

    def forward(self, lr_latent: torch.Tensor) -> torch.Tensor:
        """Extract control features from LR latent.

        Args:
            lr_latent: LR latent from VAE, shape (B, in_channels, D, H, W)

        Returns:
            Control features, shape (B, control_channels, D, H, W)
        """
        h = self.conv_in(lr_latent)
        h = self.block1(h)
        h = self.block2(h)
        control = self.conv_out(h)

        return control


class MedicalLatentDiffusion3D(nn.Module):
    """Complete Medical Latent Diffusion Model with ControlNet conditioning.

    Combines:
    - Medical VAE for latent compression
    - 3D Diffusion UNet for denoising
    - IRControlNet for LR conditioning
    - ResShift for efficient sampling
    """

    def __init__(
        self,
        vae,
        unet,
        latent_channels: int = 4
    ):
        super().__init__()

        self.vae = vae
        self.unet = unet
        self.controlnet = IRControlNet3D(
            in_channels=latent_channels,
            control_channels=latent_channels,
            base_channels=64
        )

        self.latent_channels = latent_channels

    def encode_to_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Encode volume to latent space."""
        with torch.no_grad():
            return self.vae.encode_to_latent(x, sample=sample)

    def decode_from_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to volume."""
        with torch.no_grad():
            return self.vae.decode(z)

    def forward(
        self,
        noisy_latent: torch.Tensor,
        timesteps: torch.Tensor,
        lr_latent: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass through diffusion model.

        Args:
            noisy_latent: Noisy HR latent, shape (B, C, D, H, W)
            timesteps: Timesteps, shape (B,)
            lr_latent: LR conditioning latent, shape (B, C, D, H, W)

        Returns:
            Predicted residual, shape (B, C, D, H, W)
        """
        # Get control features from LR
        control_features = self.controlnet(lr_latent)

        # Concatenate noisy latent with control features
        unet_input = torch.cat([noisy_latent, control_features], dim=1)

        # Predict residual
        predicted_residual = self.unet(unet_input, timesteps)

        return predicted_residual

    @torch.no_grad()
    def prepare_lr_condition(
        self,
        lr_volume: torch.Tensor,
        target_shape: tuple
    ) -> torch.Tensor:
        """Prepare LR volume as conditioning.

        Args:
            lr_volume: Low-resolution volume, shape (B, 1, D/2, H, W)
            target_shape: Target HR shape (D, H, W)

        Returns:
            LR latent upsampled to target resolution
        """
        # Upsample LR to HR size along z-axis
        lr_upsampled = F.interpolate(
            lr_volume,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )

        # Encode to latent
        lr_latent = self.vae.encode_to_latent(lr_upsampled, sample=False)

        return lr_latent


def create_medical_latent_diffusion(
    vae,
    unet,
    latent_channels: int = 4
) -> MedicalLatentDiffusion3D:
    """Create complete Medical Latent Diffusion model.

    Args:
        vae: Pre-trained or initialized Medical VAE
        unet: Latent Diffusion UNet
        latent_channels: Number of latent channels

    Returns:
        MedicalLatentDiffusion3D model
    """
    model = MedicalLatentDiffusion3D(
        vae=vae,
        unet=unet,
        latent_channels=latent_channels
    )

    # Count total parameters
    vae_params = sum(p.numel() for p in vae.parameters())
    unet_params = sum(p.numel() for p in unet.parameters())
    control_params = sum(p.numel() for p in model.controlnet.parameters())

    print(f"Medical Latent Diffusion Model created:")
    print(f"  VAE parameters: {vae_params:,}")
    print(f"  UNet parameters: {unet_params:,}")
    print(f"  ControlNet parameters: {control_params:,}")
    print(f"  Total parameters: {vae_params + unet_params + control_params:,}")

    return model
