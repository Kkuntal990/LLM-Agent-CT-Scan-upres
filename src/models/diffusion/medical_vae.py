"""Medical VAE for 3D CT latent space encoding/decoding.

Adapted from microsoft/mri-autoencoder-v0.1 for CT imaging.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ResidualBlock3D(nn.Module):
    """3D Residual block for VAE."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, 3, padding=1)
        self.conv2 = nn.Conv3d(out_channels, out_channels, 3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.norm2 = nn.GroupNorm(8, out_channels)
        self.activation = nn.SiLU()

        self.skip = nn.Conv3d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.skip(x)

        h = self.conv1(x)
        h = self.norm1(h)
        h = self.activation(h)

        h = self.conv2(h)
        h = self.norm2(h)
        h = self.activation(h)

        return h + residual


class Encoder3D(nn.Module):
    """3D Encoder for medical images."""

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    ):
        super().__init__()

        # Initial convolution
        self.conv_in = nn.Conv3d(in_channels, base_channels, 3, padding=1)

        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        ch = base_channels

        for mult in channel_mult:
            out_ch = base_channels * mult
            self.down_blocks.append(nn.ModuleList([
                ResidualBlock3D(ch, out_ch),
                ResidualBlock3D(out_ch, out_ch),
                nn.Conv3d(out_ch, out_ch, 3, stride=2, padding=1)  # Downsample
            ]))
            ch = out_ch

        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock3D(ch, ch),
            ResidualBlock3D(ch, ch)
        ])

        # Output to latent space (mean and logvar)
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv3d(ch, 2 * latent_channels, 1)  # 2x for mean and logvar

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.conv_in(x)

        # Downsample
        for res_block1, res_block2, downsample in self.down_blocks:
            h = res_block1(h)
            h = res_block2(h)
            h = downsample(h)

        # Middle
        for block in self.mid_blocks:
            h = block(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        # Split into mean and logvar
        mean, logvar = torch.chunk(h, 2, dim=1)
        return mean, logvar


class Decoder3D(nn.Module):
    """3D Decoder for medical images."""

    def __init__(
        self,
        latent_channels: int = 4,
        out_channels: int = 1,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    ):
        super().__init__()

        # Input from latent
        ch = base_channels * channel_mult[-1]
        self.conv_in = nn.Conv3d(latent_channels, ch, 3, padding=1)

        # Middle blocks
        self.mid_blocks = nn.ModuleList([
            ResidualBlock3D(ch, ch),
            ResidualBlock3D(ch, ch)
        ])

        # Upsampling blocks
        self.up_blocks = nn.ModuleList()

        for mult in reversed(channel_mult):
            out_ch = base_channels * mult
            self.up_blocks.append(nn.ModuleList([
                ResidualBlock3D(ch, out_ch),
                ResidualBlock3D(out_ch, out_ch),
                nn.ConvTranspose3d(out_ch, out_ch, 2, stride=2)  # Upsample
            ]))
            ch = out_ch

        # Output convolution
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(z)

        # Middle
        for block in self.mid_blocks:
            h = block(h)

        # Upsample
        for res_block1, res_block2, upsample in self.up_blocks:
            h = res_block1(h)
            h = res_block2(h)
            h = upsample(h)

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


class MedicalVAE3D(nn.Module):
    """3D Variational Autoencoder for CT/MRI medical imaging.

    Provides latent space compression for efficient diffusion.
    """

    def __init__(
        self,
        in_channels: int = 1,
        latent_channels: int = 4,
        base_channels: int = 64,
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8)
    ):
        super().__init__()

        self.encoder = Encoder3D(in_channels, latent_channels, base_channels, channel_mult)
        self.decoder = Decoder3D(latent_channels, in_channels, base_channels, channel_mult)

        self.latent_channels = latent_channels

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode input to latent distribution parameters."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent to reconstruction."""
        return self.decoder(z)

    def reparameterize(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(self, x: torch.Tensor, sample: bool = True) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through VAE.

        Args:
            x: Input volume, shape (B, C, D, H, W)
            sample: If True, sample from latent distribution; if False, use mean

        Returns:
            reconstruction: Decoded output
            mean: Latent mean
            logvar: Latent log variance
        """
        mean, logvar = self.encode(x)

        if sample:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean

        reconstruction = self.decode(z)

        return reconstruction, mean, logvar

    def encode_to_latent(self, x: torch.Tensor, sample: bool = False) -> torch.Tensor:
        """Encode input to latent space (for diffusion training)."""
        mean, logvar = self.encode(x)

        if sample:
            return self.reparameterize(mean, logvar)
        else:
            return mean

    @classmethod
    def from_pretrained(cls, model_path: str, device: str = 'cpu'):
        """Load pre-trained VAE from checkpoint.

        Args:
            model_path: Path to checkpoint or Hugging Face model ID
            device: Device to load model on
        """
        model = cls()

        # Try loading from local checkpoint
        try:
            checkpoint = torch.load(model_path, map_location=device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print(f"Loaded VAE from {model_path}")
        except:
            print(f"Could not load from {model_path}, initializing random weights")
            print("Note: Download microsoft/mri-autoencoder-v0.1 from Hugging Face for pre-trained weights")

        return model.to(device)


def vae_loss(
    reconstruction: torch.Tensor,
    target: torch.Tensor,
    mean: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 1e-6,
    mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, dict]:
    """Compute VAE loss with reconstruction and KL divergence.

    Args:
        reconstruction: Reconstructed volume
        target: Original volume
        mean: Latent mean
        logvar: Latent log variance
        kl_weight: Weight for KL divergence term
        mask: Optional mask for body region

    Returns:
        total_loss: Combined loss
        loss_dict: Dictionary of individual losses
    """
    # Handle potential size mismatch from VAE encoder-decoder
    if reconstruction.shape != target.shape:
        # Interpolate reconstruction to match target size
        reconstruction = F.interpolate(
            reconstruction,
            size=target.shape[2:],
            mode='trilinear',
            align_corners=False
        )

    # Reconstruction loss (MSE)
    if mask is not None:
        # Also resize mask if needed
        if mask.shape != target.shape:
            mask = F.interpolate(
                mask.float(),
                size=target.shape[2:],
                mode='trilinear',
                align_corners=False
            )
        recon_loss = ((reconstruction - target) ** 2 * mask).sum() / (mask.sum() + 1e-8)
    else:
        recon_loss = F.mse_loss(reconstruction, target)

    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
    kl_loss = kl_loss / mean.numel()  # Normalize by number of elements

    # Total loss
    total_loss = recon_loss + kl_weight * kl_loss

    loss_dict = {
        'total': total_loss.item(),
        'recon': recon_loss.item(),
        'kl': kl_loss.item()
    }

    return total_loss, loss_dict


def create_medical_vae(
    latent_channels: int = 4,
    base_channels: int = 64,
    device: str = 'cpu'
) -> MedicalVAE3D:
    """Create Medical VAE model.

    Args:
        latent_channels: Number of latent channels (default: 4)
        base_channels: Base channel count (default: 64)
        device: Device to create model on

    Returns:
        MedicalVAE3D model
    """
    model = MedicalVAE3D(
        in_channels=1,
        latent_channels=latent_channels,
        base_channels=base_channels,
        channel_mult=(1, 2, 4, 8)
    )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Medical VAE created with {num_params:,} parameters")
    print(f"Latent compression: 16x (spatial 8x + feature reduction)")
    print(f"Device: {device}")

    return model
