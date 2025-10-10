"""3D UNet for latent diffusion-based super-resolution.

Uses ResShift architecture with 3D convolutions and attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
from .resnet_blocks import ResNetBlock3D
from .attention import SelfAttention3D
from .timestep_embedding import TimestepEmbedding


class DownsampleBlock3D(nn.Module):
    """3D downsampling using strided convolution."""

    def __init__(self, channels: int, anisotropic: bool = True):
        super().__init__()
        # Anisotropic stride for through-plane SR: (2,2,2) or (1,2,2)
        stride = (1, 2, 2) if anisotropic else (2, 2, 2)
        self.conv = nn.Conv3d(channels, channels, 3, stride=stride, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class UpsampleBlock3D(nn.Module):
    """3D upsampling using transposed convolution."""

    def __init__(self, channels: int, anisotropic: bool = True):
        super().__init__()
        # Anisotropic stride for through-plane SR
        stride = (1, 2, 2) if anisotropic else (2, 2, 2)
        self.conv = nn.ConvTranspose3d(channels, channels, 2, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


class LatentDiffusionUNet3D(nn.Module):
    """3D UNet for diffusion in latent space.

    Optimized for CT through-plane super-resolution with:
    - ResShift residual prediction
    - Anisotropic operations for z-axis SR
    - 3D self-attention at key resolutions
    - FiLM conditioning on timestep
    """

    def __init__(
        self,
        in_channels: int = 8,  # 4 latent + 4 condition
        out_channels: int = 4,  # Latent residual
        model_channels: int = 192,
        num_res_blocks: int = 2,
        attention_levels: Tuple[int, ...] = (1, 2),  # At 1/2 and 1/4 resolution
        channel_mult: Tuple[int, ...] = (1, 2, 4, 8),
        num_heads: int = 8,
        use_3d_attention: bool = True,
        use_checkpoint: bool = False,
        anisotropic: bool = True  # For through-plane SR
    ):
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.attention_levels = attention_levels
        self.channel_mult = channel_mult
        self.num_heads = num_heads
        self.use_checkpoint = use_checkpoint
        self.anisotropic = anisotropic

        # Timestep embedding
        time_embed_dim = model_channels * 4
        self.time_embed = TimestepEmbedding(
            timestep_dim=model_channels,
            embedding_dim=time_embed_dim
        )

        # Input convolution
        self.conv_in = nn.Conv3d(in_channels, model_channels, 3, padding=1)

        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        self.downsample_blocks = nn.ModuleList()

        ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch = model_channels * mult

            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResNetBlock3D(
                        in_channels=ch,
                        out_channels=out_ch,
                        timestep_emb_dim=time_embed_dim,
                        dropout=0.0
                    )
                )
                ch = out_ch

                # Add attention at specified levels
                if level in attention_levels and use_3d_attention:
                    self.encoder_blocks.append(
                        SelfAttention3D(ch, num_heads=num_heads)
                    )

            # Downsample (except last level)
            if level < len(channel_mult) - 1:
                self.downsample_blocks.append(
                    DownsampleBlock3D(ch, anisotropic=anisotropic)
                )

        # Bottleneck (middle blocks)
        self.mid_block1 = ResNetBlock3D(ch, ch, time_embed_dim, groups=8, dropout=0.0)
        self.mid_attn = SelfAttention3D(ch, num_heads=num_heads) if use_3d_attention else nn.Identity()
        self.mid_block2 = ResNetBlock3D(ch, ch, time_embed_dim, groups=8, dropout=0.0)

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.upsample_blocks = nn.ModuleList()

        # Track skip connection channels (reverse order)
        skip_ch_list = []
        temp_ch = model_channels
        for level, mult in enumerate(channel_mult):
            out_ch_enc = model_channels * mult
            for _ in range(num_res_blocks):
                temp_ch = out_ch_enc
                skip_ch_list.append(temp_ch)
            if level < len(channel_mult) - 1:
                skip_ch_list.append(temp_ch)  # Downsample skip
        skip_ch_list = list(reversed(skip_ch_list))

        skip_idx = 0
        for level in reversed(range(len(channel_mult))):
            out_ch = model_channels * channel_mult[level]

            for i in range(num_res_blocks + 1):
                # Get skip connection channel count (0 if no skip available)
                if skip_idx < len(skip_ch_list):
                    skip_ch = skip_ch_list[skip_idx]
                    skip_idx += 1
                else:
                    skip_ch = 0  # No skip for last block

                # Input = current channels + skip channels
                in_ch = ch + skip_ch if skip_ch > 0 else ch

                self.decoder_blocks.append(
                    ResNetBlock3D(
                        in_channels=in_ch,
                        out_channels=out_ch,
                        timestep_emb_dim=time_embed_dim,
                        dropout=0.0
                    )
                )
                ch = out_ch

                # Add attention
                if level in attention_levels and use_3d_attention:
                    self.decoder_blocks.append(
                        SelfAttention3D(ch, num_heads=num_heads)
                    )

            # Upsample (except first level)
            if level > 0:
                self.upsample_blocks.append(
                    UpsampleBlock3D(ch, anisotropic=anisotropic)
                )

        # Output convolution
        self.norm_out = nn.GroupNorm(8, ch)
        self.conv_out = nn.Conv3d(ch, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        """Forward pass through UNet.

        Args:
            x: Input tensor, shape (B, in_channels, D, H, W)
               in_channels = 8: [noisy_latent(4), lr_latent_condition(4)]
            timesteps: Timestep indices, shape (B,)

        Returns:
            Predicted residual or noise, shape (B, out_channels, D, H, W)
        """
        # Timestep embedding
        t_emb = self.time_embed(timesteps)  # (B, time_embed_dim)

        # Input
        h = self.conv_in(x)

        # Encoder with skip connections
        skip_connections = []
        enc_block_idx = 0
        down_block_idx = 0

        for level in range(len(self.channel_mult)):
            for _ in range(self.num_res_blocks):
                h = self.encoder_blocks[enc_block_idx](h, t_emb)
                enc_block_idx += 1

                # Attention if present
                if enc_block_idx < len(self.encoder_blocks) and \
                   isinstance(self.encoder_blocks[enc_block_idx], SelfAttention3D):
                    h = self.encoder_blocks[enc_block_idx](h)
                    enc_block_idx += 1

                skip_connections.append(h)

            # Downsample
            if level < len(self.channel_mult) - 1:
                h = self.downsample_blocks[down_block_idx](h)
                down_block_idx += 1
                skip_connections.append(h)

        # Bottleneck
        h = self.mid_block1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_block2(h, t_emb)

        # Decoder with skip connections
        dec_block_idx = 0
        up_block_idx = 0

        for level in reversed(range(len(self.channel_mult))):
            for i in range(self.num_res_blocks + 1):
                # Concatenate skip connection if available
                if len(skip_connections) > 0:
                    skip = skip_connections.pop()

                    # Match spatial dimensions if needed (due to anisotropic operations)
                    if h.shape[2:] != skip.shape[2:]:
                        h = F.interpolate(h, size=skip.shape[2:], mode='trilinear', align_corners=False)

                    h = torch.cat([h, skip], dim=1)

                h = self.decoder_blocks[dec_block_idx](h, t_emb)
                dec_block_idx += 1

                # Attention if present
                if dec_block_idx < len(self.decoder_blocks) and \
                   isinstance(self.decoder_blocks[dec_block_idx], SelfAttention3D):
                    h = self.decoder_blocks[dec_block_idx](h)
                    dec_block_idx += 1

            # Upsample
            if level > 0:
                h = self.upsample_blocks[up_block_idx](h)
                up_block_idx += 1

        # Output
        h = self.norm_out(h)
        h = F.silu(h)
        h = self.conv_out(h)

        return h


def create_latent_unet3d(
    latent_channels: int = 4,
    model_channels: int = 192,
    num_heads: int = 8,
    use_checkpoint: bool = False,
    device: str = 'cpu'
) -> LatentDiffusionUNet3D:
    """Create 3D Latent Diffusion UNet model.

    Args:
        latent_channels: Number of latent channels from VAE (default: 4)
        model_channels: Base channel count (default: 192)
        num_heads: Number of attention heads (default: 8)
        use_checkpoint: Use gradient checkpointing (default: False)
        device: Device to create model on

    Returns:
        LatentDiffusionUNet3D model
    """
    model = LatentDiffusionUNet3D(
        in_channels=latent_channels * 2,  # Noisy latent + condition
        out_channels=latent_channels,     # Residual prediction
        model_channels=model_channels,
        num_res_blocks=2,
        attention_levels=(1, 2),
        channel_mult=(1, 2, 4, 8),
        num_heads=num_heads,
        use_3d_attention=True,
        use_checkpoint=use_checkpoint,
        anisotropic=True  # For through-plane SR
    )

    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Latent Diffusion UNet3D created with {num_params:,} parameters")
    print(f"Model channels: {model_channels}")
    print(f"Attention heads: {num_heads}")
    print(f"Anisotropic operations: True (optimized for z-axis SR)")
    print(f"Device: {device}")

    return model
