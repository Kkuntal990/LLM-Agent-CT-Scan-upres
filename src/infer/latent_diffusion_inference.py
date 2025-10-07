"""Inference pipeline for 3D Latent Diffusion super-resolution.

Supports fast DDIM sampling with ResShift for CT through-plane SR.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple
from tqdm import tqdm

from ..models.diffusion.resshift_scheduler import DDIMSchedulerResShift


class LatentDiffusionInference:
    """Inference pipeline for Medical Latent Diffusion Model.

    Performs super-resolution in latent space with DDIM sampling.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cuda',
        num_inference_steps: int = 15,
        guidance_scale: float = 1.5,
        predict_residual: bool = True
    ):
        """Initialize inference pipeline.

        Args:
            model: MedicalLatentDiffusion3D model
            device: 'cuda', 'cpu', or 'mps'
            num_inference_steps: Number of DDIM steps (15-25 recommended)
            guidance_scale: Classifier-free guidance scale (1.0-2.0)
            predict_residual: Use ResShift residual prediction
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.predict_residual = predict_residual

        # Create DDIM scheduler
        self.scheduler = DDIMSchedulerResShift(
            num_train_timesteps=1000,
            num_inference_steps=num_inference_steps,
            schedule_type='cosine'
        )

        print(f"Latent Diffusion Inference initialized:")
        print(f"  Device: {device}")
        print(f"  Inference steps: {num_inference_steps}")
        print(f"  Guidance scale: {guidance_scale}")
        print(f"  Prediction type: {'Residual (ResShift)' if predict_residual else 'Noise'}")

    @torch.no_grad()
    def infer_volume(
        self,
        lr_volume: torch.Tensor,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Perform super-resolution on a full volume.

        Args:
            lr_volume: Low-resolution volume, shape (1, 1, D/2, H, W)
            show_progress: Show progress bar

        Returns:
            Super-resolved volume, shape (1, 1, D, H, W)
        """
        self.model.eval()

        # Upsample LR to target HR size
        target_shape = (
            lr_volume.shape[2] * 2,  # Double z-dimension
            lr_volume.shape[3],
            lr_volume.shape[4]
        )

        lr_upsampled = F.interpolate(
            lr_volume,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        ).to(self.device)

        # Encode LR to latent space
        lr_latent = self.model.encode_to_latent(lr_upsampled, sample=False)

        # Generate HR latent using diffusion
        hr_latent = self.sample_latent(
            lr_latent,
            num_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            show_progress=show_progress
        )

        # Decode to pixel space
        sr_volume = self.model.decode_from_latent(hr_latent)

        return sr_volume

    @torch.no_grad()
    def sample_latent(
        self,
        lr_latent: torch.Tensor,
        num_steps: Optional[int] = None,
        guidance_scale: Optional[float] = None,
        show_progress: bool = True
    ) -> torch.Tensor:
        """Sample HR latent using DDIM.

        Args:
            lr_latent: LR conditioning latent
            num_steps: Number of sampling steps (override default)
            guidance_scale: Guidance scale (override default)
            show_progress: Show progress bar

        Returns:
            Generated HR latent
        """
        num_steps = num_steps or self.num_inference_steps
        guidance_scale = guidance_scale or self.guidance_scale

        batch_size = lr_latent.shape[0]
        latent_shape = lr_latent.shape

        # Start from pure noise
        sample = torch.randn(latent_shape, device=self.device)

        # DDIM sampling loop
        timesteps = self.scheduler.timesteps[:num_steps]

        iterator = tqdm(timesteps, desc='Diffusion sampling', disable=not show_progress)

        for t in iterator:
            t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.long)

            # Predict with condition
            pred_cond = self.model(sample, t_batch, lr_latent)

            # Classifier-free guidance
            if guidance_scale != 1.0:
                # Predict without condition
                lr_latent_uncond = torch.zeros_like(lr_latent)
                pred_uncond = self.model(sample, t_batch, lr_latent_uncond)

                # Apply guidance
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            # DDIM step
            sample = self.scheduler.step(
                pred,
                t.item(),
                sample,
                eta=0.0,  # Deterministic
                predict_residual=self.predict_residual
            )

        return sample

    @torch.no_grad()
    def infer_with_patches(
        self,
        lr_volume: torch.Tensor,
        patch_size: Tuple[int, int, int] = (32, 256, 256),
        overlap: Tuple[int, int, int] = (8, 64, 64),
        show_progress: bool = True
    ) -> torch.Tensor:
        """Perform super-resolution using overlapping patches.

        For very large volumes that don't fit in memory.

        Args:
            lr_volume: Low-resolution volume, shape (1, 1, D/2, H, W)
            patch_size: Patch size for processing (in HR space)
            overlap: Overlap between patches
            show_progress: Show progress bar

        Returns:
            Super-resolved volume
        """
        self.model.eval()

        # Upsample LR to target HR size
        target_shape = (
            lr_volume.shape[2] * 2,
            lr_volume.shape[3],
            lr_volume.shape[4]
        )

        lr_upsampled = F.interpolate(
            lr_volume,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        ).to(self.device)

        # Initialize output volume and weight map
        sr_volume = torch.zeros_like(lr_upsampled)
        weight_map = torch.zeros_like(lr_upsampled)

        # Compute patch positions
        d, h, w = target_shape
        pd, ph, pw = patch_size
        od, oh, ow = overlap

        d_positions = list(range(0, d - pd + 1, pd - od)) + [d - pd]
        h_positions = list(range(0, h - ph + 1, ph - oh)) + [h - ph]
        w_positions = list(range(0, w - pw + 1, pw - ow)) + [w - pw]

        total_patches = len(d_positions) * len(h_positions) * len(w_positions)

        # Gaussian weight for blending
        gaussian_weight = self._create_gaussian_weight(patch_size).to(self.device)

        # Process patches
        if show_progress:
            print(f"Processing {total_patches} patches...")

        pbar = tqdm(total=total_patches, desc='Patch inference', disable=not show_progress)

        for d_start in d_positions:
            for h_start in h_positions:
                for w_start in w_positions:
                    # Extract patch
                    lr_patch = lr_upsampled[
                        :, :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw
                    ]

                    # Encode to latent
                    lr_patch_latent = self.model.encode_to_latent(lr_patch, sample=False)

                    # Sample HR latent
                    hr_patch_latent = self.sample_latent(
                        lr_patch_latent,
                        num_steps=self.num_inference_steps,
                        guidance_scale=self.guidance_scale,
                        show_progress=False
                    )

                    # Decode to pixel space
                    sr_patch = self.model.decode_from_latent(hr_patch_latent)

                    # Add to output with Gaussian weighting
                    sr_volume[
                        :, :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw
                    ] += sr_patch * gaussian_weight

                    weight_map[
                        :, :,
                        d_start:d_start + pd,
                        h_start:h_start + ph,
                        w_start:w_start + pw
                    ] += gaussian_weight

                    pbar.update(1)

        pbar.close()

        # Normalize by weight map
        sr_volume = sr_volume / (weight_map + 1e-8)

        return sr_volume

    def _create_gaussian_weight(self, patch_size: Tuple[int, int, int]) -> torch.Tensor:
        """Create 3D Gaussian weight for patch blending."""
        d, h, w = patch_size

        # 1D Gaussian
        def gaussian_1d(size, sigma=None):
            if sigma is None:
                sigma = size / 6.0
            x = np.arange(size) - (size - 1) / 2.0
            g = np.exp(-(x ** 2) / (2 * sigma ** 2))
            return g / g.max()

        # 3D Gaussian
        g_d = gaussian_1d(d)
        g_h = gaussian_1d(h)
        g_w = gaussian_1d(w)

        weight_3d = np.outer(g_d, np.outer(g_h, g_w).flatten()).reshape(d, h, w)
        weight_3d = weight_3d.astype(np.float32)

        # Add batch and channel dimensions
        weight_tensor = torch.from_numpy(weight_3d).unsqueeze(0).unsqueeze(0)

        return weight_tensor

    @torch.no_grad()
    def denormalize_hu(self, volume: torch.Tensor) -> torch.Tensor:
        """Convert normalized [0, 1] volume back to HU values.

        Args:
            volume: Normalized volume

        Returns:
            Volume in HU units
        """
        # Reverse normalization: HU = volume * 4095 - 1024
        volume_hu = volume * 4095.0 - 1024.0
        return volume_hu


def create_inference_pipeline(
    model_checkpoint: str,
    vae_checkpoint: str,
    device: str = 'cuda',
    num_inference_steps: int = 15,
    guidance_scale: float = 1.5
) -> LatentDiffusionInference:
    """Create inference pipeline from checkpoints.

    Args:
        model_checkpoint: Path to latent diffusion checkpoint
        vae_checkpoint: Path to VAE checkpoint
        device: Device to run inference on
        num_inference_steps: Number of DDIM steps
        guidance_scale: Classifier-free guidance scale

    Returns:
        LatentDiffusionInference pipeline
    """
    from ..models.diffusion.medical_vae import create_medical_vae
    from ..models.diffusion.unet3d_latent import create_latent_unet3d
    from ..models.diffusion.controlnet3d import create_medical_latent_diffusion

    # Load VAE
    vae = create_medical_vae(latent_channels=4, device=device)
    vae_ckpt = torch.load(vae_checkpoint, map_location=device)
    vae.load_state_dict(vae_ckpt['model_state_dict'])

    # Load UNet
    unet = create_latent_unet3d(latent_channels=4, device=device)

    # Create full model
    model = create_medical_latent_diffusion(vae, unet, latent_channels=4)

    # Load checkpoint
    ckpt = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    # Create inference pipeline
    pipeline = LatentDiffusionInference(
        model=model,
        device=device,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        predict_residual=ckpt.get('config', {}).get('predict_residual', True)
    )

    print(f"✓ Inference pipeline loaded from checkpoints")

    return pipeline
