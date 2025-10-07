"""Diffusion models for CT super-resolution."""

from .noise_schedulers import NoiseScheduler, DDPMScheduler, DDIMScheduler
from .resshift_scheduler import ResShiftScheduler, DDIMSchedulerResShift, create_resshift_scheduler
from .medical_vae import MedicalVAE3D, create_medical_vae, vae_loss
from .unet3d_latent import LatentDiffusionUNet3D, create_latent_unet3d
from .controlnet3d import IRControlNet3D, MedicalLatentDiffusion3D, create_medical_latent_diffusion

__all__ = [
    # Original schedulers
    'NoiseScheduler',
    'DDPMScheduler',
    'DDIMScheduler',
    # ResShift schedulers (NEW - SOTA)
    'ResShiftScheduler',
    'DDIMSchedulerResShift',
    'create_resshift_scheduler',
    # Medical VAE (NEW - SOTA)
    'MedicalVAE3D',
    'create_medical_vae',
    'vae_loss',
    # Latent Diffusion UNet (NEW - SOTA)
    'LatentDiffusionUNet3D',
    'create_latent_unet3d',
    # ControlNet and full model (NEW - SOTA)
    'IRControlNet3D',
    'MedicalLatentDiffusion3D',
    'create_medical_latent_diffusion',
]
