"""ResShift scheduler for efficient diffusion-based super-resolution.

Based on "ResShift: Efficient Diffusion Model for Image Super-resolution by Residual Shifting"
(NeurIPS 2023 Spotlight, TPAMI 2024)
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class ResShiftScheduler:
    """ResShift noise scheduler for efficient diffusion.

    Key innovation: Predicts residual shift instead of noise for faster convergence.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 15,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'cosine',
        shift_scale: float = 1.0
    ):
        """Initialize ResShift scheduler.

        Args:
            num_train_timesteps: Number of diffusion timesteps for training
            num_inference_steps: Number of steps for inference (much fewer)
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: Type of noise schedule ('linear', 'cosine')
            shift_scale: Scaling factor for residual shift
        """
        self.num_train_timesteps = num_train_timesteps
        self.num_inference_steps = num_inference_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type
        self.shift_scale = shift_scale

        # Compute beta schedule
        self.betas = self._get_beta_schedule()

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])

        # For noise addition (forward process)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For ResShift residual prediction
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # Inference timesteps (subset of training timesteps)
        self.timesteps = self._get_inference_timesteps()

    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule based on schedule type."""
        if self.schedule_type == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.num_train_timesteps)

        elif self.schedule_type == 'cosine':
            # Improved DDPM cosine schedule
            return self._cosine_beta_schedule()

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from Improved DDPM paper."""
        steps = self.num_train_timesteps + 1
        x = torch.linspace(0, self.num_train_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_train_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _get_inference_timesteps(self) -> torch.Tensor:
        """Get subset of timesteps for fast inference."""
        step_ratio = self.num_train_timesteps // self.num_inference_steps
        timesteps = torch.arange(0, self.num_train_timesteps, step_ratio).long()
        return torch.flip(timesteps, dims=[0])  # Reverse for sampling

    def add_noise(
        self,
        x_start: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor
    ) -> torch.Tensor:
        """Add noise to clean images (forward diffusion).

        Args:
            x_start: Clean images, shape (B, C, D, H, W)
            noise: Noise to add, shape (B, C, D, H, W)
            timesteps: Timesteps for each sample, shape (B,)

        Returns:
            Noisy images at timestep t
        """
        # Gather coefficients for each sample's timestep
        sqrt_alpha_prod = self._extract(self.sqrt_alphas_cumprod, timesteps, x_start.shape)
        sqrt_one_minus_alpha_prod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, timesteps, x_start.shape
        )

        # Forward process: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        noisy_images = sqrt_alpha_prod * x_start + sqrt_one_minus_alpha_prod * noise
        return noisy_images

    def get_residual_from_noise(
        self,
        x_t: torch.Tensor,
        noise: torch.Tensor,
        timestep: int
    ) -> torch.Tensor:
        """Convert noise to residual shift (ResShift formulation).

        ResShift: residual = x_t - x_0
        From noise: x_0 = (x_t - sqrt(1-alpha_bar) * noise) / sqrt(alpha_bar)
        So: residual = x_t - x_0

        Args:
            x_t: Noisy image
            noise: Predicted or true noise
            timestep: Current timestep

        Returns:
            Residual shift
        """
        sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[timestep]
        sqrt_recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[timestep]

        # Predict x_0 from noise
        x_0_pred = sqrt_recip_alpha * x_t - sqrt_recipm1_alpha * noise

        # Residual = x_t - x_0
        residual = x_t - x_0_pred
        return residual * self.shift_scale

    def get_x0_from_residual(
        self,
        x_t: torch.Tensor,
        residual: torch.Tensor
    ) -> torch.Tensor:
        """Get x_0 prediction from residual shift.

        Args:
            x_t: Noisy image
            residual: Predicted residual

        Returns:
            Predicted clean image x_0
        """
        return x_t - residual / self.shift_scale

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_residual: bool = True,
        eta: float = 0.0
    ) -> torch.Tensor:
        """Perform one denoising step.

        Args:
            model_output: Output from model (residual or noise)
            timestep: Current timestep t
            sample: Current sample x_t
            predict_residual: If True, model predicts residual; if False, noise
            eta: Stochasticity parameter (0 = deterministic)

        Returns:
            Previous sample x_{t-1}
        """
        # Get previous timestep
        prev_timestep = timestep - self.num_train_timesteps // self.num_inference_steps
        prev_timestep = max(prev_timestep, 0)

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        # Predict x_0
        if predict_residual:
            # Model predicts residual shift
            pred_original_sample = self.get_x0_from_residual(sample, model_output)
        else:
            # Model predicts noise (standard DDPM)
            sqrt_recip_alpha = self.sqrt_recip_alphas_cumprod[timestep]
            sqrt_recipm1_alpha = self.sqrt_recipm1_alphas_cumprod[timestep]
            pred_original_sample = sqrt_recip_alpha * sample - sqrt_recipm1_alpha * model_output

        # Clip to valid range
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # Compute direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev) * model_output

        # Compute x_{t-1}
        pred_prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        # Add noise if eta > 0
        if eta > 0 and timestep > 0:
            variance = self._get_variance(timestep, prev_timestep)
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + eta * torch.sqrt(variance) * noise

        return pred_prev_sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for sampling."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance

    def _extract(self, a: torch.Tensor, t: torch.Tensor, x_shape: tuple) -> torch.Tensor:
        """Extract coefficients at specified timesteps and reshape for broadcasting.

        Args:
            a: Coefficient array, shape (num_timesteps,)
            t: Timesteps to extract, shape (B,)
            x_shape: Shape of target tensor (B, C, D, H, W)

        Returns:
            Extracted coefficients, shape (B, 1, 1, 1, 1)
        """
        batch_size = t.shape[0]
        out = a.to(t.device).gather(0, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))


class DDIMSchedulerResShift:
    """DDIM scheduler adapted for ResShift residual prediction."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 15,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'cosine'
    ):
        self.resshift = ResShiftScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=schedule_type
        )
        self.timesteps = self.resshift.timesteps
        self.num_inference_steps = num_inference_steps

    def add_noise(self, x_start, noise, timesteps):
        return self.resshift.add_noise(x_start, noise, timesteps)

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        predict_residual: bool = True
    ) -> torch.Tensor:
        """DDIM step with ResShift residual prediction."""
        return self.resshift.step(
            model_output,
            timestep,
            sample,
            predict_residual=predict_residual,
            eta=eta
        )


def create_resshift_scheduler(
    num_train_timesteps: int = 1000,
    num_inference_steps: int = 15,
    schedule_type: str = 'cosine'
) -> ResShiftScheduler:
    """Create ResShift scheduler with optimal settings.

    Args:
        num_train_timesteps: Training timesteps (default: 1000)
        num_inference_steps: Inference steps (default: 15 for fast sampling)
        schedule_type: 'linear' or 'cosine'

    Returns:
        ResShiftScheduler instance
    """
    scheduler = ResShiftScheduler(
        num_train_timesteps=num_train_timesteps,
        num_inference_steps=num_inference_steps,
        beta_start=0.0001,
        beta_end=0.02,
        schedule_type=schedule_type,
        shift_scale=1.0
    )

    print(f"ResShift scheduler created:")
    print(f"  Training timesteps: {num_train_timesteps}")
    print(f"  Inference steps: {num_inference_steps}")
    print(f"  Schedule: {schedule_type}")
    print(f"  Speedup: {num_train_timesteps // num_inference_steps}x")

    return scheduler
