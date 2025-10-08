"""Noise schedulers for DDPM and DDIM."""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class NoiseScheduler:
    """Base class for noise schedulers in diffusion models."""

    def __init__(
        self,
        num_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        schedule_type: str = 'linear'
    ):
        """Initialize noise scheduler.

        Args:
            num_timesteps: Number of diffusion timesteps
            beta_start: Starting beta value
            beta_end: Ending beta value
            schedule_type: Type of noise schedule ('linear', 'cosine', 'non_uniform')
        """
        self.num_timesteps = num_timesteps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.schedule_type = schedule_type

        # Compute beta schedule
        self.betas = self._get_beta_schedule()

        # Pre-compute useful quantities
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat([
            torch.tensor([1.0]), self.alphas_cumprod[:-1]
        ])

        # For noise addition
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # For denoising
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)

        # For posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_log_variance_clipped = torch.log(
            torch.clamp(self.posterior_variance, min=1e-20)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)
        )

    def _get_beta_schedule(self) -> torch.Tensor:
        """Get beta schedule based on schedule type."""
        if self.schedule_type == 'linear':
            return torch.linspace(self.beta_start, self.beta_end, self.num_timesteps)

        elif self.schedule_type == 'cosine':
            # Improved DDPM cosine schedule
            return self._cosine_beta_schedule()

        elif self.schedule_type == 'non_uniform':
            # Fast-DDPM style non-uniform schedule for super-resolution
            return self._non_uniform_beta_schedule()

        else:
            raise ValueError(f"Unknown schedule type: {self.schedule_type}")

    def _cosine_beta_schedule(self, s: float = 0.008) -> torch.Tensor:
        """Cosine schedule from Improved DDPM paper."""
        steps = self.num_timesteps + 1
        x = torch.linspace(0, self.num_timesteps, steps)
        alphas_cumprod = torch.cos(((x / self.num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clamp(betas, 0.0001, 0.9999)

    def _non_uniform_beta_schedule(self) -> torch.Tensor:
        """Non-uniform schedule optimized for super-resolution (Fast-DDPM style)."""
        # For SR, concentrate noise at critical timesteps
        # Use quadratic spacing for fewer timesteps
        t = torch.linspace(0, 1, self.num_timesteps)
        # Quadratic spacing emphasizes later timesteps
        betas = self.beta_start + (self.beta_end - self.beta_start) * (t ** 2)
        return betas

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


class DDPMScheduler(NoiseScheduler):
    """DDPM scheduler for training and sampling."""

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        predict_epsilon: bool = True,
        eta: float = 1.0
    ) -> torch.Tensor:
        """Reverse diffusion step (denoising).

        Args:
            model_output: Output from model (predicted noise or x_0)
            timestep: Current timestep
            sample: Current noisy sample x_t
            predict_epsilon: Whether model predicts noise (True) or x_0 (False)
            eta: Variance scaling (1.0 = DDPM, 0.0 = DDIM deterministic)

        Returns:
            Previous sample x_{t-1}
        """
        t = timestep

        # Get model prediction of x_0
        if predict_epsilon:
            # Model predicts noise: x_0 = (x_t - sqrt(1-alpha_bar) * epsilon) / sqrt(alpha_bar)
            pred_original_sample = (
                sample - self.sqrt_one_minus_alphas_cumprod[t] * model_output
            ) / self.sqrt_alphas_cumprod[t]
        else:
            # Model directly predicts x_0
            pred_original_sample = model_output

        # Clip to valid range
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # Compute posterior mean
        pred_prev_sample = (
            self.posterior_mean_coef1[t] * pred_original_sample +
            self.posterior_mean_coef2[t] * sample
        )

        # Add noise for stochasticity
        if t > 0:
            noise = torch.randn_like(sample)
            variance = (eta ** 2) * self.posterior_variance[t]
            pred_prev_sample = pred_prev_sample + torch.sqrt(variance) * noise

        return pred_prev_sample


class DDIMScheduler(NoiseScheduler):
    """DDIM scheduler for fast deterministic sampling."""

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 50,
        **kwargs
    ):
        """Initialize DDIM scheduler.

        Args:
            num_train_timesteps: Number of timesteps used during training
            num_inference_steps: Number of timesteps for inference (can be much smaller)
            **kwargs: Additional arguments for NoiseScheduler
        """
        super().__init__(num_timesteps=num_train_timesteps, **kwargs)
        self.num_inference_steps = num_inference_steps

        # Create subset of timesteps for inference
        step_ratio = self.num_timesteps // self.num_inference_steps
        self.timesteps = torch.arange(0, self.num_timesteps, step_ratio).long()
        self.timesteps = torch.flip(self.timesteps, dims=[0])  # Reverse for sampling

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        predict_epsilon: bool = True
    ) -> torch.Tensor:
        """DDIM reverse step (deterministic if eta=0).

        Args:
            model_output: Predicted noise from model
            timestep: Current timestep t
            sample: Current sample x_t
            eta: Stochasticity parameter (0 = deterministic)
            predict_epsilon: Whether model predicts noise

        Returns:
            Previous sample x_{t-1}
        """
        # Get previous timestep
        prev_timestep = timestep - self.num_timesteps // self.num_inference_steps
        prev_timestep = max(prev_timestep, 0)

        # Get alpha values
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        # Predict x_0
        if predict_epsilon:
            pred_original_sample = (
                sample - torch.sqrt(1 - alpha_prod_t) * model_output
            ) / torch.sqrt(alpha_prod_t)
        else:
            pred_original_sample = model_output

        # Clip
        pred_original_sample = torch.clamp(pred_original_sample, -1.0, 1.0)

        # Compute variance
        variance = self._get_variance(timestep, prev_timestep)
        std_dev_t = eta * torch.sqrt(variance)

        # Compute direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output

        # Compute x_{t-1}
        pred_prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original_sample +
            pred_sample_direction
        )

        # Add noise if eta > 0
        if eta > 0:
            noise = torch.randn_like(sample)
            pred_prev_sample = pred_prev_sample + std_dev_t * noise

        return pred_prev_sample

    def _get_variance(self, timestep: int, prev_timestep: int) -> torch.Tensor:
        """Compute variance for DDIM sampling."""
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else torch.tensor(1.0)

        beta_prod_t = 1 - alpha_prod_t
        beta_prod_t_prev = 1 - alpha_prod_t_prev

        variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        return variance
