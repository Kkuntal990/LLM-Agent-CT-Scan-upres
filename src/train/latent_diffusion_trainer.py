"""Trainer for 3D Latent Diffusion Model with CPU/GPU support.

Trains ResShift-based diffusion in latent space for CT super-resolution.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import time

from ..models.diffusion.resshift_scheduler import ResShiftScheduler, DDIMSchedulerResShift


class LatentDiffusionTrainer:
    """Trainer for Latent Diffusion Model with ResShift.

    Supports both CPU and GPU training with optional mixed precision.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 2e-5,
        num_train_timesteps: int = 1000,
        num_inference_steps: int = 15,
        predict_residual: bool = True,
        use_cfg: bool = True,
        cfg_dropout: float = 0.1,
        checkpoint_dir: str = './checkpoints/latent_diffusion',
        log_dir: str = './runs/latent_diffusion',
        use_amp: bool = False
    ):
        """Initialize Latent Diffusion trainer.

        Args:
            model: MedicalLatentDiffusion3D model
            train_loader: Training dataloader (returns latents)
            val_loader: Validation dataloader
            device: 'cuda', 'cpu', or 'mps'
            learning_rate: Learning rate
            num_train_timesteps: Training timesteps
            num_inference_steps: Inference steps (for validation sampling)
            predict_residual: Use ResShift residual prediction
            use_cfg: Use classifier-free guidance
            cfg_dropout: Dropout rate for unconditional training
            checkpoint_dir: Directory to save checkpoints
            log_dir: TensorBoard log directory
            use_amp: Use automatic mixed precision (GPU only)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.predict_residual = predict_residual
        self.use_cfg = use_cfg
        self.cfg_dropout = cfg_dropout
        self.use_amp = use_amp and device in ['cuda', 'mps']

        # Noise scheduler
        self.noise_scheduler = ResShiftScheduler(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            schedule_type='cosine'
        )

        # Sampler for validation
        self.sampler = DDIMSchedulerResShift(
            num_train_timesteps=num_train_timesteps,
            num_inference_steps=num_inference_steps,
            schedule_type='cosine'
        )

        # Optimizer
        # Only train UNet and ControlNet, freeze VAE
        trainable_params = []
        for name, param in model.named_parameters():
            if 'vae' not in name:
                trainable_params.append(param)
            else:
                param.requires_grad = False

        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=learning_rate,
            weight_decay=0.01,
            betas=(0.9, 0.999)
        )

        # Learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=len(train_loader) * 50,  # Assume 50 epochs
            pct_start=0.05,  # 5% warmup
            anneal_strategy='cos'
        )

        # Mixed precision scaler
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and device == 'cuda' else None

        # Checkpointing and logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')

        print(f"Latent Diffusion Trainer initialized on {device}")
        print(f"Prediction type: {'Residual (ResShift)' if predict_residual else 'Noise'}")
        print(f"Classifier-free guidance: {use_cfg}")
        print(f"Mixed precision: {self.use_amp}")
        print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            # Get latents
            hr_latent = batch['hr_latent'].to(self.device)
            lr_latent = batch['lr_latent'].to(self.device)

            batch_size = hr_latent.shape[0]

            # Classifier-free guidance: randomly drop condition
            if self.use_cfg and torch.rand(1).item() < self.cfg_dropout:
                # Unconditional training (zero out LR condition)
                lr_latent_cond = torch.zeros_like(lr_latent)
            else:
                lr_latent_cond = lr_latent

            # Sample random timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()

            # Sample noise
            noise = torch.randn_like(hr_latent)

            # Forward diffusion: add noise to HR latent
            noisy_hr_latent = self.noise_scheduler.add_noise(hr_latent, noise, timesteps)

            # Forward pass with mixed precision
            self.optimizer.zero_grad()

            if self.use_amp and self.device == 'cuda':
                with torch.cuda.amp.autocast():
                    # Predict residual or noise
                    prediction = self.model(noisy_hr_latent, timesteps, lr_latent_cond)

                    # Compute target
                    if self.predict_residual:
                        # ResShift: predict residual = noisy - clean
                        target = noisy_hr_latent - hr_latent
                    else:
                        # Standard DDPM: predict noise
                        target = noise

                    # Loss
                    loss = F.mse_loss(prediction, target)

                # Backward with scaler
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Predict residual or noise
                prediction = self.model(noisy_hr_latent, timesteps, lr_latent_cond)

                # Compute target
                if self.predict_residual:
                    target = noisy_hr_latent - hr_latent
                else:
                    target = noise

                # Loss
                loss = F.mse_loss(prediction, target)

                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()

            self.scheduler.step()

            # Accumulate loss
            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })

            # Log to TensorBoard every 100 steps
            if self.global_step % 100 == 0:
                self.writer.add_scalar('Train/loss_step', loss.item(), self.global_step)
                self.writer.add_scalar('Train/lr', self.optimizer.param_groups[0]['lr'], self.global_step)

        # Average loss
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        for batch in pbar:
            hr_latent = batch['hr_latent'].to(self.device)
            lr_latent = batch['lr_latent'].to(self.device)

            batch_size = hr_latent.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0, self.noise_scheduler.num_train_timesteps,
                (batch_size,), device=self.device
            ).long()

            # Sample noise
            noise = torch.randn_like(hr_latent)

            # Forward diffusion
            noisy_hr_latent = self.noise_scheduler.add_noise(hr_latent, noise, timesteps)

            # Predict
            prediction = self.model(noisy_hr_latent, timesteps, lr_latent)

            # Compute target and loss
            if self.predict_residual:
                target = noisy_hr_latent - hr_latent
            else:
                target = noise

            loss = F.mse_loss(prediction, target)

            total_loss += loss.item()
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Average loss
        avg_loss = total_loss / num_batches
        return {'loss': avg_loss}

    @torch.no_grad()
    def sample(
        self,
        lr_latent: torch.Tensor,
        num_steps: int = 15,
        guidance_scale: float = 1.0
    ) -> torch.Tensor:
        """Generate HR latent using DDIM sampling.

        Args:
            lr_latent: LR conditioning latent
            num_steps: Number of sampling steps
            guidance_scale: Classifier-free guidance scale (1.0 = no guidance)

        Returns:
            Generated HR latent
        """
        self.model.eval()

        batch_size = lr_latent.shape[0]

        # Start from pure noise
        sample = torch.randn(
            batch_size,
            self.model.latent_channels,
            lr_latent.shape[2],
            lr_latent.shape[3],
            lr_latent.shape[4],
            device=self.device
        )

        # DDIM sampling
        timesteps = self.sampler.timesteps[:num_steps]

        for t in tqdm(timesteps, desc='Sampling', leave=False):
            t_batch = torch.full((batch_size,), t.item(), device=self.device, dtype=torch.long)

            # Predict with condition
            pred_cond = self.model(sample, t_batch, lr_latent)

            # Classifier-free guidance
            if self.use_cfg and guidance_scale != 1.0:
                # Predict without condition
                lr_latent_uncond = torch.zeros_like(lr_latent)
                pred_uncond = self.model(sample, t_batch, lr_latent_uncond)

                # Apply guidance
                pred = pred_uncond + guidance_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            # Sampling step
            sample = self.sampler.step(
                pred,
                t.item(),
                sample,
                eta=0.0,
                predict_residual=self.predict_residual
            )

        return sample

    def train(
        self,
        num_epochs: int,
        save_every: int = 5,
        sample_every: int = 10
    ):
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
            sample_every: Generate samples every N epochs
        """
        print(f"Starting Latent Diffusion training on {self.device}")
        print(f"Total epochs: {num_epochs}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Log to TensorBoard
            self.writer.add_scalar('Loss/train', train_losses['loss'], epoch)
            self.writer.add_scalar('Loss/val', val_losses['loss'], epoch)

            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Train Loss: {train_losses['loss']:.4f}")
            print(f"  Val Loss: {val_losses['loss']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Generate samples
            if (epoch + 1) % sample_every == 0:
                print("  Generating samples...")
                self.generate_and_log_samples(epoch)

            # Save best model
            if val_losses['loss'] < self.best_val_loss:
                self.best_val_loss = val_losses['loss']
                self.save_checkpoint('best_latent_diffusion.pth')
                print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'latent_diffusion_epoch_{epoch}.pth')

        print("Latent Diffusion training complete!")
        self.writer.close()

    @torch.no_grad()
    def generate_and_log_samples(self, epoch: int):
        """Generate and log sample images to TensorBoard."""
        # Get one batch from validation
        batch = next(iter(self.val_loader))
        lr_latent = batch['lr_latent'][:4].to(self.device)  # Take 4 samples
        hr_latent = batch['hr_latent'][:4].to(self.device)

        # Generate samples
        generated_latent = self.sample(lr_latent, num_steps=15, guidance_scale=1.5)

        # Decode to pixel space
        lr_decoded = self.model.decode_from_latent(lr_latent)
        hr_decoded = self.model.decode_from_latent(hr_latent)
        gen_decoded = self.model.decode_from_latent(generated_latent)

        # Log central slices to TensorBoard
        for i in range(4):
            mid_slice = lr_decoded.shape[2] // 2

            # Get central slice
            lr_slice = lr_decoded[i, 0, mid_slice].cpu()
            hr_slice = hr_decoded[i, 0, mid_slice].cpu()
            gen_slice = gen_decoded[i, 0, mid_slice].cpu()

            # Normalize to [0, 1] for visualization
            lr_slice = (lr_slice - lr_slice.min()) / (lr_slice.max() - lr_slice.min() + 1e-8)
            hr_slice = (hr_slice - hr_slice.min()) / (hr_slice.max() - hr_slice.min() + 1e-8)
            gen_slice = (gen_slice - gen_slice.min()) / (gen_slice.max() - gen_slice.min() + 1e-8)

            self.writer.add_image(f'Sample_{i}/LR', lr_slice.unsqueeze(0), epoch)
            self.writer.add_image(f'Sample_{i}/Generated', gen_slice.unsqueeze(0), epoch)
            self.writer.add_image(f'Sample_{i}/HR', hr_slice.unsqueeze(0), epoch)

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': {
                'predict_residual': self.predict_residual,
                'use_cfg': self.use_cfg,
                'cfg_dropout': self.cfg_dropout
            }
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint."""
        checkpoint = torch.load(
            self.checkpoint_dir / filename,
            map_location=self.device
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']

        config = checkpoint.get('config', {})
        self.predict_residual = config.get('predict_residual', True)
        self.use_cfg = config.get('use_cfg', True)
        self.cfg_dropout = config.get('cfg_dropout', 0.1)

        print(f"Loaded checkpoint from epoch {self.current_epoch}")
        print(f"Global step: {self.global_step}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
