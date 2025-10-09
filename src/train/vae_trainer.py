"""Trainer for fine-tuning Medical VAE on CT data.

Adapts pre-trained MRI VAE to CT imaging domain.
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

from ..models.diffusion.medical_vae import vae_loss


class VAETrainer:
    """Trainer for Medical VAE fine-tuning with CPU/GPU support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'cuda',
        learning_rate: float = 1e-4,
        kl_weight: float = 1e-6,
        checkpoint_dir: str = './checkpoints/vae',
        log_dir: str = './runs/vae'
    ):
        """Initialize VAE trainer.

        Args:
            model: Medical VAE model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: 'cuda', 'cpu', or 'mps'
            learning_rate: Learning rate
            kl_weight: Weight for KL divergence loss
            checkpoint_dir: Directory to save checkpoints
            log_dir: TensorBoard log directory
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.kl_weight = kl_weight

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=0.01
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3
        )

        # Checkpointing and logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        print(f"VAE Trainer initialized on {device}")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            # Get HR volume (no need for LR in VAE training)
            hr = batch['hr'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            reconstruction, mean, logvar = self.model(hr, sample=True)

            # Compute loss
            loss, loss_dict = vae_loss(
                reconstruction, hr, mean, logvar,
                kl_weight=self.kl_weight,
                mask=mask
            )

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss_dict['total']:.4f}",
                'recon': f"{loss_dict['recon']:.4f}"
            })

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def validate(self) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        total_losses = {'total': 0.0, 'recon': 0.0, 'kl': 0.0}
        num_batches = 0

        pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
        for batch in pbar:
            hr = batch['hr'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass
            reconstruction, mean, logvar = self.model(hr, sample=False)

            # Compute loss
            loss, loss_dict = vae_loss(
                reconstruction, hr, mean, logvar,
                kl_weight=self.kl_weight,
                mask=mask
            )

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    @torch.no_grad()
    def compute_psnr(self) -> float:
        """Compute PSNR on validation set."""
        self.model.eval()
        psnr_values = []

        for batch in self.val_loader:
            hr = batch['hr'].to(self.device)
            reconstruction, _, _ = self.model(hr, sample=False)

            # Compute PSNR
            mse = F.mse_loss(reconstruction, hr, reduction='none')
            mse = mse.view(mse.shape[0], -1).mean(dim=1)
            psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
            psnr_values.extend(psnr.cpu().numpy())

        return sum(psnr_values) / len(psnr_values)

    def train(self, num_epochs: int, save_every: int = 5):
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting VAE training on {self.device}")
        print(f"Total epochs: {num_epochs}")
        print(f"KL weight: {self.kl_weight}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Compute PSNR
            val_psnr = self.compute_psnr()

            # Log to TensorBoard
            for key in train_losses:
                self.writer.add_scalar(f'Loss/train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Loss/val_{key}', val_losses[key], epoch)

            self.writer.add_scalar('PSNR/val', val_psnr, epoch)
            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Train Loss: {train_losses['total']:.4f} (Recon: {train_losses['recon']:.4f}, KL: {train_losses['kl']:.6f})")
            print(f"  Val Loss: {val_losses['total']:.4f} (PSNR: {val_psnr:.2f} dB)")
            print(f"  Time: {epoch_time:.2f}s")

            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])

            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_vae.pth')
                print(f"  ✓ Saved best model (val_loss={self.best_val_loss:.4f}, PSNR={val_psnr:.2f} dB)")

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'vae_epoch_{epoch}.pth')

        print("VAE training complete!")
        self.writer.close()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'kl_weight': self.kl_weight
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
        self.best_val_loss = checkpoint['best_val_loss']
        self.kl_weight = checkpoint.get('kl_weight', self.kl_weight)

        print(f"Loaded VAE checkpoint from epoch {self.current_epoch}")
        print(f"Best val loss: {self.best_val_loss:.4f}")
