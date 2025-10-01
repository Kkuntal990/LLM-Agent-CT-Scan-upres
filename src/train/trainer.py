"""Supervised trainer for CT super-resolution on MPS."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Dict
import time

from .losses import CombinedLoss


class SupervisedTrainer:
    """Trainer for supervised through-plane SR with MPS support."""

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = 'mps',
        learning_rate: float = 1e-4,
        checkpoint_dir: str = './checkpoints',
        log_dir: str = './runs',
        use_amp: bool = False
    ):
        """Initialize trainer.

        Args:
            model: Neural network model
            train_loader: Training dataloader
            val_loader: Validation dataloader
            device: Device ('mps', 'cuda', or 'cpu')
            learning_rate: Learning rate for Adam optimizer
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
            use_amp: Use automatic mixed precision (experimental on MPS)
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.use_amp = use_amp

        # Loss and optimizer
        self.criterion = CombinedLoss(l1_weight=1.0, ssim_weight=0.1, grad_weight=0.1)
        self.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )

        # Checkpointing and logging
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')

        # AMP scaler (for FP16 training)
        if use_amp and device == 'mps':
            print("Warning: AMP on MPS is experimental and may be unstable")

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch.

        Returns:
            Dictionary of average training metrics
        """
        self.model.train()
        total_losses = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'grad': 0.0}
        num_batches = 0

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch} [Train]')
        for batch in pbar:
            # Move to device
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)
            mask = batch.get('mask', None)
            if mask is not None:
                mask = mask.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if self.use_amp:
                with torch.amp.autocast(device_type=self.device):
                    pred = self.model(lr)
                    loss, loss_dict = self.criterion(pred, hr, mask)
            else:
                pred = self.model(lr)
                loss, loss_dict = self.criterion(pred, hr, mask)

            # Backward pass
            loss.backward()
            self.optimizer.step()

            # Accumulate losses
            for key, value in loss_dict.items():
                total_losses[key] += value
            num_batches += 1

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def validate(self) -> Dict[str, float]:
        """Validate on validation set.

        Returns:
            Dictionary of average validation metrics
        """
        self.model.eval()
        total_losses = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'grad': 0.0}
        num_batches = 0

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.current_epoch} [Val]')
            for batch in pbar:
                lr = batch['lr'].to(self.device)
                hr = batch['hr'].to(self.device)
                mask = batch.get('mask', None)
                if mask is not None:
                    mask = mask.to(self.device)

                # Forward pass
                pred = self.model(lr)
                loss, loss_dict = self.criterion(pred, hr, mask)

                # Accumulate losses
                for key, value in loss_dict.items():
                    total_losses[key] += value
                num_batches += 1

                pbar.set_postfix({'loss': f"{loss_dict['total']:.4f}"})

        # Average losses
        avg_losses = {k: v / num_batches for k, v in total_losses.items()}
        return avg_losses

    def train(self, num_epochs: int, save_every: int = 5):
        """Train for multiple epochs.

        Args:
            num_epochs: Number of epochs to train
            save_every: Save checkpoint every N epochs
        """
        print(f"Starting training on {self.device}")
        print(f"Total epochs: {num_epochs}")

        for epoch in range(num_epochs):
            self.current_epoch = epoch
            epoch_start = time.time()

            # Train
            train_losses = self.train_epoch()

            # Validate
            val_losses = self.validate()

            # Log to TensorBoard
            for key in train_losses:
                self.writer.add_scalar(f'Loss/train_{key}', train_losses[key], epoch)
                self.writer.add_scalar(f'Loss/val_{key}', val_losses[key], epoch)

            self.writer.add_scalar('Learning_Rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print summary
            epoch_time = time.time() - epoch_start
            print(f"\nEpoch {epoch}/{num_epochs-1}")
            print(f"  Train Loss: {train_losses['total']:.4f}")
            print(f"  Val Loss: {val_losses['total']:.4f}")
            print(f"  Time: {epoch_time:.2f}s")

            # Learning rate scheduling
            self.scheduler.step(val_losses['total'])

            # Save best model
            if val_losses['total'] < self.best_val_loss:
                self.best_val_loss = val_losses['total']
                self.save_checkpoint('best_model.pth')
                print(f"  Saved best model (val_loss={self.best_val_loss:.4f})")

            # Periodic checkpoint
            if (epoch + 1) % save_every == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch}.pth')

        print("Training complete!")
        self.writer.close()

    def save_checkpoint(self, filename: str):
        """Save model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss
        }
        torch.save(checkpoint, self.checkpoint_dir / filename)

    def load_checkpoint(self, filename: str):
        """Load model checkpoint.

        Args:
            filename: Checkpoint filename
        """
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.current_epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        print(f"Loaded checkpoint from epoch {self.current_epoch}")
