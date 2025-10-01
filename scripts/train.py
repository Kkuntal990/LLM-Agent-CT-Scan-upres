#!/usr/bin/env python3
"""Training script for supervised CT super-resolution."""

import argparse
import torch
from pathlib import Path

from src.models.unet3d_simple import create_simple_model
from src.train.dataset import create_dataloaders
from src.train.trainer import SupervisedTrainer
from src.data.prepare_lidc import load_split


def main():
    parser = argparse.ArgumentParser(description='Train CT SR model')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing processed NIfTI files')
    parser.add_argument('--train-split', type=str, required=True,
                        help='Path to train split file')
    parser.add_argument('--val-split', type=str, required=True,
                        help='Path to validation split file')
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log-dir', type=str, default='./runs',
                        help='Directory for TensorBoard logs')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--upscale-factor', type=int, default=2,
                        help='Z-axis upsampling factor')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[32, 128, 128],
                        help='Training patch size (D H W)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'])
    parser.add_argument('--resume', type=str, default=None,
                        help='Resume from checkpoint')

    args = parser.parse_args()

    # Check device
    if args.device == 'mps' and not torch.backends.mps.is_available():
        print("Warning: MPS not available, using CPU")
        args.device = 'cpu'

    print("=" * 60)
    print("CT Super-Resolution Training")
    print("=" * 60)
    print(f"Data directory:  {args.data_dir}")
    print(f"Device:          {args.device}")
    print(f"Epochs:          {args.epochs}")
    print(f"Batch size:      {args.batch_size}")
    print(f"Learning rate:   {args.lr}")
    print(f"Upscale factor:  {args.upscale_factor}x")
    print("=" * 60)

    # Load splits
    train_files = load_split(args.train_split)
    val_files = load_split(args.val_split)
    print(f"\nLoaded {len(train_files)} train and {len(val_files)} val cases")

    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=0,
        patch_size=tuple(args.patch_size),
        downsample_factor=args.upscale_factor
    )

    # Create model
    print("\nCreating model...")
    model = create_simple_model(device=args.device)

    # Create trainer
    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.lr,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Resume if checkpoint provided
    if args.resume:
        print(f"\nResuming from {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print("\nStarting training...\n")
    trainer.train(num_epochs=args.epochs, save_every=5)


if __name__ == '__main__':
    main()
