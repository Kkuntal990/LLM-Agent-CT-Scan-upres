"""Fine-tune Medical VAE on CT data.

Adapts pre-trained MRI VAE (microsoft/mri-autoencoder-v0.1) to CT imaging.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.diffusion.medical_vae import create_medical_vae
from src.train.vae_trainer import VAETrainer
from src.train.dataset import create_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Fine-tune Medical VAE on CT data')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing NIfTI files')
    parser.add_argument('--train-split', type=str, required=True,
                        help='Path to train split file (list of filenames)')
    parser.add_argument('--val-split', type=str, required=True,
                        help='Path to validation split file')

    # Model arguments
    parser.add_argument('--latent-channels', type=int, default=4,
                        help='Number of latent channels (default: 4)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channel count (default: 64)')
    parser.add_argument('--pretrained', type=str, default=None,
                        help='Path to pre-trained VAE checkpoint (optional)')

    # Training arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to train on (default: cuda)')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of epochs (default: 20)')
    parser.add_argument('--batch-size', type=int, default=2,
                        help='Batch size (default: 2)')
    parser.add_argument('--learning-rate', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--kl-weight', type=float, default=1e-6,
                        help='KL divergence weight (default: 1e-6)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[32, 128, 128],
                        help='Training patch size D H W (default: 32 128 128)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')

    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/vae',
                        help='Checkpoint directory (default: ./checkpoints/vae)')
    parser.add_argument('--log-dir', type=str, default='./runs/vae',
                        help='TensorBoard log directory (default: ./runs/vae)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("VAE Fine-tuning Configuration")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"KL weight: {args.kl_weight}")
    print(f"Patch size: {args.patch_size}")
    print(f"Latent channels: {args.latent_channels}")
    print("=" * 80)

    # Load file lists
    with open(args.train_split, 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]

    with open(args.val_split, 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]

    print(f"\nDataset:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")

    # Create model
    print(f"\nCreating Medical VAE...")
    model = create_medical_vae(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        device=args.device
    )

    # Load pre-trained weights if provided
    if args.pretrained:
        print(f"Loading pre-trained weights from {args.pretrained}")
        try:
            checkpoint = torch.load(args.pretrained, map_location=args.device)
            if 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            print("✓ Pre-trained weights loaded successfully")
        except Exception as e:
            print(f"⚠ Could not load pre-trained weights: {e}")
            print("  Continuing with random initialization")

    # Create dataloaders
    print(f"\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        train_files=train_files,
        val_files=val_files,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        patch_size=tuple(args.patch_size),
        downsample_factor=2,
        hr_spacing=1.0,
        use_body_mask=True
    )

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = VAETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        kl_weight=args.kl_weight,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir
    )

    # Train
    print(f"\nStarting training...")
    print("=" * 80)
    trainer.train(num_epochs=args.epochs, save_every=args.save_every)

    print("\n" + "=" * 80)
    print("VAE fine-tuning complete!")
    print(f"Best model saved to: {Path(args.checkpoint_dir) / 'best_vae.pth'}")
    print(f"TensorBoard logs: {args.log_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
