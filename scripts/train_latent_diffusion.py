"""Train 3D Latent Diffusion Model for CT super-resolution.

Uses ResShift for efficient diffusion in latent space.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.diffusion.medical_vae import create_medical_vae
from src.models.diffusion.unet3d_latent import create_latent_unet3d
from src.models.diffusion.controlnet3d import create_medical_latent_diffusion
from src.train.latent_diffusion_trainer import LatentDiffusionTrainer
from src.train.latent_dataset import create_latent_dataloaders


def main():
    parser = argparse.ArgumentParser(description='Train Latent Diffusion Model for CT SR')

    # Data arguments
    parser.add_argument('--latent-dir', type=str, required=True,
                        help='Directory containing pre-computed latent .npz files')
    parser.add_argument('--train-split', type=str, required=True,
                        help='Path to train split file (list of base filenames)')
    parser.add_argument('--val-split', type=str, required=True,
                        help='Path to validation split file')

    # Model arguments
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--latent-channels', type=int, default=4,
                        help='Number of latent channels (default: 4)')
    parser.add_argument('--model-channels', type=int, default=192,
                        help='Base model channels (default: 192)')
    parser.add_argument('--num-heads', type=int, default=8,
                        help='Number of attention heads (default: 8)')

    # Diffusion arguments
    parser.add_argument('--num-train-timesteps', type=int, default=1000,
                        help='Number of training timesteps (default: 1000)')
    parser.add_argument('--num-inference-steps', type=int, default=15,
                        help='Number of inference steps (default: 15)')
    parser.add_argument('--predict-residual', action='store_true', default=True,
                        help='Use ResShift residual prediction (default: True)')
    parser.add_argument('--no-predict-residual', dest='predict_residual', action='store_false',
                        help='Use standard noise prediction instead of ResShift')

    # Classifier-free guidance
    parser.add_argument('--use-cfg', action='store_true', default=True,
                        help='Use classifier-free guidance (default: True)')
    parser.add_argument('--no-cfg', dest='use_cfg', action='store_false',
                        help='Disable classifier-free guidance')
    parser.add_argument('--cfg-dropout', type=float, default=0.1,
                        help='CFG dropout rate for unconditional training (default: 0.1)')

    # Training arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device to train on (default: cuda)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--batch-size', type=int, default=4,
                        help='Batch size (default: 4)')
    parser.add_argument('--learning-rate', type=float, default=2e-5,
                        help='Learning rate (default: 2e-5)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[16, 64, 64],
                        help='Latent patch size D H W (default: 16 64 64)')
    parser.add_argument('--num-workers', type=int, default=0,
                        help='Number of dataloader workers (default: 0)')
    parser.add_argument('--use-amp', action='store_true',
                        help='Use automatic mixed precision (GPU only)')
    parser.add_argument('--gradient-checkpointing', action='store_true',
                        help='Use gradient checkpointing to save memory')

    # Output arguments
    parser.add_argument('--checkpoint-dir', type=str, default='./checkpoints/latent_diffusion',
                        help='Checkpoint directory (default: ./checkpoints/latent_diffusion)')
    parser.add_argument('--log-dir', type=str, default='./runs/latent_diffusion',
                        help='TensorBoard log directory (default: ./runs/latent_diffusion)')
    parser.add_argument('--save-every', type=int, default=5,
                        help='Save checkpoint every N epochs (default: 5)')
    parser.add_argument('--sample-every', type=int, default=10,
                        help='Generate samples every N epochs (default: 10)')

    # Resume training
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training from')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'

    # Disable AMP on CPU
    if args.device == 'cpu':
        args.use_amp = False

    print("=" * 80)
    print("Latent Diffusion Training Configuration")
    print("=" * 80)
    print(f"Latent directory: {args.latent_dir}")
    print(f"Device: {args.device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Patch size (latent): {args.patch_size}")
    print(f"Model channels: {args.model_channels}")
    print(f"Attention heads: {args.num_heads}")
    print(f"Training timesteps: {args.num_train_timesteps}")
    print(f"Inference steps: {args.num_inference_steps}")
    print(f"Prediction type: {'ResShift (residual)' if args.predict_residual else 'Noise'}")
    print(f"Classifier-free guidance: {args.use_cfg} (dropout={args.cfg_dropout})")
    print(f"Mixed precision: {args.use_amp}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print("=" * 80)

    # Load file lists
    with open(args.train_split, 'r') as f:
        train_files = [line.strip() for line in f if line.strip()]

    with open(args.val_split, 'r') as f:
        val_files = [line.strip() for line in f if line.strip()]

    print(f"\nDataset:")
    print(f"  Train files: {len(train_files)}")
    print(f"  Val files: {len(val_files)}")

    # Load VAE
    print(f"\nLoading VAE...")
    vae = create_medical_vae(
        latent_channels=args.latent_channels,
        device=args.device
    )
    vae_checkpoint = torch.load(args.vae_checkpoint, map_location=args.device)
    vae.load_state_dict(vae_checkpoint['model_state_dict'])
    vae.eval()
    # Freeze VAE
    for param in vae.parameters():
        param.requires_grad = False
    print("✓ VAE loaded and frozen")

    # Create UNet
    print(f"\nCreating Latent Diffusion UNet...")
    unet = create_latent_unet3d(
        latent_channels=args.latent_channels,
        model_channels=args.model_channels,
        num_heads=args.num_heads,
        use_checkpoint=args.gradient_checkpointing,
        device=args.device
    )

    # Create full model
    print(f"\nCreating Medical Latent Diffusion Model...")
    model = create_medical_latent_diffusion(
        vae=vae,
        unet=unet,
        latent_channels=args.latent_channels
    )

    # Create dataloaders
    print(f"\nCreating latent dataloaders...")
    train_loader, val_loader = create_latent_dataloaders(
        latent_dir=args.latent_dir,
        train_files=train_files,
        val_files=val_files,
        batch_size=args.batch_size,
        patch_size=tuple(args.patch_size),
        num_workers=args.num_workers
    )

    # Create trainer
    print(f"\nInitializing trainer...")
    trainer = LatentDiffusionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
        num_train_timesteps=args.num_train_timesteps,
        num_inference_steps=args.num_inference_steps,
        predict_residual=args.predict_residual,
        use_cfg=args.use_cfg,
        cfg_dropout=args.cfg_dropout,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
        use_amp=args.use_amp
    )

    # Resume from checkpoint if provided
    if args.resume:
        print(f"\nResuming from checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print(f"\nStarting training...")
    print("=" * 80)
    trainer.train(
        num_epochs=args.epochs,
        save_every=args.save_every,
        sample_every=args.sample_every
    )

    print("\n" + "=" * 80)
    print("Latent Diffusion training complete!")
    print(f"Best model saved to: {Path(args.checkpoint_dir) / 'best_latent_diffusion.pth'}")
    print(f"TensorBoard logs: {args.log_dir}")
    print("=" * 80)


if __name__ == '__main__':
    main()
