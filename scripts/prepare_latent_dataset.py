"""Pre-compute latent representations for efficient diffusion training.

Encodes CT volumes to latent space using trained VAE.
"""

import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from src.models.diffusion.medical_vae import create_medical_vae
from src.train.latent_dataset import prepare_latent_dataset


def main():
    parser = argparse.ArgumentParser(description='Prepare latent dataset from CT volumes')

    # Data arguments
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Directory containing NIfTI files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for latent .npz files')
    parser.add_argument('--file-list', type=str, required=True,
                        help='Path to file list (all files to encode)')

    # Model arguments
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                        help='Path to trained VAE checkpoint')
    parser.add_argument('--latent-channels', type=int, default=4,
                        help='Number of latent channels (default: 4)')
    parser.add_argument('--base-channels', type=int, default=64,
                        help='Base channel count (default: 64)')

    # Processing arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device for encoding (default: cuda)')
    parser.add_argument('--downsample-factor', type=int, default=2,
                        help='Z-axis downsampling factor (default: 2)')
    parser.add_argument('--hr-spacing', type=float, default=1.0,
                        help='HR z-spacing in mm (default: 1.0)')

    args = parser.parse_args()

    # Check device availability
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("Latent Dataset Preparation")
    print("=" * 80)
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Device: {args.device}")
    print(f"Downsample factor: {args.downsample_factor}")
    print("=" * 80)

    # Load file list
    with open(args.file_list, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]

    print(f"\nFiles to encode: {len(file_list)}")

    # Load VAE model
    print(f"\nLoading VAE model...")
    vae = create_medical_vae(
        latent_channels=args.latent_channels,
        base_channels=args.base_channels,
        device=args.device
    )

    # Load checkpoint
    print(f"Loading VAE checkpoint from {args.vae_checkpoint}")
    checkpoint = torch.load(args.vae_checkpoint, map_location=args.device)
    vae.load_state_dict(checkpoint['model_state_dict'])
    vae.eval()
    print("✓ VAE loaded successfully")

    # Prepare latent dataset
    print(f"\nStarting latent encoding...")
    print("=" * 80)

    prepare_latent_dataset(
        data_dir=args.data_dir,
        file_list=file_list,
        vae_model=vae,
        output_dir=args.output_dir,
        device=args.device,
        downsample_factor=args.downsample_factor,
        hr_spacing=args.hr_spacing
    )

    print("\n" + "=" * 80)
    print("Latent dataset preparation complete!")
    print(f"Output directory: {args.output_dir}")
    print(f"Encoded {len(file_list)} volumes")
    print("=" * 80)


if __name__ == '__main__':
    main()
