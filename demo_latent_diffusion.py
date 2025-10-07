"""Demo script for CT super-resolution using Latent Diffusion Model.

End-to-end inference on CT volumes.
"""

import sys
import argparse
from pathlib import Path
import time

import torch
import nibabel as nib
import numpy as np

from src.infer.latent_diffusion_inference import create_inference_pipeline


def load_nifti(path: str) -> tuple:
    """Load NIfTI file and return volume + header."""
    img = nib.load(path)
    volume = np.asarray(img.dataobj).astype(np.float32)
    return volume, img.affine, img.header


def save_nifti(volume: np.ndarray, affine, header, path: str):
    """Save volume as NIfTI file."""
    img = nib.Nifti1Image(volume, affine, header)
    nib.save(img, path)
    print(f"✓ Saved to: {path}")


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    """Normalize HU values to [0, 1] range."""
    volume = np.clip(volume, -1024, 3071)
    volume = (volume + 1024) / 4095.0
    return volume.astype(np.float32)


def denormalize_hu(volume: np.ndarray) -> np.ndarray:
    """Convert [0, 1] back to HU values."""
    volume = volume * 4095.0 - 1024.0
    return volume.astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='CT Super-Resolution using Latent Diffusion')

    # Input/Output
    parser.add_argument('input', type=str,
                        help='Path to input low-resolution NIfTI file')
    parser.add_argument('output', type=str,
                        help='Path to output super-resolved NIfTI file')

    # Model checkpoints
    parser.add_argument('--vae-checkpoint', type=str, required=True,
                        help='Path to VAE checkpoint')
    parser.add_argument('--diffusion-checkpoint', type=str, required=True,
                        help='Path to latent diffusion checkpoint')

    # Inference arguments
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu', 'mps'],
                        help='Device for inference (default: cuda)')
    parser.add_argument('--num-steps', type=int, default=15,
                        help='Number of DDIM sampling steps (default: 15)')
    parser.add_argument('--guidance-scale', type=float, default=1.5,
                        help='Classifier-free guidance scale (default: 1.5)')
    parser.add_argument('--use-patches', action='store_true',
                        help='Use patch-based inference for large volumes')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[32, 256, 256],
                        help='Patch size D H W (default: 32 256 256)')
    parser.add_argument('--overlap', type=int, nargs=3, default=[8, 64, 64],
                        help='Patch overlap D H W (default: 8 64 64)')

    args = parser.parse_args()

    # Check device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        args.device = 'cpu'
    elif args.device == 'mps' and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        args.device = 'cpu'

    print("=" * 80)
    print("CT Super-Resolution with Latent Diffusion")
    print("=" * 80)
    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Device: {args.device}")
    print(f"DDIM steps: {args.num_steps}")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Patch-based: {args.use_patches}")
    print("=" * 80)

    # Load input volume
    print(f"\nLoading input volume...")
    lr_volume, affine, header = load_nifti(args.input)
    print(f"  Shape: {lr_volume.shape}")
    print(f"  HU range: [{lr_volume.min():.1f}, {lr_volume.max():.1f}]")

    # Normalize
    lr_volume_norm = normalize_hu(lr_volume)

    # Convert to tensor
    lr_tensor = torch.from_numpy(lr_volume_norm).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

    # Create inference pipeline
    print(f"\nLoading models...")
    pipeline = create_inference_pipeline(
        model_checkpoint=args.diffusion_checkpoint,
        vae_checkpoint=args.vae_checkpoint,
        device=args.device,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale
    )

    # Run inference
    print(f"\nRunning super-resolution...")
    start_time = time.time()

    if args.use_patches:
        print(f"Using patch-based inference (patch size: {args.patch_size}, overlap: {args.overlap})")
        sr_tensor = pipeline.infer_with_patches(
            lr_tensor,
            patch_size=tuple(args.patch_size),
            overlap=tuple(args.overlap),
            show_progress=True
        )
    else:
        sr_tensor = pipeline.infer_volume(
            lr_tensor,
            show_progress=True
        )

    inference_time = time.time() - start_time

    # Denormalize
    sr_volume_norm = sr_tensor.squeeze(0).squeeze(0).cpu().numpy()
    sr_volume = denormalize_hu(sr_volume_norm)

    print(f"\n✓ Super-resolution complete!")
    print(f"  Output shape: {sr_volume.shape}")
    print(f"  HU range: [{sr_volume.min():.1f}, {sr_volume.max():.1f}]")
    print(f"  Inference time: {inference_time:.2f}s")

    # Update header for new spacing
    new_header = header.copy()
    zooms = list(header.get_zooms())
    if len(zooms) >= 3:
        zooms[2] = zooms[2] / 2.0  # Half the z-spacing (2x resolution)
    new_header.set_zooms(zooms)

    # Save output
    print(f"\nSaving output...")
    save_nifti(sr_volume, affine, new_header, args.output)

    print("\n" + "=" * 80)
    print("Super-resolution complete!")
    print(f"Upscaling factor: 2x (z-axis)")
    print(f"Input: {lr_volume.shape} → Output: {sr_volume.shape}")
    print(f"Total time: {inference_time:.2f}s")
    print("=" * 80)


if __name__ == '__main__':
    main()
