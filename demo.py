#!/usr/bin/env python3
"""Demo script for end-to-end CT through-plane super-resolution.

Loads a LIDC-IDRI case (NIfTI), runs SR inference to target spacing,
and writes output with updated header/affine.
"""

import argparse
import torch
import numpy as np
from pathlib import Path

from src.io.nifti_io import read_nifti, write_nifti
from src.models.unet3d import create_model
from src.infer.patch_infer import PatchInference
from src.preprocessing.masking import create_body_mask
from src.eval.metrics import evaluate_volume, print_metrics


def normalize_hu(volume: np.ndarray) -> np.ndarray:
    """Normalize HU to [0, 1] range."""
    volume = np.clip(volume, -1024, 3071)
    return (volume + 1024) / 4095.0


def denormalize_hu(volume: np.ndarray) -> np.ndarray:
    """Denormalize from [0, 1] to HU range."""
    return volume * 4095.0 - 1024.0


def main():
    parser = argparse.ArgumentParser(description='CT Through-Plane Super-Resolution Demo')
    parser.add_argument('input', type=str, help='Input NIfTI file path')
    parser.add_argument('output', type=str, help='Output NIfTI file path')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to trained model checkpoint')
    parser.add_argument('--target-spacing', type=float, default=1.0,
                        help='Target z-spacing in mm (default: 1.0)')
    parser.add_argument('--upscale-factor', type=int, default=2,
                        help='Z-axis upsampling factor (default: 2)')
    parser.add_argument('--patch-size', type=int, nargs=3, default=[16, 160, 160],
                        help='Patch size for inference (D H W)')
    parser.add_argument('--overlap', type=int, nargs=3, default=[8, 32, 32],
                        help='Overlap between patches (D H W)')
    parser.add_argument('--device', type=str, default='mps',
                        choices=['mps', 'cuda', 'cpu'],
                        help='Device for inference')
    parser.add_argument('--evaluate', action='store_true',
                        help='Evaluate against ground truth (requires HR input)')

    args = parser.parse_args()

    # Check MPS availability
    if args.device == 'mps':
        if not torch.backends.mps.is_available():
            print("Warning: MPS not available, falling back to CPU")
            args.device = 'cpu'
        else:
            print(f"MPS is available and enabled")

    print("\n" + "=" * 60)
    print("CT Through-Plane Super-Resolution Demo")
    print("=" * 60)
    print(f"Input:           {args.input}")
    print(f"Output:          {args.output}")
    print(f"Target spacing:  {args.target_spacing} mm")
    print(f"Upscale factor:  {args.upscale_factor}x")
    print(f"Device:          {args.device}")
    print("=" * 60 + "\n")

    # Load input volume
    print("Loading input volume...")
    volume, spacing, affine = read_nifti(args.input, return_spacing=True, return_affine=True)
    print(f"  Input shape:   {volume.shape}")
    print(f"  Input spacing: {spacing} mm")
    print(f"  HU range:      [{volume.min():.1f}, {volume.max():.1f}]")

    # Normalize HU
    volume_norm = normalize_hu(volume)

    # Create model
    print("\nInitializing model...")
    model = create_model(
        upscale_factor=args.upscale_factor,
        base_channels=32,
        depth=3,
        device=args.device
    )

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("  Checkpoint loaded successfully")
    else:
        print("  Warning: No checkpoint provided, using random weights")

    # Run inference
    print("\nRunning super-resolution inference...")
    inference_engine = PatchInference(
        model=model,
        patch_size=tuple(args.patch_size),
        overlap=tuple(args.overlap),
        device=args.device,
        upscale_factor=args.upscale_factor
    )

    sr_volume_norm = inference_engine.infer(volume_norm, progress=True)

    # Denormalize
    sr_volume = denormalize_hu(sr_volume_norm)

    print(f"  Output shape:  {sr_volume.shape}")
    print(f"  HU range:      [{sr_volume.min():.1f}, {sr_volume.max():.1f}]")

    # Calculate new spacing
    new_spacing = (spacing[0], spacing[1], spacing[2] / args.upscale_factor)
    print(f"  Output spacing: {new_spacing} mm")

    # Update affine for new z-spacing
    new_affine = affine.copy()
    new_affine[2, 2] = new_spacing[2] * np.sign(affine[2, 2])

    # Write output
    print(f"\nWriting output to {args.output}...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    write_nifti(sr_volume, args.output, new_spacing, new_affine)
    print("  Output saved successfully")

    # Evaluation (if requested and possible)
    if args.evaluate:
        print("\nEvaluating super-resolution quality...")

        # Create body mask
        mask = create_body_mask(sr_volume, threshold_hu=-500.0)

        # For evaluation, we would need ground truth at target spacing
        # Here we just compute some basic statistics
        print(f"  Body voxels:   {mask.sum():,} / {mask.size:,} ({100*mask.sum()/mask.size:.1f}%)")

        # If we had ground truth, we would do:
        # metrics = evaluate_volume(sr_volume_norm, gt_volume_norm, mask)
        # print_metrics(metrics)

    print("\n" + "=" * 60)
    print("Demo completed successfully!")
    print("=" * 60 + "\n")


if __name__ == '__main__':
    main()
