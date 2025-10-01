#!/usr/bin/env python3
"""Create synthetic sample data for testing the pipeline without downloading LIDC-IDRI.

This generates fake CT volumes with realistic properties for development and testing.
"""

import argparse
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.nifti_io import write_nifti


def generate_synthetic_ct(
    shape=(150, 256, 256),
    spacing=(0.7, 0.7, 2.5),
    add_structures=True
):
    """Generate a synthetic CT volume with realistic HU values.

    Args:
        shape: Volume shape (D, H, W)
        spacing: Voxel spacing (sx, sy, sz) in mm
        add_structures: Add synthetic anatomical structures

    Returns:
        volume: Synthetic CT volume in HU
    """
    d, h, w = shape

    # Initialize with air
    volume = np.ones(shape, dtype=np.float32) * -1000

    # Create body region (ellipsoid)
    z, y, x = np.ogrid[:d, :h, :w]

    # Body ellipsoid
    body_mask = (
        ((z - d/2) / (d/2.5))**2 +
        ((y - h/2) / (h/2.5))**2 +
        ((x - w/2) / (w/2.5))**2
    ) < 1

    # Fill body with soft tissue
    volume[body_mask] = np.random.normal(40, 20, volume[body_mask].shape)

    if add_structures:
        # Add some bones (ribs, spine)
        # Spine (vertical structure)
        spine_z = slice(int(d*0.3), int(d*0.7))
        spine_y = slice(int(h*0.35), int(h*0.5))
        spine_x = slice(int(w*0.45), int(w*0.55))
        volume[spine_z, spine_y, spine_x] += np.random.normal(350, 50,
            volume[spine_z, spine_y, spine_x].shape)

        # Ribs (curved structures)
        for i in range(5):
            rib_z = int(d*0.3 + i*d*0.1)
            rib_y_start = int(h*0.3)
            rib_y_end = int(h*0.7)
            for y_pos in range(rib_y_start, rib_y_end, 5):
                x_offset = int(20 * np.sin((y_pos - rib_y_start) / 20))
                x_pos = int(w/2) + x_offset
                if 0 <= x_pos < w:
                    volume[rib_z-1:rib_z+2, y_pos-1:y_pos+2, x_pos-1:x_pos+2] += 300

        # Lungs (low density regions)
        lung_left_x = slice(int(w*0.25), int(w*0.45))
        lung_right_x = slice(int(w*0.55), int(w*0.75))
        lung_y = slice(int(h*0.3), int(h*0.7))
        lung_z = slice(int(d*0.25), int(d*0.75))

        volume[lung_z, lung_y, lung_left_x] = np.random.normal(-800, 50,
            volume[lung_z, lung_y, lung_left_x].shape)
        volume[lung_z, lung_y, lung_right_x] = np.random.normal(-800, 50,
            volume[lung_z, lung_y, lung_right_x].shape)

        # Add a few nodules (for realism)
        for _ in range(3):
            nodule_z = np.random.randint(lung_z.start, lung_z.stop)
            nodule_y = np.random.randint(lung_y.start, lung_y.stop)
            nodule_x = np.random.randint(lung_left_x.start, lung_right_x.stop)
            nodule_radius = np.random.randint(3, 8)

            for dz in range(-nodule_radius, nodule_radius+1):
                for dy in range(-nodule_radius, nodule_radius+1):
                    for dx in range(-nodule_radius, nodule_radius+1):
                        if dz**2 + dy**2 + dx**2 <= nodule_radius**2:
                            z_idx = nodule_z + dz
                            y_idx = nodule_y + dy
                            x_idx = nodule_x + dx
                            if (0 <= z_idx < d and 0 <= y_idx < h and 0 <= x_idx < w):
                                volume[z_idx, y_idx, x_idx] = np.random.normal(-100, 30)

    # Clip to realistic CT range
    volume = np.clip(volume, -1024, 3071)

    return volume


def main():
    parser = argparse.ArgumentParser(description='Create synthetic sample data for testing')
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/unprocessed',
        help='Output directory for sample NIfTI files'
    )
    parser.add_argument(
        '--num-samples',
        type=int,
        default=5,
        help='Number of sample volumes to generate'
    )
    parser.add_argument(
        '--shape',
        type=int,
        nargs=3,
        default=[150, 256, 256],
        help='Volume shape (D H W)'
    )
    parser.add_argument(
        '--spacing',
        type=float,
        nargs=3,
        default=[0.7, 0.7, 2.5],
        help='Voxel spacing (sx sy sz) in mm'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Synthetic Sample Data Generator")
    print("=" * 70)
    print(f"Output directory: {output_dir}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Volume shape: {args.shape}")
    print(f"Spacing: {args.spacing} mm")
    print("=" * 70)
    print()

    for i in range(args.num_samples):
        patient_id = f"SAMPLE-{i+1:04d}"
        print(f"Generating {patient_id}...")

        # Generate volume
        volume = generate_synthetic_ct(
            shape=tuple(args.shape),
            spacing=tuple(args.spacing),
            add_structures=True
        )

        # Save as NIfTI
        output_path = output_dir / f"{patient_id}.nii.gz"
        write_nifti(volume, str(output_path), tuple(args.spacing))

        print(f"  ✓ Saved to {output_path}")
        print(f"    Shape: {volume.shape}")
        print(f"    HU range: [{volume.min():.1f}, {volume.max():.1f}]")
        print()

    print("=" * 70)
    print(f"Generated {args.num_samples} synthetic samples")
    print("=" * 70)
    print()
    print("Next steps:")
    print("1. These files are already in NIfTI format, so you can use them directly")
    print()
    print("2. Create train/val/test splits:")
    print(f"   python -c \"")
    print(f"import json")
    print(f"from pathlib import Path")
    print(f"files = [f.name for f in Path('{output_dir}').glob('*.nii.gz')]")
    print(f"splits_dir = Path('./data/processed/splits')")
    print(f"splits_dir.mkdir(parents=True, exist_ok=True)")
    print(f"with open(splits_dir / 'train.txt', 'w') as f: f.write('\\\\n'.join(files[:3]))")
    print(f"with open(splits_dir / 'val.txt', 'w') as f: f.write('\\\\n'.join(files[3:4]))")
    print(f"with open(splits_dir / 'test.txt', 'w') as f: f.write('\\\\n'.join(files[4:]))")
    print(f"   \"")
    print()
    print("3. Train on synthetic data:")
    print(f"   python scripts/train.py \\")
    print(f"     --data-dir {output_dir} \\")
    print(f"     --train-split ./data/processed/splits/train.txt \\")
    print(f"     --val-split ./data/processed/splits/val.txt \\")
    print(f"     --epochs 10 \\")
    print(f"     --device mps")
    print()


if __name__ == '__main__':
    main()