#!/usr/bin/env python3
"""Prepare LIDC-IDRI dataset: DICOM to NIfTI conversion and train/val/test splits."""

import argparse
from pathlib import Path

from src.data.prepare_lidc import (
    convert_dicom_to_nifti,
    create_manifest,
    split_dataset,
    save_split
)


def main():
    parser = argparse.ArgumentParser(description='Prepare LIDC-IDRI dataset')
    parser.add_argument('--dicom-root', type=str, required=True,
                        help='Root directory containing LIDC-IDRI DICOM files')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Output directory for NIfTI files')
    parser.add_argument('--max-cases', type=int, default=None,
                        help='Maximum number of cases to process (default: all)')
    parser.add_argument('--train-ratio', type=float, default=0.7,
                        help='Training set ratio')
    parser.add_argument('--val-ratio', type=float, default=0.15,
                        help='Validation set ratio')
    parser.add_argument('--test-ratio', type=float, default=0.15,
                        help='Test set ratio')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for split')

    args = parser.parse_args()

    print("=" * 60)
    print("LIDC-IDRI Dataset Preparation")
    print("=" * 60)
    print(f"DICOM root:  {args.dicom_root}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Max cases:   {args.max_cases if args.max_cases else 'all'}")
    print("=" * 60 + "\n")

    # Convert DICOM to NIfTI
    print("Step 1: Converting DICOM to NIfTI...")
    metadata_list = convert_dicom_to_nifti(
        dicom_root=args.dicom_root,
        output_dir=args.output_dir,
        max_cases=args.max_cases,
        verbose=True
    )
    print(f"Converted {len(metadata_list)} cases\n")

    # Create manifest
    print("Step 2: Creating manifest...")
    manifest_path = Path(args.output_dir) / 'manifest.json'
    create_manifest(metadata_list, manifest_path, format='json')
    print()

    # Split dataset
    print("Step 3: Creating train/val/test splits...")
    train_files, val_files, test_files = split_dataset(
        metadata_list=metadata_list,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed
    )
    print()

    # Save splits
    print("Step 4: Saving split files...")
    splits_dir = Path(args.output_dir) / 'splits'
    save_split(train_files, val_files, test_files, splits_dir)
    print()

    print("=" * 60)
    print("Dataset preparation complete!")
    print("=" * 60)
    print(f"\nNext steps:")
    print(f"1. Verify NIfTI files in: {args.output_dir}")
    print(f"2. Check splits in: {splits_dir}")
    print(f"3. Use splits for training:")
    print(f"   python scripts/train.py \\")
    print(f"     --data-dir {args.output_dir} \\")
    print(f"     --train-split {splits_dir}/train.txt \\")
    print(f"     --val-split {splits_dir}/val.txt")
    print()


if __name__ == '__main__':
    main()
