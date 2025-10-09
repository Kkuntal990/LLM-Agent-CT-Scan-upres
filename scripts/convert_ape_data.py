#!/usr/bin/env python3
"""
Convert APE-data from ZIP/DICOM to NIfTI format

Extracts DICOM series from APE-data ZIP files and converts them to NIfTI
for use with the existing custom latent diffusion training pipeline.
"""

import argparse
import os
import zipfile
import tempfile
import shutil
from pathlib import Path
from tqdm import tqdm
import pydicom
import numpy as np
import nibabel as nib
from collections import defaultdict


def parse_args():
    parser = argparse.ArgumentParser(description="Convert APE-data to NIfTI format")

    parser.add_argument("--ape-cache-dir", type=str, required=True,
                        help="Path to APE-data cache directory")
    parser.add_argument("--subset", type=str, default="APE", choices=["APE", "non APE"],
                        help="Which subset to convert")
    parser.add_argument("--output-dir", type=str, default="./data/ape-nifti",
                        help="Output directory for NIfTI files")
    parser.add_argument("--train-split", type=float, default=0.8,
                        help="Train/val split ratio")
    parser.add_argument("--val-split", type=float, default=0.1,
                        help="Validation split ratio (test = 1 - train - val)")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to process (for debugging)")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip already converted files")

    return parser.parse_args()


def extract_dicom_from_zip(zip_path: Path, temp_dir: Path) -> list:
    """
    Extract ZIP file and find all DICOM files.

    Returns:
        List of DICOM file paths
    """
    # Extract ZIP
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    # Find DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(temp_dir):
        for file in files:
            filepath = os.path.join(root, file)

            # Try to read as DICOM
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True)
                dicom_files.append(filepath)
            except:
                # Not a DICOM file
                pass

    return dicom_files


def group_dicom_by_series(dicom_files: list) -> dict:
    """
    Group DICOM files by SeriesInstanceUID.

    Returns:
        Dictionary mapping SeriesInstanceUID to list of DICOM files
    """
    series_dict = defaultdict(list)

    for dcm_path in dicom_files:
        try:
            ds = pydicom.dcmread(dcm_path, stop_before_pixels=True)
            series_uid = ds.SeriesInstanceUID if hasattr(ds, 'SeriesInstanceUID') else 'unknown'
            series_dict[series_uid].append(dcm_path)
        except:
            pass

    return series_dict


def convert_dicom_series_to_nifti(dicom_files: list, output_path: Path) -> bool:
    """
    Convert a DICOM series to NIfTI format.

    Returns:
        True if successful, False otherwise
    """
    try:
        # Read all DICOM slices
        slices = []
        for dcm_path in dicom_files:
            ds = pydicom.dcmread(dcm_path)
            if hasattr(ds, 'pixel_array'):
                slices.append(ds)

        if not slices:
            return False

        # Sort by ImagePositionPatient (Z coordinate)
        slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)

        # Stack into 3D volume
        volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

        # Convert to Hounsfield Units
        if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
            slope = float(slices[0].RescaleSlope)
            intercept = float(slices[0].RescaleIntercept)
            volume = volume * slope + intercept

        # Create affine matrix from DICOM metadata
        try:
            # Get ImagePositionPatient and ImageOrientationPatient
            ipp = np.array(slices[0].ImagePositionPatient)
            iop = np.array(slices[0].ImageOrientationPatient)

            # Get pixel spacing
            pixel_spacing = slices[0].PixelSpacing
            dx, dy = float(pixel_spacing[0]), float(pixel_spacing[1])

            # Calculate slice thickness
            if len(slices) > 1:
                dz = abs(float(slices[1].ImagePositionPatient[2]) - float(slices[0].ImagePositionPatient[2]))
            else:
                dz = float(slices[0].SliceThickness) if hasattr(slices[0], 'SliceThickness') else 1.0

            # Construct affine matrix
            # Row direction (X)
            row_dir = iop[:3]
            # Column direction (Y)
            col_dir = iop[3:]
            # Slice direction (Z) - cross product
            slice_dir = np.cross(row_dir, col_dir)

            affine = np.eye(4)
            affine[:3, 0] = row_dir * dx
            affine[:3, 1] = col_dir * dy
            affine[:3, 2] = slice_dir * dz
            affine[:3, 3] = ipp

        except:
            # Fallback to identity affine
            affine = np.eye(4)

        # Save as NIfTI
        nii = nib.Nifti1Image(volume, affine)
        nib.save(nii, output_path)

        return True

    except Exception as e:
        print(f"Error converting series: {e}")
        return False


def process_zip_file(zip_path: Path, output_dir: Path, skip_existing: bool = False) -> int:
    """
    Process a single ZIP file and convert to NIfTI.

    Returns:
        Number of series converted
    """
    # Create output filename
    patient_name = zip_path.stem  # e.g., "201612140637 RONG GUI FANG"
    safe_name = patient_name.replace(" ", "_")

    # Check if already converted (look for any file with this patient name)
    if skip_existing:
        existing_files = list(output_dir.glob(f"{safe_name}*.nii.gz"))
        if existing_files:
            # Already converted, skip extraction entirely
            return len(existing_files)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)

        # Extract DICOM files
        dicom_files = extract_dicom_from_zip(zip_path, temp_path)

        if not dicom_files:
            print(f"No DICOM files found in {zip_path.name}")
            return 0

        # Group by series
        series_dict = group_dicom_by_series(dicom_files)

        # Convert each series
        num_converted = 0
        for series_idx, (series_uid, dcm_files) in enumerate(series_dict.items()):
            # Output path
            if len(series_dict) > 1:
                output_path = output_dir / f"{safe_name}_series{series_idx:02d}.nii.gz"
            else:
                output_path = output_dir / f"{safe_name}.nii.gz"

            # Convert to NIfTI
            success = convert_dicom_series_to_nifti(dcm_files, output_path)
            if success:
                num_converted += 1

        return num_converted


def create_train_val_test_splits(nifti_files: list, train_ratio: float, val_ratio: float, output_dir: Path):
    """
    Create train/val/test splits and save file lists.
    """
    import random
    random.shuffle(nifti_files)

    n_total = len(nifti_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_files = nifti_files[:n_train]
    val_files = nifti_files[n_train:n_train+n_val]
    test_files = nifti_files[n_train+n_val:]

    # Save file lists with only filenames (not full paths)
    # The dataset loader will prepend the data_dir
    with open(output_dir / "train_files.txt", "w") as f:
        f.write("\n".join([p.name for p in train_files]))

    with open(output_dir / "val_files.txt", "w") as f:
        f.write("\n".join([p.name for p in val_files]))

    with open(output_dir / "test_files.txt", "w") as f:
        f.write("\n".join([p.name for p in test_files]))

    print(f"\nDataset splits:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val: {len(val_files)} files")
    print(f"  Test: {len(test_files)} files")


def main():
    args = parse_args()

    # Setup paths
    ape_cache_dir = Path(args.ape_cache_dir)
    subset_dir = ape_cache_dir / args.subset
    output_dir = Path(args.output_dir) / args.subset.replace(" ", "_")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all ZIP files
    zip_files = sorted(list(subset_dir.glob("*.zip")))
    if args.max_samples:
        zip_files = zip_files[:args.max_samples]

    print(f"Found {len(zip_files)} ZIP files in {subset_dir}")
    print(f"Output directory: {output_dir}")

    # Process each ZIP file
    total_series = 0
    for zip_path in tqdm(zip_files, desc="Converting ZIP files"):
        num_series = process_zip_file(zip_path, output_dir, args.skip_existing)
        total_series += num_series

    print(f"\nConversion complete!")
    print(f"Total series converted: {total_series}")

    # Find all converted NIfTI files
    nifti_files = sorted(list(output_dir.glob("*.nii.gz")))
    print(f"Total NIfTI files: {len(nifti_files)}")

    # Create train/val/test splits
    if nifti_files:
        create_train_val_test_splits(
            nifti_files,
            args.train_split,
            args.val_split,
            output_dir
        )

    print(f"\nDataset ready at: {output_dir}")


if __name__ == "__main__":
    main()
