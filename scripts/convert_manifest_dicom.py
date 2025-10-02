#!/usr/bin/env python3
"""Convert manifest DICOM files to NIfTI format."""

import os
import sys
from pathlib import Path
import numpy as np
import nibabel as nib
import pydicom
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.io.dicom_reader import read_dicom_series
from src.io.nifti_io import write_nifti


def convert_patient(patient_dir, output_dir):
    """Convert a single patient's DICOM series to NIfTI."""
    patient_name = patient_dir.name

    # Find all series directories
    series_dirs = []
    for study_dir in patient_dir.iterdir():
        if study_dir.is_dir():
            for series_dir in study_dir.iterdir():
                if series_dir.is_dir() and any(series_dir.glob("*.dcm")):
                    series_dirs.append(series_dir)

    if not series_dirs:
        print(f"  No DICOM series found for {patient_name}")
        return None

    # Use the first series (usually the CT scan)
    series_dir = series_dirs[0]

    try:
        # Read DICOM series
        volume, spacing = read_dicom_series(str(series_dir), return_spacing=True)

        # Create output filename
        output_path = output_dir / f"{patient_name}.nii.gz"

        # Write NIfTI
        write_nifti(volume, str(output_path), spacing)

        print(f"  ✓ {patient_name}: shape={volume.shape}, spacing={spacing}")
        return str(output_path)

    except Exception as e:
        print(f"  ✗ {patient_name}: {str(e)}")
        return None


def main():
    # Paths
    dicom_root = Path("./data/data/manifest-1600709154662/LIDC-IDRI")
    output_dir = Path("./data/lidc-processed")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all patient directories
    patient_dirs = sorted([d for d in dicom_root.iterdir()
                          if d.is_dir() and d.name.startswith("LIDC-IDRI-")])

    print(f"Found {len(patient_dirs)} patients")
    print("=" * 60)

    # Convert each patient
    converted_files = []
    for patient_dir in patient_dirs:
        result = convert_patient(patient_dir, output_dir)
        if result:
            converted_files.append(result)

    print("=" * 60)
    print(f"Successfully converted {len(converted_files)}/{len(patient_dirs)} patients")

    # Create train/val/test splits (60/20/20)
    np.random.seed(42)
    indices = np.random.permutation(len(converted_files))

    n_train = int(0.6 * len(converted_files))
    n_val = int(0.2 * len(converted_files))

    train_files = [Path(converted_files[i]).name for i in indices[:n_train]]
    val_files = [Path(converted_files[i]).name for i in indices[n_train:n_train+n_val]]
    test_files = [Path(converted_files[i]).name for i in indices[n_train+n_val:]]

    # Write split files
    with open(output_dir / "train_files.txt", "w") as f:
        f.write("\n".join(train_files))

    with open(output_dir / "val_files.txt", "w") as f:
        f.write("\n".join(val_files))

    with open(output_dir / "test_files.txt", "w") as f:
        f.write("\n".join(test_files))

    print(f"\nSplits created:")
    print(f"  Train: {len(train_files)} files")
    print(f"  Val:   {len(val_files)} files")
    print(f"  Test:  {len(test_files)} files")


if __name__ == "__main__":
    main()
