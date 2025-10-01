"""LIDC-IDRI dataset preparation scripts.

Converts DICOM series to NIfTI with HU preservation and creates
train/val/test splits with manifest files.
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
from tqdm import tqdm

from ..io.dicom_reader import read_dicom_series, get_series_uid
from ..io.nifti_io import write_nifti


def convert_dicom_to_nifti(
    dicom_root: str,
    output_dir: str,
    max_cases: int = None,
    verbose: bool = True
) -> List[Dict[str, str]]:
    """Convert LIDC-IDRI DICOM series to NIfTI format.

    Processes DICOM directories, extracts SeriesInstanceUID for traceability,
    applies HU calibration, and writes NIfTI files.

    Args:
        dicom_root: Root directory containing LIDC-IDRI DICOM folders
        output_dir: Output directory for NIfTI files
        max_cases: Maximum number of cases to process (None for all)
        verbose: Print progress

    Returns:
        List of metadata dictionaries for processed cases
    """
    dicom_root = Path(dicom_root)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all patient directories (LIDC-IDRI-XXXX)
    patient_dirs = sorted([d for d in dicom_root.iterdir() if d.is_dir()])

    if max_cases is not None:
        patient_dirs = patient_dirs[:max_cases]

    metadata_list = []

    iterator = tqdm(patient_dirs, desc="Converting DICOM to NIfTI") if verbose else patient_dirs

    for patient_dir in iterator:
        patient_id = patient_dir.name

        try:
            # Find DICOM series directory
            # LIDC-IDRI structure: LIDC-IDRI-XXXX/StudyInstanceUID/SeriesInstanceUID/*.dcm
            study_dirs = [d for d in patient_dir.iterdir() if d.is_dir()]

            for study_dir in study_dirs:
                series_dirs = [d for d in study_dir.iterdir() if d.is_dir()]

                for series_dir in series_dirs:
                    # Check if directory contains DICOM files
                    dicom_files = list(series_dir.glob("*.dcm"))
                    if not dicom_files:
                        continue

                    # Read DICOM series
                    volume, spacing = read_dicom_series(str(series_dir), return_spacing=True)
                    series_uid = get_series_uid(str(series_dir))

                    # Create output filename
                    output_filename = f"{patient_id}.nii.gz"
                    output_path = output_dir / output_filename

                    # Write NIfTI
                    write_nifti(volume, str(output_path), spacing)

                    # Store metadata
                    metadata = {
                        'patient_id': patient_id,
                        'series_uid': series_uid,
                        'nifti_path': str(output_path.relative_to(output_dir)),
                        'spacing': spacing,
                        'shape': volume.shape
                    }
                    metadata_list.append(metadata)

                    if verbose and len(metadata_list) % 10 == 0:
                        print(f"Processed {len(metadata_list)} series")

        except Exception as e:
            print(f"Error processing {patient_id}: {e}")
            continue

    return metadata_list


def create_manifest(
    metadata_list: List[Dict],
    output_path: str,
    format: str = 'json'
):
    """Create manifest file with dataset metadata.

    Args:
        metadata_list: List of metadata dictionaries
        output_path: Output path for manifest
        format: Output format ('json' or 'csv')
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if format == 'json':
        with open(output_path, 'w') as f:
            json.dump(metadata_list, f, indent=2)
    elif format == 'csv':
        df = pd.DataFrame(metadata_list)
        df.to_csv(output_path, index=False)
    else:
        raise ValueError(f"Unknown format: {format}")

    print(f"Manifest saved to {output_path}")


def split_dataset(
    metadata_list: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[List[str], List[str], List[str]]:
    """Split dataset into train/val/test sets.

    Args:
        metadata_list: List of metadata dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility

    Returns:
        train_files, val_files, test_files (lists of NIfTI filenames)
    """
    import numpy as np

    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    # Set random seed
    np.random.seed(seed)

    # Get all filenames
    filenames = [meta['nifti_path'] for meta in metadata_list]

    # Shuffle
    indices = np.random.permutation(len(filenames))
    filenames = [filenames[i] for i in indices]

    # Calculate split points
    n_train = int(len(filenames) * train_ratio)
    n_val = int(len(filenames) * val_ratio)

    # Split
    train_files = filenames[:n_train]
    val_files = filenames[n_train:n_train + n_val]
    test_files = filenames[n_train + n_val:]

    print(f"Dataset split:")
    print(f"  Train: {len(train_files)} cases")
    print(f"  Val:   {len(val_files)} cases")
    print(f"  Test:  {len(test_files)} cases")

    return train_files, val_files, test_files


def save_split(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    output_dir: str
):
    """Save train/val/test split to text files.

    Args:
        train_files: Training filenames
        val_files: Validation filenames
        test_files: Test filenames
        output_dir: Output directory for split files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each split
    for split_name, files in [('train', train_files), ('val', val_files), ('test', test_files)]:
        output_path = output_dir / f"{split_name}.txt"
        with open(output_path, 'w') as f:
            f.write('\n'.join(files))
        print(f"Saved {split_name} split to {output_path}")


def load_split(split_file: str) -> List[str]:
    """Load split from text file.

    Args:
        split_file: Path to split file

    Returns:
        List of filenames
    """
    with open(split_file, 'r') as f:
        files = [line.strip() for line in f if line.strip()]
    return files
