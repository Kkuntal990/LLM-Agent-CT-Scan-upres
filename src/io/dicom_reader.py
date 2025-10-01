"""DICOM series reading with HU calibration for LIDC-IDRI."""

import os
from pathlib import Path
from typing import Tuple, Optional
import numpy as np
import pydicom
from pydicom.dataset import FileDataset


def get_series_uid(dicom_dir: str) -> str:
    """Extract SeriesInstanceUID from DICOM directory.

    Args:
        dicom_dir: Path to directory containing DICOM files

    Returns:
        SeriesInstanceUID string for traceability
    """
    dcm_files = sorted(Path(dicom_dir).glob("*.dcm"))
    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    ds = pydicom.dcmread(dcm_files[0], stop_before_pixels=True)
    return str(ds.SeriesInstanceUID)


def read_dicom_series(
    dicom_dir: str,
    return_spacing: bool = True
) -> Tuple[np.ndarray, Optional[Tuple[float, float, float]]]:
    """Read DICOM series and convert to HU with proper calibration.

    Reads all DICOM files in directory, sorts by ImagePositionPatient,
    applies RescaleSlope/Intercept for HU calibration, and extracts
    voxel spacing (sx, sy, sz).

    Args:
        dicom_dir: Path to directory containing DICOM files
        return_spacing: If True, return (volume, spacing), else just volume

    Returns:
        volume: 3D numpy array in HU units, shape (D, H, W)
        spacing: (sx, sy, sz) in mm if return_spacing=True

    Raises:
        ValueError: If no DICOM files found or inconsistent series
    """
    dicom_dir = Path(dicom_dir)
    dcm_files = sorted(dicom_dir.glob("*.dcm"))

    if not dcm_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    # Read all slices with position information
    slices = []
    for f in dcm_files:
        ds = pydicom.dcmread(f)
        if not hasattr(ds, 'ImagePositionPatient'):
            continue
        slices.append(ds)

    if not slices:
        raise ValueError(f"No valid DICOM slices with ImagePositionPatient in {dicom_dir}")

    # Sort by z-position (ImagePositionPatient[2])
    slices.sort(key=lambda s: float(s.ImagePositionPatient[2]))

    # Extract spacing
    ref_slice = slices[0]
    pixel_spacing = ref_slice.PixelSpacing  # [row_spacing, col_spacing]
    sx, sy = float(pixel_spacing[1]), float(pixel_spacing[0])

    # Calculate slice thickness from positions
    if len(slices) > 1:
        positions = np.array([float(s.ImagePositionPatient[2]) for s in slices])
        sz_calc = np.median(np.diff(positions))
        # Use SliceThickness if available and consistent
        if hasattr(ref_slice, 'SliceThickness'):
            sz_tag = float(ref_slice.SliceThickness)
            sz = sz_tag if abs(sz_tag - sz_calc) < 0.1 else sz_calc
        else:
            sz = sz_calc
    else:
        sz = float(ref_slice.SliceThickness) if hasattr(ref_slice, 'SliceThickness') else 1.0

    spacing = (sx, sy, sz)

    # Build volume with HU calibration
    volume_slices = []
    for s in slices:
        arr = s.pixel_array.astype(np.float32)

        # Apply HU calibration: HU = pixel_value * RescaleSlope + RescaleIntercept
        slope = float(s.RescaleSlope) if hasattr(s, 'RescaleSlope') else 1.0
        intercept = float(s.RescaleIntercept) if hasattr(s, 'RescaleIntercept') else 0.0

        hu_slice = arr * slope + intercept
        volume_slices.append(hu_slice)

    volume = np.stack(volume_slices, axis=0)  # Shape: (D, H, W)

    if return_spacing:
        return volume, spacing
    return volume
