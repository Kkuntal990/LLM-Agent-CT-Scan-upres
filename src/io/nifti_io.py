"""NIfTI I/O with proper spacing and orientation handling."""

import numpy as np
import nibabel as nib
from typing import Tuple, Optional


def read_nifti(
    nifti_path: str,
    return_spacing: bool = True,
    return_affine: bool = False
) -> Tuple[np.ndarray, ...]:
    """Read NIfTI file and extract volume with metadata.

    Args:
        nifti_path: Path to .nii or .nii.gz file
        return_spacing: If True, include spacing in output
        return_affine: If True, include affine matrix in output

    Returns:
        volume: 3D numpy array, shape (D, H, W)
        spacing: (sx, sy, sz) in mm if return_spacing=True
        affine: 4x4 affine matrix if return_affine=True
    """
    img = nib.load(nifti_path)
    volume = np.asarray(img.dataobj).astype(np.float32)

    result = [volume]

    if return_spacing:
        # Extract spacing from header zooms
        spacing = tuple(img.header.get_zooms()[:3])
        result.append(spacing)

    if return_affine:
        result.append(img.affine)

    return tuple(result) if len(result) > 1 else result[0]


def write_nifti(
    volume: np.ndarray,
    output_path: str,
    spacing: Tuple[float, float, float],
    affine: Optional[np.ndarray] = None,
    header: Optional[nib.Nifti1Header] = None
) -> None:
    """Write volume to NIfTI with correct spacing and affine.

    Args:
        volume: 3D numpy array to save
        output_path: Output path for .nii or .nii.gz
        spacing: (sx, sy, sz) voxel spacing in mm
        affine: Optional 4x4 affine matrix; if None, creates from spacing
        header: Optional header template; if provided, updates spacing
    """
    if affine is None:
        # Create simple affine with diagonal spacing and no rotation
        affine = np.eye(4)
        affine[0, 0] = spacing[0]
        affine[1, 1] = spacing[1]
        affine[2, 2] = spacing[2]

    # Create NIfTI image
    img = nib.Nifti1Image(volume.astype(np.float32), affine)

    # Update header with spacing
    if header is not None:
        img.header.set_zooms(spacing)
        # Copy relevant fields from template
        img.header['pixdim'] = header['pixdim'].copy()
        img.header['pixdim'][1:4] = spacing
    else:
        img.header.set_zooms(spacing)

    # Save
    nib.save(img, output_path)


def update_spacing(
    input_path: str,
    output_path: str,
    new_spacing: Tuple[float, float, float]
) -> None:
    """Update spacing in NIfTI header without changing voxel data.

    Useful for correcting metadata after super-resolution.

    Args:
        input_path: Input NIfTI path
        output_path: Output NIfTI path
        new_spacing: New (sx, sy, sz) spacing in mm
    """
    img = nib.load(input_path)
    volume = np.asarray(img.dataobj)

    # Update affine diagonal with new spacing
    affine = img.affine.copy()
    affine[0, 0] = new_spacing[0] * np.sign(affine[0, 0])
    affine[1, 1] = new_spacing[1] * np.sign(affine[1, 1])
    affine[2, 2] = new_spacing[2] * np.sign(affine[2, 2])

    write_nifti(volume, output_path, new_spacing, affine, img.header)
