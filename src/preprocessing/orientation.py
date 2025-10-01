"""Volume orientation normalization to RAS."""

import numpy as np
import nibabel as nib
from typing import Tuple


def reorient_to_ras(
    volume: np.ndarray,
    affine: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Reorient volume to RAS (Right-Anterior-Superior) orientation.

    Args:
        volume: 3D numpy array
        affine: 4x4 affine matrix

    Returns:
        reoriented_volume: Volume in RAS orientation
        reoriented_affine: Corresponding affine matrix
    """
    # Create NIfTI image
    img = nib.Nifti1Image(volume, affine)

    # Reorient to RAS
    img_ras = nib.as_closest_canonical(img)

    # Extract reoriented data
    volume_ras = np.asarray(img_ras.dataobj).astype(volume.dtype)
    affine_ras = img_ras.affine

    return volume_ras, affine_ras


def get_orientation_string(affine: np.ndarray) -> str:
    """Get orientation string (e.g., 'RAS', 'LPS') from affine matrix.

    Args:
        affine: 4x4 affine matrix

    Returns:
        Three-letter orientation code
    """
    ornt = nib.orientations.io_orientation(affine)
    return nib.orientations.ornt2axcodes(ornt)
