"""Body masking for CT volumes to exclude air."""

import numpy as np
from scipy.ndimage import binary_erosion, binary_dilation, binary_fill_holes


def create_body_mask(
    volume: np.ndarray,
    threshold_hu: float = -500.0,
    erosion_iter: int = 2,
    dilation_iter: int = 3
) -> np.ndarray:
    """Create body mask to exclude air and background.

    Uses HU thresholding followed by morphological operations
    to create a clean body mask for computing metrics.

    Args:
        volume: 3D CT volume in HU units
        threshold_hu: HU threshold for body vs. air (default -500)
        erosion_iter: Number of erosion iterations to clean mask
        dilation_iter: Number of dilation iterations to recover boundary

    Returns:
        mask: Binary mask (bool), True for body voxels
    """
    # Threshold
    mask = volume > threshold_hu

    # Morphological cleaning
    if erosion_iter > 0:
        mask = binary_erosion(mask, iterations=erosion_iter)

    # Fill holes slice-by-slice
    for i in range(mask.shape[0]):
        mask[i] = binary_fill_holes(mask[i])

    # Dilate to recover boundary
    if dilation_iter > 0:
        mask = binary_dilation(mask, iterations=dilation_iter)

    return mask.astype(bool)


def apply_mask(
    volume: np.ndarray,
    mask: np.ndarray,
    fill_value: float = -1024.0
) -> np.ndarray:
    """Apply mask to volume, setting masked regions to fill_value.

    Args:
        volume: 3D volume
        mask: Binary mask
        fill_value: Value for masked-out regions (default air HU)

    Returns:
        Masked volume
    """
    masked = volume.copy()
    masked[~mask] = fill_value
    return masked
