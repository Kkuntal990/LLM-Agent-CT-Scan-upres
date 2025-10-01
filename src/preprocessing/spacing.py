"""Voxel spacing manipulation and resampling."""

import numpy as np
from scipy.ndimage import zoom
from typing import Tuple, Optional


def resample_volume(
    volume: np.ndarray,
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float],
    order: int = 1
) -> np.ndarray:
    """Resample volume to target spacing using interpolation.

    Args:
        volume: 3D numpy array
        current_spacing: Current (sx, sy, sz) in mm
        target_spacing: Target (sx, sy, sz) in mm
        order: Interpolation order (0=nearest, 1=linear, 3=cubic)

    Returns:
        Resampled volume
    """
    # Calculate zoom factors for each axis
    zoom_factors = tuple(
        c / t for c, t in zip(current_spacing, target_spacing)
    )

    # Resample
    resampled = zoom(volume, zoom_factors, order=order)
    return resampled


def match_spacing(
    volume: np.ndarray,
    spacing: Tuple[float, float, float],
    target_z_spacing: float,
    keep_xy: bool = True
) -> Tuple[np.ndarray, Tuple[float, float, float]]:
    """Match through-plane spacing while optionally preserving in-plane.

    Args:
        volume: 3D numpy array, shape (D, H, W)
        spacing: Current (sx, sy, sz)
        target_z_spacing: Target z-spacing in mm
        keep_xy: If True, preserve x,y spacing; only resample z-axis

    Returns:
        resampled_volume: Volume with updated z-spacing
        new_spacing: (sx, sy, new_sz)
    """
    sx, sy, sz = spacing

    if keep_xy:
        # Only resample along z-axis
        z_factor = sz / target_z_spacing
        new_shape = (int(volume.shape[0] * z_factor), volume.shape[1], volume.shape[2])

        # Use order=1 (linear) for HU values
        resampled = zoom(volume, (z_factor, 1.0, 1.0), order=1)
        new_spacing = (sx, sy, target_z_spacing)
    else:
        target_spacing = (sx, sy, target_z_spacing)
        resampled = resample_volume(volume, spacing, target_spacing, order=1)
        new_spacing = target_spacing

    return resampled, new_spacing


def calculate_new_shape(
    current_shape: Tuple[int, int, int],
    current_spacing: Tuple[float, float, float],
    target_spacing: Tuple[float, float, float]
) -> Tuple[int, int, int]:
    """Calculate output shape after resampling to target spacing.

    Args:
        current_shape: (D, H, W)
        current_spacing: (sx, sy, sz) in mm
        target_spacing: Target (sx, sy, sz) in mm

    Returns:
        new_shape: (D', H', W')
    """
    new_shape = tuple(
        int(np.round(s * c / t))
        for s, c, t in zip(current_shape, current_spacing, target_spacing)
    )
    return new_shape
