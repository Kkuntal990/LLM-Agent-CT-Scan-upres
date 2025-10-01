"""Slice profile simulation for thick-slice CT synthesis.

Implements 1D PSF convolution along z-axis followed by decimation
to simulate acquisition with larger slice thickness and spacing.
"""

import numpy as np
from scipy.ndimage import convolve1d
from typing import Literal, Optional


def gaussian_kernel_1d(thickness: float, spacing: float, sigma_factor: float = 0.42) -> np.ndarray:
    """Generate 1D Gaussian slice profile kernel.

    Args:
        thickness: Slice thickness in mm (FWHM of profile)
        spacing: Voxel spacing in z in mm
        sigma_factor: Factor relating thickness to sigma (default 0.42 for FWHM)

    Returns:
        kernel: 1D array, normalized to sum=1
    """
    sigma = thickness * sigma_factor / spacing
    kernel_radius = int(np.ceil(3 * sigma))
    x = np.arange(-kernel_radius, kernel_radius + 1)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def triangular_kernel_1d(thickness: float, spacing: float) -> np.ndarray:
    """Generate 1D triangular slice profile kernel.

    Args:
        thickness: Slice thickness in mm
        spacing: Voxel spacing in z in mm

    Returns:
        kernel: 1D array, normalized to sum=1
    """
    half_width = int(np.ceil(thickness / (2 * spacing)))
    x = np.arange(-half_width, half_width + 1)
    kernel = np.maximum(0, 1 - np.abs(x * spacing) / thickness)
    kernel /= kernel.sum()
    return kernel


class SliceProfileSimulator:
    """Simulate thick-slice acquisition from thin-slice HR volume.

    Applies parameterized 1D slice profile (Gaussian or triangular)
    along z-axis, then decimates to target slice spacing.
    """

    def __init__(
        self,
        profile_type: Literal['gaussian', 'triangular'] = 'gaussian',
        sigma_factor: float = 0.42
    ):
        """Initialize simulator.

        Args:
            profile_type: Type of slice profile ('gaussian' or 'triangular')
            sigma_factor: Gaussian sigma factor (used if profile_type='gaussian')
        """
        self.profile_type = profile_type
        self.sigma_factor = sigma_factor

    def simulate(
        self,
        volume: np.ndarray,
        hr_spacing: float,
        target_thickness: float,
        target_spacing: float,
        overlap_fraction: float = 0.0
    ) -> np.ndarray:
        """Simulate thick-slice volume from thin-slice HR volume.

        Args:
            volume: High-resolution volume, shape (D, H, W)
            hr_spacing: HR z-spacing in mm
            target_thickness: Target slice thickness in mm
            target_spacing: Target slice spacing in mm
            overlap_fraction: Overlap fraction (0=no overlap, 0.5=50% overlap)

        Returns:
            lr_volume: Simulated low-resolution volume
        """
        # Build kernel
        if self.profile_type == 'gaussian':
            kernel = gaussian_kernel_1d(target_thickness, hr_spacing, self.sigma_factor)
        elif self.profile_type == 'triangular':
            kernel = triangular_kernel_1d(target_thickness, hr_spacing)
        else:
            raise ValueError(f"Unknown profile type: {self.profile_type}")

        # Convolve along z-axis
        blurred = convolve1d(volume, kernel, axis=0, mode='nearest')

        # Calculate effective spacing with overlap
        # spacing = thickness * (1 - overlap_fraction)
        effective_spacing = target_spacing

        # Decimation factor
        decimate_factor = int(np.round(effective_spacing / hr_spacing))
        decimate_factor = max(1, decimate_factor)

        # Decimate
        lr_volume = blurred[::decimate_factor, :, :]

        return lr_volume

    def simulate_with_exact_spacing(
        self,
        volume: np.ndarray,
        hr_spacing: float,
        lr_spacing: float
    ) -> np.ndarray:
        """Simulate LR volume matching exact target spacing.

        Uses thickness = spacing (no overlap) for simplicity.

        Args:
            volume: HR volume
            hr_spacing: HR z-spacing in mm
            lr_spacing: Target LR z-spacing in mm

        Returns:
            lr_volume: Simulated LR volume
        """
        return self.simulate(
            volume,
            hr_spacing,
            target_thickness=lr_spacing,
            target_spacing=lr_spacing,
            overlap_fraction=0.0
        )

    def create_training_pair(
        self,
        hr_volume: np.ndarray,
        hr_spacing: float,
        downsample_factor: int = 2
    ) -> tuple[np.ndarray, np.ndarray]:
        """Create LR-HR training pair from HR volume.

        Args:
            hr_volume: High-resolution volume
            hr_spacing: HR z-spacing in mm
            downsample_factor: Factor to downsample (e.g., 2x, 3x)

        Returns:
            lr_volume: Simulated LR input
            hr_volume: Original HR target
        """
        lr_spacing = hr_spacing * downsample_factor
        lr_volume = self.simulate_with_exact_spacing(hr_volume, hr_spacing, lr_spacing)
        return lr_volume, hr_volume
