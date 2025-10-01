"""Tests for slice profile simulation."""

import pytest
import numpy as np

from src.sim.slice_profile import SliceProfileSimulator, gaussian_kernel_1d, triangular_kernel_1d


def test_gaussian_kernel_normalized():
    """Test Gaussian kernel sums to 1."""
    kernel = gaussian_kernel_1d(thickness=5.0, spacing=1.0)
    assert np.abs(kernel.sum() - 1.0) < 1e-6


def test_triangular_kernel_normalized():
    """Test triangular kernel sums to 1."""
    kernel = triangular_kernel_1d(thickness=5.0, spacing=1.0)
    assert np.abs(kernel.sum() - 1.0) < 1e-6


def test_simulator_reduces_resolution():
    """Test that simulation reduces through-plane resolution."""
    # Create volume with sharp edges
    volume = np.zeros((100, 128, 128), dtype=np.float32)
    volume[40:60, :, :] = 100  # Sharp slab

    simulator = SliceProfileSimulator(profile_type='gaussian')

    # Simulate with 2x larger spacing
    lr_volume = simulator.simulate_with_exact_spacing(
        volume, hr_spacing=1.0, lr_spacing=2.0
    )

    # Check output is smaller along z
    assert lr_volume.shape[0] < volume.shape[0]
    assert lr_volume.shape[1] == volume.shape[1]
    assert lr_volume.shape[2] == volume.shape[2]


def test_training_pair_generation():
    """Test LR-HR training pair has correct relationship."""
    volume = np.random.randn(80, 128, 128).astype(np.float32) * 100 + 40

    simulator = SliceProfileSimulator(profile_type='gaussian')
    lr_volume, hr_volume = simulator.create_training_pair(
        volume, hr_spacing=1.0, downsample_factor=2
    )

    # HR should be original
    np.testing.assert_array_equal(hr_volume, volume)

    # LR should be downsampled
    assert lr_volume.shape[0] == volume.shape[0] // 2
    assert lr_volume.shape[1] == volume.shape[1]
    assert lr_volume.shape[2] == volume.shape[2]


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
