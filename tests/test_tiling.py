"""Tests for seamless patch-wise tiling with Gaussian blending."""

import pytest
import numpy as np
import torch
import torch.nn as nn

from src.infer.patch_infer import PatchInference, create_gaussian_weight


class DummyModel(nn.Module):
    """Dummy model that returns input * 2 for testing."""

    def __init__(self, upscale_factor=2):
        super().__init__()
        self.upscale_factor = upscale_factor

    def forward(self, x):
        # Upsample z-axis by repeating
        b, c, d, h, w = x.shape
        # Simple nearest-neighbor upsampling for testing
        x_up = x.repeat_interleave(self.upscale_factor, dim=2)
        return x_up * 2  # Scale values for testing


def test_gaussian_weight_creation():
    """Test Gaussian weight map has correct properties."""
    shape = (16, 128, 128)
    weight = create_gaussian_weight(shape)

    # Check shape
    assert weight.shape == shape

    # Check all positive
    assert (weight > 0).all()

    # Check center has max value
    center = (shape[0] // 2, shape[1] // 2, shape[2] // 2)
    center_val = weight[center]
    assert center_val == weight.max()


def test_patch_inference_deterministic():
    """Test that patch inference gives deterministic results."""
    np.random.seed(42)
    torch.manual_seed(42)

    volume = np.random.randn(32, 128, 128).astype(np.float32)

    model = DummyModel(upscale_factor=2)
    inference = PatchInference(
        model=model,
        patch_size=(16, 64, 64),
        overlap=(4, 16, 16),
        device='cpu',
        upscale_factor=2
    )

    # Run twice
    output1 = inference.infer(volume, progress=False)
    output2 = inference.infer(volume, progress=False)

    # Should be identical
    np.testing.assert_allclose(output1, output2, rtol=1e-6)


def test_patch_inference_no_seams():
    """Test that Gaussian blending eliminates seams."""
    # Create synthetic volume with constant value
    volume = np.ones((32, 128, 128), dtype=np.float32) * 0.5

    model = DummyModel(upscale_factor=2)
    inference = PatchInference(
        model=model,
        patch_size=(16, 64, 64),
        overlap=(8, 32, 32),
        device='cpu',
        upscale_factor=2
    )

    output = inference.infer(volume, progress=False)

    # For constant input, output should be approximately constant (scaled by 2)
    # Check variance is very small
    output_std = output.std()
    assert output_std < 0.1, f"High variance detected: {output_std}, may indicate seams"

    # Check mean is close to expected value
    expected_mean = 0.5 * 2  # Input * 2 from model
    np.testing.assert_allclose(output.mean(), expected_mean, rtol=0.1)


def test_patch_inference_shape():
    """Test output shape is correctly upsampled."""
    volume = np.random.randn(24, 100, 100).astype(np.float32)
    upscale_factor = 2

    model = DummyModel(upscale_factor=upscale_factor)
    inference = PatchInference(
        model=model,
        patch_size=(8, 64, 64),
        overlap=(4, 16, 16),
        device='cpu',
        upscale_factor=upscale_factor
    )

    output = inference.infer(volume, progress=False)

    # Check z-axis is upsampled
    expected_shape = (volume.shape[0] * upscale_factor, volume.shape[1], volume.shape[2])
    assert output.shape == expected_shape


def test_single_patch_vs_tiled():
    """Test single patch inference matches expected behavior."""
    patch = np.random.randn(16, 64, 64).astype(np.float32)

    model = DummyModel(upscale_factor=2)
    inference = PatchInference(
        model=model,
        device='cpu',
        upscale_factor=2
    )

    output = inference.infer_single_patch(patch)

    # Check shape
    assert output.shape == (32, 64, 64)

    # Check values (should be input * 2)
    # For repeated z-slices
    for i in range(0, 32, 2):
        original_slice_idx = i // 2
        expected = patch[original_slice_idx] * 2
        np.testing.assert_allclose(output[i], expected, rtol=1e-5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
