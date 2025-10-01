"""Tests for HU integrity through DICOM-NIfTI-SR pipeline."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from src.io.nifti_io import write_nifti, read_nifti


def test_hu_roundtrip():
    """Test HU values are preserved through NIfTI write-read cycle."""
    # Create synthetic CT volume with known HU values
    volume = np.random.randn(50, 128, 128).astype(np.float32) * 200 + 40  # Soft tissue range
    spacing = (0.7, 0.7, 2.5)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / 'test.nii.gz'

        # Write
        write_nifti(volume, str(nifti_path), spacing)

        # Read
        volume_read, spacing_read = read_nifti(str(nifti_path), return_spacing=True)

        # Check values preserved
        np.testing.assert_allclose(volume, volume_read, rtol=1e-5)

        # Check spacing preserved
        np.testing.assert_allclose(spacing, spacing_read, rtol=1e-6)


def test_hu_range_preservation():
    """Test that typical HU ranges are preserved correctly."""
    # Create volume with specific HU values
    volume = np.zeros((30, 64, 64), dtype=np.float32)

    # Set specific tissue types
    volume[10:20, 20:40, 20:40] = -1000  # Air
    volume[15:18, 25:35, 25:35] = 40     # Soft tissue
    volume[16, 30:33, 30:33] = 400       # Bone

    spacing = (1.0, 1.0, 1.0)

    with tempfile.TemporaryDirectory() as tmpdir:
        nifti_path = Path(tmpdir) / 'test_ranges.nii.gz'

        write_nifti(volume, str(nifti_path), spacing)
        volume_read = read_nifti(str(nifti_path), return_spacing=False)

        # Verify HU values
        assert np.allclose(volume_read[10, 30, 30], -1000, atol=1)
        assert np.allclose(volume_read[15, 30, 30], 40, atol=1)
        assert np.allclose(volume_read[16, 31, 31], 400, atol=1)


def test_spacing_update():
    """Test spacing can be updated correctly in header."""
    from src.io.nifti_io import update_spacing

    volume = np.random.randn(40, 100, 100).astype(np.float32) * 100
    original_spacing = (0.7, 0.7, 2.5)
    new_spacing = (0.7, 0.7, 1.25)  # 2x upsampled in z

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = Path(tmpdir) / 'input.nii.gz'
        output_path = Path(tmpdir) / 'output.nii.gz'

        # Write with original spacing
        write_nifti(volume, str(input_path), original_spacing)

        # Update spacing
        update_spacing(str(input_path), str(output_path), new_spacing)

        # Read and verify
        _, spacing_read = read_nifti(str(output_path), return_spacing=True)
        np.testing.assert_allclose(new_spacing, spacing_read, rtol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
