"""
APE-data Dataset Loader for Hugging Face Diffusers Approach

Loads CT scans from the APE-data Hugging Face dataset cache and creates
low-resolution/high-resolution pairs for diffusion model training.
"""

import os
import zipfile
import pydicom
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional
import tempfile
import shutil
from scipy.ndimage import gaussian_filter, zoom


class APEDataset(Dataset):
    """
    Dataset for APE CT scans with slice profile simulation.

    Extracts DICOM files from ZIP archives, converts to volumes,
    and creates LR-HR pairs using Gaussian blur + decimation.
    """

    def __init__(
        self,
        ape_cache_dir: str,
        subset: str = "APE",  # "APE" or "non APE"
        scale_factor: int = 3,
        patch_size: Tuple[int, int, int] = (64, 128, 128),
        normalize: bool = True,
        hu_window: Tuple[float, float] = (-1000, 400),
        cache_extracted: bool = True,
        max_samples: Optional[int] = None,
    ):
        """
        Args:
            ape_cache_dir: Path to APE-data cache (e.g., ~/.cache/huggingface/hub/datasets--t2ance--APE-data/snapshots/...)
            subset: "APE" or "non APE" subdirectory
            scale_factor: Upsampling factor for z-axis (default: 3)
            patch_size: 3D patch size (Z, H, W) for training
            normalize: Normalize to [-1, 1] range
            hu_window: HU windowing range for CT
            cache_extracted: Cache extracted DICOM volumes to disk
            max_samples: Limit number of samples (for debugging)
        """
        self.ape_cache_dir = Path(ape_cache_dir)
        self.subset_dir = self.ape_cache_dir / subset
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.normalize = normalize
        self.hu_window = hu_window
        self.cache_extracted = cache_extracted

        # Find all ZIP files
        self.zip_files = sorted(list(self.subset_dir.glob("*.zip")))
        if max_samples:
            self.zip_files = self.zip_files[:max_samples]

        print(f"Found {len(self.zip_files)} ZIP files in {self.subset_dir}")

        # Cache directory for extracted volumes
        if cache_extracted:
            self.cache_dir = self.ape_cache_dir / f"extracted_{subset.replace(' ', '_')}"
            self.cache_dir.mkdir(exist_ok=True)
        else:
            self.cache_dir = None

    def __len__(self) -> int:
        return len(self.zip_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lr_volume: Low-resolution CT volume [1, Z//scale, H, W]
            hr_volume: High-resolution CT volume [1, Z, H, W]
        """
        zip_path = self.zip_files[idx]

        # Check cache first
        if self.cache_dir:
            cache_file = self.cache_dir / f"{zip_path.stem}.npy"
            if cache_file.exists():
                hr_volume = np.load(cache_file)
            else:
                hr_volume = self._extract_volume_from_zip(zip_path)
                np.save(cache_file, hr_volume)
        else:
            hr_volume = self._extract_volume_from_zip(zip_path)

        # Apply HU windowing
        hr_volume = np.clip(hr_volume, self.hu_window[0], self.hu_window[1])

        # Normalize to [-1, 1] if requested
        if self.normalize:
            hr_volume = 2 * (hr_volume - self.hu_window[0]) / (self.hu_window[1] - self.hu_window[0]) - 1

        # Extract random 3D patch
        hr_patch = self._extract_random_patch(hr_volume, self.patch_size)

        # Create LR version using slice profile simulation
        lr_patch = self._simulate_slice_profile(hr_patch, self.scale_factor)

        # Convert to tensors [C, Z, H, W]
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0).float()
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).float()

        return lr_tensor, hr_tensor

    def _extract_volume_from_zip(self, zip_path: Path) -> np.ndarray:
        """Extract DICOM series from ZIP and convert to 3D volume."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Extract ZIP
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(tmpdir)

            # Find all DICOM files
            dicom_files = []
            for root, dirs, files in os.walk(tmpdir):
                for file in files:
                    if file.endswith('.dcm') or not '.' in file:  # DICOM files may have no extension
                        filepath = os.path.join(root, file)
                        try:
                            pydicom.dcmread(filepath, stop_before_pixels=True)
                            dicom_files.append(filepath)
                        except:
                            pass

            if not dicom_files:
                raise ValueError(f"No DICOM files found in {zip_path}")

            # Read DICOM slices
            slices = []
            for dcm_path in dicom_files:
                ds = pydicom.dcmread(dcm_path)
                if hasattr(ds, 'pixel_array'):
                    slices.append(ds)

            # Sort by ImagePositionPatient (Z coordinate)
            slices.sort(key=lambda x: float(x.ImagePositionPatient[2]) if hasattr(x, 'ImagePositionPatient') else 0)

            # Stack into 3D volume
            volume = np.stack([s.pixel_array.astype(np.float32) for s in slices], axis=0)

            # Convert to Hounsfield Units
            if hasattr(slices[0], 'RescaleSlope') and hasattr(slices[0], 'RescaleIntercept'):
                slope = float(slices[0].RescaleSlope)
                intercept = float(slices[0].RescaleIntercept)
                volume = volume * slope + intercept

            return volume

    def _simulate_slice_profile(self, hr_volume: np.ndarray, scale: int) -> np.ndarray:
        """
        Simulate through-plane slice profile using Gaussian blur + decimation.

        Args:
            hr_volume: High-resolution volume [Z, H, W]
            scale: Downsampling factor for Z-axis

        Returns:
            lr_volume: Low-resolution volume [Z//scale, H, W]
        """
        # Gaussian blur along Z-axis (simulate thick slices)
        sigma = scale / 2.0
        blurred = gaussian_filter(hr_volume, sigma=(sigma, 0, 0), mode='nearest')

        # Decimate along Z-axis
        lr_volume = blurred[::scale, :, :]

        return lr_volume

    def _extract_random_patch(self, volume: np.ndarray, patch_size: Tuple[int, int, int]) -> np.ndarray:
        """Extract random 3D patch from volume."""
        z, h, w = volume.shape
        pz, ph, pw = patch_size

        # Ensure volume is large enough
        if z < pz or h < ph or w < pw:
            # Pad if needed
            pad_z = max(0, pz - z)
            pad_h = max(0, ph - h)
            pad_w = max(0, pw - w)
            volume = np.pad(volume, ((0, pad_z), (0, pad_h), (0, pad_w)), mode='edge')
            z, h, w = volume.shape

        # Random crop
        z_start = np.random.randint(0, z - pz + 1)
        h_start = np.random.randint(0, h - ph + 1)
        w_start = np.random.randint(0, w - pw + 1)

        patch = volume[z_start:z_start+pz, h_start:h_start+ph, w_start:w_start+pw]

        return patch


def collate_fn(batch):
    """Custom collate function for APE dataset."""
    lr_volumes = torch.stack([item[0] for item in batch])
    hr_volumes = torch.stack([item[1] for item in batch])
    return lr_volumes, hr_volumes


if __name__ == "__main__":
    # Test dataset loading
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ape-cache-dir", type=str, required=True,
                        help="Path to APE-data cache directory")
    parser.add_argument("--subset", type=str, default="APE", choices=["APE", "non APE"])
    parser.add_argument("--max-samples", type=int, default=5)
    args = parser.parse_args()

    dataset = APEDataset(
        ape_cache_dir=args.ape_cache_dir,
        subset=args.subset,
        max_samples=args.max_samples,
    )

    print(f"\nDataset size: {len(dataset)}")

    # Test loading
    print("\nTesting data loading...")
    lr, hr = dataset[0]
    print(f"LR shape: {lr.shape}, range: [{lr.min():.2f}, {lr.max():.2f}]")
    print(f"HR shape: {hr.shape}, range: [{hr.min():.2f}, {hr.max():.2f}]")
    print("\nDataset test passed!")
