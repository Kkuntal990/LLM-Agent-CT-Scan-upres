"""Dataset and dataloader for LIDC-IDRI training."""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import nibabel as nib

from ..sim.slice_profile import SliceProfileSimulator
from ..preprocessing.masking import create_body_mask


class LIDCDataset(Dataset):
    """LIDC-IDRI dataset for supervised through-plane SR training.

    Loads high-resolution NIfTI volumes and generates LR-HR pairs
    on-the-fly using slice profile simulation.
    """

    def __init__(
        self,
        data_dir: str,
        file_list: List[str],
        patch_size: Tuple[int, int, int] = (32, 128, 128),
        downsample_factor: int = 2,
        hr_spacing: float = 1.0,
        use_body_mask: bool = True,
        augment: bool = True
    ):
        """Initialize dataset.

        Args:
            data_dir: Root directory containing NIfTI files
            file_list: List of NIfTI filenames to use
            patch_size: 3D patch size (D, H, W) for training
            downsample_factor: Z-axis downsampling factor (2x, 3x, etc.)
            hr_spacing: HR z-spacing in mm (for simulation)
            use_body_mask: Create body masks for loss computation
            augment: Apply random flips for augmentation
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.hr_spacing = hr_spacing
        self.use_body_mask = use_body_mask
        self.augment = augment

        self.simulator = SliceProfileSimulator(profile_type='gaussian')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        """Load volume and generate LR-HR patch pair.

        Returns:
            Dictionary with keys:
                - 'lr': LR input patch, shape (1, D//k, H, W)
                - 'hr': HR target patch, shape (1, D, H, W)
                - 'mask': Body mask (if use_body_mask=True)
        """
        # Load volume
        nifti_path = self.data_dir / self.file_list[idx]
        img = nib.load(nifti_path)
        volume = np.asarray(img.dataobj).astype(np.float32)

        # Generate LR-HR pair
        lr_volume, hr_volume = self.simulator.create_training_pair(
            volume, self.hr_spacing, self.downsample_factor
        )

        # Extract random patch
        hr_patch, lr_patch = self._extract_patch(hr_volume, lr_volume)

        # Augmentation
        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        # Normalize to [0, 1] range approximately (HU: -1024 to 3071)
        hr_patch = self._normalize_hu(hr_patch)
        lr_patch = self._normalize_hu(lr_patch)

        # Convert to tensors
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0)  # (1, D, H, W)
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0)  # (1, D//k, H, W)

        result = {'lr': lr_tensor, 'hr': hr_tensor}

        # Body mask
        if self.use_body_mask:
            mask = create_body_mask(hr_patch, threshold_hu=-500.0)
            result['mask'] = torch.from_numpy(mask).unsqueeze(0)

        return result

    def _extract_patch(
        self,
        hr_volume: np.ndarray,
        lr_volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random 3D patch from HR and corresponding LR region."""
        d_hr, h, w = hr_volume.shape
        d_lr = lr_volume.shape[0]
        patch_d, patch_h, patch_w = self.patch_size

        # Pad volumes if smaller than patch size
        if d_hr < patch_d or h < patch_h or w < patch_w:
            pad_d = max(0, patch_d - d_hr)
            pad_h = max(0, patch_h - h)
            pad_w = max(0, patch_w - w)

            hr_volume = np.pad(
                hr_volume,
                ((0, pad_d), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )

            lr_pad_d = max(0, (patch_d // self.downsample_factor) - d_lr)
            lr_volume = np.pad(
                lr_volume,
                ((0, lr_pad_d), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )

            d_hr, h, w = hr_volume.shape

        # Random starting position for HR patch
        d_start = np.random.randint(0, max(1, d_hr - patch_d + 1))
        h_start = np.random.randint(0, max(1, h - patch_h + 1))
        w_start = np.random.randint(0, max(1, w - patch_w + 1))

        hr_patch = hr_volume[
            d_start:d_start + patch_d,
            h_start:h_start + patch_h,
            w_start:w_start + patch_w
        ]

        # Corresponding LR patch
        lr_patch_d = patch_d // self.downsample_factor
        lr_d_start = d_start // self.downsample_factor

        lr_patch = lr_volume[
            lr_d_start:lr_d_start + lr_patch_d,
            h_start:h_start + patch_h,
            w_start:w_start + patch_w
        ]

        return hr_patch, lr_patch

    def _augment(
        self,
        hr_patch: np.ndarray,
        lr_patch: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random flips for augmentation."""
        # Random horizontal flip
        if np.random.rand() > 0.5:
            hr_patch = np.flip(hr_patch, axis=2).copy()
            lr_patch = np.flip(lr_patch, axis=2).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            hr_patch = np.flip(hr_patch, axis=1).copy()
            lr_patch = np.flip(lr_patch, axis=1).copy()

        return hr_patch, lr_patch

    def _normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        """Normalize HU values to approximate [0, 1] range."""
        # Clip to typical CT range
        volume = np.clip(volume, -1024, 3071)
        # Normalize to [0, 1]
        volume = (volume + 1024) / 4095.0
        return volume.astype(np.float32)


def create_dataloaders(
    train_files: List[str],
    val_files: List[str],
    data_dir: str,
    batch_size: int = 4,
    num_workers: int = 0,
    **dataset_kwargs
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders.

    Args:
        train_files: List of training NIfTI filenames
        val_files: List of validation NIfTI filenames
        data_dir: Root data directory
        batch_size: Batch size
        num_workers: Number of workers (use 0 for MPS compatibility)
        **dataset_kwargs: Additional args for LIDCDataset

    Returns:
        train_loader, val_loader
    """
    train_dataset = LIDCDataset(
        data_dir=data_dir,
        file_list=train_files,
        augment=True,
        **dataset_kwargs
    )

    val_dataset = LIDCDataset(
        data_dir=data_dir,
        file_list=val_files,
        augment=False,
        **dataset_kwargs
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False  # MPS doesn't support pin_memory
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False
    )

    return train_loader, val_loader
