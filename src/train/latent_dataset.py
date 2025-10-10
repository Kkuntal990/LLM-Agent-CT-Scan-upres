"""Dataset for pre-computed latent representations.

Prepares and loads latent codes from VAE for efficient diffusion training.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Optional, Tuple, List
import nibabel as nib
from tqdm import tqdm

from ..sim.slice_profile import SliceProfileSimulator
from ..preprocessing.masking import create_body_mask


class LatentDataset(Dataset):
    """Dataset of pre-computed latent representations.

    Loads latents computed by VAE encoder for fast diffusion training.
    """

    def __init__(
        self,
        latent_dir: str,
        file_list: List[str],
        patch_size: Tuple[int, int, int] = (16, 64, 64),
        augment: bool = True
    ):
        """Initialize latent dataset.

        Args:
            latent_dir: Directory containing .npz files with latents
            file_list: List of base filenames (without _latent.npz suffix)
            patch_size: 3D patch size (D, H, W) in latent space
            augment: Apply random flips for augmentation
        """
        self.latent_dir = Path(latent_dir)
        self.file_list = file_list
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        """Load latent patch pair.

        Returns:
            Dictionary with keys:
                - 'hr_latent': HR latent patch, shape (C, D, H, W)
                - 'lr_latent': LR latent patch, shape (C, D, H, W)
        """
        # Load latent file
        # Handle both .nii.gz and .nii extensions (Path.stem only removes last extension)
        filename = self.file_list[idx]
        base_name = Path(filename).stem  # Removes .gz, leaving .nii for .nii.gz files
        latent_file = self.latent_dir / f"{base_name}_latent.npz"
        data = np.load(latent_file)

        hr_latent = data['hr_latent'].astype(np.float32)
        lr_latent = data['lr_latent'].astype(np.float32)

        # Extract random patch
        hr_patch, lr_patch = self._extract_patch(hr_latent, lr_latent)

        # Augmentation
        if self.augment:
            hr_patch, lr_patch = self._augment(hr_patch, lr_patch)

        # Convert to tensors
        hr_tensor = torch.from_numpy(hr_patch)
        lr_tensor = torch.from_numpy(lr_patch)

        return {
            'hr_latent': hr_tensor,
            'lr_latent': lr_tensor
        }

    def _extract_patch(
        self,
        hr_latent: np.ndarray,
        lr_latent: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random 3D patch from latents."""
        c, d_hr, h, w = hr_latent.shape
        patch_d, patch_h, patch_w = self.patch_size

        # If volume is smaller than patch size, pad it
        if d_hr < patch_d or h < patch_h or w < patch_w:
            pad_d = max(0, patch_d - d_hr)
            pad_h = max(0, patch_h - h)
            pad_w = max(0, patch_w - w)

            hr_latent = np.pad(
                hr_latent,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
            lr_latent = np.pad(
                lr_latent,
                ((0, 0), (0, pad_d), (0, pad_h), (0, pad_w)),
                mode='reflect'
            )
            d_hr, h, w = hr_latent.shape[1], hr_latent.shape[2], hr_latent.shape[3]

        # Random starting position
        d_start = np.random.randint(0, max(1, d_hr - patch_d + 1))
        h_start = np.random.randint(0, max(1, h - patch_h + 1))
        w_start = np.random.randint(0, max(1, w - patch_w + 1))

        hr_patch = hr_latent[
            :,
            d_start:d_start + patch_d,
            h_start:h_start + patch_h,
            w_start:w_start + patch_w
        ]

        lr_patch = lr_latent[
            :,
            d_start:d_start + patch_d,
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
            hr_patch = np.flip(hr_patch, axis=3).copy()
            lr_patch = np.flip(lr_patch, axis=3).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            hr_patch = np.flip(hr_patch, axis=2).copy()
            lr_patch = np.flip(lr_patch, axis=2).copy()

        return hr_patch, lr_patch


class VolumeDatasetForLatent(Dataset):
    """Dataset for generating latents from volumes (on-the-fly encoding).

    Used when latents are not pre-computed.
    """

    def __init__(
        self,
        data_dir: str,
        file_list: List[str],
        vae_model,
        device: str = 'cpu',
        patch_size: Tuple[int, int, int] = (32, 128, 128),
        downsample_factor: int = 2,
        hr_spacing: float = 1.0,
        augment: bool = True
    ):
        """Initialize volume dataset with on-the-fly VAE encoding.

        Args:
            data_dir: Root directory containing NIfTI files
            file_list: List of NIfTI filenames
            vae_model: Pre-trained VAE model for encoding
            device: Device for VAE encoding
            patch_size: 3D patch size (D, H, W) in pixel space
            downsample_factor: Z-axis downsampling factor
            hr_spacing: HR z-spacing in mm
            augment: Apply random flips
        """
        self.data_dir = Path(data_dir)
        self.file_list = file_list
        self.vae_model = vae_model.to(device)
        self.vae_model.eval()
        self.device = device
        self.patch_size = patch_size
        self.downsample_factor = downsample_factor
        self.hr_spacing = hr_spacing
        self.augment = augment

        self.simulator = SliceProfileSimulator(profile_type='gaussian')

    def __len__(self) -> int:
        return len(self.file_list)

    def __getitem__(self, idx: int) -> dict:
        """Load volume, create LR-HR pair, and encode to latents."""
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

        # Normalize to [0, 1] range
        hr_patch = self._normalize_hu(hr_patch)
        lr_patch = self._normalize_hu(lr_patch)

        # Upsample LR to HR size (for consistent latent dimensions)
        lr_upsampled = self._upsample_lr(lr_patch, hr_patch.shape)

        # Convert to tensors and add channel dimension
        hr_tensor = torch.from_numpy(hr_patch).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
        lr_tensor = torch.from_numpy(lr_upsampled).unsqueeze(0).unsqueeze(0)

        # Encode to latent space
        with torch.no_grad():
            hr_tensor = hr_tensor.to(self.device)
            lr_tensor = lr_tensor.to(self.device)

            hr_latent = self.vae_model.encode_to_latent(hr_tensor, sample=False)
            lr_latent = self.vae_model.encode_to_latent(lr_tensor, sample=False)

        return {
            'hr_latent': hr_latent.squeeze(0).cpu(),  # (C, D, H, W)
            'lr_latent': lr_latent.squeeze(0).cpu()
        }

    def _extract_patch(
        self,
        hr_volume: np.ndarray,
        lr_volume: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract random 3D patch."""
        d_hr, h, w = hr_volume.shape
        d_lr = lr_volume.shape[0]
        patch_d, patch_h, patch_w = self.patch_size

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
        """Apply random flips."""
        if np.random.rand() > 0.5:
            hr_patch = np.flip(hr_patch, axis=2).copy()
            lr_patch = np.flip(lr_patch, axis=2).copy()

        if np.random.rand() > 0.5:
            hr_patch = np.flip(hr_patch, axis=1).copy()
            lr_patch = np.flip(lr_patch, axis=1).copy()

        return hr_patch, lr_patch

    def _normalize_hu(self, volume: np.ndarray) -> np.ndarray:
        """Normalize HU values to approximate [0, 1] range."""
        volume = np.clip(volume, -1024, 3071)
        volume = (volume + 1024) / 4095.0
        return volume.astype(np.float32)

    def _upsample_lr(self, lr_patch: np.ndarray, target_shape: tuple) -> np.ndarray:
        """Upsample LR patch to target HR shape."""
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0)
        upsampled = torch.nn.functional.interpolate(
            lr_tensor,
            size=target_shape,
            mode='trilinear',
            align_corners=False
        )
        return upsampled.squeeze(0).squeeze(0).numpy()


def prepare_latent_dataset(
    data_dir: str,
    file_list: List[str],
    vae_model,
    output_dir: str,
    device: str = 'cpu',
    downsample_factor: int = 2,
    hr_spacing: float = 1.0
):
    """Pre-compute and save latent representations.

    Args:
        data_dir: Directory with NIfTI volumes
        file_list: List of volume filenames
        vae_model: Trained VAE model
        output_dir: Output directory for latent .npz files
        device: Device for encoding
        downsample_factor: Z-axis downsampling
        hr_spacing: HR z-spacing in mm
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    vae_model = vae_model.to(device)
    vae_model.eval()

    simulator = SliceProfileSimulator(profile_type='gaussian')

    print(f"Preparing latent dataset...")
    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Files: {len(file_list)}")

    for filename in tqdm(file_list, desc="Encoding volumes"):
        # Load volume
        nifti_path = Path(data_dir) / filename
        img = nib.load(nifti_path)
        volume = np.asarray(img.dataobj).astype(np.float32)

        # Generate LR-HR pair
        lr_volume, hr_volume = simulator.create_training_pair(
            volume, hr_spacing, downsample_factor
        )

        # Normalize
        hr_volume = _normalize_hu(hr_volume)
        lr_volume = _normalize_hu(lr_volume)

        # Upsample LR to HR size
        lr_upsampled = _upsample_volume(lr_volume, hr_volume.shape)

        # Process in chunks to avoid OOM
        chunk_size = 32  # Process 32 slices at a time
        depth = hr_volume.shape[0]

        hr_latent_chunks = []
        lr_latent_chunks = []

        with torch.no_grad():
            for start_idx in range(0, depth, chunk_size):
                end_idx = min(start_idx + chunk_size, depth)

                # Extract chunk
                hr_chunk = hr_volume[start_idx:end_idx]
                lr_chunk = lr_upsampled[start_idx:end_idx]

                # Convert to tensors
                hr_tensor = torch.from_numpy(hr_chunk).unsqueeze(0).unsqueeze(0).to(device)
                lr_tensor = torch.from_numpy(lr_chunk).unsqueeze(0).unsqueeze(0).to(device)

                # Encode chunk
                hr_latent_chunk = vae_model.encode_to_latent(hr_tensor, sample=False)
                lr_latent_chunk = vae_model.encode_to_latent(lr_tensor, sample=False)

                # Move to CPU and store
                hr_latent_chunks.append(hr_latent_chunk.squeeze(0).cpu())
                lr_latent_chunks.append(lr_latent_chunk.squeeze(0).cpu())

                # Clear GPU memory
                del hr_tensor, lr_tensor, hr_latent_chunk, lr_latent_chunk
                torch.cuda.empty_cache()

        # Concatenate chunks along depth dimension
        hr_latent = torch.cat(hr_latent_chunks, dim=1)  # Concatenate along D dimension (C, D, H, W)
        lr_latent = torch.cat(lr_latent_chunks, dim=1)

        # Save as .npz
        hr_latent_np = hr_latent.numpy()
        lr_latent_np = lr_latent.numpy()

        base_name = Path(filename).stem
        output_file = output_path / f"{base_name}_latent.npz"

        np.savez_compressed(
            output_file,
            hr_latent=hr_latent_np,
            lr_latent=lr_latent_np
        )

    print(f"✓ Latent dataset prepared: {len(file_list)} volumes encoded")


def _normalize_hu(volume: np.ndarray) -> np.ndarray:
    """Normalize HU values."""
    volume = np.clip(volume, -1024, 3071)
    volume = (volume + 1024) / 4095.0
    return volume.astype(np.float32)


def _upsample_volume(lr_volume: np.ndarray, target_shape: tuple) -> np.ndarray:
    """Upsample LR volume to target shape."""
    lr_tensor = torch.from_numpy(lr_volume).unsqueeze(0).unsqueeze(0)
    upsampled = torch.nn.functional.interpolate(
        lr_tensor,
        size=target_shape,
        mode='trilinear',
        align_corners=False
    )
    return upsampled.squeeze(0).squeeze(0).numpy()


def create_latent_dataloaders(
    latent_dir: str,
    train_files: List[str],
    val_files: List[str],
    batch_size: int = 4,
    patch_size: Tuple[int, int, int] = (16, 64, 64),
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders for latents.

    Args:
        latent_dir: Directory with pre-computed latents
        train_files: List of training filenames
        val_files: List of validation filenames
        batch_size: Batch size
        patch_size: Patch size in latent space
        num_workers: Number of workers

    Returns:
        train_loader, val_loader
    """
    train_dataset = LatentDataset(
        latent_dir=latent_dir,
        file_list=train_files,
        patch_size=patch_size,
        augment=True
    )

    val_dataset = LatentDataset(
        latent_dir=latent_dir,
        file_list=val_files,
        patch_size=patch_size,
        augment=False
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    print(f"Latent dataloaders created:")
    print(f"  Train: {len(train_dataset)} volumes")
    print(f"  Val: {len(val_dataset)} volumes")
    print(f"  Batch size: {batch_size}")
    print(f"  Patch size (latent): {patch_size}")

    return train_loader, val_loader
