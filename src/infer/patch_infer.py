"""Patch-wise inference with overlap-tiling and Gaussian blending.

Implements sliding window inference to avoid memory issues and seams.
Optimized for Apple Silicon MPS backend.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Optional
from tqdm import tqdm


def create_gaussian_weight(
    shape: Tuple[int, int, int],
    sigma_scale: float = 0.125
) -> np.ndarray:
    """Create 3D Gaussian weight map for blending overlapping patches.

    Args:
        shape: (D, H, W) patch shape
        sigma_scale: Scale factor for sigma relative to patch size

    Returns:
        Gaussian weight map, shape (D, H, W)
    """
    d, h, w = shape

    # Create 1D Gaussian for each dimension
    def gaussian_1d(length, sigma_scale):
        sigma = length * sigma_scale
        center = length / 2
        x = np.arange(length)
        g = np.exp(-((x - center) ** 2) / (2 * sigma ** 2))
        return g

    # Create 3D Gaussian as outer product
    gz = gaussian_1d(d, sigma_scale)
    gy = gaussian_1d(h, sigma_scale)
    gx = gaussian_1d(w, sigma_scale)

    weight = np.outer(gz, np.outer(gy, gx).ravel()).reshape(d, h, w)
    return weight.astype(np.float32)


class PatchInference:
    """Patch-wise inference engine with Gaussian blending for seamless output.

    Processes large volumes in overlapping tiles to avoid memory issues
    and eliminates seams using Gaussian-weighted blending.
    """

    def __init__(
        self,
        model: nn.Module,
        patch_size: Tuple[int, int, int] = (16, 160, 160),
        overlap: Tuple[int, int, int] = (8, 32, 32),
        device: str = 'mps',
        batch_size: int = 1,
        upscale_factor: int = 2
    ):
        """Initialize patch inference engine.

        Args:
            model: Trained super-resolution model
            patch_size: Size of input patches (D, H, W)
            overlap: Overlap between adjacent patches (D, H, W)
            device: Device for inference ('mps', 'cuda', or 'cpu')
            batch_size: Number of patches to process in parallel
            upscale_factor: Z-axis upsampling factor
        """
        self.model = model.to(device).eval()
        self.patch_size = patch_size
        self.overlap = overlap
        self.device = device
        self.batch_size = batch_size
        self.upscale_factor = upscale_factor

        # Create Gaussian weight for input patch size
        self.input_weight = create_gaussian_weight(patch_size)

        # Create Gaussian weight for output patch size (z-axis upsampled)
        output_patch_size = (
            patch_size[0] * upscale_factor,
            patch_size[1],
            patch_size[2]
        )
        self.output_weight = create_gaussian_weight(output_patch_size)

    def __call__(
        self,
        volume: np.ndarray,
        progress: bool = True
    ) -> np.ndarray:
        """Run inference on full volume with patch-wise processing.

        Args:
            volume: Input LR volume, shape (D, H, W)
            progress: Show progress bar

        Returns:
            Super-resolved volume, shape (D*upscale_factor, H, W)
        """
        return self.infer(volume, progress)

    def infer(
        self,
        volume: np.ndarray,
        progress: bool = True
    ) -> np.ndarray:
        """Run patch-wise inference with Gaussian blending.

        Args:
            volume: Input volume (D, H, W) in normalized HU
            progress: Show progress bar

        Returns:
            Super-resolved volume
        """
        d, h, w = volume.shape
        patch_d, patch_h, patch_w = self.patch_size
        overlap_d, overlap_h, overlap_w = self.overlap
        stride_d = patch_d - overlap_d
        stride_h = patch_h - overlap_h
        stride_w = patch_w - overlap_w

        # Calculate output dimensions
        output_d = d * self.upscale_factor
        output_h = h
        output_w = w

        # Initialize output accumulator and weight accumulator
        output = np.zeros((output_d, output_h, output_w), dtype=np.float32)
        weight_sum = np.zeros((output_d, output_h, output_w), dtype=np.float32)

        # Calculate patch positions
        positions = []
        for z in range(0, d - overlap_d, stride_d):
            for y in range(0, h - overlap_h, stride_h):
                for x in range(0, w - overlap_w, stride_w):
                    # Ensure patch doesn't exceed volume bounds
                    z_end = min(z + patch_d, d)
                    y_end = min(y + patch_h, h)
                    x_end = min(x + patch_w, w)

                    # Adjust start if needed to maintain patch size
                    z_start = max(0, z_end - patch_d)
                    y_start = max(0, y_end - patch_h)
                    x_start = max(0, x_end - patch_w)

                    positions.append((z_start, y_start, x_start))

        # Process patches
        iterator = tqdm(positions, desc="Processing patches") if progress else positions

        with torch.no_grad():
            for z_start, y_start, x_start in iterator:
                # Extract patch
                patch = volume[
                    z_start:z_start + patch_d,
                    y_start:y_start + patch_h,
                    x_start:x_start + patch_w
                ]

                # Handle edge cases where patch is smaller than expected
                if patch.shape != self.patch_size:
                    # Pad to expected size
                    padded = np.zeros(self.patch_size, dtype=np.float32)
                    padded[:patch.shape[0], :patch.shape[1], :patch.shape[2]] = patch
                    patch = padded

                # Convert to tensor and add batch/channel dims
                patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)
                patch_tensor = patch_tensor.to(self.device)

                # Inference
                output_patch_tensor = self.model(patch_tensor)

                # Move to CPU and remove batch/channel dims
                output_patch = output_patch_tensor.squeeze(0).squeeze(0).cpu().numpy()

                # Calculate output position (z is upsampled)
                out_z_start = z_start * self.upscale_factor
                out_z_end = out_z_start + output_patch.shape[0]
                out_y_start = y_start
                out_y_end = y_start + patch_h
                out_x_start = x_start
                out_x_end = x_start + patch_w

                # Get weight for this patch
                weight = self.output_weight

                # Handle size mismatch for edge patches
                actual_shape = output_patch.shape
                weight = weight[:actual_shape[0], :actual_shape[1], :actual_shape[2]]

                # Accumulate with Gaussian weighting
                output[
                    out_z_start:out_z_end,
                    out_y_start:out_y_end,
                    out_x_start:out_x_end
                ] += output_patch * weight

                weight_sum[
                    out_z_start:out_z_end,
                    out_y_start:out_y_end,
                    out_x_start:out_x_end
                ] += weight

        # Normalize by accumulated weights to get final blended result
        output = output / (weight_sum + 1e-8)

        return output

    def infer_single_patch(self, patch: np.ndarray) -> np.ndarray:
        """Infer on a single patch without blending.

        Args:
            patch: Input patch, shape (D, H, W)

        Returns:
            Output patch, shape (D*upscale_factor, H, W)
        """
        with torch.no_grad():
            patch_tensor = torch.from_numpy(patch).unsqueeze(0).unsqueeze(0)
            patch_tensor = patch_tensor.to(self.device)
            output_tensor = self.model(patch_tensor)
            output = output_tensor.squeeze(0).squeeze(0).cpu().numpy()
        return output


def run_inference(
    model: nn.Module,
    volume: np.ndarray,
    device: str = 'mps',
    patch_size: Tuple[int, int, int] = (16, 160, 160),
    overlap: Tuple[int, int, int] = (8, 32, 32),
    upscale_factor: int = 2
) -> np.ndarray:
    """Convenience function for running patch-wise inference.

    Args:
        model: Trained SR model
        volume: Input volume in normalized HU
        device: Device for inference
        patch_size: Patch size for processing
        overlap: Overlap between patches
        upscale_factor: Z-axis upsampling factor

    Returns:
        Super-resolved volume
    """
    inference_engine = PatchInference(
        model=model,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        upscale_factor=upscale_factor
    )

    return inference_engine.infer(volume, progress=True)
