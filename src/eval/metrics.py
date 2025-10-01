"""Evaluation metrics for CT super-resolution quality assessment."""

import numpy as np
from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from typing import Optional, Dict
import torch


def compute_psnr(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0
) -> float:
    """Compute Peak Signal-to-Noise Ratio.

    Args:
        pred: Predicted volume
        target: Ground truth volume
        mask: Optional binary mask for body region
        data_range: Maximum value range (default 1.0 for normalized data)

    Returns:
        PSNR in dB
    """
    if mask is not None:
        pred = pred[mask]
        target = target[mask]

    mse = np.mean((pred - target) ** 2)
    if mse == 0:
        return float('inf')

    psnr = 20 * np.log10(data_range / np.sqrt(mse))
    return float(psnr)


def compute_ssim(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    data_range: float = 1.0
) -> float:
    """Compute Structural Similarity Index.

    Computes SSIM slice-by-slice and averages.

    Args:
        pred: Predicted volume, shape (D, H, W)
        target: Ground truth volume
        mask: Optional binary mask
        data_range: Data range

    Returns:
        Average SSIM value
    """
    ssim_values = []

    for i in range(pred.shape[0]):
        pred_slice = pred[i]
        target_slice = target[i]

        if mask is not None:
            mask_slice = mask[i]
            # Only compute SSIM if slice has sufficient body content
            if mask_slice.sum() < 100:
                continue

        ssim = structural_similarity(
            target_slice,
            pred_slice,
            data_range=data_range,
            win_size=11
        )
        ssim_values.append(ssim)

    return float(np.mean(ssim_values)) if ssim_values else 0.0


def compute_hu_mae(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    denormalize: bool = True
) -> float:
    """Compute Mean Absolute Error in HU units.

    Args:
        pred: Predicted volume (normalized or HU)
        target: Ground truth volume
        mask: Optional body mask
        denormalize: If True, assumes input is normalized [0,1] and converts to HU

    Returns:
        MAE in Hounsfield Units
    """
    pred_hu = pred.copy()
    target_hu = target.copy()

    if denormalize:
        # Convert from [0, 1] normalized to HU range [-1024, 3071]
        pred_hu = pred_hu * 4095.0 - 1024.0
        target_hu = target_hu * 4095.0 - 1024.0

    if mask is not None:
        pred_hu = pred_hu[mask]
        target_hu = target_hu[mask]

    mae = np.mean(np.abs(pred_hu - target_hu))
    return float(mae)


def compute_lpips(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    device: str = 'mps'
) -> float:
    """Compute LPIPS (Learned Perceptual Image Patch Similarity).

    Computes LPIPS on central slices using pretrained network.

    Args:
        pred: Predicted volume
        target: Ground truth volume
        mask: Optional mask
        device: Device for computation

    Returns:
        Average LPIPS score (lower is better)
    """
    try:
        import lpips
    except ImportError:
        print("Warning: lpips not installed, returning 0.0")
        return 0.0

    # Initialize LPIPS model
    loss_fn = lpips.LPIPS(net='alex').to(device)
    loss_fn.eval()

    lpips_values = []

    # Sample slices evenly through volume
    num_samples = min(10, pred.shape[0])
    indices = np.linspace(0, pred.shape[0] - 1, num_samples, dtype=int)

    with torch.no_grad():
        for idx in indices:
            pred_slice = pred[idx]
            target_slice = target[idx]

            # Convert to 3-channel (RGB) by repeating
            pred_rgb = np.stack([pred_slice] * 3, axis=0)  # (3, H, W)
            target_rgb = np.stack([target_slice] * 3, axis=0)

            # Convert to tensors and normalize to [-1, 1]
            pred_tensor = torch.from_numpy(pred_rgb).unsqueeze(0).float()
            target_tensor = torch.from_numpy(target_rgb).unsqueeze(0).float()

            pred_tensor = (pred_tensor - 0.5) * 2  # [0,1] -> [-1,1]
            target_tensor = (target_tensor - 0.5) * 2

            pred_tensor = pred_tensor.to(device)
            target_tensor = target_tensor.to(device)

            # Compute LPIPS
            lpips_val = loss_fn(pred_tensor, target_tensor)
            lpips_values.append(lpips_val.item())

    return float(np.mean(lpips_values)) if lpips_values else 0.0


def evaluate_volume(
    pred: np.ndarray,
    target: np.ndarray,
    mask: Optional[np.ndarray] = None,
    compute_lpips_metric: bool = False,
    device: str = 'mps'
) -> Dict[str, float]:
    """Compute all evaluation metrics for a volume.

    Args:
        pred: Predicted volume (normalized [0,1])
        target: Ground truth volume (normalized [0,1])
        mask: Optional body mask
        compute_lpips_metric: Whether to compute LPIPS (slower)
        device: Device for LPIPS computation

    Returns:
        Dictionary with all metrics
    """
    metrics = {
        'psnr': compute_psnr(pred, target, mask, data_range=1.0),
        'ssim': compute_ssim(pred, target, mask, data_range=1.0),
        'hu_mae': compute_hu_mae(pred, target, mask, denormalize=True)
    }

    if compute_lpips_metric:
        metrics['lpips'] = compute_lpips(pred, target, mask, device)

    return metrics


def print_metrics(metrics: Dict[str, float]):
    """Pretty print evaluation metrics.

    Args:
        metrics: Dictionary of metric values
    """
    print("\n" + "=" * 50)
    print("Evaluation Metrics")
    print("=" * 50)
    print(f"  PSNR:     {metrics['psnr']:.2f} dB")
    print(f"  SSIM:     {metrics['ssim']:.4f}")
    print(f"  HU-MAE:   {metrics['hu_mae']:.2f} HU")
    if 'lpips' in metrics:
        print(f"  LPIPS:    {metrics['lpips']:.4f}")
    print("=" * 50 + "\n")
