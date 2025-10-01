"""Loss functions for CT super-resolution with HU awareness."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HUL1Loss(nn.Module):
    """L1 loss with optional body mask for HU values."""

    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Compute masked L1 loss.

        Args:
            pred: Predicted HU values, shape (B, C, D, H, W)
            target: Ground truth HU values
            mask: Optional binary mask (True=body, False=air)

        Returns:
            Loss value
        """
        diff = torch.abs(pred - target)

        if mask is not None:
            diff = diff * mask.float()
            if self.reduction == 'mean':
                return diff.sum() / (mask.sum() + 1e-8)
            elif self.reduction == 'sum':
                return diff.sum()
        else:
            if self.reduction == 'mean':
                return diff.mean()
            elif self.reduction == 'sum':
                return diff.sum()

        return diff


class SSIMLoss(nn.Module):
    """SSIM loss for structural similarity."""

    def __init__(self, window_size: int = 11, c1: float = 0.01**2, c2: float = 0.03**2):
        super().__init__()
        self.window_size = window_size
        self.c1 = c1
        self.c2 = c2

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute 1 - SSIM.

        Args:
            pred: Predicted values
            target: Ground truth

        Returns:
            SSIM loss (lower is better)
        """
        # Use 2D SSIM on central slices for efficiency
        # Extract middle slice from each volume
        mid_idx = pred.shape[2] // 2
        pred_2d = pred[:, :, mid_idx, :, :]
        target_2d = target[:, :, mid_idx, :, :]

        # Compute means
        mu_pred = F.avg_pool2d(pred_2d, self.window_size, stride=1, padding=self.window_size // 2)
        mu_target = F.avg_pool2d(target_2d, self.window_size, stride=1, padding=self.window_size // 2)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_pred_target = mu_pred * mu_target

        # Compute variances and covariance
        sigma_pred_sq = F.avg_pool2d(
            pred_2d ** 2, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_pred_sq
        sigma_target_sq = F.avg_pool2d(
            target_2d ** 2, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_target_sq
        sigma_pred_target = F.avg_pool2d(
            pred_2d * target_2d, self.window_size, stride=1, padding=self.window_size // 2
        ) - mu_pred_target

        # SSIM formula
        numerator = (2 * mu_pred_target + self.c1) * (2 * sigma_pred_target + self.c2)
        denominator = (mu_pred_sq + mu_target_sq + self.c1) * (sigma_pred_sq + sigma_target_sq + self.c2)

        ssim = numerator / (denominator + 1e-8)
        return 1 - ssim.mean()


class GradientLoss(nn.Module):
    """Z-gradient loss to encourage sharp through-plane edges."""

    def __init__(self):
        super().__init__()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute L1 loss on z-gradients.

        Args:
            pred: Predicted volume, shape (B, C, D, H, W)
            target: Target volume

        Returns:
            Gradient loss
        """
        # Compute gradients along z-axis
        pred_grad = pred[:, :, 1:, :, :] - pred[:, :, :-1, :, :]
        target_grad = target[:, :, 1:, :, :] - target[:, :, :-1, :, :]

        return F.l1_loss(pred_grad, target_grad)


class CombinedLoss(nn.Module):
    """Combined loss with HU L1, SSIM, and z-gradient terms."""

    def __init__(
        self,
        l1_weight: float = 1.0,
        ssim_weight: float = 0.1,
        grad_weight: float = 0.1
    ):
        super().__init__()
        self.l1_weight = l1_weight
        self.ssim_weight = ssim_weight
        self.grad_weight = grad_weight

        self.l1_loss = HUL1Loss()
        self.ssim_loss = SSIMLoss()
        self.grad_loss = GradientLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> tuple[torch.Tensor, dict]:
        """Compute combined loss.

        Args:
            pred: Predicted volume
            target: Target volume
            mask: Optional body mask

        Returns:
            total_loss: Weighted sum of all losses
            loss_dict: Dictionary of individual loss values
        """
        l1 = self.l1_loss(pred, target, mask)
        ssim = self.ssim_loss(pred, target)
        grad = self.grad_loss(pred, target)

        total = self.l1_weight * l1 + self.ssim_weight * ssim + self.grad_weight * grad

        loss_dict = {
            'total': total.item(),
            'l1': l1.item(),
            'ssim': ssim.item(),
            'grad': grad.item()
        }

        return total, loss_dict
