"""Simplified 3D U-Net that works on CPU (no MPS 3D ops)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleUNet3D(nn.Module):
    """Simple 3D U-Net for through-plane super-resolution.

    Works on CPU without MPS-unsupported operations.
    """

    def __init__(self, in_channels=1, out_channels=1, base_channels=16):
        super().__init__()

        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down1 = nn.Conv3d(base_channels, base_channels, 3, stride=2, padding=1)

        self.enc2 = nn.Sequential(
            nn.Conv3d(base_channels, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Conv3d(base_channels*2, base_channels*2, 3, stride=2, padding=1)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*4, base_channels*4, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Decoder
        self.up2 = nn.ConvTranspose3d(base_channels*4, base_channels*2, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv3d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels*2, base_channels*2, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = nn.ConvTranspose3d(base_channels*2, base_channels, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv3d(base_channels*2, base_channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(base_channels, base_channels, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        self.out = nn.Conv3d(base_channels, out_channels, 1)

    def forward(self, x):
        input_shape = x.shape[2:]  # (D, H, W)

        # Encoder
        e1 = self.enc1(x)
        e1_down = self.down1(e1)

        e2 = self.enc2(e1_down)
        e2_down = self.down2(e2)

        # Bottleneck
        b = self.bottleneck(e2_down)

        # Decoder
        d2 = self.up2(b)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode='trilinear', align_corners=False)
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode='trilinear', align_corners=False)
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        out = self.out(d1)

        # Final upsampling: 2x along z-axis only
        target_shape = (input_shape[0] * 2, input_shape[1], input_shape[2])
        out = F.interpolate(out, size=target_shape, mode='trilinear', align_corners=False)

        return out


def create_simple_model(device='cpu'):
    """Create simple U-Net for CPU training."""
    model = SimpleUNet3D(in_channels=1, out_channels=1, base_channels=16)
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params:,} parameters")
    print(f"Device: {device}")

    return model