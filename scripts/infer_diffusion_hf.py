#!/usr/bin/env python3
"""
Inference using Hugging Face Diffusers 3D Diffusion Model

Loads fine-tuned UNet and performs DDIM sampling for fast inference.
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from diffusers import UNet3DConditionModel, DDIMScheduler, AutoencoderKL
from tqdm.auto import tqdm
import nibabel as nib
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with HF diffusion model")

    # Model paths
    parser.add_argument("--unet-path", type=str, required=True,
                        help="Path to fine-tuned UNet")
    parser.add_argument("--vae-model", type=str, default="microsoft/mri-autoencoder-v0.1",
                        help="VAE model")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use EMA weights")

    # Input/output
    parser.add_argument("--input-lr", type=str, required=True,
                        help="Path to LR NIfTI volume")
    parser.add_argument("--output-hr", type=str, required=True,
                        help="Path to save HR NIfTI volume")

    # Inference parameters
    parser.add_argument("--num-inference-steps", type=int, default=25,
                        help="Number of DDIM sampling steps")
    parser.add_argument("--guidance-scale", type=float, default=1.0,
                        help="Classifier-free guidance scale (1.0 = no guidance)")
    parser.add_argument("--eta", type=float, default=0.0,
                        help="DDIM eta (0.0 = deterministic)")

    # Processing
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 128, 128],
                        help="3D patch size for inference (Z H W)")
    parser.add_argument("--patch-overlap", type=int, default=16,
                        help="Overlap between patches")
    parser.add_argument("--normalize", action="store_true", default=True,
                        help="Normalize input to [-1, 1]")
    parser.add_argument("--hu-window", type=float, nargs=2, default=[-1000, 400],
                        help="HU windowing range")

    return parser.parse_args()


def load_nifti(path: str) -> tuple[np.ndarray, nib.Nifti1Image]:
    """Load NIfTI volume."""
    nii = nib.load(path)
    volume = nii.get_fdata().astype(np.float32)
    return volume, nii


def save_nifti(volume: np.ndarray, affine, path: str):
    """Save NIfTI volume."""
    nii = nib.Nifti1Image(volume, affine)
    nib.save(nii, path)


def normalize_volume(volume: np.ndarray, hu_window: tuple) -> np.ndarray:
    """Normalize volume to [-1, 1]."""
    volume = np.clip(volume, hu_window[0], hu_window[1])
    volume = 2 * (volume - hu_window[0]) / (hu_window[1] - hu_window[0]) - 1
    return volume


def denormalize_volume(volume: np.ndarray, hu_window: tuple) -> np.ndarray:
    """Denormalize volume from [-1, 1] to HU."""
    volume = (volume + 1) / 2 * (hu_window[1] - hu_window[0]) + hu_window[0]
    return volume


def extract_patches_3d(volume: np.ndarray, patch_size: tuple, overlap: int):
    """
    Extract overlapping 3D patches from volume.

    Returns:
        patches: List of (patch, (z_start, h_start, w_start))
    """
    z, h, w = volume.shape
    pz, ph, pw = patch_size
    stride = (pz - overlap, ph - overlap, pw - overlap)

    patches = []
    positions = []

    for z_start in range(0, max(1, z - pz + 1), stride[0]):
        for h_start in range(0, max(1, h - ph + 1), stride[1]):
            for w_start in range(0, max(1, w - pw + 1), stride[2]):
                # Ensure we don't go out of bounds
                z_end = min(z_start + pz, z)
                h_end = min(h_start + ph, h)
                w_end = min(w_start + pw, w)

                z_start = z_end - pz
                h_start = h_end - ph
                w_start = w_end - pw

                patch = volume[z_start:z_end, h_start:h_end, w_start:w_end]
                patches.append(patch)
                positions.append((z_start, h_start, w_start))

    return patches, positions


def merge_patches_3d(patches, positions, output_shape, overlap: int):
    """Merge overlapping patches using Gaussian weighting."""
    output = np.zeros(output_shape, dtype=np.float32)
    weight_map = np.zeros(output_shape, dtype=np.float32)

    pz, ph, pw = patches[0].shape

    # Create Gaussian weight
    z_weight = np.exp(-((np.arange(pz) - pz/2)**2) / (2 * (pz/4)**2))
    h_weight = np.exp(-((np.arange(ph) - ph/2)**2) / (2 * (ph/4)**2))
    w_weight = np.exp(-((np.arange(pw) - pw/2)**2) / (2 * (pw/4)**2))
    weight = z_weight[:, None, None] * h_weight[None, :, None] * w_weight[None, None, :]

    # Merge patches
    for patch, (z_start, h_start, w_start) in zip(patches, positions):
        output[z_start:z_start+pz, h_start:h_start+ph, w_start:w_start+pw] += patch * weight
        weight_map[z_start:z_start+pz, h_start:h_start+ph, w_start:w_start+pw] += weight

    # Normalize by weight
    output = output / (weight_map + 1e-8)

    return output


@torch.no_grad()
def run_inference(
    lr_volume: np.ndarray,
    unet: UNet3DConditionModel,
    vae: AutoencoderKL,
    scheduler: DDIMScheduler,
    device: str,
    patch_size: tuple,
    patch_overlap: int,
    num_inference_steps: int = 25,
    guidance_scale: float = 1.0,
    eta: float = 0.0,
):
    """
    Run diffusion inference on LR volume.

    Args:
        lr_volume: Low-resolution volume [Z, H, W]
        unet: Fine-tuned UNet model
        vae: VAE model
        scheduler: DDIM scheduler
        device: Device to run on
        patch_size: 3D patch size
        patch_overlap: Overlap between patches
        num_inference_steps: Number of sampling steps
        guidance_scale: CFG scale
        eta: DDIM eta parameter

    Returns:
        hr_volume: High-resolution volume [Z*scale, H, W]
    """
    unet.eval()
    vae.eval()

    # Extract patches
    lr_patches, positions = extract_patches_3d(lr_volume, patch_size, patch_overlap)

    hr_patches = []

    for lr_patch in tqdm(lr_patches, desc="Processing patches"):
        # Convert to tensor [1, 1, Z, H, W]
        lr_tensor = torch.from_numpy(lr_patch).unsqueeze(0).unsqueeze(0).to(device)

        # Encode to latent
        lr_latent = vae.encode(lr_tensor).latent_dist.sample()
        lr_latent = lr_latent * 0.18215

        # Initialize random noise
        latent_shape = lr_latent.shape
        latent = torch.randn(latent_shape, device=device)

        # Set timesteps
        scheduler.set_timesteps(num_inference_steps)

        # DDIM sampling loop
        for t in scheduler.timesteps:
            # Concatenate with LR conditioning
            model_input = torch.cat([latent, lr_latent], dim=1)

            # Predict noise
            noise_pred = unet(model_input, t, return_dict=False)[0]

            # Classifier-free guidance (if scale > 1.0)
            if guidance_scale != 1.0:
                # Predict unconditional
                model_input_uncond = torch.cat([latent, torch.zeros_like(lr_latent)], dim=1)
                noise_pred_uncond = unet(model_input_uncond, t, return_dict=False)[0]

                # Apply guidance
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred - noise_pred_uncond)

            # DDIM step
            latent = scheduler.step(noise_pred, t, latent, eta=eta).prev_sample

        # Decode latent to image
        latent = latent / 0.18215
        hr_tensor = vae.decode(latent).sample

        # Convert to numpy
        hr_patch = hr_tensor[0, 0].cpu().numpy()
        hr_patches.append(hr_patch)

    # Merge patches
    hr_shape = (
        lr_volume.shape[0],  # Assume same Z (through-plane SR handled by model)
        lr_volume.shape[1],
        lr_volume.shape[2]
    )
    hr_volume = merge_patches_3d(hr_patches, positions, hr_shape, patch_overlap)

    return hr_volume


def main():
    args = parse_args()

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"
    elif args.device == "mps" and not torch.backends.mps.is_available():
        print("MPS not available, falling back to CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")

    # Load models
    print(f"Loading UNet from {args.unet_path}")
    unet = UNet3DConditionModel.from_pretrained(args.unet_path)
    unet = unet.to(device)

    print(f"Loading VAE: {args.vae_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    vae = vae.to(device)

    # Create scheduler
    scheduler = DDIMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Load LR volume
    print(f"Loading LR volume: {args.input_lr}")
    lr_volume, lr_nii = load_nifti(args.input_lr)

    # Normalize
    if args.normalize:
        lr_volume = normalize_volume(lr_volume, tuple(args.hu_window))

    print(f"LR volume shape: {lr_volume.shape}")

    # Run inference
    print(f"Running inference with {args.num_inference_steps} steps...")
    hr_volume = run_inference(
        lr_volume=lr_volume,
        unet=unet,
        vae=vae,
        scheduler=scheduler,
        device=device,
        patch_size=tuple(args.patch_size),
        patch_overlap=args.patch_overlap,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        eta=args.eta,
    )

    # Denormalize
    if args.normalize:
        hr_volume = denormalize_volume(hr_volume, tuple(args.hu_window))

    print(f"HR volume shape: {hr_volume.shape}")

    # Save result
    print(f"Saving HR volume: {args.output_hr}")
    save_nifti(hr_volume, lr_nii.affine, args.output_hr)

    print("Inference complete!")


if __name__ == "__main__":
    main()
