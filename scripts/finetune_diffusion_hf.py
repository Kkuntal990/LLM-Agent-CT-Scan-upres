#!/usr/bin/env python3
"""
Fine-tune 3D Diffusion Model using Hugging Face Diffusers Library

Simplified approach using diffusers library with minimal custom code.
Uses microsoft/mri-autoencoder-v0.1 as VAE and custom 3D UNet.
"""

import argparse
import os
from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from diffusers import UNet3DConditionModel, DDPMScheduler, AutoencoderKL
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from accelerate import Accelerator
from tqdm.auto import tqdm
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from data.ape_dataset_hf import APEDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune 3D diffusion model for CT super-resolution")

    # Data arguments
    parser.add_argument("--ape-cache-dir", type=str, required=True,
                        help="Path to APE-data cache directory")
    parser.add_argument("--subset", type=str, default="APE", choices=["APE", "non APE"])
    parser.add_argument("--scale-factor", type=int, default=3)
    parser.add_argument("--patch-size", type=int, nargs=3, default=[64, 128, 128],
                        help="3D patch size (Z H W)")
    parser.add_argument("--train-split", type=float, default=0.9)

    # Model arguments
    parser.add_argument("--vae-model", type=str, default="microsoft/mri-autoencoder-v0.1",
                        help="Pretrained VAE model")
    parser.add_argument("--unet-channels", type=int, default=128,
                        help="Base channels for UNet")
    parser.add_argument("--use-ema", action="store_true",
                        help="Use exponential moving average")
    parser.add_argument("--ema-decay", type=float, default=0.9999)

    # Training arguments
    parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--lr-scheduler", type=str, default="cosine",
                        choices=["linear", "cosine", "constant"])
    parser.add_argument("--warmup-steps", type=int, default=500)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=4)
    parser.add_argument("--mixed-precision", type=str, default="no",
                        choices=["no", "fp16", "bf16"],
                        help="Mixed precision training (fp16/bf16 only work on CUDA)")
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # Diffusion arguments
    parser.add_argument("--num-train-timesteps", type=int, default=1000)
    parser.add_argument("--beta-schedule", type=str, default="scaled_linear",
                        choices=["linear", "scaled_linear", "squaredcos_cap_v2"])
    parser.add_argument("--prediction-type", type=str, default="epsilon",
                        choices=["epsilon", "v_prediction", "sample"])

    # Checkpointing
    parser.add_argument("--output-dir", type=str, default="./checkpoints/diffusion_hf")
    parser.add_argument("--save-every", type=int, default=5)
    parser.add_argument("--resume-from", type=str, default=None)

    # Other
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Limit dataset size (for debugging)")

    return parser.parse_args()


def create_unet_3d(
    in_channels: int = 8,  # 4 (noisy latent) + 4 (LR conditioning)
    out_channels: int = 4,
    base_channels: int = 128,
    block_out_channels: tuple = (128, 256, 384, 512),
    down_block_types: tuple = (
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "CrossAttnDownBlock3D",
        "DownBlock3D",
    ),
    up_block_types: tuple = (
        "UpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
        "CrossAttnUpBlock3D",
    ),
    layers_per_block: int = 2,
    cross_attention_dim: int = 64,
):
    """Create 3D UNet for latent diffusion."""
    return UNet3DConditionModel(
        in_channels=in_channels,
        out_channels=out_channels,
        block_out_channels=block_out_channels,
        down_block_types=down_block_types,
        up_block_types=up_block_types,
        layers_per_block=layers_per_block,
        cross_attention_dim=cross_attention_dim,
    )


def train_one_epoch(
    model,
    vae,
    noise_scheduler,
    dataloader,
    optimizer,
    lr_scheduler,
    accelerator,
    epoch,
    args,
    ema_model=None,
):
    """Train for one epoch."""
    model.train()
    vae.eval()  # Keep VAE frozen

    progress_bar = tqdm(total=len(dataloader), disable=not accelerator.is_local_main_process)
    progress_bar.set_description(f"Epoch {epoch}")

    total_loss = 0.0

    for step, (lr_volumes, hr_volumes) in enumerate(dataloader):
        with accelerator.accumulate(model):
            # Encode to latent space (slice-by-slice for 2D VAE)
            with torch.no_grad():
                # VAE is 2D, so we need to process each slice
                batch_size = lr_volumes.shape[0]
                channels = lr_volumes.shape[1]
                lr_depth = lr_volumes.shape[2]
                hr_depth = hr_volumes.shape[2]
                height = lr_volumes.shape[3]
                width = lr_volumes.shape[4]

                # Process LR volumes
                lr_2d = lr_volumes.permute(0, 2, 1, 3, 4).reshape(batch_size * lr_depth, channels, height, width)
                lr_latent_2d = vae.encode(lr_2d).latent_dist.sample()

                # Process HR volumes
                hr_2d = hr_volumes.permute(0, 2, 1, 3, 4).reshape(batch_size * hr_depth, channels, height, width)
                hr_latent_2d = vae.encode(hr_2d).latent_dist.sample()

                # Reshape back to 3D: (batch, latent_channels, depth, latent_h, latent_w)
                latent_channels = lr_latent_2d.shape[1]
                latent_h = lr_latent_2d.shape[2]
                latent_w = lr_latent_2d.shape[3]

                lr_latents = lr_latent_2d.reshape(batch_size, lr_depth, latent_channels, latent_h, latent_w)
                lr_latents = lr_latents.permute(0, 2, 1, 3, 4)  # (batch, latent_channels, lr_depth, latent_h, latent_w)

                hr_latents = hr_latent_2d.reshape(batch_size, hr_depth, latent_channels, latent_h, latent_w)
                hr_latents = hr_latents.permute(0, 2, 1, 3, 4)  # (batch, latent_channels, hr_depth, latent_h, latent_w)

                # Scale latents (VAE uses 0.18215 scaling factor)
                lr_latents = lr_latents * 0.18215
                hr_latents = hr_latents * 0.18215

            # Sample noise
            noise = torch.randn_like(hr_latents)
            batch_size = hr_latents.shape[0]

            # Sample random timesteps
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (batch_size,),
                device=hr_latents.device, dtype=torch.long
            )

            # Add noise to HR latents
            noisy_latents = noise_scheduler.add_noise(hr_latents, noise, timesteps)

            # Concatenate noisy latents with LR conditioning
            model_input = torch.cat([noisy_latents, lr_latents], dim=1)

            # Create dummy encoder_hidden_states for cross attention
            # Shape: (batch, seq_len, cross_attention_dim)
            encoder_hidden_states = torch.zeros(
                batch_size, 1, 64,
                device=model_input.device,
                dtype=model_input.dtype
            )

            # Predict noise (or v-prediction/sample depending on config)
            model_pred = model(
                model_input,
                timesteps,
                encoder_hidden_states=encoder_hidden_states,
                return_dict=False
            )[0]

            # Calculate loss based on prediction type
            if noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif noise_scheduler.config.prediction_type == "v_prediction":
                target = noise_scheduler.get_velocity(hr_latents, noise, timesteps)
            elif noise_scheduler.config.prediction_type == "sample":
                target = hr_latents
            else:
                raise ValueError(f"Unknown prediction type: {noise_scheduler.config.prediction_type}")

            loss = F.mse_loss(model_pred, target)

            # Backpropagation
            accelerator.backward(loss)

            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

            # Update EMA
            if ema_model is not None and accelerator.sync_gradients:
                ema_model.step(model.parameters())

        # Logging
        total_loss += loss.item()

        logs = {
            "loss": loss.item(),
            "lr": lr_scheduler.get_last_lr()[0],
            "step": epoch * len(dataloader) + step,
        }
        progress_bar.set_postfix(**logs)
        progress_bar.update(1)

    progress_bar.close()

    avg_loss = total_loss / len(dataloader)
    accelerator.print(f"Epoch {epoch} - Average Loss: {avg_loss:.4f}")

    return avg_loss


def main():
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Force mixed_precision to "no" for MPS and CPU devices
    if args.device in ["mps", "cpu"]:
        if args.mixed_precision != "no":
            print(f"Warning: Mixed precision not supported on {args.device}, setting to 'no'")
            args.mixed_precision = "no"

    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
    )

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dataset
    accelerator.print("Loading APE-data dataset...")
    full_dataset = APEDataset(
        ape_cache_dir=args.ape_cache_dir,
        subset=args.subset,
        scale_factor=args.scale_factor,
        patch_size=tuple(args.patch_size),
        max_samples=args.max_samples,
    )

    # Train/val split
    train_size = int(args.train_split * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    accelerator.print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    # Load VAE
    accelerator.print(f"Loading VAE: {args.vae_model}")
    vae = AutoencoderKL.from_pretrained(args.vae_model)
    vae.requires_grad_(False)  # Freeze VAE

    # Create UNet
    accelerator.print("Creating 3D UNet...")
    unet = create_unet_3d(
        base_channels=args.unet_channels,
    )

    # Create noise scheduler
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.num_train_timesteps,
        beta_schedule=args.beta_schedule,
        prediction_type=args.prediction_type,
    )

    # Create EMA model
    ema_model = None
    if args.use_ema:
        ema_model = EMAModel(
            unet.parameters(),
            decay=args.ema_decay,
            model_cls=UNet3DConditionModel,
            model_config=unet.config,
        )

    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=args.learning_rate,
        betas=(0.9, 0.999),
        weight_decay=0.01,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.warmup_steps,
        num_training_steps=args.num_epochs * len(train_loader),
    )

    # Prepare for distributed training
    unet, vae, optimizer, train_loader, lr_scheduler = accelerator.prepare(
        unet, vae, optimizer, train_loader, lr_scheduler
    )

    # Resume from checkpoint
    start_epoch = 0
    if args.resume_from:
        accelerator.print(f"Resuming from {args.resume_from}")
        accelerator.load_state(args.resume_from)

    # Training loop
    accelerator.print("Starting training...")
    for epoch in range(start_epoch, args.num_epochs):
        avg_loss = train_one_epoch(
            unet, vae, noise_scheduler, train_loader,
            optimizer, lr_scheduler, accelerator, epoch, args, ema_model
        )

        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = output_dir / f"checkpoint-epoch-{epoch+1}"
            accelerator.save_state(save_path)
            accelerator.print(f"Saved checkpoint to {save_path}")

            # Save UNet separately for easy loading
            unwrapped_unet = accelerator.unwrap_model(unet)
            unwrapped_unet.save_pretrained(output_dir / f"unet-epoch-{epoch+1}")

            # Save EMA model if used
            if ema_model is not None:
                ema_model.save_pretrained(output_dir / f"unet-ema-epoch-{epoch+1}")

    # Final save
    accelerator.print("Training complete! Saving final model...")
    unwrapped_unet = accelerator.unwrap_model(unet)
    unwrapped_unet.save_pretrained(output_dir / "unet-final")

    if ema_model is not None:
        ema_model.save_pretrained(output_dir / "unet-ema-final")

    accelerator.print(f"Model saved to {output_dir}")


if __name__ == "__main__":
    main()
