#!/bin/bash

# Complete training pipeline for Custom Implementation - GOOGLE COLAB VERSION
# Uses APE-data with full SOTA features (ResShift, IRControlNet, etc.)

set -e  # Exit on error

#==============================================================================
# CONFIGURATION
#==============================================================================
# Dataset options
DOWNLOAD_DATA=true             # Set to false if you already have the data
DOWNLOAD_DIR="./data/ape-raw"  # Where to download raw data
OUTPUT_DIR="./data/ape-nifti"  # Where to store converted NIfTI files
SUBSET="APE"                   # "APE" or "non APE"
MAX_SAMPLES=""                 # Leave empty for all samples, or set a number (e.g., 10)

# Training options
DEVICE="cuda"                  # Google Colab uses CUDA
VAE_EPOCHS=10
DIFFUSION_EPOCHS=20
BATCH_SIZE=4                   # Colab can handle larger batches

# Advanced: Use cached HF data if available
# If you already have data in HF cache, set this path
HF_CACHE_PATH="/root/.cache/huggingface/hub/datasets--t2ance--APE-data/snapshots"
USE_HF_CACHE=false  # Set to true to use cached data instead of downloading

#==============================================================================
# Determine APE_DIR
#==============================================================================
if [ "$USE_HF_CACHE" = true ] && [ -d "$HF_CACHE_PATH" ]; then
    # Find the latest snapshot in cache
    LATEST_SNAPSHOT=$(ls -t "$HF_CACHE_PATH" | head -1)
    APE_DIR="$HF_CACHE_PATH/$LATEST_SNAPSHOT"
    echo "Using cached HuggingFace data: $APE_DIR"
    DOWNLOAD_DATA=false
else
    APE_DIR="$DOWNLOAD_DIR"
fi

echo "=== Custom Implementation Training Pipeline (Google Colab) ==="
echo "Configuration:"
echo "  Device: $DEVICE"
echo "  Dataset: $SUBSET"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo "  Download data: $DOWNLOAD_DATA"
echo "  Data source: $APE_DIR"
echo ""

#==============================================================================
# Step 0: Download APE-data from HuggingFace (if needed)
#==============================================================================
if [ "$DOWNLOAD_DATA" = true ]; then
    echo "Step 0/4: Downloading APE-data from HuggingFace..."

    if [ -d "$DOWNLOAD_DIR/$SUBSET" ] && [ "$(ls -A $DOWNLOAD_DIR/$SUBSET 2>/dev/null)" ]; then
        echo "  → Data already downloaded, skipping..."
    else
        python scripts/download_ape_data.py \
            --output-dir "$DOWNLOAD_DIR" \
            --subset "$SUBSET"
    fi
    echo ""
fi

#==============================================================================
# Step 1: Convert APE-data to NIfTI
#==============================================================================
echo "Step 1/4: Converting APE-data to NIfTI format..."
python scripts/convert_ape_data.py \
    --ape-cache-dir "$APE_DIR" \
    --subset "$SUBSET" \
    --output-dir "$OUTPUT_DIR" \
    --train-split 0.8 \
    --val-split 0.1 \
    ${MAX_SAMPLES:+--max-samples $MAX_SAMPLES} \
    --skip-existing
echo "  → Conversion complete (existing files were skipped)"

#==============================================================================
# Step 2: Fine-tune VAE on CT data
#==============================================================================
echo ""
echo "Step 2/4: Fine-tuning VAE on CT data..."
python scripts/finetune_vae.py \
    --data-dir "$OUTPUT_DIR/$SUBSET" \
    --train-split "$OUTPUT_DIR/$SUBSET/train_files.txt" \
    --val-split "$OUTPUT_DIR/$SUBSET/val_files.txt" \
    --device $DEVICE \
    --epochs $VAE_EPOCHS \
    --batch-size $BATCH_SIZE \
    --learning-rate 1e-4 \
    --kl-weight 0.0001 \
    --checkpoint-dir ./checkpoints/vae_ape

#==============================================================================
# Step 3: Prepare latent dataset
#==============================================================================
echo ""
echo "Step 3/4: Preparing latent dataset..."
python scripts/prepare_latent_dataset.py \
    --data-dir "$OUTPUT_DIR/$SUBSET" \
    --file-list "$OUTPUT_DIR/$SUBSET/train_files.txt" \
    --vae-checkpoint ./checkpoints/vae_ape/best_vae.pth \
    --output-dir ./data/ape-latents \
    --device $DEVICE

#==============================================================================
# Step 4: Train latent diffusion model
#==============================================================================
echo ""
echo "Step 4/4: Training latent diffusion model..."
python scripts/train_latent_diffusion.py \
    --latent-dir ./data/ape-latents \
    --train-files "$OUTPUT_DIR/$SUBSET/train_files.txt" \
    --val-files "$OUTPUT_DIR/$SUBSET/val_files.txt" \
    --vae-checkpoint ./checkpoints/vae_ape/best_vae.pth \
    --device $DEVICE \
    --epochs $DIFFUSION_EPOCHS \
    --batch-size 4 \
    --learning-rate 2e-5 \
    --predict-residual \
    --use-cfg \
    --checkpoint-dir ./checkpoints/latent_diffusion_ape

echo ""
echo "=== Training Complete! ==="
echo "Checkpoints saved:"
echo "  VAE: ./checkpoints/vae_ape/best_vae.pth"
echo "  Diffusion: ./checkpoints/latent_diffusion_ape/best_latent_diffusion.pth"
echo ""
echo "To run inference:"
echo "  python demo_latent_diffusion.py input.nii.gz output.nii.gz \\"
echo "      --vae-checkpoint ./checkpoints/vae_ape/best_vae.pth \\"
echo "      --diffusion-checkpoint ./checkpoints/latent_diffusion_ape/best_latent_diffusion.pth \\"
echo "      --device cuda"
