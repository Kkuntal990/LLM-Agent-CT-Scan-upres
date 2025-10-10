#!/bin/bash

# Quick test script - Train on just 3 samples for testing
# This verifies the entire pipeline works before running full training

set -e  # Exit on error

echo "=========================================="
echo "Quick Test Training Pipeline"
echo "Testing with 3 samples only"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="./data/ape-nifti/APE"
LATENT_DIR="./data/ape-latents-test"
VAE_CHECKPOINT="./checkpoints/vae_ape/best_vae.pth"
OUTPUT_DIR="./checkpoints/test_run"
DEVICE="mps"  # Change to 'cpu' if needed
NUM_TEST_SAMPLES=3

# Create temporary file list with just 3 samples
echo "Creating test file list with $NUM_TEST_SAMPLES samples..."
head -n $NUM_TEST_SAMPLES "$DATA_DIR/train_files.txt" > /tmp/test_train_files.txt
head -n 1 "$DATA_DIR/val_files.txt" > /tmp/test_val_files.txt

echo "Test samples:"
cat /tmp/test_train_files.txt
echo ""

# Step 1: Prepare latents (just for test samples)
echo "Step 1/2: Preparing latents for test samples..."
python scripts/prepare_latent_dataset.py \
    --data-dir "$DATA_DIR" \
    --file-list /tmp/test_train_files.txt \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --output-dir "$LATENT_DIR" \
    --device cpu

echo ""
echo "Step 2/2: Training diffusion model (2 epochs only)..."
python scripts/train_latent_diffusion.py \
    --latent-dir "$LATENT_DIR" \
    --train-split /tmp/test_train_files.txt \
    --val-split /tmp/test_val_files.txt \
    --vae-checkpoint "$VAE_CHECKPOINT" \
    --device $DEVICE \
    --epochs 2 \
    --batch-size 1 \
    --learning-rate 2e-5 \
    --predict-residual \
    --use-cfg \
    --gradient-checkpointing \
    --checkpoint-dir "$OUTPUT_DIR" \
    --save-every 1

echo ""
echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Latents: $LATENT_DIR"
echo "  Checkpoints: $OUTPUT_DIR"
echo ""
echo "If this worked, you can run full training with:"
echo "  ./run_custom_training.sh"
echo ""

# Cleanup
rm -f /tmp/test_train_files.txt /tmp/test_val_files.txt
