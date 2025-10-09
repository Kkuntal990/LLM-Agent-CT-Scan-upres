#!/bin/bash

# Download pre-trained weights for Medical Latent Diffusion Model
#
# This script downloads:
# 1. Microsoft MRI AutoencoderKL (for VAE initialization)
# 2. ResShift architecture reference
# 3. DiffBIR for ControlNet reference

set -e  # Exit on error

echo "========================================================================"
echo "Downloading Pre-trained Weights for Medical Latent Diffusion"
echo "========================================================================"

# Create directories
PRETRAINED_DIR="./pretrained_weights"
mkdir -p "$PRETRAINED_DIR"
mkdir -p "$PRETRAINED_DIR/microsoft-mri-vae"
mkdir -p "$PRETRAINED_DIR/resshift"
mkdir -p "$PRETRAINED_DIR/diffbir"

echo ""
echo "Created directories:"
echo "  $PRETRAINED_DIR/"
echo ""

# ========================================================================
# 1. Microsoft MRI Autoencoder (Hugging Face)
# ========================================================================
echo "--------------------------------------------------------------------"
echo "1. Downloading Microsoft MRI AutoencoderKL..."
echo "--------------------------------------------------------------------"

# Check if huggingface-cli is installed
if ! command -v huggingface-cli &> /dev/null; then
    echo "⚠ huggingface-cli not found. Installing huggingface_hub..."
    pip install -q huggingface_hub
fi

# Download using Python (more reliable than huggingface-cli)
python -c "
from huggingface_hub import hf_hub_download
import os

model_id = 'microsoft/mri-autoencoder-v0.1'
cache_dir = '$PRETRAINED_DIR/microsoft-mri-vae'

print(f'Downloading {model_id}...')
try:
    # Download config and weights
    config_path = hf_hub_download(
        repo_id=model_id,
        filename='config.json',
        cache_dir=cache_dir
    )
    weights_path = hf_hub_download(
        repo_id=model_id,
        filename='diffusion_pytorch_model.safetensors',
        cache_dir=cache_dir
    )
    print(f'✓ Downloaded to {cache_dir}')
    print(f'  Config: {config_path}')
    print(f'  Weights: {weights_path}')
except Exception as e:
    print(f'⚠ Could not download {model_id}: {e}')
    print('  You can download manually from: https://huggingface.co/{model_id}')
    print('  Note: This is optional for initialization. Training will work without it.')
"

echo ""

# ========================================================================
# 2. ResShift (GitHub)
# ========================================================================
echo "--------------------------------------------------------------------"
echo "2. Downloading ResShift architecture reference..."
echo "--------------------------------------------------------------------"

if [ ! -d "$PRETRAINED_DIR/resshift/.git" ]; then
    echo "Cloning ResShift repository..."
    git clone https://github.com/zsyOAOA/ResShift.git "$PRETRAINED_DIR/resshift" || {
        echo "⚠ Could not clone ResShift repository"
        echo "  You can clone manually: git clone https://github.com/zsyOAOA/ResShift.git"
    }
    echo "✓ ResShift cloned"
else
    echo "✓ ResShift already exists"
fi

echo ""

# ========================================================================
# 3. DiffBIR (GitHub)
# ========================================================================
echo "--------------------------------------------------------------------"
echo "3. Downloading DiffBIR ControlNet reference..."
echo "--------------------------------------------------------------------"

if [ ! -d "$PRETRAINED_DIR/diffbir/.git" ]; then
    echo "Cloning DiffBIR repository..."
    git clone https://github.com/XPixelGroup/DiffBIR.git "$PRETRAINED_DIR/diffbir" || {
        echo "⚠ Could not clone DiffBIR repository"
        echo "  You can clone manually: git clone https://github.com/XPixelGroup/DiffBIR.git"
    }
    echo "✓ DiffBIR cloned"
else
    echo "✓ DiffBIR already exists"
fi

# Optionally download DiffBIR pre-trained weights (large file)
echo ""
echo "Do you want to download DiffBIR pre-trained weights (~1.5GB)? (y/N)"
read -r response

if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
    echo "Downloading DiffBIR weights..."
    cd "$PRETRAINED_DIR/diffbir"

    if [ ! -f "general_full_v1.pth" ]; then
        wget -q --show-progress https://github.com/XPixelGroup/DiffBIR/releases/download/v1.0/general_full_v1.pth || {
            echo "⚠ Could not download DiffBIR weights"
            echo "  You can download manually from: https://github.com/XPixelGroup/DiffBIR/releases"
        }
        echo "✓ DiffBIR weights downloaded"
    else
        echo "✓ DiffBIR weights already exist"
    fi

    cd - > /dev/null
else
    echo "Skipping DiffBIR weights download"
fi

echo ""

# ========================================================================
# Summary
# ========================================================================
echo "========================================================================"
echo "Download Summary"
echo "========================================================================"
echo ""
echo "Pre-trained weights location: $PRETRAINED_DIR/"
echo ""
echo "Files:"
echo "  1. Microsoft MRI VAE: $PRETRAINED_DIR/microsoft-mri-vae/"
echo "  2. ResShift reference: $PRETRAINED_DIR/resshift/"
echo "  3. DiffBIR reference: $PRETRAINED_DIR/diffbir/"
echo ""
echo "========================================================================"
echo "Next Steps"
echo "========================================================================"
echo ""
echo "1. Fine-tune VAE on CT data:"
echo "   python scripts/finetune_vae.py \\"
echo "     --data-dir ./data/lidc-processed \\"
echo "     --train-split ./data/lidc-processed/train_files.txt \\"
echo "     --val-split ./data/lidc-processed/val_files.txt \\"
echo "     --device cuda \\"
echo "     --epochs 20"
echo ""
echo "2. Prepare latent dataset:"
echo "   python scripts/prepare_latent_dataset.py \\"
echo "     --data-dir ./data/lidc-processed \\"
echo "     --output-dir ./data/lidc-latents \\"
echo "     --file-list ./data/lidc-processed/all_files.txt \\"
echo "     --vae-checkpoint ./checkpoints/vae/best_vae.pth \\"
echo "     --device cuda"
echo ""
echo "3. Train latent diffusion:"
echo "   python scripts/train_latent_diffusion.py \\"
echo "     --latent-dir ./data/lidc-latents \\"
echo "     --train-split ./data/lidc-processed/train_files.txt \\"
echo "     --val-split ./data/lidc-processed/val_files.txt \\"
echo "     --vae-checkpoint ./checkpoints/vae/best_vae.pth \\"
echo "     --device cuda \\"
echo "     --epochs 50"
echo ""
echo "4. Run inference:"
echo "   python demo_latent_diffusion.py \\"
echo "     input_lr.nii.gz output_sr.nii.gz \\"
echo "     --vae-checkpoint ./checkpoints/vae/best_vae.pth \\"
echo "     --diffusion-checkpoint ./checkpoints/latent_diffusion/best_latent_diffusion.pth \\"
echo "     --device cuda"
echo ""
echo "========================================================================"
echo "Download complete!"
echo "========================================================================"
