#!/bin/bash

# One-command setup for APE-data
# Downloads from HuggingFace and converts to NIfTI

set -e  # Exit on error

echo "==================================================================="
echo "APE-data Setup Script"
echo "==================================================================="
echo ""

# Configuration
DOWNLOAD_DIR="./data/ape-raw"
NIFTI_DIR="./data/ape-nifti"
SUBSET="APE"
MAX_SAMPLES=""  # Empty = all samples

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --max-samples)
            MAX_SAMPLES="--max-samples $2"
            shift 2
            ;;
        --subset)
            SUBSET="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [options]"
            echo ""
            echo "Options:"
            echo "  --max-samples N    Process only first N samples (for testing)"
            echo "  --subset NAME      Which subset to use (APE or 'non APE')"
            echo "  --help            Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                          # Download and convert all APE data"
            echo "  $0 --max-samples 10         # Process only 10 samples (quick test)"
            echo "  $0 --subset 'non APE'       # Use non-APE subset"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run with --help for usage information"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Download directory: $DOWNLOAD_DIR"
echo "  NIfTI directory: $NIFTI_DIR"
echo "  Subset: $SUBSET"
echo "  Max samples: ${MAX_SAMPLES:-all}"
echo ""

# Step 1: Check authentication
echo "Step 1/3: Checking HuggingFace authentication..."
if huggingface-cli whoami &> /dev/null; then
    echo "  ✓ Already logged in"
else
    echo "  ✗ Not logged in to HuggingFace"
    echo ""
    echo "You need to login to download APE-data (gated dataset)"
    echo ""
    read -p "Would you like to login now? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        huggingface-cli login
    else
        echo "Please login manually: huggingface-cli login"
        exit 1
    fi
fi

# Check if already downloaded
if [ -d "$DOWNLOAD_DIR/$SUBSET" ] && [ "$(ls -A $DOWNLOAD_DIR/$SUBSET)" ]; then
    echo ""
    echo "Dataset already downloaded at: $DOWNLOAD_DIR"
    read -p "Skip download and use existing data? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "  → Using existing download"
    else
        echo ""
        echo "Step 2/3: Downloading APE-data from HuggingFace..."
        python scripts/download_ape_data.py \
            --output-dir "$DOWNLOAD_DIR" \
            --subset "$SUBSET" \
            --force
    fi
else
    echo ""
    echo "Step 2/3: Downloading APE-data from HuggingFace..."
    python scripts/download_ape_data.py \
        --output-dir "$DOWNLOAD_DIR" \
        --subset "$SUBSET"
fi

# Step 3: Convert to NIfTI
echo ""
echo "Step 3/3: Converting DICOM to NIfTI..."
python scripts/convert_ape_data.py \
    --ape-cache-dir "$DOWNLOAD_DIR" \
    --subset "$SUBSET" \
    --output-dir "$NIFTI_DIR" \
    --skip-existing \
    $MAX_SAMPLES

echo ""
echo "==================================================================="
echo "Setup Complete!"
echo "==================================================================="
echo ""
echo "Dataset locations:"
echo "  Raw data (ZIP): $DOWNLOAD_DIR/$SUBSET/"
echo "  NIfTI data: $NIFTI_DIR/$SUBSET/"
echo ""
echo "Next steps:"
echo "  1. Review the train/val/test splits:"
echo "     cat $NIFTI_DIR/$SUBSET/train_files.txt"
echo ""
echo "  2. Start training:"
echo "     ./run_custom_training.sh"
echo ""
echo "  Or train HF Diffusers approach:"
echo "     python scripts/finetune_diffusion_hf.py \\"
echo "         --ape-cache-dir $DOWNLOAD_DIR \\"
echo "         --subset $SUBSET \\"
echo "         --device mps"
echo ""
echo "==================================================================="
