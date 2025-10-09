#!/bin/bash

# Quick test training on APE-data with MPS

APE_DIR="/Users/kuntalkokate/.cache/huggingface/hub/datasets--t2ance--APE-data/snapshots/5d20b5abd8504294335446f836fd0c61bf6f2d6a"

python scripts/finetune_diffusion_hf.py \
    --ape-cache-dir "$APE_DIR" \
    --subset APE \
    --device mps \
    --batch-size 1 \
    --max-samples 5 \
    --num-epochs 2 \
    --output-dir ./checkpoints/diffusion_hf_test
