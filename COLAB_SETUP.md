# Google Colab Setup Guide

This guide will help you run the custom implementation training pipeline on Google Colab.

## Prerequisites

1. Google account with access to Colab
2. HuggingFace account with access to APE-data dataset
3. HuggingFace token (for downloading gated datasets)

## Setup Steps

### 1. Start a Colab Session

Go to [Google Colab](https://colab.research.google.com/) and create a new notebook.

**IMPORTANT**: Enable GPU runtime
- Click `Runtime` → `Change runtime type`
- Select `Hardware accelerator`: **T4 GPU** (free tier) or **A100** (Colab Pro)
- Click `Save`

### 2. Clone Your Repository

```python
# In a Colab cell:
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO.git
%cd YOUR_REPO
```

### 3. Install Dependencies

```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q nibabel pydicom tqdm huggingface_hub SimpleITK scipy
```

### 4. Authenticate with HuggingFace

```python
from huggingface_hub import login
login()
# Enter your HuggingFace token when prompted
```

Alternatively, set token as environment variable:
```python
import os
os.environ['HF_TOKEN'] = 'your_token_here'
```

### 5. Verify GPU is Available

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### 6. Run Training Pipeline

```bash
!bash run_custom_training_colab.sh
```

## Key Differences from Local Version

| Setting | Local (Mac) | Colab |
|---------|-------------|-------|
| Device | `mps` | `cuda` |
| Batch Size | 1-2 | 4-8 |
| Cache Path | `~/.cache/huggingface/...` | `/root/.cache/huggingface/...` |
| GPU Memory | 20GB unified | 15GB VRAM (T4) / 40GB (A100) |
| Training Time | 10-12 hours | 3-4 hours (T4) / 1-2 hours (A100) |

## Configuration Options

Edit the configuration section in `run_custom_training_colab.sh`:

```bash
# For faster testing with fewer samples:
MAX_SAMPLES="10"  # Process only 10 samples

# For full training:
MAX_SAMPLES=""    # Process all 206 samples

# Adjust batch size based on GPU:
BATCH_SIZE=4      # T4 GPU (15GB)
BATCH_SIZE=8      # A100 GPU (40GB)

# Number of epochs:
VAE_EPOCHS=10
DIFFUSION_EPOCHS=20
```

## Monitoring Training

### View Real-time Logs

Training progress is printed to stdout. In Colab, you'll see:
```
Step 2/4: Fine-tuning VAE on CT data...
Epoch [1/10] - Train Loss: 0.0234, Val Loss: 0.0198
Epoch [2/10] - Train Loss: 0.0189, Val Loss: 0.0176
...
```

### Check GPU Memory Usage

```python
# Run in a separate cell while training:
!nvidia-smi
```

### TensorBoard (if enabled)

```python
%load_ext tensorboard
%tensorboard --logdir ./checkpoints
```

## Saving Checkpoints

Colab sessions timeout after 12 hours (free) or 24 hours (Pro). To avoid losing checkpoints:

### Option 1: Mount Google Drive

```python
from google.colab import drive
drive.mount('/content/drive')

# Then modify the script to save to Drive:
# --checkpoint-dir /content/drive/MyDrive/ct_checkpoints
```

### Option 2: Download Checkpoints Periodically

```python
from google.colab import files

# After VAE training completes:
!zip -r vae_checkpoint.zip ./checkpoints/vae_ape
files.download('vae_checkpoint.zip')

# After diffusion training completes:
!zip -r diffusion_checkpoint.zip ./checkpoints/latent_diffusion_ape
files.download('diffusion_checkpoint.zip')
```

### Option 3: Push to GitHub (careful with size)

```bash
# Only if checkpoints are small
!git lfs install
!git lfs track "*.pth"
!git add checkpoints/
!git commit -m "Add trained checkpoints"
!git push
```

## Expected Training Times

### T4 GPU (Free Tier)
- **VAE training**: ~2 hours (10 epochs, 206 samples)
- **Latent preparation**: ~30 minutes
- **Diffusion training**: ~3 hours (20 epochs)
- **Total**: ~5.5 hours

### A100 GPU (Colab Pro - $10/month)
- **VAE training**: ~45 minutes
- **Latent preparation**: ~10 minutes
- **Diffusion training**: ~1.5 hours
- **Total**: ~2.5 hours

## Troubleshooting

### "Runtime disconnected" Error

Colab free tier has usage limits. Solutions:
1. Use Colab Pro ($10/month) for longer sessions
2. Save checkpoints frequently
3. Train in stages (VAE → save → diffusion)

### "Out of Memory" Error

```python
# Reduce batch size in run_custom_training_colab.sh:
BATCH_SIZE=2  # or even 1

# Also reduce in step 4:
--batch-size 2
```

### "No module named 'nibabel'" Error

Re-run the dependencies installation:
```python
!pip install nibabel pydicom tqdm huggingface_hub SimpleITK scipy
```

### HuggingFace Authentication Failed

```python
# Check if token is set:
import os
print(os.environ.get('HF_TOKEN', 'Not set'))

# Re-authenticate:
from huggingface_hub import login
login(token='your_token_here')
```

## Running Inference After Training

Once training completes, test your model:

```python
# Download a test CT scan or use one from validation set
!python demo_latent_diffusion.py \
    data/ape-nifti/APE/test_sample.nii.gz \
    output_super_res.nii.gz \
    --vae-checkpoint ./checkpoints/vae_ape/best_vae.pth \
    --diffusion-checkpoint ./checkpoints/latent_diffusion_ape/best_latent_diffusion.pth \
    --device cuda
```

## Cost Comparison

| Option | GPU | Memory | Time | Cost |
|--------|-----|--------|------|------|
| Mac M3 | MPS | 20GB unified | ❌ OOM | Free |
| Colab Free | T4 | 15GB VRAM | 5.5 hours | Free |
| Colab Pro | A100 | 40GB VRAM | 2.5 hours | $10/month |
| Vast.ai | RTX 4090 | 24GB VRAM | 3 hours | ~$0.30/hour (~$1 total) |
| RunPod | A100 | 40GB VRAM | 2.5 hours | ~$1.00/hour (~$2.50 total) |

## Next Steps

After successful training on Colab:

1. Download checkpoints to local machine
2. Run inference on your own CT scans
3. Evaluate PSNR/SSIM metrics on test set
4. Fine-tune hyperparameters if needed

## Additional Resources

- [Google Colab FAQ](https://research.google.com/colaboratory/faq.html)
- [HuggingFace Authentication](https://huggingface.co/docs/huggingface_hub/quick-start#authentication)
- [Colab Pro Features](https://colab.research.google.com/signup)
