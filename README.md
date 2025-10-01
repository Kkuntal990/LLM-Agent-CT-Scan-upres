# CT Through-Plane Super-Resolution for macOS Apple Silicon

Learning-based pipeline for upsampling low through-plane resolution CT volumes to higher resolution while preserving Hounsfield Unit (HU) consistency and correct voxel spacing. Optimized for macOS 14+ with Apple Silicon MPS backend.

## Features

- **3D Residual U-Net** with z-axis-only upsampling for through-plane super-resolution
- **HU-aware training** with L1, SSIM, and gradient losses
- **Patch-wise inference** with Gaussian blending to avoid seams and memory issues
- **LIDC-IDRI dataset support** with DICOM→NIfTI conversion and HU calibration
- **Apple Silicon MPS optimization** for training and inference
- **Comprehensive evaluation** with PSNR, SSIM, HU-MAE, and optional LPIPS metrics
- **Optional self-supervised training** framework inspired by SR4ZCT

## Requirements

- macOS 14.0 or later
- Apple Silicon (M1, M2, M3, or later)
- Python 3.11
- PyTorch 2.2+ with MPS support

## Installation

### 1. Clone and setup environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ct-superres-mps

# Verify MPS availability
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

Expected output:
```
MPS available: True
```

### 2. Verify installation

```bash
# Run unit tests
pytest tests/ -v

# Check imports
python -c "from src.models.unet3d import ResidualUNet3D; print('✓ Imports OK')"
```

## Quick Start

### 1. Prepare LIDC-IDRI Dataset

Download LIDC-IDRI from [TCIA](https://www.cancerimagingarchive.net/collection/lidc-idri/) and convert to NIfTI:

```bash
python scripts/prepare_dataset.py \
  --dicom-root /path/to/LIDC-IDRI \
  --output-dir ./data/processed \
  --max-cases 100
```

This creates:
- `data/processed/*.nii.gz` - Converted volumes with HU calibration
- `data/processed/manifest.json` - Dataset metadata with SeriesInstanceUID
- `data/processed/splits/train.txt` - Training split
- `data/processed/splits/val.txt` - Validation split
- `data/processed/splits/test.txt` - Test split

### 2. Train Model

```bash
python scripts/train.py \
  --data-dir ./data/processed \
  --train-split ./data/processed/splits/train.txt \
  --val-split ./data/processed/splits/val.txt \
  --epochs 100 \
  --batch-size 4 \
  --upscale-factor 2 \
  --device mps
```

**Training Parameters:**
- `--patch-size D H W`: Training patch size (default: 32 128 128)
- `--lr`: Learning rate (default: 1e-4)
- `--checkpoint-dir`: Where to save models (default: ./checkpoints)
- `--log-dir`: TensorBoard logs (default: ./runs)

**Monitor training:**
```bash
tensorboard --logdir=./runs
```

### 3. Run Inference

```bash
python demo.py \
  input.nii.gz \
  output_sr.nii.gz \
  --checkpoint ./checkpoints/best_model.pth \
  --target-spacing 1.0 \
  --upscale-factor 2 \
  --device mps
```

**Before and After:**
- Input: `(100, 512, 512)` at spacing `(0.7, 0.7, 2.5)` mm
- Output: `(200, 512, 512)` at spacing `(0.7, 0.7, 1.25)` mm

## Dataset: LIDC-IDRI

The Lung Image Database Consortium (LIDC-IDRI) contains 1,018 thoracic CT cases with annotations. For this project:

- **Format**: DICOM series → NIfTI with HU calibration
- **HU Calibration**: `HU = pixel_value × RescaleSlope + RescaleIntercept`
- **Orientation**: Normalized to RAS (Right-Anterior-Superior)
- **Typical spacing**: ~0.6-0.8 mm in-plane, 1-5 mm through-plane
- **Recommended subset**: 100-200 cases for experimentation on M-series

## Architecture

### 3D Residual U-Net

```
Input: (B, 1, D, H, W) - LR volume
  ↓
Initial Conv3D → BN → ReLU
  ↓
Encoder (3 levels):
  - ResidualBlock3D
  - MaxPool3D(2,2,2)
  ↓
Bottleneck
  ↓
Decoder (3 levels):
  - ConvTranspose3D (z-only: kernel=(2,1,1))
  - Skip connection
  - ResidualBlock3D
  ↓
Output Conv3D
  ↓
Output: (B, 1, D×k, H, W) - SR volume
```

**Key features:**
- **Z-only upsampling**: Preserves in-plane resolution
- **Residual blocks**: Stabilize deep network training
- **~2M parameters** (base_channels=32, depth=3)

## Training Strategy

### Supervised Training

1. **Data generation**: Simulate LR volumes from HR using 1D slice profile convolution + decimation
2. **Losses**:
   - HU L1 loss (primary, weight=1.0)
   - SSIM loss (structural, weight=0.1)
   - Z-gradient loss (sharpness, weight=0.1)
3. **Body masking**: Exclude air regions from loss computation
4. **Augmentation**: Random horizontal/vertical flips

### Self-Supervised Option (SR4ZCT-style)

Framework provided in `src/train/selfsupervised.py` for off-axis training using in-plane views as supervision. Requires additional development for full SR4ZCT implementation.

## Inference

### Patch-wise Processing

Large volumes are processed in overlapping tiles to:
1. **Avoid memory overflow** on GPU/MPS
2. **Eliminate seams** using Gaussian-weighted blending

**Recommended settings:**
- Patch size: `(16, 160, 160)`
- Overlap: `(8, 32, 32)`
- Batch size: 1 (MPS limitation)

**Example:**
```python
from src.infer.patch_infer import PatchInference
from src.models.unet3d import create_model

model = create_model(upscale_factor=2, device='mps')
model.load_state_dict(torch.load('checkpoint.pth')['model_state_dict'])

inference = PatchInference(
    model=model,
    patch_size=(16, 160, 160),
    overlap=(8, 32, 32),
    device='mps',
    upscale_factor=2
)

sr_volume = inference.infer(lr_volume_normalized, progress=True)
```

## Evaluation Metrics

Computed on body-masked regions:

- **PSNR**: Peak Signal-to-Noise Ratio (dB, higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **HU-MAE**: Mean Absolute Error in Hounsfield Units (lower is better)
- **LPIPS** (optional): Perceptual similarity (0-1, lower is better)

**Example:**
```python
from src.eval.metrics import evaluate_volume, print_metrics
from src.preprocessing.masking import create_body_mask

mask = create_body_mask(hr_volume, threshold_hu=-500.0)
metrics = evaluate_volume(pred_norm, target_norm, mask, compute_lpips_metric=True)
print_metrics(metrics)
```

## MPS Backend Notes

### Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

# Test tensor operation
x = torch.randn(2, 3, 4, 4, 4).to('mps')
print(f"✓ MPS tensor created: {x.device}")
```

### Performance Tips

1. **Use FP32 for HU math**: AMP on MPS is experimental and may cause numerical issues
2. **Batch size 1-4**: Higher batches may exceed unified memory
3. **num_workers=0**: MPS doesn't support multi-process data loading
4. **Monitor memory**: Use Activity Monitor → GPU History

### Known Limitations

- **No pin_memory**: Set `pin_memory=False` in DataLoader
- **Limited AMP support**: Avoid mixed precision for now
- **Some ops unsupported**: Falls back to CPU if needed

## Project Structure

```
.
├── src/
│   ├── io/              # DICOM and NIfTI I/O
│   ├── preprocessing/   # Orientation, spacing, masking
│   ├── sim/             # Slice profile simulation
│   ├── models/          # 3D U-Net architecture
│   ├── train/           # Training, losses, dataset
│   ├── infer/           # Patch-wise inference
│   ├── eval/            # Metrics (PSNR, SSIM, HU-MAE)
│   └── data/            # LIDC-IDRI preparation
├── scripts/
│   ├── prepare_dataset.py  # DICOM→NIfTI conversion
│   └── train.py            # Training script
├── tests/
│   ├── test_hu_integrity.py  # HU preservation tests
│   ├── test_tiling.py        # Seamless blending tests
│   └── test_slice_profile.py # Simulation tests
├── demo.py              # End-to-end SR demo
├── environment.yml      # Conda environment
└── README.md
```

## Unit Tests

Run all tests:
```bash
pytest tests/ -v
```

Individual test suites:
```bash
pytest tests/test_hu_integrity.py -v    # HU preservation
pytest tests/test_tiling.py -v          # Gaussian blending
pytest tests/test_slice_profile.py -v   # Degradation simulation
```

## References

1. **SR4ZCT**: [Self-supervised Through-plane Resolution Enhancement for CT](https://arxiv.org/abs/2405.02515)
2. **LIDC-IDRI**: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/)
3. **PyTorch MPS**: [MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)

## Citation

If you use this code, please cite:

```bibtex
@article{shi2024sr4zct,
  title={SR4ZCT: Self-supervised Through-plane Resolution Enhancement for CT},
  author={Shi, Jiayang and others},
  journal={arXiv preprint arXiv:2405.02515},
  year={2024}
}
```

## License

This project is for research purposes. LIDC-IDRI dataset usage must comply with TCIA data usage policies.

## Troubleshooting

### MPS not available
```bash
# Check macOS version
sw_vers

# Check PyTorch installation
pip show torch
```

### Out of memory
Reduce batch size or patch size:
```bash
python scripts/train.py --batch-size 2 --patch-size 16 96 96
```

### Slow training
Check MPS is being used:
```python
# In training script
print(f"Model device: {next(model.parameters()).device}")  # Should show 'mps'
```

## Contact

For issues and questions, please open a GitHub issue.
