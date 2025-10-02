# CT Through-Plane Super-Resolution for macOS Apple Silicon

Learning-based pipeline for upsampling low through-plane resolution CT volumes to higher resolution while preserving Hounsfield Unit (HU) consistency and correct voxel spacing.

## Features

- **SimpleUNet3D** with z-axis-only upsampling for through-plane super-resolution (373K parameters)
- **HU-aware training** with L1, SSIM, and gradient losses
- **Patch-wise inference** with Gaussian blending to avoid seams and memory issues
- **LIDC-IDRI dataset support** with DICOM→NIfTI conversion and HU calibration
- **CPU training** (MPS not supported for 3D operations)
- **Comprehensive evaluation** with PSNR, SSIM, HU-MAE metrics
- **Real data training** on LIDC-IDRI thoracic CT scans

## Requirements

- macOS 14.0 or later
- Python 3.11
- PyTorch 2.2+
- Conda or pip

## Installation

### 1. Clone and setup environment

```bash
# Create conda environment
conda env create -f environment.yml
conda activate ct-superres-mps
```

### 2. Verify installation

```bash
# Check imports
python -c "from src.models.unet3d_simple import SimpleUNet3D; print('✓ Imports OK')"
```

## Quick Start

### 1. Prepare LIDC-IDRI Dataset

**Option A: Use NBIA Data Retriever (Recommended)**

1. Download NBIA Data Retriever from [TCIA](https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images)
2. Download desired LIDC-IDRI cases using the tool
3. Convert DICOM to NIfTI:

```bash
python scripts/convert_manifest_dicom.py
```

This will:
- Scan `./data/data/manifest-1600709154662/LIDC-IDRI` for DICOM files
- Convert to NIfTI format in `./data/lidc-processed/`
- Create train/val/test splits (60/20/20)

**Option B: Use Synthetic Data (Testing)**

Generate synthetic CT volumes for pipeline testing:

```bash
python scripts/create_sample_data.py
```

### 2. Train Model

```bash
python scripts/train.py \
  --data-dir ./data/lidc-processed \
  --train-split ./data/lidc-processed/train_files.txt \
  --val-split ./data/lidc-processed/val_files.txt \
  --device cpu \
  --epochs 50 \
  --batch-size 2 \
  --patch-size 16 64 64
```

**Training Parameters:**
- `--data-dir`: Directory with NIfTI files
- `--train-split`: Path to train split file
- `--val-split`: Path to validation split file
- `--device`: cpu (MPS not supported for 3D ops)
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
  --device cpu
```

**Before and After:**
- Input: `(100, 512, 512)` at spacing `(0.7, 0.7, 2.5)` mm
- Output: `(200, 512, 512)` at spacing `(0.7, 0.7, 1.25)` mm

## Dataset: LIDC-IDRI

The Lung Image Database Consortium (LIDC-IDRI) contains 1,018 thoracic CT cases with annotations.

- **Format**: DICOM series → NIfTI with HU calibration
- **HU Calibration**: `HU = pixel_value × RescaleSlope + RescaleIntercept`
- **Typical spacing**: ~0.6-0.9 mm in-plane, 1.25-2.5 mm through-plane
- **Tested with**: 5 real LIDC-IDRI patients (LIDC-IDRI-0002, 0007, 0009, 0010, 0012)

## Architecture

### SimpleUNet3D

```
Input: (B, 1, D, H, W) - LR volume
  ↓
Initial Conv3D → BN → ReLU
  ↓
Encoder (4 levels):
  - Conv3D → BN → ReLU
  - Strided Conv3D (downsampling)
  ↓
Decoder (4 levels):
  - Trilinear Interpolation (upsampling)
  - Concat skip connection
  - Conv3D → BN → ReLU
  ↓
Output Conv3D
  ↓
Final Z-axis 2× upsampling (trilinear)
  ↓
Output: (B, 1, D×2, H, W) - SR volume
```

**Key features:**
- **Z-only upsampling**: Preserves in-plane resolution
- **Strided convolutions**: MPS-compatible downsampling (no pooling)
- **Trilinear interpolation**: MPS-compatible upsampling (no transposed conv)
- **373,857 parameters** (base_channels=16, depth=4)

## Training Strategy

### Supervised Training

1. **Data generation**: Simulate LR volumes from HR using Gaussian slice profile convolution + decimation
2. **Losses**:
   - HU L1 loss (primary, weight=1.0)
   - SSIM loss (structural, weight=0.1)
   - Z-gradient loss (sharpness, weight=0.1)
3. **Body masking**: Exclude air regions from loss computation
4. **Optimizer**: Adam with ReduceLROnPlateau scheduler

### Slice Profile Simulation

The pipeline simulates thick-slice acquisition from thin-slice ground truth:
- Gaussian kernel blurring along z-axis
- Decimation to target spacing
- Realistic training pairs for supervised learning

File: `src/sim/slice_profile.py`

## Inference

### Patch-wise Processing

Large volumes are processed in overlapping tiles to:
1. **Avoid memory overflow** on CPU
2. **Eliminate seams** using Gaussian-weighted blending

**Recommended settings:**
- Patch size: `(16, 160, 160)`
- Overlap: `(8, 32, 32)`

**Example:**
```python
from src.infer.patch_infer import PatchInference
from src.models.unet3d_simple import create_simple_model

model = create_simple_model(device='cpu')
model.load_state_dict(torch.load('best_model.pth')['model_state_dict'])

inference = PatchInference(
    model=model,
    patch_size=(16, 160, 160),
    overlap=(8, 32, 32),
    device='cpu',
    upscale_factor=2
)

sr_volume = inference.infer(lr_volume_normalized, progress=True)
```

## Evaluation Metrics

Computed on body-masked regions:

- **PSNR**: Peak Signal-to-Noise Ratio (dB, higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)
- **HU-MAE**: Mean Absolute Error in Hounsfield Units (lower is better)

**Example:**
```python
from src.eval.metrics import evaluate_volume, print_metrics
from src.preprocessing.masking import create_body_mask

mask = create_body_mask(hr_volume, threshold_hu=-500.0)
metrics = evaluate_volume(pred_norm, target_norm, mask)
print_metrics(metrics)
```

## Training Results

### Configuration
- **Data**: 5 real LIDC-IDRI patients
- **Device**: CPU (MPS lacks 3D op support)
- **Epochs**: 10
- **Batch Size**: 2
- **Patch Size**: 16×64×64
- **Optimizer**: Adam (lr=1e-4)

### Performance
- **Best Validation Loss**: 0.0690 (epoch 8)
- **Training Time**: ~4.7s per epoch
- **Model**: `./checkpoints/best_model.pth`

## MPS Backend Notes

### Known Limitations

**PyTorch MPS does not support these 3D operations:**
- `max_pool3d`, `avg_pool3d`
- `conv_transpose3d`

**Solution:** Use CPU training with optimized architecture:
- Strided convolutions for downsampling
- Trilinear interpolation for upsampling

### Verification

```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
```

For 3D medical imaging, use `--device cpu` until PyTorch adds MPS support for these operations.

## Project Structure

```
.
├── src/
│   ├── io/              # DICOM and NIfTI I/O
│   ├── preprocessing/   # Orientation, spacing, masking, normalization
│   ├── sim/             # Slice profile simulation
│   ├── models/          # SimpleUNet3D architecture
│   ├── train/           # Training, losses, dataset
│   ├── infer/           # Patch-wise inference
│   ├── eval/            # Metrics (PSNR, SSIM, HU-MAE)
│   └── data/            # LIDC-IDRI preparation utilities
├── scripts/
│   ├── convert_manifest_dicom.py  # DICOM→NIfTI conversion
│   ├── create_sample_data.py      # Synthetic data generator
│   └── train.py                   # Training script
├── data/
│   ├── lidc-processed/            # Converted NIfTI files
│   └── unprocessed/               # Synthetic test data
├── checkpoints/
│   └── best_model.pth             # Trained model weights
├── demo.py                        # End-to-end SR demo
├── environment.yml                # Conda environment
└── README.md
```

## Common Issues & Solutions

### Issue 1: ModuleNotFoundError: No module named 'src'
**Solution:** Activate conda environment first:
```bash
conda activate ct-superres-mps
```

### Issue 2: MPS operations not supported
**Error:** `NotImplementedError: 'aten::max_pool3d' not implemented for MPS`

**Solution:** Use CPU device:
```bash
python scripts/train.py --device cpu ...
```

### Issue 3: RuntimeError: Numpy is not available
**Solution:** Reinstall numpy and PyTorch:
```bash
pip uninstall numpy torch
pip install numpy torch torchvision
```

### Issue 4: Channel mismatch in U-Net
**Error:** `RuntimeError: expected input[1, 384, ...] to have 256 channels`

**Solution:** Use SimpleUNet3D instead of ResidualUNet3D:
```python
from src.models.unet3d_simple import create_simple_model
model = create_simple_model(device='cpu')
```

### Issue 5: Out of memory
**Solution:** Reduce batch size or patch size:
```bash
python scripts/train.py --batch-size 1 --patch-size 16 64 64
```

## Development History

### Major Milestones
1. ✅ Initial repository setup with modular architecture
2. ✅ DICOM→NIfTI conversion pipeline
3. ✅ Slice profile simulation for training data
4. ✅ Original ResidualUNet3D architecture (had MPS issues)
5. ✅ SimpleUNet3D architecture (CPU-compatible)
6. ✅ Real LIDC-IDRI data integration (5 patients)
7. ✅ Successful training on real medical data
8. ✅ Best validation loss: 0.0690

### Challenges Overcome
- MPS backend incompatibility with 3D operations
- U-Net skip connection channel mismatches
- DICOM metadata handling and HU calibration
- Dataset format compatibility (2D slices vs 3D volumes)

## References

1. **LIDC-IDRI**: [The Cancer Imaging Archive](https://www.cancerimagingarchive.net/collection/lidc-idri/)
2. **PyTorch MPS**: [MPS Backend Documentation](https://pytorch.org/docs/stable/notes/mps.html)
3. Medical image super-resolution techniques

## License

This project is for research purposes. LIDC-IDRI dataset usage must comply with TCIA data usage policies.

## Contact

For issues and questions, please open a GitHub issue.
