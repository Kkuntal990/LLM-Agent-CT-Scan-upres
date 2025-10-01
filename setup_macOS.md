# macOS Setup Guide for CT Super-Resolution

Complete setup instructions for macOS 14+ with Apple Silicon (M1/M2/M3).

## System Requirements

### Hardware
- **Mac**: Apple Silicon (M1, M2, M3, or later)
- **RAM**: 16 GB minimum, 32 GB recommended
- **Storage**: 50 GB free (for code, models, and subset of LIDC-IDRI)

### Software
- **macOS**: 14.0 (Sonoma) or later
- **Xcode Command Line Tools**: For compilation

## Step 1: Install Prerequisites

### 1.1 Install Xcode Command Line Tools

```bash
xcode-select --install
```

Verify:
```bash
xcode-select -p
# Should output: /Library/Developer/CommandLineTools
```

### 1.2 Install Conda/Mamba

**Option A: Miniforge (recommended for Apple Silicon)**
```bash
# Download Miniforge for arm64
curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-MacOSX-arm64.sh

# Install
bash Miniforge3-MacOSX-arm64.sh

# Follow prompts, then reload shell
source ~/.zshrc  # or ~/.bash_profile
```

**Option B: Anaconda**
Download from [anaconda.com](https://www.anaconda.com/download) (choose Apple Silicon version).

Verify:
```bash
conda --version
# Should output: conda 23.x.x or later
```

## Step 2: Clone Repository

```bash
cd ~/Desktop
git clone <repository-url> "LLM Agent - CT scan upres"
cd "LLM Agent - CT scan upres"
```

Or if working with existing directory:
```bash
cd "~/Desktop/LLM Agent - CT scan upres"
```

## Step 3: Create Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate environment
conda activate ct-superres-mps
```

Expected output:
```
Collecting package metadata (repodata.json): done
Solving environment: done
...
# To activate this environment, use:
#     conda activate ct-superres-mps
```

### Alternative: Manual Environment Setup

If `environment.yml` fails:

```bash
# Create environment
conda create -n ct-superres-mps python=3.11 -y
conda activate ct-superres-mps

# Install PyTorch with MPS support
conda install pytorch::pytorch torchvision -c pytorch -y

# Install scientific stack
conda install -c conda-forge numpy scipy nibabel pydicom scikit-image tqdm pyyaml pandas matplotlib tensorboard pytest -y

# Install pip packages
pip install lpips einops
```

## Step 4: Verify MPS Backend

Create test script `test_mps.py`:

```python
import torch
import sys

print("=" * 60)
print("PyTorch MPS Backend Verification")
print("=" * 60)

print(f"PyTorch version:     {torch.__version__}")
print(f"Python version:      {sys.version.split()[0]}")
print(f"MPS available:       {torch.backends.mps.is_available()}")
print(f"MPS built:           {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    print("\n✓ MPS is available!")

    # Test tensor creation
    try:
        x = torch.randn(2, 3, 4, 4, 4, device='mps')
        print(f"✓ Created MPS tensor: {x.device}")

        # Test operation
        y = x * 2 + 1
        print(f"✓ MPS operations work")

        # Test model
        model = torch.nn.Conv3d(3, 16, 3, padding=1).to('mps')
        out = model(x)
        print(f"✓ MPS Conv3D works: {out.shape}")

        print("\nMPS backend is fully functional! ✓")
    except Exception as e:
        print(f"\n✗ MPS test failed: {e}")
else:
    print("\n✗ MPS is NOT available!")
    print("Check:")
    print("  1. macOS version >= 14.0")
    print("  2. PyTorch version >= 2.2.0")
    print("  3. Apple Silicon Mac (M1/M2/M3)")

print("=" * 60)
```

Run verification:
```bash
python test_mps.py
```

**Expected output:**
```
============================================================
PyTorch MPS Backend Verification
============================================================
PyTorch version:     2.2.0
Python version:      3.11.x
MPS available:       True
MPS built:           True

✓ MPS is available!
✓ Created MPS tensor: mps:0
✓ MPS operations work
✓ MPS Conv3D works: torch.Size([2, 16, 4, 4, 4])

MPS backend is fully functional! ✓
============================================================
```

## Step 5: Run Tests

```bash
# Run all unit tests
pytest tests/ -v

# Test specific functionality
pytest tests/test_hu_integrity.py -v
pytest tests/test_tiling.py -v
```

Expected: All tests should pass.

## Step 6: Download LIDC-IDRI Dataset (Optional)

### 6.1 Register and Download

1. Visit [TCIA LIDC-IDRI](https://www.cancerimagingarchive.net/collection/lidc-idri/)
2. Register for free account
3. Download using NBIA Data Retriever (available for macOS)

### 6.2 Organize Data

```bash
# Expected structure after download
/path/to/LIDC-IDRI/
├── LIDC-IDRI-0001/
│   └── <StudyInstanceUID>/
│       └── <SeriesInstanceUID>/
│           ├── 1-001.dcm
│           ├── 1-002.dcm
│           └── ...
├── LIDC-IDRI-0002/
└── ...
```

### 6.3 Convert to NIfTI

```bash
python scripts/prepare_dataset.py \
  --dicom-root /path/to/LIDC-IDRI \
  --output-dir ./data/processed \
  --max-cases 50
```

**Storage estimate:**
- 50 cases: ~5 GB
- 100 cases: ~10 GB
- Full dataset (1018 cases): ~120 GB

## Step 7: Test Training (Smoke Test)

Create minimal test:

```bash
# Quick training test (1 epoch, small batch)
python scripts/train.py \
  --data-dir ./data/processed \
  --train-split ./data/processed/splits/train.txt \
  --val-split ./data/processed/splits/val.txt \
  --epochs 1 \
  --batch-size 2 \
  --device mps
```

Watch for:
- ✓ Model loads to MPS
- ✓ Data loads without errors
- ✓ Training step completes
- ✓ Validation runs
- ✓ Checkpoint saves

## Troubleshooting

### Issue: `conda: command not found`

**Solution:**
```bash
# Add conda to PATH
echo 'export PATH="$HOME/miniforge3/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Issue: MPS not available despite M1/M2 Mac

**Check:**
```bash
# macOS version
sw_vers
# ProductVersion should be >= 14.0

# PyTorch version
python -c "import torch; print(torch.__version__)"
# Should be >= 2.2.0
```

**Solution:** Update macOS or PyTorch:
```bash
conda install pytorch::pytorch>=2.2.0 -c pytorch
```

### Issue: Out of memory during training

**Solution:** Reduce batch size and patch size:
```bash
python scripts/train.py \
  --batch-size 1 \
  --patch-size 16 96 96 \
  ...
```

### Issue: `ImportError: cannot import name 'X'`

**Solution:** Reinstall dependencies:
```bash
conda env remove -n ct-superres-mps
conda env create -f environment.yml
```

### Issue: Slow data loading

**Solution:** Ensure `num_workers=0` in DataLoader (MPS requirement):
```python
# In dataset.py (already set)
train_loader = DataLoader(..., num_workers=0)
```

### Issue: Tests fail with "No module named 'src'"

**Solution:** Run from repository root:
```bash
cd "~/Desktop/LLM Agent - CT scan upres"
pytest tests/ -v
```

## Performance Monitoring

### Activity Monitor
1. Open Activity Monitor (Applications → Utilities)
2. Go to "GPU" tab
3. Watch "GPU" graph during training/inference

### TensorBoard
```bash
# Start TensorBoard
tensorboard --logdir=./runs --port=6006

# Open browser
open http://localhost:6006
```

### Memory Usage
```bash
# During training, check unified memory
vm_stat | grep "Pages active"
```

## Recommended Workflow

```bash
# 1. Activate environment
conda activate ct-superres-mps

# 2. Verify MPS
python test_mps.py

# 3. Prepare data (if needed)
python scripts/prepare_dataset.py --dicom-root /path/to/LIDC-IDRI --output-dir ./data/processed --max-cases 100

# 4. Train
python scripts/train.py --data-dir ./data/processed --train-split ./data/processed/splits/train.txt --val-split ./data/processed/splits/val.txt --epochs 100 --device mps

# 5. Monitor
tensorboard --logdir=./runs

# 6. Run inference
python demo.py input.nii.gz output_sr.nii.gz --checkpoint ./checkpoints/best_model.pth --device mps
```

## Next Steps

1. ✓ Environment set up
2. ✓ MPS verified
3. → Prepare LIDC-IDRI subset (50-100 cases)
4. → Train model (start with 10-20 epochs)
5. → Evaluate on test set
6. → Fine-tune hyperparameters

## Resources

- **PyTorch MPS Docs**: https://pytorch.org/docs/stable/notes/mps.html
- **Apple ML Docs**: https://developer.apple.com/metal/pytorch/
- **LIDC-IDRI**: https://www.cancerimagingarchive.net/collection/lidc-idri/

## Support

For issues specific to:
- **MPS backend**: Check PyTorch GitHub issues
- **LIDC-IDRI**: TCIA help desk
- **This code**: Open GitHub issue in repository
