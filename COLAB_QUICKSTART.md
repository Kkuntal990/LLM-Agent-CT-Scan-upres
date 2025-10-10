# Google Colab Quick Start

## Step-by-Step Instructions

### 1. Open Google Colab
Go to: https://colab.research.google.com/

### 2. Enable GPU
- Click **Runtime** → **Change runtime type**
- Select **Hardware accelerator**: **T4 GPU**
- Click **Save**

### 3. Run These Commands in Colab Cells

#### Cell 1: Clone Repository
```python
# First, you need to push your code to GitHub
# Then clone it here (replace with your actual repo URL)
!git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
%cd YOUR_REPO_NAME

# Verify files are present
!ls -la
```

**IMPORTANT**: You need to push your local code to GitHub first!

---

## If You Haven't Pushed to GitHub Yet

### Option A: Upload Entire Project as ZIP

#### On Your Local Machine:
```bash
cd "/Users/kuntalkokate/Desktop/LLM Agent - CT scan upres"
zip -r ct_upres_project.zip . -x "*.git*" -x "data/*" -x "checkpoints/*" -x "__pycache__/*"
```

#### In Colab:
```python
# Cell 1: Upload the ZIP file
from google.colab import files
uploaded = files.upload()  # Select ct_upres_project.zip

# Cell 2: Extract and setup
!unzip -q ct_upres_project.zip -d ct_upres
%cd ct_upres
!ls -la
```

### Option B: Push to GitHub (Recommended)

#### On Your Local Machine:
```bash
cd "/Users/kuntalkokate/Desktop/LLM Agent - CT scan upres"

# Check what will be committed (should NOT include data/)
git status

# Add all code files (data/ is already in .gitignore)
git add .
git commit -m "Add Colab training support"
git push origin main
```

Then use Cell 1 from "Step 3" above in Colab.

---

## After Your Code is in Colab

### Cell 2: Install Dependencies
```python
!pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
!pip install -q nibabel pydicom tqdm huggingface_hub SimpleITK scipy
```

### Cell 3: Authenticate with HuggingFace
```python
from huggingface_hub import login
login()
# Paste your HuggingFace token when prompted
```

### Cell 4: Verify GPU
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

### Cell 5: Make Script Executable
```python
!chmod +x run_custom_training_colab.sh
```

### Cell 6: Start Training
```python
!bash run_custom_training_colab.sh
```

---

## Quick Test (10 samples only)

Edit `run_custom_training_colab.sh` before running:
```bash
MAX_SAMPLES="10"  # Line 16
```

This will test the pipeline with only 10 samples (~1 hour total).

---

## Monitoring Training

### Check GPU Usage (Run in a separate cell)
```python
!nvidia-smi
```

### View Checkpoints
```python
!ls -lh checkpoints/vae_ape/
!ls -lh checkpoints/latent_diffusion_ape/
```

### Download Checkpoints
```python
from google.colab import files

# After training completes
!zip -r all_checkpoints.zip checkpoints/
files.download('all_checkpoints.zip')
```

---

## Expected Output

You should see:
```
=== Custom Implementation Training Pipeline (Google Colab) ===
Configuration:
  Device: cuda
  Dataset: APE
  Max samples: all
  Download data: true
  Data source: ./data/ape-raw

Step 0/4: Downloading APE-data from HuggingFace...
...
Step 1/4: Converting APE-data to NIfTI format...
...
Step 2/4: Fine-tuning VAE on CT data...
Epoch [1/10] - Train Loss: 0.0234, Val Loss: 0.0198
...
```

---

## Troubleshooting

**Error: "No such file or directory"**
→ You forgot to clone/upload your code. See options above.

**Error: "CUDA out of memory"**
→ Reduce batch size in `run_custom_training_colab.sh`:
```bash
BATCH_SIZE=2  # Line 22
```

**Error: "Access to this dataset is restricted"**
→ Re-authenticate with HuggingFace:
```python
from huggingface_hub import login
login(token='your_token_here')
```
