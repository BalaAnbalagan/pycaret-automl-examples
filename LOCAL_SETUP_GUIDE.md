# Local Setup Guide - Run PyCaret Notebooks on Your Mac

## Overview

This guide will help you set up and run all 6 PyCaret notebooks locally on your Mac without needing Google Colab.

**Advantages of running locally:**
- ✅ No internet required (after initial setup)
- ✅ Faster execution (uses your full CPU/GPU)
- ✅ No session timeouts
- ✅ Better control over environment
- ✅ Can save models and data permanently

---

## Prerequisites

- **macOS** (you have Mac mini)
- **Python 3.8 or higher** (you have Python 3.9)
- **~2 GB free disk space** (for packages + datasets)
- **8 GB RAM minimum** (16 GB recommended)

---

## Step 1: Set Up Python Virtual Environment

### Why Virtual Environment?
Isolates project dependencies from your system Python, preventing conflicts.

```bash
# Navigate to project directory
cd /Users/banbalagan/Projects/pycaret-automl-examples

# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# You should see (venv) in your terminal prompt
# Example: (venv) banbalagan@Mac-mini pycaret-automl-examples %
```

### To deactivate later (when done):
```bash
deactivate
```

---

## Step 2: Install Dependencies

### Option A: Install PyCaret with all dependencies (Recommended)

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Upgrade pip first
pip install --upgrade pip

# Install PyCaret with full functionality
pip install 'pycaret[full]'

# This installs:
# - PyCaret and all ML algorithms
# - Jupyter Notebook/Lab
# - All visualization libraries
# - Takes 5-10 minutes, installs ~500 MB of packages
```

### Option B: Minimal installation (faster, smaller)

```bash
pip install pycaret jupyter pandas numpy matplotlib seaborn scikit-learn
```

### Verify Installation

```bash
python3 -c "import pycaret; print(f'PyCaret version: {pycaret.__version__}')"
```

Expected output: `PyCaret version: 3.3.2` (or similar)

---

## Step 3: Install Jupyter Notebook/Lab

### Option A: Jupyter Notebook (Classic interface)

```bash
pip install notebook

# Launch Jupyter Notebook
jupyter notebook

# Browser will open automatically at http://localhost:8888
# Navigate to any notebook and open it
```

### Option B: JupyterLab (Modern interface, recommended)

```bash
pip install jupyterlab

# Launch JupyterLab
jupyter lab

# Browser opens at http://localhost:8888/lab
# Better UI, more features
```

### Option C: VS Code (Alternative)

If you prefer VS Code:
1. Install VS Code: https://code.visualstudio.com/
2. Install "Jupyter" extension in VS Code
3. Open project folder in VS Code
4. Click any .ipynb file
5. Select Python interpreter (choose the venv one)
6. Run cells directly in VS Code

---

## Step 4: Download Datasets

### Option A: Use the automated script (requires Kaggle API)

```bash
# First, set up Kaggle API (see KAGGLE_SETUP_INSTRUCTIONS.md)
# Then run:
./download_datasets.sh

# This downloads all 6 datasets automatically
```

### Option B: Manual download from Kaggle

1. **Binary Classification - Heart Disease**
   - URL: https://www.kaggle.com/datasets/yasserh/heart-disease-dataset
   - Download → Extract → Place `heart.csv` in `datasets/binary-classification/`

2. **Multiclass Classification - Dry Bean**
   - URL: https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset
   - Download → Extract → Place CSV in `datasets/multiclass-classification/`

3. **Regression - Medical Insurance**
   - URL: https://www.kaggle.com/datasets/mirichoi0218/insurance
   - Download → Extract → Place `insurance.csv` in `datasets/regression/`

4. **Clustering - Wholesale Customers**
   - URL: https://www.kaggle.com/binovi/wholesale-customers-data-set
   - Download → Extract → Place CSV in `datasets/clustering/`

5. **Anomaly Detection - Network Intrusion**
   - URL: https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection
   - Download → Extract → Place CSV in `datasets/anomaly-detection/`

6. **Time Series - Energy Consumption**
   - URL: https://www.kaggle.com/datasets/atharvasoundankar/global-energy-consumption-2000-2024
   - Download → Extract → Place CSV in `datasets/time-series/`

### Option C: Load directly from URLs (some notebooks support this)

Some notebooks have code to load data directly from public URLs. No download needed!

---

## Step 5: Update Notebook Data Paths (if using local datasets)

If you downloaded datasets locally, update the data loading cells:

```python
# OLD (loading from URL)
df = pd.read_csv('https://example.com/data.csv')

# NEW (loading from local file)
df = pd.read_csv('../datasets/binary-classification/heart.csv')
```

---

## Step 6: Run Notebooks

### Using Jupyter Notebook/Lab

```bash
# Activate virtual environment
cd /Users/banbalagan/Projects/pycaret-automl-examples
source venv/bin/activate

# Launch Jupyter
jupyter lab
# or
jupyter notebook

# In browser:
# 1. Navigate to any folder (e.g., binary-classification/)
# 2. Click on the .ipynb file
# 3. Run cells one by one (Shift + Enter)
# 4. Or run all cells: Cell → Run All
```

### Using VS Code

```bash
# Open VS Code
code /Users/banbalagan/Projects/pycaret-automl-examples

# Open any .ipynb file
# Select kernel: Choose 'venv' Python interpreter
# Click "Run All" or run cells individually
```

---

## Step 7: Modify Installation Cell in Notebooks

Since you're running locally (not Colab), modify the first code cell:

```python
# Comment out or skip the Colab installation code
# The installation section will detect you're not in Colab
# and won't try to install anything

# Just import the libraries:
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pycaret.classification import *  # or regression, clustering, etc.
```

Or simply skip the installation cell entirely since packages are already installed in your venv!

---

## Complete Local Setup Script

Create a file `setup_local.sh` and run it:

```bash
#!/bin/bash
# Complete local setup script

cd /Users/banbalagan/Projects/pycaret-automl-examples

echo "🚀 Setting up PyCaret notebooks locally..."

# Create virtual environment
echo "📦 Creating virtual environment..."
python3 -m venv venv

# Activate
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install PyCaret and Jupyter
echo "📚 Installing PyCaret and Jupyter Lab..."
pip install 'pycaret[full]' jupyterlab

# Verify
echo "✅ Verifying installation..."
python3 -c "import pycaret; print(f'PyCaret {pycaret.__version__} installed!')"

echo ""
echo "✅ Setup complete!"
echo ""
echo "Next steps:"
echo "1. Download datasets (see KAGGLE_SETUP_INSTRUCTIONS.md)"
echo "2. Launch Jupyter: jupyter lab"
echo "3. Open any notebook and run!"
echo ""
echo "To activate environment later:"
echo "  cd /Users/banbalagan/Projects/pycaret-automl-examples"
echo "  source venv/bin/activate"
```

**Run it:**
```bash
chmod +x setup_local.sh
./setup_local.sh
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pycaret'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Install PyCaret
pip install pycaret
```

### Issue: "Kernel died" or notebook crashes

**Solution:**
- Not enough RAM (close other apps)
- Try reducing dataset size for testing:
  ```python
  df = df.sample(n=1000, random_state=42)  # Use 1000 rows instead of all
  ```

### Issue: Slow training times

**Solution:**
- Reduce cross-validation folds:
  ```python
  setup(data=df, target='target', fold=5)  # Use 5 instead of 10
  ```
- Reduce number of models compared:
  ```python
  compare_models(n_select=3)  # Compare fewer models
  ```

### Issue: Jupyter not opening in browser

**Solution:**
```bash
# Get the URL manually
jupyter lab --no-browser

# Copy the URL shown (e.g., http://localhost:8888/lab?token=...)
# Paste in your browser
```

### Issue: "Port 8888 already in use"

**Solution:**
```bash
# Use a different port
jupyter lab --port 8889
```

---

## Performance Comparison: Local vs Colab

| Aspect | Local (Your Mac) | Google Colab |
|--------|------------------|--------------|
| **Speed** | Faster (dedicated CPU) | Slower (shared resources) |
| **RAM** | Your full RAM | Limited (12 GB free tier) |
| **Storage** | Permanent | Temporary (deleted after session) |
| **Internet** | Not needed after setup | Required always |
| **Timeout** | Never | 90 min idle, 12 hr max |
| **Cost** | Free (uses your machine) | Free tier available |
| **GPU** | Not available (Mac mini) | Free GPU (T4) |

**Recommendation:** Run locally unless you need GPU acceleration!

---

## Recommended Workflow

### For Development/Testing:
1. Work locally with Jupyter Lab
2. Faster iterations
3. Permanent storage

### For Sharing/Demos:
1. Push to GitHub
2. Open in Colab (via "Open in Colab" badge)
3. Others can run without local setup

### For Production:
1. Develop locally
2. Save models locally
3. Deploy as Flask API or Streamlit app

---

## Directory Structure After Setup

```
pycaret-automl-examples/
├── venv/                          # Virtual environment (created)
├── datasets/                      # Downloaded datasets (created)
│   ├── binary-classification/
│   ├── multiclass-classification/
│   ├── regression/
│   ├── clustering/
│   ├── anomaly-detection/
│   └── time-series/
├── binary-classification/
│   ├── heart_disease_classification.ipynb
│   ├── heart_disease_model.pkl   # Saved model (after running)
│   └── ...
├── setup_local.sh                 # Setup script
└── ...
```

---

## Quick Start Commands

```bash
# 1. Setup (first time only)
cd /Users/banbalagan/Projects/pycaret-automl-examples
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install 'pycaret[full]' jupyterlab

# 2. Download datasets (optional, use URL loading instead)
./download_datasets.sh

# 3. Launch Jupyter (every time you want to work)
source venv/bin/activate
jupyter lab

# 4. Open any notebook in browser and run!
```

---

## System Requirements Check

Run this to check if your Mac is ready:

```bash
# Check Python version (need 3.8+)
python3 --version

# Check available disk space (need ~2 GB)
df -h ~

# Check RAM (need 8 GB+)
sysctl hw.memsize | awk '{print $2/1024/1024/1024 " GB"}'

# Check CPU cores
sysctl hw.ncpu
```

---

## FAQ

**Q: Do I need GPU for PyCaret?**
A: No, PyCaret works great on CPU. Your Mac mini is fine!

**Q: How long does training take locally?**
A:
- Binary/Regression: 2-5 minutes
- Multiclass: 5-10 minutes
- Clustering: 1-3 minutes
- Time series: 10-15 minutes

**Q: Can I use both local and Colab?**
A: Yes! Develop locally, demo in Colab.

**Q: Will this affect my system Python?**
A: No, virtual environment keeps everything isolated.

**Q: Can I delete venv/ folder?**
A: Yes, but you'll need to reinstall. Keep it!

---

## Next Steps After Setup

1. ✅ Run one notebook end-to-end
2. ✅ Experiment with different parameters
3. ✅ Try your own datasets
4. ✅ Build custom models
5. ✅ Deploy as web app (Flask/Streamlit)

---

**Author:** Bala Anbalagan
**Email:** bala.anbalagan@sjsu.edu
**Last Updated:** January 2025

---

**Ready to run locally? Follow the Quick Start Commands above!** 🚀
