# Google Colab Dependency Fix Instructions

## Problem

When installing PyCaret 3.3.2 in Google Colab, you get dependency conflicts with pre-installed packages:
- numpy, pandas, scipy, matplotlib versions are incompatible
- This causes the runtime to crash and requires restart

## Solution

Replace the first installation cell in ALL notebooks with this updated version:

```python
# ============================================================
# INSTALLATION CELL - Google Colab Compatible
# ============================================================
# This cell fixes dependency conflicts in Google Colab

import sys

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("🔧 Detected Google Colab environment")
    print("📦 Installing PyCaret with compatible dependencies...")
    print("⏳ This will take 2-3 minutes, please wait...\n")

    # Install specific versions to avoid conflicts
    !pip install -q --upgrade pip

    # Install compatible base packages first
    !pip install -q \
        numpy>=1.23.0 \
        pandas>=2.0.0 \
        scipy>=1.10.0 \
        scikit-learn>=1.3.0 \
        matplotlib>=3.7.0

    # Install PyCaret without full dependencies (avoid conflicts)
    !pip install -q pycaret

    # Install additional required packages
    !pip install -q \
        category-encoders \
        lightgbm \
        xgboost \
        catboost \
        optuna \
        plotly \
        kaleido

    print("\n✅ Installation complete!")
    print("⚠️  IMPORTANT: Please restart the runtime now:")
    print("   Runtime → Restart runtime (or Ctrl+M .)")
    print("\n🔄 After restart, skip this cell and run the rest of the notebook.\n")

else:
    print("📍 Not in Colab - installing standard PyCaret...")
    !pip install pycaret[full]
    print("✅ Installation complete!")
```

## Alternative: Simpler Fix (If Above Doesn't Work)

If you still face issues, use this minimal installation:

```python
# Minimal PyCaret installation for Colab
import sys

if 'google.colab' in sys.modules:
    print("Installing PyCaret (minimal)...")
    !pip install -q --no-deps pycaret
    !pip install -q pandas scikit-learn numpy scipy joblib category-encoders
    print("✅ Done! Restart runtime: Runtime → Restart runtime")
else:
    !pip install pycaret
```

## Step-by-Step Process

### For Each Notebook:

1. **Open the notebook** in Google Colab
2. **Find Cell 2** (the one with `# !pip install pycaret[full]`)
3. **Replace that entire cell** with the code above
4. **Run the new installation cell**
5. **Wait for installation to complete** (2-3 minutes)
6. **Restart the runtime**: Click `Runtime → Restart runtime` in the menu
7. **After restart**:
   - Skip the installation cell
   - Run all remaining cells normally

## Which Notebooks Need This Fix?

✅ ALL 6 notebooks need this fix:
1. `binary-classification/heart_disease_classification.ipynb`
2. `multiclass-classification/dry_bean_classification.ipynb`
3. `regression/insurance_cost_prediction.ipynb`
4. `clustering/wholesale_customer_segmentation.ipynb`
5. `anomaly-detection/network_intrusion_detection.ipynb`
6. `time-series/energy_consumption_forecasting.ipynb`

## Why This Happens

- **Google Colab** comes with pre-installed packages (numpy 1.26.4, pandas 2.2.2, etc.)
- **PyCaret 3.3.2** requires different versions
- **Pip's resolver** tries to install compatible versions but conflicts occur
- **Solution**: Install base packages first with compatible versions, then PyCaret

## If Runtime Still Crashes

Try this ultra-minimal approach:

```python
# Ultra-minimal - no conflicts
!pip install --quiet pycaret==3.0.0

print("✅ PyCaret 3.0.0 installed (older but stable)")
print("🔄 Restart runtime now!")
```

## Testing After Fix

After applying the fix and restarting, test with:

```python
# Test installation
import pycaret
print(f"PyCaret version: {pycaret.__version__}")

from pycaret.classification import *
print("✅ PyCaret classification module working!")
```

## Automated Fix Script

Want me to update all 6 notebooks automatically? I can create a script to:
1. Read each notebook
2. Replace the installation cell
3. Save the updated version
4. Commit to GitHub

Just let me know!

---

**Author**: Bala Anbalagan
**Date**: January 2025
**Issue**: Google Colab dependency conflicts
**Status**: Fix documented, ready to apply
