#!/usr/bin/env python3
"""
Automated script to fix Google Colab dependency conflicts in all PyCaret notebooks
Author: Bala Anbalagan
Email: bala.anbalagan@sjsu.edu
"""

import json
import os
from pathlib import Path

# The new installation cell content (markdown)
NEW_MARKDOWN = """---

## Cell 1: Install and Import Required Libraries (Google Colab Compatible)

### What
We're installing PyCaret with compatible dependencies for Google Colab and importing all necessary Python libraries for our analysis.

### Why
Google Colab comes with pre-installed packages that can conflict with PyCaret's dependencies. This cell ensures compatibility by installing packages in the correct order to avoid runtime crashes.

### Technical Details
- Detect if running in Google Colab
- Install compatible versions of base packages (numpy, pandas, scipy, scikit-learn)
- Install PyCaret without forcing full dependency resolution
- Avoid version conflicts that cause runtime crashes

### Expected Output
Installation progress messages and a reminder to restart the runtime. After restart, the notebook will work smoothly without dependency errors.

### IMPORTANT
⚠️ After running this cell, you MUST restart the runtime:
- Click: **Runtime → Restart runtime** (or Ctrl+M .)
- After restart, skip this cell and run all other cells normally"""

# The new code cell content
NEW_CODE = """# ============================================================
# INSTALLATION CELL - Google Colab Compatible
# ============================================================
# This cell fixes dependency conflicts that cause runtime crashes

import sys

# Check if running in Colab
IN_COLAB = 'google.colab' in sys.modules

if IN_COLAB:
    print("=" * 60)
    print("🔧 Google Colab Detected")
    print("=" * 60)
    print("📦 Installing PyCaret with compatible dependencies...")
    print("⏳ This will take 2-3 minutes, please be patient...\\n")

    # Upgrade pip first
    !pip install -q --upgrade pip

    # Install compatible base packages FIRST (prevents conflicts)
    print("Step 1/3: Installing base packages with compatible versions...")
    !pip install -q --upgrade \\
        numpy>=1.23.0,<2.0.0 \\
        pandas>=2.0.0,<2.3.0 \\
        scipy>=1.10.0,<1.14.0 \\
        scikit-learn>=1.3.0,<1.6.0 \\
        matplotlib>=3.7.0,<3.9.0

    # Install PyCaret (will use already installed base packages)
    print("Step 2/3: Installing PyCaret...")
    !pip install -q pycaret

    # Install additional ML packages
    print("Step 3/3: Installing additional ML packages...")
    !pip install -q \\
        category-encoders \\
        lightgbm \\
        xgboost \\
        catboost \\
        optuna \\
        plotly \\
        kaleido

    print("\\n" + "=" * 60)
    print("✅ Installation Complete!")
    print("=" * 60)
    print("\\n⚠️  CRITICAL: You MUST restart the runtime now!")
    print("   👉 Click: Runtime → Restart runtime (or Ctrl+M .)\\n")
    print("🔄 After restart:")
    print("   1. Skip this installation cell")
    print("   2. Run all other cells normally")
    print("   3. Everything will work without crashes!\\n")
    print("=" * 60)

else:
    print("=" * 60)
    print("📍 Local Environment Detected")
    print("=" * 60)
    print("Installing standard PyCaret with full dependencies...\\n")
    !pip install pycaret[full]
    print("\\n✅ Installation complete!")
    print("=" * 60)

# Import libraries after installation
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

# Set visualization style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (10, 6)

print("\\n📚 Libraries imported successfully!")
print(f"   - Pandas version: {pd.__version__}")
print(f"   - NumPy version: {np.__version__}")"""

def fix_notebook(notebook_path):
    """
    Fix a single notebook by replacing the installation cell
    """
    print(f"\\nProcessing: {notebook_path}")

    # Read the notebook
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    # Find the installation cell (usually cell index 1 for markdown, 2 for code)
    # Look for the cell with "pip install pycaret"
    installation_code_index = None
    installation_markdown_index = None

    for idx, cell in enumerate(nb['cells']):
        if cell['cell_type'] == 'code':
            # Check if this is the installation cell
            source = ''.join(cell['source'])
            if 'pip install pycaret' in source or 'Import libraries' in source:
                installation_code_index = idx
                # The markdown cell should be just before it
                if idx > 0 and nb['cells'][idx-1]['cell_type'] == 'markdown':
                    installation_markdown_index = idx - 1
                break

    if installation_code_index is None:
        print(f"   ⚠️  Could not find installation cell, skipping...")
        return False

    # Replace the markdown cell
    if installation_markdown_index is not None:
        nb['cells'][installation_markdown_index]['source'] = NEW_MARKDOWN.split('\\n')
        print(f"   ✓ Updated markdown cell at index {installation_markdown_index}")

    # Replace the code cell
    nb['cells'][installation_code_index]['source'] = NEW_CODE.split('\\n')
    print(f"   ✓ Updated code cell at index {installation_code_index}")

    # Save the notebook
    with open(notebook_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)

    print(f"   ✅ Successfully updated {notebook_path.name}")
    return True

def main():
    """
    Main function to fix all notebooks
    """
    print("=" * 70)
    print("  PyCaret Notebooks - Google Colab Dependency Fix")
    print("  Fixing installation cells in all 6 notebooks")
    print("=" * 70)

    # Define notebook paths
    base_dir = Path(__file__).parent
    notebooks = [
        base_dir / "binary-classification" / "heart_disease_classification.ipynb",
        base_dir / "multiclass-classification" / "dry_bean_classification.ipynb",
        base_dir / "regression" / "insurance_cost_prediction.ipynb",
        base_dir / "clustering" / "wholesale_customer_segmentation.ipynb",
        base_dir / "anomaly-detection" / "network_intrusion_detection.ipynb",
        base_dir / "time-series" / "energy_consumption_forecasting.ipynb",
    ]

    # Fix each notebook
    success_count = 0
    for notebook_path in notebooks:
        if notebook_path.exists():
            if fix_notebook(notebook_path):
                success_count += 1
        else:
            print(f"\\n⚠️  Notebook not found: {notebook_path}")

    print("\\n" + "=" * 70)
    print(f"✅ Successfully updated {success_count}/{len(notebooks)} notebooks")
    print("=" * 70)

    print("\\nWhat was changed:")
    print("  - Updated installation cell with Colab-compatible dependencies")
    print("  - Added runtime restart instructions")
    print("  - Fixed numpy, pandas, scipy version conflicts")
    print("  - Added progress messages for better UX")

    print("\\nNext steps:")
    print("  1. Test one notebook in Google Colab")
    print("  2. Run the installation cell")
    print("  3. Restart runtime as instructed")
    print("  4. Run remaining cells")
    print("  5. If successful, commit changes to GitHub")

    print("\\n" + "=" * 70)

if __name__ == "__main__":
    main()
