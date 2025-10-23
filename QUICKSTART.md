# Quick Start Guide - PyCaret Notebooks

## ✅ Good News: Your Python 3.9.6 is Perfect!

**PyCaret 3.3.2 supports Python 3.9, 3.10, and 3.11**

Your Mac has Python 3.9.6, which is **fully compatible**! ✅

---

## 🚀 Quick Start (3 Steps)

### Step 1: Activate Virtual Environment
```bash
cd /Users/banbalagan/Projects/pycaret-automl-examples
source venv/bin/activate
```

You'll see `(venv)` in your terminal prompt.

### Step 2: Launch Jupyter Lab
```bash
jupyter lab
```

Browser opens automatically at http://localhost:8888/lab

### Step 3: Run Any Notebook
- Navigate to any folder (e.g., `binary-classification/`)
- Click the notebook file
- **IMPORTANT:** Skip Cell 2 (installation cell) - already installed!
- Run all other cells normally

---

## ✅ Virtual Environment Already Created!

I've already created and set up your virtual environment with:
- ✅ Python 3.9.6
- ✅ PyCaret 3.3.2
- ✅ JupyterLab 4.4.10
- ✅ All dependencies (~500 MB)

**Location:** `/Users/banbalagan/Projects/pycaret-automl-examples/venv/`

---

## 📊 Python Version Compatibility

| PyCaret Version | Python Support | Your Version |
|----------------|----------------|--------------|
| PyCaret 3.3.2 | 3.9, 3.10, 3.11 | ✅ 3.9.6 |
| PyCaret 3.1+ | 3.9+ | ✅ Compatible |
| PyCaret 3.0 | 3.8+ | ✅ Compatible |

**You're all set!** No need to install older Python versions.

---

## 🔄 Daily Workflow

### When you want to work:
```bash
cd /Users/banbalagan/Projects/pycaret-automl-examples
source venv/bin/activate
jupyter lab
```

### When you're done:
Press `Ctrl+C` in terminal, then:
```bash
deactivate
```

---

## 📁 What's in the Virtual Environment

```bash
source venv/bin/activate
pip list
```

**Key packages installed:**
- pycaret==3.3.2
- jupyterlab==4.4.10
- scikit-learn==1.4.2
- pandas==2.1.4
- numpy==1.26.4
- matplotlib==3.7.5
- lightgbm==4.6.0
- plotly==6.3.1
- And 150+ more dependencies!

---

## 🎯 Test Your Setup

### Test 1: Check Python Version
```bash
source venv/bin/activate
python --version
# Output: Python 3.9.6 ✅
```

### Test 2: Verify PyCaret
```bash
source venv/bin/activate
python -c "import pycaret; print(pycaret.__version__)"
# Output: 3.3.2 ✅
```

### Test 3: Launch Jupyter
```bash
source venv/bin/activate
jupyter lab
# Browser should open ✅
```

---

## 📖 Running Your First Notebook

1. **Open Terminal**
   ```bash
   cd /Users/banbalagan/Projects/pycaret-automl-examples
   source venv/bin/activate
   jupyter lab
   ```

2. **In Jupyter Lab (browser)**
   - Click `binary-classification/` folder
   - Click `heart_disease_classification.ipynb`
   - Jupyter opens the notebook

3. **Run the Notebook**
   - **Cell 1 (Markdown):** Just info, skip it
   - **Cell 2 (Installation):** **SKIP THIS!** Already installed
   - **Cell 3+:** Run normally with Shift+Enter

4. **Expected Results**
   - No installation needed
   - No crashes
   - Training completes in 2-5 minutes
   - Models saved successfully

---

## 🆘 Troubleshooting

### Issue: "venv not found" or "No module named 'pycaret'"

**Solution:**
```bash
# Recreate virtual environment
cd /Users/banbalagan/Projects/pycaret-automl-examples
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install pycaret jupyterlab
```

### Issue: "jupyter: command not found"

**Solution:**
```bash
# Make sure venv is activated
source venv/bin/activate

# Install jupyter
pip install jupyterlab
```

### Issue: Notebook kernel keeps dying

**Solution:**
```bash
# Not enough RAM or too many processes
# Close other applications
# Or reduce dataset size in notebook:
df = df.sample(n=1000, random_state=42)  # Use 1000 rows for testing
```

### Issue: "Port 8888 already in use"

**Solution:**
```bash
# Use different port
jupyter lab --port 8889
```

---

## 💡 Pro Tips

### Tip 1: Use Aliases
Add to your `~/.zshrc` or `~/.bash_profile`:
```bash
alias pycaret-activate='cd /Users/banbalagan/Projects/pycaret-automl-examples && source venv/bin/activate'
alias pycaret-lab='pycaret-activate && jupyter lab'
```

Then just run:
```bash
pycaret-lab
```

### Tip 2: Skip Installation Cells
In each notebook, Cell 2 has installation code. You can:
- Skip it entirely (recommended)
- Or convert it to Markdown so it doesn't run

### Tip 3: Faster Training
```python
# In setup() cell
setup(
    data=df,
    target='target',
    fold=5,  # Instead of 10 - 2x faster
    n_jobs=-1  # Use all CPU cores
)
```

### Tip 4: Save Models Automatically
PyCaret saves models as `.pkl` files. They're saved in the notebook's directory.

---

## 📚 Notebook Execution Order

### Start Here (Easiest):
1. **Binary Classification** - Heart Disease (2-5 min)
2. **Regression** - Insurance Cost (2-5 min)
3. **Clustering** - Wholesale Customers (1-3 min)

### Then Try (Medium):
4. **Multiclass Classification** - Dry Bean (5-10 min)
5. **Anomaly Detection** - Network Intrusion (3-5 min)

### Finally (Longest):
6. **Time Series** - Energy Consumption (10-15 min)

---

## 🔧 Virtual Environment Commands

```bash
# Activate (do this every time)
source venv/bin/activate

# Deactivate (when done)
deactivate

# Check what's installed
pip list

# Update a package
pip install --upgrade pycaret

# Check Python version
python --version

# Remove virtual environment (careful!)
rm -rf venv
```

---

## ✨ Summary

### ✅ What's Ready:
- Python 3.9.6 (compatible with PyCaret 3.3.2)
- Virtual environment created at `venv/`
- PyCaret 3.3.2 installed with all dependencies
- JupyterLab 4.4.10 ready to launch
- All 6 notebooks ready to run

### 🚀 Next Step:
```bash
cd /Users/banbalagan/Projects/pycaret-automl-examples
source venv/bin/activate
jupyter lab
```

**That's it! You're ready to go!** 🎉

---

## 🎓 Learning Path

1. **Day 1:** Run Binary Classification notebook
2. **Day 2:** Run Regression notebook
3. **Day 3:** Run Clustering notebook
4. **Day 4:** Run Multiclass Classification notebook
5. **Day 5:** Run Anomaly Detection notebook
6. **Day 6:** Run Time Series notebook
7. **Day 7:** Experiment with your own datasets!

---

**Questions?** Check the full [LOCAL_SETUP_GUIDE.md](LOCAL_SETUP_GUIDE.md) for more details.

**Author:** Bala Anbalagan (bala.anbalagan@sjsu.edu)
**Date:** January 2025
