#!/bin/bash
# Local Setup Script for PyCaret AutoML Examples
# Author: Bala Anbalagan (bala.anbalagan@sjsu.edu)
# Description: Sets up complete local environment for running all 6 notebooks

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  PyCaret AutoML Examples - Local Setup${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# Get project directory
PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$PROJECT_DIR"

echo -e "${GREEN}📍 Project directory: ${PROJECT_DIR}${NC}"
echo ""

# Check Python version
echo -e "${BLUE}[1/6] Checking Python version...${NC}"
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

if [ "$PYTHON_MAJOR" -ge 3 ] && [ "$PYTHON_MINOR" -ge 8 ]; then
    echo -e "${GREEN}✓ Python $PYTHON_VERSION detected (requirement: 3.8+)${NC}"
else
    echo -e "${RED}✗ Python 3.8+ required, found $PYTHON_VERSION${NC}"
    exit 1
fi
echo ""

# Check available disk space
echo -e "${BLUE}[2/6] Checking disk space...${NC}"
AVAILABLE_SPACE=$(df -h . | awk 'NR==2 {print $4}')
echo -e "${GREEN}✓ Available disk space: $AVAILABLE_SPACE${NC}"
echo -e "${YELLOW}  (Need ~2 GB for packages + datasets)${NC}"
echo ""

# Create virtual environment
echo -e "${BLUE}[3/6] Creating virtual environment...${NC}"
if [ -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment already exists${NC}"
    read -p "Delete and recreate? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf venv
        python3 -m venv venv
        echo -e "${GREEN}✓ Virtual environment recreated${NC}"
    else
        echo -e "${YELLOW}→ Using existing virtual environment${NC}"
    fi
else
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${BLUE}[4/6] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${BLUE}[5/6] Upgrading pip...${NC}"
pip install --upgrade pip --quiet
PIP_VERSION=$(pip --version | awk '{print $2}')
echo -e "${GREEN}✓ pip upgraded to version $PIP_VERSION${NC}"
echo ""

# Install packages
echo -e "${BLUE}[6/6] Installing PyCaret and dependencies...${NC}"
echo -e "${YELLOW}⏳ This will take 5-10 minutes. Please be patient...${NC}"
echo ""

# Ask user for installation type
echo "Choose installation type:"
echo "  1) Full installation (recommended) - includes all features (~1 GB)"
echo "  2) Minimal installation (faster) - core features only (~300 MB)"
read -p "Enter choice (1 or 2): " -n 1 -r
echo ""

if [[ $REPLY == "1" ]]; then
    echo -e "${BLUE}Installing PyCaret with full dependencies...${NC}"
    pip install 'pycaret[full]' jupyterlab --quiet
    echo -e "${GREEN}✓ Full installation complete${NC}"
else
    echo -e "${BLUE}Installing minimal PyCaret...${NC}"
    pip install pycaret jupyterlab pandas numpy matplotlib seaborn scikit-learn --quiet
    echo -e "${GREEN}✓ Minimal installation complete${NC}"
fi
echo ""

# Verify installation
echo -e "${BLUE}Verifying installation...${NC}"
PYCARET_VERSION=$(python3 -c "import pycaret; print(pycaret.__version__)" 2>/dev/null)
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ PyCaret $PYCARET_VERSION installed successfully!${NC}"
else
    echo -e "${RED}✗ PyCaret installation failed${NC}"
    exit 1
fi
echo ""

# Create datasets directory
echo -e "${BLUE}Creating datasets directory...${NC}"
mkdir -p datasets/binary-classification
mkdir -p datasets/multiclass-classification
mkdir -p datasets/regression
mkdir -p datasets/clustering
mkdir -p datasets/anomaly-detection
mkdir -p datasets/time-series
echo -e "${GREEN}✓ Dataset directories created${NC}"
echo ""

# Success message
echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}✅ Setup Complete!${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

echo -e "${GREEN}Installation Summary:${NC}"
echo "  ✓ Python $PYTHON_VERSION"
echo "  ✓ PyCaret $PYCARET_VERSION"
echo "  ✓ JupyterLab installed"
echo "  ✓ Virtual environment: venv/"
echo "  ✓ Dataset directories created"
echo ""

echo -e "${BLUE}Next Steps:${NC}"
echo ""
echo -e "${YELLOW}1. Download Datasets (Optional):${NC}"
echo "   Option A: Use Kaggle API script"
echo "     ./download_datasets.sh"
echo ""
echo "   Option B: Manual download from Kaggle"
echo "     See KAGGLE_SETUP_INSTRUCTIONS.md"
echo ""
echo "   Option C: Skip - notebooks can load from URLs"
echo ""

echo -e "${YELLOW}2. Launch Jupyter Lab:${NC}"
echo "   source venv/bin/activate"
echo "   jupyter lab"
echo ""

echo -e "${YELLOW}3. Open any notebook and start coding!${NC}"
echo "   - binary-classification/heart_disease_classification.ipynb"
echo "   - multiclass-classification/dry_bean_classification.ipynb"
echo "   - regression/insurance_cost_prediction.ipynb"
echo "   - clustering/wholesale_customer_segmentation.ipynb"
echo "   - anomaly-detection/network_intrusion_detection.ipynb"
echo "   - time-series/energy_consumption_forecasting.ipynb"
echo ""

echo -e "${BLUE}================================================================${NC}"
echo -e "${GREEN}Useful Commands:${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""
echo "# Activate environment (run every time):"
echo "  source venv/bin/activate"
echo ""
echo "# Deactivate environment (when done):"
echo "  deactivate"
echo ""
echo "# Launch Jupyter Lab:"
echo "  jupyter lab"
echo ""
echo "# Launch Jupyter Notebook (classic):"
echo "  jupyter notebook"
echo ""
echo "# Update PyCaret:"
echo "  pip install --upgrade pycaret"
echo ""
echo "# List installed packages:"
echo "  pip list"
echo ""

echo -e "${GREEN}Happy coding! 🚀${NC}"
echo ""
