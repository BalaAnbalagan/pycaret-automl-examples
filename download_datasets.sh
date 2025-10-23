#!/bin/bash
# Download All Kaggle Datasets for PyCaret AutoML Examples
# Author: Bala Anbalagan
# Email: bala.anbalagan@sjsu.edu

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Kaggle command path
KAGGLE_CMD="/Users/banbalagan/Library/Python/3.9/bin/kaggle"

echo -e "${BLUE}================================================${NC}"
echo -e "${BLUE}  PyCaret AutoML - Dataset Download Script${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""

# Check if kaggle is installed
if [ ! -f "$KAGGLE_CMD" ]; then
    echo -e "${RED}Error: Kaggle CLI not found at $KAGGLE_CMD${NC}"
    echo "Please install: python3 -m pip install kaggle --user"
    exit 1
fi

# Check if kaggle.json exists
if [ ! -f ~/.kaggle/kaggle.json ]; then
    echo -e "${RED}Error: Kaggle API credentials not found!${NC}"
    echo "Please follow KAGGLE_SETUP_INSTRUCTIONS.md to set up your API key"
    exit 1
fi

# Create datasets directory
mkdir -p datasets
cd datasets

echo -e "${GREEN}Starting dataset downloads...${NC}"
echo ""

# Dataset 1: Heart Disease
echo -e "${BLUE}[1/6] Downloading Heart Disease Dataset...${NC}"
mkdir -p binary-classification
$KAGGLE_CMD datasets download -d yasserh/heart-disease-dataset -p binary-classification --unzip
echo -e "${GREEN}✓ Heart Disease dataset downloaded (1,025 rows)${NC}"
echo ""

# Dataset 2: Dry Bean
echo -e "${BLUE}[2/6] Downloading Dry Bean Dataset...${NC}"
mkdir -p multiclass-classification
$KAGGLE_CMD datasets download -d sansuthi/dry-bean-dataset -p multiclass-classification --unzip
echo -e "${GREEN}✓ Dry Bean dataset downloaded (13,611 rows)${NC}"
echo ""

# Dataset 3: Medical Insurance
echo -e "${BLUE}[3/6] Downloading Medical Insurance Dataset...${NC}"
mkdir -p regression
$KAGGLE_CMD datasets download -d mirichoi0218/insurance -p regression --unzip
echo -e "${GREEN}✓ Medical Insurance dataset downloaded (1,338 rows)${NC}"
echo ""

# Dataset 4: Wholesale Customers
echo -e "${BLUE}[4/6] Downloading Wholesale Customers Dataset...${NC}"
mkdir -p clustering
$KAGGLE_CMD datasets download -d binovi/wholesale-customers-data-set -p clustering --unzip
echo -e "${GREEN}✓ Wholesale Customers dataset downloaded (440 rows)${NC}"
echo ""

# Dataset 5: Network Intrusion
echo -e "${BLUE}[5/6] Downloading Network Intrusion Dataset...${NC}"
mkdir -p anomaly-detection
$KAGGLE_CMD datasets download -d bcccdatasets/network-intrusion-detection -p anomaly-detection --unzip
echo -e "${GREEN}✓ Network Intrusion dataset downloaded${NC}"
echo ""

# Dataset 6: Energy Consumption
echo -e "${BLUE}[6/6] Downloading Energy Consumption Dataset...${NC}"
mkdir -p time-series
$KAGGLE_CMD datasets download -d atharvasoundankar/global-energy-consumption-2000-2024 -p time-series --unzip
echo -e "${GREEN}✓ Energy Consumption dataset downloaded${NC}"
echo ""

# Summary
echo -e "${BLUE}================================================${NC}"
echo -e "${GREEN}All datasets downloaded successfully!${NC}"
echo -e "${BLUE}================================================${NC}"
echo ""
echo "Dataset locations:"
echo "  1. Heart Disease: datasets/binary-classification/"
echo "  2. Dry Bean: datasets/multiclass-classification/"
echo "  3. Medical Insurance: datasets/regression/"
echo "  4. Wholesale Customers: datasets/clustering/"
echo "  5. Network Intrusion: datasets/anomaly-detection/"
echo "  6. Energy Consumption: datasets/time-series/"
echo ""
echo "Total storage used: ~3-4 MB"
echo ""
echo -e "${GREEN}You can now run the notebooks!${NC}"
echo "Example: jupyter notebook binary-classification/heart_disease_classification.ipynb"
