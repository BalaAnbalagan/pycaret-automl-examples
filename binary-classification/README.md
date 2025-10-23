# Binary Classification - Heart Disease Prediction

## Overview

This notebook demonstrates binary classification using PyCaret's AutoML capabilities to predict heart disease in patients based on medical attributes.

## Problem Statement

Predict whether a patient has heart disease (binary classification: 0 = No Disease, 1 = Disease) using 13 medical features.

## Dataset

- **Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset)
- **Rows**: 1,025 patients
- **Features**: 13 medical attributes + 1 target variable
- **License**: CC0: Public Domain

### Features

| Feature | Description | Type |
|---------|-------------|------|
| age | Age of patient | Numerical |
| sex | Sex (1=male, 0=female) | Categorical |
| cp | Chest pain type (0-3) | Categorical |
| trestbps | Resting blood pressure (mm Hg) | Numerical |
| chol | Serum cholesterol (mg/dl) | Numerical |
| fbs | Fasting blood sugar > 120 mg/dl | Binary |
| restecg | Resting ECG results (0-2) | Categorical |
| thalach | Maximum heart rate achieved | Numerical |
| exang | Exercise induced angina | Binary |
| oldpeak | ST depression induced by exercise | Numerical |
| slope | Slope of peak exercise ST segment | Categorical |
| ca | Number of major vessels (0-3) | Numerical |
| thal | Thalassemia (0-3) | Categorical |
| **target** | Heart disease (0=no, 1=yes) | **Target** |

## What You'll Learn

1. **PyCaret Setup**: Initialize classification environment
2. **Model Comparison**: Automatically compare 15+ algorithms
3. **Hyperparameter Tuning**: Optimize model performance
4. **Ensemble Methods**:
   - Blending (soft voting)
   - Stacking (meta-learning)
5. **Model Calibration**: Improve probability estimates
6. **Evaluation Metrics**: Accuracy, AUC, Precision, Recall, F1
7. **Feature Importance**: Understand key predictive factors
8. **Model Deployment**: Save and load models

## PyCaret Features Demonstrated

- `setup()`: Initialize classification environment
- `compare_models()`: Compare multiple algorithms automatically
- `tune_model()`: Hyperparameter optimization
- `blend_models()`: Create blended ensemble
- `stack_models()`: Create stacked ensemble
- `calibrate_model()`: Calibrate probability predictions
- `plot_model()`: Visualization (AUC, confusion matrix, feature importance)
- `finalize_model()`: Train on full dataset
- `predict_model()`: Make predictions
- `save_model()` / `load_model()`: Model persistence

## Running the Notebook

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/pycaret-automl-examples/blob/main/binary-classification/heart_disease_classification.ipynb)

### Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook heart_disease_classification.ipynb
```

## Expected Results

The final model achieves:
- High accuracy in predicting heart disease
- Strong AUC indicating good discrimination
- Balanced precision and recall
- Calibrated probabilities for risk assessment

## Business Value

- **Healthcare Providers**: Early identification of high-risk patients
- **Patients**: Early detection and preventive care
- **Insurance**: Better risk assessment and premium calculation

## Key Insights

The model identifies the most important factors contributing to heart disease, helping healthcare providers:
- Focus on critical risk factors
- Make data-driven decisions
- Allocate resources effectively

## Notebook Structure

Each code cell includes ELI20 (Explain Like I'm 20) explanations covering:
- **What**: What the cell does
- **Why**: Purpose and importance
- **Technical Details**: Parameters and functions
- **Expected Output**: What to look for

## Prerequisites

- Python 3.8+
- Basic understanding of classification problems
- Familiarity with pandas and numpy

## Time to Complete

Approximately 30-45 minutes

## Author

**Bala Anbalagan**

## Disclaimer

This model is for educational purposes only. Always consult qualified healthcare professionals for medical decisions.
