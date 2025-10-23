# Regression - Medical Insurance Cost Prediction

## Overview

This notebook demonstrates regression analysis using PyCaret to predict medical insurance costs (continuous dollar amounts) based on personal attributes like age, BMI, smoking status, and family size.

## Problem Statement

Predict annual medical insurance charges (in USD) for individuals based on 6 personal attributes to help insurance companies set appropriate premiums and help customers plan financially.

## Dataset

- **Source**: [Kaggle - Medical Insurance Cost Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Rows**: 1,338 individuals
- **Features**: 6 attributes (mixed numerical and categorical)
- **Target**: Insurance charges (continuous, USD)
- **License**: ODbL

### Features

| Feature | Type | Description |
|---------|------|-------------|
| age | Numerical | Age of beneficiary (18-64 years) |
| sex | Categorical | Gender (male/female) |
| bmi | Numerical | Body Mass Index (15-55) |
| children | Numerical | Number of dependents (0-5) |
| smoker | Categorical | Smoking status (yes/no) |
| region | Categorical | US region (4 regions) |

### Target Variable

- **charges**: Individual medical costs billed by insurance in USD
- Range: ~$1,000 - $65,000
- Distribution: Right-skewed (most low costs, few very high)

## What You'll Learn

1. **Regression Analysis**: Predicting continuous values (not categories)
2. **Regression Metrics**: R², RMSE, MAE, MAPE interpretation
3. **Residual Analysis**: Understanding prediction errors
4. **Target Transformation**: Handling skewed distributions
5. **Mixed Features**: Both numerical and categorical variables
6. **Feature Importance**: Identifying cost drivers
7. **Ensemble Regression**: Combining models for better predictions
8. **Real-world Application**: Insurance premium calculation

## PyCaret Features Demonstrated

- `setup()`: Configure regression environment with target transformation
- `compare_models()`: Compare 15+ regression algorithms
- `tune_model()`: Optimize for RMSE/R²
- `blend_models()`: Average predictions from multiple models
- `stack_models()`: Meta-learner for regression
- `plot_model()`: Residuals, prediction error, feature importance, learning curve
- `predict_model()`: Make cost predictions
- `finalize_model()`: Train on full dataset
- `save_model()` / `load_model()`: Model persistence

## Running the Notebook

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/pycaret-automl-examples/blob/main/regression/insurance_cost_prediction.ipynb)

### Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook insurance_cost_prediction.ipynb
```

## Expected Results

The model achieves:
- High R² (typically >0.85, meaning 85% variance explained)
- Low RMSE (predictions within ~$3,000-$5,000 on average)
- Low MAE (mean absolute error)
- Accurate predictions for most cases

## Business Value

- **Insurance Companies**: Accurate premium calculation and risk assessment
- **Customers**: Instant quotes and financial planning
- **Employers**: Budget employee health benefits
- **Policy Makers**: Data-driven healthcare decisions

## Key Insights

### Major Cost Drivers

1. **Smoking Status** (STRONGEST):
   - Smokers pay 2-3x more than non-smokers
   - Average difference: ~$20,000-$25,000 per year
   - Clear financial incentive for smoking cessation

2. **Age**:
   - Costs increase steadily with age
   - Acceleration after 50 years old
   - Non-linear relationship

3. **BMI**:
   - Higher BMI correlates with higher costs
   - More pronounced effect for smokers
   - Health risk indicator

4. **Region**:
   - Geographic variation in healthcare costs
   - Some regions 10-15% costlier

5. **Children**:
   - Minor impact on individual costs
   - More children = slightly higher costs

### Regression vs Classification

| Aspect | Classification | Regression |
|--------|---------------|------------|
| Output | Categories | Continuous numbers |
| Example | "Has disease: Yes/No" | "Cost: $12,345.67" |
| Metrics | Accuracy, AUC, F1 | RMSE, MAE, R² |
| Plot | Confusion Matrix | Residual Plot |

## Notebook Structure

Each code cell includes ELI20 explanations covering:
- **What**: Purpose of the code
- **Why**: Importance for regression
- **Technical Details**: How it works
- **Expected Output**: What results to expect

## Prerequisites

- Python 3.8+
- Understanding of regression concepts
- Familiarity with statistical metrics

## Time to Complete

Approximately 30-40 minutes

## Author

**Bala Anbalagan**

## Disclaimer

For educational purposes. Real insurance pricing involves actuarial science, regulatory compliance, and many additional factors beyond this model.
