# PyCaret AutoML Examples

> **📌 Important Note for Local Development**
>
> This repository is **optimized for local execution** with Python 3.9+ using virtual environments. Running locally is **recommended** because:
> - ✅ **No GPU Required**: All examples run efficiently on CPU (tested on Apple M4 and Intel processors)
> - ✅ **Version Control**: Avoid Python version conflicts between Google Colab and PyCaret by using isolated virtual environments (venv/conda)
> - ✅ **Reproducibility**: Consistent package versions (PyCaret 3.3.2 with Python 3.9-3.11) ensure stable execution
> - ✅ **Faster Setup**: Pre-configured notebooks with optimized model comparison (1-2 minutes vs 5-10 minutes)
> - ✅ **Local Datasets**: All datasets included in the repository - no internet dependency during execution
>
> **Google Colab** is supported but may require runtime restarts due to package conflicts. See [Local Setup Guide](QUICKSTART.md) for detailed instructions.

---

A comprehensive collection of six practical examples demonstrating PyCaret's powerful AutoML capabilities across different machine learning tasks using real-world Kaggle datasets.

## What is PyCaret?

PyCaret is an open-source, low-code machine learning library in Python that automates machine learning workflows. It is an end-to-end machine learning and model management tool that speeds up the experiment cycle exponentially and makes you more productive. PyCaret essentially wraps around several machine learning libraries (scikit-learn, XGBoost, LightGBM, CatBoost, and many more) and provides a unified interface for all of them.

### 🎥 Quick Introduction to PyCaret (30 seconds)

[![PyCaret Introduction](https://img.youtube.com/vi/FH0kJZJbses/0.jpg)](https://youtu.be/FH0kJZJbses)

*Click to watch a quick 30-second overview of PyCaret's capabilities*

### Key Features of PyCaret:
- **Low-Code**: Write less code, do more
- **AutoML**: Automatically compare multiple models
- **Hyperparameter Tuning**: Optimize model performance automatically
- **Ensemble Methods**: Create blended and stacked models
- **Model Interpretability**: Built-in SHAP and feature importance plots
- **Deployment Ready**: Easy model saving and loading

## Project Overview

This repository contains six comprehensive Jupyter notebooks, each tackling a different machine learning task using PyCaret's full AutoML capabilities. All datasets are sourced from Kaggle and are different from PyCaret's official tutorial examples to ensure originality and provide fresh learning perspectives.

## Examples Included

### 1. Binary Classification - Heart Disease Prediction

[![Binary Classification Video](https://img.youtube.com/vi/ACPdD7sVdbc/0.jpg)](https://youtu.be/ACPdD7sVdbc)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Heart Disease Dataset
- **Source**: [Kaggle - Heart Disease Dataset](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset)
- **Problem**: Predict whether a patient has heart disease based on medical attributes
- **Features**: 14 attributes including age, sex, chest pain type, blood pressure, cholesterol, etc.
- **Target**: Binary (0 = No disease, 1 = Disease)
- **Key Learnings**: Model comparison, hyperparameter tuning, ensemble methods (blending, stacking), threshold optimization, calibration
- **Notebook**: [binary-classification/heart_disease_classification.ipynb](binary-classification/heart_disease_classification.ipynb)

### 2. Multiclass Classification - Dry Bean Classification

[![Multiclass Classification Video](https://img.youtube.com/vi/6k5lvXAYQ20/0.jpg)](https://youtu.be/6k5lvXAYQ20)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Dry Bean Dataset (UCI)
- **Source**: [Kaggle - Dry Bean Dataset](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset)
- **Problem**: Classify dry beans into 7 different varieties based on morphological features
- **Features**: 16 morphological features (area, perimeter, major/minor axis lengths, shape factors)
- **Target**: 7 bean varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira)
- **Key Learnings**: Multi-class metrics, confusion matrix analysis, model interpretation, feature importance
- **Notebook**: [multiclass-classification/dry_bean_classification.ipynb](multiclass-classification/dry_bean_classification.ipynb)

### 3. Regression - Medical Insurance Cost Prediction

[![Regression Video](https://img.youtube.com/vi/m2bq8WsI97E/0.jpg)](https://youtu.be/m2bq8WsI97E)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Medical Insurance Cost Dataset
- **Source**: [Kaggle - Medical Insurance Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)
- **Problem**: Predict medical insurance costs based on personal attributes
- **Features**: Age, sex, BMI, number of children, smoker status, region
- **Target**: Insurance charges (continuous variable in USD)
- **Key Learnings**: Regression metrics (RMSE, MAE, R²), residual analysis, ensemble regression models, feature importance
- **Notebook**: [regression/insurance_cost_prediction.ipynb](regression/insurance_cost_prediction.ipynb)

### 4. Clustering - Wholesale Customer Segmentation

[![Clustering Video](https://img.youtube.com/vi/YtmirEPIfSQ/0.jpg)](https://youtu.be/YtmirEPIfSQ)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Wholesale Customers Dataset
- **Source**: [Kaggle - Wholesale Customers Dataset](https://www.kaggle.com/binovi/wholesale-customers-data-set)
- **Problem**: Segment wholesale customers based on annual spending patterns
- **Features**: Annual spending on Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicatessen products
- **Task**: Unsupervised customer segmentation
- **Key Learnings**: Clustering algorithms (KMeans, DBSCAN, Hierarchical), elbow method, silhouette analysis, cluster visualization
- **Notebook**: [clustering/wholesale_customer_segmentation.ipynb](clustering/wholesale_customer_segmentation.ipynb)

### 5. Anomaly Detection - Network Intrusion Detection

[![Anomaly Detection Video](https://img.youtube.com/vi/CTwEzOeetPg/0.jpg)](https://youtu.be/CTwEzOeetPg)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Network Intrusion Detection Dataset (2024)
- **Source**: [Kaggle - Network Intrusion Detection](https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection)
- **Problem**: Detect anomalous network behavior and potential intrusions
- **Features**: Network traffic features from BCCC-CIC-IDS-2017 dataset
- **Task**: Identify outliers and anomalies in network traffic
- **Key Learnings**: Anomaly detection algorithms (Isolation Forest, LOF, One-Class SVM), anomaly scoring, threshold selection
- **Notebook**: [anomaly-detection/network_intrusion_detection.ipynb](anomaly-detection/network_intrusion_detection.ipynb)

### 6. Time Series Forecasting - Energy Consumption Prediction

[![Time Series Video](https://img.youtube.com/vi/6LczHlVbROM/0.jpg)](https://youtu.be/6LczHlVbROM)

*🎥 Watch the video walkthrough of this notebook*

- **Dataset**: Global Energy Consumption (2000-2024)
- **Source**: [Kaggle - Global Energy Consumption](https://www.kaggle.com/datasets/atharvasoundankar/global-energy-consumption-2000-2024)
- **Problem**: Forecast future energy consumption based on historical data
- **Features**: Temporal energy consumption data across multiple countries
- **Task**: Time series forecasting
- **Key Learnings**: Time series models, trend and seasonality analysis, forecasting metrics (MAPE, RMSE), prediction intervals
- **Notebook**: [time-series/energy_consumption_forecasting.ipynb](time-series/energy_consumption_forecasting.ipynb)

## 📊 Results Summary

| Project | Best Model | Key Metric | Performance |
|---------|-----------|------------|-------------|
| **Binary Classification** | Gradient Boosting | Accuracy / AUC | 87.50% / 0.9338 |
| **Multiclass Classification** | Extra Trees | Accuracy (7 classes) | 93.06% |
| **Regression** | Gradient Boosting | R² / MAE | 0.8689 / $2,618 |
| **Clustering** | K-Means (4 clusters) | Silhouette Score | 0.4529 |
| **Anomaly Detection** | LOF | F1-Score | 0.183* |
| **Time Series** | Random Forest | MAE (vs baseline) | 2.52 MW (72% improvement) |

*Low F1 due to realistic imbalanced data (18.8% attack rate with 7 attack types)

**Key Takeaways:**
- Ensemble methods (Gradient Boosting, Extra Trees) consistently achieve best performance
- All supervised models: >85% accuracy/R² across diverse domains
- Feature engineering critical for time series (3x improvement over naive baseline)
- PyCaret AutoML workflow: Compare 10-15 models in minutes vs. days of manual coding

## Installation & Setup

### Prerequisites

Python 3.8 or higher is required. It's recommended to use a virtual environment.

### Install PyCaret

```bash
# Basic installation
pip install pycaret

# For full functionality (recommended)
pip install pycaret[full]
```

### Additional Dependencies

Each folder contains its own `requirements.txt` file with specific dependencies. Install them using:

```bash
cd <folder-name>
pip install -r requirements.txt
```

## Running the Notebooks

### Option 1: Google Colab (Recommended for Beginners)

1. Navigate to the specific notebook folder in this repository
2. Click on the notebook (.ipynb file)
3. Click the "Open in Colab" badge at the top of the notebook
4. Run cells sequentially using Shift+Enter
5. Install PyCaret in the first cell if needed:
   ```python
   !pip install pycaret
   ```

### Option 2: Local Jupyter Notebook/Lab

1. Clone this repository:
   ```bash
   git clone https://github.com/BalaAnbalagan/pycaret-automl-examples.git
   cd pycaret-automl-examples
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies for the specific example:
   ```bash
   cd binary-classification  # or any other folder
   pip install -r requirements.txt
   ```

4. Launch Jupyter:
   ```bash
   jupyter notebook
   # or
   jupyter lab
   ```

5. Open the notebook and run cells sequentially

### Option 3: VS Code with Jupyter Extension

1. Install the Jupyter extension in VS Code
2. Open the notebook file
3. Select your Python interpreter
4. Run cells using the play button or Shift+Enter

## Repository Structure

```
pycaret-automl-examples/
├── README.md                          # This file
├── .gitignore                         # Git ignore rules
├── binary-classification/
│   ├── heart_disease_classification.ipynb
│   ├── README.md
│   └── requirements.txt
├── multiclass-classification/
│   ├── dry_bean_classification.ipynb
│   ├── README.md
│   └── requirements.txt
├── regression/
│   ├── insurance_cost_prediction.ipynb
│   ├── README.md
│   └── requirements.txt
├── clustering/
│   ├── wholesale_customer_segmentation.ipynb
│   ├── README.md
│   └── requirements.txt
├── anomaly-detection/
│   ├── network_intrusion_detection.ipynb
│   ├── README.md
│   └── requirements.txt
└── time-series/
    ├── energy_consumption_forecasting.ipynb
    ├── README.md
    └── requirements.txt
```

## Datasets Used

| Example | Dataset | Source | Rows | Features | License |
|---------|---------|--------|------|----------|---------|
| Binary Classification | Heart Disease | [Kaggle](https://www.kaggle.com/datasets/yasserh/heart-disease-dataset) | 1,025 | 14 | CC0: Public Domain |
| Multiclass Classification | Dry Bean | [Kaggle](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset) | 13,611 | 16 | CC0: Public Domain |
| Regression | Medical Insurance | [Kaggle](https://www.kaggle.com/datasets/mirichoi0218/insurance) | 1,338 | 7 | ODbL |
| Clustering | Wholesale Customers | [Kaggle](https://www.kaggle.com/binovi/wholesale-customers-data-set) | 440 | 8 | CC BY 4.0 |
| Anomaly Detection | Network Intrusion | [Kaggle](https://www.kaggle.com/datasets/bcccdatasets/network-intrusion-detection) | Varies | Multiple | CC0: Public Domain |
| Time Series | Energy Consumption | [Kaggle](https://www.kaggle.com/datasets/atharvasoundankar/global-energy-consumption-2000-2024) | Varies | Temporal | CC0: Public Domain |

## Key PyCaret Features Demonstrated

### All Examples Include:
- **Data Preparation**: Automated preprocessing and feature engineering
- **Model Comparison**: Compare multiple algorithms automatically
- **Hyperparameter Tuning**: Optimize model performance
- **Model Evaluation**: Comprehensive metrics and visualizations
- **Model Saving**: Serialization for deployment

### Classification & Regression Specific:
- **Ensemble Methods**: Blending and stacking multiple models
- **Model Calibration**: Probability calibration for better predictions
- **Threshold Optimization**: Finding optimal decision thresholds
- **Feature Importance**: Understanding feature contributions

### Clustering Specific:
- **Multiple Algorithms**: KMeans, DBSCAN, Hierarchical clustering
- **Cluster Quality Metrics**: Silhouette score, Calinski-Harabasz index
- **Optimal Clusters**: Elbow method and silhouette analysis
- **Cluster Assignment**: Assigning data points to clusters

### Anomaly Detection Specific:
- **Multiple Detectors**: Isolation Forest, LOF, One-Class SVM
- **Anomaly Scoring**: Scoring anomalies for ranking
- **Visualization**: Plotting anomalies in feature space

### Time Series Specific:
- **Multiple Models**: ARIMA, ETS, Prophet, ML models
- **Decomposition**: Trend, seasonality, and residuals
- **Forecasting**: Future predictions with confidence intervals
- **Diagnostics**: Residual plots and statistical tests

## Notebook Structure

Each notebook follows a consistent structure:

1. **Introduction**: Problem context and business value
2. **Library Imports**: Explanation of each library's purpose
3. **Data Loading**: Dataset source and structure
4. **Exploratory Data Analysis**: Understanding the data through visualizations
5. **PyCaret Setup**: Environment configuration with parameter explanations
6. **Model Comparison**: Automated comparison of multiple algorithms
7. **Best Model Analysis**: Deep dive into top-performing models
8. **Hyperparameter Tuning**: Optimizing model performance
9. **Ensemble Methods**: Combining models for better results (classification/regression)
10. **Model Evaluation**: Comprehensive metrics and visualizations
11. **Predictions**: Making predictions on new data
12. **Model Saving**: Serializing models for deployment
13. **Conclusions**: Key takeaways and insights

### Documentation Format

Each code cell is preceded by a markdown cell explaining:
- **What**: Clear description of what the cell does
- **Why**: The purpose and importance of this step
- **Technical Details**: Parameters, functions, and how they work
- **Expected Output**: What to look for in the results

## Learning Outcomes

After completing these examples, you will learn:

1. **PyCaret Fundamentals**: How to set up and use PyCaret for various ML tasks
2. **AutoML Workflow**: Automated machine learning pipeline from data to deployment
3. **Model Comparison**: How to compare multiple algorithms efficiently
4. **Hyperparameter Tuning**: Optimizing models for better performance
5. **Ensemble Methods**: Combining models through blending and stacking
6. **Model Interpretation**: Understanding model decisions and feature importance
7. **Deployment Preparation**: Saving and loading models for production
8. **Domain Knowledge**: Practical applications across healthcare, agriculture, insurance, retail, cybersecurity, and energy sectors

## Troubleshooting

### Common Issues and Solutions

**Issue**: ImportError when importing PyCaret
```
Solution: Reinstall PyCaret with full dependencies
pip install --upgrade pycaret[full]
```

**Issue**: Memory errors during model comparison
```
Solution: Reduce the number of models to compare or use a subset of data
```

**Issue**: Slow training times
```
Solution:
- Reduce cross-validation folds in setup()
- Limit the number of iterations in tune_model()
- Use a smaller dataset for testing
```

**Issue**: Unable to download Kaggle datasets
```
Solution:
1. Create a Kaggle account
2. Download dataset manually from Kaggle
3. Upload to Google Colab or place in local directory
4. Update the data loading path in the notebook
```

**Issue**: Visualization not displaying
```
Solution:
- In Jupyter: Make sure matplotlib inline is enabled (%matplotlib inline)
- In Colab: Visualizations should work by default
- Try restarting the kernel
```

## Best Practices

1. **Start Small**: Begin with a subset of data to test the pipeline quickly
2. **Save Progress**: Save models and results frequently
3. **Document Changes**: If you modify notebooks, document your changes
4. **Version Control**: Use git to track your experiments
5. **Resource Management**: Monitor memory usage, especially with large datasets
6. **Reproducibility**: Set random seeds for reproducible results

## Contributing

This is an educational project. If you find issues or have suggestions:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## References

- [PyCaret Official Documentation](https://pycaret.gitbook.io/docs/)
- [PyCaret GitHub Repository](https://github.com/pycaret/pycaret)
- [Kaggle Datasets](https://www.kaggle.com/datasets)
- [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php)

## License

This project is licensed under the MIT License - see individual notebooks for dataset-specific licenses.

## Author

**Bala Anbalagan**
- GitHub: [@BalaAnbalagan](https://github.com/BalaAnbalagan)

## Acknowledgments

- Thanks to the PyCaret team for creating an amazing AutoML library
- Thanks to Kaggle and UCI ML Repository for providing quality datasets
- Thanks to the open-source community for various ML libraries used in this project

---

**Note**: All notebooks are designed to be educational and demonstrate PyCaret's capabilities. They are not production-ready solutions but serve as learning templates that can be adapted for real-world applications.

**Last Updated**: January 2025
