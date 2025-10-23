# Clustering - Wholesale Customer Segmentation

## Overview

This notebook demonstrates unsupervised learning using PyCaret's clustering capabilities to discover natural customer segments based on purchasing patterns. Unlike classification and regression, clustering has **no target variable** - we discover hidden patterns in the data.

## Problem Statement

Segment wholesale customers into meaningful groups based on their annual spending across 6 product categories to enable targeted marketing, optimized inventory, and personalized service strategies.

## Dataset

- **Source**: [Kaggle - Wholesale Customers Dataset](https://www.kaggle.com/binovi/wholesale-customers-data-set)
- **Original**: UCI Machine Learning Repository
- **Rows**: 440 wholesale customers
- **Features**: 8 attributes (2 categorical, 6 spending features)
- **Task**: Unsupervised clustering (no target variable)
- **License**: CC BY 4.0

### Features

| Feature | Type | Description |
|---------|------|-------------|
| Channel | Categorical | 1=Horeca (Hotel/Restaurant/Cafe), 2=Retail |
| Region | Categorical | Geographic region (3 regions) |
| Fresh | Numerical | Annual spending on fresh products |
| Milk | Numerical | Annual spending on milk products |
| Grocery | Numerical | Annual spending on grocery products |
| Frozen | Numerical | Annual spending on frozen products |
| Detergents_Paper | Numerical | Annual spending on detergents/paper |
| Delicassen | Numerical | Annual spending on delicatessen |

**Note**: All spending in monetary units (Mu).

## What You'll Learn

1. **Unsupervised Learning**: Discovering patterns without labels
2. **Clustering Algorithms**: KMeans, DBSCAN, Hierarchical clustering
3. **Optimal K Selection**: Elbow method and Silhouette analysis
4. **Cluster Evaluation**: Silhouette score, Calinski-Harabasz, Davies-Bouldin
5. **Cluster Profiling**: Understanding segment characteristics
6. **PCA Visualization**: Reducing dimensions for visualization
7. **Business Translation**: Converting clusters to actionable segments

## PyCaret Features Demonstrated

- `setup()`: Configure clustering environment with normalization
- `create_model()`: Create different clustering algorithms
- `assign_model()`: Assign customers to clusters
- `plot_model()`: Elbow, silhouette, cluster, distribution plots
- `save_model()` / `load_model()`: Model persistence
- **No compare_models()**: Unsupervised learning evaluates differently

## Running the Notebook

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/pycaret-automl-examples/blob/main/clustering/wholesale_customer_segmentation.ipynb)

### Local Environment

```bash
pip install -r requirements.txt
jupyter notebook wholesale_customer_segmentation.ipynb
```

## Expected Results

Discover 3-5 distinct customer segments with characteristics like:
- **Restaurant/Cafe Segment**: High fresh product spending
- **Retail Segment**: High grocery/milk/detergents spending
- **Specialized Segments**: Focused on specific categories

## Business Value

- **Marketing**: Segment-specific campaigns and messaging
- **Sales**: Targeted account management strategies
- **Inventory**: Optimize stock by segment demand
- **Pricing**: Segment-based pricing strategies
- **Service**: Tiered service levels per segment

## Key Insights

### Unsupervised vs Supervised

| Aspect | Supervised | Unsupervised |
|--------|-----------|--------------|
| Labels | Has target variable | No labels |
| Goal | Predict outcomes | Find patterns |
| Evaluation | Accuracy, RMSE | Silhouette, DB Index |
| Split | Train/Test | Use all data |

### Clustering Metrics

- **Silhouette Score** (0-1, higher better): Cluster separation quality
- **Calinski-Harabasz** (higher better): Between vs within cluster variance
- **Davies-Bouldin** (lower better): Average cluster similarity

### Customer Segments Discovered

Typical segments found:
1. **Horeca (Hotel/Restaurant/Cafe)**: High fresh, frozen spending
2. **Retail Stores**: High grocery, milk, detergents spending
3. **Specialty Customers**: Balanced or niche category focus

## Notebook Structure

Each cell includes ELI20 explanations:
- **What**: Purpose of the code
- **Why**: Importance for clustering
- **Technical Details**: How unsupervised learning works
- **Expected Output**: What to look for

## Prerequisites

- Python 3.8+
- Understanding of unsupervised learning
- Basic knowledge of distance metrics

## Time to Complete

Approximately 35-45 minutes

## Author

**Bala Anbalagan**

## Disclaimer

For educational purposes. Real customer segmentation should include additional data (demographics, temporal patterns, transaction history) and business validation.
