# Multiclass Classification - Dry Bean Classification

## Overview

This notebook demonstrates multiclass classification using PyCaret to automatically classify dry beans into 7 different varieties based on morphological features extracted from images.

## Problem Statement

Classify dry beans into 7 varieties (Barbunya, Bombay, Cali, Dermason, Horoz, Seker, Sira) using 16 morphological features for automated agricultural sorting.

## Dataset

- **Source**: [Kaggle - Dry Bean Dataset (UCI)](https://www.kaggle.com/datasets/sansuthi/dry-bean-dataset)
- **Rows**: 13,611 bean samples
- **Features**: 16 morphological attributes
- **Classes**: 7 bean varieties
- **License**: CC0: Public Domain

### Features

All features are derived from image processing of bean samples:

| Feature | Description |
|---------|-------------|
| Area | Bean area in pixels |
| Perimeter | Bean perimeter length |
| MajorAxisLength | Length of major axis |
| MinorAxisLength | Length of minor axis |
| AspectRatio | Major/Minor axis ratio |
| Eccentricity | Ellipse eccentricity |
| ConvexArea | Convex hull area |
| EquivDiameter | Equivalent diameter |
| Extent | Bean area / bounding box |
| Solidity | Bean area / convex area |
| Roundness | Circularity measure |
| Compactness | Compactness measure |
| ShapeFactor1-4 | Various shape factors |

### Bean Varieties (7 Classes)

1. **Barbunya** - Red kidney bean type
2. **Bombay** - Smaller round variety
3. **Cali** - White kidney bean
4. **Dermason** - Small white variety
5. **Horoz** - Long shaped bean
6. **Seker** - Sugar bean variety
7. **Sira** - Yellow-brown variety

## What You'll Learn

1. **Multiclass Classification**: Handle 7 classes (vs binary 2 classes)
2. **Large Dataset Handling**: Work with 13,611 samples efficiently
3. **Multiclass Metrics**: Macro/weighted averages, per-class analysis
4. **Confusion Matrix (7x7)**: Analyze complex classification patterns
5. **Feature Importance**: Identify discriminative morphological features
6. **Misclassification Analysis**: Understand which varieties are confused
7. **Ensemble Methods**: Combine models for multiclass problems

## PyCaret Features Demonstrated

- `setup()`: Configure for multiclass with feature scaling
- `compare_models()`: Compare algorithms on 7-class problem
- `tune_model()`: Optimize for multiclass accuracy
- `blend_models()`: Soft voting across 7 classes
- `stack_models()`: Meta-learning for multiclass
- `plot_model()`: Multiclass confusion matrix, AUC, class report
- `predict_model()`: Predictions with probabilities for all 7 classes
- `finalize_model()`: Train on full dataset
- `save_model()` / `load_model()`: Model persistence

## Running the Notebook

### Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BalaAnbalagan/pycaret-automl-examples/blob/main/multiclass-classification/dry_bean_classification.ipynb)

### Local Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Launch Jupyter
jupyter notebook dry_bean_classification.ipynb
```

## Expected Results

The model achieves:
- High overall accuracy (typically >90% with ensemble methods)
- Balanced performance across all 7 varieties
- Clear identification of variety-specific features
- Production-ready classification for automated sorting

## Business Value

- **Agricultural Industry**: Automate bean sorting, increase throughput
- **Farmers**: Ensure correct variety identification for pricing
- **Food Processing**: Quality assurance and standardization
- **Seed Companies**: Verify seed purity and prevent contamination

## Key Insights

### Multiclass Complexity
- 7 classes require more complex decision boundaries than binary problems
- Some variety pairs may be morphologically similar (harder to distinguish)
- Ensemble methods particularly effective for multiclass
- Per-class analysis reveals variety-specific performance

### Feature Importance
- Shape features (AspectRatio, Roundness) often highly discriminative
- Size features (Area, Perimeter) help distinguish varieties
- Combination of features provides robust classification

## Deployment Applications

1. **Automated Sorting Systems**: Real-time bean classification on conveyor belts
2. **Quality Control**: Batch purity verification
3. **Mobile Apps**: Field identification by farmers
4. **Research Tools**: Agricultural variety studies

## Notebook Structure

Each code cell includes ELI20 explanations covering:
- **What**: Purpose of the code
- **Why**: Importance for multiclass classification
- **Technical Details**: How it works with 7 classes
- **Expected Output**: What results to expect

## Prerequisites

- Python 3.8+
- Understanding of classification concepts
- Familiarity with evaluation metrics

## Time to Complete

Approximately 40-60 minutes (larger dataset than binary classification)

## Author

**Bala Anbalagan**

## Disclaimer

For educational purposes. Production deployment should include validation with real agricultural data and quality assurance procedures.
