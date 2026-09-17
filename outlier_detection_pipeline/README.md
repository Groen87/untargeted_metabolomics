# Outlier Detection Pipeline

A modular pipeline for outlier detection in metabolomics data using Extended Isolation Forest with cross-validation.

## Overview

This pipeline performs outlier detection on `merged_data_with_classification.csv` using Extended Isolation Forest. It's designed to:

- Train on 80% of samples with Classification = 0 (normal)
- Use 10% of normal samples for validation
- Use 10% of normal samples for testing
- Split samples with Classification = 1, 2, or 3 between validation and test sets
- Compute accuracy, F1 score, precision, recall, and ROC AUC

## Directory Structure

```
outlier_detection_pipeline/
├── __init__.py
├── main.py                  # Main entry point
├── config/
│   ├── __init__.py
│   ├── config.py            # Configuration loader
│   └── config.yaml          # Default configuration
└── pipeline/
    ├── __init__.py
    ├── data_loader.py        # Data loading and splitting
    ├── model.py              # Extended Isolation Forest model
    └── evaluation.py         # Metrics and evaluation
```

## Requirements

```bash
pip install pandas numpy scikit-learn pyyaml joblib
```

## Usage

### Basic Usage

```bash
python outlier_detection_pipeline/main.py \
    --input data/merged_data_with_classification.csv \
    --output outputs/my_outlier_run
```

### With Custom Configuration

```bash
python outlier_detection_pipeline/main.py \
    --input data/merged_data_with_classification.csv \
    --config config/my_config.yaml \
    --output outputs/my_outlier_run
```

### Using Default Configuration

If no `--input` is provided, it uses the path from `config/config.yaml`.

```bash
python outlier_detection_pipeline/main.py
```

## Configuration

Edit `config/config.yaml` to customize the pipeline:

### Data Configuration

```yaml
input_file: "data/merged_data_with_classification.csv"
output_dir: "outputs/outlier_detection"

# Column names
non_feature_columns:
  - "Oordeel trageted"
  - "Classification"

# Classification values
normal_classification: 0
outlier_classifications:
  - 1
  - 2
  - 3
```

### Data Splitting

```yaml
# Split ratios for Classification 0 samples
train_ratio: 0.8
test_ratio: 0.1
val_ratio: 0.1

random_seed: 42
```

### Extended Isolation Forest Parameters

```yaml
n_estimators: 100
max_samples: "auto"
max_features: 1.0
bootstrap: false
n_jobs: -1
random_state: 42
contamination: "auto"
```

### Cross-Validation

```yaml
n_splits: 5
```

### Evaluation Metrics

```yaml
metrics:
  - accuracy
  - f1
  - f1_weighted
  - precision
  - recall
  - roc_auc
  - confusion_matrix
```

### Output Options

```yaml
save_predictions: true
save_model: true
save_plots: true
```

## Input Data Format

The input CSV (`merged_data_with_classification.csv`) should have:

- **Rows**: Patient IDs (will be used as index)
- **Columns**: 
  - Feature columns (metabolite intensities)
  - `Oordeel trageted` (non-feature metadata)
  - `Classification` (0 = normal, 1/2/3 = outliers)

Example:

```csv
,Feature1,Feature2,Feature3,Oordeel trageted,Classification
Patient1,1.2,3.4,5.6,Targeted,0
Patient2,2.3,4.5,6.7,Non-targeted,1
Patient3,3.4,5.6,7.8,Targeted,0
...
```

## Output Files

The pipeline generates:

```
outputs/outlier_detection/
├── model.joblib              # Trained model
├── logs/
│   └── outlier_detection.log # Log file
├── validation/
│   ├── metrics.json          # Validation metrics
│   └── validation_predictions.csv
├── test/
│   ├── metrics.json          # Test metrics
│   └── test_predictions.csv
└── train_cv_predictions.csv   # Cross-validated training predictions
```

## Metrics

The pipeline computes:

- **Accuracy**: Overall correctness
- **F1 Score**: Harmonic mean of precision and recall
- **F1 Weighted**: F1 score weighted by class support
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **ROC AUC**: Area under ROC curve (requires anomaly scores)
- **Confusion Matrix**: Full classification performance breakdown

## Modular Design

Each component is separated into its own module:

- **`data_loader.py`**: Handles data loading and train/val/test splitting
- **`model.py`**: Extended Isolation Forest implementation with CV
- **`evaluation.py`**: Metrics computation and reporting
- **`config/config.py`**: Configuration management

This makes it easy to:
- Modify individual components
- Reuse modules in other pipelines
- Test components independently

## Example Configuration for Different Scenarios

### Conservative Detection (Fewer False Positives)

```yaml
contamination: 0.05  # Expect 5% outliers
n_estimators: 200
max_samples: 0.5      # Use fewer samples per tree
```

### Aggressive Detection (Catch More Outliers)

```yaml
contamination: 0.2   # Expect 20% outliers
n_estimators: 50
max_samples: "auto"
```

### Fast Training

```yaml
n_estimators: 50
n_jobs: 4
max_samples: 0.3
```

## Notes

- The pipeline uses **IsolationForest** from scikit-learn (Extended Isolation Forest is the same algorithm)
- Anomaly scores are negated for ROC AUC calculation (higher score = more anomalous)
- Cross-validation uses StratifiedKFold to maintain class distribution
- All random operations use the configured `random_seed` for reproducibility
