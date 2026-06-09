# Multi-Batch Metabolomics Pipeline

This pipeline processes multiple batch folders (e.g., "MZ25_36", "MZ26_10") simultaneously through the full metabolomics data processing workflow and performs multi-batch ComBat correction.

## Overview

The multi-batch pipeline:
1. Processes each batch folder through the full single-batch pipeline (data processing → drift correction → median normalization)
2. Merges all median-normalized files from all batches
3. Performs ComBat batch correction on the merged data
4. Optionally generates QC reports

## Structure

```
multi_batch_pipeline/
├── __init__.py
├── README.md
├── main.py              # Main entry point for multi-batch processing
├── config/
│   ├── __init__.py
│   └── config.yaml      # Configuration file
└── pipeline/
    ├── __init__.py
    ├── data_processing.py
    ├── injection_order.py
    ├── loess_drift_correction.py
    ├── merge_batches_for_combat.py
    ├── multi_batch_combat.py  # Multi-batch merging and ComBat
    ├── median_normalization.py
    ├── combat_utils.py
    └── quality_control.py
```

## Usage

### Basic Usage

Process all batches in the data directory:
```bash
python multi_batch_pipeline/main.py
```

Process specific batches:
```bash
python multi_batch_pipeline/main.py --batches MZ25_36 MZ26_10 MZ27_15
```

Process only NEG mode:
```bash
python multi_batch_pipeline/main.py --batches MZ25_36 MZ26_10 --mode NEG
```

### Command-line Options

```
--batches BATCH1 BATCH2 ...   Specify batch folders to process (default: auto-detect)
--mode {NEG,POS}              Process only NEG or POS mode (default: both)
--data-dir DIRECTORY          Base data directory (default: data)
--output-dir DIRECTORY        Output directory for results (default: data/multi_batch_output/)
--rt-threshold FLOAT          RT threshold for feature matching (default: 0.02)
--no-plots                    Disable plot generation
--show-plots                 Show plots interactively
--no-qc                      Disable QC report generation
--config FILE                Path to config file
```

## Requirements

- Python >= 3.8
- pandas >= 1.3.0
- numpy >= 1.21.0
- scipy >= 1.7.0
- statsmodels >= 0.12.0
- matplotlib >= 3.4.0
- pyyaml >= 6.0
- openpyxl >= 3.0.0
- inmoose (optional, for QC reports)
- umap-learn (optional, for visualizations)
- seaborn (optional, for visualizations)

## Input Data Structure

Each batch folder should contain:
```
data/
├── MZ25_36/
│   ├── MZ25_36_NEG.csv          # Raw data for NEG mode
│   ├── MZ25_36_POS.csv          # Raw data for POS mode
│   └── MZ25_36_meta.xlsx        # Metadata file
├── MZ26_10/
│   ├── MZ26_10_NEG.csv
│   ├── MZ26_10_POS.csv
│   └── MZ26_10_meta.xlsx
└── ...
```

## Output

Results are saved to `data/multi_batch_output/` by default:
```
data/multi_batch_output/
├── NEG/
│   ├── combat_input/
│   │   ├── merged_data_for_combat.csv
│   │   ├── merged_batch_for_combat.csv
│   │   └── *only_features.csv (batch-specific features)
│   └── combat_corrected/
│       ├── combat_corrected_data.csv
│       ├── before_combat_*.png
│       └── after_combat_*.png
└── POS/
    └── ... (same structure as NEG)
```

## Configuration

Edit `multi_batch_pipeline/config/config.yaml` to customize pipeline parameters:

```yaml
# File paths
batch_folder: "MZ25_36"
reference_batch: null  # Not used in multi-batch mode
intensity_threshold: 10000
qc_pattern: "expQC"
qc_intensity_threshold: 1
frac: 0.5

# Logging
log_level: "INFO"
log_file: "logs/multi_batch_pipeline.log"
```

## Differences from Single-Batch Pipeline

1. **Batch Processing**: Processes all specified batches before merging
2. **Feature Matching**: Uses RT-based matching to align features across batches
3. **Batch-Specific Features**: Identifies and saves features unique to each batch
4. **Multi-Batch ComBat**: Performs ComBat correction across all batches simultaneously
5. **QC Reports**: Generates comprehensive QC reports for the merged data

## Notes

- The pipeline reuses the single-batch processing logic from `metabolomics_pipeline`
- Each batch is processed independently first, then merged
- Features present in only one batch are removed before ComBat correction
- expQC samples are automatically excluded from the merged data
