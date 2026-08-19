# Combined Batch Pipeline

A pipeline for processing metabolomics data from multiple batches combined into a single CSV file.

## Overview

This pipeline is designed for data where:
- All batches are in **one CSV file** (not separate files per batch)
- Column headers follow the format: `Area: posneg_MZ25_36_25230101131_1.raw (F1)`
- Batch names are embedded in filenames (e.g., `MZ25_36`)
- Duplicate samples have `_1.raw` and `_2.raw` suffixes (will be averaged)
- Features are already aligned by Compound Discoverer (no RT-based alignment needed)

## Installation

```bash
# Clone the repository
cd untargeted_metabolomics

# Install dependencies
pip install pandas numpy matplotlib seaborn statsmodels pyyaml

# Optional: Install inmoose for ComBat and QC reports
pip install inmoose
```

## Usage

### Command Line

```bash
# Basic usage
python combined_batch_pipeline/main.py --input data/combined_all_batches.csv

# With custom output directory
python combined_batch_pipeline/main.py \
    --input data/combined_all_batches.csv \
    --output output/my_run

# Skip QC analysis
python combined_batch_pipeline/main.py \
    --input data/combined_all_batches.csv \
    --no-qc

# All options
python combined_batch_pipeline/main.py --help
```

### Python API

```python
from combined_batch_pipeline.main import run_full_pipeline

results = run_full_pipeline(
    input_file="data/combined_all_batches.csv",
    output_dir="output/my_run",
    qc_pattern="expQC",
    fallback_qc_pattern="QC3",
    frac=0.5,
    ref_batch=None,
    run_qc=True,
    save_plots=True,
    show_plots=False,
)

# Access results
corrected_data = results['corrected_data']
metadata = results['metadata']
combat_metrics = results['combat_metrics']
```

## Configuration

Create a `config.yaml` file in `combined_batch_pipeline/config/`:

```yaml
# Input/Output
input_file: "data/combined_all_batches.csv"
output_dir: "output/combined_batch_pipeline"

# Processing parameters
intensity_threshold: 10000
qc_pattern: "expQC"
fallback_qc_pattern: "QC3"
qc_intensity_threshold: 0.1
frac: 0.5
rt_threshold: 0.02
ref_batch: null

# Logging
log_level: "INFO"
log_file: "logs/combined_batch_pipeline.log"

# Options
run_qc: true
save_plots: true
show_plots: false
```

## Pipeline Steps

1. **Data Loading** (`pipeline/data_loader.py`)
   - Read combined CSV
   - Extract batch names from column headers
   - Identify sample types (QC, blanco, etc.)
   - Filter low-intensity features

2. **Batch Processing** (`pipeline/batch_processing.py`)
   - For each batch:
     - Average duplicate samples (_1 + _2)
     - Apply median normalization using QC samples
     - Apply LOESS drift correction

3. **Merge Batches**
   - Combine all processed batches into single DataFrame

4. **ComBat Correction** (`pipeline/combat_correction.py`)
   - Run ComBat to remove batch effects
   - Generate diagnostic plots

5. **QC Analysis** (`pipeline/quality_control.py`)
   - Generate QC reports using inmoose (if available)

## Input File Format

The input CSV should have:
- First column: `Name` (feature names)
- Subsequent columns: `Area: {filename} ({F#})`

Example:
```
Name,Area: posneg_MZ25_36_25230101131_1.raw (F1),Area: posneg_MZ25_36_25230101131_2.raw (F2),...
feature1,100000,120000,...
feature2,5000,6000,...
```

## Batch Name Extraction

Batch names are extracted from filenames using regex patterns:
- `posneg_{BATCH}_...` → extracts `BATCH`
- `Posneg_{BATCH}_...` → extracts `BATCH`
- `{BATCH}_...` → extracts `BATCH`

Valid batch names contain underscores and digits (e.g., `MZ25_36`).

## Sample Type Classification

Sample types are identified from sample IDs:
- `expQC`, `QC3`, `QC4` → "QC"
- `blanco` → "blanco"
- `blauw` → "blauw"
- `mix` → "Mix"
- Everything else → "Sample"

## Output

The pipeline creates the following directory structure:
```
output/combined_batch_pipeline/
├── batch_outputs/
│   └── {batch_name}/
│       ├── processed_data.csv
│       └── metadata.csv
├── merged/
│   ├── merged_data.csv
│   └── merged_metadata.csv
├── combat/
│   ├── combat_corrected_data.csv
│   ├── before_combat_boxplot.png
│   └── after_combat_boxplot.png
├── qc_reports/
│   └── qc_report.html
└── final/
    ├── final_corrected_data.csv
    └── final_metadata.csv
```

## Dependencies

### Required
- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- statsmodels
- pyyaml

### Optional
- inmoose (for ComBat and QC reports)

## Troubleshooting

### "No QC samples found"
Check your `--qc-pattern` and `--fallback-qc` arguments. Your QC samples should contain one of these patterns in their filenames.

### "Could not extract batch from column"
Ensure your column names follow the expected format. The pipeline looks for patterns like `posneg_MZ25_36_...` or `MZ25_36_...`.

### ComBat fails
Make sure you have `inmoose` installed: `pip install inmoose`

## License

This pipeline is part of the Groen87/untargeted_metabolomics project.
