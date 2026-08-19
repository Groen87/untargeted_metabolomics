"""
Combined Batch Pipeline for Untargeted Metabolomics

This package provides a pipeline for processing metabolomics data from multiple
batches that are combined into a single CSV file.

Features:
- Load combined CSV with all batches
- Extract batch information from column names
- Average duplicate samples (_1.raw + _2.raw)
- Per-batch median normalization and LOESS drift correction
- ComBat batch correction across all batches
- QC report generation

Author: Based on Groen87/untargeted_metabolomics
Version: 1.0.0
"""

__version__ = "1.0.0"
