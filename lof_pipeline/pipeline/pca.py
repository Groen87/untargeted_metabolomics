"""
PCA module for LOF pipeline - same as outlier_detection_pipeline/pipeline/pca.py
Reusing the existing SparsePCAWrapper from the isolation forest pipeline.
"""

import sys
import os

# Add the parent pipeline directory to path so we can import SparsePCAWrapper
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from outlier_detection_pipeline.pipeline.pca import SparsePCAWrapper

__all__ = ['SparsePCAWrapper']
