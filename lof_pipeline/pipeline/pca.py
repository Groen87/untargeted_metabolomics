"""
PCA module for LOF pipeline.
Reuses the SparsePCAWrapper from the outlier_detection_pipeline.
"""

import sys
import os

# Add the parent directory to path so we can import SparsePCAWrapper
parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, parent_dir)

from outlier_detection_pipeline.pipeline.pca import SparsePCAWrapper

__all__ = ['SparsePCAWrapper']
