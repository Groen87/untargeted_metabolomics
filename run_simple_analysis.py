#!/usr/bin/env python3
"""Simple, clean entry point for pathway analysis.

This script implements a streamlined pathway analysis that:
1. Maps features to pathways, removing unmapped features
2. Calculates Z-scores for normals and IMDs only  
3. Combines Z-scores into compound scores per pathway using absolute Stouffer's Z
4. Finds optimal cutoffs empirically from the normal distribution
5. Flags samples based on extreme pathway deviations

Usage:
    python run_simple_analysis.py --input combined_batch_pipeline/dummy.csv --output outputs/simple_test
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd
import numpy as np

from pathway_pipeline.pipeline.pathway_analysis_simple import run_simple_pathway_analysis


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def _setup_logging(output_dir: Path) -> None:
    """Add file logging alongside the stream handler."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(output_dir / "simple_analysis.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(fh)


def load_data(input_file: str) -> tuple:
    """Load and prepare data.
    
    The input CSV has features as rows and samples as columns.
    We need to transpose it to have samples as rows and features as columns.
    """
    df = pd.read_csv(input_file, index_col=0)
    logger.info(f"Loaded {input_file}: {df.shape[0]} features x {df.shape[1]} samples")
    
    # Transpose: features (rows) -> columns, samples (columns) -> rows
    df_t = df.T
    logger.info(f"Transposed: {df_t.shape[0]} samples x {df_t.shape[1]} features")
    
    # Extract sample IDs from column names (original row index)
    # Column names are like "Area: posneg_MZ25_36_25230101131_1.raw (F1)"
    # We want to extract the sample identifier (F1, F2, etc.)
    sample_ids = []
    for col in df.columns:
        # Try to extract the (F#) part
        if isinstance(col, str) and '(' in col and ')' in col:
            sample_id = col.split('(')[1].split(')')[0]
        else:
            sample_id = str(col)
        sample_ids.append(sample_id)
    
    df_t.index = sample_ids
    
    # Features are now the columns
    features = df_t
    
    # Clean feature names - remove any NaN or non-string columns
    features = features.loc[:, features.columns.notna()]
    features.columns = [str(c) for c in features.columns]
    
    # Create dummy metadata with Classification and Oordeel columns
    # For now, let's assume first 217 samples are normals (Class 0, Oordeel 0)
    # and next 100 are IMDs (Class 1, Oordeel 1)
    # This is a placeholder - in real usage, you'd load actual classification
    n_samples = len(sample_ids)
    classification = [0] * n_samples  # All normals by default
    oordeel = [0] * n_samples
    
    # Mark some as IMDs (Class 1, Oordeel 1)
    # For testing, let's mark samples F100-F199 as IMDs
    for i, sid in enumerate(sample_ids):
        if isinstance(sid, str) and sid.startswith('F') and sid[1:].isdigit():
            num = int(sid[1:])
            if 100 <= num < 200:
                classification[i] = 1
                oordeel[i] = 1
    
    metadata = pd.DataFrame({
        'Classification': classification,
        'Oordeel targeted': oordeel
    }, index=sample_ids)
    
    logger.info(f"Features: {len(features.columns)} columns, Metadata: {metadata.shape[1]} columns")
    logger.info(f"Classification: {metadata['Classification'].value_counts().to_dict()}")
    logger.info(f"Oordeel: {metadata['Oordeel targeted'].value_counts().to_dict()}")
    return features, metadata


def create_simple_mapping(features: pd.DataFrame) -> pd.DataFrame:
    """Create a simple feature-to-pathway mapping based on feature names.
    
    Since we don't have HMDB XML or pathways.tsv, we'll create a simple mapping
    based on the feature names (which contain HMDB IDs).
    """
    rows = []
    for feature in features.columns:
        # Convert to string if needed
        feature_str = str(feature)
        
        # Extract HMDB ID from feature name if present
        # Feature names are like "Propionylcarnitine.HMDB0000824"
        hmdb_id = None
        if '.HMDB' in feature_str:
            hmdb_id = feature_str.split('.HMDB')[1]
        
        # Assign to a pathway based on metabolite class
        # This is a simple placeholder - in real usage, use actual pathways
        pathway_name = "Unknown"
        feature_upper = feature_str.upper()
        if 'CARNITINE' in feature_upper:
            pathway_name = "Fatty Acid Metabolism"
        elif 'PHENYLALANINE' in feature_upper:
            pathway_name = "Amino Acid Metabolism"
        elif 'CREATININE' in feature_upper:
            pathway_name = "Nitrogen Metabolism"
        
        rows.append({
            'feature': feature_str,
            'hmdb_id': hmdb_id,
            'pathway_name': pathway_name,
            'smp_id': f"PATH_{pathway_name}"
        })
    
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(
        description="Simple pathway analysis: feature -> HMDB -> pathway -> Stouffer's Z"
    )
    parser.add_argument("--input", default="combined_batch_pipeline/dummy.csv",
                        help="Path to the feature matrix CSV.")
    parser.add_argument("--output", default="outputs/simple_analysis",
                        help="Output directory.")
    args = parser.parse_args()
    
    # Setup
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    _setup_logging(out)
    
    logger.info(f"Input: {args.input}")
    logger.info(f"Output: {out}")
    
    # Step 1: Load data
    logger.info("\n" + "="*70)
    logger.info("STEP 1: Load data")
    logger.info("="*70)
    features, metadata = load_data(args.input)
    
    # Step 2: Create simple feature-to-pathway mapping
    logger.info("\n" + "="*70)
    logger.info("STEP 2: Create feature-to-pathway mapping")
    logger.info("="*70)
    feature_to_pathway = create_simple_mapping(features)
    logger.info(f"Created mapping for {len(feature_to_pathway)} features")
    logger.info(f"Pathways: {feature_to_pathway['pathway_name'].unique().tolist()}")
    
    # Save mapping
    feature_to_pathway.to_csv(out / "feature_to_pathway.csv", index=False)
    
    # Step 3: Filter features to only those in pathways
    logger.info("\n" + "="*70)
    logger.info("STEP 3: Filter to pathway-mapped features")
    logger.info("="*70)
    pathway_features = [f for f in feature_to_pathway['feature'].unique() if pd.notna(f)]
    features_filtered = features[pathway_features]
    logger.info(f"Filtered to {len(pathway_features)} pathway-mapped features")
    
    # Step 4: Compute z-scores
    logger.info("\n" + "="*70)
    logger.info("STEP 4: Compute z-scores")
    logger.info("="*70)
    
    # Use class1_imd classification
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
    normal_mask = (cls == 0) & (oor == 0)
    n_normal = int(normal_mask.sum())
    logger.info(f"Normal reference: {n_normal} samples")
    
    # Simple z-score computation (no age adjustment)
    from scipy import stats as scipy_stats
    
    zscores = pd.DataFrame(index=features_filtered.index, columns=features_filtered.columns)
    for feature in features_filtered.columns:
        normal_values = features_filtered.loc[normal_mask, feature]
        sample_values = features_filtered[feature]
        
        # Compute median and IQR on normals
        normal_median = np.nanmedian(normal_values)
        normal_iqr = np.percentile(normal_values.dropna(), 75) - np.percentile(normal_values.dropna(), 25)
        
        if normal_iqr > 0:
            # Robust z-score: (x - median) / IQR
            zscores[feature] = (sample_values - normal_median) / normal_iqr
        else:
            zscores[feature] = 0.0
    
    logger.info(f"Computed z-scores: {zscores.shape[0]} samples x {zscores.shape[1]} features")
    
    # Save z-scores
    zscores.to_csv(out / "metabolite_zscores.csv")
    
    # Step 5: Run simple pathway analysis
    logger.info("\n" + "="*70)
    logger.info("STEP 5: Run simple pathway analysis")
    logger.info("="*70)
    
    results = run_simple_pathway_analysis(
        zscores=zscores,
        feature_to_pathway=feature_to_pathway,
        metadata=metadata,
        output_dir=out,
        min_pathway_size=1,  # Lower since we have few features
        classification_scheme="class1_imd",
        min_detection=0.80,
        max_contamination=0.05,
        min_flagged_pathways=1,
        min_percentile=95.0,
        max_percentile=99.9999,
        n_percentiles=20,
    )
    
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE")
    logger.info("="*70)
    logger.info(f"Optimal threshold: {results['threshold_info']['optimal_threshold']:.2f}")
    logger.info(f"IMD detection: {results['validation']['detection_rate']*100:.1f}%")
    logger.info(f"Normal contamination: {results['validation']['contamination_rate']*100:.1f}%")
    logger.info(f"Flagged normals: {results['validation']['normals_flagged']}")
    logger.info(f"Flagged IMDs: {results['validation']['imds_flagged']}")
    
    if results['validation']['normals_flagged'] > 0:
        logger.warning(f"WARNING: {results['validation']['normals_flagged']} normals were flagged!")
    
    if results['validation']['detection_rate'] < 0.80:
        logger.warning(f"WARNING: Detection rate below 80% target")


if __name__ == "__main__":
    main()
