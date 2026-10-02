"""
RALPS batch correction module for combined batch pipeline.

This module handles:
1. Preparing data and batch info files for RALPS
2. Running RALPS batch correction
3. Generating diagnostic visualizations (UMAP, PCA) for comparison with ComBat
4. Using QC3 samples for regularization and QC4/QC_blaauw for benchmarking
"""

import os
import sys
import subprocess
import tempfile
import shutil
from pathlib import Path
from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
import logging
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)


def prepare_ralps_files(
    df: pd.DataFrame,
    batch_groups: Dict[str, List[str]],
    sample_info: Optional[Dict[str, Dict]] = None,
    output_dir: Path = None,
    qc3_pattern: str = "QC3",
    qc4_pattern: str = "QC4",
    blanco_pattern: str = "blanco",
    blaauw_pattern: str = "blaauw",
) -> Tuple[Path, Path, Path]:
    """
    Prepare data file and batch info file for RALPS.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        batch_groups: Dictionary mapping batch names to sample column names
        sample_info: Optional dictionary with sample metadata
        output_dir: Directory to save RALPS files
        qc3_pattern: Pattern to identify QC3 samples (used for regularization)
        qc4_pattern: Pattern to identify QC4 samples (used for benchmarking)
        blanco_pattern: Pattern to identify blanco samples
        blaauw_pattern: Pattern to identify blaauw samples (used for benchmarking)
        
    Returns:
        Tuple of (data_file_path, info_file_path, ralps_output_dir)
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir) / "ralps_files"
        output_dir.mkdir(parents=True, exist_ok=True)
    
    data_file = output_dir / "data.csv"
    info_file = output_dir / "batch_info.csv"
    
    # Save data file (transposed: samples as rows, features as columns)
    # RALPS expects: rows=samples, columns=features
    df_t = df.T
    df_t.to_csv(data_file, index_label="sample_id")
    
    # Create batch info file
    # Columns: sample_id, batch, group, benchmark
    # - QC3 samples: group = same group ID (for regularization), benchmark = 0
    # - QC4 and QC_blaauw samples: group = 0, benchmark = same benchmark ID
    # - Other samples: group = 0, benchmark = 0
    
    sample_ids = list(df.columns)
    batch_info = []
    
    # Create group and benchmark assignments
    group_id = 1
    benchmark_id = 1
    
    for sample_id in sample_ids:
        # Determine batch
        batch = None
        for batch_name, samples in batch_groups.items():
            if sample_id in samples:
                batch = batch_name
                break
        
        if batch is None:
            batch = "unknown"
            logger.warning(f"Sample {sample_id} not found in any batch")
        
        # Determine if it's a QC sample
        is_qc3 = qc3_pattern in sample_id
        is_qc4 = qc4_pattern in sample_id
        is_blaauw = blaauw_pattern.lower() in sample_id.lower()
        is_blanco = blanco_pattern.lower() in sample_id.lower()
        
        # Assign group (for regularization) - QC3 samples get same group
        if is_qc3:
            group = f"reg_{group_id}"
        else:
            group = "0"
        
        # Assign benchmark - QC4 and blaauw samples get benchmark IDs
        if is_qc4 or is_blaauw:
            benchmark = f"bench_{benchmark_id}"
            benchmark_id += 1
        else:
            benchmark = "0"
        
        batch_info.append({
            "sample_id": sample_id,
            "batch": batch,
            "group": group,
            "benchmark": benchmark,
        })
    
    info_df = pd.DataFrame(batch_info)
    info_df.to_csv(info_file, index=False)
    
    logger.info(f"Created RALPS data file: {data_file}")
    logger.info(f"Created RALPS batch info file: {info_file}")
    logger.info(f"QC3 samples (regularization): {sum(is_qc3 for _ in sample_ids) if 'is_qc3' in locals() else 'N/A'}")
    logger.info(f"QC4 + blaauw samples (benchmarking): {sum(is_qc4 or is_blaauw for _ in sample_ids) if 'is_qc4' in locals() and 'is_blaauw' in locals() else 'N/A'}")
    
    return data_file, info_file, output_dir


def run_ralps(
    data_file: Path,
    info_file: Path,
    output_dir: Path,
    config_params: Optional[Dict] = None,
) -> Path:
    """
    Run RALPS batch correction.
    
    Args:
        data_file: Path to RALPS data file
        info_file: Path to RALPS batch info file
        output_dir: Directory for RALPS output
        config_params: Optional dictionary of RALPS config parameters
        
    Returns:
        Path to RALPS output directory
    """
    # Create config file
    config_file = output_dir / "config.csv"
    
    default_config = {
        "data_path": str(data_file),
        "info_path": str(info_file),
        "out_path": str(output_dir / "ralps_output"),
        "latent_dim": "-1",
        "variance_ratio": "0.9,0.95,0.99",
        "n_replicates": "3",
        "grid_size": "1",
        "d_lr": "0.00005-0.005",
        "g_lr": "0.00005-0.005",
        "d_lambda": "0.-10.",
        "g_lambda": "0.-10.",
        "v_lambda": "0.-10.",
        "train_ratio": "0.9",
        "batch_size": "32,64,128",
        "epochs": "30",
        "skip_epochs": "3",
        "keep_checkpoints": "False",
        "device": "cpu",
        "plots_extension": "png",
        "min_relevant_intensity": "1000",
        "allowed_vc_increase": "0.05",
    }
    
    if config_params:
        default_config.update(config_params)
    
    # Write config file
    with open(config_file, 'w') as f:
        for key, value in default_config.items():
            f.write(f"{key},{value}\n")
    
    # Check if RALPS is available
    ralps_script = None
    # Try to find ralps.py in common locations
    possible_paths = [
        "ralps/src/ralps.py",
        "RALPS/src/ralps.py",
        "/opt/ralps/src/ralps.py",
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            ralps_script = path
            break
    
    if ralps_script is None:
        logger.error("RALPS script not found. Please clone RALPS repository and ensure ralps.py is accessible.")
        logger.error("You can clone it from: https://github.com/zamboni-lab/RALPS")
        return output_dir
    
    # Run RALPS
    output_path = output_dir / "ralps_output"
    output_path.mkdir(parents=True, exist_ok=True)
    
    cmd = [
        sys.executable,
        ralps_script,
        "-n", str(config_file),
    ]
    
    logger.info(f"Running RALPS: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            cwd=output_dir,
            capture_output=True,
            text=True,
            timeout=3600,  # 1 hour timeout
        )
        
        if result.returncode != 0:
            logger.error(f"RALPS failed with return code {result.returncode}")
            logger.error(f"STDOUT: {result.stdout}")
            logger.error(f"STDERR: {result.stderr}")
        else:
            logger.info("RALPS completed successfully")
            logger.info(f"STDOUT: {result.stdout}")
    
    except subprocess.TimeoutExpired:
        logger.error("RALPS timed out after 1 hour")
    except Exception as e:
        logger.error(f"Error running RALPS: {e}")
    
    return output_path


def load_ralps_results(ralps_output_dir: Path) -> pd.DataFrame:
    """
    Load normalized data from RALPS output.
    
    Args:
        ralps_output_dir: Path to RALPS output directory
        
    Returns:
        DataFrame with normalized features (rows) and samples (columns)
    """
    # RALPS saves normalized data as data_normalized.csv
    normalized_file = ralps_output_dir / "data_normalized.csv"
    
    if not normalized_file.exists():
        logger.error(f"RALPS normalized data not found at {normalized_file}")
        return pd.DataFrame()
    
    # Load and transpose back to features x samples
    df = pd.read_csv(normalized_file, index_col=0)
    df = df.T
    
    logger.info(f"Loaded RALPS normalized data: {df.shape}")
    return df


def generate_comparison_plots(
    data_combat: pd.DataFrame,
    data_ralps: pd.DataFrame,
    batch_groups: Dict[str, List[str]],
    output_dir: Path,
    sample_info: Optional[Dict[str, Dict]] = None,
    qc3_pattern: str = "QC3",
) -> None:
    """
    Generate identical UMAP and PCA plots for ComBat vs RALPS comparison.
    
    Args:
        data_combat: ComBat-corrected data
        data_ralps: RALPS-corrected data
        batch_groups: Dictionary mapping batch names to sample column names
        output_dir: Directory to save comparison plots
        sample_info: Optional dictionary with sample metadata
        qc3_pattern: Pattern to identify QC3 samples
    """
    output_dir = Path(output_dir) / "comparison_plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create numeric batch vector
    all_samples = list(data_combat.columns)
    numeric_batch_vector = np.zeros(len(all_samples), dtype=int)
    batch_names = sorted(batch_groups.keys())
    
    for i, batch_name in enumerate(batch_names):
        for sample in batch_groups[batch_name]:
            if sample in all_samples:
                idx = all_samples.index(sample)
                numeric_batch_vector[idx] = i + 1
    
    unique_batches = sorted(batch_groups.keys())
    num_batches = len(unique_batches)
    
    # Create palette with distinct colors per batch
    palette = sns.color_palette("husl", n_colors=num_batches)
    
    # Identify QC3 samples
    qc3_samples = [col for col in all_samples if qc3_pattern in col]
    
    def calculate_batch_asw(embedding: np.ndarray, batch_labels: np.ndarray) -> float:
        """Calculate Average Silhouette Width."""
        from sklearn.metrics import silhouette_score
        unique_batches = np.unique(batch_labels)
        if len(unique_batches) < 2:
            return float('nan')
        try:
            score = silhouette_score(embedding, batch_labels, metric='euclidean')
            return float(score)
        except ValueError:
            return float('nan')
    
    def calculate_batch_mixing_entropy(embedding: np.ndarray, batch_labels: np.ndarray, n_neighbors: int = 10) -> float:
        """Calculate entropy of batch mixing."""
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import entropy as scipy_entropy
        unique_batches = np.unique(batch_labels)
        n_samples = len(batch_labels)
        if len(unique_batches) < 2 or n_samples < n_neighbors + 1:
            return float('nan')
        try:
            nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding)
            distances, indices = nbrs.kneighbors(embedding)
            entropies = []
            for i in range(n_samples):
                neighbor_idx = indices[i]
                neighbor_batches = batch_labels[neighbor_idx]
                unique, counts = np.unique(neighbor_batches, return_counts=True)
                probs = counts / counts.sum()
                ent = scipy_entropy(probs)
                entropies.append(ent)
            return float(np.mean(entropies))
        except Exception as e:
            logger.debug(f"Entropy calculation failed: {e}")
            return float('nan')
    
    # Generate comparison plots for all samples
    for method_name, df in [("ComBat", data_combat), ("RALPS", data_ralps)]:
        # UMAP
        try:
            import umap.umap_ as umap
            
            df_filled = df.copy()
            if df_filled.isna().any().any():
                min_positive = df_filled[df_filled > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df_filled = df_filled.fillna(small_value)
            
            emb = umap.UMAP(random_state=42, n_jobs=1, n_neighbors=30, min_dist=0.1).fit_transform(df_filled.T)
            
            plt.figure(figsize=(14, 12))
            for i, batch_label in enumerate(unique_batches):
                batch_mask = numeric_batch_vector == (i + 1)
                if np.any(batch_mask):
                    plt.scatter(
                        emb[batch_mask, 0], emb[batch_mask, 1],
                        c=[palette[i]], alpha=0.7, s=30,
                        label=f'Batch {batch_label}', edgecolors='black', linewidth=0.3
                    )
            
            plt.title(f"{method_name} - UMAP (All Samples)")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_dir / f"{method_name.lower()}_umap_all.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Metrics
            batch_asw = calculate_batch_asw(emb, numeric_batch_vector)
            batch_entropy = calculate_batch_mixing_entropy(emb, numeric_batch_vector, n_neighbors=10)
            logger.info(f"{method_name} UMAP (All) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
        
        except ImportError:
            logger.warning(f"UMAP not available. Skipping {method_name} UMAP plots.")
        
        # PCA
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            df_filled = df.copy()
            if df_filled.isna().any().any():
                min_positive = df_filled[df_filled > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df_filled = df_filled.fillna(small_value)
            
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_filled.T)
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(df_scaled)
            
            plt.figure(figsize=(14, 12))
            for i, batch_label in enumerate(unique_batches):
                batch_mask = numeric_batch_vector == (i + 1)
                if np.any(batch_mask):
                    plt.scatter(
                        emb[batch_mask, 0], emb[batch_mask, 1],
                        c=[palette[i]], alpha=0.7, s=30,
                        label=f'Batch {batch_label}', edgecolors='black', linewidth=0.3
                    )
            
            plt.title(f"{method_name} - PCA (All Samples)")
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_dir / f"{method_name.lower()}_pca_all.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            # Metrics
            batch_asw = calculate_batch_asw(emb, numeric_batch_vector)
            batch_entropy = calculate_batch_mixing_entropy(emb, numeric_batch_vector, n_neighbors=10)
            logger.info(f"{method_name} PCA (All) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
        
        except ImportError:
            logger.warning(f"sklearn not available. Skipping {method_name} PCA plots.")
    
    # Generate QC-only comparison plots
    if len(qc3_samples) >= 2:
        logger.info(f"Generating QC-only comparison plots with {len(qc3_samples)} QC3 samples")
        
        qc_batch_vec = numeric_batch_vector[[col in qc3_samples for col in all_samples]]
        qc_unique_batches = np.unique(qc_batch_vec)
        
        for method_name, df in [("ComBat", data_combat), ("RALPS", data_ralps)]:
            # UMAP for QC only
            try:
                import umap.umap_ as umap
                
                qc_df = df[qc3_samples].copy()
                qc_batch = numeric_batch_vector[[col in qc3_samples for col in df.columns]]
                
                if qc_df.isna().any().any():
                    min_positive = qc_df[qc_df > 0].min().min()
                    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                    qc_df = qc_df.fillna(small_value)
                
                n_neighbors = min(30, len(qc3_samples) - 1)
                emb = umap.UMAP(random_state=42, n_jobs=1, n_neighbors=n_neighbors, min_dist=0.1).fit_transform(qc_df.T)
                
                plt.figure(figsize=(14, 12))
                for i, batch_label in enumerate(qc_unique_batches):
                    batch_mask = qc_batch == (i + 1)
                    if np.any(batch_mask):
                        plt.scatter(
                            emb[batch_mask, 0], emb[batch_mask, 1],
                            c=[palette[i]], alpha=0.7, s=30,
                            label=f'Batch {batch_label}', edgecolors='black', linewidth=0.3
                        )
                
                plt.title(f"{method_name} - UMAP (QC3 Only)")
                plt.xlabel("UMAP 1")
                plt.ylabel("UMAP 2")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(output_dir / f"{method_name.lower()}_umap_qc3.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                batch_asw = calculate_batch_asw(emb, qc_batch)
                batch_entropy = calculate_batch_mixing_entropy(emb, qc_batch, n_neighbors=10)
                logger.info(f"{method_name} UMAP (QC3) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
            
            except ImportError:
                logger.warning(f"UMAP not available. Skipping {method_name} QC-only UMAP plots.")
            
            # PCA for QC only
            try:
                from sklearn.decomposition import PCA
                from sklearn.preprocessing import StandardScaler
                
                qc_df = df[qc3_samples].copy()
                qc_batch = numeric_batch_vector[[col in qc3_samples for col in df.columns]]
                
                if qc_df.isna().any().any():
                    min_positive = qc_df[qc_df > 0].min().min()
                    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                    qc_df = qc_df.fillna(small_value)
                
                scaler = StandardScaler()
                qc_scaled = scaler.fit_transform(qc_df.T)
                pca = PCA(n_components=2, random_state=42)
                emb = pca.fit_transform(qc_scaled)
                
                plt.figure(figsize=(14, 12))
                for i, batch_label in enumerate(qc_unique_batches):
                    batch_mask = qc_batch == (i + 1)
                    if np.any(batch_mask):
                        plt.scatter(
                            emb[batch_mask, 0], emb[batch_mask, 1],
                            c=[palette[i]], alpha=0.7, s=30,
                            label=f'Batch {batch_label}', edgecolors='black', linewidth=0.3
                        )
                
                plt.title(f"{method_name} - PCA (QC3 Only)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(output_dir / f"{method_name.lower()}_pca_qc3.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                batch_asw = calculate_batch_asw(emb, qc_batch)
                batch_entropy = calculate_batch_mixing_entropy(emb, qc_batch, n_neighbors=10)
                logger.info(f"{method_name} PCA (QC3) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
            
            except ImportError:
                logger.warning(f"sklearn not available. Skipping {method_name} QC-only PCA plots.")
    
    logger.info(f"Saved comparison plots to {output_dir}")


def run_ralps_correction(
    df: pd.DataFrame,
    batch_groups: Dict[str, List[str]],
    sample_info: Optional[Dict[str, Dict]] = None,
    output_dir: Path = None,
    qc3_pattern: str = "QC3",
    qc4_pattern: str = "QC4",
    blanco_pattern: str = "blanco",
    blaauw_pattern: str = "blaauw",
    config_params: Optional[Dict] = None,
) -> Tuple[pd.DataFrame, Path]:
    """
    Complete RALPS correction pipeline.
    
    Args:
        df: DataFrame with features as rows, samples as columns
        batch_groups: Dictionary mapping batch names to sample column names
        sample_info: Optional dictionary with sample metadata
        output_dir: Directory to save all RALPS files and results
        qc3_pattern: Pattern to identify QC3 samples (for regularization)
        qc4_pattern: Pattern to identify QC4 samples (for benchmarking)
        blanco_pattern: Pattern to identify blanco samples
        blaauw_pattern: Pattern to identify blaauw samples (for benchmarking)
        config_params: Optional dictionary of RALPS config parameters
        
    Returns:
        Tuple of (RALPS-corrected DataFrame, Path to RALPS output directory)
    """
    if output_dir is None:
        output_dir = Path(tempfile.mkdtemp())
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"RALPS Batch Correction")
    logger.info(f"{'='*70}")
    
    # Step 1: Prepare RALPS files
    data_file, info_file, ralps_files_dir = prepare_ralps_files(
        df, batch_groups, sample_info, output_dir,
        qc3_pattern, qc4_pattern, blanco_pattern, blaauw_pattern
    )
    
    # Step 2: Run RALPS
    ralps_output_dir = run_ralps(data_file, info_file, ralps_files_dir, config_params)
    
    # Step 3: Load RALPS results
    if ralps_output_dir.exists():
        df_ralps = load_ralps_results(ralps_output_dir)
    else:
        logger.error("RALPS output directory not found. Returning original data.")
        return df.copy(), output_dir
    
    # Step 4: Generate comparison plots
    generate_comparison_plots(
        df, df_ralps, batch_groups, output_dir,
        sample_info, qc3_pattern
    )
    
    logger.info(f"RALPS correction completed. Output saved to {output_dir}")
    
    return df_ralps, output_dir
