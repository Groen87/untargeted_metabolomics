"""
ComBat batch correction module for combined batch pipeline.

This module handles:
1. Running ComBat batch correction on merged data
2. Generating diagnostic visualizations (UMAP, PCA, Boxplots)
3. Calculating batch effect removal metrics
"""

from typing import Tuple, Dict, Optional, List
import pandas as pd
import numpy as np
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

try:
    from inmoose.pycombat import pycombat_norm
    COMBAT_AVAILABLE = True
except ImportError:
    COMBAT_AVAILABLE = False

logger = logging.getLogger(__name__)


def identify_qc_samples_from_columns(sample_cols, qc_pattern="QC3"):
    """Identify QC samples from column names using pattern matching."""
    qc_samples = []
    for col in sample_cols:
        if qc_pattern in col:
            qc_samples.append(col)
    return qc_samples


def identify_blank_samples_from_columns(sample_cols, blank_pattern="blanco"):
    """Identify blank samples from column names."""
    blank_samples = []
    for col in sample_cols:
        if blank_pattern.lower() in col.lower():
            blank_samples.append(col)
    return blank_samples


def run_combat_on_merged_data(
    merged_data: pd.DataFrame,
    merged_metadata: pd.DataFrame,
    output_dir: Path,
    ref_batch: Optional[int] = None,
    show_plots: bool = False,
    save_plots: bool = True,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run ComBat batch correction on merged data.
    
    Args:
        merged_data: DataFrame with all features (rows) and samples (columns)
        merged_metadata: DataFrame with sample metadata (must have 'batch' column)
        output_dir: Directory to save results
        ref_batch: Reference batch for ComBat (optional)
        show_plots: Whether to display plots
        save_plots: Whether to save plots
        
    Returns:
        Tuple of:
        - corrected_data: ComBat-corrected DataFrame
        - metrics: Dictionary with correction metrics
    """
    if not COMBAT_AVAILABLE:
        raise ImportError("pycombat_norm from inmoose is required but not installed")
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("\nRunning ComBat batch correction...")
    
    # Align data and metadata
    common_samples = list(set(merged_data.columns) & set(merged_metadata['original_col']))
    merged_data = merged_data[common_samples]
    merged_metadata = merged_metadata[merged_metadata['original_col'].isin(common_samples)]
    
    # Create batch vector
    batch_dict = dict(zip(merged_metadata['original_col'], merged_metadata['batch']))
    batch_vector = np.array([batch_dict[col] for col in merged_data.columns])
    
    # Get unique batches and create numeric batch labels
    unique_batches = sorted(set(batch_vector))
    batch_to_num = {b: i+1 for i, b in enumerate(unique_batches)}
    numeric_batch_vector = np.array([batch_to_num[b] for b in batch_vector])
    
    logger.info(f"Batches: {unique_batches}")
    logger.info(f"Batch sizes: {[(b, np.sum(numeric_batch_vector == batch_to_num[b])) for b in unique_batches]}")
    
    # Handle NaN values (replace with small value)
    data_for_combat = merged_data.copy()
    min_positive = data_for_combat[data_for_combat > 0].min().min()
    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
    data_for_combat = data_for_combat.fillna(small_value)
    
    # Filter out zero-variance features
    feature_variances = data_for_combat.var(axis=1)
    nonzero_var_mask = feature_variances > 0
    data_for_combat = data_for_combat.loc[nonzero_var_mask]
    
    logger.info(f"Features with non-zero variance: {nonzero_var_mask.sum()}/{len(nonzero_var_mask)}")
    
    # Add small epsilon to prevent numerical issues
    epsilon = 1e-10
    data_for_combat = data_for_combat + epsilon
    
    # Run ComBat
    logger.info("Running pycombat_norm...")
    combat_result = pycombat_norm(
        data_for_combat,
        numeric_batch_vector,
        covar_mod=np.zeros((len(numeric_batch_vector), 0)),
        ref_batch=ref_batch,
        na_cov_action="na.omit"
    )
    
    # Create corrected DataFrame
    corrected_data = pd.DataFrame(
        combat_result,
        index=data_for_combat.index,
        columns=data_for_combat.columns
    )
    corrected_data = corrected_data.clip(lower=0)
    
    # Restore zero-variance features
    zero_var_features = feature_variances[~nonzero_var_mask].index
    for feat in zero_var_features:
        corrected_data.loc[feat] = merged_data.loc[feat]
    
    # Save corrected data
    corrected_data.to_csv(output_dir / "combat_corrected_data.csv")
    logger.info(f"Saved ComBat-corrected data to {output_dir / 'combat_corrected_data.csv'}")
    
    # Calculate metrics
    def calculate_batch_metrics(df: pd.DataFrame, batch_vec: np.ndarray) -> Dict:
        unique_batches = np.unique(batch_vec)
        if len(unique_batches) < 2:
            return {}
        
        metrics = {}
        for i, batch_label in enumerate(unique_batches):
            batch_data = df.loc[:, batch_vec == batch_label]
            metrics[f'batch_{batch_label}_median'] = float(batch_data.median().median())
        
        # Calculate median CV between batches
        batch_medians = [metrics[f'batch_{b}_median'] for b in unique_batches]
        metrics['median_cv'] = float(np.std(batch_medians) / np.mean(batch_medians) * 100)
        
        return metrics
    
    metrics = {
        'pre_correction': calculate_batch_metrics(merged_data, numeric_batch_vector),
        'post_correction': calculate_batch_metrics(corrected_data, numeric_batch_vector),
    }
    
    # Generate plots if requested
    if save_plots:
        generate_combat_plots(
            merged_data,
            corrected_data,
            batch_vector,
            batch_to_num,
            numeric_batch_vector,
            output_dir,
        )
    
    return corrected_data, metrics


def generate_combat_plots(
    data_before: pd.DataFrame,
    data_after: pd.DataFrame,
    batch_vector: np.ndarray,
    batch_to_num: Dict,
    numeric_batch_vector: np.ndarray,
    output_dir: Path,
) -> None:
    """Generate diagnostic plots for ComBat correction including UMAP, PCA, and Boxplots."""
    unique_batches = sorted(batch_to_num.keys())
    num_batches = len(unique_batches)
    
    # Create color palette - use high-contrast qualitative colors
    # For up to 12 batches, use distinct colors from seaborn's qualitative palettes
    if num_batches <= 10:
        palette = sns.color_palette('tab10', n_colors=num_batches)
    elif num_batches <= 20:
        palette = sns.color_palette('tab20', n_colors=num_batches)
    else:
        # For more than 20 batches, use husl which has good perceptual uniformity
        palette = sns.color_palette('husl', n_colors=num_batches)
    batch_colors = {b: palette[i] for i, b in enumerate(unique_batches)}
    
    # Convert batch_vector elements to standard Python types for dictionary lookup
    batch_numbers = []
    for b in batch_vector:
        if hasattr(b, 'item'):
            b_native = b.item()
        else:
            b_native = b
        batch_numbers.append(batch_to_num[b_native])
    
    # Convert to numpy array for consistency
    batch_vector_native = np.array([b.item() if hasattr(b, 'item') else b for b in batch_vector])
    
    # ========================================================================
    # BOXPLOTS
    # ========================================================================
    
    # Boxplot before correction
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=batch_numbers,
        y=data_before.T.mean(axis=1),
        hue=batch_numbers,
        palette=palette,
        showfliers=False,
        legend=False
    )
    plt.title("Before ComBat - Mean Intensity by Batch")
    plt.xlabel("Batch")
    plt.ylabel("Mean Intensity")
    plt.xticks(range(1, num_batches + 1), unique_batches)
    plt.savefig(output_dir / "before_combat_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Boxplot after correction
    plt.figure(figsize=(12, 6))
    sns.boxplot(
        x=batch_numbers,
        y=data_after.T.mean(axis=1),
        hue=batch_numbers,
        palette=palette,
        showfliers=False,
        legend=False
    )
    plt.title("After ComBat - Mean Intensity by Batch")
    plt.xlabel("Batch")
    plt.ylabel("Mean Intensity")
    plt.xticks(range(1, num_batches + 1), unique_batches)
    plt.savefig(output_dir / "after_combat_boxplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # ========================================================================
    # UMAP PLOTS
    # ========================================================================
    
    try:
        import umap.umap_ as umap
        from sklearn.metrics import silhouette_score
        from sklearn.neighbors import NearestNeighbors
        from scipy.stats import entropy
        
        def calculate_batch_asw(embedding: np.ndarray, batch_labels: np.ndarray) -> float:
            """Calculate Average Silhouette Width."""
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
                    ent = entropy(probs)
                    entropies.append(ent)
                return float(np.mean(entropies))
            except Exception as e:
                logger.debug(f"Entropy calculation failed: {e}")
                return float('nan')
        
        # Generate UMAP plots for before and after
        for label, df, batch_vec in [("Before ComBat", data_before, numeric_batch_vector), 
                                       ("After ComBat", data_after, numeric_batch_vector)]:
            # Handle NaN values
            df_filled = df.copy()
            if df_filled.isna().any().any():
                min_positive = df_filled[df_filled > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df_filled = df_filled.fillna(small_value)
            
            # Generate UMAP - use more neighbors for better global structure
            emb = umap.UMAP(random_state=42, n_jobs=1, n_neighbors=30, min_dist=0.1).fit_transform(df_filled.T)
            
            # Create figure
            plt.figure(figsize=(14, 12))
            
            # Plot each batch with distinct color
            for i, batch_label in enumerate(unique_batches):
                batch_mask = batch_vec == (i + 1)  # numeric_batch_vector uses 1-indexed
                if np.any(batch_mask):
                    plt.scatter(
                        emb[batch_mask, 0],
                        emb[batch_mask, 1],
                        c=[palette[i]],
                        alpha=0.7,
                        s=30,
                        label=f'Batch {batch_label}',
                        edgecolors='black',
                        linewidth=0.3
                    )
            
            plt.title(f"{label} (UMAP) - n_neighbors=30, min_dist=0.1")
            plt.xlabel("UMAP 1")
            plt.ylabel("UMAP 2")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_dir / f"{label.lower().replace(' ', '_')}_umap.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and log metrics with numeric batch labels
            batch_asw = calculate_batch_asw(emb, batch_vec)
            batch_entropy = calculate_batch_mixing_entropy(emb, batch_vec, n_neighbors=10)
            logger.info(f"{label} UMAP - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
            
            # Print interpretation
            if batch_asw > 0.1:
                logger.info(f"  -> Samples show batch separation (ASW={batch_asw:.4f})")
            else:
                logger.info(f"  -> Batch effect removed (ASW={batch_asw:.4f})")
            
            if not np.isnan(batch_entropy):
                max_entropy = np.log(len(unique_batches))
                logger.info(f"  -> Mixing entropy: {batch_entropy:.4f} / {max_entropy:.4f} (higher = better mixing)")
        
    except ImportError as e:
        logger.warning(f"UMAP not available: {e}. Skipping UMAP plots.")
    
    # ========================================================================
    # PCA PLOTS
    # ========================================================================
    
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Generate PCA plots for before and after
        for label, df, batch_vec in [("Before ComBat", data_before, numeric_batch_vector), 
                                       ("After ComBat", data_after, numeric_batch_vector)]:
            # Handle NaN values
            df_filled = df.copy()
            if df_filled.isna().any().any():
                min_positive = df_filled[df_filled > 0].min().min()
                small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                df_filled = df_filled.fillna(small_value)
            
            # Standardize data
            scaler = StandardScaler()
            df_scaled = scaler.fit_transform(df_filled.T)
            
            # Generate PCA
            pca = PCA(n_components=2, random_state=42)
            emb = pca.fit_transform(df_scaled)
            explained_variance = pca.explained_variance_ratio_
            
            # Create figure
            plt.figure(figsize=(14, 12))
            
            # Plot each batch with distinct color
            for i, batch_label in enumerate(unique_batches):
                batch_mask = batch_vec == (i + 1)  # numeric_batch_vector uses 1-indexed
                if np.any(batch_mask):
                    plt.scatter(
                        emb[batch_mask, 0],
                        emb[batch_mask, 1],
                        c=[palette[i]],
                        alpha=0.7,
                        s=30,
                        label=f'Batch {batch_label}',
                        edgecolors='black',
                        linewidth=0.3
                    )
            
            plt.title(f"{label} (PCA)\nExplained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
            plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
            plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
            plt.grid(True, alpha=0.2)
            plt.tight_layout()
            plt.savefig(output_dir / f"{label.lower().replace(' ', '_')}_pca.png", 
                       dpi=300, bbox_inches='tight')
            plt.close()
            
            # Calculate and log metrics with numeric batch labels
            batch_asw = calculate_batch_asw(emb, batch_vec)
            batch_entropy = calculate_batch_mixing_entropy(emb, batch_vec, n_neighbors=10)
            logger.info(f"{label} PCA - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
            
            # Print interpretation
            if batch_asw > 0.1:
                logger.info(f"  -> Samples show batch separation (ASW={batch_asw:.4f})")
            else:
                logger.info(f"  -> Batch effect removed (ASW={batch_asw:.4f})")
            
            if not np.isnan(batch_entropy):
                max_entropy = np.log(len(unique_batches))
                logger.info(f"  -> Mixing entropy: {batch_entropy:.4f} / {max_entropy:.4f} (higher = better mixing)")
        
    except ImportError as e:
        logger.warning(f"sklearn not available: {e}. Skipping PCA plots.")
    
    
    # ========================================================================
    # QC-ONLY PLOTS (for better batch effect assessment)
    # ========================================================================
    
    # Identify QC3 samples
    qc3_samples = identify_qc_samples_from_columns(data_after.columns, qc_pattern="QC3")
    
    if len(qc3_samples) >= 2:
        logger.info(f"Generating QC-only plots with {len(qc3_samples)} QC3 samples")
        
        # Get batch labels for QC samples
        qc_batch_vec = numeric_batch_vector[[col in qc3_samples for col in data_after.columns]]
        qc_unique_batches = np.unique(qc_batch_vec)
        
        # UMAP for QC only
        try:
            import umap.umap_ as umap
            
            for label, df in [("Before ComBat", data_before), ("After ComBat", data_after)]:
                qc_df = df[qc3_samples].copy()
                qc_batch = numeric_batch_vector[[col in qc3_samples for col in df.columns]]
                
                # Handle NaN
                if qc_df.isna().any().any():
                    min_positive = qc_df[qc_df > 0].min().min()
                    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                    qc_df = qc_df.fillna(small_value)
                
                # UMAP
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
                
                plt.title(f"{label} - UMAP (QC3 samples only)")
                plt.xlabel("UMAP 1")
                plt.ylabel("UMAP 2")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(output_dir / f"{label.lower().replace(' ', '_')}_umap_qc3_only.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Metrics
                batch_asw = calculate_batch_asw(emb, qc_batch)
                batch_entropy = calculate_batch_mixing_entropy(emb, qc_batch, n_neighbors=10)
                logger.info(f"{label} UMAP (QC3 only) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
                
                if batch_asw > 0.1:
                    logger.info(f"  -> QC samples show batch separation (ASW={batch_asw:.4f})")
                else:
                    logger.info(f"  -> Batch effect removed in QC samples (ASW={batch_asw:.4f})")
                
                if not np.isnan(batch_entropy):
                    max_entropy = np.log(len(qc_unique_batches))
                    logger.info(f"  -> Mixing entropy: {batch_entropy:.4f} / {max_entropy:.4f}")
        
        except ImportError:
            logger.warning("UMAP not available. Skipping QC-only UMAP plots.")
        
        # PCA for QC only
        try:
            from sklearn.decomposition import PCA
            from sklearn.preprocessing import StandardScaler
            
            for label, df in [("Before ComBat", data_before), ("After ComBat", data_after)]:
                qc_df = df[qc3_samples].copy()
                qc_batch = numeric_batch_vector[[col in qc3_samples for col in df.columns]]
                
                # Handle NaN
                if qc_df.isna().any().any():
                    min_positive = qc_df[qc_df > 0].min().min()
                    small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
                    qc_df = qc_df.fillna(small_value)
                
                # Standardize and PCA
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
                
                plt.title(f"{label} - PCA (QC3 samples only)")
                plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
                plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
                plt.grid(True, alpha=0.2)
                plt.tight_layout()
                plt.savefig(output_dir / f"{label.lower().replace(' ', '_')}_pca_qc3_only.png", 
                           dpi=300, bbox_inches='tight')
                plt.close()
                
                # Metrics
                batch_asw = calculate_batch_asw(emb, qc_batch)
                batch_entropy = calculate_batch_mixing_entropy(emb, qc_batch, n_neighbors=10)
                logger.info(f"{label} PCA (QC3 only) - Batch ASW: {batch_asw:.4f}, Mixing Entropy: {batch_entropy:.4f}")
                
                if batch_asw > 0.1:
                    logger.info(f"  -> QC samples show batch separation (ASW={batch_asw:.4f})")
                else:
                    logger.info(f"  -> Batch effect removed in QC samples (ASW={batch_asw:.4f})")
                
                if not np.isnan(batch_entropy):
                    max_entropy = np.log(len(qc_unique_batches))
                    logger.info(f"  -> Mixing entropy: {batch_entropy:.4f} / {max_entropy:.4f}")
        
        except ImportError:
            logger.warning("sklearn not available. Skipping QC-only PCA plots.")
    else:
        logger.warning(f"Need at least 2 QC3 samples for QC-only plots. Found {len(qc3_samples)}.")
    
    # ========================================================================
    # END QC-ONLY PLOTS
    # ========================================================================

    logger.info(f"Saved all plots to {output_dir}")
