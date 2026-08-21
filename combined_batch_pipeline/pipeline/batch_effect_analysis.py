"""
Batch effect analysis module for combined batch pipeline.

This module provides comprehensive analysis of batch effects before correction,
including:
- PCA-based batch effect metrics (batch contribution to PCs)
- Batch ASW (Average Silhouette Width)
- Batch mixing entropy
- Between-batch vs within-batch variance ratios
- Batch effect size estimates
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

logger = logging.getLogger(__name__)


def calculate_batch_asw(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
) -> float:
    """
    Calculate Average Silhouette Width (ASW) for batch labels.
    
    Higher ASW (>0.1) indicates batch separation (batch effect present).
    Lower ASW (<0.1) indicates good mixing (no batch effect).
    
    Args:
        embeddings: 2D array of sample embeddings (e.g., from PCA)
        batch_labels: Array of batch labels for each sample
        
    Returns:
        ASW score (float)
    """
    from sklearn.metrics import silhouette_score
    
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0
    
    try:
        asw = silhouette_score(embeddings, batch_labels)
        return asw
    except Exception as e:
        logger.warning(f"Could not calculate ASW: {e}")
        return 0.0


def calculate_batch_mixing_entropy(
    embeddings: np.ndarray,
    batch_labels: np.ndarray,
    n_neighbors: int = 10,
) -> float:
    """
    Calculate batch mixing entropy.
    
    Higher entropy (closer to max entropy) indicates better mixing.
    
    Args:
        embeddings: 2D array of sample embeddings
        batch_labels: Array of batch labels for each sample
        n_neighbors: Number of neighbors for k-NN graph
        
    Returns:
        Mixing entropy (float)
    """
    from sklearn.neighbors import NearestNeighbors
    
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return 0.0
    
    try:
        # Build k-NN graph
        nbrs = NearestNeighbors(n_neighbors=min(n_neighbors, len(embeddings) - 1)).fit(embeddings)
        distances, indices = nbrs.kneighbors(embeddings)
        
        # Calculate entropy for each sample
        entropies = []
        for i in range(len(embeddings)):
            neighbor_batches = batch_labels[indices[i]]
            if len(neighbor_batches) == 0:
                continue
            # Calculate probability distribution of batches in neighborhood
            unique, counts = np.unique(neighbor_batches, return_counts=True)
            probs = counts / len(neighbor_batches)
            # Entropy
            entropy = -np.sum(probs * np.log2(probs + 1e-10))
            entropies.append(entropy)
        
        return np.mean(entropies)
    except Exception as e:
        logger.warning(f"Could not calculate mixing entropy: {e}")
        return 0.0


def calculate_batch_variance_ratio(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
) -> Dict[str, float]:
    """
    Calculate the ratio of between-batch to within-batch variance for each feature.
    
    Higher ratios indicate stronger batch effects.
    
    Args:
        data: DataFrame with samples as columns
        batch_labels: Array of batch labels for each sample (aligned with data.columns)
        
    Returns:
        Dictionary with:
        - 'mean_ratio': Mean variance ratio across all features
        - 'median_ratio': Median variance ratio across all features
        - 'features_above_2': Number of features with ratio > 2
        - 'features_above_5': Number of features with ratio > 5
    """
    unique_batches = np.unique(batch_labels)
    if len(unique_batches) < 2:
        return {'mean_ratio': 0.0, 'median_ratio': 0.0, 'features_above_2': 0, 'features_above_5': 0}
    
    # Calculate for each feature
    ratios = []
    for feature in data.index:
        feature_values = data.loc[feature].values
        
        # Within-batch variance (average variance within each batch)
        within_vars = []
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            if np.sum(batch_mask) > 1:
                within_vars.append(np.var(feature_values[batch_mask]))
        
        within_var = np.mean(within_vars) if within_vars else 0
        
        # Between-batch variance
        batch_means = []
        for batch in unique_batches:
            batch_mask = batch_labels == batch
            if np.sum(batch_mask) > 0:
                batch_means.append(np.mean(feature_values[batch_mask]))
        
        between_var = np.var(batch_means) if len(batch_means) > 1 else 0
        
        # Ratio (avoid division by zero)
        if within_var > 0:
            ratio = between_var / within_var
        else:
            ratio = 0.0
        
        ratios.append(ratio)
    
    ratios = np.array(ratios)
    
    return {
        'mean_ratio': float(np.mean(ratios)),
        'median_ratio': float(np.median(ratios)),
        'features_above_2': int(np.sum(ratios > 2)),
        'features_above_5': int(np.sum(ratios > 5)),
        'max_ratio': float(np.max(ratios)) if len(ratios) > 0 else 0.0,
    }


def calculate_batch_contribution_to_pcs(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    n_components: int = 10,
) -> Dict[str, float]:
    """
    Calculate how much each PC is explained by batch effects.
    
    Uses PERMANOVA-like approach: calculate R^2 for batch vs PC scores.
    
    Args:
        data: DataFrame with samples as columns
        batch_labels: Array of batch labels for each sample
        n_components: Number of PC components to analyze
        
    Returns:
        Dictionary with batch contribution for each PC
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import LinearRegression
        
        # Handle NaN
        df_filled = data.copy()
        if df_filled.isna().any().any():
            min_positive = df_filled[df_filled > 0].min().min()
            small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
            df_filled = df_filled.fillna(small_value)
        
        # Standardize and do PCA
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filled.T)
        
        pca = PCA(n_components=n_components, random_state=42)
        pc_scores = pca.fit_transform(df_scaled)
        
        # For each PC, calculate how much variance is explained by batch
        batch_contributions = {}
        for i in range(min(n_components, pc_scores.shape[1])):
            pc = pc_scores[:, i]
            
            # One-way ANOVA: batch explains variance in PC
            unique_batches = np.unique(batch_labels)
            if len(unique_batches) < 2:
                batch_contributions[f'PC{i+1}'] = 0.0
                continue
            
            # Calculate between-batch variance
            batch_means = []
            for batch in unique_batches:
                mask = batch_labels == batch
                if np.sum(mask) > 0:
                    batch_means.append(np.mean(pc[mask]))
            
            overall_mean = np.mean(pc)
            ss_between = sum(len(batch_means) * (m - overall_mean)**2 for m in batch_means)
            
            # Calculate within-batch variance
            ss_within = 0
            for batch in unique_batches:
                mask = batch_labels == batch
                if np.sum(mask) > 1:
                    ss_within += np.sum((pc[mask] - np.mean(pc[mask]))**2)
            
            # Total variance
            ss_total = ss_between + ss_between
            
            if ss_total > 0:
                r2 = ss_between / ss_total
            else:
                r2 = 0.0
            
            batch_contributions[f'PC{i+1}'] = float(r2)
        
        return batch_contributions
        
    except ImportError:
        logger.warning("sklearn not available. Cannot calculate batch contribution to PCs.")
        return {}
    except Exception as e:
        logger.warning(f"Error calculating batch contribution to PCs: {e}")
        return {}


def calculate_permanova_p_value(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    n_permutations: int = 1000,
) -> float:
    """
    Calculate PERMANOVA p-value for batch effect.
    
    Args:
        data: DataFrame with samples as columns
        batch_labels: Array of batch labels for each sample
        n_permutations: Number of permutations for p-value calculation
        
    Returns:
        PERMANOVA p-value (float)
    """
    try:
        from skbio.diversity.beta import permanova
        
        # Convert to distance matrix
        df_filled = data.copy()
        if df_filled.isna().any().any():
            min_positive = df_filled[df_filled > 0].min().min()
            small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
            df_filled = df_filled.fillna(small_value)
        
        # Use Euclidean distance
        from scipy.spatial.distance import pdist, squareform
        dist_matrix = squareform(pdist(df_filled.T, 'euclidean'))
        
        # PERMANOVA
        results = permanova(dist_matrix, batch_labels, permutations=n_permutations)
        return float(results['p-value'])
    except ImportError:
        logger.warning("skbio not available. Cannot calculate PERMANOVA p-value.")
        return 1.0
    except Exception as e:
        logger.warning(f"Error calculating PERMANOVA: {e}")
        return 1.0


def plot_batch_pca(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    output_dir: Path,
    title: str = "PCA",
) -> None:
    """
    Generate PCA plot colored by batch.
    
    Args:
        data: DataFrame with samples as columns
        batch_labels: Array of batch labels for each sample
        output_dir: Directory to save plot
        title: Plot title
    """
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Handle NaN
        df_filled = data.copy()
        if df_filled.isna().any().any():
            min_positive = df_filled[df_filled > 0].min().min()
            small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
            df_filled = df_filled.fillna(small_value)
        
        # Standardize and do PCA
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filled.T)
        
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(df_scaled)
        explained_variance = pca.explained_variance_ratio_
        
        # Create figure
        plt.figure(figsize=(14, 12))
        
        unique_batches = np.unique(batch_labels)
        palette = sns.color_palette("husl", len(unique_batches))
        
        for i, batch_label in enumerate(unique_batches):
            batch_mask = batch_labels == batch_label
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
        
        plt.title(f"{title}\nExplained Variance: PC1={explained_variance[0]:.2%}, PC2={explained_variance[1]:.2%}")
        plt.xlabel(f"PC1 ({explained_variance[0]:.1%})")
        plt.ylabel(f"PC2 ({explained_variance[1]:.1%})")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        plt.grid(True, alpha=0.2)
        plt.tight_layout()
        plt.savefig(output_dir / f"{title.lower().replace(' ', '_')}_batch_pca.png", 
                   dpi=300, bbox_inches='tight')
        plt.close()
        
    except ImportError as e:
        logger.warning(f"sklearn not available: {e}. Skipping batch PCA plot.")


def analyze_batch_effects(
    data: pd.DataFrame,
    batch_labels: np.ndarray,
    output_dir: Path,
    prefix: str = "pre_combat",
) -> Dict[str, float]:
    """
    Perform comprehensive batch effect analysis.
    
    Args:
        data: DataFrame with samples as columns
        batch_labels: Array of batch labels for each sample (aligned with data.columns)
        output_dir: Directory to save plots and results
        prefix: Prefix for output files
        
    Returns:
        Dictionary with all batch effect metrics
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"BATCH EFFECT ANALYSIS: {prefix}")
    logger.info(f"{'='*70}")
    
    results = {}
    
    # 1. Basic info
    unique_batches = np.unique(batch_labels)
    n_samples = len(batch_labels)
    n_features = data.shape[0]
    
    logger.info(f"Number of batches: {len(unique_batches)}")
    logger.info(f"Batch sizes: {[(b, np.sum(batch_labels == b)) for b in unique_batches]}")
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Number of samples: {n_samples}")
    
    results['n_batches'] = len(unique_batches)
    results['n_features'] = n_features
    results['n_samples'] = n_samples
    
    # 2. PCA-based metrics
    try:
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler
        
        # Handle NaN
        df_filled = data.copy()
        if df_filled.isna().any().any():
            min_positive = df_filled[df_filled > 0].min().min()
            small_value = min_positive / 2 if not pd.isna(min_positive) else 1e-10
            df_filled = df_filled.fillna(small_value)
        
        # Standardize and do PCA
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_filled.T)
        
        pca = PCA(n_components=2, random_state=42)
        emb = pca.fit_transform(df_scaled)
        explained_variance = pca.explained_variance_ratio_
        
        # Calculate ASW
        asw = calculate_batch_asw(emb, batch_labels)
        results['pca_asw'] = asw
        logger.info(f"\nPCA ASW: {asw:.4f}")
        if asw > 0.1:
            logger.info("  -> Strong batch effect detected (ASW > 0.1)")
        else:
            logger.info("  -> Batch effect is minimal (ASW <= 0.1)")
        
        # Calculate mixing entropy
        entropy = calculate_batch_mixing_entropy(emb, batch_labels)
        results['pca_mixing_entropy'] = entropy
        max_entropy = np.log(len(unique_batches))
        logger.info(f"PCA Mixing Entropy: {entropy:.4f} / {max_entropy:.4f}")
        if entropy / max_entropy > 0.8:
            logger.info("  -> Good mixing (entropy > 80% of max)")
        else:
            logger.info("  -> Batch separation detected (entropy < 80% of max)")
        
        # Generate PCA plot
        plot_batch_pca(data, batch_labels, output_dir, f"{prefix}_batch_effect")
        
    except ImportError as e:
        logger.warning(f"sklearn not available: {e}. Skipping PCA-based metrics.")
    
    # 3. Variance ratio analysis
    var_results = calculate_batch_variance_ratio(data, batch_labels)
    results.update(var_results)
    
    logger.info(f"\nBatch Variance Ratio Analysis:")
    logger.info(f"  Mean ratio (between/within): {var_results['mean_ratio']:.2f}")
    logger.info(f"  Median ratio: {var_results['median_ratio']:.2f}")
    logger.info(f"  Max ratio: {var_results['max_ratio']:.2f}")
    logger.info(f"  Features with ratio > 2: {var_results['features_above_2']}")
    logger.info(f"  Features with ratio > 5: {var_results['features_above_5']}")
    
    if var_results['mean_ratio'] > 1.0:
        logger.info("  -> Significant batch effects detected (mean ratio > 1)")
    
    # 4. Batch contribution to PCs
    pc_contrib = calculate_batch_contribution_to_pcs(data, batch_labels)
    results['pc_batch_contribution'] = pc_contrib
    
    if pc_contrib:
        logger.info(f"\nBatch Contribution to Principal Components:")
        for pc, contrib in pc_contrib.items():
            logger.info(f"  {pc}: {contrib:.2%} variance explained by batch")
            if contrib > 0.1:
                logger.info(f"    -> Strong batch effect in {pc}")
    
    # 5. PERMANOVA (if available)
    permanova_p = calculate_permanova_p_value(data, batch_labels)
    results['permanova_p_value'] = permanova_p
    logger.info(f"\nPERMANOVA p-value: {permanova_p:.4e}")
    if permanova_p < 0.05:
        logger.info("  -> Significant batch effect (p < 0.05)")
    else:
        logger.info("  -> No significant batch effect detected (p >= 0.05)")
    
    # 6. Save results to file
    results_file = output_dir / f"{prefix}_batch_effect_metrics.txt"
    with open(results_file, 'w') as f:
        f.write("BATCH EFFECT ANALYSIS RESULTS\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Prefix: {prefix}\n")
        f.write(f"Number of batches: {results.get('n_batches', 'N/A')}\n")
        f.write(f"Number of features: {results.get('n_features', 'N/A')}\n")
        f.write(f"Number of samples: {results.get('n_samples', 'N/A')}\n\n")
        
        f.write("PCA Metrics:\n")
        f.write(f"  ASW: {results.get('pca_asw', 'N/A'):.4f}\n")
        f.write(f"  Mixing Entropy: {results.get('pca_mixing_entropy', 'N/A'):.4f}\n\n")
        
        f.write("Variance Ratio Analysis:\n")
        f.write(f"  Mean ratio: {results.get('mean_ratio', 'N/A'):.2f}\n")
        f.write(f"  Median ratio: {results.get('median_ratio', 'N/A'):.2f}\n")
        f.write(f"  Max ratio: {results.get('max_ratio', 'N/A'):.2f}\n")
        f.write(f"  Features > 2: {results.get('features_above_2', 'N/A')}\n")
        f.write(f"  Features > 5: {results.get('features_above_5', 'N/A')}\n\n")
        
        f.write("PC Batch Contribution:\n")
        for pc, contrib in results.get('pc_batch_contribution', {}).items():
            f.write(f"  {pc}: {contrib:.2%}\n")
        f.write("\n")
        
        f.write(f"PERMANOVA p-value: {results.get('permanova_p_value', 'N/A'):.4e}\n")
    
    logger.info(f"\nBatch effect metrics saved to {results_file}")
    
    return results
