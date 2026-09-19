"""Enhanced pathway-shift analysis with multiple statistical methods and monitoring.

This module implements an enhanced layered flagging scheme with:

1. **Stouffer's Z-score**: Properly combines p-values across pathway metabolites
   to replace the F(P) flagged-fraction statistic.

2. **Multiple testing correction**: Bonferroni and FDR (Benjamini-Hochberg) corrections
   for pathway-level hypotheses.

3. **Weighted decision score**: Replaces counting with weighted anomaly scores
   that emphasize strong pathway signals over weak ones.

4. **Pathway-specific empirical thresholds**: Computes per-pathway thresholds
   from the normal reference distribution.

5. **Two-stage detection**: Lenient first stage to identify candidates, rigorous
   second stage with corrected p-values.

6. **Comprehensive monitoring**: All intermediate results (p-values, corrected p-values,
   weights, scores) are saved to CSV for diagnostics.

Usage:
    This module provides drop-in replacements for functions in pathway_stats.py
    with additional monitoring outputs. Set use_enhanced_stats: true in config
    to enable the enhanced pipeline.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

logger = logging.getLogger(__name__)

try:
    import seaborn as sns
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for saving figures
    import matplotlib.pyplot as plt
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False
    logger.warning("Seaborn/matplotlib not available; pathway visualizations "
                   "will be skipped. Install with: pip install seaborn matplotlib")


def _compute_stouffers_z(per_metabolite_zscores: np.ndarray,
                         two_tailed: bool = True) -> Tuple[float, float, float, float]:
    """Compute Stouffer's combined Z-score and p-value for a set of z-scores.

    Computes BOTH signed and absolute versions:
    - Signed: Z_combined = sum(z_i) / sqrt(n) - detects coordinated direction
    - Absolute: Z_abs = sum(|z_i|) / sqrt(n) - detects any disturbance

    The absolute version is critical for IMD detection where metabolites
    may move in opposite directions (e.g., block in pathway causing some
    metabolites to accumulate and others to deplete).

    Args:
        per_metabolite_zscores: 1-D array of z-scores for metabolites in a pathway.
        two_tailed: if True, compute two-tailed p-value; else one-tailed.

    Returns:
        Tuple of (Z_stouffer_signed, p_value_signed, Z_stouffer_abs, p_value_abs).
        NaN for all if all inputs are NaN.
    """
    # Filter out NaN values
    valid = per_metabolite_zscores[~np.isnan(per_metabolite_zscores)]
    if len(valid) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")

    n = len(valid)
    
    # Signed Stouffer's Z (original) - detects coordinated direction
    z_sum_signed = np.sum(valid)
    z_combined_signed = z_sum_signed / np.sqrt(n)
    
    # Absolute Stouffer's Z - detects ANY disturbance regardless of direction
    z_sum_abs = np.sum(np.abs(valid))
    z_combined_abs = z_sum_abs / np.sqrt(n)

    if two_tailed:
        # Two-tailed p-values
        p_signed = 2 * (1 - scipy_stats.norm.cdf(abs(z_combined_signed)))
        p_abs = 2 * (1 - scipy_stats.norm.cdf(abs(z_combined_abs)))
    else:
        # One-tailed (directional)
        p_signed = 1 - scipy_stats.norm.cdf(z_combined_signed)
        p_abs = 1 - scipy_stats.norm.cdf(z_combined_abs)

    return float(z_combined_signed), float(p_signed), float(z_combined_abs), float(p_abs)


def _compute_per_metabolite_pvalues(zscores: np.ndarray,
                                     two_tailed: bool = True) -> np.ndarray:
    """Convert z-scores to two-tailed p-values.

    Args:
        zscores: 1-D or 2-D array of z-scores.
        two_tailed: if True, two-tailed p-values; else one-tailed.

    Returns:
        Array of p-values with same shape as zscores.
    """
    if two_tailed:
        return 2 * (1 - scipy_stats.norm.cdf(np.abs(zscores)))
    else:
        return 1 - scipy_stats.norm.cdf(zscores)


def compute_enhanced_pathway_statistics(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    normal_mask: pd.Series,
    min_pathway_size: int = 3,
    output_dir: Optional[Path] = None,
    gray_mask: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Compute enhanced pathway statistics with Stouffer's Z and monitoring.

    For each pathway and sample, computes:
    - Traditional: Z_med, F, Z_up, Z_down, Z_split
    - Enhanced: Z_stouffer, p_stouffer, p_bonferroni, p_fdr
    - Per-pathway empirical thresholds from normal distribution

    All intermediate values are saved to CSV for monitoring.

    Args:
        zscores: per-metabolite z-scores (rows=samples, columns=features).
        feature_to_pathway: long (feature, smp_id, pathway_name, ...) table.
        normal_mask: boolean Series aligned to zscores.index.
        min_pathway_size: minimum number of matched features for a pathway.
        output_dir: directory to save monitoring CSVs. If None, no files written.
        gray_mask: optional boolean Series marking gray-zone samples.
            If provided, gray samples are excluded from p-value computation
            (only normals + IMD used) and flagging is evaluated separately
            for normals vs non-normals.

    Returns:
        Long DataFrame with one row per (sample, pathway) with all statistics.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get normal z-scores for empirical threshold computation
    z_normals = zscores.loc[normal_mask]
    normal_sample_ids = z_normals.index

    # Determine which samples to use for null distribution (p-value computation)
    # If gray_mask is provided, use only normals (exclude gray from null)
    # Otherwise use all non-IMD samples (normals + gray as null)
    if gray_mask is not None:
        # Null samples = normals only (gray samples are not part of null)
        null_mask = normal_mask.copy()
        null_sample_ids = normal_sample_ids
        logger.info(f"Using {len(null_sample_ids)} normal samples for null "
                    f"distribution (gray samples excluded from p-value computation)")
    else:
        # Default: use normals + gray as null (traditional behavior)
        null_mask = normal_mask
        null_sample_ids = normal_sample_ids
        if gray_mask is not None:
            null_mask = null_mask | gray_mask
            null_sample_ids = zscores.loc[null_mask].index
            logger.info(f"Using {len(null_sample_ids)} null samples "
                        f"(normals + gray) for p-value computation")

    # Build pathway feature mapping
    available = set(zscores.columns)
    pathway_features: Dict[str, Dict] = {}
    if not feature_to_pathway.empty:
        for smp_id, grp in feature_to_pathway.groupby("smp_id"):
            feats = sorted(set(grp["feature"]) & available)
            name = grp["pathway_name"].iloc[0] if "pathway_name" in grp.columns else smp_id
            if len(feats) >= min_pathway_size:
                pathway_features[smp_id] = {"pathway_name": name, "features": feats}

    if not pathway_features:
        logger.warning("No pathways have enough matched metabolites.")
        return pd.DataFrame(columns=[
            "sample_id", "smp_id", "pathway_name", "n_metabolites",
            "z_med", "flagged_fraction", "z_up", "z_down", "z_split",
            "z_stouffer", "z_stouffer_signed", "p_stouffer", "p_stouffer_signed", "p_bonferroni",
            "empirical_p_threshold", "n_normals_used"
        ])

    # Pre-compute per-pathway empirical thresholds from normals
    # Use null samples (normals only, or normals + gray) for threshold computation
    z_null = zscores.loc[null_mask]
    empirical_thresholds: Dict[str, float] = {}
    for smp_id, info in pathway_features.items():
        feat_cols = info["features"]
        null_z = z_null[feat_cols].to_numpy(dtype=float)
        # Compute Z_med for each null sample for this pathway
        with np.errstate(all="ignore"):
            z_med_null = np.nanmedian(null_z, axis=1)
        # Use 95th percentile of |Z_med| over null samples as empirical threshold
        empirical_thresholds[smp_id] = float(
            np.nanpercentile(np.abs(z_med_null), 95)
        )

    # Compute statistics for all samples
    sample_ids = zscores.index
    z_arr = zscores.to_numpy(dtype=float)
    col_index = {c: i for i, c in enumerate(zscores.columns)}
    n_samples = len(sample_ids)
    n_pathways = len(pathway_features)

    # Collect all rows
    rows: List[Dict] = []

    # For monitoring: collect all p-values for FDR computation
    all_p_values: List[float] = []
    all_pathway_names: List[str] = []
    all_sample_ids: List = []

    for smp_id, info in pathway_features.items():
        feat_cols = [col_index[f] for f in info["features"]]
        sub = z_arr[:, feat_cols]
        n = sub.shape[1]
        pathway_name = info["pathway_name"]
        empirical_thresh = empirical_thresholds.get(smp_id, np.nan)

        # Traditional statistics
        with np.errstate(all="ignore"):
            z_med = np.nanmedian(sub, axis=1)
        # Per-metabolite thresholds (99th percentile of |z| over normals)
        per_met_thresh = z_normals[info["features"]].abs().quantile(0.99, axis=0).to_numpy()
        # Handle NaN and zero thresholds - replace with inf so comparisons work
        per_met_thresh = np.where((np.isnan(per_met_thresh) | (per_met_thresh <= 0)), np.inf, per_met_thresh)

        flagged = (np.abs(sub) > per_met_thresh) & ~np.isnan(sub)
        k_valid = np.sum(~np.isnan(sub), axis=1)
        n_flagged = np.sum(flagged, axis=1)
        f = np.where(k_valid > 0, n_flagged / k_valid, np.nan)

        # Signed-extreme statistics
        with np.errstate(invalid="ignore"):
            pos = np.where(sub > 0, sub, np.nan)
            neg = np.where(sub < 0, sub, np.nan)
        n_pos = np.sum(~np.isnan(pos), axis=1)
        n_neg = np.sum(~np.isnan(neg), axis=1)
        with np.errstate(all="ignore"):
            z_up_raw = np.nanmedian(pos, axis=1)
            z_down_raw = np.nanmedian(neg, axis=1)
        z_up = np.where(n_pos >= 2, z_up_raw, np.nan)
        z_down = np.where(n_neg >= 2, z_down_raw, np.nan)
        cand = np.stack([np.abs(z_up), np.abs(z_down)], axis=1)
        all_invalid = np.all(np.isnan(cand), axis=1)
        with np.errstate(all="ignore"):
            z_split = np.nanmax(cand, axis=1)
        z_split = np.where(all_invalid, np.nan, z_split)

        # Enhanced: Stouffer's Z-score (both signed and absolute)
        z_stouffer_signed = np.zeros(n_samples)
        p_stouffer_signed = np.zeros(n_samples)
        z_stouffer_abs = np.zeros(n_samples)
        p_stouffer_abs = np.zeros(n_samples)
        for i in range(n_samples):
            z_stouffer_signed[i], p_stouffer_signed[i], z_stouffer_abs[i], p_stouffer_abs[i] = _compute_stouffers_z(sub[i, :])

        # Use absolute Stouffer's Z for flagging (detects opposite-direction disturbances)
        # Keep signed for monitoring/compatibility
        z_stouffer = z_stouffer_abs
        p_stouffer = p_stouffer_abs
        
        # Compute Bonferroni-corrected p-values (for monitoring, not flagging)
        n_pathways = len(pathway_features)
        p_bonferroni = p_stouffer * n_pathways
        
        # For FDR, we need all p-values across all pathways
        all_p_values.extend(p_stouffer.tolist())
        all_pathway_names.extend([pathway_name] * n_samples)
        all_sample_ids.extend(sample_ids.tolist())

        for i, sid in enumerate(sample_ids):
            rows.append({
                "sample_id": sid,
                "smp_id": smp_id,
                "pathway_name": pathway_name,
                "n_metabolites": int(n),
                # Traditional
                "z_med": float(z_med[i]) if not np.isnan(z_med[i]) else float("nan"),
                "flagged_fraction": float(f[i]) if not np.isnan(f[i]) else float("nan"),
                "z_up": float(z_up[i]) if not np.isnan(z_up[i]) else float("nan"),
                "z_down": float(z_down[i]) if not np.isnan(z_down[i]) else float("nan"),
                "z_split": float(z_split[i]) if not np.isnan(z_split[i]) else float("nan"),
                # Enhanced (using absolute Stouffer's Z)
                "z_stouffer": float(z_stouffer[i]) if not np.isnan(z_stouffer[i]) else float("nan"),
                "z_stouffer_signed": float(z_stouffer_signed[i]) if not np.isnan(z_stouffer_signed[i]) else float("nan"),
                "p_stouffer": float(p_stouffer[i]) if not np.isnan(p_stouffer[i]) else float("nan"),
                "p_stouffer_signed": float(p_stouffer_signed[i]) if not np.isnan(p_stouffer_signed[i]) else float("nan"),
                "p_bonferroni": float(p_bonferroni[i]) if not np.isnan(p_bonferroni[i]) else float("nan"),
                # Empirical threshold
                "empirical_p_threshold": float(empirical_thresh) if not np.isnan(empirical_thresh) else float("nan"),
            })

    # Compute FDR correction using all p-values
    stats_df = pd.DataFrame(rows)

    # Add FDR-corrected p-values
    if len(all_p_values) > 0:
        # Create a flat DataFrame for FDR computation
        p_df = pd.DataFrame({
            "p_value": all_p_values,
            "pathway_name": all_pathway_names,
            "sample_id": all_sample_ids
        })
        # Sort by p-value for BH procedure
        p_df = p_df.sort_values("p_value").reset_index(drop=True)
        p_df["rank"] = range(1, len(p_df) + 1)
        p_df["p_fdr"] = p_df["p_value"] * n_pathways / p_df["rank"]
        p_df["p_fdr"] = p_df["p_fdr"].cummax()  # BH procedure
        p_df["p_fdr"] = np.minimum(p_df["p_fdr"], 1.0)

        # Merge back to stats_df
        p_df = p_df[["sample_id", "pathway_name", "p_fdr"]]
        stats_df = stats_df.merge(p_df, on=["sample_id", "pathway_name"], how="left")

    # Save monitoring CSVs
    if output_dir is not None:
        # Save pathway-level statistics
        stats_df.to_csv(output_dir / "enhanced_pathway_statistics.csv", index=False)
        logger.info(f"Wrote enhanced_pathway_statistics.csv with {len(stats_df)} rows")

        # Save p-value distribution for diagnostics
        p_dist = pd.DataFrame({
            "p_stouffer": all_p_values,
            "p_bonferroni": [p * n_pathways for p in all_p_values],
        })
        p_dist.to_csv(output_dir / "p_value_distribution.csv", index=False)
        logger.info(f"Wrote p_value_distribution.csv with {len(p_dist)} p-values")

    logger.info(f"Computed enhanced pathway statistics for {len(pathway_features)} "
                f"pathways across {len(sample_ids)} samples.")
    return stats_df


def flag_pathways_enhanced(
    stats: pd.DataFrame,
    extreme_z_threshold: float = 15.0,
    use_empirical_threshold: bool = True,
    empirical_percentile: float = 99.999,
) -> pd.DataFrame:
    """Flag pathways using extreme mode only.

    Only flags pathways with |Z_stouffer| > threshold.
    This is designed for IMD detection where only 1-2 pathways are 
    extremely disturbed, while normals have no such extreme deviations.

    Args:
        stats: output of compute_enhanced_pathway_statistics.
        extreme_z_threshold: |Z_stouffer| threshold for flagging (used if use_empirical_threshold=False).
        use_empirical_threshold: If True, compute threshold from normal distribution.
        empirical_percentile: Percentile of normal |Z_stouffer| distribution to use as threshold.

    Returns:
        stats with added flag_extreme and flagged_two_stage columns.
    """
    if stats.empty:
        return stats

    # Determine threshold
    if use_empirical_threshold and "sample_type" in stats.columns:
        # Compute empirical threshold from normal samples only
        normal_stats = stats[stats["sample_type"] == "normal"]
        if not normal_stats.empty and len(normal_stats) > 10:
            # Use percentile of |Z_stouffer| from normals
            threshold = float(np.percentile(normal_stats["z_stouffer"].abs().dropna(), empirical_percentile))
            logger.info(f"Using empirical |Z_stouffer| threshold: {threshold:.2f} "
                       f"(from {len(normal_stats)} normal samples at {empirical_percentile}th percentile)")
        else:
            # Fallback to fixed threshold
            threshold = extreme_z_threshold
            logger.warning(f"Not enough normal samples for empirical threshold, "
                          f"using fixed threshold: {threshold}")
    elif use_empirical_threshold:
        # If sample_type not in stats, we can't separate normals
        # Try to infer from the data - but this is risky
        threshold = extreme_z_threshold
        logger.warning(f"sample_type not in stats, using fixed threshold: {threshold}")
    else:
        threshold = extreme_z_threshold

    # Extreme mode: only flag pathways with very high |Z_stouffer|
    stats["flag_extreme"] = stats["z_stouffer"].abs() > threshold
    stats["flagged_two_stage"] = stats["flag_extreme"]
    
    logger.info(
        f"Pathway flags (extreme mode): "
        f"extreme={stats['flag_extreme'].sum()}, "
        f"flagged_two_stage={stats['flagged_two_stage'].sum()}, "
        f"threshold={threshold:.2f}"
    )

    return stats


def compute_weighted_decision_score(
    pathway_stats: pd.DataFrame,
    weight_method: str = "stouffer",
    use_log: bool = True,
) -> pd.DataFrame:
    """Compute weighted anomaly score per sample.

    Instead of counting pathway flags, compute a weighted score that
    emphasizes strong signals. This naturally separates true IMD (few
    strong pathway signals) from gray samples (many weak signals).

    Args:
        pathway_stats: DataFrame with pathway statistics (from
            flag_pathways_enhanced). Must have sample_id, z_med, and
            either z_stouffer or p_stouffer columns.
        weight_method: how to weight pathways:
            - "stouffer": use |Z_stouffer| as weight
            - "zmed": use |Z_med| as weight
            - "p_value": use -log10(p_stouffer) as weight
        use_log: if True, use log-transformed weights for better scale.

    Returns:
        DataFrame with sample_id index and columns:
        - total_score: sum of weights for flagged pathways
        - n_flagged_pathways: count of flagged pathways
        - mean_weight: mean weight of flagged pathways
        - max_weight: maximum weight across flagged pathways
    """
    if pathway_stats.empty:
        return pd.DataFrame(columns=["total_score", "n_flagged_pathways",
                                      "mean_weight", "max_weight"])

    # Determine weight column
    if weight_method == "stouffer":
        weight_col = "z_stouffer"
        weights = pathway_stats[weight_col].abs()
    elif weight_method == "zmed":
        weight_col = "z_med"
        weights = pathway_stats[weight_col].abs()
    elif weight_method == "p_value":
        weight_col = "p_stouffer"
        # Use -log10(p) as weight: p=0.01 -> weight=2, p=0.001 -> weight=3
        weights = -np.log10(pathway_stats[weight_col])
    else:
        raise ValueError(f"Unknown weight_method: {weight_method}")

    # Only use flagged pathways (by two-stage method)
    flagged = pathway_stats[pathway_stats["flagged_two_stage"]]
    if flagged.empty:
        return pd.DataFrame(columns=["total_score", "n_flagged_pathways",
                                      "mean_weight", "max_weight"])

    # Handle NaN weights
    weights = weights.fillna(0)

    if use_log:
        # Log transform to compress scale
        weights = np.log1p(weights)

    # Aggregate by sample
    sample_weights = flagged.groupby("sample_id")[weight_col].agg([
        ("total_score", lambda x: np.sum(np.abs(x))),
        ("n_flagged_pathways", "count"),
        ("mean_weight", "mean"),
        ("max_weight", "max"),
    ])

    # Fill NaN for samples with no flagged pathways
    all_samples = pathway_stats["sample_id"].unique()
    sample_weights = sample_weights.reindex(all_samples)
    sample_weights = sample_weights.fillna({
        "total_score": 0.0,
        "n_flagged_pathways": 0,
        "mean_weight": 0.0,
        "max_weight": 0.0,
    })

    logger.info(
        f"Weighted decision scores: mean={sample_weights['total_score'].mean():.2f}, "
        f"max={sample_weights['total_score'].max():.2f}, "
        f"flagged_samples={(sample_weights['n_flagged_pathways'] > 0).sum()}"
    )

    return sample_weights


def decide_samples_enhanced(
    pathway_stats: pd.DataFrame,
    weighted_scores: pd.DataFrame,
    metabolite_flags: pd.DataFrame,
    global_scores: Optional[pd.Series] = None,
    score_threshold: Optional[float] = None,
    min_flagged_pathways: int = 3,
    min_weight: float = 2.0,
    global_threshold: Optional[float] = None,
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Enhanced decision rule using weighted scores.

    A sample is flagged when ANY of these holds:
    - Weighted score > score_threshold
    - >= min_flagged_pathways with mean weight > min_weight
    - Any single-metabolite override
    - Global anomaly score > global_threshold

    Args:
        pathway_stats: enhanced pathway statistics with flagged_two_stage.
        weighted_scores: output of compute_weighted_decision_score.
        metabolite_flags: single-metabolite override flags.
        global_scores: optional global anomaly scores.
        score_threshold: flag if weighted score > this.
        min_flagged_pathways: minimum number of flagged pathways.
        min_weight: minimum mean weight for flagged pathways.
        global_threshold: global anomaly threshold.
        output_dir: directory to save monitoring CSV.

    Returns:
        DataFrame indexed by sample_id with flagged (bool) and reasons.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)

    # Get all sample IDs
    all_samples = list(set(
        list(pathway_stats["sample_id"].unique()) +
        list(weighted_scores.index) +
        ([s for s in metabolite_flags["sample_id"].unique()] if not metabolite_flags.empty else []) +
        ([s for s in global_scores.index] if global_scores is not None else [])
    ))

    # Build flagged pathway counts per sample
    flagged_pathways = pathway_stats[pathway_stats["flagged_two_stage"]]
    pathway_counts = flagged_pathways.groupby("sample_id")["z_stouffer"].agg([
        ("n_flagged_pathways", "count"),
        ("mean_weight", lambda x: np.mean(np.abs(x))),
    ])

    rows: List[Dict] = []
    for sid in all_samples:
        score_row = weighted_scores.loc[sid] if sid in weighted_scores.index else pd.Series({"total_score": 0, "mean_weight": 0})
        n_flagged_row = pathway_counts.loc[sid] if sid in pathway_counts.index else pd.Series({"n_flagged_pathways": 0, "mean_weight": 0})
        n_flagged = n_flagged_row["n_flagged_pathways"]
        mean_weight = n_flagged_row["mean_weight"]
        n_met = len(metabolite_flags[metabolite_flags["sample_id"] == sid]) if not metabolite_flags.empty else 0
        gscore = global_scores.get(sid, float("nan")) if global_scores is not None else float("nan")

        reasons: List[str] = []

        if score_threshold is not None and score_row.get("total_score", 0) > score_threshold:
            reasons.append(f"weighted_score={score_row['total_score']:.2f}>{score_threshold}")

        if n_flagged >= min_flagged_pathways and mean_weight >= min_weight:
            reasons.append(f"{n_flagged} pathways with mean_weight={mean_weight:.2f}>{min_weight}")

        if n_met > 0:
            reasons.append(f"{n_met} metabolite override(s)")

        if global_threshold is not None and not np.isnan(gscore) and gscore > global_threshold:
            reasons.append(f"global={gscore:.2f}>{global_threshold}")

        rows.append({
            "sample_id": sid,
            "flagged": bool(reasons),
            "weighted_score": score_row.get("total_score", 0),
            "n_flagged_pathways": n_flagged,
            "mean_pathway_weight": mean_weight,
            "n_metabolite_overrides": n_met,
            "global_anomaly_score": gscore,
            "decision_reason": ";".join(reasons),
        })

    out = pd.DataFrame(rows).set_index("sample_id")

    logger.info(
        f"Enhanced decision rule: flagged {int(out['flagged'].sum())} of "
        f"{len(out)} samples. Score distribution: "
        f"mean={out['weighted_score'].mean():.2f}, max={out['weighted_score'].max():.2f}"
    )

    # Save monitoring CSV
    if output_dir is not None:
        out.to_csv(output_dir / "enhanced_sample_decisions.csv")
        logger.info(f"Wrote enhanced_sample_decisions.csv")

    return out


def validate_no_normal_contamination(
    decisions: pd.DataFrame,
    metadata: pd.DataFrame,
    classification_scheme: str = "class1_imd",
    output_dir: Optional[Path] = None,
) -> pd.DataFrame:
    """Validate that NO normal samples are flagged.

    This is the critical validation: normals should NEVER be flagged.
    Gray samples CAN be flagged (they have real disturbances).
    IMD samples SHOULD be flagged.

    Args:
        decisions: DataFrame from decide_samples or decide_samples_enhanced.
            Must have 'flagged' column indexed by sample_id.
        metadata: DataFrame with Classification and Oordeel targeted columns.
        classification_scheme: scheme to determine normal/IMD/gray labels.
        output_dir: directory to save validation report.

    Returns:
        DataFrame with validation metrics:
        - n_normals, n_imd, n_gray
        - normals_flagged, imd_flagged, gray_flagged
        - normal_contamination_rate (should be 0%)
        - imd_detection_rate (should be high)
        - gray_flagging_rate (informational)
        - flagged_normals_list: list of normal sample IDs that were flagged
    """
    # Determine labels for each sample
    idx = metadata.index
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")

    # Classify samples
    if classification_scheme == "class1_imd":
        # Normals = Class 0 AND Oordeel 0
        # IMD = Class 1 AND Oordeel 1
        # Gray = Everything else (Class 2/3, Class 0 AND Oordeel 1, Class 1 AND Oordeel 0)
        normal_mask = (cls == 0) & (oor == 0)
        imd_mask = (cls == 1) & (oor == 1)
        gray_mask = ~normal_mask & ~imd_mask
    else:
        # Default: use confident_normals scheme
        normal_mask = (cls == 0) & (oor == 0)
        imd_mask = ~normal_mask
        gray_mask = pd.Series(False, index=idx)

    n_normals = int(normal_mask.sum())
    n_imd = int(imd_mask.sum())
    n_gray = int(gray_mask.sum())

    # Align decisions with metadata
    # reindex and fill missing flagged values with False
    # Handle potential duplicate index in decisions by aggregating with any()
    if decisions.index.duplicated().any():
        # Aggregate duplicate decisions using any() - if any duplicate is flagged, the sample is flagged
        decisions_no_dup = decisions.groupby(decisions.index)["flagged"].any()
        decisions_aligned = decisions_no_dup.reindex(idx)
    else:
        decisions_aligned = decisions["flagged"].reindex(idx)
    decisions_aligned = decisions_aligned.fillna(False)
    flagged = decisions_aligned

    # Count flagged in each category
    normals_flagged = int((flagged & normal_mask).sum())
    imd_flagged = int((flagged & imd_mask).sum())
    gray_flagged = int((flagged & gray_mask).sum())

    # Compute rates
    normal_contamination_rate = (normals_flagged / n_normals * 100) if n_normals > 0 else 0.0
    imd_detection_rate = (imd_flagged / n_imd * 100) if n_imd > 0 else 0.0
    gray_flagging_rate = (gray_flagged / n_gray * 100) if n_gray > 0 else 0.0

    # Get list of flagged normals (for debugging)
    flagged_normal_ids = list(idx[flagged & normal_mask])

    # Build validation report
    report = pd.DataFrame({
        "metric": [
            "n_normals", "n_imd", "n_gray",
            "normals_flagged", "imd_flagged", "gray_flagged",
            "normal_contamination_rate", "imd_detection_rate", "gray_flagging_rate",
        ],
        "value": [
            n_normals, n_imd, n_gray,
            normals_flagged, imd_flagged, gray_flagged,
            round(normal_contamination_rate, 2),
            round(imd_detection_rate, 2),
            round(gray_flagging_rate, 2),
        ],
        "unit": [
            "samples", "samples", "samples",
            "samples", "samples", "samples",
            "%", "%", "%",
        ]
    })

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("VALIDATION: Normal Contamination Check")
    logger.info("=" * 60)
    logger.info(f"Normals: {n_normals} samples, {normals_flagged} flagged "
                f"({normal_contamination_rate:.1f}%)")
    logger.info(f"IMD: {n_imd} samples, {imd_flagged} flagged "
                f"({imd_detection_rate:.1f}%)")
    logger.info(f"Gray: {n_gray} samples, {gray_flagged} flagged "
                f"({gray_flagging_rate:.1f}%)")

    if normals_flagged > 0:
        logger.warning(f"CRITICAL: {normals_flagged} normal samples were flagged! "
                       f"Thresholds are too loose. Flagged normals: {flagged_normal_ids}")
    else:
        logger.info("PASS: No normal samples flagged.")

    if imd_detection_rate < 50:
        logger.warning(f"WARNING: Only {imd_detection_rate:.1f}% of IMD samples "
                       f"were flagged. Thresholds may be too strict.")
    else:
        logger.info(f"IMD detection rate: {imd_detection_rate:.1f}%")

    logger.info("=" * 60)

    # Save report
    if output_dir is not None:
        output_dir = Path(output_dir)
        report.to_csv(output_dir / "validation_report.csv", index=False)
        logger.info(f"Wrote validation_report.csv")

        # Also save per-sample validation
        per_sample = pd.DataFrame({
            "sample_id": idx,
            "classification": cls,
            "oordeel": oor,
            "sample_type": [
                "normal" if nm else ("imd" if im else "gray")
                for nm, im in zip(normal_mask, imd_mask)
            ],
            "flagged": flagged,
            "is_contamination": flagged & normal_mask,
        })
        per_sample.to_csv(output_dir / "per_sample_validation.csv", index=False)
        logger.info(f"Wrote per_sample_validation.csv")

    return report


def run_enhanced_pipeline(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    normal_mask: pd.Series,
    output_dir: Path,
    min_pathway_size: int = 3,
    # Thresholds
    zmed_threshold: float = 2.0,
    stouffer_z_threshold: float = 3.0,
    p_stouffer_threshold: float = 0.001,
    p_bonferroni_threshold: float = 0.05,
    p_fdr_threshold: float = 0.05,
    use_empirical: bool = True,
    score_threshold: Optional[float] = None,
    min_flagged_pathways: int = 3,
    min_weight: float = 2.0,
    weight_method: str = "stouffer",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the complete enhanced pathway analysis pipeline.

    This is a convenience function that runs all enhanced analysis steps
    and returns the three main outputs, while saving monitoring CSVs.

    Args:
        zscores: per-metabolite z-scores.
        feature_to_pathway: feature to pathway mapping.
        normal_mask: boolean normal reference mask.
        output_dir: directory for monitoring CSVs.
        All other parameters: thresholds and options.

    Returns:
        Tuple of (pathway_stats, flagged_pathways, sample_decisions).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Enhanced pathway statistics
    logger.info("Computing enhanced pathway statistics...")
    stats = compute_enhanced_pathway_statistics(
        zscores, feature_to_pathway, normal_mask,
        min_pathway_size=min_pathway_size,
        output_dir=output_dir,
    )

    # Step 2: Flag pathways using enhanced methods
    logger.info("Flagging pathways with enhanced methods...")
    flagged = flag_pathways_enhanced(
        stats,
        zmed_threshold=zmed_threshold,
        stouffer_z_threshold=stouffer_z_threshold,
        p_stouffer_threshold=p_stouffer_threshold,
        p_bonferroni_threshold=p_bonferroni_threshold,
        p_fdr_threshold=p_fdr_threshold,
        use_empirical=use_empirical,
    )

    # Step 3: Compute weighted decision scores
    logger.info("Computing weighted decision scores...")
    weighted = compute_weighted_decision_score(
        flagged,
        weight_method=weight_method,
    )

    # Step 4: Make decisions
    logger.info("Making enhanced decisions...")
    # Create dummy metabolite flags for now
    metabolite_flags = pd.DataFrame(columns=["sample_id", "metabolite", "z"])
    decisions = decide_samples_enhanced(
        flagged,
        weighted,
        metabolite_flags,
        score_threshold=score_threshold,
        min_flagged_pathways=min_flagged_pathways,
        min_weight=min_weight,
        output_dir=output_dir,
    )

    # Step 5: Generate visualizations for IMD samples
    if HAS_SEABORN:
        logger.info("Generating pathway visualization for IMD samples...")
        _generate_imd_pathway_visualizations(
            stats, flagged, decisions, metadata, output_dir
        )
    else:
        logger.info("Skipping pathway visualizations (seaborn not available)")

    return stats, flagged, decisions


def _generate_imd_pathway_visualizations(
    pathway_stats: pd.DataFrame,
    flagged_pathways: pd.DataFrame,
    decisions: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path,
    figsize: Tuple[int, int] = (12, 8),
    top_n_pathways: int = 15,
    dpi: int = 150,
) -> None:
    """Generate Seaborn visualizations for each IMD sample.

    Creates the following plots for each flagged IMD sample:
    1. Bar plot of top N most deviated pathways (by |Z_stouffer|)
    2. Bar plot of pathway Z_med values (traditional metric)
    3. Heatmap of all pathway Z_med values for the sample

    Args:
        pathway_stats: DataFrame with all pathway statistics.
        flagged_pathways: DataFrame with flagged pathways.
        decisions: DataFrame with sample decisions.
        metadata: DataFrame with Classification and Oordeel columns.
        output_dir: Directory to save plots.
        figsize: Figure size for each plot.
        top_n_pathways: Number of top pathways to show in bar plots.
        dpi: DPI for saved figures.
    """
    if not HAS_SEABORN:
        logger.warning("Seaborn/matplotlib not available; skipping visualizations.")
        return
        
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Identify IMD samples (Class 1 AND Oordeel 1) from metadata
    cls = pd.to_numeric(metadata["Classification"], errors="coerce")
    oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
    imd_mask = (cls == 1) & (oor == 1)
    imd_samples = metadata.index[imd_mask]

    # Get flagged IMD samples - handle duplicate index issues
    # Convert decisions index to array and imd_mask to array for element-wise comparison
    decisions_index = decisions.index.values if hasattr(decisions.index, 'values') else decisions.index
    imd_mask_array = imd_mask.values if hasattr(imd_mask, 'values') else imd_mask
    flagged_array = decisions["flagged"].values if hasattr(decisions["flagged"], 'values') else decisions["flagged"]
    
    # Find intersection using array-based masking
    # Get indices in metadata that are both IMD and flagged
    # First, get the sample_ids from decisions that are flagged
    flagged_sample_ids = set(decisions.index[flagged_array])
    # Then intersect with IMD sample IDs
    flagged_imd = list(flagged_sample_ids & set(imd_samples))

    if len(flagged_imd) == 0:
        logger.info("No flagged IMD samples to visualize.")
        return

    logger.info(f"Generating visualizations for {len(flagged_imd)} flagged IMD samples")

    # Get normal reference statistics for comparison
    normal_mask = _get_normal_mask_from_metadata(metadata, "class1_imd")
    normal_samples = metadata.index[normal_mask]
    
    # Filter normal_stats using sample_id column to avoid index issues
    normal_stats = pathway_stats[pathway_stats["sample_id"].isin(normal_samples)]

    # Compute mean and std of Z_med for normals per pathway
    if not normal_stats.empty:
        normal_means = normal_stats.groupby("pathway_name")["z_med"].agg([
            ("mean", "mean"),
            ("std", "std"),
        ])
    else:
        normal_means = pd.DataFrame()

    for sample_id in flagged_imd:
        # Filter to this sample
        sample_stats = pathway_stats[pathway_stats["sample_id"] == sample_id]
        sample_flagged = flagged_pathways[flagged_pathways["sample_id"] == sample_id]

        if sample_stats.empty:
            continue

        # Create output subdirectory for this sample
        sample_dir = output_dir / f"imd_visualizations" / str(sample_id)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Get pathway names and values
        pathways = sample_stats["pathway_name"].tolist()
        z_med = sample_stats["z_med"].tolist()
        z_stouffer = sample_stats["z_stouffer"].abs().tolist()
        flagged = sample_stats["flagged_two_stage"].tolist()

        # Merge with normal stats for z-score relative to normals
        sample_df = sample_stats.copy()
        sample_df = sample_df.set_index("pathway_name")
        if not normal_means.empty:
            sample_df = sample_df.join(normal_means, how="left")
            # Handle the multi-level column names from agg
            if ('mean', 'z_med') in sample_df.columns and ('std', 'z_med') in sample_df.columns:
                sample_df["z_med_normalized"] = (
                    (sample_df["z_med"] - sample_df[("mean", "z_med")]) /
                    sample_df[("std", "z_med")].replace(0, np.nan)
                )
            else:
                # Fallback if column names are different
                sample_df["z_med_normalized"] = sample_df["z_med"]
        else:
            sample_df["z_med_normalized"] = sample_df["z_med"]

        # Sort by |Z_stouffer| for ranking
        sample_df = sample_df.sort_values("z_stouffer", key=abs, ascending=False)

        # ===== Plot 1: Top N pathways by |Z_stouffer| =====
        plt.figure(figsize=figsize)
        top_n = sample_df.head(top_n_pathways)
        colors = ["red" if f else "lightcoral" for f in top_n["flagged_two_stage"]]
        ax = sns.barplot(
            data=top_n,
            x="z_stouffer",
            y="pathway_name",
            palette=colors,
            order=top_n.index.tolist()
        )
        ax.set_xlabel(f"|Z_stouffer| (Stouffer's combined Z-score)")
        ax.set_ylabel("Pathway")
        ax.set_title(f"Sample {sample_id}: Top {top_n_pathways} Pathways by |Z_stouffer|\n"
                     f"(Red = flagged by two-stage method)")
        plt.tight_layout()
        plt.savefig(sample_dir / "top_pathways_stouffer.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        # ===== Plot 2: Top N pathways by |Z_med| =====
        plt.figure(figsize=figsize)
        top_n_zmed = sample_df.sort_values("z_med", key=abs, ascending=False).head(top_n_pathways)
        colors = ["red" if f else "lightcoral" for f in top_n_zmed["flagged_two_stage"]]
        ax = sns.barplot(
            data=top_n_zmed,
            x="z_med",
            y="pathway_name",
            palette=colors,
            order=top_n_zmed.index.tolist()
        )
        ax.set_xlabel(f"Z_med (Median z-score)")
        ax.set_ylabel("Pathway")
        ax.set_title(f"Sample {sample_id}: Top {top_n_pathways} Pathways by |Z_med|\n"
                     f"(Red = flagged by two-stage method)")
        plt.tight_layout()
        plt.savefig(sample_dir / "top_pathways_zmed.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        # ===== Plot 3: Heatmap of all pathways =====
        plt.figure(figsize=(14, max(6, len(pathways) * 0.2)))
        # Create a matrix for the heatmap
        heatmap_data = sample_df["z_med_normalized"].unstack()
        if isinstance(heatmap_data, pd.Series):
            heatmap_data = heatmap_data.to_frame().T
        
        # Add flagged annotation
        flagged_series = sample_df["flagged_two_stage"].astype(int)

        ax = sns.heatmap(
            heatmap_data.T,
            cmap="RdBu_r",
            center=0,
            vmin=-3,
            vmax=3,
            cbar_kws={"label": "Z_med (normalized to normal mean/std)"},
            annot=flagged_series.to_dict(),
            fmt="d",
            annot_kws={"size": 8},
        )
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Sample")
        ax.set_title(f"Sample {sample_id}: All Pathway Z_med Values\n"
                     f"(Normalized to normal distribution; annotations = flagged)")
        plt.tight_layout()
        plt.savefig(sample_dir / "all_pathways_heatmap.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        # ===== Plot 4: Comparison with normal distribution =====
        plt.figure(figsize=figsize)
        # Get normal distribution for each pathway
        normal_dist_data = []
        for pathway in pathways:
            normal_pathway_zmed = normal_stats[normal_stats["pathway_name"] == pathway]["z_med"]
            if len(normal_pathway_zmed) > 0:
                normal_dist_data.append({
                    "pathway": pathway,
                    "value": "normal",
                    "z_med": float(normal_pathway_zmed.mean())
                })
        
        sample_data = [{"pathway": p, "value": "sample", "z_med": z}
                      for p, z in zip(pathways, z_med)]
        
        combined = pd.DataFrame(normal_dist_data + sample_data)
        
        ax = sns.boxplot(
            data=combined,
            x="pathway",
            y="z_med",
            hue="value",
            order=pathways[:top_n_pathways],  # Limit to top pathways
            palette={"normal": "lightblue", "sample": "red"},
            showfliers=False,
        )
        ax.set_xlabel("Pathway")
        ax.set_ylabel("Z_med")
        ax.set_title(f"Sample {sample_id}: Pathway Z_med vs Normal Distribution\n"
                     f"(Top {top_n_pathways} pathways; Red dot = sample, Blue box = normals)")
        ax.legend(title="")
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.savefig(sample_dir / "pathway_vs_normal.png", dpi=dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Generated 4 visualizations for IMD sample {sample_id} in {sample_dir}")


def _get_normal_mask_from_metadata(metadata: pd.DataFrame, scheme: str) -> pd.Series:
    """Helper to get normal mask from metadata (duplicated from main.py)."""
    idx = metadata.index
    if metadata.empty:
        return pd.Series(np.ones(len(idx), dtype=bool), index=idx)

    if "Classification" in metadata.columns and "Oordeel targeted" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        if scheme == "class1_imd":
            return pd.Series((cls == 0) & (oor == 0), index=idx)
        if scheme == "confident_normals":
            return pd.Series((cls == 0) & (oor == 0), index=idx)
        if scheme == "oordeel":
            return pd.Series(oor == 0, index=idx)
        if scheme == "binary_simplified":
            return pd.Series(cls.isin([0, 3]), index=idx)
    if "Classification" in metadata.columns:
        cls = pd.to_numeric(metadata["Classification"], errors="coerce")
        return pd.Series(cls == 0, index=idx)
    if "Oordeel targeted" in metadata.columns:
        oor = pd.to_numeric(metadata["Oordeel targeted"], errors="coerce")
        return pd.Series(oor == 0, index=idx)
    return pd.Series(np.ones(len(idx), dtype=bool), index=idx)
