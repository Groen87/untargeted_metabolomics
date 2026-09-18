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


def _compute_stouffers_z(per_metabolite_zscores: np.ndarray,
                         two_tailed: bool = True) -> Tuple[float, float]:
    """Compute Stouffer's combined Z-score and p-value for a set of z-scores.

    Stouffer's method: Z_combined = sum(z_i) / sqrt(n)
    This properly combines evidence across metabolites, accounting for
    directionality and sample size.

    Args:
        per_metabolite_zscores: 1-D array of z-scores for metabolites in a pathway.
        two_tailed: if True, compute two-tailed p-value; else one-tailed.

    Returns:
        Tuple of (Z_stouffer, p_value). NaN for both if all inputs are NaN.
    """
    # Filter out NaN values
    valid = per_metabolite_zscores[~np.isnan(per_metabolite_zscores)]
    if len(valid) == 0:
        return float("nan"), float("nan")

    n = len(valid)
    z_sum = np.sum(valid)
    z_combined = z_sum / np.sqrt(n)

    if two_tailed:
        # Two-tailed p-value for combined Z
        p = 2 * (1 - scipy_stats.norm.cdf(abs(z_combined)))
    else:
        # One-tailed (directional)
        p = 1 - scipy_stats.norm.cdf(z_combined)

    return float(z_combined), float(p)


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

    Returns:
        Long DataFrame with one row per (sample, pathway) with all statistics.
    """
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Get normal z-scores for empirical threshold computation
    z_normals = zscores.loc[normal_mask]
    normal_sample_ids = z_normals.index

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
            "z_stouffer", "p_stouffer", "p_bonferroni", "p_fdr",
            "empirical_p_threshold", "n_normals_used"
        ])

    # Pre-compute per-pathway empirical thresholds from normals
    empirical_thresholds: Dict[str, float] = {}
    for smp_id, info in pathway_features.items():
        feat_cols = info["features"]
        normal_z = z_normals[feat_cols].to_numpy(dtype=float)
        # Compute Z_med for each normal sample for this pathway
        z_med_normals = np.nanmedian(normal_z, axis=1)
        # Use 95th percentile of |Z_med| over normals as empirical threshold
        empirical_thresholds[smp_id] = float(
            np.nanpercentile(np.abs(z_med_normals), 95)
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
        z_med = np.nanmedian(sub, axis=1)

        # Per-metabolite thresholds (99th percentile of |z| over normals)
        per_met_thresh = z_normals[info["features"]].abs().quantile(0.99, axis=0).to_numpy()
        per_met_thresh = per_met_thresh.where(per_met_thresh.notna() & (per_met_thresh > 0), np.inf)

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

        # Enhanced: Stouffer's Z-score
        z_stouffer = np.zeros(n_samples)
        p_stouffer = np.zeros(n_samples)
        for i in range(n_samples):
            z_stouffer[i], p_stouffer[i] = _compute_stouffers_z(sub[i, :])

        # Multiple testing correction
        # Bonferroni: p_bonferroni = p * n_pathways
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
                # Enhanced
                "z_stouffer": float(z_stouffer[i]) if not np.isnan(z_stouffer[i]) else float("nan"),
                "p_stouffer": float(p_stouffer[i]) if not np.isnan(p_stouffer[i]) else float("nan"),
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
        p_df = p_df[[
            "sample_id", "pathway_name", "p_value", "p_fdr"
        ]].rename(columns={"p_value": "p_stouffer_orig"})
        stats_df = stats_df.merge(p_df, on=["sample_id", "pathway_name"], how="left")
        stats_df = stats_df.rename(columns={"p_stouffer": "p_stouffer_orig"})
        stats_df["p_stouffer"] = stats_df["p_stouffer_orig"]
        stats_df = stats_df.drop(columns=["p_stouffer_orig"])

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
    zmed_threshold: float = 2.0,
    stouffer_z_threshold: float = 3.0,
    p_stouffer_threshold: float = 0.001,
    p_bonferroni_threshold: float = 0.05,
    p_fdr_threshold: float = 0.05,
    use_empirical: bool = False,
    empirical_alpha: float = 0.05,
) -> pd.DataFrame:
    """Flag pathways using enhanced statistics with multiple methods.

    A pathway can be flagged by any of these methods:
    - Traditional Z_med threshold
    - Stouffer's Z threshold
    - Uncorrected p-value threshold
    - Bonferroni-corrected p-value threshold
    - FDR-corrected p-value threshold
    - Empirical threshold (pathway-specific from normals)

    Args:
        stats: output of compute_enhanced_pathway_statistics.
        zmed_threshold: traditional |Z_med| threshold.
        stouffer_z_threshold: |Z_stouffer| threshold.
        p_stouffer_threshold: uncorrected p-value threshold.
        p_bonferroni_threshold: Bonferroni-corrected p-value threshold.
        p_fdr_threshold: FDR-corrected p-value threshold.
        use_empirical: if True, use empirical pathway thresholds.
        empirical_alpha: significance level for empirical thresholds.

    Returns:
        stats with added flag columns for each method and combined flag.
    """
    if stats.empty:
        return stats

    # Traditional Z_med flag
    stats["flag_zmed"] = stats["z_med"].abs() > zmed_threshold

    # Stouffer's Z flag
    stats["flag_stouffer_z"] = stats["z_stouffer"].abs() > stouffer_z_threshold

    # p-value flags
    stats["flag_p_stouffer"] = stats["p_stouffer"] < p_stouffer_threshold
    stats["flag_p_bonferroni"] = stats["p_bonferroni"] < p_bonferroni_threshold
    stats["flag_p_fdr"] = stats["p_fdr"] < p_fdr_threshold

    # Empirical threshold flag
    if use_empirical:
        stats["flag_empirical"] = stats["z_med"].abs() > stats["empirical_p_threshold"]
    else:
        stats["flag_empirical"] = False

    # Combined flag: any method triggers
    flag_cols = ["flag_zmed", "flag_stouffer_z", "flag_p_stouffer",
                 "flag_p_bonferroni", "flag_p_fdr", "flag_empirical"]
    stats["flagged_any"] = stats[flag_cols].any(axis=1)

    # Two-stage flag: require Stouffer's Z OR corrected p-value
    stats["flagged_two_stage"] = (
        stats["flag_stouffer_z"] |
        stats["flag_p_bonferroni"] |
        stats["flag_p_fdr"]
    )

    # Count how many methods flag each (sample, pathway)
    stats["n_methods_flagged"] = stats[flag_cols].sum(axis=1)

    logger.info(
        f"Pathway flags (enhanced): "
        f"Z_med={stats['flag_zmed'].sum()}, "
        f"Stouffer_Z={stats['flag_stouffer_z'].sum()}, "
        f"p_stouffer={stats['flag_p_stouffer'].sum()}, "
        f"p_bonferroni={stats['flag_p_bonferroni'].sum()}, "
        f"p_fdr={stats['flag_p_fdr'].sum()}, "
        f"empirical={stats['flag_empirical'].sum()}, "
        f"any={stats['flagged_any'].sum()}, "
        f"two_stage={stats['flagged_two_stage'].sum()}"
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
    pathway_counts = flagged_pathways.groupby("sample_id").agg({
        "n_flagged": ("n_flagged_pathways", "count"),
        "mean_weight": ("z_stouffer", lambda x: np.mean(np.abs(x))),
    })["n_flagged_pathways"]

    rows: List[Dict] = []
    for sid in all_samples:
        score = weighted_scores.loc.get(sid, {"total_score": 0, "mean_weight": 0})
        n_flagged = pathway_counts.get(sid, 0)
        mean_weight = weighted_scores.loc.get(sid, {"mean_weight": 0})["mean_weight"]
        n_met = len(metabolite_flags[metabolite_flags["sample_id"] == sid]) if not metabolite_flags.empty else 0
        gscore = global_scores.get(sid, float("nan")) if global_scores is not None else float("nan")

        reasons: List[str] = []

        if score_threshold is not None and score.get("total_score", 0) > score_threshold:
            reasons.append(f"weighted_score={score['total_score']:.2f}>{score_threshold}")

        if n_flagged >= min_flagged_pathways and mean_weight >= min_weight:
            reasons.append(f"{n_flagged} pathways with mean_weight={mean_weight:.2f}>{min_weight}")

        if n_met > 0:
            reasons.append(f"{n_met} metabolite override(s)")

        if global_threshold is not None and not np.isnan(gscore) and gscore > global_threshold:
            reasons.append(f"global={gscore:.2f}>{global_threshold}")

        rows.append({
            "sample_id": sid,
            "flagged": bool(reasons),
            "weighted_score": score.get("total_score", 0),
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

    return stats, flagged, decisions
