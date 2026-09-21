"""
Enhanced per-sample pathway flagging for the pathway pipeline.

Implements a three-statistic flagging scheme, each calibrated on the
reference controls (normals):

1. **Correlation-adjusted Stouffer** (global disturbance flag): directional
   Stouffer's Z that divides by sqrt(1' Sigma 1), where Sigma is the
   pathway feature correlation matrix estimated from the controls
   (Ledoit-Wolf shrinkage). Adjusting for correlation prevents a block of
   co-regulated metabolites from inflating the combined score the way the
   independence assumption does.

2. **Top-k max-z with empirical null** (localized block flag): the mean of
   the k largest |z| in a pathway, p-valued against an empirical null
   obtained by bootstrapping the controls' own top-k statistics. This
   catches a single disturbed block (few extreme metabolites) that the
   global Stouffer dilutes across all pathway features.

3. **Substrate:product ratio z-scores** (block-specific flag): robust
   (median/IQR) z-score of the log sum-ratio against controls, for each
   configured ratio (e.g. propionylcarnitine : acetylcarnitine). Ratios are
   the most direct biochemical signature of a blocked enzyme.

A sample is flagged when any pathway's combined p-value survives
Benjamini-Hochberg FDR control across the tested pathways at the target
FDR level. Inputs are the robust covariate-matched z-scores from
``compute_metabolite_zscores`` (age-adjusted residuals, median/IQR scaled
on the normal reference).
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

logger = logging.getLogger(__name__)


def _bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg adjusted p-values (q-values) for a 1-D array.

    Args:
        pvals: 1-D array of p-values (may contain NaN, which are kept NaN).

    Returns:
        Array of BH-adjusted q-values, same order/length as the input.
    """
    pvals = np.asarray(pvals, dtype=float)
    q = np.full_like(pvals, np.nan)
    valid = ~np.isnan(pvals)
    m = int(valid.sum())
    if m == 0:
        return q
    order = np.argsort(pvals[valid])
    sorted_p = pvals[valid][order]
    sorted_q = np.minimum.accumulate((sorted_p * m / np.arange(1, m + 1))[::-1])[::-1]
    sorted_q = np.clip(sorted_q, 0.0, 1.0)
    q[valid] = sorted_q[np.argsort(order)]
    return q


def _build_pathway_features(
    feature_to_pathway: pd.DataFrame,
    available_features: List[str],
    min_pathway_size: int = 3,
) -> Dict[str, List[str]]:
    """Map pathway_name -> list of features present in the z-score matrix."""
    pathway_features: Dict[str, List[str]] = {}
    feat_set = set(available_features)
    for _, row in feature_to_pathway.iterrows():
        pathway = row["pathway_name"]
        feature = str(row["feature"])
        if feature in feat_set:
            pathway_features.setdefault(pathway, [])
            if feature not in pathway_features[pathway]:
                pathway_features[pathway].append(feature)
    too_small = [p for p, f in pathway_features.items() if len(f) < min_pathway_size]
    pathway_features = {p: f for p, f in pathway_features.items()
                        if len(f) >= min_pathway_size}
    if too_small:
        logger.info(f"Enhanced flagging: skipping {len(too_small)} pathways with "
                    f"< {min_pathway_size} matched features")
    return pathway_features


def compute_correlation_adjusted_stouffer(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    normal_mask: pd.Series,
    min_pathway_size: int = 3,
) -> pd.DataFrame:
    """
    Per (sample, pathway): directional Stouffer's Z adjusted for the
    correlation structure estimated on the controls.

    T = 1'z / sqrt(1' Sigma 1), Sigma from Ledoit-Wolf shrinkage of the
    controls' pathway z-scores (falls back to the identity when there are
    too few controls). p-value is two-sided standard normal (global
    disturbance in either direction).
    """
    from sklearn.covariance import LedoitWolf

    pathway_features = _build_pathway_features(
        feature_to_pathway, list(zscores.columns), min_pathway_size
    )
    if not pathway_features:
        return pd.DataFrame(columns=["sample_id", "pathway_name",
                                      "z_stouffer_corr", "p_stouffer"])

    ctrl = zscores.loc[normal_mask]

    # Pre-fit Sigma per pathway on controls
    sigmas: Dict[str, Optional[np.ndarray]] = {}
    for pathway, feats in pathway_features.items():
        X = ctrl[feats].dropna(axis=0, how="any").to_numpy(dtype=float)
        k = len(feats)
        if len(X) >= k + 2:
            try:
                sigmas[pathway] = LedoitWolf().fit(X).covariance_
            except Exception as e:  # pragma: no cover - degenerate covariance
                logger.debug(f"Ledoit-Wolf failed for {pathway}: {e}; using independence")
                sigmas[pathway] = np.eye(k)
        else:
            sigmas[pathway] = np.eye(k)

    ones_cache: Dict[int, np.ndarray] = {}

    rows = []
    for pathway, feats in pathway_features.items():
        k = len(feats)
        Sigma = sigmas[pathway]
        if k not in ones_cache:
            ones_cache[k] = np.ones(k)
        ones = ones_cache[k]
        denom = np.sqrt(float(ones @ Sigma @ ones))
        if denom <= 0:
            denom = np.sqrt(k)

        Z = zscores[feats].to_numpy(dtype=float)
        # Row-wise valid-count; only score rows with no NaN (impute 0 for
        # isolated NaNs so a single missing feature does not void the test).
        nan_rows = np.isnan(Z).any(axis=1)
        Z = np.nan_to_num(Z, nan=0.0)
        T = (Z @ ones) / denom
        p = 2.0 * norm.sf(np.abs(T))
        for i, sample_id in enumerate(zscores.index):
            rows.append({
                "sample_id": sample_id,
                "pathway_name": pathway,
                "z_stouffer_corr": float(T[i]),
                "p_stouffer": float(p[i]) if not nan_rows[i] else np.nan,
                "n_features": k,
            })

    return pd.DataFrame(rows)


def compute_topk_statistic(
    zscores: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    normal_mask: pd.Series,
    k: int = 3,
    n_boot: int = 1000,
    random_state: int = 42,
    min_pathway_size: int = 3,
) -> pd.DataFrame:
    """
    Per (sample, pathway): mean of the k largest |z| (top-k statistic) with
    a p-value from an empirical null built by bootstrapping the controls'
    own top-k statistics.

    Also records the signed directional max-z (the most extreme metabolite
    with its sign) so the flagged block's direction is inspectable.
    """
    pathway_features = _build_pathway_features(
        feature_to_pathway, list(zscores.columns), min_pathway_size
    )
    if not pathway_features:
        return pd.DataFrame(columns=["sample_id", "pathway_name",
                                      "topk_stat", "p_topk", "max_z"])

    ctrl = zscores.loc[normal_mask]
    rng = np.random.default_rng(random_state)

    def topk_of(matrix: np.ndarray, k_eff: int) -> np.ndarray:
        """Row-wise mean of the k_eff largest absolute values."""
        A = np.abs(matrix)
        if k_eff < A.shape[1]:
            A = np.partition(A, A.shape[1] - k_eff, axis=1)[:, -k_eff:]
        return A.mean(axis=1)

    rows = []
    for pathway, feats in pathway_features.items():
        k_eff = min(k, len(feats))
        Z = zscores[feats].to_numpy(dtype=float)
        stat = topk_of(np.nan_to_num(Z, nan=0.0), k_eff)

        # Empirical null: bootstrap resamples of the controls' top-k stats.
        ctrl_Z = ctrl[feats].to_numpy(dtype=float)
        ctrl_stat = topk_of(np.nan_to_num(ctrl_Z, nan=0.0), k_eff)
        ctrl_stat = ctrl_stat[np.isfinite(ctrl_stat)]
        if len(ctrl_stat) == 0:
            continue
        null_pool = rng.choice(ctrl_stat, size=n_boot, replace=True)
        # add-1 smoothing so p is never 0
        p = (1.0 + np.array([np.sum(null_pool >= s) for s in stat])) / (n_boot + 1.0)

        # Signed directional max-z (for interpretation only)
        with np.errstate(invalid="ignore"):
            max_idx = np.nanargmax(np.abs(np.nan_to_num(Z, nan=0.0)), axis=1)
            max_z = Z[np.arange(Z.shape[0]), max_idx]

        for i, sample_id in enumerate(zscores.index):
            rows.append({
                "sample_id": sample_id,
                "pathway_name": pathway,
                "topk_stat": float(stat[i]),
                "p_topk": float(p[i]),
                "max_z": float(max_z[i]),
                "n_features": len(feats),
            })

    return pd.DataFrame(rows)


def compute_ratio_zscores(
    features: pd.DataFrame,
    normal_mask: pd.Series,
    ratio_specs: List[Dict],
    iqr_scale: bool = True,
) -> pd.DataFrame:
    """
    Robust z-score of log10 substrate:product sum-ratios against controls.

    Each spec: {name, numerator: [features], denominator: [features],
    pathway: optional pathway_name to attach the ratio to}. Values are
    log10(sum(numerator) / sum(denominator)) on the RAW feature matrix
    (not z-scores), robustly scaled (median/IQR) on the normal reference.
    """
    rows = []
    if not ratio_specs:
        return pd.DataFrame(columns=["sample_id", "ratio_name",
                                     "ratio_z", "p_ratio", "pathway_name"])

    ctrl = features.loc[normal_mask]
    for spec in ratio_specs:
        name = spec.get("name")
        num = spec.get("numerator", [])
        den = spec.get("denominator", [])
        pathway = spec.get("pathway")
        if not name or not num or not den:
            logger.warning(f"Ratio spec missing name/numerator/denominator: {spec}")
            continue
        num_feats = [f for f in num if f in features.columns]
        den_feats = [f for f in den if f in features.columns]
        missing = (set(num) | set(den)) - set(num_feats) - set(den_feats)
        if missing:
            logger.warning(f"Ratio '{name}': missing features {sorted(missing)}; skipped")
            continue
        if not num_feats or not den_feats:
            continue

        num_sum = features[num_feats].to_numpy(dtype=float).sum(axis=1)
        den_sum = features[den_feats].to_numpy(dtype=float).sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_ratio = np.log10(num_sum) - np.log10(den_sum)
        log_ratio = np.where(np.isfinite(log_ratio), log_ratio, np.nan)

        ctrl_vals = log_ratio[np.asarray(normal_mask)]
        ctrl_vals = ctrl_vals[np.isfinite(ctrl_vals)]
        if len(ctrl_vals) < 5:
            logger.warning(f"Ratio '{name}': <5 control values; skipped")
            continue
        median = np.median(ctrl_vals)
        if iqr_scale:
            scale = np.percentile(ctrl_vals, 75) - np.percentile(ctrl_vals, 25)
        else:
            scale = np.std(ctrl_vals)
        if not np.isfinite(scale) or scale <= 0:
            logger.warning(f"Ratio '{name}': degenerate scale; skipped")
            continue
        z = (log_ratio - median) / scale
        p = 2.0 * norm.sf(np.abs(z))

        for i, sample_id in enumerate(features.index):
            rows.append({
                "sample_id": sample_id,
                "pathway_name": pathway if pathway else name,
                "ratio_name": name,
                "ratio_z": float(z[i]) if np.isfinite(z[i]) else np.nan,
                "p_ratio": float(p[i]) if np.isfinite(z[i]) else np.nan,
            })

    return pd.DataFrame(rows)


def flag_samples_enhanced(
    zscores: pd.DataFrame,
    features: pd.DataFrame,
    feature_to_pathway: pd.DataFrame,
    normal_mask: pd.Series,
    ratio_specs: Optional[List[Dict]] = None,
    target_fdr: float = 0.05,
    topk_k: int = 3,
    n_boot: int = 1000,
    random_state: int = 42,
    min_pathway_size: int = 3,
    iqr_scale: bool = True,
) -> Dict:
    """
    Full enhanced flagging: three statistics -> combined p per pathway ->
    BH-FDR across pathways -> sample flags.

    Returns a dict with:
      pathway_stats: per (sample, pathway) statistics and flags
      sample_summary: per-sample flag status and top flagged pathways
      diagnostics: control-side false-flag rate per statistic
    """
    stouffer_df = compute_correlation_adjusted_stouffer(
        zscores, feature_to_pathway, normal_mask, min_pathway_size
    )
    topk_df = compute_topk_statistic(
        zscores, feature_to_pathway, normal_mask,
        k=topk_k, n_boot=n_boot, random_state=random_state,
        min_pathway_size=min_pathway_size,
    )
    ratio_df = compute_ratio_zscores(
        features, normal_mask, ratio_specs or [], iqr_scale=iqr_scale
    )

    # Merge the three tables on (sample_id, pathway_name)
    topk_in = topk_df.drop(columns=["n_features"], errors="ignore") if not topk_df.empty else topk_df
    pathway_stats = stouffer_df.merge(
        topk_in, on=["sample_id", "pathway_name"], how="outer"
    )
    if not ratio_df.empty:
        pathway_stats = pathway_stats.merge(
            ratio_df, on=["sample_id", "pathway_name"], how="outer"
        )

    # Combined p per (sample, pathway): Bonferroni over the statistics that
    # were computed for that pair.
    stat_cols = [c for c in ("p_stouffer", "p_topk", "p_ratio") if c in pathway_stats.columns]
    pvals = pathway_stats[stat_cols].to_numpy(dtype=float)
    n_available = (~np.isnan(pvals)).sum(axis=1)
    with np.errstate(invalid="ignore"):
        p_min = np.nanmin(np.where(np.isnan(pvals), np.inf, pvals), axis=1)
    combined_p = np.where(
        n_available > 0,
        np.clip(p_min * np.maximum(n_available, 1), 0.0, 1.0),
        np.nan,
    )
    pathway_stats["p_combined"] = combined_p

    # BH-FDR across pathways per sample (positional indices from groupby)
    q = np.full(len(pathway_stats), np.nan)
    for sample_id, idx in pathway_stats.groupby("sample_id").indices.items():
        p_s = pathway_stats["p_combined"].iloc[idx].to_numpy(dtype=float)
        q[idx] = _bh_qvalues(p_s)
    pathway_stats["q_value"] = q
    pathway_stats["flagged"] = pathway_stats["q_value"] <= target_fdr

    # Diagnostics: control-side false-flag rate (per pathway test)
    ctrl_rows = pathway_stats[pathway_stats["sample_id"].isin(
        zscores.index[normal_mask]
    )]
    diagnostics = {
        "n_pathways_tested": int(pathway_stats["pathway_name"].nunique()),
        "control_false_flag_rate": (
            float(ctrl_rows["flagged"].mean()) if len(ctrl_rows) else 0.0
        ),
        "target_fdr": target_fdr,
    }

    # Per-sample summary
    flagged_rows = pathway_stats[pathway_stats["flagged"]]
    sample_summary = pd.DataFrame({
        "sample_id": list(zscores.index),
    })
    n_flagged = flagged_rows.groupby("sample_id").size()
    top_pathways = flagged_rows.sort_values("q_value").groupby("sample_id")["pathway_name"].apply(
        lambda s: "; ".join(list(s)[:3])
    )
    sample_summary["n_flagged_pathways"] = sample_summary["sample_id"].map(n_flagged).fillna(0).astype(int)
    sample_summary["flagged"] = sample_summary["n_flagged_pathways"] > 0
    sample_summary["top_flagged_pathways"] = sample_summary["sample_id"].map(top_pathways).fillna("")

    logger.info(
        f"Enhanced flagging: {diagnostics['n_pathways_tested']} pathways tested, "
        f"target FDR {target_fdr}; control-side flag rate "
        f"{diagnostics['control_false_flag_rate']:.4f}"
    )

    return {
        "pathway_stats": pathway_stats,
        "sample_summary": sample_summary,
        "diagnostics": diagnostics,
    }
