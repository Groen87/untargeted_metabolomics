"""Layered, direction-aware pathway-shift analysis.

This module implements the layered flagging scheme for the pathway pipeline.
There is NO outlier-detection model here -- every layer is a z-based analysis
and flagging rule, escalating in statistical complexity only where the simpler
layer fails:

1. **Per-metabolite z-scores (age-adjusted, robustly scaled)** -- the atomic
   evidence. ``compute_metabolite_zscores`` produces, for each metabolite,
   ``z_i(s) = (x_i(s) - ref_i(age)) / IQR_i`` where the reference is estimated
   over the NORMAL set only (median/IQR, optionally age-regressed). A single
   metabolite at z = +9 with a known disease association is clinically
   meaningful even without pathway support, so ``flag_metabolites`` flags
   individually extreme metabolites and these act as single-metabolite
   overrides in the decision rule.

2. **Pathway-level statistics** -- the primary detector. For each pathway P
   with k matched metabolites and each sample s::

       Z_med(P, s)  = median(z_1 ... z_k)          (direction-aware median)
       F(P, s)      = (1/k) * sum 1[|z_i| > t_i]    (flagged-fraction breadth)
       Z_up(P, s)   = median of the positive z_i   (valid only if >=2 pos members)
       Z_down(P, s) = median of the negative z_i   (valid only if >=2 neg members)
       Z_split(P, s) = max(|Z_up|, |Z_down|) over the VALID sides

   ``t_i`` is the per-metabolite empirical threshold (the configured percentile
   of ``|z_i|`` over normals). ``flag_pathways`` assigns a **severity tier**
   (moderate / severe) to each pathway flag, testing the signed-extreme guard
   against ``Z_split``. The signed extremes catch the cancel-out problem
   (upstream pileup + downstream depletion averaging toward a near-zero Z_med);

3. **The decision rule** (``decide_samples``) -- the operating point. A sample
   is flagged when::

       >=1 SEVERE pathway flag
       OR  >=2 MODERATE pathway flags
       OR  any single-metabolite override (|z_i| > override_threshold)
       OR  (optional) global anomaly score > global_threshold

   The thresholds are tuned on the inner IMD split by
   ``tune_decision_thresholds``.

4. **Optional global anomaly score** (``compute_global_anomaly_score``) -- the
   "odd sample" safety light. A parameter-light z-aggregate (mean of the top-k
   |z| across metabolites) that lights up for globally odd samples the
   pathway layers miss, with no model involved.
"""

import logging
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)


def _iqr(values: np.ndarray) -> float:
    """Robust interquartile range of a 1-D array. 0 for a constant array."""
    q1 = np.nanpercentile(values, 25)
    q3 = np.nanpercentile(values, 75)
    return float(q3 - q1)


def _age_adjust(features: pd.DataFrame, ages: pd.Series,
                normal_mask: pd.Series,
                method: str = "ols",
                loess_frac: float = 0.5) -> pd.DataFrame:
    """Regress each metabolite on age over normals; return the residuals.

    Removes the age trend (e.g. creatinine rising with muscle mass) so the
    downstream median/IQR scaling is not confounded by age. Two regression
    methods are supported, each fit per metabolite using only the normal
    reference samples, then evaluated for ALL samples (``x - fitted(age)``):

      * ``ols``  -- a linear OLS fit ``x = a + b*age`` (one closed-form solve
        for all metabolites at once). The default; robust for monotonic
        linear age trends.
      * ``loess`` -- a per-metabolite LOESS (locally linear) smoothing of ``x``
        on age via ``statsmodels.nonparametric.lowess``. More flexible for
        non-linear age trends (childhood growth curves, hormone decline) at
        the cost of the ``loess_frac`` smoothing parameter. The prediction for
        a sample age outside the normal range is clamped to the nearest
        fitted age (LOESS does not extrapolate).

    Samples with a missing age are imputed at the normal mean age (so they
    stay in the same residual space as the rest of the data, keeping the
    per-metabolite median/IQR consistent).

    Args:
        features: metabolite matrix (rows=samples, cols=metabolites).
        ages: numeric age per sample, aligned to ``features.index``.
        normal_mask: boolean Series; True for the normal reference samples
            used to fit the age regression.
        method: ``"ols"`` (default) or ``"loess"``.
        loess_frac: LOESS smoothing fraction (bandwidth) used only when
            ``method == "loess"``.

    Returns:
        DataFrame of residuals, same shape/index/columns as ``features``.
    """
    age_norm = pd.to_numeric(ages, errors="coerce")
    # Normals with a usable age are the regression training set.
    norm_idx = normal_mask & age_norm.notna()
    if int(norm_idx.sum()) < 3:
        logger.info("Age adjustment skipped: fewer than 3 normals with a "
                    "usable age; using unadjusted values.")
        return features.copy()

    a = age_norm.loc[norm_idx].to_numpy(dtype=float)
    # Centre age so the intercept is the normal mean at the reference age.
    a_mean = a.mean()
    a_c = a - a_mean
    feat_norm = features.loc[norm_idx].to_numpy(dtype=float)

    if method == "loess":
        # Per-metabolite LOESS (no shared closed form). Predictions for ages
        # outside the fitted normal range are clamped to the nearest fitted
        # point, since LOESS does not extrapolate.
        from statsmodels.nonparametric.smoothers_lowess import lowess

        n_points = len(a_c)
        if n_points < 20:
            # Small normal set: widen the bandwidth for numerical stability.
            frac = max(float(loess_frac), 0.5)
        else:
            frac = float(loess_frac)

        # Fit each metabolite on the sorted normal ages.
        order = np.argsort(a_c)
        a_sorted = a_c[order]
        # predictions at all (centred) sample ages, clamped to the fitted range.
        age_filled = age_norm.fillna(a_mean)
        all_a_c = (age_filled - a_mean).to_numpy(dtype=float)
        lo_hi = (a_sorted.min(), a_sorted.max())

        preds = np.empty((features.shape[0], features.shape[1]), dtype=float)
        for j in range(feat_norm.shape[1]):
            y_sorted = feat_norm[order, j]
            # NaN-aware: statsmodels lowess cannot handle NaN y; if a column
            # has any NaN among the normals, fall back to a constant fit at
            # the normal mean for that metabolite (residuals = x - mean).
            if np.isnan(y_sorted).any():
                preds[:, j] = np.nanmean(y_sorted)
                continue
            try:
                sm = lowess(y_sorted, a_sorted, frac=frac, return_sorted=True)
            except (ValueError, IndexError):
                preds[:, j] = np.nanmean(y_sorted)
                continue
            xs, ys = sm[:, 0], sm[:, 1]
            # Clamp query ages to the fitted range (no extrapolation), then
            # linearly interpolate the LOESS curve to each sample's age.
            clamped = np.clip(all_a_c, lo_hi[0], lo_hi[1])
            preds[:, j] = np.interp(clamped, xs, ys)
        pred = pd.DataFrame(preds, index=features.index, columns=features.columns)
        return features - pred

    # OLS fit per metabolite (one closed-form solve for all columns at once).
    X = np.column_stack([np.ones_like(a_c), a_c])
    coef, *_ = np.linalg.lstsq(X, feat_norm, rcond=None)
    intercept, slope = coef[0], coef[1]

    # Impute missing ages with the normal mean age (centred = 0) so the
    # prediction is the population-average level and the sample stays in the
    # residual space (consistent with the per-metabolite median/IQR).
    age_filled = age_norm.fillna(a_mean)
    pred = intercept + slope * (age_filled - a_mean).to_numpy()[:, None]
    pred = pd.DataFrame(pred, index=features.index, columns=features.columns)
    return features - pred


def compute_metabolite_zscores(features: pd.DataFrame,
                                normal_mask: pd.Series,
                                iqr_scale: bool = True,
                                ages: Optional[pd.Series] = None,
                                age_adjustment_method: str = "ols",
                                age_loess_frac: float = 0.5,
                                ) -> pd.DataFrame:
    """Compute per-metabolite, per-sample (age-adjusted) robust z-scores.

    For each metabolite column::

        reference = x_i over the NORMAL set (optionally age-regressed)
        median_i  = median(reference)
        scale_i   = IQR_i (iqr_scale) else std_i, over the normal set
        z_i(s)    = (x_i(s) - median_i) / scale_i

    Median and scale are estimated from normals only, so abnormal samples are
    scored against the normal distribution with no leakage. A metabolite whose
    scale is zero (constant over normals) is left as NaN so it is excluded from
    every pathway statistic it belongs to.

    Args:
        features: DataFrame rows=samples, columns=metabolites.
        normal_mask: boolean Series aligned to ``features.index``; True for the
            normal reference samples used to estimate median/scale.
        iqr_scale: when True (default) scale by IQR (matches the outlier-
            detection pipeline's log-IQR convention); when False scale by std.
        ages: optional numeric age Series aligned to ``features.index``. When
            provided, each metabolite is age-regressed over the normals first
            (see :func:`_age_adjust`) so the z-scores are age-adjusted.
        age_adjustment_method: age-regression method passed to
            :func:`_age_adjust` when ``ages`` is given: ``"ols"`` (default,
            linear) or ``"loess"`` (locally linear, non-linear age trends).
        age_loess_frac: LOESS smoothing fraction, used only when
            ``age_adjustment_method == "loess"``.

    Returns:
        DataFrame of the same shape as ``features`` holding z-scores. Cells
        for metabolites with zero scale are NaN.
    """
    base = features
    if ages is not None:
        base = _age_adjust(features, ages, normal_mask,
                          method=age_adjustment_method,
                          loess_frac=age_loess_frac)

    normals = base.loc[normal_mask]
    medians = normals.median(axis=0)
    if iqr_scale:
        scales = normals.apply(lambda s: _iqr(s.to_numpy(dtype=float)), axis=0)
    else:
        scales = normals.std(axis=0)
    safe_scales = scales.where(scales > 1e-10, np.nan)
    z = (base - medians) / safe_scales
    logger.info(f"Computed {z.shape[1]} metabolite z-scores over "
                f"{int(normal_mask.sum())} normals"
                f"{(' (age-adjusted, ' + age_adjustment_method + ')') if ages is not None else ''}; "
                f"{int((safe_scales.isna() | (safe_scales <= 1e-10)).sum())} "
                f"metabolites dropped (zero scale).")
    return z


def flag_metabolites(zscores: pd.DataFrame,
                     normal_mask: pd.Series,
                     override_threshold: float = 6.0,
                     flag_percentile: float = 99,
                     ) -> pd.DataFrame:
    """Atomic per-metabolite flag layer.

    A metabolite is flagged for a sample when ``|z_i| > override_threshold``
    (the single-metabolite override used by the decision rule). The
    per-metabolite empirical threshold ``t_i`` (the configured percentile of
    ``|z_i|`` over normals) is also recorded so the breadth can be reported
    alongside the hard override.

    Args:
        zscores: per-metabolite z-scores from :func:`compute_metabolite_zscores`.
        normal_mask: boolean Series aligned to ``zscores.index``; the normal
            reference set used to derive ``t_i``.
        override_threshold: fixed high z-magnitude a single metabolite must
            exceed to act as a standalone override.
        flag_percentile: percentile (0-100) of ``|z_i|`` over normals used as
            the per-metabolite empirical threshold ``t_i``.

    Returns:
        Long DataFrame with one row per (sample, metabolite) that is flagged,
        columns ``sample_id``, ``metabolite``, ``z``, ``abs_z``,
        ``override_threshold``, ``per_metabolite_threshold``, ``direction``.
    """
    z_normals = zscores.loc[normal_mask]
    per_met = z_normals.abs().quantile(flag_percentile / 100.0, axis=0)

    rows: List[Dict] = []
    # Iterate by integer position (not label): the sample index may contain
    # duplicate IDs, in which case ``col_z.loc[sid]`` would return a Series
    # instead of a scalar and break the float() cast below.
    sample_ids = zscores.index
    for col in zscores.columns:
        t_i = per_met.get(col, np.nan)
        col_vals = zscores[col].to_numpy(dtype=float)
        flagged_mask = np.abs(col_vals) > override_threshold
        for pos in np.flatnonzero(flagged_mask):
            z = float(col_vals[pos])
            rows.append({
                "sample_id": sample_ids[pos],
                "metabolite": col,
                "z": z,
                "abs_z": abs(z),
                "override_threshold": float(override_threshold),
                "per_metabolite_threshold": float(t_i) if not np.isnan(t_i) else float("nan"),
                "direction": "increased" if z >= 0 else "decreased",
            })
    out = pd.DataFrame(rows, columns=["sample_id", "metabolite", "z", "abs_z",
                                      "override_threshold",
                                      "per_metabolite_threshold", "direction"])
    logger.info(f"Atomic metabolite layer: flagged {len(out)} (sample, "
                f"metabolite) pairs at |z| > {override_threshold}.")
    return out


def compute_pathway_statistics(zscores: pd.DataFrame,
                                feature_to_pathway: pd.DataFrame,
                                normal_mask: pd.Series,
                                flag_percentile: float = 99,
                                min_pathway_size: int = 3,
                                ) -> pd.DataFrame:
    """Compute per-sample, per-pathway Z_med / F / Z_up / Z_down statistics.

    Args:
        zscores: per-metabolite z-scores from :func:`compute_metabolite_zscores`
            (rows=samples, columns=features).
        feature_to_pathway: long (feature, smp_id, pathway_name, ...) table
            from :func:`link_features_to_pathways`. Each pathway is scored over
            the union of its matched features.
        normal_mask: boolean Series aligned to ``zscores.index``; the normal
            reference set used to derive the per-metabolite empirical
            threshold ``t_i``.
        flag_percentile: percentile (0-100) of ``|z_i|`` over normals used as
            the per-metabolite threshold ``t_i`` for F. Default 99.
        min_pathway_size: pathways with fewer matched metabolites than this
            are skipped.

    Returns:
        Long DataFrame with one row per (sample, pathway) and columns
        ``sample_id``, ``smp_id``, ``pathway_name``, ``n_metabolites`` (k),
        ``z_med``, ``flagged_fraction`` (F), ``z_up``, ``z_down``, ``z_split``,
        ``threshold_percentile``. ``z_up`` / ``z_down`` are the medians of the
        positive / negative z's, valid only when the side has >=2 members
        (else NaN); ``z_split`` = max(|Z_up|, |Z_down|) over the valid sides
        (NaN when neither side has >=2 members).
    """
    z_normals = zscores.loc[normal_mask]
    per_metabolite_threshold = z_normals.abs().quantile(flag_percentile / 100.0, axis=0)
    thresholds = per_metabolite_threshold.where(
        per_metabolite_threshold.notna() & (per_metabolite_threshold > 0),
        np.inf,
    )

    available = set(zscores.columns)
    pathway_features: Dict[str, Dict] = {}
    if not feature_to_pathway.empty:
        for smp_id, grp in feature_to_pathway.groupby("smp_id"):
            feats = sorted(set(grp["feature"]) & available)
            name = grp["pathway_name"].iloc[0] if "pathway_name" in grp.columns else smp_id
            if len(feats) >= min_pathway_size:
                pathway_features[smp_id] = {"pathway_name": name, "features": feats}

    if not pathway_features:
        logger.warning("No pathways have enough matched metabolites with "
                       "usable z-scores; statistics table will be empty.")
        return pd.DataFrame(columns=["sample_id", "smp_id", "pathway_name",
                                      "n_metabolites", "z_med",
                                      "flagged_fraction", "z_up", "z_down",
                                      "z_split", "threshold_percentile"])

    rows: List[Dict] = []
    sample_ids = zscores.index
    z_arr = zscores.to_numpy(dtype=float)
    col_index = {c: i for i, c in enumerate(zscores.columns)}
    thr_arr = thresholds.reindex(zscores.columns).to_numpy(dtype=float)

    for smp_id, info in pathway_features.items():
        feat_cols = [col_index[f] for f in info["features"]]
        sub = z_arr[:, feat_cols]
        thr_sub = thr_arr[feat_cols]
        n = sub.shape[1]

        z_med = np.nanmedian(sub, axis=1)
        flagged = (np.abs(sub) > thr_sub) & ~np.isnan(sub)
        k_valid = np.sum(~np.isnan(sub), axis=1)
        n_flagged = np.sum(flagged, axis=1)
        f = np.where(k_valid > 0, n_flagged / k_valid, np.nan)
        # Signed-extreme medians, valid only when a side has >=2 members (a
        # single positive or negative z is not a stable side median). Z_split
        # = max(|Z_up|, |Z_down|) over the VALID sides (NaN when neither side
        # has >=2 members).
        with np.errstate(invalid="ignore"):
            pos = np.where(sub > 0, sub, np.nan)
            neg = np.where(sub < 0, sub, np.nan)
        n_pos = np.sum(~np.isnan(pos), axis=1)
        n_neg = np.sum(~np.isnan(neg), axis=1)
        # Suppress "All-NaN slice" warnings from nanmedian when a row has
        # no positive/negative z-scores for a pathway.
        with np.errstate(all="ignore"):
            z_up_raw = np.nanmedian(pos, axis=1)
            z_down_raw = np.nanmedian(neg, axis=1)
        z_up = np.where(n_pos >= 2, z_up_raw, np.nan)
        z_down = np.where(n_neg >= 2, z_down_raw, np.nan)
        # Z_split: largest side-median magnitude among the valid sides.
        cand = np.stack([np.abs(z_up), np.abs(z_down)], axis=1)
        all_invalid = np.all(np.isnan(cand), axis=1)
        with np.errstate(all="ignore"):
            z_split = np.nanmax(cand, axis=1)
        z_split = np.where(all_invalid, np.nan, z_split)

        for i, sid in enumerate(sample_ids):
            rows.append({
                "sample_id": sid,
                "smp_id": smp_id,
                "pathway_name": info["pathway_name"],
                "n_metabolites": int(n),
                "z_med": float(z_med[i]) if not np.isnan(z_med[i]) else float("nan"),
                "flagged_fraction": float(f[i]) if not np.isnan(f[i]) else float("nan"),
                "z_up": float(z_up[i]) if not np.isnan(z_up[i]) else float("nan"),
                "z_down": float(z_down[i]) if not np.isnan(z_down[i]) else float("nan"),
                "z_split": float(z_split[i]) if not np.isnan(z_split[i]) else float("nan"),
                "threshold_percentile": float(flag_percentile),
            })

    stats = pd.DataFrame(rows)
    logger.info(f"Computed pathway statistics for {len(pathway_features)} pathways "
                f"across {len(sample_ids)} samples.")
    return stats


def _pathway_severity(row: pd.Series, mod: dict, sev: dict) -> Tuple[str, List[str]]:
    """Return (severity, reasons) for one (sample, pathway) stats row.

    ``mod`` / ``sev`` hold the moderate / severe threshold tuples
    ``(zmed, flagged_fraction, signed_extreme)``. The signed-extreme element is
    tested against ``z_split`` = max(|Z_up|, |Z_down|) over the VALID sides
    (each side needs >=2 members to be valid), so a single outlier member on
    one side cannot trip the guard on its own. A row is flagged (moderate or
    severe) when ANY moderate condition holds; it is SEVERE when any severe
    condition holds. Returns severity in {"none","moderate","severe"} and the
    list of human-readable triggering reasons.
    """
    zmed = row.get("z_med")
    f = row.get("flagged_fraction")
    zup = row.get("z_up")
    zsplit = row.get("z_split")

    reasons: List[str] = []
    is_severe = False

    def _check(thr, label, severe=False):
        nonlocal is_severe
        if pd.notna(zmed) and abs(zmed) > thr[0]:
            reasons.append(f"|Z_med|={abs(zmed):.2f}>{thr[0]:g}")
            is_severe = is_severe or severe
        if pd.notna(f) and f > thr[1]:
            reasons.append(f"F={f:.2f}>{thr[1]:g}")
            is_severe = is_severe or severe
        # Signed-extreme guard via Z_split (max valid side median magnitude).
        if pd.notna(zsplit) and zsplit > thr[2]:
            side = "Z_up" if (pd.notna(zup) and abs(zup) == zsplit) else "Z_down"
            reasons.append(f"{side}={zsplit:.2f}")
            is_severe = is_severe or severe

    # Severe conditions take precedence; check them first so a row that
    # qualifies as severe is labelled severe even if moderate also fires.
    _check(sev, "severe", severe=True)
    if not reasons:
        _check(mod, "moderate", severe=False)
    if not reasons:
        return "none", []
    return ("severe" if is_severe else "moderate"), reasons


def flag_pathways(stats: pd.DataFrame,
                   zmed_threshold: float = 2.0,
                   flagged_fraction_threshold: float = 0.5,
                   signed_extreme_threshold: float = 2.5,
                   severe_zmed_threshold: float = 3.0,
                   severe_flagged_fraction_threshold: float = 0.7,
                   severe_signed_extreme_threshold: float = 4.0,
                   ) -> pd.DataFrame:
    """Flag pathways and assign a severity tier (moderate / severe).

    A pathway is **flagged** when any moderate condition holds::

        |Z_med|   > zmed_threshold
        F          > flagged_fraction_threshold
        Z_split    > signed_extreme_threshold

    where ``Z_split = max(|Z_up|, |Z_down|)`` over the VALID sides (a side is
    valid only if it has >=2 members). It is **severe** when any of the stricter
    ``severe_*`` thresholds hold; otherwise it is **moderate**. ``flagged`` is
    True for both tiers. The signed extremes (via Z_split) catch the
    cancel-out case where upstream pileup and downstream depletion average
    toward a near-zero Z_med, and the >=2-member rule prevents a single outlier
    member from tripping the guard on its own.

    Args:
        stats: output of :func:`compute_pathway_statistics`.
        zmed_threshold, flagged_fraction_threshold,
        signed_extreme_threshold: moderate-tier cutoffs.
        severe_zmed_threshold, severe_flagged_fraction_threshold,
        severe_signed_extreme_threshold: severe-tier cutoffs.

    Returns:
        ``stats`` with added ``flagged`` (bool), ``severity``
        ("none"/"moderate"/"severe"), and ``flag_reason`` (';'-joined) columns.
    """
    mod = (zmed_threshold, flagged_fraction_threshold, signed_extreme_threshold)
    sev = (severe_zmed_threshold, severe_flagged_fraction_threshold,
           severe_signed_extreme_threshold)

    if stats.empty:
        return stats.assign(flagged=False, severity="none", flag_reason="")

    severities: List[str] = []
    flags: List[bool] = []
    reasons_out: List[str] = []
    for _, r in stats.iterrows():
        sev_tier, reasons = _pathway_severity(r, mod, sev)
        severities.append(sev_tier)
        flags.append(sev_tier != "none")
        reasons_out.append(";".join(reasons))

    out = stats.copy()
    out["flagged"] = flags
    out["severity"] = severities
    out["flag_reason"] = reasons_out
    n_mod = int((out["severity"] == "moderate").sum())
    n_sev = int((out["severity"] == "severe").sum())
    logger.info(f"Pathway flags: {n_sev} severe, {n_mod} moderate "
                f"({int(out['flagged'].sum())} total).")
    return out


def compute_global_anomaly_score(zscores: pd.DataFrame,
                                  top_k: int = 10,
                                  ) -> pd.Series:
    """Global "odd sample" safety-light z-aggregate (no model).

    For each sample, the mean of its top-``top_k`` metabolite |z| values. This
    is a parameter-light summary that lights up for globally odd samples the
    pathway layers may miss (e.g. a diffuse, multi-pathway perturbation with no
    single pathway crossing its threshold). Lower = more normal.

    Args:
        zscores: per-metabolite z-scores from :func:`compute_metabolite_zscores`.
        top_k: number of largest |z| metabolites to average per sample.

    Returns:
        Series indexed by sample_id of the global anomaly score.
    """
    abs_z = zscores.abs()
    k = min(top_k, abs_z.shape[1])
    if k < 1:
        return pd.Series(np.nan, index=zscores.index, name="global_anomaly_score")
    # Mean of the k largest |z| per row (NaN-aware: at least one finite value).
    top_vals = np.partition(abs_z.to_numpy(dtype=float), -k, axis=1)[:, -k:]
    with np.errstate(invalid="ignore"):
        scores = np.nanmean(np.where(np.isfinite(top_vals), top_vals, np.nan), axis=1)
    scores = np.where(np.all(~np.isfinite(top_vals), axis=1), np.nan, scores)
    return pd.Series(scores, index=zscores.index, name="global_anomaly_score")


def decide_samples(pathway_flags: pd.DataFrame,
                    metabolite_flags: pd.DataFrame,
                    global_scores: Optional[pd.Series] = None,
                    min_moderate: int = 2,
                    min_severe: int = 1,
                    global_threshold: Optional[float] = None,
                    ) -> pd.DataFrame:
    """Apply the sample-level decision rule (the operating point).

    A sample is **flagged** when ANY holds::

        >= min_severe SEVERE pathway flags
        OR  >= min_moderate MODERATE pathway flags (severe ones excluded)
        OR  any single-metabolite override (rows in ``metabolite_flags``)
        OR  (when global_threshold is set) global anomaly score > global_threshold

    The triggering reasons are recorded in ``decision_reason``.

    Args:
        pathway_flags: output of :func:`flag_pathways` (one row per
            sample, pathway with a ``severity`` column).
        metabolite_flags: output of :func:`flag_metabolites` (the single-
            metabolite overrides).
        global_scores: optional Series from
            :func:`compute_global_anomaly_score`. Only used when
            ``global_threshold`` is not None.
        min_moderate: minimum MODERATE (non-severe) pathway flags to flag a
            sample on breadth alone.
        min_severe: minimum SEVERE pathway flags to flag a sample.
        global_threshold: when not None, flag any sample whose global anomaly
            score exceeds it (the "odd sample" safety light).

    Returns:
        DataFrame indexed by sample_id with columns ``flagged`` (bool),
        ``n_severe_pathways``, ``n_moderate_pathways``,
        ``n_metabolite_overrides``, ``global_anomaly_score`` (when provided),
        and ``decision_reason`` (';'-joined).
    """
    sample_ids = pathway_flags["sample_id"].unique() if not pathway_flags.empty else []
    if len(sample_ids) == 0 and metabolite_flags.empty and global_scores is None:
        return pd.DataFrame(columns=["flagged", "n_severe_pathways",
                                      "n_moderate_pathways",
                                      "n_metabolite_overrides",
                                      "global_anomaly_score", "decision_reason"])

    all_ids = list(sample_ids)
    if not metabolite_flags.empty:
        all_ids += [s for s in metabolite_flags["sample_id"].unique() if s not in all_ids]
    if global_scores is not None:
        all_ids += [s for s in global_scores.index if s not in all_ids]

    # Build scalar lookups for counts. Use .to_dict() so duplicate sample IDs
    # in the Series are handled (keeps last, which is fine since duplicates
    # represent the same sample and should share the same counts/scores).
    sev_counts = (pathway_flags[pathway_flags["severity"] == "severe"]
                   .groupby("sample_id").size().to_dict())
    mod_counts = (pathway_flags[pathway_flags["severity"] == "moderate"]
                   .groupby("sample_id").size().to_dict())
    met_counts = (metabolite_flags.groupby("sample_id").size().to_dict()
                  if not metabolite_flags.empty else {})
    gscores_dict = (global_scores.to_dict() if global_scores is not None else {})

    rows: List[Dict] = []
    for sid in all_ids:
        n_sev = int(sev_counts.get(sid, 0))
        n_mod = int(mod_counts.get(sid, 0))
        n_met = int(met_counts.get(sid, 0))
        reasons: List[str] = []
        if n_sev >= min_severe:
            reasons.append(f"{n_sev} severe pathway flag(s)")
        if n_mod >= min_moderate:
            reasons.append(f"{n_mod} moderate pathway flag(s)")
        if n_met > 0:
            reasons.append(f"{n_met} metabolite override(s)")
        gscore = gscores_dict.get(sid, float("nan"))
        if (global_threshold is not None and not np.isnan(gscore)
                and gscore > global_threshold):
            reasons.append(f"global={gscore:.2f}>{global_threshold:g}")
        rows.append({
            "sample_id": sid,
            "flagged": bool(reasons),
            "n_severe_pathways": n_sev,
            "n_moderate_pathways": n_mod,
            "n_metabolite_overrides": n_met,
            "global_anomaly_score": gscore,
            "decision_reason": ";".join(reasons),
        })

    out = pd.DataFrame(rows).set_index("sample_id")
    logger.info(f"Decision rule flagged {int(out['flagged'].sum())} of "
                f"{len(out)} samples.")
    return out


def _f1_at_prevalence(detection_rate: float, fpr: float, prevalence: float) -> float:
    """Precision/F1 at a target deployment prevalence (analytic)."""
    if np.isnan(detection_rate) or np.isnan(fpr):
        return float("nan")
    denom = prevalence * detection_rate + (1.0 - prevalence) * fpr
    if denom <= 0:
        return 0.0
    precision = (prevalence * detection_rate) / denom
    if precision + detection_rate <= 0:
        return 0.0
    return float(2.0 * precision * detection_rate / (precision + detection_rate))


def tune_decision_thresholds(pathway_stats: pd.DataFrame,
                               zscores: pd.DataFrame,
                               normal_mask: pd.Series,
                               labels: pd.Series,
                               metabolite_override_grid: List[float],
                               moderate_zmed_grid: List[float],
                               severe_zmed_grid: List[float],
                               flag_percentile: float = 99,
                               signed_extreme_grid: Optional[List[float]] = None,
                               prevalence: float = 0.02,
                               metric: str = "f1",
                               ) -> pd.DataFrame:
    """Sweep the operating point on the inner IMD split.

    Grid-searches the single-metabolite override threshold and the moderate /
    severe pathway |Z_med| thresholds (with the flagged fraction and signed
    extremes derived consistently), re-deriving the per-sample decision each
    time, and scores each setting against the binary ``labels`` (0 = normal,
    1 = IMD) at the target deployment ``prevalence``. Returns the full sweep
    table sorted by the chosen ``metric`` so the user can pick the operating
    point.

    This is calibration on the inner (normal vs IMD) split only; it never
    touches an external test set.

    Args:
        pathway_stats: raw per-pathway statistics (output of
            :func:`compute_pathway_statistics`) BEFORE flagging.
        zscores: per-metabolite z-scores used for the metabolite override and
            global score layers.
        normal_mask: boolean normal-reference mask aligned to ``zscores.index``.
        labels: binary 0/1 label per sample aligned to ``zscores.index`` (1 =
            IMD). Used only to score each threshold setting.
        metabolite_override_grid: candidate single-metabolite |z| thresholds.
        moderate_zmed_grid: candidate moderate-tier |Z_med| thresholds.
        severe_zmed_grid: candidate severe-tier |Z_med| thresholds.
        flag_percentile: percentile for the per-metabolite empirical threshold.
        signed_extreme_grid: candidate signed-extreme thresholds (defaults to a
            small grid around the moderate |Z_med| values).
        prevalence: assumed deployment prevalence for the analytic F1/precision.

    Returns:
        DataFrame with one row per threshold combination and columns
        ``override_threshold``, ``moderate_zmed``, ``severe_zmed``,
        ``signed_extreme``, ``n_flagged``, ``detection_rate``, ``fpr``,
        ``precision``, ``f1``, ``accuracy``.
    """
    if signed_extreme_grid is None:
        signed_extreme_grid = [2.5, 3.0, 4.0]

    # Fixed moderate flagged-fraction and severe multipliers relative to the
    # swept |Z_med|, so the grid stays tractable and the tiers move together.
    rows: List[Dict] = []
    y = pd.to_numeric(labels, errors="coerce").fillna(0).astype(int)
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())

    for override in metabolite_override_grid:
        met_flags = flag_metabolites(zscores, normal_mask,
                                     override_threshold=override,
                                     flag_percentile=flag_percentile)
        for mod_zmed in moderate_zmed_grid:
            mod_ff = min(mod_zmed / 4.0, 0.9)
            for sev_zmed in severe_zmed_grid:
                sev_ff = min(sev_zmed / 4.0, 0.95)
                for ext in signed_extreme_grid:
                    flagged = flag_pathways(
                        pathway_stats,
                        zmed_threshold=mod_zmed,
                        flagged_fraction_threshold=mod_ff,
                        signed_extreme_threshold=ext,
                        severe_zmed_threshold=sev_zmed,
                        severe_flagged_fraction_threshold=sev_ff,
                        severe_signed_extreme_threshold=ext + 1.0,
                    )
                    decision = decide_samples(flagged, met_flags,
                                              global_scores=None)
                    # Align decision with zscores.index. zscores.index may have duplicate
                    # sample IDs, so we cannot use reindex directly. Instead, build a
                    # flagged Series aligned to zscores.index using dict lookup.
                    flagged_dict = decision["flagged"].to_dict()
                    pred = pd.Series(
                        [flagged_dict.get(sid, False) for sid in zscores.index],
                        index=zscores.index
                    ).astype(int).to_numpy()
                    tp = int(((pred == 1) & (y.to_numpy() == 1)).sum())
                    fp = int(((pred == 1) & (y.to_numpy() == 0)).sum())
                    detection = (tp / n_pos) if n_pos > 0 else float("nan")
                    fpr = (fp / n_neg) if n_neg > 0 else float("nan")
                    # Precision/F1 at the target deployment prevalence.
                    denom = prevalence * detection + (1.0 - prevalence) * fpr
                    prec = ((prevalence * detection) / denom) if denom > 0 else 0.0
                    f1 = (2 * prec * detection / (prec + detection)) \
                        if (prec + detection) > 0 else 0.0
                    acc = float((1 - prevalence) * (1 - fpr) + prevalence * detection) \
                        if not (np.isnan(detection) or np.isnan(fpr)) else float("nan")
                    score = {"f1": f1, "precision": prec, "detection": detection,
                             "accuracy": acc}.get(metric, f1)
                    rows.append({
                        "override_threshold": override,
                        "moderate_zmed": mod_zmed,
                        "severe_zmed": sev_zmed,
                        "signed_extreme": ext,
                        "n_flagged": int(pred.sum()),
                        "detection_rate": detection if not np.isnan(detection) else float("nan"),
                        "fpr": fpr if not np.isnan(fpr) else float("nan"),
                        "precision": prec,
                        "f1": f1,
                        "accuracy": acc,
                        "score": score,
                    })
    sweep = pd.DataFrame(rows).sort_values("score", ascending=False).reset_index(drop=True)
    logger.info(f"Threshold sweep: {len(sweep)} settings; best "
                f"{metric}={sweep['score'].iloc[0]:.4f} at "
                f"override={sweep['override_threshold'].iloc[0]}, "
                f"mod_zmed={sweep['moderate_zmed'].iloc[0]}, "
                f"sev_zmed={sweep['severe_zmed'].iloc[0]}.")
    return sweep
