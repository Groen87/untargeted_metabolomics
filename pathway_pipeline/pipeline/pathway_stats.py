"""Per-metabolite z-scores against the normal reference set.

This is the scoring basis for the pathway pipeline: every pathway-mapped
feature is turned into a robust z-score using the normal samples as the
reference distribution, so that pathway-level statistics (Stouffer scores) can
combine metabolites on a common scale.

Reference definition: normals are the samples with ``Classification == 0``
AND ``Oordeel targeted == 0``; every other sample is scored but never
contributes to the reference statistics.

Scaling is robust: ``z = (x - median_i) / IQR_i`` with the median and IQR
estimated over the normals only (``iqr_scale: false`` switches the denominator
to the standard deviation). Features whose reference scale is zero (flat in the
normals) or that have no normal values at all cannot be calibrated and are
dropped -- they would contribute z = 0 everywhere and dilute the pathway
scores.
"""

import logging
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


logger = logging.getLogger(__name__)

CLASSIFICATION_COLUMN = "Classification"
OORDEEL_COLUMN = "Oordeel targeted"


def classify_samples(metadata: pd.DataFrame,
                     normal_classification: int = 0,
                     normal_oordeel: int = 0) -> pd.Series:
    """Return a boolean Series marking the normal reference samples.

    Normals are the samples with ``Classification == normal_classification``
    AND ``Oordeel targeted == normal_oordeel`` (defaults 0/0).

    Args:
        metadata: DataFrame holding at least the ``Classification`` and
            ``Oordeel targeted`` columns (the configured non-feature columns).
        normal_classification: classification value marking normals.
        normal_oordeel: oordeel value marking normals.

    Returns:
        Boolean Series indexed like ``metadata``; True for normals. All-False
        (with a logged error) when the columns are missing.
    """
    for col in (CLASSIFICATION_COLUMN, OORDEEL_COLUMN):
        if col not in metadata.columns:
            logger.error(f"Metadata is missing the '{col}' column; "
                         f"no normal reference can be defined.")
            return pd.Series(False, index=metadata.index)

    cls = pd.to_numeric(metadata[CLASSIFICATION_COLUMN], errors="coerce")
    oor = pd.to_numeric(metadata[OORDEEL_COLUMN], errors="coerce")
    normal_mask = (cls == normal_classification) & (oor == normal_oordeel)
    logger.info(f"Sample classification: {int(normal_mask.sum())} normals "
                f"of {len(metadata)} samples.")
    return normal_mask


def compute_metabolite_zscores(features: pd.DataFrame,
                               normal_mask: pd.Series,
                               iqr_scale: bool = True
                               ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute robust z-scores for every feature column.

    The reference (center and scale) is estimated over the normal samples
    only. A feature whose reference scale is zero, or that has no usable
    normal values, is dropped: it cannot be calibrated and would contribute
    z = 0 for every sample, diluting downstream pathway scores.

    Args:
        features: DataFrame with samples as rows and features as columns
            (log10-transformed upstream).
        normal_mask: boolean Series marking the normal reference samples.
        iqr_scale: scale by the normals' IQR (default) or by their standard
            deviation when false.

    Returns:
        Tuple ``(zscores, reference_stats, dropped_features)``:

        - ``zscores``: DataFrame of the same sample index with one z-score
          column per calibrated feature (dropped features removed).
        - ``reference_stats``: per-feature reference table with columns
          ``feature``, ``median``, ``scale``, ``n_normal_values``,
          ``p_normal_missing``.
        - ``dropped_features``: per dropped feature with columns ``feature``
          and ``reason`` ('no_normal_values' / 'zero_scale').
    """
    if features.empty:
        empty_stats = pd.DataFrame(columns=["feature", "median", "scale",
                                            "n_normal_values", "p_normal_missing"])
        return (pd.DataFrame(index=features.index),
                empty_stats,
                pd.DataFrame(columns=["feature", "reason"]))

    normal_mask = normal_mask.reindex(features.index, fill_value=False)
    if not normal_mask.any():
        logger.error("No normal samples; z-scores cannot be calibrated.")
        empty_stats = pd.DataFrame(columns=["feature", "median", "scale",
                                            "n_normal_values", "p_normal_missing"])
        return (pd.DataFrame(index=features.index),
                empty_stats,
                pd.DataFrame({"feature": list(features.columns),
                              "reason": ["no_normal_values"] * features.shape[1]}))

    numeric = features.apply(pd.to_numeric, errors="coerce")
    normal_values = numeric.loc[normal_mask]

    medians = normal_values.median()
    if iqr_scale:
        scales = normal_values.quantile(0.75) - normal_values.quantile(0.25)
    else:
        scales = normal_values.std(ddof=0)

    n_normal_values = normal_values.notna().sum()
    p_normal_missing = normal_values.isna().mean()

    dropped_rows: List[Dict] = []
    keep: List[str] = []
    for col in numeric.columns:
        if n_normal_values[col] == 0:
            dropped_rows.append({"feature": col, "reason": "no_normal_values"})
        elif not np.isfinite(scales[col]) or scales[col] == 0:
            dropped_rows.append({"feature": col, "reason": "zero_scale"})
        else:
            keep.append(col)

    dropped = pd.DataFrame(dropped_rows, columns=["feature", "reason"])
    if len(dropped):
        logger.info(f"Dropped {len(dropped)} features before z-scoring "
                    f"({int((dropped['reason'] == 'zero_scale').sum())} zero-scale, "
                    f"{int((dropped['reason'] == 'no_normal_values').sum())} without "
                    f"normal values).")

    zscores = (numeric[keep] - medians[keep]) / scales[keep]

    reference_stats = pd.DataFrame({
        "feature": keep,
        "median": medians[keep].to_numpy(),
        "scale": scales[keep].to_numpy(),
        "n_normal_values": n_normal_values[keep].to_numpy(),
        "p_normal_missing": p_normal_missing[keep].to_numpy(),
    })

    logger.info(f"Calibrated {len(keep)} of {numeric.shape[1]} features "
                f"against {int(normal_mask.sum())} normals "
                f"({'median/IQR' if iqr_scale else 'median/std'} scaling).")
    return zscores, reference_stats, dropped


def filter_pathways_for_scoring(coverage: pd.DataFrame,
                                 feature_to_pathway: pd.DataFrame,
                                 available_features,
                                 min_pathway_features: int = 3) -> pd.DataFrame:
    """Restrict pathway coverage to the calibrated features and apply a
    minimum usable-metabolite count.

    After the z-score stage drops uncalibratable features, a pathway's usable
    matched metabolites are those with at least one surviving feature. The
    coverage counts and fraction are recomputed against the surviving links,
    and pathways with fewer than ``min_pathway_features`` usable matched
    metabolites are dropped (Stouffer on k < 3 is dominated by a single
    outlier).

    Args:
        coverage: output of :func:`pathway_mapping.pathway_coverage` (the
            post-20% table).
        feature_to_pathway: output of
            :func:`pathway_mapping.link_features_to_pathways`.
        available_features: iterable of feature columns that survived the
            z-score stage.
        min_pathway_features: minimum number of usable matched metabolites for
            a pathway to stay in the scoring set.

    Returns:
        DataFrame with the same columns as ``coverage`` (counts and coverage
        recomputed over the surviving features), sorted by coverage
        descending.
    """
    out_cols = list(coverage.columns)
    if coverage.empty:
        return coverage.copy()

    available = set(available_features)
    usable = (
        feature_to_pathway[feature_to_pathway["feature"].isin(available)]
        .groupby(["smp_id", "pathway_name"])["hmdb_id"]
        .agg(lambda s: sorted(set(s)))
        .rename("matched_metabolites")
        .reset_index()
    )
    usable["n_matched_metabolites"] = usable["matched_metabolites"].str.len()

    usable_features = (
        feature_to_pathway[feature_to_pathway["feature"].isin(available)]
        .groupby(["smp_id", "pathway_name"])["feature"]
        .agg(lambda s: sorted(set(s)))
        .rename("matched_features")
        .reset_index()
    )
    usable_features["n_matched_features"] = usable_features["matched_features"].str.len()
    usable_features["matched_features"] = usable_features["matched_features"].str.join(";")

    counts = usable.merge(usable_features, on=["smp_id", "pathway_name"], how="outer")

    # Drop the pre-z-score matched columns from coverage and re-derive them.
    drop_cols = [c for c in ("matched_metabolites", "n_matched_metabolites",
                             "matched_features", "n_matched_features", "coverage")
                 if c in coverage.columns]
    base = coverage.drop(columns=drop_cols)
    scored = base.merge(counts, on=["smp_id", "pathway_name"], how="left")
    scored["n_matched_metabolites"] = scored["n_matched_metabolites"].fillna(0).astype(int)
    scored["n_matched_features"] = scored["n_matched_features"].fillna(0).astype(int)
    scored["matched_metabolites"] = scored["matched_metabolites"].fillna("")
    scored["matched_features"] = scored["matched_features"].fillna("")
    scored["coverage"] = scored.apply(
        lambda r: (r["n_matched_metabolites"] / r["n_metabolites"])
        if pd.notna(r.get("n_metabolites")) and r["n_metabolites"] > 0 else float("nan"),
        axis=1,
    )

    before = len(scored)
    dropped_rows = scored[scored["n_matched_metabolites"] < min_pathway_features]
    scored = scored[scored["n_matched_metabolites"] >= min_pathway_features].copy()
    scored = scored.sort_values("coverage", ascending=False).reset_index(drop=True)
    dropped = before - len(scored)
    if dropped:
        logger.info(f"Dropped {dropped} pathways with fewer than "
                    f"{min_pathway_features} usable matched metabolites after "
                    f"z-scoring; {len(scored)} remain.")
        logger.debug(f"Dropped pathways: {sorted(dropped_rows['pathway_name'].unique())}")
    return scored[out_cols]


def compute_stouffer_scores(zscores: pd.DataFrame,
                             feature_to_pathway: pd.DataFrame,
                             scored_coverage: pd.DataFrame,
                             normal_mask: pd.Series,
                             min_metabolites: int = 3) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Compute per-pathway Stouffer scores for every sample.

    For each pathway P with usable metabolites (features deduplicated per
    HMDB ID -- several features mapping to the same metabolite are averaged)
    and each sample s with k usable metabolite z-scores:

        Z_signed(P, s) = sum(z_i) / sqrt(k)      (direction-aware)
        Z_abs(P, s)    = sum(|z_i|) / sqrt(k)    (disturbance regardless of
                                                  direction; the IMD default)

    Metabolites with a missing z-score for a sample are skipped for that
    sample; a sample with fewer than ``min_metabolites`` usable metabolites in
    a pathway gets NaN for it (never a shrunken score).

    Args:
        zscores: per-sample metabolite z-scores (output of
            :func:`compute_metabolite_zscores`).
        feature_to_pathway: (feature, pathway) links.
        scored_coverage: pathways kept for scoring (output of
            :func:`filter_pathways_for_scoring`).
        normal_mask: boolean Series marking the normal reference samples; used
            only for the per-pathway reference percentiles.
        min_metabolites: minimum usable metabolites for a score to be emitted.

    Returns:
        Tuple ``(scores, reference)``:

        - ``scores``: long table with columns ``sample_id``, ``smp_id``,
          ``pathway_name``, ``n_metabolites_used``, ``z_stouffer``,
          ``z_stouffer_abs`` (one row per sample x pathway).
        - ``reference``: per-pathway normal reference with columns ``smp_id``,
          ``pathway_name``, ``n_metabolites`` (total usable metabolites),
          ``normal_z_stouffer_abs_p50/p95/p99`` (empirical percentiles of the
          absolute Stouffer score over the normal samples).
    """
    score_cols = ["sample_id", "smp_id", "pathway_name",
                  "n_metabolites_used", "z_stouffer", "z_stouffer_abs"]
    ref_cols = ["smp_id", "pathway_name", "n_metabolites",
                "normal_z_stouffer_abs_p50", "normal_z_stouffer_abs_p95",
                "normal_z_stouffer_abs_p99"]
    if zscores.empty or scored_coverage.empty:
        return (pd.DataFrame(columns=score_cols),
                pd.DataFrame(columns=ref_cols))

    normal_mask = normal_mask.reindex(zscores.index, fill_value=False)

    kept_pathways = scored_coverage[["smp_id", "pathway_name"]].drop_duplicates()
    links = feature_to_pathway.merge(kept_pathways,
                                     on=["smp_id", "pathway_name"], how="inner")
    links = links[links["feature"].isin(zscores.columns)
                  & links["hmdb_id"].notna()]

    # Pathway -> {metabolite (hmdb_id) -> [features]}
    pathway_metabolite_features: Dict[str, Dict[str, List[str]]] = {}
    for _, row in links.iterrows():
        pathway_metabolite_features.setdefault(row["smp_id"], {}).setdefault(
            row["hmdb_id"], []).append(row["feature"])

    score_rows: List[Dict] = []
    ref_rows: List[Dict] = []
    n_scored_pathways = 0
    for smp_id, metabolites in pathway_metabolite_features.items():
        metabolite_z = []
        metabolite_names = []
        for hmdb_id, feats in metabolites.items():
            feats = [f for f in feats if f in zscores.columns]
            if feats:
                metabolite_z.append(zscores[feats].mean(axis=1))
                metabolite_names.append(hmdb_id)
        if len(metabolite_z) < min_metabolites:
            continue
        n_scored_pathways += 1
        metab_matrix = pd.concat(metabolite_z, axis=1)

        usable = metab_matrix.notna().sum(axis=1)
        k_eff = usable.astype(float)
        signed = metab_matrix.sum(axis=1) / np.sqrt(k_eff.where(k_eff > 0))
        absolute = metab_matrix.abs().sum(axis=1) / np.sqrt(k_eff.where(k_eff > 0))
        signed[usable < min_metabolites] = np.nan
        absolute[usable < min_metabolites] = np.nan

        pathway_name = kept_pathways.loc[
            kept_pathways["smp_id"] == smp_id, "pathway_name"].iloc[0]

        for sample_id in zscores.index:
            score_rows.append({
                "sample_id": sample_id,
                "smp_id": smp_id,
                "pathway_name": pathway_name,
                "n_metabolites_used": int(usable.loc[sample_id]),
                "z_stouffer": signed.loc[sample_id],
                "z_stouffer_abs": absolute.loc[sample_id],
            })

        normal_abs = absolute.loc[normal_mask].dropna()
        ref_rows.append({
            "smp_id": smp_id,
            "pathway_name": pathway_name,
            "n_metabolites": len(metabolite_names),
            "normal_z_stouffer_abs_p50": float(normal_abs.quantile(0.50)) if len(normal_abs) else np.nan,
            "normal_z_stouffer_abs_p95": float(normal_abs.quantile(0.95)) if len(normal_abs) else np.nan,
            "normal_z_stouffer_abs_p99": float(normal_abs.quantile(0.99)) if len(normal_abs) else np.nan,
        })

    scores = pd.DataFrame(score_rows, columns=score_cols)
    reference = pd.DataFrame(ref_rows, columns=ref_cols)
    logger.info(f"Computed Stouffer scores for {n_scored_pathways} pathways x "
                f"{zscores.index.nunique()} samples "
                f"(min {min_metabolites} usable metabolites per score).")
    return scores, reference


def flag_pathway_scores(pathway_scores: pd.DataFrame,
                         normal_mask: pd.Series,
                         threshold_percentile: float = 99.0
                         ) -> pd.DataFrame:
    """Flag (sample, pathway) pairs against each pathway's own normal range.

    For every pathway the threshold is the ``threshold_percentile`` percentile
    of the absolute Stouffer score over the NORMAL samples only (empirical
    calibration: a noisy pathway automatically gets a wider range). A pair is
    flagged when its ``z_stouffer_abs`` strictly exceeds its pathway's
    threshold.

    Args:
        pathway_scores: output of :func:`compute_stouffer_scores`.
        normal_mask: boolean Series (sample_id -> is-normal).
        threshold_percentile: percentile of the normals' absolute score used
            as the per-pathway threshold (default 99).

    Returns:
        DataFrame like ``pathway_scores`` plus ``threshold`` and ``excess``
        (z_stouffer_abs / threshold, > 1 when flagged) and ``flagged``.
    """
    flag_cols = list(pathway_scores.columns) + ["threshold", "excess", "flagged"]
    if pathway_scores.empty:
        return pd.DataFrame(columns=flag_cols)

    normal_mask = normal_mask.reindex(pathway_scores["sample_id"].unique(),
                                      fill_value=False)
    normal_scores = pathway_scores[pathway_scores["sample_id"].map(normal_mask)]

    thresholds = (
        normal_scores.groupby("smp_id")["z_stouffer_abs"]
        .quantile(threshold_percentile / 100.0)
        .rename("threshold")
        .reset_index()
    )
    flags = pathway_scores.merge(thresholds, on="smp_id", how="left")
    flags["excess"] = flags["z_stouffer_abs"] / flags["threshold"]
    flags["flagged"] = flags["excess"] > 1.0
    flags = flags[flag_cols]
    n_flagged_pairs = int(flags["flagged"].sum())
    logger.info(f"Flagged {n_flagged_pairs} (sample, pathway) pairs at the "
                f"{threshold_percentile}th normal percentile "
                f"({flags['smp_id'].nunique()} pathways scored).")
    return flags


def summarize_sample_flags(pathway_flags: pd.DataFrame,
                            min_flagged_pathways: int = 1
                            ) -> pd.DataFrame:
    """Summarize the per-pathway flags into a per-sample decision.

    A sample is flagged when at least ``min_flagged_pathways`` of its pathways
    are flagged. The summary keeps the sample's top evidence (largest excess
    ratio) for review.

    Args:
        pathway_flags: output of :func:`flag_pathway_scores`.
        min_flagged_pathways: minimum flagged pathways for a sample decision.

    Returns:
        DataFrame with one row per sample: ``sample_id``,
        ``n_flagged_pathways``, ``n_scored_pathways``, ``flagged``,
        ``top_pathway_name``, ``top_z_stouffer_abs``, ``top_excess``.
    """
    out_cols = ["sample_id", "n_flagged_pathways", "n_scored_pathways", "flagged",
                "top_pathway_name", "top_z_stouffer_abs", "top_excess"]
    if pathway_flags.empty:
        return pd.DataFrame(columns=out_cols)

    def _agg(g):
        candidates = g.dropna(subset=["excess"])
        if candidates.empty:
            top = g.iloc[0]
        else:
            top = candidates.loc[candidates["excess"].idxmax()]
        return pd.Series({
            "n_flagged_pathways": int(g["flagged"].sum()),
            "n_scored_pathways": int(g["flagged"].sum() + (~g["flagged"].astype(bool)).sum()),
            "flagged": bool(g["flagged"].sum() >= min_flagged_pathways),
            "top_pathway_name": top["pathway_name"],
            "top_z_stouffer_abs": top["z_stouffer_abs"],
            "top_excess": top["excess"],
        })

    group_cols = [c for c in pathway_flags.columns if c != "sample_id"]
    summary = (pathway_flags[group_cols]
               .groupby(pathway_flags["sample_id"].values)
               .apply(_agg)
               .reset_index()
               .rename(columns={"index": "sample_id"}))
    n_flagged_samples = int(summary["flagged"].sum())
    logger.info(f"Flagged {n_flagged_samples} of {len(summary)} samples "
                f"(>= {min_flagged_pathways} flagged pathway(s)).")
    return summary[out_cols]
