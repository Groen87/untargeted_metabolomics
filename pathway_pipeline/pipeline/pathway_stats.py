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
