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
from math import exp, lgamma, log
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
                               iqr_scale: bool = True,
                               min_reference_scale: float = None
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
        min_reference_scale: drop features whose reference scale is below
            this floor (in the feature's log10 units); a razor-thin normal
            spread turns trivial absolute shifts into huge z-scores
            (noise-floor z-magnifiers). None disables the floor.

    Returns:
        Tuple ``(zscores, reference_stats, dropped_features)``:

        - ``zscores``: DataFrame of the same sample index with one z-score
          column per calibrated feature (dropped features removed).
        - ``reference_stats``: per-feature reference table with columns
          ``feature``, ``median``, ``scale``, ``n_normal_values``,
          ``p_normal_missing``.
        - ``dropped_features``: per dropped feature with columns ``feature``
          and ``reason`` ('no_normal_values' / 'zero_scale' / 'small_scale').
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
        elif min_reference_scale is not None and scales[col] < min_reference_scale:
            dropped_rows.append({"feature": col, "reason": "small_scale"})
        else:
            keep.append(col)

    dropped = pd.DataFrame(dropped_rows, columns=["feature", "reason"])
    if len(dropped):
        n_small = int((dropped["reason"] == "small_scale").sum())
        logger.info(f"Dropped {len(dropped)} features before z-scoring "
                    f"({int((dropped['reason'] == 'zero_scale').sum())} zero-scale, "
                    f"{int((dropped['reason'] == 'no_normal_values').sum())} without "
                    f"normal values"
                    + (f", {n_small} below the {min_reference_scale} scale "
                       f"floor" if n_small else "") + ").")

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


def prune_redundant_pathways(scored_coverage: pd.DataFrame,
                             min_jaccard: float = 0.8
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Collapse near-duplicate pathways to one representative per group.

    PathBank disease pathways are mechanism cartoons sharing a handful of
    metabolites with their parent metabolic pathway: five pathways driven
    by the same three metabolites produce five identical flags and inflate
    multiplicity. Pathways whose SCORED metabolite sets have Jaccard >=
    ``min_jaccard`` are grouped (connected components over the similarity
    graph) and each group keeps a single representative, chosen by a
    pre-declared, composition-based preference -- never by flag
    performance and never by a disease label:

    1. General metabolic pathway over a disease-specific cartoon
       (name without 'deficiency'/'disease'/'aciduria', case-insensitive).
    2. Larger scored metabolite set (more information).
    3. Alphabetical by pathway name (determinism).

    Args:
        scored_coverage: output of :func:`filter_pathways_for_scoring`
            (one row per scored pathway, with ``matched_metabolites``).
        min_jaccard: Jaccard threshold above which two pathways are
            near-duplicates.

    Returns:
        Tuple ``(pruned, dropped)``: the coverage table reduced to the
        representatives, and a per-dropped-pathway table with columns
        ``smp_id``, ``pathway_name``, ``represented_by``.
    """
    drop_cols = ["smp_id", "pathway_name", "represented_by"]
    if scored_coverage.empty:
        return scored_coverage.copy(), pd.DataFrame(columns=drop_cols)

    metabolite_sets = {
        row["smp_id"]: set(str(row["matched_metabolites"]).split(";"))
        - {""}
        for _, row in scored_coverage.iterrows()
    }
    metabolite_sets = {k: v for k, v in metabolite_sets.items() if v}
    names = dict(zip(scored_coverage["smp_id"],
                     scored_coverage["pathway_name"]))
    smp_ids = sorted(metabolite_sets)

    # Connected components over the >= min_jaccard similarity graph.
    parent = {s: s for s in smp_ids}

    def find(s):
        while parent[s] != s:
            parent[s] = parent[parent[s]]
            s = parent[s]
        return s

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(smp_ids)):
        for j in range(i + 1, len(smp_ids)):
            a, b = metabolite_sets[smp_ids[i]], metabolite_sets[smp_ids[j]]
            union_size = len(a | b)
            if union_size and len(a & b) / union_size >= min_jaccard:
                union(smp_ids[i], smp_ids[j])

    groups = {}
    for s in smp_ids:
        groups.setdefault(find(s), []).append(s)

    def is_general(smp_id):
        name = str(names.get(smp_id, "")).lower()
        return not any(k in name for k in ("deficiency", "disease", "aciduria"))

    def rep_key(smp_id):
        return (0 if is_general(smp_id) else 1,
                -len(metabolite_sets[smp_id]),
                str(names.get(smp_id, smp_id)))

    drop_rows = []
    keep = set()
    for members in groups.values():
        rep = sorted(members, key=rep_key)[0]
        keep.add(rep)
        for m in members:
            if m != rep:
                drop_rows.append({
                    "smp_id": m,
                    "pathway_name": names.get(m, m),
                    "represented_by": rep,
                })

    dropped = pd.DataFrame(drop_rows, columns=drop_cols)
    pruned = scored_coverage[scored_coverage["smp_id"].isin(keep)].copy()
    if len(dropped):
        logger.info(f"Pathway redundancy pruning: collapsed {len(dropped)} "
                    f"near-duplicate pathway(s) (Jaccard >= {min_jaccard}) "
                    f"into {len(groups)} representative group(s); "
                    f"{len(pruned)} pathways remain for scoring.")
        by_rep = dropped.groupby("represented_by")["pathway_name"]
        for rep, members in by_rep.agg(list).items():
            logger.debug(f"  kept {names.get(rep, rep)} over: {members}")
    return pruned, dropped


def compute_stouffer_scores(zscores: pd.DataFrame,
                             feature_to_pathway: pd.DataFrame,
                             scored_coverage: pd.DataFrame,
                             normal_mask: pd.Series,
                             min_metabolites: int = 3,
                             max_abs_z: float = None,
                             feature_scale_weights: Dict[str, float] = None
                             ) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        max_abs_z: cap on |z| applied before the Stouffer sum (a single
            artifact feature can otherwise dominate a whole pathway); None
            disables the cap.
        feature_scale_weights: optional feature -> weight map (typically
            the reference scale squared). When given, features mapping to
            the same metabolite (HMDB ID) are combined as a weighted
            average instead of a plain mean, so a razor-thin duplicate
            feature (a noise-floor z-magnifier) cannot dominate the
            metabolite's z. Missing features get weight 1.

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

    weights = feature_scale_weights or {}

    if max_abs_z is not None and max_abs_z > 0:
        zscores = zscores.clip(lower=-max_abs_z, upper=max_abs_z)
        logger.info(f"Capping metabolite |z| at {max_abs_z} before the "
                    f"Stouffer sum.")

    normal_mask = normal_mask[~normal_mask.index.duplicated(keep="first")]
    normal_mask = normal_mask.reindex(zscores.index, fill_value=False)
    normal_positions = np.flatnonzero(normal_mask.to_numpy())

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
                if len(feats) == 1 or not weights:
                    metabolite_z.append(zscores[feats].mean(axis=1))
                else:
                    sub = zscores[feats]
                    w = pd.Series({f: float(weights.get(f, 1.0))
                                   for f in feats})
                    present_w = sub.notna().mul(w, axis=1).sum(axis=1)
                    weighted = (sub.fillna(0.0).mul(w, axis=1).sum(axis=1)
                                / present_w.where(present_w > 0))
                    metabolite_z.append(weighted)
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

        for pos, sample_id in enumerate(zscores.index):
            score_rows.append({
                "sample_id": sample_id,
                "smp_id": smp_id,
                "pathway_name": pathway_name,
                "n_metabolites_used": int(usable.iloc[pos]),
                "z_stouffer": signed.iloc[pos],
                "z_stouffer_abs": absolute.iloc[pos],
            })

        normal_abs = absolute.iloc[normal_positions].dropna()
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


def _binomial_sf(k: int, n: int, p: float) -> float:
    """Survival function P(X >= k) for X ~ Binomial(n, p).

    Exact tail sum in log space (no SciPy dependency); underflows to 0 for
    vanishingly small tails.

    Args:
        k: minimum number of successes.
        n: number of trials.
        p: per-trial success probability.

    Returns:
        P(X >= k); 1.0 for k <= 0, 0.0 for k > n.
    """
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    log_p = log(p)
    log_1mp = log(1.0 - p)
    total = 0.0
    for i in range(k, n + 1):
        log_term = (lgamma(n + 1) - lgamma(i + 1) - lgamma(n - i + 1)
                    + i * log_p + (n - i) * log_1mp)
        total += exp(log_term)
    return min(total, 1.0)


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

    normal_mask = normal_mask[~normal_mask.index.duplicated(keep="first")]
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


def flag_metabolite_scores(zscores: pd.DataFrame,
                           normal_mask: pd.Series,
                           threshold_percentile: float = 99.0
                           ) -> pd.DataFrame:
    """Flag metabolite z-scores against each metabolite's own normal range.

    Mirrors :func:`flag_pathway_scores` at the metabolite level: a metabolite
    is flagged for a sample when its absolute z-score exceeds the
    ``threshold_percentile`` percentile of that metabolite's absolute
    z-scores over the reference normals. Per-metabolite calibration absorbs
    noisy features (a metabolite with wide normal spread gets a wider range).

    Args:
        zscores: metabolite z-score matrix (samples x metabolites).
        normal_mask: boolean Series marking the normal reference samples.
        threshold_percentile: percentile of the normal |z| distribution used
            as the per-metabolite threshold.

    Returns:
        Long-format flags: one row per (sample, metabolite) with
        ``abs_z``, ``threshold``, ``excess``, ``flagged``.
    """
    out_cols = ["sample_id", "metabolite", "abs_z", "threshold",
                "excess", "flagged"]
    if zscores.empty:
        return pd.DataFrame(columns=out_cols)
    abs_z = zscores.abs()
    index_name = abs_z.index.name or "sample_id"
    abs_z = abs_z.rename_axis(index_name)
    dedup = normal_mask[~normal_mask.index.duplicated(keep="first")]
    normal_values = abs_z[dedup.reindex(abs_z.index, fill_value=False)]
    with np.errstate(all="ignore"):
        thresholds = normal_values.quantile(
            threshold_percentile / 100.0, axis=0)
    thresholds = thresholds.fillna(float("inf"))
    flags = (abs_z.reset_index()
             .melt(id_vars=index_name, var_name="metabolite",
                   value_name="abs_z")
             .rename(columns={index_name: "sample_id"}))
    flags["threshold"] = flags["metabolite"].map(thresholds)
    flags["excess"] = flags["abs_z"] / flags["threshold"]
    flags["flagged"] = flags["excess"] > 1.0
    n_flagged = int(flags["flagged"].sum())
    logger.info(f"Flagged {n_flagged} (sample, metabolite) pairs at the "
                f"{threshold_percentile}th normal percentile "
                f"({flags['metabolite'].nunique()} metabolites scored).")
    return flags[out_cols]


def summarize_metabolite_flags(metabolite_flags: pd.DataFrame,
                               normal_mask: pd.Series
                               ) -> pd.DataFrame:
    """Per-sample metabolite-depth summary (report-only).

    The metabolite depth p-value is the fraction of reference normals whose
    maximum metabolite |z| reaches at least the sample's maximum. It is the
    metabolite-level analogue of the max_excess pathway rule: a sample with
    one grossly elevated metabolite beats it; a sample with a broad mild
    shift does not.

    Args:
        metabolite_flags: output of :func:`flag_metabolite_scores`.
        normal_mask: boolean Series marking the normal reference samples.

    Returns:
        DataFrame with one row per sample: ``sample_id``,
        ``max_metabolite_z``, ``n_flagged_metabolites``,
        ``metabolite_depth_p``, ``top_metabolite``,
        ``top_metabolite_z``.
    """
    out_cols = ["sample_id", "max_metabolite_z", "n_flagged_metabolites",
                "metabolite_depth_p", "top_metabolite", "top_metabolite_z"]
    if metabolite_flags.empty:
        return pd.DataFrame(columns=out_cols)
    dedup = normal_mask[~normal_mask.index.duplicated(keep="first")]
    is_normal = dedup.reindex(metabolite_flags["sample_id"].unique(),
                              fill_value=False)
    normal_samples = [s for s, v in is_normal.items() if v]
    normal_max = (metabolite_flags[metabolite_flags["sample_id"]
                                   .isin(normal_samples)]
                  .groupby("sample_id")["abs_z"].max())
    logger.info(f"Metabolite-depth null: maximum |z| of {len(normal_max)} "
                f"normals (median {float(normal_max.median()):.2f}, "
                f"p95 {float(normal_max.quantile(0.95)):.2f}).")

    def _agg(g):
        top = g.loc[g["abs_z"].idxmax()]
        max_z = float(top["abs_z"])
        p = float((normal_max >= max_z).mean()) if len(normal_max) else float("nan")
        return pd.Series({
            "max_metabolite_z": max_z,
            "n_flagged_metabolites": int(g["flagged"].sum()),
            "metabolite_depth_p": p,
            "top_metabolite": top["metabolite"],
            "top_metabolite_z": max_z,
        })

    summary = (metabolite_flags.groupby("sample_id", sort=False)
               .apply(_agg).reset_index())
    return summary[out_cols]


def summarize_sample_flags(pathway_flags: pd.DataFrame,
                            min_flagged_pathways: int = 1,
                            per_pathway_flag_rate: float = None,
                            max_sample_p: float = 0.05,
                            normal_mask: pd.Series = None,
                            sample_rule: str = "empirical"
                            ) -> pd.DataFrame:
    """Summarize the per-pathway flags into a per-sample decision.

    With 200+ pathways per sample, a few chance pathway flags are expected
    for every sample. The decision combines the count rule (at least
    ``min_flagged_pathways`` flagged) with a null model:

    - ``sample_rule='empirical'``: the null is the observed flagged-pathway
      count distribution of the NORMAL samples. PathBank pathways share
      metabolites, so flags are correlated and the binomial null is
      anti-conservative; the empirical distribution absorbs that correlation
      automatically. ``sample_p_value`` is the fraction of normals with at
      least as many flagged pathways. Measures breadth.
    - ``sample_rule='max_excess'``: the null is the observed distribution of
      each normal's maximum pathway excess. ``sample_p_value`` is the
      fraction of normals whose most extreme pathway reaches at least the
      sample's maximum excess. Measures depth: a sample disturbing a few
      pathways profoundly beats it, a sample shifting many pathways mildly
      does not.
    - ``sample_rule='binomial'``: the null is
      Binomial(n_scored_pathways, per_pathway_flag_rate); valid only when
      pathway flags are (near-)independent.
    - ``sample_rule='none'``: count rule only.

    Args:
        pathway_flags: output of :func:`flag_pathway_scores`.
        min_flagged_pathways: minimum flagged pathways for a sample decision.
        per_pathway_flag_rate: per-pathway flag rate for the binomial rule
            (typically 1 - threshold_percentile/100).
        max_sample_p: p-value cutoff for the null rules (default 0.05).
        normal_mask: boolean Series (sample_id -> is-normal); required by
            the empirical and max_excess rules (falls back with a warning
            when absent).
        sample_rule: 'empirical', 'max_excess', 'binomial', or 'none'.

    Returns:
        DataFrame with one row per sample: ``sample_id``,
        ``n_flagged_pathways``, ``n_scored_pathways``, ``flagged``,
        ``sample_p_value`` (NaN when no null rule applies),
        ``top_pathway_name``, ``top_z_stouffer_abs``, ``top_excess``.
    """
    out_cols = ["sample_id", "n_flagged_pathways", "n_scored_pathways", "flagged",
                "sample_p_value", "top_pathway_name", "top_z_stouffer_abs",
                "top_excess"]
    if pathway_flags.empty:
        return pd.DataFrame(columns=out_cols)

    sample_rule = str(sample_rule).lower()
    if sample_rule in ("empirical", "max_excess"):
        if normal_mask is None or not normal_mask.any():
            logger.warning(f"{sample_rule} sample rule needs normal samples; "
                           "falling back to the count rule.")
            sample_rule = "none"
    elif sample_rule == "binomial" and per_pathway_flag_rate is None:
        logger.warning("Binomial sample rule needs per_pathway_flag_rate; "
                       "falling back to the count rule.")
        sample_rule = "none"

    flagged_counts = pathway_flags.groupby("sample_id")["flagged"].sum()

    normal_counts = None
    normal_max_excess = None
    if sample_rule == "empirical":
        dedup = normal_mask[~normal_mask.index.duplicated(keep="first")]
        is_normal = dedup.reindex(flagged_counts.index, fill_value=False)
        normal_counts = flagged_counts[is_normal.to_numpy()]
        logger.info(f"Empirical null: flagged-pathway counts of "
                    f"{len(normal_counts)} normals (median "
                    f"{float(normal_counts.median()):.1f}, p99 "
                    f"{float(normal_counts.quantile(0.99)):.1f}).")
    if sample_rule == "max_excess":
        excess = pathway_flags.dropna(subset=["excess"])
        if excess.empty:
            max_excess_by_sample = pd.Series(dtype=float)
        else:
            max_excess_by_sample = excess.groupby("sample_id")["excess"].max()
        dedup = normal_mask[~normal_mask.index.duplicated(keep="first")]
        is_normal = dedup.reindex(max_excess_by_sample.index, fill_value=False)
        normal_max_excess = max_excess_by_sample[is_normal.to_numpy()]
        logger.info(f"Max-excess null: maximum pathway excess of "
                    f"{len(normal_max_excess)} normals (median "
                    f"{float(normal_max_excess.median()):.2f}, p95 "
                    f"{float(normal_max_excess.quantile(0.95)):.2f}).")

    def _agg(g):
        candidates = g.dropna(subset=["excess"])
        if candidates.empty:
            top = g.iloc[0]
        else:
            top = candidates.loc[candidates["excess"].idxmax()]
        n_flagged = int(g["flagged"].sum())
        n_scored = int(g["flagged"].sum() + (~g["flagged"].astype(bool)).sum())
        if sample_rule == "empirical":
            p_value = float((normal_counts >= n_flagged).mean())
        elif sample_rule == "max_excess":
            if normal_max_excess.empty or not candidates.empty:
                p_value = float(
                    (normal_max_excess >= float(top["excess"])).mean())
            else:
                p_value = float("nan")
        elif sample_rule == "binomial":
            p_value = _binomial_sf(n_flagged, n_scored, per_pathway_flag_rate)
        else:
            p_value = float("nan")
        if sample_rule == "none":
            decision = n_flagged >= min_flagged_pathways
        else:
            decision = (n_flagged >= min_flagged_pathways
                        and p_value <= max_sample_p)
        return pd.Series({
            "n_flagged_pathways": n_flagged,
            "n_scored_pathways": n_scored,
            "flagged": bool(decision),
            "sample_p_value": p_value,
            "top_pathway_name": top["pathway_name"],
            "top_z_stouffer_abs": top["z_stouffer_abs"],
            "top_excess": top["excess"],
        })

    summary = (pathway_flags.drop(columns=["sample_id"])
               .groupby(pathway_flags["sample_id"], sort=False)
               .apply(_agg)
               .reset_index())
    n_flagged_samples = int(summary["flagged"].sum())
    if sample_rule == "empirical":
        logger.info(f"Flagged {n_flagged_samples} of {len(summary)} samples "
                    f"(>= {min_flagged_pathways} flagged pathway(s) AND "
                    f"empirical p <= {max_sample_p} against the normals).")
    elif sample_rule == "max_excess":
        logger.info(f"Flagged {n_flagged_samples} of {len(summary)} samples "
                    f"(>= {min_flagged_pathways} flagged pathway(s) AND "
                    f"max-excess p <= {max_sample_p} against the normals).")
    elif sample_rule == "binomial":
        logger.info(f"Flagged {n_flagged_samples} of {len(summary)} samples "
                    f"(>= {min_flagged_pathways} flagged pathway(s) AND "
                    f"binomial p <= {max_sample_p} at rate "
                    f"{per_pathway_flag_rate:.2%}).")
    else:
        logger.info(f"Flagged {n_flagged_samples} of {len(summary)} samples "
                    f"(>= {min_flagged_pathways} flagged pathway(s)).")
    return summary[out_cols]



