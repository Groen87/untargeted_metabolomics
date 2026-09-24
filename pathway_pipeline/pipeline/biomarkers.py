"""Literature-biomarker attachment channel (parallel to PathBank pathways).

PathBank disease pathways are intracellular mechanism cartoons and often
omit the clinically diagnostic biomarkers of the disease they depict
(e.g. the MCADD pathway without octanoylcarnitine). This channel injects
curated prior knowledge -- published disease biomarkers attached to their
disease pathways -- as a SEPARATE, openly declared channel: the PathBank
pathway definitions are never modified, so any performance difference
between the channels is attributable and auditable.

Evidence budget #3: the attachment table is gene/disease-level textbook
knowledge assembled from literature sources (cited per row in the
``source`` column), frozen in the configuration before any evaluation
read. It is label-blind by construction -- the same table would be
declared for any cohort. Attachments are NEVER chosen by looking at
which samples the pipeline flagged or missed.

A sample flags a disease pathway through this channel when an attached
biomarker's z-score exceeds its own normal-percentile threshold AND the
sample's maximum attached-biomarker |z| beats the biomarker-restricted
depth null of the reference normals (the metabolite-level analogue of
the pathway max_excess rule). The channel ORs into the sample decision.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from pathway_pipeline.pipeline.pathway_stats import (
    flag_metabolite_scores,
    summarize_metabolite_flags,
)

logger = logging.getLogger(__name__)

ATTACHMENT_COLUMNS = ["smp_id", "pathway_name", "hmdb_id", "source"]
RESOLVED_COLUMNS = ATTACHMENT_COLUMNS + ["features", "n_features"]


def load_biomarker_attachments(csv_path: str) -> pd.DataFrame:
    """Load the curated biomarker -> pathway attachment table.

    The CSV is user-curated (literature biomarker tables) with columns
    ``smp_id`` OR ``pathway_name`` (the target disease pathway),
    ``hmdb_id`` (the biomarker's HMDB accession), and ``source``
    (citation for the biomarker-disease link). A missing file disables
    the channel gracefully (the curator has not provided a list yet).

    Returns:
        DataFrame with ``smp_id``, ``pathway_name``, ``hmdb_id``,
        ``source``; empty when the file is absent or unreadable.
    """
    path = Path(csv_path)
    if not path.exists():
        logger.warning(f"Biomarker attachments file not found ({csv_path}); "
                       "biomarker channel disabled until the table is "
                       "provided.")
        return pd.DataFrame(columns=ATTACHMENT_COLUMNS)
    try:
        table = pd.read_csv(path)
    except (OSError, pd.errors.ParserError) as exc:
        logger.warning(f"Could not read biomarker attachments file "
                       f"({csv_path}): {exc}; channel disabled.")
        return pd.DataFrame(columns=ATTACHMENT_COLUMNS)

    table.columns = [str(c).strip().lower() for c in table.columns]
    for col in ("smp_id", "pathway_name", "hmdb_id", "source"):
        if col not in table.columns:
            table[col] = None
    table = table[ATTACHMENT_COLUMNS].copy()
    for col in ("smp_id", "pathway_name", "hmdb_id", "source"):
        table[col] = (table[col].astype("string").str.strip()
                      if table[col].notna().any() else pd.NA)
    table["hmdb_id"] = table["hmdb_id"].str.upper()
    before = len(table)
    table = table[table["hmdb_id"].notna()
                  & (table["hmdb_id"] != "")
                  & (table["smp_id"].notna() | table["pathway_name"].notna())]
    dropped = before - len(table)
    if dropped:
        logger.warning(f"Dropped {dropped} attachment row(s) without an "
                       "hmdb_id or without a pathway reference.")
    logger.info(f"Loaded {len(table)} biomarker attachment(s) from {csv_path}.")
    return table.reset_index(drop=True)


def resolve_biomarker_attachments(attachments: pd.DataFrame,
                                   feature_to_hmdb: pd.DataFrame,
                                   coverage: pd.DataFrame
                                   ) -> Tuple[pd.DataFrame, List[str]]:
    """Resolve attachment targets to kept pathways and dataset features.

    Target pathways are matched by ``smp_id`` when given, otherwise by
    exact ``pathway_name`` (case-insensitive); attachments pointing at
    pathways that did not survive the coverage/keyword filters are
    dropped with a warning (they cannot be scored). Biomarker HMDB IDs
    are resolved to dataset feature columns via the feature -> HMDB
    mapping; biomarkers with no matched feature are dropped with a
    warning.

    Returns:
        Tuple ``(resolved, features)``: the resolved attachment table
        (``smp_id``, ``pathway_name``, ``hmdb_id``, ``source``,
        ``features`` ';'-joined, ``n_features``) and the sorted list of
        dataset features carrying attached biomarkers (these must be
        z-scored even when no kept PathBank pathway maps them).
    """
    empty = pd.DataFrame(columns=RESOLVED_COLUMNS)
    if attachments.empty:
        return empty, []
    if coverage.empty:
        logger.warning("No kept pathways; all biomarker attachments are "
                       "unresolvable.")
        return empty, []

    cov = coverage[["smp_id", "pathway_name"]].drop_duplicates().copy()
    cov["name_key"] = cov["pathway_name"].astype("string").str.strip().str.lower()
    name_to_smp = {}
    ambiguous = set()
    for _, row in cov.iterrows():
        key = row["name_key"]
        if pd.isna(key) or key == "":
            continue
        if key in name_to_smp and name_to_smp[key] != row["smp_id"]:
            ambiguous.add(key)
        name_to_smp[key] = row["smp_id"]

    feature_sets = (feature_to_hmdb.dropna(subset=["hmdb_id"])
                    .groupby("hmdb_id")["feature"]
                    .agg(lambda s: sorted(set(s))))

    rows = []
    unmatched_pathways = []
    unmatched_features = []
    for _, att in attachments.iterrows():
        smp_id = att["smp_id"] if pd.notna(att["smp_id"]) else None
        if smp_id is None or pd.isna(smp_id):
            key = str(att["pathway_name"]).strip().lower()
            if key in ambiguous:
                logger.warning(f"Ambiguous pathway name "
                               f"'{att['pathway_name']}' for biomarker "
                               f"{att['hmdb_id']}; attachment dropped.")
                continue
            smp_id = name_to_smp.get(key)
        if smp_id is None or smp_id not in set(cov["smp_id"]):
            unmatched_pathways.append(str(att.get("pathway_name") or smp_id))
            continue
        name = cov.loc[cov["smp_id"] == smp_id, "pathway_name"].iloc[0]
        feats = feature_sets.get(att["hmdb_id"], [])
        if not feats:
            unmatched_features.append(att["hmdb_id"])
            continue
        rows.append({
            "smp_id": smp_id,
            "pathway_name": name,
            "hmdb_id": att["hmdb_id"],
            "source": att["source"] if pd.notna(att["source"]) else None,
            "features": ";".join(feats),
            "n_features": len(feats),
        })

    if unmatched_pathways:
        logger.warning(f"{len(unmatched_pathways)} attachment(s) target "
                       "pathways outside the kept set (dropped from the "
                       f"channel): {sorted(set(unmatched_pathways))}")
    if unmatched_features:
        logger.warning(f"{len(unmatched_features)} attached biomarker(s) "
                       "match no dataset feature (dropped from the "
                       f"channel): {sorted(set(unmatched_features))}")

    resolved = pd.DataFrame(rows, columns=RESOLVED_COLUMNS)
    features = sorted({f for feats in resolved["features"] for f in
                       str(feats).split(";") if f})
    logger.info(f"Resolved {len(resolved)} biomarker attachment(s) to kept "
                f"pathways ({resolved['smp_id'].nunique() if len(resolved) else 0} "
                f"pathways, {len(features)} dataset features).")
    return resolved, features


def aggregate_metabolite_zscores(zscores: pd.DataFrame,
                                 feature_to_hmdb: pd.DataFrame,
                                 hmdb_ids: List[str],
                                 feature_scale_weights: Dict[str, float] = None
                                 ) -> pd.DataFrame:
    """Combine per-feature z-scores into per-metabolite (HMDB) z-scores.

    Features mapping to the same HMDB ID are averaged (scale^2 weighted
    when weights are given -- the same rule as the Stouffer channel),
    mirroring how a metabolite's z is treated inside a pathway score.
    Biomarkers with no z-scored feature are skipped with a warning.
    """
    if not hmdb_ids:
        return pd.DataFrame(index=zscores.index)
    id_to_features = (feature_to_hmdb[feature_to_hmdb["hmdb_id"]
                                       .isin(hmdb_ids)]
                      .groupby("hmdb_id")["feature"]
                      .agg(lambda s: sorted(set(s))))
    columns = {}
    missing = []
    for hmdb_id in dict.fromkeys(hmdb_ids):
        feats = [f for f in id_to_features.get(hmdb_id, [])
                 if f in zscores.columns]
        if not feats:
            missing.append(hmdb_id)
            continue
        if len(feats) == 1 or not feature_scale_weights:
            columns[hmdb_id] = zscores[feats].mean(axis=1)
        else:
            sub = zscores[feats]
            w = pd.Series({f: float(feature_scale_weights.get(f, 1.0))
                           for f in feats})
            present_w = sub.notna().mul(w, axis=1).sum(axis=1)
            columns[hmdb_id] = (sub.fillna(0.0).mul(w, axis=1).sum(axis=1)
                                / present_w.where(present_w > 0))
    if missing:
        logger.warning(f"{len(missing)} attached biomarker(s) have no "
                       "z-scored feature (demoted or dropped before "
                       f"scoring); skipped: {sorted(set(missing))}")
    if not columns:
        return pd.DataFrame(index=zscores.index)
    return pd.DataFrame(columns, index=zscores.index)


def flag_biomarker_attachments(zscores: pd.DataFrame,
                               resolved_attachments: pd.DataFrame,
                               feature_to_hmdb: pd.DataFrame,
                               normal_mask: pd.Series,
                               threshold_percentile: float = 99.0,
                               feature_scale_weights: Dict[str, float] = None
                               ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag samples through attached biomarkers (per pathway and sample).

    Reuses the metabolite-flag machinery on the biomarker-restricted
    z-matrix: per-biomarker thresholds are the ``threshold_percentile``
    percentile of the reference normals' |z|, and the per-sample
    ``biomarker_depth_p`` is the fraction of reference normals whose
    maximum attached-biomarker |z| reaches the sample's maximum (the
    biomarker-restricted depth null). The channel decision is declared
    in ``main`` (at least one flagged biomarker AND depth p <= the
    frozen max_sample_p), ORed with the pathway-channel decision.

    Returns:
        Tuple ``(biomarker_flags, biomarker_summary)``:

        - ``biomarker_flags``: one row per (sample, pathway, biomarker)
          with ``abs_z``, ``threshold``, ``excess``, ``flagged``.
        - ``biomarker_summary``: one row per sample with
          ``biomarker_flagged``, ``n_flagged_biomarkers``,
          ``biomarker_depth_p``, ``max_biomarker_z``, ``top_biomarker``,
          ``top_biomarker_z``.
    """
    flag_cols = ["sample_id", "smp_id", "pathway_name", "hmdb_id",
                 "abs_z", "threshold", "excess", "flagged"]
    summary_cols = ["sample_id", "biomarker_flagged", "n_flagged_biomarkers",
                    "biomarker_depth_p", "max_biomarker_z",
                    "top_biomarker", "top_biomarker_z"]
    if resolved_attachments.empty or zscores.empty:
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))

    biomarker_z = aggregate_metabolite_zscores(
        zscores, feature_to_hmdb,
        list(resolved_attachments["hmdb_id"].unique()),
        feature_scale_weights=feature_scale_weights)
    if biomarker_z.empty:
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))

    metabolite_flags = flag_metabolite_scores(
        biomarker_z, normal_mask=normal_mask,
        threshold_percentile=threshold_percentile)
    if metabolite_flags.empty:
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))

    pathway_view = (resolved_attachments[["smp_id", "pathway_name", "hmdb_id"]]
                    .drop_duplicates())
    biomarker_flags = metabolite_flags.merge(
        pathway_view, left_on="metabolite", right_on="hmdb_id", how="inner")
    biomarker_flags = biomarker_flags[flag_cols]

    depth = summarize_metabolite_flags(metabolite_flags,
                                       normal_mask=normal_mask)
    depth = depth.rename(columns={
        "max_metabolite_z": "max_biomarker_z",
        "n_flagged_metabolites": "n_flagged_biomarkers",
        "metabolite_depth_p": "biomarker_depth_p",
        "top_metabolite": "top_biomarker",
        "top_metabolite_z": "top_biomarker_z",
    })
    depth["biomarker_flagged"] = (
        depth["n_flagged_biomarkers"] >= 1)
    depth = depth[summary_cols]
    return biomarker_flags, depth
