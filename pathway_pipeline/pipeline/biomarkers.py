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
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
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


# ---------------------------------------------------------------------------
# IEMbase-style disease biomarker table (Excel/CSV) -> disease-keyed channel
# ---------------------------------------------------------------------------

DISEASE_TABLE_LONG_COLUMNS = ["disease", "biomarker", "hmdb_id", "direction",
                              "omim", "smp_id", "pathway_name", "source"]
DISEASE_RESOLVED_COLUMNS = DISEASE_TABLE_LONG_COLUMNS + ["features",
                                                         "n_features"]
DISEASE_AUDIT_COLUMNS = ["disease", "biomarker", "hmdb_id", "direction",
                         "omim", "smp_id", "pathway_name", "pathway_status",
                         "features", "n_features", "status"]

_ARROWS = {"\u2191": "up", "\u2193": "down"}


def _split_top_level(text: str) -> List[str]:
    """Split on semicolons that are NOT inside parentheses/brackets."""
    parts, depth, current = [], 0, ""
    for char in str(text or ""):
        if char in "([":
            depth += 1
        elif char in ")]":
            depth = max(0, depth - 1)
        if char == ";" and depth == 0:
            parts.append(current)
            current = ""
        else:
            current += char
    parts.append(current)
    return [p.strip() for p in parts if p.strip()]


def _marker_direction(marker: str) -> str:
    for arrow, direction in _ARROWS.items():
        if arrow in marker:
            return direction
    return ""


def _clean_marker_name(marker: str) -> str:
    name = re.sub(r"[\u2191\u2193]", "", str(marker or ""))
    name = name.replace("*", " ").strip()
    return re.sub(r"\s+", " ", name)


def _clean_pathway_name(name: str) -> str:
    return re.sub(r"\s*\(PathBank PW\d+\)\s*$", "",
                  str(name or "").strip()).strip()


def _normalize_header(header: str) -> str:
    return re.sub(r"\s+", " ", str(header or "").strip().lower()
                  .replace("_", " "))


def _column_lookup(headers: List[str]) -> Dict[str, str]:
    """Map semantic roles to the actual header by prefix (case-insensitive)."""
    roles = {
        "disease": "disease",
        "omim": "omim",
        "markers": "biochemical markers",
        "pathway": "pathbank",
        "smp": "smpdb code",
        "hmdb": "hmdb codes",
    }
    lookup = {}
    for header in headers:
        norm = _normalize_header(header)
        for role, prefix in roles.items():
            if norm == prefix or norm.startswith(prefix):
                lookup.setdefault(role, header)
    return lookup


def _read_table_rows(table_path: str) -> List[List]:
    path = Path(table_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        try:
            import openpyxl
        except ImportError:
            logger.error(f"Reading {path.name} requires openpyxl "
                         "(pip install openpyxl); disease biomarker "
                         "table disabled.")
            return []
        workbook = openpyxl.load_workbook(str(path), read_only=True)
        sheet = workbook[workbook.sheetnames[0]]
        rows = [list(r) for r in sheet.iter_rows(values_only=True)]
        workbook.close()
        return rows
    frame = pd.read_csv(path)
    return [list(frame.columns)] + frame.astype(object).values.tolist()


def load_disease_biomarker_table(table_path: str) -> pd.DataFrame:
    """Load an IEMbase-style disease-biomarker table (Excel or CSV).

    Expected columns (header matching is prefix-based and
    case-insensitive): ``Disease``, ``OMIM``, ``Biochemical_Markers``
    (marker names with arrows: up, down, or no arrow for either
    direction), ``PathBank disease pathway``, ``SMPDB code (SMP)``, and
    ``HMDB codes of named metabolites`` (semicolon-separated entries
    positionally aligned with the markers; entries may carry multiple
    HMDB codes, and ``(no HMDB entry found)`` entries are skipped with
    a warning).

    Returns:
        DataFrame with one row per (disease, biomarker, HMDB code):
        ``disease``, ``biomarker``, ``hmdb_id``, ``direction`` ('up'/
        'down'/'' ), ``omim``, ``smp_id``, ``pathway_name``, ``source``.
        Empty when the file is absent or unreadable.
    """
    path = Path(table_path)
    if not path.exists():
        logger.warning(f"Disease biomarker table not found ({table_path}); "
                       "disease table source disabled.")
        return pd.DataFrame(columns=DISEASE_TABLE_LONG_COLUMNS)
    try:
        rows = _read_table_rows(table_path)
    except (OSError, pd.errors.ParserError, ValueError) as exc:
        logger.warning(f"Could not read disease biomarker table "
                       f"({table_path}): {exc}; source disabled.")
        return pd.DataFrame(columns=DISEASE_TABLE_LONG_COLUMNS)
    if len(rows) < 2:
        logger.warning(f"Disease biomarker table {table_path} has no data "
                       "rows; source disabled.")
        return pd.DataFrame(columns=DISEASE_TABLE_LONG_COLUMNS)

    def _cell(row, role):
        header = lookup.get(role)
        if header is None:
            return None
        idx = rows[0].index(header)
        value = row[idx] if idx < len(row) else None
        return None if value is None or pd.isna(value) else value

    lookup = _column_lookup(rows[0])
    if "disease" not in lookup or "markers" not in lookup \
            or "hmdb" not in lookup:
        logger.warning(f"Disease biomarker table {table_path} lacks the "
                       "Disease / Biochemical_Markers / HMDB-codes columns; "
                       "source disabled.")
        return pd.DataFrame(columns=DISEASE_TABLE_LONG_COLUMNS)

    out_rows = []
    n_no_code = 0
    n_misaligned = 0
    for row in rows[1:]:
        disease = str(_cell(row, "disease") or "").strip()
        if not disease:
            continue
        omim = str(_cell(row, "omim") or "").strip() or None
        smp_match = re.search(r"SMP\d+",
                              str(_cell(row, "smp") or "").upper())
        smp_id = smp_match.group() if smp_match else None
        pathway_name = _clean_pathway_name(_cell(row, "pathway"))
        source = f"IEMbase; OMIM {omim}" if omim else "IEMbase"
        raw_markers = str(_cell(row, "markers") or "").strip()
        if raw_markers.lower() in ("none", ""):
            continue
        markers = _split_top_level(raw_markers)
        entries = _split_top_level(_cell(row, "hmdb"))
        if len(markers) != len(entries):
            n_misaligned += 1
            logger.warning(f"Disease '{disease}': {len(markers)} marker(s) "
                           f"but {len(entries)} HMDB entry(ies); aligning "
                           "the shorter list.")
        for marker, entry in zip(markers, entries):
            direction = _marker_direction(marker)
            biomarker = _clean_marker_name(marker)
            codes = [c.upper() for c in re.findall(r"HMDB\d+", entry)]
            if not codes:
                n_no_code += 1
                continue
            for hmdb_id in codes:
                out_rows.append({
                    "disease": disease,
                    "biomarker": biomarker,
                    "hmdb_id": hmdb_id,
                    "direction": direction,
                    "omim": omim,
                    "smp_id": smp_id,
                    "pathway_name": pathway_name or None,
                    "source": source,
                })

    table = pd.DataFrame(out_rows, columns=DISEASE_TABLE_LONG_COLUMNS)
    if n_no_code:
        logger.info(f"Disease table: skipped {n_no_code} marker(s) without "
                    "an HMDB code ('no HMDB entry found').")
    if n_misaligned:
        logger.warning(f"Disease table: {n_misaligned} disease row(s) had "
                       "misaligned marker/HMDB lists.")
    logger.info(f"Loaded {len(table)} disease-biomarker attachment(s) "
                f"({table['disease'].nunique() if len(table) else 0} "
                f"diseases) from {table_path}.")
    return table


def resolve_disease_biomarkers(table: pd.DataFrame,
                               feature_to_hmdb: pd.DataFrame,
                               coverage: pd.DataFrame
                               ) -> Tuple[pd.DataFrame, List[str], pd.DataFrame]:
    """Resolve disease-biomarker rows to dataset features and kept pathways.

    HMDB codes resolve to dataset feature columns via the feature -> HMDB
    mapping (rows without a matched feature are reported in the audit and
    skipped). The workbook's SMP code links the disease to a KEPT pathway
    when one survives the coverage/keyword/pruning filters -- purely for
    reporting; a disease with no kept pathway still scores through the
    channel (the disease itself is the group).

    Returns:
        Tuple ``(resolved, features, audit)``: resolved attachments
        (DISEASE_RESOLVED_COLUMNS), the sorted list of dataset features
        carrying disease biomarkers, and the full audit table
        (DISEASE_AUDIT_COLUMNS, one row per table row).
    """
    empty_resolved = pd.DataFrame(columns=DISEASE_RESOLVED_COLUMNS)
    if table.empty:
        return empty_resolved, [], pd.DataFrame(columns=DISEASE_AUDIT_COLUMNS)

    feature_sets = (feature_to_hmdb.dropna(subset=["hmdb_id"])
                    .groupby("hmdb_id")["feature"]
                    .agg(lambda s: sorted(set(s))))
    kept = (coverage[["smp_id", "pathway_name"]].drop_duplicates()
            if not coverage.empty else pd.DataFrame(
                columns=["smp_id", "pathway_name"]))
    kept_smps = set(kept["smp_id"])
    smp_to_name = dict(zip(kept["smp_id"], kept["pathway_name"]))

    resolved_rows = []
    audit_rows = []
    for _, row in table.iterrows():
        feats = feature_sets.get(row["hmdb_id"], [])
        has_smp = pd.notna(row["smp_id"]) and str(row["smp_id"]).strip()
        if has_smp and row["smp_id"] in kept_smps:
            pathway_status = "kept"
            smp_id, pathway_name = row["smp_id"], smp_to_name[row["smp_id"]]
        elif has_smp:
            pathway_status = "not_in_kept_set"
            smp_id, pathway_name = None, row["pathway_name"]
        else:
            pathway_status = "no_pathbank_pathway"
            smp_id, pathway_name = None, row["pathway_name"]
        status = "scored" if feats else "no_dataset_feature"
        audit_rows.append({
            "disease": row["disease"], "biomarker": row["biomarker"],
            "hmdb_id": row["hmdb_id"], "direction": row["direction"],
            "omim": row["omim"], "smp_id": row["smp_id"],
            "pathway_name": row["pathway_name"],
            "pathway_status": pathway_status,
            "features": ";".join(feats), "n_features": len(feats),
            "status": status,
        })
        if feats:
            resolved_rows.append({
                "disease": row["disease"], "biomarker": row["biomarker"],
                "hmdb_id": row["hmdb_id"], "direction": row["direction"],
                "omim": row["omim"], "smp_id": smp_id,
                "pathway_name": pathway_name, "source": row["source"],
                "features": ";".join(feats), "n_features": len(feats),
            })

    resolved = pd.DataFrame(resolved_rows,
                            columns=DISEASE_RESOLVED_COLUMNS)
    audit = pd.DataFrame(audit_rows, columns=DISEASE_AUDIT_COLUMNS)
    features = sorted({f for feats in resolved["features"] for f in
                       str(feats).split(";") if f})
    n_no_feature = int((audit["status"] == "no_dataset_feature").sum())
    if n_no_feature:
        logger.warning(f"{n_no_feature} disease-biomarker row(s) match no "
                       "dataset feature (see disease_table_audit.csv).")
    n_kept = int((audit["pathway_status"] == "kept").sum())
    logger.info(f"Disease table: {len(resolved)} attachment(s) resolved to "
                f"dataset features across {resolved['disease'].nunique() if len(resolved) else 0} "
                f"disease(s); {n_kept} row(s) link to a kept pathway.")
    return resolved, features, audit


def flag_disease_biomarkers(zscores: pd.DataFrame,
                            resolved: pd.DataFrame,
                            feature_to_hmdb: pd.DataFrame,
                            normal_mask: pd.Series,
                            threshold_percentile: float = 99.0,
                            feature_scale_weights: Dict[str, float] = None,
                            max_sample_p: float = 0.05,
                            ratio_tests: pd.DataFrame = None
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Flag samples through disease-keyed, direction-aware biomarkers.

    Per disease, each biomarker's z is directed by its literature
    direction ('up' keeps only increases, 'down' only decrease
    magnitude, no arrow keeps |z|) -- directions are honored per
    (disease, biomarker), so a metabolite that rises in one disease and
    falls in another is scored correctly for both. Thresholds stay the
    direction-agnostic ``threshold_percentile`` percentile of the
    reference normals' |z| (the prior restricts WHICH tail may flag,
    not the normal range). The per-disease depth null is the maximum
    directed z of the reference normals over that disease's biomarkers;
    a sample flags the disease when at least one biomarker exceeds its
    threshold AND its maximum directed z beats that null at the frozen
    ``max_sample_p``. The channel flags the sample when ANY disease
    flags.

    ``ratio_tests`` (columns ``disease``, ``ratio``, ``direction``)
    adds declared diagnostic ratios (e.g. acylcarnitine C8/C2 for
    MCADD) as extra (disease, ratio) tests scored directly from the
    z-score matrix -- ratios have no HMDB identity, so they bypass
    the feature aggregation and use their own ratio column's z-score.
    Ratio tests join the SAME global depth null, so the channel's
    multiple-testing budget is not inflated by adding them.

    Returns:
        Tuple ``(flags, summary)``: ``flags`` with one row per (sample,
        disease, biomarker) carrying ``abs_z`` (directed magnitude),
        ``threshold``, ``excess``, ``flagged``; ``summary`` with one row
        per sample: ``biomarker_flagged`` (final channel decision),
        ``n_flagged_biomarkers``, ``biomarker_depth_p`` (depth p of the
        best-flagging disease), ``max_biomarker_z``, ``top_biomarker``,
        ``top_disease``.
    """
    flag_cols = ["sample_id", "disease", "smp_id", "pathway_name",
                 "biomarker", "hmdb_id", "direction", "abs_z", "threshold",
                 "excess", "flagged"]
    summary_cols = ["sample_id", "biomarker_flagged", "n_flagged_biomarkers",
                    "biomarker_depth_p", "max_biomarker_z", "top_biomarker",
                    "top_disease"]
    ratio_rows = (ratio_tests.drop_duplicates(subset=["disease", "ratio"])
                  if ratio_tests is not None and not ratio_tests.empty
                  else pd.DataFrame(columns=["disease", "ratio",
                                             "direction"]))
    if not ratio_rows.empty and not resolved.empty:
        known = set(resolved["disease"].dropna().astype(str).str.strip())
        unknown = sorted({str(d).strip() for d in ratio_rows["disease"]}
                         - known)
        if unknown:
            logger.warning(
                f"{len(unknown)} ratio-test disease name(s) not found in "
                "the IEMbase table (check the spelling; the test still "
                "scores but its evidence stays attributed to this name "
                "instead of merging with the disease's metabolite tests): "
                + "; ".join(unknown))
    if ((resolved.empty and ratio_rows.empty) or zscores.empty):
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))

    if resolved.empty:
        biomarker_z = pd.DataFrame(index=zscores.index)
    else:
        biomarker_z = aggregate_metabolite_zscores(
            zscores, feature_to_hmdb, list(resolved["hmdb_id"].unique()),
            feature_scale_weights=feature_scale_weights)
        biomarker_z = biomarker_z.dropna(axis=1, how="all")
    dedup = normal_mask[~normal_mask.index.duplicated(keep="first")]
    if biomarker_z.empty:
        thresholds = pd.Series(dtype=float)
    else:
        normal_values = biomarker_z.abs()[dedup.reindex(biomarker_z.index,
                                                        fill_value=False)]
        with np.errstate(all="ignore"):
            thresholds = normal_values.quantile(
                threshold_percentile / 100.0, axis=0).fillna(float("inf"))

    # One GLOBAL directed-z matrix across all (disease, biomarker)
    # tests: the depth null must span every test the channel runs.
    # Running per-disease tests at p <= max_sample_p each would give
    # ~1 - (1 - p)^n_diseases false positives per sample (73 tests at
    # 0.05 => ~97%). Diseases stay in the output for attribution only.
    test_columns = []
    column_meta = []
    ratio_thresholds = {}
    if not biomarker_z.empty:
        for _, row in resolved.drop_duplicates(
                subset=["disease", "hmdb_id"]).iterrows():
            hmdb_id = row["hmdb_id"]
            if hmdb_id not in biomarker_z.columns:
                continue
            z = biomarker_z[hmdb_id]
            direction = row["direction"]
            if direction == "up":
                directed_col = z.clip(lower=0)
            elif direction == "down":
                directed_col = (-z).clip(lower=0)
            else:
                directed_col = z.abs()
            test_columns.append(directed_col)
            column_meta.append({
                "test_id": len(test_columns) - 1,
                "disease": row["disease"], "biomarker": row.get("biomarker"),
                "hmdb_id": hmdb_id,
                "direction": direction, "smp_id": row["smp_id"],
                "pathway_name": row["pathway_name"]})
    for _, row in ratio_rows.iterrows():
        ratio_name = str(row["ratio"]).strip()
        if ratio_name not in zscores.columns:
            logger.warning(
                f"Ratio biomarker '{ratio_name}' for disease "
                f"'{row['disease']}' has no z-scored column; "
                "test skipped.")
            continue
        z = zscores[ratio_name]
        direction = str(row.get("direction") or "").strip().lower()
        if direction == "up":
            directed_col = z.clip(lower=0)
        elif direction == "down":
            directed_col = (-z).clip(lower=0)
        else:
            directed_col = z.abs()
        normal_z = z[dedup.reindex(z.index, fill_value=False)].abs().dropna()
        ratio_thr = (float(np.nanquantile(
            normal_z, threshold_percentile / 100.0))
            if len(normal_z) else float("inf"))
        ratio_thresholds[ratio_name] = ratio_thr
        test_columns.append(directed_col)
        column_meta.append({
            "test_id": len(test_columns) - 1,
            "disease": row["disease"], "biomarker": ratio_name,
            "hmdb_id": ratio_name,
            "direction": direction, "smp_id": None,
            "pathway_name": None})
    if not test_columns:
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))
    if not test_columns:
        return (pd.DataFrame(columns=flag_cols),
                pd.DataFrame(columns=summary_cols))

    directed = pd.concat(test_columns, axis=1)
    directed.columns = range(len(test_columns))
    meta = pd.DataFrame(column_meta)
    # Thresholds stay per-biomarker (each biomarker's own normal range);
    # the depth null is ONE global test: the maximum directed z over all
    # (disease, biomarker) tests.
    thr_by_test = np.asarray(
        [ratio_thresholds[m["hmdb_id"]]
         if m["hmdb_id"] in ratio_thresholds
         else thresholds.get(m["hmdb_id"], float("inf"))
         for m in column_meta], dtype=float)
    directed_arr = directed.values
    sample_max = np.nanmax(directed_arr, axis=1)

    is_normal = dedup.reindex(directed.index, fill_value=False).values
    normal_rows = directed_arr[is_normal]
    with np.errstate(all="ignore"):
        normal_null = (np.nanmax(normal_rows, axis=1)
                       if normal_rows.size else np.array([]))
        normal_null = normal_null[~np.isnan(normal_null)]
        if len(normal_null):
            depth_p = np.mean(sample_max[:, None] <= normal_null[None, :],
                              axis=1)
        else:
            depth_p = np.full(len(sample_max), np.nan)

    flags = pd.DataFrame({
        "sample_id": np.repeat(directed.index.values, directed.shape[1]),
        "test_id": np.tile(np.arange(directed.shape[1]), directed.shape[0]),
        "abs_z": directed_arr.ravel(order="C"),
    })
    flags["threshold"] = flags["test_id"].map(pd.Series(thr_by_test))
    flags = flags.merge(meta, on="test_id", how="left")
    with np.errstate(all="ignore"):
        flags["excess"] = flags["abs_z"] / flags["threshold"]
    flags["flagged"] = flags["abs_z"] > flags["threshold"]

    pair_flags = flags[flags["flagged"]] if not flags.empty else flags
    any_pair = (pair_flags.groupby("sample_id").size() > 0
                if len(pair_flags) else pd.Series(dtype=bool))
    ok_depth = pd.Series(depth_p <= max_sample_p, index=directed.index)
    channel_decision = (any_pair.reindex(directed.index, fill_value=False)
                        & ok_depth.reindex(directed.index, fill_value=False))
    channel_decision = channel_decision.fillna(False)

    best = (pair_flags.assign(
        depth_p=pd.Series(depth_p, index=directed.index)
        .reindex(pair_flags["sample_id"]).values)
        if len(pair_flags) else pd.DataFrame(
            columns=["sample_id", "disease", "depth_p", "abs_z"]))
    if len(best):
        best = best[best["depth_p"] <= max_sample_p]
        best = (best.sort_values("abs_z", ascending=False)
                .groupby("sample_id", as_index=False).first())

    samples = pd.Index(zscores.index, name="sample_id")
    summary = pd.DataFrame({"sample_id": samples})
    summary = summary.merge(best[["sample_id", "disease", "depth_p"]]
                            .rename(columns={
                                "disease": "top_disease",
                                "depth_p": "biomarker_depth_p"}),
                            on="sample_id", how="left")
    if not pair_flags.empty:
        counts = pair_flags.groupby("sample_id").agg(
            n_flagged_biomarkers=("flagged", "size"),
            max_biomarker_z=("abs_z", "max")).reset_index()
        top = (pair_flags.sort_values("abs_z", ascending=False)
               .groupby("sample_id", as_index=False).first()[
                   ["sample_id", "hmdb_id"]])
        summary = summary.merge(counts, on="sample_id", how="left")
        summary = summary.merge(top.rename(columns={
            "hmdb_id": "top_biomarker"}), on="sample_id", how="left")
    else:
        summary["n_flagged_biomarkers"] = 0
        summary["max_biomarker_z"] = np.nan
        summary["top_biomarker"] = None
    summary["biomarker_flagged"] = summary["sample_id"].isin(
        set(channel_decision[channel_decision].index))
    for col, fill in (("n_flagged_biomarkers", 0), ("max_biomarker_z", np.nan),
                      ("top_biomarker", None), ("top_disease", None),
                      ("biomarker_depth_p", np.nan)):
        if col in summary.columns:
            summary[col] = summary[col].where(summary[col].notna(), fill)
    summary = summary[summary_cols]
    n_diseases = resolved["disease"].nunique()
    n_tests = int(directed.shape[1])
    logger.info(f"Disease biomarker channel: {n_diseases} disease group(s) "
                f"({n_tests} directed tests, ONE global depth null at "
                f"p <= {max_sample_p}); "
                f"{int(summary['biomarker_flagged'].sum())} of "
                f"{len(summary)} samples flagged through the channel.")
    return flags, summary


def audit_unlinked_disease_markers(table_path: str,
                                   feature_columns: List[str],
                                   name_index: Dict[str, object] = None
                                   ) -> pd.DataFrame:
    """Report disease markers skipped for having no HMDB code.

    The IEMbase loader skips every marker whose aligned HMDB entry says
    '(no HMDB entry found)'. Those markers cannot join the biomarker
    channel through accession resolution, but their names may still
    match dataset feature columns. This audit re-reads the table with
    the same parsing rules, collects the skipped (disease, marker)
    pairs, and attempts exact and loose name matches against the
    feature columns -- without changing any scoring. Readout-only, for
    deciding whether a name-resolution build is worthwhile.

    Args:
        table_path: path to the IEMbase-style table (Excel or CSV).
        feature_columns: dataset feature column names.
        name_index: optional HMDB name index; when given, a marker
            whose name resolves to accessions through it is also
            reported (those accessions could be added to the table).

    Returns:
        DataFrame with columns ``disease``, ``marker``, ``direction``,
        ``matched_features`` (';'-joined dataset columns matched by
        name, '' when none), ``name_index_accessions`` (';'-joined
        accessions from the HMDB name index, '' when none).
    """
    from pathway_pipeline.pipeline.name_utils import (
        normalize_name, normalize_loose)
    from pathway_pipeline.pipeline.pathway_mapping import (
        _split_feature_name_and_hmdb)

    cols = ["disease", "marker", "direction",
            "matched_features", "name_index_accessions"]
    rows = _read_table_rows(table_path) if table_path else []
    if len(rows) < 2:
        return pd.DataFrame(columns=cols)
    lookup = _column_lookup(rows[0])
    if "disease" not in lookup or "markers" not in lookup:
        return pd.DataFrame(columns=cols)

    def _cell(row, role):
        header = lookup.get(role)
        if header is None:
            return None
        idx = rows[0].index(header)
        value = row[idx] if idx < len(row) else None
        return None if value is None or pd.isna(value) else value

    feat_exact = {}
    feat_loose = {}
    for col in feature_columns:
        stripped = normalize_name(col)
        tag = _split_feature_name_and_hmdb(col)
        if tag and stripped.endswith(tag):
            stripped = stripped[: -len(tag)].rstrip(".")
        if stripped:
            feat_exact.setdefault(stripped, []).append(col)
        norm = normalize_name(col)
        if norm:
            feat_exact.setdefault(norm, []).append(col)
        loose = normalize_loose(col)
        if tag and loose.endswith(tag):
            loose = loose[: -len(tag)].rstrip(".")
        if loose:
            feat_loose.setdefault(loose, []).append(col)
        loose_full = normalize_loose(col)
        if loose_full:
            feat_loose.setdefault(loose_full, []).append(col)

    out_rows = []
    for row in rows[1:]:
        disease = str(_cell(row, "disease") or "").strip()
        if not disease:
            continue
        raw_markers = str(_cell(row, "markers") or "").strip()
        if raw_markers.lower() in ("none", ""):
            continue
        markers = _split_top_level(raw_markers)
        entries = _split_top_level(_cell(row, "hmdb"))
        for marker, entry in zip(markers, entries):
            if re.findall(r"HMDB\d+", str(entry or "")):
                continue
            biomarker = _clean_marker_name(marker)
            direction = _marker_direction(marker)
            norm = normalize_name(biomarker)
            loose = normalize_loose(biomarker)
            matched = (feat_exact.get(norm, [])
                       if norm else [])
            if not matched and loose:
                matched = feat_loose.get(loose, [])
            accessions = ""
            if name_index is not None and norm:
                hits = name_index.get(norm, set())
                if hits:
                    accessions = ";".join(sorted(hits))
            out_rows.append({
                "disease": disease,
                "marker": biomarker,
                "direction": direction,
                "matched_features": ";".join(matched),
                "name_index_accessions": accessions,
            })
    return pd.DataFrame(out_rows, columns=cols)


def build_disease_panel_scores(zscores: pd.DataFrame,
                               resolved: pd.DataFrame,
                               feature_to_hmdb: pd.DataFrame,
                               normal_mask: pd.Series,
                               ratio_tests: pd.DataFrame = None,
                               min_metabolites: int = 2,
                               max_abs_z: float = None,
                               feature_scale_weights: Dict[str, float] = None,
                               name_matched: pd.DataFrame = None
                               ) -> pd.DataFrame:
    """Score each IEMbase disease panel as a direction-aware Stouffer sum.

    A promoted panel behaves like a pathway: one row per (sample,
    disease) carrying the panel's aggregate statistic and its
    per-direction member terms. Each member's z is directed by its
    literature arrow before the sum -- an 'up' marker contributes +z, a
    'down' marker -z, a marker without an arrow contributes |z| -- so
    a marker moving against its curated direction weakens the panel
    instead of strengthening it. Declared ratio tests for the disease
    join the same sum (their column's z is directed the same way).
    Members are combined per HMDB metabolite first (scale^2-weighted
    mean of same-metabolite features, the pathway channel's rule), so
    duplicate features cannot double a metabolite's vote.

    Args:
        zscores: per-sample metabolite z-scores (demoted features
            already excluded by the caller).
        resolved: resolved IEMbase disease biomarker table (one row per
            (disease, biomarker, hmdb_id) with ``direction``).
        feature_to_hmdb: feature -> HMDB mapping.
        normal_mask: boolean Series marking normal reference samples
            (unused here; the caller thresholds against normals).
        ratio_tests: declared ratio tests (columns ``disease``,
            ``ratio``, ``direction``); matching diseases add their
            ratio z as a member term.
        min_metabolites: minimum usable member terms for a panel score
            to be emitted.
        max_abs_z: optional cap on |z| before the sum (mirrors the
            pathway channel's artifact guard).
        feature_scale_weights: optional feature -> weight map for
            combining same-metabolite features.
        name_matched: optional output of the unlinked-marker audit
            (columns ``disease``, ``marker``, ``direction``,
            ``matched_features``); markers whose Excel entry carries
            no HMDB code but whose name matches dataset features join
            the panel as extra member terms.

    Returns:
        Long-format frame with columns ``sample_id``, ``smp_id``
        (the disease's SMP code or a synthetic DISEASE-<slug> key),
        ``pathway_name`` (the disease name), ``n_metabolites_used``,
        ``z_stouffer`` (directed sum / sqrt(k)), ``z_stouffer_abs``.
    """
    score_cols = ["sample_id", "smp_id", "pathway_name",
                  "n_metabolites_used", "z_stouffer", "z_stouffer_abs"]
    if resolved is None or resolved.empty or zscores.empty:
        return pd.DataFrame(columns=score_cols)
    z = zscores.copy()
    if max_abs_z is not None and max_abs_z > 0:
        z = z.clip(lower=-max_abs_z, upper=max_abs_z)

    # feature -> metabolite edges, as the pathway channel uses them.
    f2h = feature_to_hmdb[["feature", "hmdb_id"]].dropna()
    f2h = f2h[f2h["feature"].isin(z.columns)]

    # disease -> {metabolite key -> (direction, [features])}
    panels = {}
    for _, row in resolved.drop_duplicates(
            subset=["disease", "hmdb_id"]).iterrows():
        disease = row["disease"]
        feats = f2h.loc[f2h["hmdb_id"] == row["hmdb_id"], "feature"]
        feats = [f for f in feats if f in z.columns]
        if feats:
            key = f"HMDB:{row['hmdb_id']}"
            entry = panels.setdefault(
                disease, {"smp_id": row.get("smp_id"),
                          "members": {}})
            entry["members"].setdefault(
                key, {"direction": row["direction"], "features": []})
            entry["members"][key]["features"].extend(feats)
    if name_matched is not None and not name_matched.empty:
        nm = name_matched[name_matched["matched_features"].fillna("") != ""]
        for _, row in nm.iterrows():
            disease = row["disease"]
            feats = [f for f in str(row["matched_features"]).split(";")
                     if f in z.columns]
            if not feats:
                continue
            key = f"NAME:{row['marker']}"
            entry = panels.setdefault(
                disease, {"smp_id": None, "members": {}})
            if key in entry["members"]:
                entry["members"][key]["features"].extend(feats)
            else:
                entry["members"][key] = {
                    "direction": row["direction"], "features": feats}
    if ratio_tests is not None and not ratio_tests.empty:
        for _, row in ratio_tests.drop_duplicates(
                subset=["disease", "ratio"]).iterrows():
            ratio_name = str(row["ratio"]).strip()
            if ratio_name not in z.columns:
                continue
            disease = row["disease"]
            key = f"RATIO:{ratio_name}"
            entry = panels.setdefault(
                disease, {"smp_id": None, "members": {}})
            entry["members"][key] = {
                "direction": str(row.get("direction") or ""), 
                "features": [ratio_name]}

    weights = feature_scale_weights or {}

    def _member_z(feats, direction):
        sub = z[feats]
        if len(feats) == 1 or not weights:
            mean_z = sub.mean(axis=1)
        else:
            w = pd.Series({f: float(weights.get(f, 1.0)) for f in feats})
            present_w = sub.notna().mul(w, axis=1).sum(axis=1)
            mean_z = (sub.fillna(0.0).mul(w, axis=1).sum(axis=1)
                      / present_w.where(present_w > 0))
        direction = str(direction or "").strip().lower()
        if direction == "up":
            return mean_z.clip(lower=0)
        if direction == "down":
            return (-mean_z).clip(lower=0)
        return mean_z.abs()

    rows = []
    for disease, entry in sorted(panels.items()):
        member_cols = []
        for key, m in entry["members"].items():
            if not m["features"]:
                continue
            mz = _member_z(sorted(set(m["features"])), m["direction"])
            if mz is not None:
                member_cols.append((key, mz))
        member_cols = [(k, c) for k, c in member_cols
                       if c.notna().any()]
        if len(member_cols) < min_metabolites:
            continue
        matrix = pd.concat([c for _, c in member_cols], axis=1)
        matrix.columns = range(len(member_cols))
        k = matrix.notna().sum(axis=1)
        with np.errstate(all="ignore"):
            signed = matrix.sum(axis=1, skipna=True) / np.sqrt(
                k.replace(0, np.nan))
        usable = k >= min_metabolites
        smp = entry.get("smp_id")
        if smp is None or pd.isna(smp) or not str(smp).strip():
            smp = f"DISEASE-{_slug(disease)}"
        for sample_id in z.index:
            if not usable.get(sample_id, False):
                continue
            s = float(signed.get(sample_id, np.nan))
            if np.isnan(s):
                continue
            rows.append({
                "sample_id": sample_id,
                "smp_id": smp,
                "pathway_name": disease,
                "n_metabolites_used": int(k[sample_id]),
                "z_stouffer": s,
                "z_stouffer_abs": abs(s),
            })
    return pd.DataFrame(rows, columns=score_cols)


def _slug(text: str) -> str:
    out = []
    for ch in str(text):
        if ch.isalnum():
            out.append(ch)
        else:
            out.append("-")
    slug = "".join(out).strip("-").lower()
    while "--" in slug:
        slug = slug.replace("--", "-")
    return slug or "disease"
