"""Pathway mapping preprocessing: feature column -> HMDB accession -> pathways.

This is the first stage of the pathway pipeline. It produces the three
annotation tables the later per-pathway shift statistics consume:

1. ``feature_to_hmdb.csv`` -- every input feature column mapped to one (or
   more) HMDB accessions, with the match method that resolved it.
2. ``feature_to_pathway.csv`` -- every (feature, pathway) link, exploding the
   one-to-many relationship between a feature's HMDB accessions and the
   pathways each accession participates in.
3. ``pathway_coverage.csv`` -- one row per pathway with the count and list of
   matched features, plus the pathway's total compound count from the TSV.

Feature columns come in three shapes (mirroring the outlier-detection
pipeline):

- a bare HMDB accession (``HMDB0000063``) -> matched directly by accession;
- a compound name with an ``.HMDB########`` suffix (``Cortisol.HMDB0000063``)
  -> the suffix HMDB is taken authoritatively;
- a plain compound name (``Cortisol``) -> matched against the HMDB name index
  (primary name + synonyms), with a loose (non-alphanumeric-stripped) fallback
  that catches hyphenation/spacing differences.

A plain name that resolves to several HMDB accessions is linked to ALL of
them (and therefore to the union of their pathways); ambiguity is recorded in
the mapping tables so the user can inspect it. A feature that matches nothing
is reported but excluded from the pathway coverage.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

from .name_utils import normalize_name, normalize_loose


logger = logging.getLogger(__name__)


def load_pathways_tsv(pathways_file: str) -> pd.DataFrame:
    """Load the SMPDB-derived pathways TSV.

    The expected structure is::

        smp_id\tpathway_name\tn_compounds\thmdb_ids
        SMP0000575\t11-beta-Hydroxylase Deficiency (CYP11B1)\t41\tHMDB0000015;HMDB0000016;...
        ...

    ``hmdb_ids`` is a ';'-separated list of HMDB accessions. Returns a
    DataFrame with columns ``smp_id``, ``pathway_name``, ``n_compounds``,
    and a list-typed ``hmdb_ids`` column (each row a list of accessions).
    """
    path = Path(pathways_file)
    if not path.exists():
        logger.error(f"Pathways TSV not found at {pathways_file}")
        return pd.DataFrame(columns=["smp_id", "pathway_name", "n_compounds", "hmdb_ids"])

    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"smp_id", "pathway_name", "n_compounds", "hmdb_ids"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"Pathways TSV {pathways_file} missing columns: {missing}")
        return pd.DataFrame(columns=list(required))

    df = df.dropna(subset=["smp_id", "pathway_name"]).copy()
    df["n_compounds"] = pd.to_numeric(df["n_compounds"], errors="coerce")
    # Split the ';'-separated accession list and strip whitespace/empties.
    df["hmdb_ids"] = df["hmdb_ids"].fillna("").apply(
        lambda s: [x.strip() for x in str(s).split(";") if x.strip()]
    )
    df = df.reset_index(drop=True)
    logger.info(f"Loaded {len(df)} pathways from {pathways_file}")
    return df[["smp_id", "pathway_name", "n_compounds", "hmdb_ids"]]


def _split_feature_name_and_hmdb(col: str) -> Optional[str]:
    """Extract a trailing HMDB accession from a feature column name.

    Handles the two annotated forms used in the dataset:
        'Cortisol.HMDB0000063' -> 'HMDB0000063'
        'HMDB0000063'          -> 'HMDB0000063'
    Returns the accession (uppercased) when present, else None so the caller
    falls back to name-based matching.
    """
    norm = normalize_name(col)
    if norm.startswith("HMDB") and len(norm) >= 7 and norm[4:].isdigit():
        return norm
    if "." in norm:
        suffix = norm.rsplit(".", 1)[-1]
        if suffix.startswith("HMDB") and len(suffix) >= 7 and suffix[4:].isdigit():
            return suffix
    return None


def match_features_to_hmdb(feature_columns: List[str],
                            name_index: Dict[str, Set[str]],
                            min_name_length: int = 3
                            ) -> pd.DataFrame:
    """Match every feature column to one or more HMDB accessions.

    Matching priority (first hit wins; the method is recorded):

    1. **HMDB tag**: the column contains a trailing HMDB accession (bare
       ``HMDB########`` or ``Name.HMDB########``). The tagged accession is
       taken authoritatively -- the name index is NOT consulted -- because the
       tag signals a confident identification.
    2. **Exact name**: the full normalized column name matches an entry in the
       HMDB name index (which holds primary names, synonyms, and the bare
       accessions).
    3. **Loose name**: the loose-normalized column name (all non-alphanumeric
       chars removed) matches a loose-normalized name index entry, catching
       hyphenation/spacing/punctuation differences.

    A feature that resolves to multiple accessions (ambiguous synonym) is
    linked to all of them; ``n_hmdb_ids`` records the count.

    Args:
        feature_columns: list of feature column names.
        name_index: normalized-name -> {HMDB accessions} from
            :func:`hmdb_parser.build_name_index`.
        min_name_length: skip names shorter than this for name-based matching.

    Returns:
        DataFrame with columns ``feature``, ``hmdb_id`` (one row per matched
        accession), ``match_method`` ('hmdb_tag' / 'name_exact' / 'name_loose'),
        and ``n_hmdb_ids`` (number of accessions matched for the feature).
        Features that match nothing get a single row with hmdb_id NaN and
        match_method 'unmatched'.
    """
    if not name_index:
        logger.warning("HMDB name index is empty; no feature can be matched by name.")

    # Precompute a loose view of the index only if any name-based matching is
    # attempted (it can be large). Built lazily from the index's keys.
    loose_index: Dict[str, Set[str]] = {}
    for norm_name, accs in name_index.items():
        loose = normalize_loose(norm_name)
        if loose and len(loose) >= min_name_length:
            # Merge accession sets when two normalized names share a loose form.
            if loose in loose_index:
                loose_index[loose] |= accs
            else:
                loose_index[loose] = set(accs)

    rows: List[Dict] = []
    n_tagged = 0
    n_exact = 0
    n_loose = 0
    n_unmatched = 0

    for col in feature_columns:
        hmdb_ids: Set[str] = set()
        method = "unmatched"

        # 1) HMDB tag -- authoritative, ignore the name index.
        tagged = _split_feature_name_and_hmdb(col)
        if tagged:
            hmdb_ids = {tagged}
            method = "hmdb_tag"
            n_tagged += 1
        else:
            # 2) Exact normalized name match.
            norm = normalize_name(col)
            if norm and len(norm) >= min_name_length and norm in name_index:
                hmdb_ids = set(name_index[norm])
                method = "name_exact"
                n_exact += 1
            elif norm and len(norm) >= min_name_length:
                # 3) Loose (non-alphanumeric-stripped) fallback.
                loose = normalize_loose(col)
                if loose and len(loose) >= min_name_length and loose in loose_index:
                    hmdb_ids = set(loose_index[loose])
                    method = "name_loose"
                    n_loose += 1

        if not hmdb_ids:
            n_unmatched += 1
            rows.append({
                "feature": col,
                "hmdb_id": None,
                "match_method": "unmatched",
                "n_hmdb_ids": 0,
            })
        else:
            for acc in sorted(hmdb_ids):
                rows.append({
                    "feature": col,
                    "hmdb_id": acc,
                    "match_method": method,
                    "n_hmdb_ids": len(hmdb_ids),
                })

    logger.info(f"Feature -> HMDB matching: {n_tagged} by HMDB tag, {n_exact} by "
                f"exact name, {n_loose} by loose name, {n_unmatched} unmatched "
                f"(of {len(feature_columns)} features).")
    return pd.DataFrame(rows, columns=["feature", "hmdb_id", "match_method", "n_hmdb_ids"])


def link_features_to_pathways(feature_to_hmdb: pd.DataFrame,
                               pathways: pd.DataFrame) -> pd.DataFrame:
    """Explode feature -> HMDB -> pathway links into a long table.

    Joins each matched HMDB accession against the pathways TSV (a pathway row
    matches if the accession is in its ``hmdb_ids`` list), producing one row
    per (feature, pathway) link. Unmatched features are excluded. A feature
    whose accession sits in several pathways yields several rows.

    Args:
        feature_to_hmdb: output of :func:`match_features_to_hmdb`.
        pathways: output of :func:`load_pathways_tsv`.

    Returns:
        DataFrame with columns ``feature``, ``hmdb_id``, ``smp_id``,
        ``pathway_name``, ``n_compounds``.
    """
    matched = feature_to_hmdb.dropna(subset=["hmdb_id"]).copy()
    if matched.empty or pathways.empty:
        logger.info("No matched features or no pathways; feature->pathway table empty.")
        return pd.DataFrame(columns=["feature", "hmdb_id", "smp_id",
                                      "pathway_name", "n_compounds"])

    # Build accession -> list of (smp_id, pathway_name, n_compounds) once.
    acc_to_pathways: Dict[str, List[tuple]] = {}
    for _, prow in pathways.iterrows():
        for acc in prow["hmdb_ids"]:
            acc_to_pathways.setdefault(acc, []).append(
                (prow["smp_id"], prow["pathway_name"], prow["n_compounds"])
            )

    rows: List[Dict] = []
    for _, mrow in matched.iterrows():
        acc = mrow["hmdb_id"]
        for smp_id, pname, ncomp in acc_to_pathways.get(acc, []):
            rows.append({
                "feature": mrow["feature"],
                "hmdb_id": acc,
                "smp_id": smp_id,
                "pathway_name": pname,
                "n_compounds": ncomp,
            })

    links = pd.DataFrame(rows, columns=["feature", "hmdb_id", "smp_id",
                                         "pathway_name", "n_compounds"])
    logger.info(f"Built {len(links)} feature->pathway links across "
                f"{links['smp_id'].nunique() if len(links) else 0} pathways.")
    return links


def pathway_coverage(feature_to_pathway: pd.DataFrame,
                      min_pathway_size: int = 3) -> pd.DataFrame:
    """Summarize, per pathway, the matched features and coverage.

    Args:
        feature_to_pathway: output of :func:`link_features_to_pathways`.
        min_pathway_size: drop pathways with fewer matched features than this.

    Returns:
        DataFrame with one row per pathway and columns ``smp_id``,
        ``pathway_name``, ``n_compounds`` (total compounds in the pathway from
        the TSV), ``n_matched_features`` (number of distinct dataset features
        mapped to the pathway), ``matched_features`` (';'-joined list), and
        ``coverage`` (n_matched_features / n_compounds).
    """
    if feature_to_pathway.empty:
        return pd.DataFrame(columns=["smp_id", "pathway_name", "n_compounds",
                                      "n_matched_features", "matched_features",
                                      "coverage"])

    grouped = (
        feature_to_pathway
        .groupby(["smp_id", "pathway_name", "n_compounds"], dropna=False)["feature"]
        .agg(lambda s: sorted(set(s)))
        .reset_index(name="matched_features")
    )
    grouped["n_matched_features"] = grouped["matched_features"].str.len()
    grouped["matched_features"] = grouped["matched_features"].str.join(";")
    grouped["n_compounds"] = pd.to_numeric(grouped["n_compounds"], errors="coerce")
    grouped["coverage"] = grouped.apply(
        lambda r: (r["n_matched_features"] / r["n_compounds"])
        if pd.notna(r["n_compounds"]) and r["n_compounds"] > 0 else float("nan"),
        axis=1,
    )
    grouped = grouped.sort_values("n_matched_features", ascending=False)
    before = len(grouped)
    dropped_rows = grouped[grouped["n_matched_features"] < min_pathway_size]
    grouped = grouped[grouped["n_matched_features"] >= min_pathway_size].reset_index(drop=True)
    dropped = before - len(grouped)
    if dropped:
        dropped_names = sorted(dropped_rows["pathway_name"].unique())
        logger.info(f"Dropped {dropped} pathways with fewer than "
                    f"{min_pathway_size} matched features; {len(grouped)} remain.")
        logger.info(f"Dropped pathways: {dropped_names}")
    return grouped[["smp_id", "pathway_name", "n_compounds",
                    "n_matched_features", "matched_features", "coverage"]]
