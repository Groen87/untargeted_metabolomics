"""Pathway mapping preprocessing: feature column -> HMDB accession -> pathways.

This is the feature-engineering stage of the pathway pipeline. It produces the
annotation tables the later per-pathway statistics will consume:

1. ``feature_to_hmdb.csv`` -- every input feature column mapped to one (or
   more) HMDB accessions, with the match method that resolved it.
2. ``feature_to_pathway.csv`` -- every (feature, pathway) link, exploding the
   one-to-many relationship between a feature's HMDB accessions and the
   pathways each accession participates in.
3. ``pathway_coverage.csv`` -- one row per pathway with the matched
   metabolites/features and the fraction of the pathway's metabolites that are
   mapped to dataset features.

Feature columns come in three shapes (mirroring the outlier-detection
pipeline):

- a bare HMDB accession (``HMDB0000063``) -> matched directly by accession;
- a compound name with an ``.HMDB########`` suffix (``Cortisol.HMDB0000063``)
  -> the suffix HMDB is taken authoritatively;
- a plain compound name (``Cortisol``) -> matched against the HMDB name index
  (primary name + synonyms), with a loose (non-alphanumeric-stripped) fallback
  that catches hyphenation/spacing differences.

Pathway data comes from the PathBank all-metabolites CSV
(``pathbank_all_metabolites.csv``), which lists one row per (pathway,
metabolite) pair with the metabolite's HMDB ID. Only ``Metabolic`` and
``Disease`` pathways for ``Homo sapiens`` are kept. A pathway is retained only
when at least ``min_coverage`` (default 20%) of its listed metabolites are
mapped to features in the dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .name_utils import normalize_name, normalize_loose


logger = logging.getLogger(__name__)


PATHBANK_SPECIES = "Homo sapiens"
PATHBANK_SUBJECTS = ("Metabolic", "Disease")


def load_pathbank_pathways(pathbank_file: str) -> pd.DataFrame:
    """Load the PathBank all-metabolites CSV into a per-(pathway, metabolite) table.

    The expected structure is one row per (pathway, metabolite) pair with the
    columns ``PathBank ID``, ``Pathway Name``, ``Pathway Subject``, ``Species``,
    ``Metabolite ID``, ``Metabolite Name``, and ``HMDB ID``. Only ``Metabolic``
    and ``Disease`` pathways for ``Homo sapiens`` are kept; rows without an
    HMDB ID cannot link to the dataset and are dropped.

    Returns a DataFrame with the columns ``smp_id``, ``pathway_name``,
    ``pathway_subject``, ``species``, ``metabolite_id``, ``metabolite_name``,
    and ``hmdb_id`` (one row per (pathway, metabolite) pair).
    """
    path = Path(pathbank_file)
    if not path.exists():
        logger.error(f"PathBank all-metabolites CSV not found at {pathbank_file}")
        return pd.DataFrame(columns=["smp_id", "pathway_name", "pathway_subject",
                                     "species", "metabolite_id",
                                     "metabolite_name", "hmdb_id"])

    df = pd.read_csv(path, dtype=str)
    rename = {
        "PathBank ID": "smp_id",
        "Pathway Name": "pathway_name",
        "Pathway Subject": "pathway_subject",
        "Species": "species",
        "Metabolite ID": "metabolite_id",
        "Metabolite Name": "metabolite_name",
        "HMDB ID": "hmdb_id",
    }
    missing = [src for src in rename if src not in df.columns]
    if missing:
        logger.error(f"PathBank CSV {pathbank_file} missing columns: {missing}")
        return pd.DataFrame(columns=list(rename.values()))

    df = df.rename(columns=rename)
    n_raw = len(df)

    df["species"] = df["species"].fillna("").str.strip()
    df["pathway_subject"] = df["pathway_subject"].fillna("").str.strip()
    df["hmdb_id"] = df["hmdb_id"].fillna("").str.strip().str.upper()

    keep = (
        (df["species"] == PATHBANK_SPECIES)
        & (df["pathway_subject"].isin(PATHBANK_SUBJECTS))
        & (df["hmdb_id"] != "")
    )
    df = df.loc[keep, list(rename.values())].copy()
    df = df.drop_duplicates().reset_index(drop=True)

    logger.info(
        f"Loaded {n_raw} rows from {pathbank_file}; kept {len(df)} "
        f"({PATHBANK_SPECIES} {PATHBANK_SUBJECTS} rows with an HMDB ID) across "
        f"{df['smp_id'].nunique()} pathways."
    )
    return df


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

    Joins each matched HMDB accession against the PathBank metabolite table
    (a pathway matches if the accession appears as one of its metabolite
    HMDB IDs), producing one row per (feature, pathway) link. Unmatched
    features are excluded. A feature whose accession sits in several pathways
    yields several rows.

    Args:
        feature_to_hmdb: output of :func:`match_features_to_hmdb`.
        pathways: output of :func:`load_pathbank_pathways`.

    Returns:
        DataFrame with columns ``feature``, ``hmdb_id``, ``smp_id``,
        ``pathway_name``, ``metabolite_id``, ``metabolite_name``.
    """
    empty = pd.DataFrame(columns=["feature", "hmdb_id", "smp_id",
                                  "pathway_name", "metabolite_id",
                                  "metabolite_name"])
    matched = feature_to_hmdb.dropna(subset=["hmdb_id"]).copy()
    if matched.empty or pathways.empty:
        logger.info("No matched features or no pathways; feature->pathway table empty.")
        return empty

    # Build accession -> list of (pathway, metabolite) rows once.
    acc_to_pathways: Dict[str, List[Tuple[str, str, str, str]]] = {}
    for _, prow in pathways.iterrows():
        acc_to_pathways.setdefault(prow["hmdb_id"], []).append(
            (prow["smp_id"], prow["pathway_name"],
             prow["metabolite_id"], prow["metabolite_name"])
        )

    rows: List[Dict] = []
    for _, mrow in matched.iterrows():
        acc = mrow["hmdb_id"]
        for smp_id, pname, met_id, met_name in acc_to_pathways.get(acc, []):
            rows.append({
                "feature": mrow["feature"],
                "hmdb_id": acc,
                "smp_id": smp_id,
                "pathway_name": pname,
                "metabolite_id": met_id,
                "metabolite_name": met_name,
            })

    links = pd.DataFrame(rows, columns=["feature", "hmdb_id", "smp_id",
                                         "pathway_name", "metabolite_id",
                                         "metabolite_name"])
    logger.info(f"Built {len(links)} feature->pathway links across "
                f"{links['smp_id'].nunique() if len(links) else 0} pathways.")
    return links


def pathway_coverage(feature_to_pathway: pd.DataFrame,
                      pathways: pd.DataFrame,
                      min_coverage: float = 0.20) -> pd.DataFrame:
    """Summarize, per pathway, the matched metabolites and their coverage.

    A pathway's total metabolite count is the number of distinct HMDB IDs
    listed for it in the PathBank table (after the species/subject filters).
    ``coverage`` is the fraction of those metabolites that are mapped to
    features in the dataset. Only pathways with ``coverage >= min_coverage``
    are returned.

    Args:
        feature_to_pathway: output of :func:`link_features_to_pathways`.
        pathways: output of :func:`load_pathbank_pathways`; used for the
            per-pathway total metabolite counts.
        min_coverage: keep pathways where at least this fraction of the
            pathway's metabolites are mapped to dataset features.

    Returns:
        DataFrame with one row per kept pathway and columns ``smp_id``,
        ``pathway_name``, ``n_metabolites`` (total distinct HMDB IDs listed in
        PathBank), ``n_matched_metabolites``, ``matched_metabolites``
        (';'-joined HMDB IDs), ``n_matched_features``, ``matched_features``
        (';'-joined dataset feature columns), and ``coverage``.
    """
    out_cols = ["smp_id", "pathway_name", "n_metabolites",
                "n_matched_metabolites", "matched_metabolites",
                "n_matched_features", "matched_features", "coverage"]
    if feature_to_pathway.empty or pathways.empty:
        return pd.DataFrame(columns=out_cols)

    # Distinct HMDB IDs per pathway from the PathBank table.
    pw_totals = (
        pathways.groupby(["smp_id", "pathway_name"])["hmdb_id"]
        .agg(lambda s: sorted(set(s)))
        .rename("metabolites")
        .reset_index()
    )
    pw_totals["n_metabolites"] = pw_totals["metabolites"].str.len()

    # Distinct matched HMDB IDs (and their dataset features) per pathway.
    grouped = (
        feature_to_pathway
        .groupby(["smp_id", "pathway_name"])
        .agg(matched_metabolites=("hmdb_id", lambda s: sorted(set(s))),
             matched_features=("feature", lambda s: sorted(set(s))))
        .reset_index()
    )
    grouped["n_matched_metabolites"] = grouped["matched_metabolites"].str.len()
    grouped["n_matched_features"] = grouped["matched_features"].str.len()
    grouped["matched_metabolites"] = grouped["matched_metabolites"].str.join(";")
    grouped["matched_features"] = grouped["matched_features"].str.join(";")

    coverage = pw_totals.merge(grouped, on=["smp_id", "pathway_name"], how="left")
    coverage["n_matched_metabolites"] = coverage["n_matched_metabolites"].fillna(0).astype(int)
    coverage["n_matched_features"] = coverage["n_matched_features"].fillna(0).astype(int)
    coverage["matched_metabolites"] = coverage["matched_metabolites"].fillna("")
    coverage["matched_features"] = coverage["matched_features"].fillna("")
    coverage["coverage"] = coverage["n_matched_metabolites"] / coverage["n_metabolites"]

    before = len(coverage)
    dropped_rows = coverage[coverage["coverage"] < min_coverage]
    coverage = coverage[coverage["coverage"] >= min_coverage].copy()
    coverage = coverage.sort_values("coverage", ascending=False).reset_index(drop=True)
    dropped = before - len(coverage)
    if dropped:
        logger.info(f"Dropped {dropped} pathways with coverage < {min_coverage:.0%}; "
                    f"{len(coverage)} remain.")
        logger.debug(f"Dropped pathways: {sorted(dropped_rows['pathway_name'].unique())}")
    return coverage[out_cols]
