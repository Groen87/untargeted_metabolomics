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

Pathway data comes from the PathBank primary-pathways metabolites CSV
(``pathbank_all_metabolites.csv``), which lists one row per (pathway,
metabolite) pair with the metabolite's HMDB ID. Only rows for the configured
``species`` (default ``Homo sapiens``) are kept. The file carries no pathway
name -- pathways are identified by their PathBank ``pathway_id`` (e.g.
``SMP0000055``), which is used as the pathway name throughout. A pathway is
retained only when at least ``min_coverage`` (default 20%) of its listed
metabolites are mapped to features in the dataset.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pandas as pd

from .name_utils import normalize_name, normalize_loose


logger = logging.getLogger(__name__)


DEFAULT_SPECIES = "Homo sapiens"

_PATHWAY_ID_COLUMNS = ("smp id", "pathway_id", "smpid", "pathbank id")
_PATHWAY_NAME_COLUMNS = ("pathway name", "pathway_name", "name")


def load_pathway_names(names_file: str) -> Dict[str, str]:
    """Load pathway names from the PathBank pathways description CSV.

    Accepts the PathBank pathways CSV layouts (``SMP ID`` / ``Name`` in the
    official download, or lowercase ``pathway_id`` / ``pathway_name``);
    columns are matched case- and punctuation-insensitively.

    Args:
        names_file: path to ``pathbank_pathways.csv``.

    Returns:
        Dict mapping SMP ID (e.g. ``SMP0000055``) to pathway name. Empty dict
        when the file is missing or has no recognizable columns (callers then
        fall back to the SMP ID as the name).
    """
    path = Path(names_file)
    if not path.exists():
        logger.warning(f"Pathway names CSV not found at {names_file}; "
                       f"pathways will be named by their SMP ID.")
        return {}

    df = pd.read_csv(path, dtype=str)
    cols = {str(c).strip().lower().replace(" ", "_"): c for c in df.columns}
    cols.update({str(c).strip().lower(): c for c in df.columns})

    def find_col(candidates):
        for cand in candidates:
            if cand in cols:
                return cols[cand]
            cand2 = cand.replace(" ", "_")
            if cand2 in cols:
                return cols[cand2]
        return None

    id_col = find_col(_PATHWAY_ID_COLUMNS)
    name_col = find_col(_PATHWAY_NAME_COLUMNS)
    if id_col is None or name_col is None:
        logger.warning(f"Pathway names CSV {names_file} has no recognizable "
                       f"SMP-ID / name columns (found: {list(df.columns)}); "
                       f"pathways will be named by their SMP ID.")
        return {}

    names = (
        df[[id_col, name_col]]
        .dropna()
        .assign(**{id_col: df[id_col].str.strip().str.upper(),
                   name_col: df[name_col].str.strip()})
        .drop_duplicates(subset=[id_col])
    )
    mapping = dict(zip(names[id_col], names[name_col]))
    logger.info(f"Loaded {len(mapping)} pathway names from {names_file}")
    return mapping


def load_pathbank_pathways(pathbank_file: str,
                             species: str = DEFAULT_SPECIES,
                             pathway_names_file: str = None) -> pd.DataFrame:
    """Load the PathBank primary-pathways metabolites CSV into a per-(pathway,
    metabolite) table.

    The expected structure is one row per (pathway, metabolite) pair with the
    columns ``pathway_id``, ``metabolite_name``, ``metabolite_id``, ``hmdb_id``,
    ``species``, and ``relation`` (plus unused chemical-identifier columns).
    Only rows whose ``species`` matches ``species`` are kept; rows without an
    HMDB ID cannot link to the dataset and are dropped. The file carries no
    pathway name; ``pathway_name`` is filled from ``pathway_names_file`` (the
    PathBank pathways description CSV) when given, otherwise with the
    ``pathway_id``.

    Args:
        pathbank_file: path to ``pathbank_all_metabolites.csv``.
        species: species to keep (default ``'Homo sapiens'``). Matched exactly
            (after whitespace stripping) against the CSV's ``species`` column.
        pathway_names_file: optional path to ``pathbank_pathways.csv``
            (SMP ID -> pathway name). Pathways missing from it fall back to
            their SMP ID.

    Returns:
        DataFrame with the columns ``smp_id``, ``pathway_name`` (equal to
        ``smp_id``), ``metabolite_id``, ``metabolite_name``, ``hmdb_id``,
        ``species``, and ``relation`` (one row per (pathway, metabolite) pair).
    """
    out_columns = ["smp_id", "pathway_name", "metabolite_id",
                   "metabolite_name", "hmdb_id", "species", "relation"]

    species = str(species).strip()
    path = Path(pathbank_file)
    if not path.exists():
        logger.error(f"PathBank all-metabolites CSV not found at {pathbank_file}")
        return pd.DataFrame(columns=out_columns)

    df = pd.read_csv(path, dtype=str)
    required = {"pathway_id", "hmdb_id", "species"}
    missing = required - set(df.columns)
    if missing:
        logger.error(f"PathBank CSV {pathbank_file} missing columns: {missing}")
        return pd.DataFrame(columns=out_columns)

    optional = {"metabolite_id": "metabolite_id", "metabolite_name": "metabolite_name",
                "relation": "relation"}
    for src in optional:
        if src not in df.columns:
            df[src] = ""

    df = df.rename(columns={"pathway_id": "smp_id", **optional})
    n_raw = len(df)

    df["smp_id"] = df["smp_id"].fillna("").str.strip()
    df["species"] = df["species"].fillna("").str.strip()
    df["hmdb_id"] = df["hmdb_id"].fillna("").str.strip().str.upper()
    for col in ("metabolite_id", "metabolite_name", "relation"):
        df[col] = df[col].fillna("").str.strip()
    df = df[df["smp_id"] != ""].copy()

    # Fail fast on a filter value that matches nothing (a config typo would
    # otherwise silently produce an empty pathway set).
    if len(df) and not df["species"].eq(species).any():
        available = sorted(df["species"].unique())
        logger.error(f"Species '{species}' not found in PathBank CSV. Available: {available}")
        return pd.DataFrame(columns=out_columns)

    df = df[df["species"] == species].copy()
    df = df[df["hmdb_id"] != ""].copy()
    pathway_names = load_pathway_names(pathway_names_file) if pathway_names_file else {}
    df["pathway_name"] = df["smp_id"].map(pathway_names).fillna(df["smp_id"])
    n_unnamed = int(df.loc[df["pathway_name"] == df["smp_id"], "smp_id"].nunique())
    if pathway_names_file and n_unnamed:
        logger.warning(f"{n_unnamed} pathways not found in {pathway_names_file}; "
                       f"they keep their SMP ID as name.")
    df = df[out_columns].drop_duplicates().reset_index(drop=True)

    logger.info(
        f"Loaded {n_raw} rows from {pathbank_file}; kept {len(df)} "
        f"({species} rows with an HMDB ID) across {df['smp_id'].nunique()} pathways."
    )
    if len(df):
        by_relation = df["relation"].value_counts().to_dict()
        logger.info(f"Metabolite relations: {by_relation}")
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
                            min_name_length: int = 3,
                            overrides: Dict[str, str] = None
                            ) -> pd.DataFrame:
    """Match every feature column to one or more HMDB accessions.

    Matching priority (first hit wins; the method is recorded):

    0. **Manual override**: the exact feature name is a key of
       ``overrides`` (from the ``feature_hmdb_overrides`` config). Used to
       fix name collisions the automatic chain resolves wrongly (e.g. a
       feature named 'Niacin' landing on the nicotinamide accession).
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
        overrides: optional exact feature name -> HMDB accession map applied
            before the automatic chain.

    Returns:
        DataFrame with columns ``feature``, ``hmdb_id`` (one row per matched
        accession), ``match_method`` ('override' / 'hmdb_tag' / 'name_exact' /
        'name_loose'), and ``n_hmdb_ids`` (number of accessions matched for
        the feature). Features that match nothing get a single row with
        hmdb_id NaN and match_method 'unmatched'.
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
    n_override = 0
    n_tagged = 0
    n_exact = 0
    n_loose = 0
    n_unmatched = 0

    for col in feature_columns:
        hmdb_ids: Set[str] = set()
        method = "unmatched"

        # 0) Manual override -- highest priority, fixes known collisions.
        if overrides and col in overrides:
            hmdb_ids = {str(overrides[col]).strip().upper()}
            method = "override"
            n_override += 1
        # 1) HMDB tag -- authoritative, ignore the name index.
        elif (tagged := _split_feature_name_and_hmdb(col)):
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

    logger.info(f"Feature -> HMDB matching: {n_override} by manual override, "
                f"{n_tagged} by HMDB tag, {n_exact} by "
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
    listed for it in the PathBank table (after the species filter). ``coverage``
    is the fraction of those metabolites that are mapped to features in the
    dataset. Only pathways with ``coverage >= min_coverage`` are returned.

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


def filter_pathways_by_keywords(coverage: pd.DataFrame,
                                exclude_keywords: List[str]) -> pd.DataFrame:
    """Drop pathways whose name matches any exclusion keyword (case-insensitive).

    Chemistry-based pathway curation (evidence budget #3): pathways whose
    metabolites the sample preparation cannot recover -- e.g. complex lipids
    under a phase-extraction protocol -- are analytically invalid regardless
    of how well they calibrate, and leaving them in inflates the normals'
    null distributions. The criterion is the keyword list in the frozen
    configuration, applied as a block to the pathway NAME; it never reads
    disease labels or flag performance.

    Args:
        coverage: output of :func:`pathway_coverage` (one row per pathway).
        exclude_keywords: substrings matched case-insensitively against
            ``pathway_name``; a pathway matching any keyword is dropped.

    Returns:
        The coverage frame without the matching pathways.
    """
    if exclude_keywords is None:
        exclude_keywords = []
    elif isinstance(exclude_keywords, str):
        exclude_keywords = [exclude_keywords]
    keywords = [str(k).strip().lower() for k in exclude_keywords
                if str(k).strip()]
    if coverage.empty or not keywords:
        return coverage
    name = coverage["pathway_name"].fillna("").str.lower()
    hit = name.apply(lambda n: any(k in n for k in keywords))
    dropped = coverage[hit]
    kept = coverage[~hit]
    if len(dropped):
        names = sorted(dropped["pathway_name"].unique())
        logger.info(f"Pathway keyword curation: dropped {len(dropped)} "
                    f"pathway(s) matching {keywords}: {names}")
    return kept


def prefer_tagged_features(feature_to_hmdb: pd.DataFrame,
                            feature_columns: List[str]) -> Tuple[pd.DataFrame,
                                                                 pd.DataFrame]:
    """Drop plain-named features superseded by a standard-confirmed twin.

    Chemistry-based identity curation (evidence budget #3): feature columns
    carrying a trailing ``.HMDB########`` tag (or a bare accession) are
    identified by pure-standard injection upstream, so within a resolved
    HMDB metabolite the tagged feature is the molecule described and its
    plain-named twins -- usually co-eluting interlopers caught by name
    matching, often with razor-thin reference IQRs -- are not. Mixing the
    two corrupts the metabolite's z at ANY aggregation weight, so the plain
    twins are removed from scoring before any downstream stage sees them.

    The rule is class-level and label-blind: it reads only column naming
    and resolved HMDB identity, never flag performance. Tagged-vs-tagged
    pairs sharing an ID are both kept (both are confirmed; scale^2
    weighting handles their relative noise). Manual overrides are also
    identity claims, so they confer the same precedence as a tag.

    Args:
        feature_to_hmdb: output of :func:`match_features_to_hmdb`.
        feature_columns: the dataset feature column names (for the
            tagged-column classification).

    Returns:
        Tuple ``(curated, dropped)``: the feature->HMDB table without the
        superseded plain twins, and a per-dropped-feature table with
        columns ``feature``, ``hmdb_id``, ``superseded_by``.
    """
    if feature_to_hmdb.empty:
        return feature_to_hmdb.copy(), pd.DataFrame(
            columns=["feature", "hmdb_id", "superseded_by"])

    is_confirmed = {col: (_split_feature_name_and_hmdb(col) is not None)
                    for col in feature_columns}
    override_features = set(feature_to_hmdb.loc[
        feature_to_hmdb["match_method"] == "override", "feature"])
    for col in override_features:
        is_confirmed[col] = True

    matched = feature_to_hmdb[feature_to_hmdb["hmdb_id"].notna()]
    all_by_id = (matched.groupby("hmdb_id")["feature"]
                 .agg(lambda s: sorted(set(s))).to_dict())

    drop_rows = []
    drop_features = set()
    for acc, twins in all_by_id.items():
        confirmed = [t for t in twins if is_confirmed.get(t, False)]
        plain = [t for t in twins if not is_confirmed.get(t, False)]
        if not confirmed or not plain:
            continue
        survivor = confirmed[0]
        for feat in plain:
            drop_rows.append({
                "feature": feat,
                "hmdb_id": acc,
                "superseded_by": survivor,
            })
            drop_features.add(feat)

    dropped = pd.DataFrame(drop_rows,
                           columns=["feature", "hmdb_id", "superseded_by"])
    curated = feature_to_hmdb[~feature_to_hmdb["feature"].isin(drop_features)]

    if len(dropped):
        logger.info(f"Identity curation: dropped {len(dropped)} plain-named "
                    f"feature(s) superseded by a standard-confirmed "
                    f"(.HMDB-tagged) twin sharing the same metabolite; "
                    f"{curated['feature'].nunique()} features remain matched.")
        for r in dropped.sort_values("feature").itertuples():
            logger.debug(f"  dropped {r.feature} (HMDB {r.hmdb_id}) in "
                         f"favor of {r.superseded_by}")
    return curated, dropped


def ambiguous_feature_report(feature_to_hmdb: pd.DataFrame,
                             exclude: Optional[List[str]] = None
                             ) -> pd.Series:
    """Collect features whose plain name matched multiple HMDB accessions.

    A feature matched by name to more than one accession cannot be
    identified from the name alone, yet it scores into every pathway
    containing any of its identities. This report surfaces those
    features so each can be demoted or given an explicit override
    before the configuration is frozen.

    Args:
        feature_to_hmdb: output of :func:`match_features_to_hmdb`.
        exclude: feature names to omit (e.g. already-demoted artifacts
            whose ambiguity has been resolved by a demotion decision).

    Returns:
        A Series indexed by feature name whose values are the
        comma-joined sorted HMDB accessions it matched.
    """
    excluded = set(exclude or [])
    multi = feature_to_hmdb[feature_to_hmdb["n_hmdb_ids"] > 1]
    multi = multi[~multi["feature"].isin(excluded)]
    return (multi.groupby("feature")["hmdb_id"]
            .apply(lambda s: ",".join(sorted(str(x) for x in s))))


def ambiguous_feature_impact(ambiguous: pd.Series,
                             feature_to_pathway: pd.DataFrame,
                             scored_coverage: pd.DataFrame,
                             disease_resolved: pd.DataFrame = None
                             ) -> pd.DataFrame:
    """Annotate ambiguous features with scoring impact for triage.

    Most multi-matched names are xenobiotics that never reach a scored
    pathway or a disease test; the actionable subset is the features
    that actually contribute to the pathway or biomarker channels.
    This joins the ambiguity report to scored PathBank pathways and
    resolved IEMbase disease tests so the review can be prioritized.

    Args:
        ambiguous: output of :func:`ambiguous_feature_report` (feature
            name -> comma-joined HMDB accessions).
        feature_to_pathway: (feature, pathway) links.
        scored_coverage: pathways kept for scoring (one row per
            pathway, with a ``smp_id`` column).
        disease_resolved: resolved IEMbase disease biomarker tests
            (one row per (disease, biomarker), with a ``features``
            column); None or empty disables the disease annotation.

    Returns:
        A frame with columns ``feature``, ``hmdb_ids``,
        ``n_scored_pathways``, ``n_disease_tests``, sorted by
        descending impact (scored pathways first, then disease tests,
        then feature name).
    """
    if ambiguous.empty:
        return pd.DataFrame(columns=["feature", "hmdb_ids",
                                    "n_scored_pathways",
                                    "n_disease_tests"])
    scored_ids = set(scored_coverage["smp_id"]) \
        if scored_coverage is not None and not scored_coverage.empty else set()
    links = feature_to_pathway
    if links is None or links.empty:
        links = pd.DataFrame(columns=["feature", "smp_id"])
    links = links[links["smp_id"].isin(scored_ids)]
    n_pathways = links.groupby("feature")["smp_id"].nunique()
    if disease_resolved is None or disease_resolved.empty:
        n_disease = pd.Series(dtype="int64")
    else:
        tests = disease_resolved.copy()
        tests["features"] = tests["features"].fillna("")
        mask = tests["features"].str.split(";").apply(
            lambda fs: any(f in ambiguous.index for f in fs))
        tests = tests[mask]
        n_disease = (tests.groupby("features")["disease"].nunique())
        n_disease.index = [fs.split(";")[0] for fs in n_disease.index]
    out = pd.DataFrame({
        "feature": ambiguous.index,
        "hmdb_ids": ambiguous.values,
    })
    out["n_scored_pathways"] = (out["feature"].map(n_pathways)
                                .fillna(0).astype("int64"))
    out["n_disease_tests"] = (out["feature"].map(n_disease)
                              .fillna(0).astype("int64"))
    return out.sort_values(
        by=["n_scored_pathways", "n_disease_tests", "feature"],
        ascending=[False, False, True]).reset_index(drop=True)
