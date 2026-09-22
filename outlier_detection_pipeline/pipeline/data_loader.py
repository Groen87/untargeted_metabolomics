"""
Data loading and preprocessing module for outlier detection pipeline.

Handles:
- Loading merged_data_with_classification.csv
- Identifying feature vs non-feature columns
- Filtering features against an HMDB endogenous metabolites keep-list
  - Reads a precomputed TSV produced by hmdb_drug_filter.py
    (columns: HMDB_ID, Name, Synonyms) listing metabolites with
    Metabolic or Disease pathways
  - Keeps only feature columns that match a name in the keep-list
  - HMDB-prefixed feature columns are always kept
- Splitting data into train/validation/test sets based on Classification
"""

import pandas as pd
import numpy as np
import unicodedata
import re
from typing import Tuple, Dict, List, Optional, Set
from sklearn.model_selection import train_test_split
import logging
from pathlib import Path
import pickle
import hashlib

from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathways_tsv,
    match_features_to_hmdb,
)
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.name_utils import normalize_name, normalize_loose


# Greek symbols -> spelled-out English canonical token. Metabolomics data
# mixes three spellings of the same concept, e.g. omega:
#   - Greek symbol : 18:2ω7
#   - Latin letter : 18:2W7         (lipid shorthand in the HMDB keep-list)
#   - spelled out  : omega-3
# NFKC folds ω to Ω but NOT to W or 'omega', so without explicit folding a
# feature named 'PS(18:2ω7)' never matches the keep-list entry 'PS(18:2W7)',
# and lipid features silently drop out. Similarly 'α-Keto...' (symbol) must
# match 'alpha-Keto...' (spelled out). We fold both to a single canonical
# spelled-out token so all three spellings collapse to one.
_GREEK_SYMBOL_TO_WORD = {
    'α': 'ALPHA', 'Α': 'ALPHA',  # alpha
    'β': 'BETA', 'Β': 'BETA',  # beta
    'γ': 'GAMMA', 'Γ': 'GAMMA',  # gamma
    'δ': 'DELTA', 'Δ': 'DELTA',  # delta
    'ε': 'EPSILON', 'Ε': 'EPSILON',  # epsilon
    'ζ': 'ZETA', 'Ζ': 'ZETA',  # zeta
    'η': 'ETA', 'Η': 'ETA',  # eta
    'θ': 'THETA', 'Θ': 'THETA', 'ϑ': 'THETA',  # theta
    'ι': 'IOTA', 'Ι': 'IOTA',  # iota
    'κ': 'KAPPA', 'Κ': 'KAPPA', 'ϰ': 'KAPPA',  # kappa
    'λ': 'LAMBDA', 'Λ': 'LAMBDA',  # lambda
    'μ': 'MU', 'Μ': 'MU',  # mu
    'ν': 'NU', 'Ν': 'NU',  # nu
    'ξ': 'XI', 'Ξ': 'XI',  # xi
    'ο': 'OMICRON', 'Ο': 'OMICRON',  # omicron
    'π': 'PI', 'Π': 'PI', 'ϖ': 'PI',  # pi
    'ρ': 'RHO', 'Ρ': 'RHO', 'ϱ': 'RHO',  # rho
    'σ': 'SIGMA', 'Σ': 'SIGMA', 'ς': 'SIGMA',  # sigma
    'τ': 'TAU', 'Τ': 'TAU',  # tau
    'υ': 'UPSILON', 'Υ': 'UPSILON',  # upsilon
    'φ': 'PHI', 'Φ': 'PHI', 'ϕ': 'PHI',  # phi
    'χ': 'CHI', 'Χ': 'CHI',  # chi
    'ψ': 'PSI', 'Ψ': 'PSI',  # psi
    'ω': 'OMEGA', 'Ω': 'OMEGA',  # omega
}

# Spelled-out Greek words (case-insensitive) -> same canonical token as the
# symbols above. Lets 'alpha' match 'α', 'omega' match 'ω', etc.
_GREEK_WORD_PATTERNS = [
    (re.compile(r'(?i)\balpha\b'), 'ALPHA'),
    (re.compile(r'(?i)\bbeta\b'), 'BETA'),
    (re.compile(r'(?i)\bgamma\b'), 'GAMMA'),
    (re.compile(r'(?i)\bdelta\b'), 'DELTA'),
    (re.compile(r'(?i)\bepsilon\b'), 'EPSILON'),
    (re.compile(r'(?i)\bzeta\b'), 'ZETA'),
    (re.compile(r'(?i)\beta\b'), 'ETA'),
    (re.compile(r'(?i)\btheta\b'), 'THETA'),
    (re.compile(r'(?i)\biota\b'), 'IOTA'),
    (re.compile(r'(?i)\bkappa\b'), 'KAPPA'),
    (re.compile(r'(?i)\blambda\b'), 'LAMBDA'),
    (re.compile(r'(?i)\bmu\b'), 'MU'),
    (re.compile(r'(?i)\bnu\b'), 'NU'),
    (re.compile(r'(?i)\bxi\b'), 'XI'),
    (re.compile(r'(?i)\bomicron\b'), 'OMICRON'),
    (re.compile(r'(?i)\bpi\b'), 'PI'),
    (re.compile(r'(?i)\brho\b'), 'RHO'),
    (re.compile(r'(?i)\bsigma\b'), 'SIGMA'),
    (re.compile(r'(?i)\btau\b'), 'TAU'),
    (re.compile(r'(?i)\bupsilon\b'), 'UPSILON'),
    (re.compile(r'(?i)\bphi\b'), 'PHI'),
    (re.compile(r'(?i)\bchi\b'), 'CHI'),
    (re.compile(r'(?i)\bpsi\b'), 'PSI'),
    (re.compile(r'(?i)\bomega\b'), 'OMEGA'),
]

# Lipid-shorthand omega: a lone 'W' sitting between two digits (e.g.
# '18:2W7', '24:1W9') denotes an omega double bond. Fold it to OMEGA so the
# HMDB keep-list's 'PS(18:2W6/24:1W9)' form matches feature columns using the
# Greek symbol ('PS(18:2ω6/24:1ω9)') or the spelled word.
_LIPID_W = re.compile(r'(?<=\d)W(?=\d)', re.IGNORECASE)

# Strip every non-alphanumeric character (used only as a loose fallback for
# endogenous matching, after the exact normalized match misses). Collapses
# hyphen/space/punctuation/quote differences such as 'Coproporphyrin III' vs
# 'Coproporphyrin-III' -> 'COPROPORPHYRINIII'. Kept separate from
# _normalize_name so the primary (exact) path is unchanged.
_NON_ALNUM = re.compile(r'[^A-Z0-9]+')

# Bump this when _normalize_name's output changes (e.g. Greek-letter folding
# added in v2) so the endogenous keep-list cache is rebuilt instead of
# reusing a keep-list normalized with the old logic.
_NORMALIZATION_VERSION = 3


def _normalize_name(name: str) -> str:
    """
    Normalize a metabolite or feature-column name for exact matching.

    - Strips surrounding whitespace and a leading UTF-8 BOM
    - Applies Unicode NFKC normalization (folds full-width digits, unifies
      some compatibility forms)
    - Folds Greek symbols (α, β, ω, ...) AND spelled-out Greek words
      (alpha, beta, omega, ...) to a single spelled-out English canonical
      token, and folds lipid-shorthand 'W' between digits to OMEGA. This is
      the critical step: the HMDB keep-list spells lipid omega as '18:2W7'
      while feature columns use '18:2ω7' or '18:2omega7'; without this folding
      they never match and lipid features are silently dropped.
    - Uppercases

    Returns the normalized name. Exact (not partial) matching is preserved.
    """
    if name is None:
        return ''
    s = str(name)
    if s.startswith('\ufeff'):
        s = s[1:]
    s = unicodedata.normalize('NFKC', s)
    # Symbols -> spelled-out canonical token.
    s = ''.join(_GREEK_SYMBOL_TO_WORD.get(ch, ch) for ch in s)
    # Spelled-out words -> same canonical token (so 'alpha' == 'α').
    for pattern, repl in _GREEK_WORD_PATTERNS:
        s = pattern.sub(repl, s)
    # Lipid-shorthand 'W' between digits -> OMEGA (so '18:2W7' == '18:2ω7').
    s = _LIPID_W.sub('OMEGA', s)
    return s.strip().upper()


def _normalize_loose(name: str) -> str:
    """
    Aggressive normalization used only as a fallback for endogenous matching.

    Applies _normalize_name (NFKC + Greek folding + uppercasing) and then
    removes every non-alphanumeric character. This collapses differences in
    hyphenation, spacing, punctuation, and quotes that exact matching cannot
    bridge, e.g.:
        'Coproporphyrin III' -> 'COPROPORPHYRINIII'
        'Coproporphyrin-III' -> 'COPROPORPHYRINIII'
        'D-Mannoheptulose'    -> 'DMANNOHEPTULOSE'
        'D-Manno-heptulose'   -> 'DMANNOHEPTULOSE'
    Returns '' for names that reduce to nothing usable.
    """
    s = _normalize_name(name)
    s = _NON_ALNUM.sub('', s)
    return s

logger = logging.getLogger(__name__)


def _get_hmdb_cache_path(endogenous_file: str) -> Path:
    """
    Get the cache file path for a given endogenous metabolites file.

    The cache key combines the file PATH with the file's CONTENT hash (md5 of
    the bytes) and last-modified time. This means the cache is invalidated
    automatically whenever the TSV is edited OR the normalization logic changes
    its output (a content change), so a stale keep-list from an older
    normalization version is never silently reused. The path alone is not
    enough: the same path can hold different contents after the user regenerates
    the TSV or after a normalization fix.

    Args:
        endogenous_file: Path to endogenous_metabolites.tsv

    Returns:
        Path to the cache pickle file
    """
    cache_dir = Path.home() / ".cache" / "hmdb_metabolomics"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.md5(endogenous_file.encode()).hexdigest()[:16]
    content_hash = ""
    mtime = ""
    try:
        p = Path(endogenous_file)
        if p.exists():
            mtime = str(int(p.stat().st_mtime))
            content_hash = hashlib.md5(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    # Bump _NORMALIZATION_VERSION whenever _normalize_name's output changes
    # (e.g. Greek-letter folding) so a stale cache from the old logic is never
    # reused, even when the TSV content itself is unchanged.
    key = f"{path_hash}_{content_hash}_{mtime}_v{_NORMALIZATION_VERSION}"
    return cache_dir / f"endogenous_names_{key}.pkl"


def _load_endogenous_metabolite_names(endogenous_file: str, use_cache: bool = True) -> Tuple[Set[str], Set[str]]:
    """
    Load endogenous metabolite name sets from a precomputed TSV file.

    The TSV is produced by hmdb_drug_filter.py and contains metabolites that
    have at least one Metabolic or Disease pathway. It has the columns:
        HMDB_ID <TAB> Name <TAB> Synonyms
    where Synonyms is a '; '-separated list of alternative names.

    Returns two sets, both as uppercase normalized strings:
      - exact set: the HMDB ID, the primary Name, and each synonym, normalized
        with _normalize_name (NFKC + Greek folding). Used for exact matching.
      - loose set: the same names further reduced with _normalize_loose
        (all non-alphanumeric chars removed), used as a fallback for exact
        matching so hyphenation/spacing/punctuation differences (e.g.
        'Coproporphyrin III' vs 'Coproporphyrin-III') still match.

    Very short names (shorter than 3 characters) are skipped to avoid
    false-positive matches.

    Args:
        endogenous_file: Path to endogenous_metabolites.tsv
        use_cache: Whether to use cached results if available

    Returns:
        Tuple of (exact_names, loose_names). Both empty on failure.
    """
    try:
        cache_path = _get_hmdb_cache_path(endogenous_file)

        # Try to load from cache first
        if use_cache and cache_path.exists():
            with open(cache_path, 'rb') as f:
                cached_data = pickle.load(f)
            logger.info(f"Loaded endogenous metabolite names from cache: {cache_path}")
            return cached_data['endogenous'], cached_data['endogenous_loose']

        endogenous_names: Set[str] = set()
        endogenous_names_loose: Set[str] = set()

        endogenous_path = Path(endogenous_file)
        with open(endogenous_path, 'r', encoding='utf-8') as f:
            header = f.readline().rstrip('\n')
            expected_cols = ['HMDB_ID', 'Name', 'Synonyms']
            cols = [c.strip() for c in header.split('\t')]
            if cols != expected_cols:
                logger.warning(
                    f"Unexpected header in {endogenous_file}: {cols}. "
                    f"Expected {expected_cols}. Proceeding by column position."
                )

            for line in f:
                line = line.rstrip('\n')
                if not line:
                    continue
                fields = line.split('\t')
                # Pad in case Synonyms is missing
                while len(fields) < 3:
                    fields.append('')
                hmdb_id = fields[0]
                name = fields[1]
                synonyms_str = fields[2]

                for raw in (hmdb_id, name):
                    if not raw:
                        continue
                    n_exact = _normalize_name(raw)
                    if n_exact:
                        endogenous_names.add(n_exact)
                    n_loose = _normalize_loose(raw)
                    if n_loose:
                        endogenous_names_loose.add(n_loose)
                if synonyms_str:
                    for syn in synonyms_str.split(';'):
                        if not syn:
                            continue
                        n_exact = _normalize_name(syn)
                        if n_exact:
                            endogenous_names.add(n_exact)
                        n_loose = _normalize_loose(syn)
                        if n_loose:
                            endogenous_names_loose.add(n_loose)

        # Drop very short names that cause false positives (on both sets).
        endogenous_names = {n for n in endogenous_names if len(n) >= 3}
        endogenous_names_loose = {n for n in endogenous_names_loose if len(n) >= 3}

        # Save to cache
        if use_cache:
            with open(cache_path, 'wb') as f:
                pickle.dump({'endogenous': endogenous_names,
                            'endogenous_loose': endogenous_names_loose}, f)
            logger.info(f"Saved endogenous metabolite names cache to {cache_path}")

        logger.info(f"Loaded {len(endogenous_names)} endogenous metabolite names "
                    f"({len(endogenous_names_loose)} loose) from {endogenous_file}")

        if len(endogenous_names) > 0:
            sample_endogenous = list(endogenous_names)[:10]
            logger.info(f"Sample endogenous metabolite names: {sample_endogenous}{'...' if len(endogenous_names) > 10 else ''}")

        return endogenous_names, endogenous_names_loose

    except Exception as e:
        logger.error(f"Failed to load endogenous metabolites file {endogenous_file}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return set(), set()


def _filter_to_endogenous_features(
    features: pd.DataFrame,
    endogenous_names: set,
    endogenous_names_loose: Optional[set] = None,
) -> pd.DataFrame:
    """
    Keep feature columns that match the HMDB endogenous metabolite keep-list.

    Feature columns in metabolomics data are commonly named either as a plain
    metabolite name (e.g. ``Coproporphyrin III``), a bare HMDB ID
    (``HMDB0000063``), or a compound name with an ``.HMDB########`` suffix
    (``Cortisol.HMDB0000063``). Retention rules (all matching is
    case-insensitive after Unicode NFKC normalization):

      - ANY column whose name contains ``HMDB`` is ALWAYS kept. HMDB-annotated
        features are confidently identified metabolites, so the keep-list is
        not applied to them (this mirrors the unconditional HMDB retention used
        elsewhere in the pipeline, e.g. ``feature_filter: 'hmdb'``). OR
      - the full column name exactly matches a Name or synonym in the
        keep-list (exact, after _normalize_name). OR
      - if a loose keep-list is provided and the exact match missed, the
        column's loose-normalized form (all non-alphanumeric chars removed)
        exactly matches a loose-normalized keep-list entry. This catches
        hyphenation/spacing/punctuation differences (e.g.
        'Coproporphyrin III' == 'Coproporphyrin-III' == 'CoproporphyrinIII')
        that exact matching cannot bridge.

    Plain-name columns (no ``HMDB`` token) that do not match either keep-list
    are removed. This is a positive keep-list for plain names only.

    Args:
        features: DataFrame with feature columns
        endogenous_names: Set of HMDB endogenous metabolite names (uppercase,
            normalized) -- includes HMDB_ID, primary Name, and synonyms.
            Used for exact matching.
        endogenous_names_loose: Optional set of the same names reduced with
            _normalize_loose (non-alphanumeric chars removed), used as a
            fallback for exact matching. When None, only exact matching is
            used.

    Returns:
        Filtered DataFrame containing only kept feature columns
    """
    if not endogenous_names:
        logger.warning("No endogenous metabolite names provided. Returning all features.")
        return features

    original_cols = set(features.columns)

    kept_columns = []
    removed_cols = []
    n_matched_hmdb = 0
    n_matched_name = 0
    n_matched_loose = 0
    matched_loose_names = []
    kept_plain_names = []

    for col in features.columns:
        col_norm = _normalize_name(col)

        # 1) HMDB-annotated columns are always kept (the HMDB tag signals a
        #    confidently identified metabolite; the keep-list is not applied
        #    to them, matching the unconditional HMDB retention used elsewhere).
        if 'HMDB' in col_norm:
            kept_columns.append(col)
            n_matched_hmdb += 1
            continue

        # 2) Exact full-name match against the keep-list. The keep-list holds
        #    the primary Name AND every synonym (split on ';') from the TSV, so
        #    this matches plain-name columns by either their name or a synonym.
        if col_norm in endogenous_names:
            kept_columns.append(col)
            n_matched_name += 1
            kept_plain_names.append(col)
            continue

        # 3) Loose fallback: collapse all non-alphanumeric chars and match
        #    against the loose keep-list. Catches hyphenation/spacing/
        #    punctuation differences that exact matching misses, e.g.
        #    'Coproporphyrin III' (feature, space) vs 'Coproporphyrin-III'
        #    (TSV, hyphen) both reduce to 'COPROPORPHYRINIII'.
        if endogenous_names_loose:
            col_loose = _normalize_loose(col)
            if col_loose and len(col_loose) >= 3 and col_loose in endogenous_names_loose:
                kept_columns.append(col)
                n_matched_loose += 1
                matched_loose_names.append(col)
                continue

        removed_cols.append(col)

    filtered_features = features[kept_columns]
    n_removed = len(original_cols) - len(kept_columns)

    logger.info(
        f"Filtered to endogenous metabolite features: {n_removed} features removed, "
        f"{len(kept_columns)} endogenous features retained "
        f"({n_matched_hmdb} HMDB-annotated columns kept always, "
        f"{n_matched_name} plain-name matched exact, "
        f"{n_matched_loose} plain-name matched loose fallback)"
    )
    # Log the kept PLAIN names (exact) so the user can verify directly that
    # endogenous metabolites are being retained.
    if kept_plain_names:
        logger.info(
            f"Kept plain-name endogenous features (exact, {len(kept_plain_names)}): "
            f"{kept_plain_names}"
        )
    # Log the names recovered ONLY by the loose fallback -- these are the
    # ones that previously dropped out due to hyphenation/spacing/punctuation.
    if matched_loose_names:
        logger.info(
            f"Kept plain-name endogenous features (loose fallback, "
            f"{len(matched_loose_names)}): {matched_loose_names}"
        )

    if n_removed > 0:
        # Separately report dropped PLAIN names (no HMDB token); these are the
        # only ones where a keep-list miss can be a real normalization bug
        # rather than a deliberately excluded exogenous compound.
        dropped_plain = [c for c in removed_cols if 'HMDB' not in _normalize_name(c)]
        logger.info(
            f"Dropped {len(dropped_plain)} plain-name features "
            f"(of {n_removed} total dropped); first 10: "
            f"{dropped_plain[:10]}"
        )
        # Near-miss diagnostic: surface dropped plain names that look like a
        # real endogenous metabolite with a naming/annotation difference (e.g.
        # 'Cortisol sulfate' vs keep-list 'Cortisol'), NOT exogenous drugs that
        # merely contain a generic chemistry fragment. A genuine near-miss is
        # a keep-list metabolite name appearing as a WORD-BOUNDARY substring of
        # the feature name, where the keep-list token is a specific metabolite
        # (not a chemical class). We exclude keep-list tokens that are generic
        # chemistry suffixes/fragments and require the feature name to be a
        # plausible metabolite name (no long IUPAC/stereochemistry form).
        if dropped_plain and len(endogenous_names) <= 400000:
            # Generic chemistry-class fragments that appear in many IUPAC names
            # and are NOT specific endogenous metabolites. Dropped names
            # matching only these are almost certainly exogenous.
            generic_fragments = {
                'SULFATE', 'SULFONATE', 'SULFITE', 'PHOSPHATE', 'PHOSPHONATE',
                'CARBOXYLIC ACID', 'DICARBOXYLIC ACID', 'BUTANOIC ACID',
                'PROPANOIC ACID', 'PENTANOIC ACID', 'HEXANOIC ACID',
                'OCTANOIC ACID', 'DECANOIC ACID', 'DODECANOIC ACID',
                'TETRADECANOIC ACID', 'HEXADECANOIC ACID', 'OCTADECANOIC ACID',
                'EICOSANOIC ACID', 'DOCOSANOIC ACID', 'TETRACOSANOIC ACID',
                'DECANOATE', 'DODECANOATE', 'TETRADECANOATE', 'HEXADECANOATE',
                'OCTADECANOATE', 'TETRACOSANOATE', 'DOCOSAHEXAENOATE',
                'PROPANOATE', 'BUTANOATE', 'ACETATE', 'FORMATE', 'BENZOATE',
                'GLUCOSIDE', 'GALACTOSIDE', 'RIBOSIDE', 'NUCLEOSIDE',
                'PHOSPHATE', 'PHOSPHONOOXY', 'HYDROXY', 'AMINO', 'METHYL',
                'ETHYL', 'PHENYL', 'BENZYL', 'GLYCEROL', 'PHOSPHOLIPID',
            }
            min_token_len = 6
            # Candidate keep-list tokens: specific enough (length) to be a real
            # metabolite, not a stereochemistry marker or fragment.
            endo_candidates = [
                n for n in endogenous_names
                if len(n) >= min_token_len and n not in generic_fragments
            ]
            # Heuristic: a feature name is a "plausible metabolite name" (not
            # a long IUPAC/stereochemistry form) if it has no stereochemistry /
            # locant markers and is not excessively long.
            iupac_markers = re.compile(
                r'(\(\d+[RS]?,?\d*[RS]?\)|^\(\d[EZ]?,'  # (1R,2S)-  / (11E)...
                r'|\[[\d.]+[RS]?,'                          # [23.2.2.1...
                r'|\bN\-[A-Z]|\bO\-[A-Z]|'                  # N-Acetyl, O-Methyl
                r'|\bDIHYDROXY\b|\bTRIHYDROXY\b|\bPENTAHYDROXY\b)',
                re.IGNORECASE,
            )

            near_misses = []
            for col in dropped_plain[:200]:
                cn = _normalize_name(col)
                if not cn or len(cn) > 80 or iupac_markers.search(cn):
                    # Skip long IUPAC / stereochemistry names: they are almost
                    # never simple endogenous metabolites even if they share a
                    # fragment.
                    continue
                for en in endo_candidates:
                    # Word-boundary containment, both directions, but only when
                    # the shorter string is a specific metabolite (length>=6).
                    shorter, longer = (en, cn) if len(en) <= len(cn) else (cn, en)
                    if len(shorter) < min_token_len:
                        continue
                    if re.search(r'(?<![A-Z0-9])' + re.escape(shorter) + r'(?![A-Z0-9])', longer):
                        near_misses.append((col, en))
                        break
            if near_misses:
                logger.info(
                    f"Near-miss plain names (word-boundary match to a specific "
                    f"keep-list metabolite; likely a naming/annotation "
                    f"difference, not exogenous): {near_misses[:15]}"
                )
            else:
                logger.info(
                    "No near-miss plain names found among the first 200 dropped "
                    "(dropped plain names appear genuinely absent from the "
                    "keep-list)."
                )
        if len(endogenous_names) > 0:
            sample_keep = list(endogenous_names)[:5]
            logger.info(f"Example keep-list names (normalized): {sample_keep}")

    return filtered_features


def _filter_to_smpdb_hmdb_features(
    features: pd.DataFrame,
    smpdb_pathways_file: str,
    hmdb_xml_file: Optional[str] = None,
    min_name_length: int = 3,
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Filter features to only those resolving to HMDB accessions that appear in
    the SMPDB pathways TSV.

    Uses the EXACT same matching chain as the pathway_pipeline
    (pathway_pipeline.pipeline.pathway_mapping.match_features_to_hmdb):

      1. HMDB tag: a trailing HMDB accession in the column name (bare
         ``HMDB########`` or ``Name.HMDB########``) is taken authoritatively.
      2. Exact name: the normalized column name matches the HMDB XML name
         index (primary names, synonyms, and bare accessions).
      3. Loose name: the non-alphanumeric-stripped column name matches a
         loose-normalized index entry (catches hyphenation/spacing).

    A feature is kept when AT LEAST ONE of its resolved HMDB accessions
    appears in the union of the pathways' ``hmdb_ids`` lists. Pathways are
    never turned into features -- this is purely a feature keep-filter.

    Args:
        features: DataFrame with feature columns
        smpdb_pathways_file: Path to smpdb_kept_pathways.tsv
        hmdb_xml_file: Path to hmdb_metabolites.xml (same file the
            pathway_pipeline uses to build its name index). Required for
            plain-name features; without it only HMDB-tagged features can
            match.
        min_name_length: Skip names shorter than this for name-based matching
            (same default as pathway_pipeline).
        use_cache: Whether to use cached HMDB name index

    Returns:
        Filtered DataFrame containing only features with HMDB codes in pathways
    """
    # Load the pathways TSV and collect the union of pathway HMDB accessions.
    pathways = load_pathways_tsv(smpdb_pathways_file)
    if pathways.empty:
        logger.warning(f"Could not load pathways from {smpdb_pathways_file}. Using all features.")
        return features

    smpdb_hmdb_ids: Set[str] = set()
    for _, row in pathways.iterrows():
        smpdb_hmdb_ids.update(acc.upper() for acc in row['hmdb_ids'])

    if not smpdb_hmdb_ids:
        logger.warning(f"No HMDB IDs found in {smpdb_pathways_file}. Using all features.")
        return features

    logger.info(f"Loaded {len(smpdb_hmdb_ids)} unique HMDB accessions from SMPDB pathways")

    # Build the SAME name index the pathway_pipeline uses (HMDB XML primary
    # names + synonyms -> accessions), so plain-name features resolve to the
    # same accessions as in the pathway_pipeline.
    name_index: Dict[str, Set[str]] = {}
    if hmdb_xml_file:
        name_index = build_name_index(
            hmdb_xml_file, min_name_length=min_name_length, use_cache=use_cache
        )
        if not name_index:
            logger.warning(
                "HMDB name index is empty (XML missing or unreadable); only "
                "HMDB-tagged features can match the SMPDB filter."
            )
    else:
        logger.warning(
            "No hmdb_xml_file configured for the SMPDB filter; only HMDB-tagged "
            "features can match. Set hmdb_xml_file (same file as pathway_pipeline)."
        )

    # Resolve every feature column to HMDB accessions using the pathway
    # pipeline's own matcher, then keep those intersecting the SMPDB set.
    feature_to_hmdb = match_features_to_hmdb(
        feature_columns=list(features.columns),
        name_index=name_index,
        min_name_length=min_name_length,
    )
    matched = feature_to_hmdb.dropna(subset=["hmdb_id"])
    # HMDB-tagged features are ALWAYS kept: the trailing accession in the
    # feature name signals a confident identification, so the feature must
    # not be dropped just because its accession is not on a kept SMPDB pathway.
    tagged_features: Set[str] = {
        feat for feat, method in zip(matched["feature"], matched["match_method"])
        if method == "hmdb_tag"
    }
    smpdb_hits: Set[str] = {
        feat for feat, acc in zip(matched["feature"], matched["hmdb_id"])
        if acc.upper() in smpdb_hmdb_ids
    }
    feature_hits: Set[str] = smpdb_hits | tagged_features

    # Report match-method breakdown for the kept features (same diagnostics
    # style as the pathway_pipeline logging).
    kept_matched = matched[matched["feature"].isin(feature_hits)]
    method_counts = kept_matched["match_method"].value_counts().to_dict()

    kept_columns = [col for col in features.columns if col in feature_hits]
    n_removed = len(features.columns) - len(kept_columns)
    n_tagged_only = len(tagged_features - smpdb_hits)
    logger.info(
        f"Filtered to SMPDB pathway features: {n_removed} features removed, "
        f"{len(kept_columns)} features retained "
        f"(match methods among retained: {method_counts}; {n_tagged_only} "
        f"HMDB-tagged features kept despite no SMPDB pathway)"
    )

    return features[kept_columns]


def _split_feature_name_and_hmdb(col: str) -> Optional[str]:
    """Extract a trailing HMDB accession from a feature column name.
    
    This is the same function as in pathway_pipeline/pathway_mapping.py
    Handles the two annotated forms used in the dataset:
        'Cortisol.HMDB0000063' -> 'HMDB0000063'
        'HMDB0000063'          -> 'HMDB0000063'
    Returns the accession (uppercased) when present, else None.
    """
    norm = normalize_name(col)
    if norm.startswith("HMDB") and len(norm) >= 7 and norm[4:].isdigit():
        return norm
    if "." in norm:
        suffix = norm.rsplit(".", 1)[-1]
        if suffix.startswith("HMDB") and len(suffix) >= 7 and suffix[4:].isdigit():
            return suffix
    return None


def _exclude_metabolites(
    features: pd.DataFrame,
    exclude_names: List[str],
) -> pd.DataFrame:
    """
    Drop feature columns whose names match a user-supplied exclude list.

    Matching is exact and case-insensitive after Unicode normalization (the
    same normalization used for the HMDB keep-list), so a user can enter
    metabolite names with any casing or surrounding whitespace.

    Args:
        features: DataFrame with feature columns
        exclude_names: List of metabolite/feature names to exclude

    Returns:
        DataFrame with the matched feature columns removed
    """
    if not exclude_names:
        return features

    exclude_set = {
        _normalize_name(name)
        for name in exclude_names
        if name is not None and str(name).strip() != ''
    }
    exclude_set = {n for n in exclude_set if n}
    if not exclude_set:
        return features

    original_cols = list(features.columns)
    kept_columns = [
        col for col in original_cols
        if _normalize_name(col) not in exclude_set
    ]
    removed_cols = [col for col in original_cols if col not in kept_columns]

    filtered_features = features[kept_columns]

    logger.info(
        f"Excluded {len(removed_cols)} metabolite features, "
        f"{len(kept_columns)} features remaining"
    )
    if removed_cols:
        logger.info(
            f"Excluded features: {removed_cols[:10]}"
            f"{'...' if len(removed_cols) > 10 else ''}"
        )

    return filtered_features


def _exclude_by_substring(
    features: pd.DataFrame,
    substrings: List[str],
) -> pd.DataFrame:
    """
    Drop feature columns whose names contain any of the given substrings.

    Matching is case-insensitive (after Unicode normalization) substring
    containment, not exact match -- this is for dropping features that mention
    a non-endogenous atom/group, e.g. 'bromo', 'iodo', 'chloro', 'fluoro',
    'silyl', 'cyano', 'boronic'. Complements (does not replace) the
    endogenous keep-list: the keep-list is a positive allow-list, while this
    is a negative deny-list applied to whatever survives.

    Args:
        features: DataFrame with feature columns
        substrings: List of substrings to exclude on (case-insensitive)

    Returns:
        DataFrame with the matched feature columns removed
    """
    if not substrings:
        return features

    norm_subs = {
        _normalize_name(s)
        for s in substrings
        if s is not None and str(s).strip() != ''
    }
    norm_subs = {s for s in norm_subs if s}
    if not norm_subs:
        return features

    original_cols = list(features.columns)
    kept_columns = [
        col for col in original_cols
        if not any(sub in _normalize_name(col) for sub in norm_subs)
    ]
    removed_cols = [col for col in original_cols if col not in kept_columns]

    filtered_features = features[kept_columns]

    logger.info(
        f"Excluded {len(removed_cols)} features by substring filter "
        f"({list(norm_subs)}), {len(kept_columns)} features remaining"
    )
    if removed_cols:
        logger.info(
            f"Excluded features: {removed_cols[:10]}"
            f"{'...' if len(removed_cols) > 10 else ''}"
        )

    return filtered_features


def load_data(
    input_file: str,
    non_feature_columns: List[str],
    patient_id_column: Optional[str] = None,
    endogenous_metabolites_file: Optional[str] = None,
    filter_to_endogenous: bool = False,
    use_hmdb_cache: bool = True,
    exclude_metabolites: Optional[List[str]] = None,
    exclude_medications: Optional[List[str]] = None,
    classification_scheme: str = "default",
    exclude_substrings: Optional[List[str]] = None,
    smpdb_pathways_file: Optional[str] = None,
    filter_to_smpdb_hmdb: bool = False,
    hmdb_xml_file: Optional[str] = None,
    log_hmdb_tagged_features: bool = False,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Load data from CSV file and optionally filter to endogenous metabolite features.

    Args:
        input_file: Path to CSV file
        non_feature_columns: List of column names that are NOT features
        patient_id_column: Column name for patient IDs (if not index)
        endogenous_metabolites_file: Path to endogenous_metabolites.tsv (the
            output of hmdb_drug_filter.py) used as a positive keep-list
        filter_to_endogenous: Whether to filter features to the endogenous
            metabolite keep-list
        use_hmdb_cache: Whether to use cached HMDB data if available
        exclude_metabolites: Optional list of metabolite/feature names to drop
            from the analysis (exact, case-insensitive match)
        exclude_medications: Optional list of medication/drug feature names to
            drop (same matching as exclude_metabolites; kept as a separate key
            so drugs and endogenous metabolites can be curated independently)
        classification_scheme: How to build the binary label from the raw
            Classification column. 'default' keeps the legacy Oordeel-based
            cleaning (drop ambiguous 2/3 with Oordeel=0; reclassify 0 with
            Oordeel!=0 as outlier). 'binary_simplified' ignores Oordeel
            entirely and remaps to a clean binary label: Classification 1 ->
            outlier (1), Classification 2 -> dropped, Classification 0 or 3
            -> inlier (0). 'oordeel' ignores the raw Classification entirely and
            derives the label from Oordeel targeted: 0 -> inlier (0),
            1 -> outlier (1); samples with any other Oordeel value are dropped.
            'confident_normals' is the semi-supervised strategy: training
            inliers are ONLY the confident normals (Classification 0 AND
            Oordeel targeted 0); every other sample is labelled outlier (1)
            and used only for testing. No samples are dropped.
            The 'Non-treated' column (when present) does not change the binary
            label; it is carried through so the lab-protocol roles can split
            the (Classification 1, Oordeel 1) group into non-treated IMD
            (true outliers) and treated IMD (gray).
        smpdb_pathways_file: Path to smpdb_kept_pathways.tsv for filtering to
            HMDB codes in pathways
        filter_to_smpdb_hmdb: Whether to filter features to those with HMDB
            codes present in the SMPDB pathways file
        hmdb_xml_file: Path to hmdb_metabolites.xml (same file as
            pathway_pipeline) used to resolve plain-name features to HMDB
            accessions for the SMPDB filter

    Returns:
        Tuple of:
        - features: DataFrame of features (rows = samples, columns = features)
        - classification: Series with the binary label (0=inlier, 1=outlier)
        - oordeel: Series with the raw Oordeel targeted values
        - raw_classification: Series with the raw Classification values
        - non_treated: Series with the raw 'Non-treated' values (all-NaN when
          the input has no such column)
    """
    logger.info(f"Loading data from {input_file}")

    # Load CSV
    df = pd.read_csv(input_file, index_col=0 if patient_id_column is None else None)

    if patient_id_column is not None:
        df = df.set_index(patient_id_column)
    
    logger.info(f"Loaded data with shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    # Data cleaning: build the binary label from the raw Classification
    # (and Oordeel targeted) columns. Several schemes are supported.
    classification_col = df['Classification']
    oordeel_col = df['Oordeel targeted']

    # Drop samples missing a label in either non-feature column: a NaN in
    # 'Oordeel targeted' or 'Classification' leaves the sample's role
    # undefined, so it cannot be trained or scored against ground truth.
    label_nan_mask = classification_col.isna() | oordeel_col.isna()
    n_label_nan = int(label_nan_mask.sum())
    if n_label_nan > 0:
        logger.info(f"Dropping {n_label_nan} samples with NaN in 'Oordeel "
                    f"targeted' or 'Classification'.")
        df = df[~label_nan_mask]
        classification_col = df['Classification']
        oordeel_col = df['Oordeel targeted']

    # Preserve the raw Classification (before any remap) so the per-group
    # evaluation can break results down by (Class, Oordeel).
    raw_classification = pd.Series(classification_col.values, index=df.index,
                                   name='raw_Classification')

    if classification_scheme == "confident_normals":
        # Semi-supervised strategy (lab protocol):
        #   Training inliers = ONLY confident normals: Classification 0 AND
        #     Oordeel targeted 0 (least ambiguous). Every other sample is
        #     labelled outlier (1) and used only for testing. No samples are
        #     dropped, so all groups (1/2/3 and Oordeel=1) appear in the test set
        #     for the per-group breakdown.
        oordeel_num = pd.to_numeric(pd.Series(oordeel_col), errors='coerce')
        class_num = pd.to_numeric(pd.Series(classification_col), errors='coerce')
        confident_normal = (class_num == 0) & (oordeel_num == 0)
        df['Classification'] = np.where(confident_normal.values, 0, 1)
        n_inlier = int((df['Classification'] == 0).sum())
        n_outlier = int((df['Classification'] == 1).sum())
        logger.info(f"confident_normals: {n_inlier} confident-normal inliers "
                    f"(Class=0 & Oordeel=0), {n_outlier} outliers (all others).")
    elif classification_scheme == "oordeel":
        # Derive the label straight from Oordeel targeted, ignoring the raw
        # Classification column: 0 -> inlier (0), 1 -> outlier (1). Samples with
        # any other Oordeel value are dropped.
        unique_oordeel = sorted(pd.Series(oordeel_col).dropna().unique().tolist())
        logger.info(f"oordeel: Oordeel targeted unique values found: {unique_oordeel}")
        # Coerce to numeric so '0.0'/'1.0'/' 1' also map correctly.
        oordeel_num = pd.to_numeric(pd.Series(oordeel_col), errors='coerce')
        keep_mask = oordeel_num.isin([0, 1])
        n_dropped = int((~keep_mask).sum())
        if n_dropped > 0:
            logger.info(f"oordeel: dropping {n_dropped} samples whose Oordeel "
                        f"targeted is not 0 or 1.")
            df = df[keep_mask]
            oordeel_num = oordeel_num[keep_mask]
        if len(df) == 0:
            raise ValueError(
                f"oordeel scheme: all {len(oordeel_col)} samples were dropped "
                f"because none had Oordeel targeted == 0 or 1. Unique values "
                f"found: {unique_oordeel}. Check that the 'Oordeel targeted' "
                f"column exists and uses 0/1 (the config non_feature_columns "
                f"must list the exact CSV column name, e.g. 'Oordeel trageted' "
                f"if that is the spelling in the CSV)."
            )
        df['Classification'] = np.where(oordeel_num.values == 1, 1, 0)
        n_inlier = int((df['Classification'] == 0).sum())
        n_outlier = int((df['Classification'] == 1).sum())
        logger.info(f"oordeel: {n_inlier} inliers (0), {n_outlier} outliers (1) "
                    f"from Oordeel targeted.")
    elif classification_scheme == "binary_simplified":
        # Ignore Oordeel entirely. Remap to a clean binary label:
        #   Classification 1      -> outlier (1)
        #   Classification 2     -> dropped (ambiguous; not used)
        #   Classification 0 or 3 -> inlier (0)
        drop_mask = classification_col.isin([2])
        n_dropped = int(drop_mask.sum())
        if n_dropped > 0:
            logger.info(f"binary_simplified: dropping {n_dropped} samples "
                        f"with Classification=2 (ignored).")
            df = df[~drop_mask]
            classification_col = df['Classification']
        inlier_mask = classification_col.isin([0, 3])
        outlier_mask = (classification_col == 1)
        n_inlier = int(inlier_mask.sum())
        n_outlier = int(outlier_mask.sum())
        n_other = int((~inlier_mask & ~outlier_mask).sum())
        if n_other > 0:
            logger.warning(f"binary_simplified: {n_other} samples had an "
                           f"unrecognised Classification value; dropping them.")
            df = df[inlier_mask | outlier_mask]
            classification_col = df['Classification']
        df['Classification'] = np.where(df['Classification'] == 1, 1, 0)
        logger.info(f"binary_simplified: {n_inlier} inliers (0), {n_outlier} "
                    f"outliers (1) after remap.")
    else:
        # Default (legacy) Oordeel-based cleaning.
        # Remove ambiguous: Classification 2 or 3 with Oordeel targeted = 0
        ambiguous_mask = ((classification_col.isin([2, 3])) & (oordeel_col == 0))
        n_ambiguous = ambiguous_mask.sum()

        if n_ambiguous > 0:
            ambiguous_indices = df.index[ambiguous_mask]
            logger.warning(f"Found {n_ambiguous} ambiguous samples (Classification 2/3 with Oordeel targeted=0). Removing these.")
            df = df[~ambiguous_mask]
            logger.warning(f"Removed samples: {list(ambiguous_indices[:5])}{'...' if n_ambiguous > 5 else ''}")

        # After removing ambiguous, update classification to be consistent
        inconsistent_mask = (df['Classification'] == 0) & (df['Oordeel targeted'] != 0)
        n_inconsistent = inconsistent_mask.sum()

        if n_inconsistent > 0:
            inconsistent_indices = df.index[inconsistent_mask]
            logger.warning(f"Found {n_inconsistent} samples with Classification=0 but Oordeel targeted!=0. "
                          f"Updating Classification to 1 (outlier) for consistency.")
            df.loc[inconsistent_mask, 'Classification'] = 1
            logger.warning(f"Updated samples: {list(inconsistent_indices[:5])}{'...' if n_inconsistent > 5 else ''}")
    
    # Extract non-feature columns
    classification = df['Classification']
    oordeel = df['Oordeel targeted']
    # Keep raw_classification aligned to the (possibly subsetted) df index.
    raw_classification = raw_classification.reindex(df.index)
    if 'Non-treated' in df.columns:
        non_treated = pd.Series(df['Non-treated'], index=df.index, name='non_treated')
    else:
        non_treated = pd.Series(np.nan, index=df.index, name='non_treated')
        logger.info("No 'Non-treated' column in the input; non-treated roles "
                    "fall back to the legacy (Class 1 & Oordeel 1) definition.")
    
    # Get feature columns (all columns except non-feature columns). The
    # 'Non-treated' metadata column is kept out of the features even when it
    # is not listed in non_feature_columns, so it can never leak in as a
    # feature.
    feature_cols = [col for col in df.columns if col not in non_feature_columns
                    and col != 'Non-treated']
    features = df[feature_cols]
    
    # Exclude user-specified metabolite features (exact, case-insensitive match)
    if exclude_metabolites:
        features = _exclude_metabolites(features, exclude_metabolites)

    # Exclude feature columns whose names contain non-endogenous atom/group
    # substrings (e.g. 'bromo', 'iodo', 'chloro', 'fluoro', 'silyl', 'cyano',
    # 'boronic'). Case-insensitive substring deny-list; complements the
    # endogenous keep-list.
    if exclude_substrings:
        features = _exclude_by_substring(features, exclude_substrings)

    # Filter to endogenous metabolite features if requested
    if filter_to_endogenous and endogenous_metabolites_file:
        endogenous_path = Path(endogenous_metabolites_file)
        if endogenous_path.exists():
            endogenous_names, endogenous_names_loose = _load_endogenous_metabolite_names(str(endogenous_path), use_cache=use_hmdb_cache)
            if endogenous_names:
                features = _filter_to_endogenous_features(
                    features, endogenous_names, endogenous_names_loose
                )
            else:
                logger.warning(f"Could not load endogenous metabolite names from {endogenous_metabolites_file}. Using all features.")
        else:
            logger.warning(f"Endogenous metabolites file not found at {endogenous_metabolites_file}. Using all features.")
    
    # Filter to SMPDB pathway HMDB features if requested
    if filter_to_smpdb_hmdb and smpdb_pathways_file:
        features = _filter_to_smpdb_hmdb_features(
            features,
            smpdb_pathways_file,
            hmdb_xml_file=hmdb_xml_file,
            use_cache=use_hmdb_cache,
        )
    
    # Optionally print all HMDB-tagged feature names (one per line) so they
    # can be reviewed and curated into e.g. a medication exclusion list.
    if log_hmdb_tagged_features:
        tagged = [col for col in features.columns if _split_feature_name_and_hmdb(col)]
        logger.info(
            f"HMDB-tagged features ({len(tagged)}):\n" + "\n".join(tagged)
        )

    # Exclude medication/drug features AFTER the SMPDB/HMDB-tag keep-filter,
    # so medications are removed from exactly the set the model will train on.
    # Same exact, case-insensitive match as exclude_metabolites.
    if exclude_medications:
        features = _exclude_metabolites(features, exclude_medications)
    
    logger.info(f"Feature columns: {len(features.columns)}")
    logger.info(f"Non-feature columns: {non_feature_columns}")
    
    return features, classification, oordeel, raw_classification, non_treated


def split_data(
    features: pd.DataFrame,
    classification: pd.Series,
    normal_classification: int,
    outlier_classifications: List[int],
    train_ratio: float = 0.8,
    test_ratio: float = 0.2,
    random_seed: int = 42,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """
    Split data into train and test sets using stratified split.
    
    For Extended Isolation Forest (unsupervised):
    - Stratified train-test split (80-20) to maintain class distribution
    - Train set contains both normal and abnormal samples
    - Test set contains both normal and abnormal samples
    - During CV: train only on normal samples from training folds
    - Validate on full validation folds (including abnormalities)
    
    Args:
        features: DataFrame of features
        classification: Series with Classification values
        normal_classification: Classification value for normal samples
        outlier_classifications: List of outlier classification values
        train_ratio: Ratio for training set (default: 0.8)
        test_ratio: Ratio for test set (default: 0.2)
        random_seed: Random seed for reproducibility
    
    Returns:
        Dictionary with keys: 'train', 'test'
        Each value is a tuple of (features, classification)
    """
    # Check for NaN in classification and drop if present
    df_combined = pd.concat([features, classification.rename('Classification')], axis=1)
    df_combined = df_combined.dropna(subset=['Classification'])
    
    if classification.isna().any():
        n_dropped = classification.isna().sum()
        logger.warning(f"Found {n_dropped} NaN values in Classification. Dropping these samples.")
    
    features = df_combined[features.columns]
    classification = df_combined['Classification']
    
    # Stratified train-test split (maintains class distribution)
    X_for_split = pd.DataFrame(index=features.index)
    X_for_split['classification'] = classification.values
    
    train_df, test_df = train_test_split(
        X_for_split,
        train_size=train_ratio,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=classification,
    )
    
    train_indices = train_df.index
    test_indices = test_df.index
    
    logger.info(f"Train set: {len(train_indices)} samples")
    logger.info(f"Test set: {len(test_indices)} samples")
    logger.info(f"Train class distribution: {classification[train_indices].value_counts().to_dict()}")
    logger.info(f"Test class distribution: {classification[test_indices].value_counts().to_dict()}")
    
    # Create splits
    splits = {}
    for name, indices in [('train', train_indices), ('test', test_indices)]:
        splits[name] = (
            features.loc[indices].copy(),
            classification.loc[indices].copy(),
        )
    
    return splits


def get_class_distribution(classification: pd.Series) -> Dict[int, int]:
    """Get distribution of classification values."""
    return classification.value_counts().to_dict()
