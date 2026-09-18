"""Streaming parser for the HMDB metabolite XML file.

The HMDB XML (``hmdb_metabolites.xml``) is a single root element containing many
``<metabolite>`` elements, each with an ``<accession>`` (e.g. HMDB0000063), a
primary ``<name>``, and zero or more ``<synonyms>`` (each a free-text
alternate name). The file is typically several hundred MB, so it must be
parsed incrementally with :func:`xml.etree.ElementTree.iterparse` (one
``<metabolite>`` at a time, clearing each element after extraction) rather
than loaded whole.

This module produces a name -> HMDB accession index:

    {
        normalized_name: set([HMDB accession, ...]),
        ...
    }

mapping every primary name and synonym (normalized via
:mod:`pathway_pipeline.pipeline.name_utils`) to the set of HMDB accessions
that share it. A normalized name can map to several accessions (two metabolites
share a synonym), so the value is a set.

The index is cached to disk keyed by the XML file's content hash + the
normalization version + ``min_name_length``, so the expensive parse runs only
once per (file, normalization) pair.
"""

import hashlib
import logging
import pickle
import re
from pathlib import Path
from typing import Dict, Set
from xml.etree import ElementTree as ET

from .name_utils import normalize_name, _NORMALIZATION_VERSION


logger = logging.getLogger(__name__)

_HMDB_ACCESSION_RE = re.compile(r"^HMDB\d+$", re.IGNORECASE)


def _cache_path(hmdb_xml_file: str, min_name_length: int) -> Path:
    """Return the on-disk cache path for a parsed HMDB name index.

    The cache key combines the file PATH, the file's CONTENT hash (md5 of the
    bytes), its mtime, the normalization version, and ``min_name_length``. A
    stale cache from an older normalization version or a regenerated XML is
    therefore never silently reused.
    """
    cache_dir = Path.home() / ".cache" / "pathway_pipeline_hmdb"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path_hash = hashlib.md5(hmdb_xml_file.encode()).hexdigest()[:16]
    content_hash = ""
    mtime = ""
    try:
        p = Path(hmdb_xml_file)
        if p.exists():
            mtime = str(int(p.stat().st_mtime))
            content_hash = hashlib.md5(p.read_bytes()).hexdigest()[:16]
    except Exception:
        pass
    key = f"{path_hash}_{content_hash}_{mtime}_v{_NORMALIZATION_VERSION}_min{min_name_length}"
    return cache_dir / f"hmdb_name_index_{key}.pkl"


def _extract_all_names(metabolite: ET.Element):
    """Yield (accession, raw_name) for the accession, primary name, and synonyms.

    Walks the metabolite once and yields every name-bearing element. The
    accession is yielded first (indexed as a bare 'HMDB########' so a feature
    column named exactly that matches directly), then the primary ``<name>``,
    then each ``<synonym>`` (whether nested under a ``<synonyms>`` container or
    a direct child). Yields nothing when the metabolite has no valid HMDB
    accession.
    """
    accession = ""
    primary_name = ""
    synonyms = []
    for child in metabolite:
        tag = child.tag
        if tag == "accession" and child.text:
            accession = child.text.strip()
        elif tag == "name" and child.text:
            primary_name = child.text.strip()
        elif tag == "synonyms":
            for sub in child:
                if sub.tag == "synonym" and sub.text:
                    synonyms.append(sub.text.strip())
        elif tag == "synonym" and child.text:
            synonyms.append(child.text.strip())

    if not accession or not _HMDB_ACCESSION_RE.match(accession):
        return
    yield accession, accession
    if primary_name:
        yield accession, primary_name
    for syn in synonyms:
        if syn:
            yield accession, syn


def build_name_index(hmdb_xml_file: str, min_name_length: int = 3,
                     use_cache: bool = True) -> Dict[str, Set[str]]:
    """Build (or load the cached) normalized-name -> HMDB-accession index.

    Args:
        hmdb_xml_file: Path to ``hmdb_metabolites.xml``.
        min_name_length: Skip names shorter than this (after normalization).
        use_cache: If true, load/save the parsed index from/to disk.

    Returns:
        Dict mapping each normalized name (uppercase, Greek-folded) to the set
        of HMDB accessions that use it. Empty dict on failure.
    """
    if use_cache:
        cache = _cache_path(hmdb_xml_file, min_name_length)
        if cache.exists():
            try:
                with open(cache, "rb") as f:
                    index = pickle.load(f)
                logger.info(f"Loaded cached HMDB name index from {cache} "
                            f"({len(index)} names).")
                return index
            except Exception as e:
                logger.warning(f"Failed to load HMDB cache {cache}: {e}; re-parsing.")

    xml_path = Path(hmdb_xml_file)
    if not xml_path.exists():
        logger.error(f"HMDB XML file not found at {hmdb_xml_file}")
        return {}

    index: Dict[str, Set[str]] = {}
    n_metabolites = 0
    n_names = 0
    context = ET.iterparse(str(xml_path), events=("end",))
    for event, elem in context:
        if elem.tag != "metabolite":
            continue
        n_metabolites += 1
        for accession, raw_name in _extract_all_names(elem):
            norm = normalize_name(raw_name)
            if not norm or len(norm) < min_name_length:
                continue
            index.setdefault(norm, set()).add(accession)
            n_names += 1
        # Free this metabolite's subtree from memory so the stream stays
        # bounded. (stdlib ElementTree lacks lxml's getprevious(), so we only
        # clear the element; that is sufficient to release its child text.)
        elem.clear()
        if n_metabolites % 10000 == 0:
            logger.info(f"Parsed {n_metabolites} metabolites, {n_names} names so far...")

    logger.info(f"Parsed {n_metabolites} metabolites, {n_names} names from "
                f"{hmdb_xml_file}; index has {len(index)} unique normalized names.")

    if use_cache and index:
        try:
            with open(cache, "wb") as f:
                pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Saved HMDB name index cache to {cache}")
        except Exception as e:
            logger.warning(f"Failed to save HMDB cache {cache}: {e}")

    return index


def parse_hmdb_xml(hmdb_xml_file: str, min_name_length: int = 3,
                    use_cache: bool = True) -> Dict[str, Set[str]]:
    """Backwards-compatible alias for :func:`build_name_index`."""
    return build_name_index(hmdb_xml_file, min_name_length=min_name_length,
                            use_cache=use_cache)
