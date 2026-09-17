"""FAIR-TPs public API client for the outlier detection pipeline.

This module queries the public FAIR-TPs knowledge graph
(https://fairtps.lcsb.uni.lu/api/v1) to identify drug-related compounds and
their transformation products (metabolites), so the pipeline can drop those
feature columns as a separate, toggleable filtering step.

Only the Python standard library is used (urllib, json, pickle, hashlib) so
this introduces no new dependency beyond what the pipeline already requires.

API reference (OAS 3.1):
    https://fairtps.lcsb.uni.lu/api/v1/openapi.json

Relevant endpoints:
    GET /api/v1/compounds?q=<name>          -> CompoundList (name/title/IUPAC/synonym search)
    GET /api/v1/compounds/{inchikey}/connections?direction=<both|incoming|outgoing>
        -> ConnectionList (transformation reactions; substrate->product)

Each Connection carries `biosystems`, `enzymes`, `dataset_reference`, and a
`direction` saying whether the queried compound is the substrate or product of
the reaction. The names we collect (parent + every connected compound's name)
are matched back against feature column names.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
import time
import urllib.error
import urllib.parse
import urllib.request
from typing import Dict, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://fairtps.lcsb.uni.lu/api/v1"
DEFAULT_TIMEOUT = 30
DEFAULT_RETRIES = 3
DEFAULT_RETRY_BACKOFF = 1.5
DEFAULT_PAGE_SIZE = 100  # API max is 100
DEFAULT_RATE_LIMIT_SECONDS = 0.2

# Status codes that mean "transient failure, retry".
_RETRY_STATUSES = {429, 500, 502, 503, 504}


class FairTPSError(RuntimeError):
    """Raised when a FAIR-TPs API request fails after retries."""


def _request_json(
    url: str,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    user_agent: str = "untargeted_metabolomics/1.0",
) -> object:
    """Perform a single GET and return parsed JSON, retrying transient errors."""
    last_exc: Optional[BaseException] = None
    for attempt in range(1, retries + 1):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "User-Agent": user_agent,
                    "Accept": "application/json",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = resp.read()
            return json.loads(raw.decode("utf-8"))
        except urllib.error.HTTPError as e:
            last_exc = e
            if e.code in _RETRY_STATUSES and attempt < retries:
                wait = retry_backoff * attempt
                logger.debug(
                    "FAIR-TPs %s returned HTTP %s; retrying in %.1fs (attempt %d/%d)",
                    url, e.code, wait, attempt, retries,
                )
                time.sleep(wait)
                continue
            raise FairTPSError(f"FAIR-TPs request failed (HTTP {e.code}): {url}") from e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last_exc = e
            if attempt < retries:
                wait = retry_backoff * attempt
                logger.debug(
                    "FAIR-TPs %s network error (%s); retrying in %.1fs (attempt %d/%d)",
                    url, e, wait, attempt, retries,
                )
                time.sleep(wait)
                continue
            raise FairTPSError(f"FAIR-TPs request failed (network): {url}: {e}") from e
    # Should be unreachable; retries exhausted without raising.
    raise FairTPSError(f"FAIR-TPs request failed: {url}: {last_exc}")


def _get(
    base_url: str,
    path: str,
    params: Optional[Dict[str, object]] = None,
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> object:
    """Build a URL from path + query params and fetch its JSON."""
    url = base_url.rstrip("/") + path
    if params:
        clean = {k: v for k, v in params.items() if v is not None}
        if clean:
            url = url + "?" + urllib.parse.urlencode(clean, doseq=True)
    return _request_json(url, timeout=timeout, retries=retries, retry_backoff=retry_backoff)


def search_compounds(
    name: str,
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
) -> List[Dict[str, object]]:
    """Search FAIR-TPs compounds by name/title/IUPAC/synonym (case-insensitive).

    Returns the list of CompoundSummary dicts (each with at least `inchikey`
    and optionally `name`/`title`/`iupac_name`). Returns an empty list if the
    API responds with no matches.
    """
    if not name or not str(name).strip():
        return []
    payload = _get(
        base_url, "/compounds",
        params={"q": str(name).strip(), "page_size": DEFAULT_PAGE_SIZE},
        timeout=timeout, retries=retries, retry_backoff=retry_backoff,
    )
    if not isinstance(payload, dict):
        return []
    data = payload.get("data")
    return list(data) if isinstance(data, list) else []


def _paginate_all(
    base_url: str,
    path: str,
    params: Dict[str, object],
    *,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
    max_pages: int = 1000,
) -> List[Dict[str, object]]:
    """Follow pagination and accumulate every `data` entry."""
    out: List[Dict[str, object]] = []
    page = 1
    while page <= max_pages:
        p = dict(params)
        p["page"] = page
        p["page_size"] = DEFAULT_PAGE_SIZE
        payload = _get(base_url, path, p, timeout=timeout, retries=retries, retry_backoff=retry_backoff)
        if not isinstance(payload, dict):
            break
        data = payload.get("data")
        if not isinstance(data, list) or not data:
            break
        out.extend(data)
        meta = payload.get("meta") or {}
        total_pages = meta.get("total_pages")
        count = meta.get("count")
        # Stop when the server says we've reached the last page, or when a
        # page returns nothing, or when the total reported count is reached.
        # We deliberately do NOT stop on "fewer items than requested page
        # size": the last page legitimately returns fewer items than the
        # requested page_size even when more pages remain is impossible, but
        # some servers underfill intermediate pages, so total_pages is the
        # authoritative signal.
        if total_pages is not None and page >= int(total_pages):
            break
        if count is not None and len(out) >= int(count):
            break
        if len(data) < 1:
            break
        page += 1
        if rate_limit:
            time.sleep(rate_limit)
    return out


def list_connections(
    inchikey: str,
    *,
    direction: str = "both",
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
) -> List[Dict[str, object]]:
    """List all transformation reactions involving `inchikey`.

    `direction`: 'both', 'outgoing' (compound is substrate), or 'incoming'.
    """
    if not inchikey:
        return []
    return _paginate_all(
        base_url,
        f"/compounds/{urllib.parse.quote(str(inchikey), safe='')}/connections",
        {"direction": direction},
        timeout=timeout, retries=retries, retry_backoff=retry_backoff,
        rate_limit=rate_limit,
    )


def _endpoint_names(endpoint: Optional[Dict[str, object]]) -> List[str]:
    """Collect name candidates from one connection endpoint."""
    if not isinstance(endpoint, dict):
        return []
    names: List[str] = []
    for key in ("name",):
        v = endpoint.get(key)
        if isinstance(v, str) and v.strip():
            names.append(v.strip())
    return names


def collect_drug_metabolite_names(
    drug_names: Iterable[str],
    *,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
    direction: str = "both",
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Resolve drug parents + all their transformation products from FAIR-TPs.

    For each drug name we (1) search FAIR-TPs compounds to resolve an InChIKey,
    then (2) list every transformation reaction touching that compound. We
    collect the parent's own name(s) plus the names of every connected
    substrate/product (the transformation products/metabolites).

    Returns:
        (names, provenance) where `names` is the set of lowercase compound
        names to drop from the feature matrix, and `provenance` maps each
        collected name -> list of drug parents that produced it (for the
        audit/log CSV).
    """
    names: Set[str] = set()
    provenance: Dict[str, List[str]] = {}

    def _add(candidate: Optional[str], parent: str) -> None:
        if not isinstance(candidate, str):
            return
        c = candidate.strip()
        if not c:
            return
        key = c.lower()
        names.add(key)
        provenance.setdefault(key, [])
        if parent and parent not in provenance[key]:
            provenance[key].append(parent)

    seen_inchikeys: Set[str] = set()

    for drug in drug_names:
        drug = str(drug).strip()
        if not drug:
            continue
        hits = search_compounds(
            drug, base_url=base_url, timeout=timeout, retries=retries,
            retry_backoff=retry_backoff,
        )
        if not hits:
            logger.info("FAIR-TPs: no compound match for drug '%s'", drug)
            continue
        parent_added = False
        for hit in hits:
            inchikey = hit.get("inchikey")
            if not isinstance(inchikey, str) or not inchikey.strip():
                continue
            inchikey = inchikey.strip()
            # Always record the parent's own name(s) under the drug.
            for key in ("name", "title", "iupac_name"):
                _add(hit.get(key), drug)
            parent_added = True
            if inchikey in seen_inchikeys:
                continue
            seen_inchikeys.add(inchikey)
            if rate_limit:
                time.sleep(rate_limit)
            connections = list_connections(
                inchikey, direction=direction, base_url=base_url,
                timeout=timeout, retries=retries, retry_backoff=retry_backoff,
                rate_limit=rate_limit,
            )
            n_conn = len(connections)
            logger.info(
                "FAIR-TPs: drug '%s' -> %s -> %d transformation reactions",
                drug, inchikey, n_conn,
            )
            for conn in connections:
                substrate = conn.get("substrate")
                product = conn.get("product")
                for ep in (substrate, product):
                    for n in _endpoint_names(ep):
                        _add(n, drug)
        if not parent_added:
            logger.info("FAIR-TPs: drug '%s' matched no compound with an InChIKey", drug)

    logger.info(
        "FAIR-TPs: collected %d distinct compound names (parents + TPs) from %d drug queries",
        len(names), sum(1 for d in drug_names if str(d).strip()),
    )
    return names, provenance


# ---------------------------------------------------------------------------
# Offline cache (so repeated runs and tests do not hit the network)
# ---------------------------------------------------------------------------

def _cache_key(drug_names: Iterable[str], base_url: str, direction: str) -> str:
    """Stable hash key for the drug-name list + base URL + direction."""
    norm = "\x1f".join(sorted(str(d).strip().lower() for d in drug_names if str(d).strip()))
    raw = f"{base_url}|{direction}|{norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def load_cache(cache_path: Optional[str]) -> Optional[Dict[str, object]]:
    if not cache_path:
        return None
    try:
        with open(cache_path, "rb") as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return None
    return None


def save_cache(cache_path: Optional[str], payload: Dict[str, object]) -> None:
    if not cache_path:
        return
    try:
        from pathlib import Path
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(payload, f)
    except OSError as e:
        logger.warning("FAIR-TPs: could not write cache %s: %s", cache_path, e)


def build_drug_name_set(
    drug_names: Iterable[str],
    *,
    cache_path: Optional[str] = None,
    base_url: str = DEFAULT_BASE_URL,
    timeout: int = DEFAULT_TIMEOUT,
    retries: int = DEFAULT_RETRIES,
    retry_backoff: float = DEFAULT_RETRY_BACKOFF,
    rate_limit: float = DEFAULT_RATE_LIMIT_SECONDS,
    direction: str = "both",
) -> Tuple[Set[str], Dict[str, List[str]]]:
    """Get the drug+TP name set, using a disk cache if present.

    The cache stores `{names, provenance}` keyed by the drug-name list + URL
    + direction, so a config change (different drug list, different endpoint)
    invalidates it automatically.
    """
    drug_list = [str(d).strip() for d in drug_names if str(d).strip()]
    if not drug_list:
        return set(), {}

    if cache_path:
        cached = load_cache(cache_path)
        if isinstance(cached, dict):
            entries = cached.get("entries")
            if isinstance(entries, dict):
                key = _cache_key(drug_list, base_url, direction)
                hit = entries.get(key)
                if isinstance(hit, dict):
                    names = set(hit.get("names") or [])
                    prov = hit.get("provenance") or {}
                    if isinstance(prov, dict):
                        logger.info(
                            "FAIR-TPs: using cached drug-metabolite set (%d names) from %s",
                            len(names), cache_path,
                        )
                        return names, {k: list(v) for k, v in prov.items()}

    names, provenance = collect_drug_metabolite_names(
        drug_list, base_url=base_url, timeout=timeout, retries=retries,
        retry_backoff=retry_backoff, rate_limit=rate_limit, direction=direction,
    )

    if cache_path:
        cached = load_cache(cache_path) or {"entries": {}}
        if not isinstance(cached, dict):
            cached = {"entries": {}}
        entries = cached.get("entries")
        if not isinstance(entries, dict):
            entries = {}
            cached["entries"] = entries
        key = _cache_key(drug_list, base_url, direction)
        entries[key] = {
            "names": sorted(names),
            "provenance": {k: sorted(v) for k, v in provenance.items()},
        }
        save_cache(cache_path, cached)

    return names, provenance
