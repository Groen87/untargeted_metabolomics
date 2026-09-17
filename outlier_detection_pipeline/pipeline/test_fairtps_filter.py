#!/usr/bin/env python3
"""Unit tests for the FAIR-TPs drug-metabolite filter (offline, mocked HTTP).

Covers the fairtps_client module and the data_loader integration filter.
All network access is monkeypatched away so the tests run in the sandbox.
"""
import io
import json
import os
import sys
import pickle
from contextlib import contextmanager
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outlier_detection_pipeline.pipeline import fairtps_client
from outlier_detection_pipeline.pipeline import data_loader


# ---------------------------------------------------------------------------
# Helpers: a fake urlopen that serves canned responses by URL pattern.
# ---------------------------------------------------------------------------

class _FakeResp:
    def __init__(self, payload: bytes):
        self._buf = io.BytesIO(payload)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self):
        return self._buf.getvalue()


def _build_urlopen(router):
    def _urlopen(req, timeout=None):
        url = req.full_url
        handler = router(url)
        if handler is None:
            raise AssertionError(f"unexpected request to {url}")
        status, body = handler
        if status >= 400:
            import urllib.error
            raise urllib.error.HTTPError(url, status, b"err", {}, io.BytesIO(body))
        return _FakeResp(body)
    return _urlopen


# ---------------------------------------------------------------------------
# fairtps_client unit tests
# ---------------------------------------------------------------------------

def test_search_compounds_parses_data(monkeypatch):
    payload = {
        "meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
        "data": [{"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine"}],
    }

    def router(url):
        assert "/api/v1/compounds" in url and "q=Caffeine" in url
        return 200, json.dumps(payload).encode()

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    hits = fairtps_client.search_compounds("Caffeine")
    assert hits == payload["data"]


def test_search_compounds_empty_name_returns_empty():
    assert fairtps_client.search_compounds("") == []
    assert fairtps_client.search_compounds("   ") == []


def test_list_connections_paginates(monkeypatch):
    page1 = {
        "meta": {"count": 3, "page": 1, "page_size": 2, "total_pages": 2},
        "data": [{"direction": "outgoing", "substrate": {"inchikey": "A"}, "product": {"inchikey": "B", "name": "TP1"}}],
    }
    page2 = {
        "meta": {"count": 3, "page": 2, "page_size": 2, "total_pages": 2},
        "data": [{"direction": "outgoing", "substrate": {"inchikey": "A"}, "product": {"inchikey": "C", "name": "TP2"}}],
    }

    seen = {"p": 0}

    def router(url):
        seen["p"] += 1
        assert "/connections" in url
        # First request has no page param or page=1 -> page1; page=2 -> page2.
        if "page=2" in url:
            return 200, json.dumps(page2).encode()
        return 200, json.dumps(page1).encode()

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    monkeypatch.setattr(fairtps_client.time, "sleep", lambda s: None)
    conns = fairtps_client.list_connections("RYYVLZVUVIJVGH-UHFFFAOYSA-N", rate_limit=0)
    assert len(conns) == 2
    assert seen["p"] == 2


def test_collect_drug_metabolite_names_parents_and_tps(monkeypatch):
    search_resp = {
        "meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
        "data": [
            {"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine", "title": None, "iupac_name": "1,3,7-trimethylxanthine"},
        ],
    }
    conns_resp = {
        "meta": {"count": 2, "page": 1, "page_size": 100, "total_pages": 1},
        "data": [
            {"direction": "outgoing",
             "substrate": {"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine"},
             "product": {"inchikey": "B", "name": "Paraxanthine"}},
            {"direction": "outgoing",
             "substrate": {"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine"},
             "product": {"inchikey": "C", "name": "Theobromine"}},
        ],
    }

    def router(url):
        if "/compounds?" in url:
            return 200, json.dumps(search_resp).encode()
        if "/connections" in url:
            return 200, json.dumps(conns_resp).encode()
        return None

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    names, prov = fairtps_client.collect_drug_metabolite_names(["Caffeine"], rate_limit=0)
    assert {"caffeine", "paraxanthine", "theobromine"} <= names
    assert "caffeine" in prov and "Caffeine" in prov["caffeine"]
    assert "paraxanthine" in prov and "Caffeine" in prov["paraxanthine"]


def test_collect_returns_empty_when_no_match(monkeypatch):
    def router(url):
        return 200, json.dumps({"meta": {"count": 0, "page": 1, "page_size": 100, "total_pages": 0}, "data": []}).encode()
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    names, prov = fairtps_client.collect_drug_metabolite_names(["NonexistentDrug"], rate_limit=0)
    assert names == set()
    assert prov == {}


def test_permanent_http_error_raises(monkeypatch):
    import urllib.error

    def _urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 404, b"nf", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _urlopen)
    with pytest.raises(fairtps_client.FairTPSError):
        fairtps_client.search_compounds("X", retries=2, retry_backoff=0)


def test_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    def _urlopen(req, timeout=None):
        calls["n"] += 1
        if calls["n"] < 2:
            import urllib.error
            raise urllib.error.HTTPError(req.full_url, 503, b"x", {}, io.BytesIO(b"{}"))
        return _FakeResp(json.dumps({"meta": {}, "data": []}).encode())

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(fairtps_client.time, "sleep", lambda s: None)
    hits = fairtps_client.search_compounds("X", retries=3, retry_backoff=0)
    assert hits == []
    assert calls["n"] == 2


def test_cache_round_trip(tmp_path, monkeypatch):
    cache = tmp_path / "ftp_cache.pkl"
    # Seed a fetch that returns a known name set.
    def router(url):
        if "/compounds?" in url:
            return 200, json.dumps({"meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
                                    "data": [{"inchikey": "K", "name": "Caffeine"}]}).encode()
        if "/connections" in url:
            return 200, json.dumps({"meta": {"count": 0, "page": 1, "page_size": 100, "total_pages": 0}, "data": []}).encode()
        return None
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    names1, _ = fairtps_client.build_drug_name_set(["Caffeine"], cache_path=str(cache), rate_limit=0)
    assert "caffeine" in names1
    assert cache.exists()

    # Second call should hit cache, not the network.
    def router_fail(url):
        raise AssertionError("network should not be hit when cache is present")
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router_fail))
    names2, _ = fairtps_client.build_drug_name_set(["Caffeine"], cache_path=str(cache), rate_limit=0)
    assert names2 == names1


def test_cache_invalidated_by_different_drug_list(tmp_path, monkeypatch):
    cache = tmp_path / "ftp_cache.pkl"
    def router(url):
        if "/compounds?" in url:
            return 200, json.dumps({"meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
                                    "data": [{"inchikey": "K2", "name": "Diclofenac"}]}).encode()
        if "/connections" in url:
            return 200, json.dumps({"meta": {"count": 0, "page": 1, "page_size": 100, "total_pages": 0}, "data": []}).encode()
        return None
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))
    # Different drug list -> different cache key -> must re-fetch.
    names, _ = fairtps_client.build_drug_name_set(["Diclofenac"], cache_path=str(cache), rate_limit=0)
    assert "diclofenac" in names


# ---------------------------------------------------------------------------
# data_loader integration
# ---------------------------------------------------------------------------

def _make_input_csv(path: Path):
    df = pd.DataFrame({
        "Caffeine": [1.0, 2.0, 3.0],
        "Paraxanthine": [4.0, 5.0, 6.0],
        "Glucose": [7.0, 8.0, 9.0],
        "Oordeel targeted": [0, 0, 1],
        "Classification": [0, 0, 1],
    }, index=["P1", "P2", "P3"])
    df.index.name = "Sample"
    df.to_csv(path)
    return path


def test_load_data_drops_fairtps_features(monkeypatch, tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")

    def router(url):
        if "/compounds?" in url:
            return 200, json.dumps({"meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
                                    "data": [{"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine"}]}).encode()
        if "/connections" in url:
            return 200, json.dumps({"meta": {"count": 1, "page": 1, "page_size": 100, "total_pages": 1},
                                    "data": [{"direction": "outgoing",
                                              "substrate": {"inchikey": "RYYVLZVUVIJVGH-UHFFFAOYSA-N", "name": "Caffeine"},
                                              "product": {"inchikey": "B", "name": "Paraxanthine"}}]}).encode()
        return None
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router))

    features, classification, oordeel, raw = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        fairtps_drug_metabolites=["Caffeine"],
        fairtps_config={"rate_limit": 0, "cache_path": str(tmp_path / "c.pkl")},
        output_dir=str(tmp_path),
    )
    assert "Caffeine" not in features.columns
    assert "Paraxanthine" not in features.columns
    assert "Glucose" in features.columns
    assert (tmp_path / "fairtps_drug_metabolites.csv").exists()


def test_load_data_no_filter_when_disabled(monkeypatch, tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")

    def router_fail(url):
        raise AssertionError("network should not be hit when filter is disabled")
    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _build_urlopen(router_fail))

    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        fairtps_drug_metabolites=None,
    )
    assert set(features.columns) == {"Caffeine", "Paraxanthine", "Glucose"}


def test_load_data_api_failure_keeps_features(monkeypatch, tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")

    import urllib.error

    def _urlopen(req, timeout=None):
        raise urllib.error.HTTPError(req.full_url, 500, b"x", {}, io.BytesIO(b"{}"))

    monkeypatch.setattr(fairtps_client.urllib.request, "urlopen", _urlopen)
    monkeypatch.setattr(fairtps_client.time, "sleep", lambda s: None)

    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        fairtps_drug_metabolites=["Caffeine"],
        fairtps_config={"retries": 2, "retry_backoff": 0, "rate_limit": 0},
    )
    # Filter disabled gracefully; all features kept.
    assert set(features.columns) == {"Caffeine", "Paraxanthine", "Glucose"}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
