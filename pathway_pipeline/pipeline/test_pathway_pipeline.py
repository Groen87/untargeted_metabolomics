#!/usr/bin/env python3
"""Unit tests for the pathway pipeline feature engineering (parser, mapping).

All inputs are synthetic and written to temp files, so the tests run without
the real HMDB XML / PathBank CSV / feature matrix.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.name_utils import normalize_name, normalize_loose
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathbank_pathways,
    match_features_to_hmdb,
    link_features_to_pathways,
    pathway_coverage,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic HMDB XML, PathBank CSV, feature columns
# ---------------------------------------------------------------------------

HMDB_XML = """<?xml version="1.0"?>
<hmdb>
  <metabolite>
    <accession>HMDB0000161</accession>
    <name>L-Alanine</name>
    <synonyms>
      <synonym>Alanine</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000538</accession>
    <name>Adenosine triphosphate</name>
    <synonyms>
      <synonym>ATP</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000045</accession>
    <name>Adenosine monophosphate</name>
    <synonyms>
      <synonym>AMP</synonym>
    </synonyms>
  </metabolite>
</hmdb>
"""


def _write_hmdb_xml(tmp_path):
    p = tmp_path / "hmdb_metabolites.xml"
    p.write_text(HMDB_XML, encoding="utf-8")
    return str(p)


PATHBANK_CSV = """PathBank ID,Pathway Name,Pathway Subject,Species,Metabolite ID,Metabolite Name,HMDB ID,KEGG ID
SMP0000055,Alanine Metabolism,Metabolic,Homo sapiens,PW_C000414,Adenosine triphosphate,HMDB0000538,C00002
SMP0000055,Alanine Metabolism,Metabolic,Homo sapiens,PW_C000105,L-Alanine,HMDB0000161,C00041
SMP0000055,Alanine Metabolism,Metabolic,Homo sapiens,PW_C000032,Adenosine monophosphate,HMDB0000045,C00020
SMP0000055,Alanine Metabolism,Metabolic,Homo sapiens,PW_C000999,Unmapped metabolite,HMDB9999999,C99999
SMP0000001,Protein Synthesis,Genetic,Homo sapiens,PW_C000105,L-Alanine,HMDB0000161,C00041
SMP0000002,Mouse Pathway,Metabolic,Mus musculus,PW_C000105,L-Alanine,HMDB0000161,C00041
SMP0000003,Drug Pathway,Drug,Human,PW_C000105,L-Alanine,HMDB0000161,C00041
SMP0000004,No HMDB,Metabolic,Homo sapiens,PW_C000105,L-Alanine,,C00041
"""


def _write_pathbank_csv(tmp_path):
    p = tmp_path / "pathbank_all_metabolites.csv"
    p.write_text(PATHBANK_CSV, encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Name normalization
# ---------------------------------------------------------------------------

def test_normalize_name_folds_greek_and_case():
    assert normalize_name("ps(18:2w6/24:1w9)") == "PS(18:2OMEGA6/24:1OMEGA9)"
    assert normalize_name("omega-3") == "OMEGA-3"


def test_normalize_loose_strips_punctuation():
    assert normalize_loose("Coproporphyrin-III") == "COPROPORPHYRINIII"
    assert normalize_loose("Coproporphyrin III") == "COPROPORPHYRINIII"


# ---------------------------------------------------------------------------
# HMDB parser
# ---------------------------------------------------------------------------

def test_build_name_index_names_and_accessions(tmp_path):
    xml = _write_hmdb_xml(tmp_path)
    index = build_name_index(xml, use_cache=False)
    assert index["L-ALANINE"] == {"HMDB0000161"}
    assert index["ALANINE"] == {"HMDB0000161"}
    assert index["HMDB0000161"] == {"HMDB0000161"}
    assert index["ATP"] == {"HMDB0000538"}


# ---------------------------------------------------------------------------
# PathBank loading
# ---------------------------------------------------------------------------

def test_load_pathbank_pathways_filters_species_and_subject(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path)
    # Only the Metabolic Homo sapiens rows with an HMDB ID survive.
    assert set(pathways["smp_id"]) == {"SMP0000055"}
    assert len(pathways) == 4
    assert set(pathways["hmdb_id"]) == {
        "HMDB0000538", "HMDB0000161", "HMDB0000045", "HMDB9999999"
    }


def test_load_pathbank_pathways_missing_file(tmp_path):
    pathways = load_pathbank_pathways(str(tmp_path / "missing.csv"))
    assert pathways.empty


def test_load_pathbank_pathways_selectable_filters(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    # Only Metabolic subject, no Disease rows in the fixture, but the mouse
    # Metabolic row must appear when the species is switched.
    pathways = load_pathbank_pathways(csv_path, species="Mus musculus",
                                      pathway_subjects="Metabolic")
    assert set(pathways["smp_id"]) == {"SMP0000002"}
    assert set(pathways["hmdb_id"]) == {"HMDB0000161"}

    # Single-string subject behaves the same as a one-element list; the mouse
    # pathway stays excluded by the (default) species filter.
    pathways = load_pathbank_pathways(csv_path,
                                      pathway_subjects=["Metabolic"])
    assert set(pathways["smp_id"]) == {"SMP0000055"}

    # Unknown species -> empty with an error logged (fail fast).
    pathways = load_pathbank_pathways(csv_path, species="Arabidopsis thaliana")
    assert pathways.empty

    # Unknown subject -> empty with an error logged (fail fast).
    pathways = load_pathbank_pathways(csv_path, pathway_subjects=["Signaling"])
    assert pathways.empty


# ---------------------------------------------------------------------------
# Feature -> HMDB matching
# ---------------------------------------------------------------------------

def test_match_features_to_hmdb_chain(tmp_path):
    xml = _write_hmdb_xml(tmp_path)
    index = build_name_index(xml, use_cache=False)
    features = [
        "Cortisol.HMDB0000161",   # HMDB tag (authoritative, ignores the name)
        "HMDB0000538",            # bare accession
        "L-Alanine",              # exact name
        "l alanine",              # loose-ish
        "Unknown compound",       # unmatched
    ]
    out = match_features_to_hmdb(features, index)
    by_feature = {f: g for f, g in out.groupby("feature")}

    assert set(by_feature["Cortisol.HMDB0000161"]["match_method"]) == {"hmdb_tag"}
    assert set(by_feature["Cortisol.HMDB0000161"]["hmdb_id"]) == {"HMDB0000161"}
    assert set(by_feature["HMDB0000538"]["match_method"]) == {"hmdb_tag"}
    assert set(by_feature["HMDB0000538"]["hmdb_id"]) == {"HMDB0000538"}
    assert set(by_feature["L-Alanine"]["match_method"]) == {"name_exact"}
    assert set(by_feature["L-Alanine"]["hmdb_id"]) == {"HMDB0000161"}
    # 'l alanine' loose-matches via non-alphanumeric stripping.
    assert "name_loose" in set(out["match_method"])
    unmatched = by_feature["Unknown compound"]
    assert len(unmatched) == 1
    assert unmatched.iloc[0]["match_method"] == "unmatched"
    assert pd.isna(unmatched.iloc[0]["hmdb_id"])


# ---------------------------------------------------------------------------
# Linking and coverage
# ---------------------------------------------------------------------------

def test_link_features_and_coverage_min_20_percent(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path)

    feature_to_hmdb = pd.DataFrame([
        {"feature": "Alanine", "hmdb_id": "HMDB0000161",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "ATP", "hmdb_id": "HMDB0000538",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "AMP", "hmdb_id": "HMDB0000045",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "Unknown compound", "hmdb_id": None,
         "match_method": "unmatched", "n_hmdb_ids": 0},
    ])

    links = link_features_to_pathways(feature_to_hmdb, pathways)
    # All three matched accessions are in Alanine Metabolism (4 metabolites).
    assert set(links["smp_id"]) == {"SMP0000055"}
    assert len(links) == 3

    coverage = pathway_coverage(links, pathways, min_coverage=0.20)
    # 3 of 4 pathway metabolites mapped -> 75% coverage, kept.
    assert len(coverage) == 1
    row = coverage.iloc[0]
    assert row["pathway_name"] == "Alanine Metabolism"
    assert row["n_metabolites"] == 4
    assert row["n_matched_metabolites"] == 3
    assert row["coverage"] == pytest.approx(0.75)


def test_coverage_drops_pathways_under_threshold(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path)

    # Only ONE of the pathway's 4 metabolites is mapped -> 25% coverage.
    feature_to_hmdb = pd.DataFrame([
        {"feature": "Alanine", "hmdb_id": "HMDB0000161",
         "match_method": "name_exact", "n_hmdb_ids": 1},
    ])
    links = link_features_to_pathways(feature_to_hmdb, pathways)

    coverage_25 = pathway_coverage(links, pathways, min_coverage=0.20)
    assert len(coverage_25) == 1  # 25% >= 20% -> kept

    coverage_50 = pathway_coverage(links, pathways, min_coverage=0.50)
    assert coverage_50.empty     # 25% < 50% -> dropped


def test_coverage_empty_inputs(tmp_path):
    empty_links = pd.DataFrame(columns=["feature", "hmdb_id", "smp_id",
                                       "pathway_name", "metabolite_id",
                                       "metabolite_name"])
    pathways = load_pathbank_pathways(str(tmp_path / "missing.csv"))
    coverage = pathway_coverage(empty_links, pathways, min_coverage=0.20)
    assert coverage.empty
