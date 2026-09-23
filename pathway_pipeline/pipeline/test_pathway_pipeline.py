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
    load_pathway_names,
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


PATHBANK_CSV = """pathway_id,metabolite_name,metabolite_id,hmdb_id,kegg_id,chebi_id,formula,smiles,iupac_name,inchi_key,species,source,relation,expected_direction,weight,msi_level,plasma_observable,measured
SMP0000055,Adenosine triphosphate,PW_C000414,HMDB0000538,C00002,15422.0,C10H16N5O13P3,SMILES,ATP-IUPAC,KEY,Homo sapiens,pathbank,direct_member,,1.0,1.0,,
SMP0000055,L-Alanine,PW_C000105,HMDB0000161,C00041,16977.0,C3H7NO2,SMILES,Alanine-IUPAC,KEY,Homo sapiens,pathbank,direct_member,,1.0,1.0,,
SMP0000055,Adenosine monophosphate,PW_C000032,HMDB0000045,C00020,16027.0,C10H14N5O7P,SMILES,AMP-IUPAC,KEY,Homo sapiens,pathbank,direct_member,,1.0,1.0,,
SMP0000055,Unmapped metabolite,PW_C000999,HMDB9999999,C99999,,FORMULA,SMILES,IUPAC,KEY,Homo sapiens,pathbank,direct_member,,1.0,1.0,,
SMP0000002,L-Alanine,PW_C000105,HMDB0000161,C00041,16977.0,C3H7NO2,SMILES,Alanine-IUPAC,KEY,Mus musculus,pathbank,direct_member,,1.0,1.0,,
SMP0000004,L-Alanine,PW_C000105,,C00041,16977.0,C3H7NO2,SMILES,Alanine-IUPAC,KEY,Homo sapiens,pathbank,direct_member,,1.0,1.0,,
"""


def _write_pathbank_csv(tmp_path):
    p = tmp_path / "pathbank_all_metabolites.csv"
    p.write_text(PATHBANK_CSV, encoding="utf-8")
    return str(p)


PATHWAY_NAMES_CSV = """pathway_id,pathbank_id,smpdb_id,name,subject,description,category,species,tier,curation_status,curation_note
SMP0000055,PW000001,SMP0000055,Alanine Metabolism,Metabolic,"A description.",Metabolic,Homo sapiens,stouffer,auto-included,
SMP0000002,PW000002,SMP0000002,Mouse Pathway,Metabolic,"A description.",Metabolic,Mus musculus,stouffer,auto-included,
"""


def _write_pathway_names_csv(tmp_path):
    p = tmp_path / "pathbank_pathways.csv"
    p.write_text(PATHWAY_NAMES_CSV, encoding="utf-8")
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

def test_load_pathbank_pathways_filters_species(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path)
    # Only the Homo sapiens rows with an HMDB ID survive; without a names file
    # pathway_name falls back to the smp_id.
    assert set(pathways["smp_id"]) == {"SMP0000055"}
    assert len(pathways) == 4
    assert set(pathways["hmdb_id"]) == {
        "HMDB0000538", "HMDB0000161", "HMDB0000045", "HMDB9999999"
    }
    assert (pathways["pathway_name"] == pathways["smp_id"]).all()


def test_load_pathbank_pathways_with_names_file(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    names_path = _write_pathway_names_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path, pathway_names_file=names_path)
    # Every SMP0000055 row is named 'Alanine Metabolism'.
    assert (pathways["pathway_name"] == "Alanine Metabolism").all()

    # Switching species picks up the mouse pathway and its name.
    pathways = load_pathbank_pathways(csv_path, species="Mus musculus",
                                      pathway_names_file=names_path)
    assert set(pathways["pathway_name"]) == {"Mouse Pathway"}


def test_load_pathway_names_fallbacks(tmp_path):
    # Missing file -> empty mapping (callers fall back to SMP IDs).
    assert load_pathway_names(str(tmp_path / "missing.csv")) == {}

    # Unrecognizable columns -> empty mapping, no crash.
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n", encoding="utf-8")
    assert load_pathway_names(str(bad)) == {}

    # Names file that lacks one pathway: that pathway keeps its SMP ID
    # (checked via load_pathbank_pathways warning path).
    partial = tmp_path / "partial.csv"
    partial.write_text("pathway_id,name\nSMP0000055,Alanine Metabolism\n",
                       encoding="utf-8")
    names = load_pathway_names(str(partial))
    assert names == {"SMP0000055": "Alanine Metabolism"}


def test_load_pathbank_pathways_missing_file(tmp_path):
    pathways = load_pathbank_pathways(str(tmp_path / "missing.csv"))
    assert pathways.empty


def test_load_pathbank_pathways_selectable_species(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    # Switching the species picks up the mouse row.
    pathways = load_pathbank_pathways(csv_path, species="Mus musculus")
    assert set(pathways["smp_id"]) == {"SMP0000002"}
    assert set(pathways["hmdb_id"]) == {"HMDB0000161"}

    # Unknown species -> empty with an error logged (fail fast).
    pathways = load_pathbank_pathways(csv_path, species="Arabidopsis thaliana")
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
    assert row["smp_id"] == "SMP0000055"
    assert row["pathway_name"] == "SMP0000055"
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


def test_coverage_with_pathway_names(tmp_path):
    csv_path = _write_pathbank_csv(tmp_path)
    names_path = _write_pathway_names_csv(tmp_path)
    pathways = load_pathbank_pathways(csv_path, pathway_names_file=names_path)

    feature_to_hmdb = pd.DataFrame([
        {"feature": "Alanine", "hmdb_id": "HMDB0000161",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "ATP", "hmdb_id": "HMDB0000538",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "AMP", "hmdb_id": "HMDB0000045",
         "match_method": "name_exact", "n_hmdb_ids": 1},
    ])
    links = link_features_to_pathways(feature_to_hmdb, pathways)
    coverage = pathway_coverage(links, pathways, min_coverage=0.20)
    assert coverage.iloc[0]["pathway_name"] == "Alanine Metabolism"


def test_coverage_empty_inputs(tmp_path):
    empty_links = pd.DataFrame(columns=["feature", "hmdb_id", "smp_id",
                                       "pathway_name", "metabolite_id",
                                       "metabolite_name"])
    pathways = load_pathbank_pathways(str(tmp_path / "missing.csv"))
    coverage = pathway_coverage(empty_links, pathways, min_coverage=0.20)
    assert coverage.empty
