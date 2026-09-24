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
    filter_pathways_by_keywords,
    prefer_tagged_features,
    ambiguous_feature_report,
    ambiguous_feature_impact,
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


def test_ambiguous_feature_report():
    """Multi-matched features are reported with all their accessions,
    unambiguous ones are not, and excluded (already-demoted) ones are
    omitted because their ambiguity has been resolved by decision."""
    feature_to_hmdb = pd.DataFrame([
        {"feature": "Dup", "hmdb_id": "HMDB0001227",
         "match_method": "name_exact", "n_hmdb_ids": 2},
        {"feature": "Dup", "hmdb_id": "HMDB0002666",
         "match_method": "name_exact", "n_hmdb_ids": 2},
        {"feature": "Settled", "hmdb_id": "HMDB0000045",
         "match_method": "name_exact", "n_hmdb_ids": 2},
        {"feature": "Settled", "hmdb_id": "HMDB0000538",
         "match_method": "name_exact", "n_hmdb_ids": 2},
        {"feature": "L-Alanine", "hmdb_id": "HMDB0000161",
         "match_method": "name_exact", "n_hmdb_ids": 1},
    ])
    report = ambiguous_feature_report(feature_to_hmdb)
    assert list(report.index) == ["Dup", "Settled"]
    assert report["Dup"] == "HMDB0001227,HMDB0002666"
    report = ambiguous_feature_report(
        feature_to_hmdb, exclude=["Settled"])
    assert list(report.index) == ["Dup"]


def test_ambiguous_feature_impact_sorts_by_scoring_reach():
    """Impact triage counts scored pathways and disease tests per
    ambiguous feature and sorts the actionable ones first."""
    ambiguous = pd.Series({
        "Dup": "HMDB0001227,HMDB0002666",
        "Idle": "HMDB0001111,HMDB0002222",
    })
    feature_to_pathway = pd.DataFrame([
        {"feature": "Dup", "smp_id": "SMP0000055"},
        {"feature": "Dup", "smp_id": "SMP0000002"},
        {"feature": "Idle", "smp_id": "SMP0000099"},
    ])
    scored_coverage = pd.DataFrame([{"smp_id": "SMP0000055"}])
    disease_resolved = pd.DataFrame([
        {"disease": "Deficiency A", "features": "Dup"},
        {"disease": "Deficiency B", "features": "Other"},
    ])
    impact = ambiguous_feature_impact(
        ambiguous, feature_to_pathway, scored_coverage,
        disease_resolved=disease_resolved)
    assert list(impact["feature"]) == ["Dup", "Idle"]
    dup = impact[impact["feature"] == "Dup"].iloc[0]
    assert dup["n_scored_pathways"] == 1
    assert dup["n_disease_tests"] == 1
    idle = impact[impact["feature"] == "Idle"].iloc[0]
    assert idle["n_scored_pathways"] == 0
    assert idle["n_disease_tests"] == 0
    empty = ambiguous_feature_impact(
        pd.Series(dtype="object"), feature_to_pathway, scored_coverage)
    assert empty.empty


def test_run_pipeline_warns_on_ambiguous_feature_names(tmp_path, caplog):
    """run_pipeline surfaces multi-matched feature names after identity
    curation: demoted and tagged-twin-superseded ambiguities stay silent,
    only unresolved multi-matches warn and land in the CSV artifact."""
    import logging
    import numpy as np
    import yaml
    from pathway_pipeline.main import run_pipeline

    xml = tmp_path / "hmdb_ambiguous.xml"
    xml.write_text("""<?xml version="1.0"?>
<hmdb>
  <metabolite>
    <accession>HMDB0001227</accession>
    <name>Ambigo</name>
  </metabolite>
  <metabolite>
    <accession>HMDB0002666</accession>
    <name>Other name</name>
    <synonyms>
      <synonym>Ambigo</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0001111</accession>
    <name>Settled</name>
  </metabolite>
  <metabolite>
    <accession>HMDB0002222</accession>
    <name>Third name</name>
    <synonyms>
      <synonym>Settled</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0003333</accession>
    <name>Curated</name>
  </metabolite>
  <metabolite>
    <accession>HMDB0004444</accession>
    <name>Fourth name</name>
    <synonyms>
      <synonym>Curated</synonym>
    </synonyms>
  </metabolite>
</hmdb>
""", encoding="utf-8")
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)
    rng = np.random.default_rng(3)
    n = 20
    data = {
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n,
        "Oordeel targeted": [0] * n,
        "Ambigo": list(rng.normal(2.0, 0.3, size=n)),
        "Settled": list(rng.normal(2.0, 0.3, size=n)),
        "Curated": list(rng.normal(2.0, 0.3, size=n)),
        "Curated.HMDB0003333": list(rng.normal(2.0, 0.3, size=n)),
        "Curated.HMDB0004444": list(rng.normal(2.0, 0.3, size=n)),
    }
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": str(xml),
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_features": 1,
        "min_stouffer_metabolites": 1,
        "demoted_features": ["Settled"],
        "save_mapping_outputs": True,
        "save_zscore_outputs": False,
        "save_stouffer_outputs": False,
        "save_flagging_outputs": False,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    with caplog.at_level(logging.WARNING, logger="pathway_pipeline.main"):
        run_pipeline(
            input_file=str(input_csv),
            output_dir=str(tmp_path / "out"),
            config_path=str(config_path),
        )
    warnings = [r for r in caplog.records
                if "match multiple HMDB accessions" in r.getMessage()]
    assert len(warnings) == 1
    msg = warnings[0].getMessage()
    assert "Ambigo -> HMDB0001227,HMDB0002666" in msg
    assert "Settled" not in msg
    # Superseded by tagged twins at identity curation -> never warned.
    assert "Curated" not in msg
    # The CSV artifact lists exactly the unresolved ambiguities.
    artifact = pd.read_csv(tmp_path / "out" / "ambiguous_features.csv")
    assert list(artifact["feature"]) == ["Ambigo"]
    assert artifact["hmdb_ids"].iloc[0] == "HMDB0001227,HMDB0002666"


def test_match_features_to_hmdb_override_beats_tag():
    """A manual override wins over the HMDB tag and fixes collisions."""
    index = {}  # even an empty index: the override never consults it
    out = match_features_to_hmdb(
        ["(+)-Estrone", "Niacin", "L-Alanine"],
        index,
        overrides={"(+)-Estrone": "HMDB0000145", "Niacin": "HMDB0001488"},
    )
    estrone = out[out["feature"] == "(+)-Estrone"].iloc[0]
    assert estrone["hmdb_id"] == "HMDB0000145"
    assert estrone["match_method"] == "override"
    niacin = out[out["feature"] == "Niacin"].iloc[0]
    assert niacin["hmdb_id"] == "HMDB0001488"
    assert niacin["match_method"] == "override"
    # Features without an override still go through the normal chain.
    alanine = out[out["feature"] == "L-Alanine"].iloc[0]
    assert alanine["match_method"] == "unmatched"


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


def test_filter_pathways_by_keywords():
    coverage = pd.DataFrame({
        "smp_id": ["SMP_A", "SMP_B", "SMP_C", "SMP_D"],
        "pathway_name": ["Sphingolipid Metabolism", "Krabbe Disease",
                         "Bile acid biosynthesis", "Urea Cycle"],
        "coverage": [0.5, 0.6, 0.4, 0.9],
    })
    kept = filter_pathways_by_keywords(coverage, ["lipid", "bile acid"])
    assert set(kept["smp_id"]) == {"SMP_B", "SMP_D"}
    # Case-insensitive substring matching; no keywords -> no-op.
    kept_none = filter_pathways_by_keywords(coverage, [])
    assert len(kept_none) == 4
    kept_one = filter_pathways_by_keywords(coverage, "urea")
    assert set(kept_one["smp_id"]) == {"SMP_A", "SMP_B", "SMP_C"}


# ---------------------------------------------------------------------------
# End-to-end wiring of the scale floor, overrides, demotion, and weights
# ---------------------------------------------------------------------------

def test_run_pipeline_wires_floor_overrides_demotion_and_weights(tmp_path):
    """Smoke test: run_pipeline honors min_reference_scale,
    feature_hmdb_overrides, demoted_features, and scale weighting."""
    import numpy as np
    import yaml

    from pathway_pipeline.main import run_pipeline

    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)

    rng = np.random.default_rng(11)
    n_normal, n_other = 30, 6
    n = n_normal + n_other
    # Feature matrix: 'Alanine' (wide), 'AMP' (thin -> scale floor drops it),
    # 'ATP' (wide), plus a demoted artifact feature and an overridden one.
    data = {
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * n_other,
        "Oordeel targeted": [0] * n_normal + [1] * n_other,
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
        # Razor-thin normal spread -> small_scale drop.
        "AMP": list(rng.normal(2.0, 0.01, size=n)),
        # Pathway-mapped artifact feature (tagged to SMP0000055's
        # HMDB9999999 row), demoted by config: z-scored but never scored.
        "ARTIFACT.HMDB9999999": list(rng.normal(5.0, 0.3, size=n - 1)) + [9.0],
        # Overridden feature: would otherwise be unmatched in this tiny
        # index; the override pins it to L-Alanine's accession.
        "(+)-Estrone": list(rng.normal(1.0, 0.3, size=n)),
    }
    df = pd.DataFrame(data)
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)

    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_coverage": 0.10,
        "min_reference_scale": 0.08,
        "feature_hmdb_overrides": {"(+)-Estrone": "HMDB0000161"},
        "demoted_features": ["ARTIFACT.HMDB9999999"],
        "min_pathway_features": 2,
        "scale_weighted_metabolites": True,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.5,
        "run_metabolite_flags": True,
        "save_mapping_outputs": True,
        "save_zscore_outputs": True,
        "save_stouffer_outputs": True,
        "save_flagging_outputs": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)

    result = run_pipeline(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
        config_path=str(config_path),
    )

    zscores = result["zscores"]
    # The thin AMP feature is dropped by the scale floor.
    assert "AMP" not in set(zscores.columns)
    dropped = result["dropped_features"]
    assert "AMP" in set(dropped.loc[dropped["reason"] == "small_scale", "feature"])

    # The override redirected (+)-Estrone away from name matching.
    f2h = result["feature_to_hmdb"]
    estrone = f2h[f2h["feature"] == "(+)-Estrone"].iloc[0]
    assert estrone["hmdb_id"] == "HMDB0000161"
    assert estrone["match_method"] == "override"

    # Demoted feature keeps its z-scores but never reaches the scores.
    assert "ARTIFACT.HMDB9999999" in set(zscores.columns)
    scores = result["pathway_scores"]
    scored_features = set(
        result["pathway_coverage_scored"]["matched_features"]
        .str.split(";").explode().dropna())
    assert "ARTIFACT.HMDB9999999" not in scored_features
    # The demoted feature never contributes a metabolite flag either.
    mflags = pd.read_csv(tmp_path / "out" / "metabolite_flags.csv")
    assert "ARTIFACT.HMDB9999999" not in set(mflags["metabolite"])

    # SMP0000055 lists ATP, L-Alanine, AMP; after the floor drops thin AMP
    # and demotion removes the artifact twin, ATP + L-Alanine remain.
    assert not result["pathway_coverage_scored"].empty
    assert not scores.empty
    metabolite_flags = result.get("sample_decisions")
    assert metabolite_flags is not None


def test_run_pipeline_keyword_exclusion_keeps_shared_features(tmp_path):
    """Chemistry curation drops lipid pathways but keeps their shared features.

    SMP0000055 (valid) shares ATP and L-Alanine with the lipid pathway
    SMP0000056; the exclusively-lipid feature DG (only in SMP0000056)
    disappears entirely, the shared ones keep their scores.
    """
    import numpy as np
    import yaml
    from pathway_pipeline.main import run_pipeline

    xml = _write_hmdb_xml(tmp_path)
    pathbank = tmp_path / "pathbank_all_metabolites.csv"
    pathbank.write_text(
        "pathway_id,metabolite_name,metabolite_id,hmdb_id,species,source\n"
        "SMP0000055,Adenosine triphosphate,PW_C000414,HMDB0000538,Homo sapiens,pathbank\n"
        "SMP0000055,L-Alanine,PW_C000105,HMDB0000161,Homo sapiens,pathbank\n"
        "SMP0000055,Adenosine monophosphate,PW_C000032,HMDB0000045,Homo sapiens,pathbank\n"
        "SMP0000055,Unmapped metabolite,PW_C000999,HMDB9999999,Homo sapiens,pathbank\n"
        "SMP0000056,Adenosine triphosphate,PW_C000414,HMDB0000538,Homo sapiens,pathbank\n"
        "SMP0000056,L-Alanine,PW_C000105,HMDB0000161,Homo sapiens,pathbank\n"
        "SMP0000056,DG(16:1(9Z)/22:0/0:0),PW_C000888,HMDB0007777,Homo sapiens,pathbank\n",
        encoding="utf-8")
    names = tmp_path / "pathbank_pathways.csv"
    names.write_text(
        "pathway_id,pathbank_id,smpdb_id,name,subject,description,category,species\n"
        "SMP0000055,PW000001,SMP0000055,Alanine Metabolism,Metabolic,d,Metabolic,Homo sapiens\n"
        "SMP0000056,PW000003,SMP0000056,Sphingolipid Metabolism,Metabolic,d,Metabolic,Homo sapiens\n",
        encoding="utf-8")

    rng = np.random.default_rng(3)
    n_normal, n_other = 30, 6
    n = n_normal + n_other
    data = {
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * n_other,
        "Oordeel targeted": [0] * n_normal + [1] * n_other,
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "AMP": list(rng.normal(2.0, 0.30, size=n)),
        "DG(16:1(9Z)/22:0/0:0)": list(rng.normal(2.0, 0.30, size=n)),
    }
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)

    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": str(pathbank),
        "pathbank_pathway_names_file": str(names),
        "min_pathway_coverage": 0.10,
        "min_pathway_features": 2,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.5,
        "exclude_pathway_keywords": ["lipid"],
        "save_mapping_outputs": True,
        "save_zscore_outputs": True,
        "save_stouffer_outputs": True,
        "save_flagging_outputs": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    result = run_pipeline(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
        config_path=str(config_path),
    )
    scored = result["pathway_coverage_scored"]
    assert set(scored["smp_id"]) == {"SMP0000055"}
    # Shared features survive: ATP and L-Alanine are still z-scored and
    # scored through the kept pathway.
    assert "ATP" in set(result["zscores"].columns)
    scored_features = set(scored["matched_features"].str.split(";").explode().dropna())
    assert {"ATP", "Alanine", "AMP"} <= scored_features
    # The exclusively-lipid feature has no kept pathway and vanishes from
    # scoring entirely.
    assert "DG(16:1(9Z)/22:0/0:0)" not in scored_features


def test_prefer_tagged_features_drops_plain_twins():
    f2h = pd.DataFrame([
        # Standard-confirmed twin (tagged) + razor-thin plain twin: the
        # Argininosuccinic acid shape.
        {"feature": "Argininosuccinic acid.HMDB0000052", "hmdb_id": "HMDB0000052",
         "match_method": "hmdb_tag", "n_hmdb_ids": 1},
        {"feature": "Argininosuccinic acid", "hmdb_id": "HMDB0000052",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        # Tagged-vs-tagged sharing an ID: both confirmed, both kept.
        {"feature": "TwinA.HMDB0000001", "hmdb_id": "HMDB0000001",
         "match_method": "hmdb_tag", "n_hmdb_ids": 1},
        {"feature": "TwinB.HMDB0000001", "hmdb_id": "HMDB0000001",
         "match_method": "hmdb_tag", "n_hmdb_ids": 1},
        # Plain-only metabolite (no tagged twin): untouched.
        {"feature": "Solo Metabolite", "hmdb_id": "HMDB0000002",
         "match_method": "name_exact", "n_hmdb_ids": 1},
        # Unmatched feature: untouched.
        {"feature": "Unknown compound", "hmdb_id": None,
         "match_method": "unmatched", "n_hmdb_ids": 0},
    ])
    cols = ["Argininosuccinic acid.HMDB0000052", "Argininosuccinic acid",
            "TwinA.HMDB0000001", "TwinB.HMDB0000001",
            "Solo Metabolite", "Unknown compound"]
    curated, dropped = prefer_tagged_features(f2h, feature_columns=cols)
    assert set(dropped["feature"]) == {"Argininosuccinic acid"}
    assert dropped.iloc[0]["superseded_by"] == "Argininosuccinic acid.HMDB0000052"
    assert set(curated["feature"]) == {
        "Argininosuccinic acid.HMDB0000052",
        "TwinA.HMDB0000001", "TwinB.HMDB0000001",
        "Solo Metabolite", "Unknown compound"}


def test_prefer_tagged_features_override_confers_precedence():
    # The overridden plain feature is an identity claim too; it supersedes
    # its name-matched plain twin.
    f2h = pd.DataFrame([
        {"feature": "Niacin", "hmdb_id": "HMDB0001488",
         "match_method": "override", "n_hmdb_ids": 1},
        {"feature": "Niacinamide.HMDB0001406", "hmdb_id": "HMDB0001406",
         "match_method": "hmdb_tag", "n_hmdb_ids": 1},
        {"feature": "Nicotinic acid(NA).HMDB0001488", "hmdb_id": "HMDB0001488",
         "match_method": "hmdb_tag", "n_hmdb_ids": 1},
    ])
    cols = ["Niacin", "Niacinamide.HMDB0001406",
            "Nicotinic acid(NA).HMDB0001488"]
    curated, dropped = prefer_tagged_features(f2h, feature_columns=cols)
    assert dropped.empty  # Niacin is itself confirmed via the override;
    # its would-be plain twin does not exist in this fixture.
    assert len(curated) == 3

    # With a plain name-matched twin on the same ID, the override wins.
    f2h2 = pd.concat([f2h, pd.DataFrame([
        {"feature": "Nicotinic acid", "hmdb_id": "HMDB0001488",
         "match_method": "name_exact", "n_hmdb_ids": 1},
    ])], ignore_index=True)
    cols2 = cols + ["Nicotinic acid"]
    curated2, dropped2 = prefer_tagged_features(f2h2, feature_columns=cols2)
    assert set(dropped2["feature"]) == {"Nicotinic acid"}
    assert "Nicotinic acid" not in set(curated2["feature"])


# ---------------------------------------------------------------------------
# STEP 8d: biomarker attachment channel wiring
# ---------------------------------------------------------------------------

def test_run_pipeline_biomarker_channel_flags_unmapped_biomarker(tmp_path):
    """Smoke test: a literature biomarker attached to a disease pathway is
    z-scored even when NO PathBank pathway maps its feature, and a spike in
    it flags the sample through the biomarker channel (OR into the
    decision), while the pathway channel alone would have missed it."""
    import numpy as np
    import yaml
    from pathway_pipeline.main import run_pipeline

    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)

    # Attach octanoylcarnitine to a fictitious MCAD pathway (SMP0000055
    # stands in for the disease pathway; the attachment targets it by
    # smp_id). The biomarker feature exists in the matrix and matches
    # HMDB0000215 via the HMDB tag, but no PathBank pathway lists it.
    attachments = tmp_path / "pathway_biomarker_attachments.csv"
    attachments.write_text(
        "smp_id,pathway_name,hmdb_id,source\n"
        "SMP0000055,,HMDB0000215,literature-ref\n",
        encoding="utf-8")

    rng = np.random.default_rng(21)
    n_normal, n_other = 30, 4
    n = n_normal + n_other
    spike = rng.normal(1.0, 0.01, size=n)
    spike[-1] = 8.0  # one 'other' sample grossly elevated
    data = {
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * n_other,
        "Oordeel targeted": [0] * n_normal + [1] * n_other,
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
        "AMP": list(rng.normal(2.0, 0.30, size=n)),
        # Biomarker feature: HMDB tag matches HMDB0000215 (add it to the
        # tiny synthetic index via a tag-style column name).
        "Octanoylcarnitine.HMDB0000215": list(spike),
    }
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_coverage": 0.10,
        "min_pathway_features": 2,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.05,
        "biomarker_channel": {
            "enable": True,
            "attachments_file": str(attachments),
        },
        "save_mapping_outputs": True,
        "save_zscore_outputs": True,
        "save_stouffer_outputs": True,
        "save_flagging_outputs": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    result = run_pipeline(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
        config_path=str(config_path),
    )
    decisions = result["sample_decisions"]
    # The biomarker feature was z-scored although no PathBank pathway maps
    # it (feature_to_pathway would never contain it).
    assert "Octanoylcarnitine.HMDB0000215" in set(result["zscores"].columns)
    f2p = result["feature_to_pathway"]
    assert f2p.empty or "Octanoylcarnitine.HMDB0000215" not in set(f2p["feature"])
    # The spiked sample is flagged through the biomarker channel.
    spiked = decisions[decisions["sample_id"] == f"s{n - 1}"].iloc[0]
    assert bool(spiked["biomarker_flagged"])
    assert bool(spiked["flagged"])
    assert bool(spiked["flagged_pathway_channel"]) is False
    # The channel columns are present for audit.
    for col in ("biomarker_flagged", "flagged_pathway_channel",
                "n_flagged_biomarkers", "biomarker_depth_p",
                "max_biomarker_z", "top_biomarker"):
        assert col in decisions.columns
    # Flags table recorded the (sample, pathway, biomarker) evidence.
    bio_flags = pd.read_csv(tmp_path / "out" / "biomarker_flags.csv")
    assert (bio_flags["hmdb_id"] == "HMDB0000215").any()


def test_run_pipeline_disease_table_channel(tmp_path):
    """Smoke test: the IEMbase-style disease table drives the biomarker
    channel end to end -- the table's HMDB codes join the z-score set,
    direction arrows gate which tail may flag, and a disease without a
    kept PathBank pathway still scores."""
    import numpy as np
    import yaml
    from pathway_pipeline.main import run_pipeline

    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)

    disease_table = tmp_path / "disease_table.csv"
    pd.DataFrame([
        {"Disease": "MCAD deficiency", "OMIM": "603361",
         "Biochemical_Markers": "Octanoylcarnitine \u2191; Glucose \u2193",
         "PathBank disease pathway": "MCAD (PathBank PW000216)",
         "SMPDB code (SMP)": "SMP0000055",
         "HMDB codes of named metabolites":
             "Octanoylcarnitine (HMDB0000215); Glucose (HMDB0000122)"},
        {"Disease": "MTHFR deficiency", "OMIM": "236250",
         "Biochemical_Markers": "Methionine \u2193",
         "PathBank disease pathway": "Not found in PathBank",
         "SMPDB code (SMP)": "",
         "HMDB codes of named metabolites":
             "Methionine (HMDB0000696)"},
    ]).to_csv(disease_table, index=False)

    rng = np.random.default_rng(23)
    n_normal, n_other = 30, 3
    n = n_normal + n_other
    oct_spikes = rng.normal(1.0, 0.01, size=n)
    oct_spikes[-1] = 8.0
    glu_drops = rng.normal(1.0, 0.01, size=n)
    glu_drops[-1] = -6.0
    data = {
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * n_other,
        "Oordeel targeted": [0] * n_normal + [1] * n_other,
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
        "AMP": list(rng.normal(2.0, 0.30, size=n)),
        "Octanoylcarnitine.HMDB0000215": list(oct_spikes),
        "Glucose.HMDB0000122": list(glu_drops),
    }
    input_csv = tmp_path / "input.csv"
    pd.DataFrame(data).to_csv(input_csv, index=False)
    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_coverage": 0.10,
        "min_pathway_features": 2,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.05,
        "biomarker_channel": {
            "enable": True,
            "disease_table_file": str(disease_table),
            "attachments_file": str(tmp_path / "missing_attachments.csv"),
        },
        "save_mapping_outputs": True,
        "save_zscore_outputs": True,
        "save_stouffer_outputs": True,
        "save_flagging_outputs": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    result = run_pipeline(
        input_file=str(input_csv),
        output_dir=str(tmp_path / "out"),
        config_path=str(config_path),
    )
    decisions = result["sample_decisions"]
    # The audit CSV exists and reports the pathway status per row: MCAD's
    # SMP is kept; MTHFR has no PathBank pathway at all.
    audit = pd.read_csv(tmp_path / "out" / "disease_table_audit.csv")
    assert len(audit) == 3
    assert set(audit.loc[audit["disease"] == "MCAD deficiency",
                        "pathway_status"]) == {"kept"}
    assert set(audit.loc[audit["disease"] == "MTHFR deficiency",
                        "pathway_status"]) == {"no_pathbank_pathway"}
    # The biomarker features joined the z-score set.
    assert "Octanoylcarnitine.HMDB0000215" in set(result["zscores"].columns)
    assert "Glucose.HMDB0000122" in set(result["zscores"].columns)
    # The spiked sample flags through the channel; the drop flags MCAD's
    # 'down' glucose too (either way the disease flags).
    spiked = decisions[decisions["sample_id"] == f"s{n - 1}"].iloc[0]
    assert bool(spiked["biomarker_flagged"])
    assert bool(spiked["flagged"])
    assert "top_disease" in decisions.columns
    assert spiked["top_disease"] == "MCAD deficiency"
    # Flags table records the (sample, disease, biomarker) evidence.
    bio_flags = pd.read_csv(tmp_path / "out" / "biomarker_flags.csv")
    assert (bio_flags["disease"] == "MCAD deficiency").any()
    assert set(bio_flags["direction"]) <= {"up", "down", ""}
