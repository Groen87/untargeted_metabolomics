#!/usr/bin/env python3
"""Unit tests for the pathway pipeline (parser, mapping, statistics).

All inputs are synthetic and written to temp files, so the tests run without
the real HMDB XML / pathways TSV / feature matrix.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.name_utils import normalize_name, normalize_loose
from pathway_pipeline.pipeline.hmdb_parser import build_name_index
from pathway_pipeline.pipeline.pathway_mapping import (
    load_pathways_tsv,
    match_features_to_hmdb,
    link_features_to_pathways,
    pathway_coverage,
)
from pathway_pipeline.pipeline.pathway_stats import (
    compute_metabolite_zscores,
    compute_pathway_statistics,
    flag_pathways,
    flag_metabolites,
    compute_global_anomaly_score,
    decide_samples,
    tune_decision_thresholds,
    _age_adjust,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic HMDB XML, pathways TSV, feature matrix
# ---------------------------------------------------------------------------

HMDB_XML = """<?xml version="1.0"?>
<hmdb>
  <metabolite>
    <accession>HMDB0000063</accession>
    <name>Cortisol</name>
    <synonyms>
      <synonym>Hydrocortisone</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000015</accession>
    <name>Cortisol sulfate</name>
    <synonyms>
      <synonym>Cortisol sulphate</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000037</accession>
    <name>PS(18:2W6/24:1W9)</name>
  </metabolite>
  <metabolite>
    <accession>HMDB0000223</accession>
    <name>Glucose</name>
    <synonyms>
      <synonym>D-Glucose</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000902</accession>
    <name>Aspartate</name>
    <synonyms>
      <synonym>L-Aspartate</synonym>
    </synonyms>
  </metabolite>
  <metabolite>
    <accession>HMDB0000148</accession>
    <name>Fumarate</name>
  </metabolite>
  <metabolite>
    <accession>HMDB0000208</accession>
    <name>2-Oxoglutarate</name>
    <synonyms>
      <synonym>Alpha-ketoglutarate</synonym>
    </synonyms>
  </metabolite>
</hmdb>
"""

PATHWAYS_TSV = """smp_id\tpathway_name\tn_compounds\thmdb_ids
SMP0000575\t11-beta-Hydroxylase Deficiency (CYP11B1)\t3\tHMDB0000063;HMDB0000015;HMDB0000037
SMP0000136\tTCA Cycle\t3\tHMDB0000208;HMDB0000148;HMDB0000902
SMP0000999\tLonely pathway\t1\tHMDB0000223
"""


def _write_hmdb(tmp_path: Path) -> Path:
    p = tmp_path / "hmdb.xml"
    p.write_text(HMDB_XML, encoding="utf-8")
    return p


def _write_pathways(tmp_path: Path) -> Path:
    p = tmp_path / "pathways.tsv"
    p.write_text(PATHWAYS_TSV, encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# name_utils
# ---------------------------------------------------------------------------

def test_normalize_folds_greek_symbols_and_words():
    assert normalize_name("PS(18:2ω6/24:1ω9)") == "PS(18:2OMEGA6/24:1OMEGA9)"
    assert normalize_name("PS(18:2W6/24:1W9)") == "PS(18:2OMEGA6/24:1OMEGA9)"
    assert normalize_name("alpha-ketoglutarate") == "ALPHA-KETOGLUTARATE"
    assert normalize_name("α-Ketoglutarate") == "ALPHA-KETOGLUTARATE"


def test_normalize_loose_collapses_punctuation():
    assert normalize_loose("Cortisol sulfate") == "CORTISOLSULFATE"
    assert normalize_loose("Cortisol-sulfate") == "CORTISOLSULFATE"


# ---------------------------------------------------------------------------
# hmdb_parser
# ---------------------------------------------------------------------------

def test_build_name_index_includes_primary_name_synonyms_and_accession(tmp_path):
    xml = _write_hmdb(tmp_path)
    idx = build_name_index(str(xml), min_name_length=3, use_cache=False)
    assert idx.get("CORTISOL") == {"HMDB0000063"}
    assert idx.get("HYDROCORTISONE") == {"HMDB0000063"}
    # The bare accession is also indexed.
    assert "HMDB0000063" in idx
    assert idx["HMDB0000063"] == {"HMDB0000063"}


def test_build_name_index_is_cached(tmp_path):
    xml = _write_hmdb(tmp_path)
    first = build_name_index(str(xml), min_name_length=3, use_cache=True)
    second = build_name_index(str(xml), min_name_length=3, use_cache=True)
    assert first == second


def test_build_name_index_missing_file_returns_empty(tmp_path):
    idx = build_name_index(str(tmp_path / "nope.xml"), use_cache=False)
    assert idx == {}


# ---------------------------------------------------------------------------
# pathway_mapping
# ---------------------------------------------------------------------------

def test_load_pathways_tsv_parses_columns(tmp_path):
    tsv = _write_pathways(tmp_path)
    df = load_pathways_tsv(str(tsv))
    assert list(df.columns) == ["smp_id", "pathway_name", "n_compounds", "hmdb_ids"]
    assert df["smp_id"].tolist() == ["SMP0000575", "SMP0000136", "SMP0000999"]
    assert df.loc[0, "hmdb_ids"] == ["HMDB0000063", "HMDB0000015", "HMDB0000037"]
    assert int(df.loc[0, "n_compounds"]) == 3


def test_match_features_hmdb_tag_exact_and_loose(tmp_path):
    xml = _write_hmdb(tmp_path)
    idx = build_name_index(str(xml), min_name_length=3, use_cache=False)
    cols = [
        "Cortisol.HMDB0000063",   # HMDB tag -> HMDB0000063
        "HMDB0000015",            # bare HMDB tag
        "Hydrocortisone",         # exact synonym
        "Cortisol sulphate",      # loose match (synonym) -> HMDB0000015
        "PS(18:2ω6/24:1ω9)",      # exact after Greek folding -> HMDB0000037
        "Unknown compound",       # unmatched
    ]
    m = match_features_to_hmdb(cols, idx, min_name_length=3)
    got = m.dropna(subset=["hmdb_id"]).groupby("feature")["hmdb_id"].apply(set).to_dict()
    assert got["Cortisol.HMDB0000063"] == {"HMDB0000063"}
    assert got["HMDB0000015"] == {"HMDB0000015"}
    assert got["Hydrocortisone"] == {"HMDB0000063"}
    assert got["Cortisol sulphate"] == {"HMDB0000015"}
    assert got["PS(18:2ω6/24:1ω9)"] == {"HMDB0000037"}
    unmatched = m[m["match_method"] == "unmatched"]
    assert set(unmatched["feature"]) == {"Unknown compound"}


def test_link_features_to_pathways_and_coverage(tmp_path):
    tsv = _write_pathways(tmp_path)
    pathways = load_pathways_tsv(str(tsv))
    f2h = pd.DataFrame([
        {"feature": "Cortisol", "hmdb_id": "HMDB0000063", "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "Cortisol sulfate", "hmdb_id": "HMDB0000015", "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "Lipid", "hmdb_id": "HMDB0000037", "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "OGT", "hmdb_id": "HMDB0000208", "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "Fum", "hmdb_id": "HMDB0000148", "match_method": "name_exact", "n_hmdb_ids": 1},
        {"feature": "Asp", "hmdb_id": "HMDB0000902", "match_method": "name_exact", "n_hmdb_ids": 1},
    ])
    links = link_features_to_pathways(f2h, pathways)
    # 3 features into SMP0000575 + 3 into SMP0000136 = 6 links.
    assert len(links) == 6
    cov = pathway_coverage(links, min_pathway_size=3)
    # The lonely pathway (1 feature) is dropped.
    assert set(cov["smp_id"]) == {"SMP0000575", "SMP0000136"}
    assert all(cov["n_matched_features"] == 3)
    assert np.isclose(cov.loc[cov["smp_id"] == "SMP0000575", "coverage"].iloc[0], 1.0)


# ---------------------------------------------------------------------------
# pathway_stats
# ---------------------------------------------------------------------------

def _toy_zscores_and_mask():
    # 40 normals (small spread) + 1 abnormal with large signed deviations;
    # 4 metabolites (2 up, 2 down) so each signed side has >=2 members and the
    # signed-extreme guard is valid. The abnormal raw values are built from the
    # normal per-metabolite medians + a fixed z magnitude * each IQR, so after
    # IQR scaling the abnormal z-scores are exactly +5,+5,-5,-5 and the two
    # sides cancel in Z_med (~0), exercising the cancel-out guard via Z_split.
    cols = ["Cortisol", "Fumarate", "Aspartate", "Glutamate"]
    n = pd.DataFrame(
        np.random.RandomState(0).normal(0, 0.2, size=(40, 4)),
        columns=cols, index=[f"N{i}" for i in range(40)],
    )
    med = n.median(axis=0)
    iqr = n.apply(lambda s: np.nanpercentile(s, 75) - np.nanpercentile(s, 25), axis=0)
    target_z = pd.Series([5.0, 5.0, -5.0, -5.0], index=cols)
    a_vals = med + target_z * iqr
    a = pd.DataFrame([a_vals.to_numpy()], columns=cols, index=["A1"])
    features = pd.concat([n, a])
    normal_mask = pd.Series([True] * 40 + [False], index=features.index)
    return features, normal_mask


def test_compute_metabolite_zscores_centers_normals():
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    # Normals should have near-zero median z.
    assert np.allclose(z.loc[normal_mask].median(), 0.0, atol=1e-7)
    # The abnormal sample has positive z on Cortisol/Fumarate, negative on
    # Aspartate/Glutamate (two up, two down).
    assert z.loc["A1", "Cortisol"] > 0
    assert z.loc["A1", "Fumarate"] > 0
    assert z.loc["A1", "Aspartate"] < 0
    assert z.loc["A1", "Glutamate"] < 0


def test_pathway_statistics_zmed_and_signed_extremes_cancel_out():
    stats = _toy_stats_one_pathway()
    a1 = stats[stats["sample_id"] == "A1"].iloc[0]
    # The abnormal sample has two +z and two -z that cancel: Z_med ~ 0, so
    # |Z_med| is small while Z_up (median of the 2 positives) is large positive
    # and Z_down (median of the 2 negatives) is large negative.
    assert abs(a1["z_med"]) < 1.0
    assert a1["z_up"] > 2.0
    assert a1["z_down"] < -2.0
    # Z_split = max(|Z_up|, |Z_down|) over the valid sides (>=2 members each).
    assert a1["z_split"] > 2.0
    assert a1["z_split"] == pytest.approx(max(abs(a1["z_up"]), abs(a1["z_down"])))
    # The flagged fraction F counts metabolites exceeding t_i. With all four
    # extreme, F == 1.0 > 0.5.
    assert a1["flagged_fraction"] > 0.5


def test_flag_pathways_catches_signed_extreme_with_zero_zmed():
    stats = _toy_stats_one_pathway()
    flagged = flag_pathways(stats, zmed_threshold=2.0,
                            flagged_fraction_threshold=0.5,
                            signed_extreme_threshold=2.5)
    a1 = flagged[flagged["sample_id"] == "A1"].iloc[0]
    assert a1["flagged"] is True or a1["flagged"] == True
    # The reason should mention a signed extreme (cancel-out guard).
    assert "Z_up" in a1["flag_reason"] or "Z_down" in a1["flag_reason"]


def test_normal_sample_not_flagged():
    stats = _toy_stats_one_pathway()
    flagged = flag_pathways(stats, zmed_threshold=2.0,
                            flagged_fraction_threshold=0.5,
                            signed_extreme_threshold=2.5)
    n0 = flagged[flagged["sample_id"] == "N0"].iloc[0]
    assert bool(n0["flagged"]) is False


# ---------------------------------------------------------------------------
# Layer 1: age-adjusted z-scores + atomic metabolite flags
# ---------------------------------------------------------------------------

def test_age_adjustment_removes_age_trend():
    # Metabolite rises strictly linearly with age over normals (C = 2*age).
    # An on-line young sample (age 1, value 2) should look "low" WITHOUT age
    # adjustment (far below the pooled median) but ~0 WITH age adjustment
    # (its value sits exactly on the age line).
    ages = pd.Series(np.linspace(1, 80, 40), index=[f"N{i}" for i in range(40)],
                     name="age")
    feat = pd.DataFrame({
        "C": ages.to_numpy() * 2.0 + 0.1 * np.random.RandomState(3).randn(40),
    }, index=ages.index)
    feat.index.name = "sample"
    abnormal = pd.DataFrame({"C": [1.0 * 2.0]}, index=["A1"])
    features = pd.concat([feat, abnormal])
    features.index.name = "sample"
    ages_full = pd.concat([ages, pd.Series([1.0], index=["A1"], name="age")])
    normal_mask = pd.Series([True] * 40 + [False], index=features.index)
    z_noage = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    z_age = compute_metabolite_zscores(features, normal_mask, iqr_scale=True,
                                        ages=ages_full.reindex(features.index))
    # The on-line young sample looks low without adjustment (|z| large) but
    # near-zero once the age trend is removed.
    assert abs(z_noage.loc["A1", "C"]) > 0.9
    assert abs(z_age.loc["A1", "C"]) < 0.5
    # The normal-set z-scores stay centred at 0 after adjustment.
    assert np.allclose(z_age.loc[normal_mask].median(), 0.0, atol=1e-7)


def test_age_adjustment_loess_removes_nonlinear_trend():
    # Metabolite follows a non-linear (concave-down quadratic) age trend over
    # normals. LOESS removes the curvature so the normal-set residuals are
    # far tighter than under the linear OLS fit, which mis-attributes the
    # curvature to the residuals. A sample sitting exactly on the curve stays
    # near zero under LOESS. This is the real benefit of the toggle on
    # non-linear age trends.
    ages = pd.Series(np.linspace(1, 80, 80), index=[f"N{i}" for i in range(80)],
                     name="age")
    a = ages.to_numpy()
    trend = 4.0 * a - 0.05 * a ** 2  # concave-down: peaks in middle age.
    rng = np.random.RandomState(7)
    feat = pd.DataFrame({
        "C": trend + 0.05 * rng.randn(80),
    }, index=ages.index)
    feat.index.name = "sample"
    normal_mask = pd.Series([True] * 80, index=feat.index)

    res_ols = _age_adjust(feat, ages, normal_mask, method="ols")
    res_loess = _age_adjust(feat, ages, normal_mask, method="loess",
                           loess_frac=0.4)
    # The quadratic curvature stays in the OLS residuals (large spread) but
    # is removed by LOESS (tight residual spread around the noise floor).
    assert res_loess["C"].std() < res_ols["C"].std() / 5
    # LOESS removes the curvature far more than OLS: the residual IQR is also
    # far smaller under LOESS.
    loess_iqr = float(np.diff(np.nanpercentile(res_loess["C"], [25, 75]))[0])
    ols_iqr = float(np.diff(np.nanpercentile(res_ols["C"], [25, 75]))[0])
    assert loess_iqr < ols_iqr / 5

    # A sample sitting exactly on the (non-linear) curve at an interior age
    # stays near zero under LOESS; the linear OLS fit leaves a curvature-driven
    # residual at that age.
    a_ab = 60.0
    on_curve = 4.0 * a_ab - 0.05 * a_ab ** 2
    abnormal = pd.DataFrame({"C": [on_curve]}, index=["A1"])
    features = pd.concat([feat, abnormal])
    ages_full = pd.concat([ages, pd.Series([a_ab], index=["A1"], name="age")])
    nm = pd.Series([True] * 80 + [False], index=features.index)
    r_loess = _age_adjust(features, ages_full, nm, method="loess",
                          loess_frac=0.4)
    r_ols = _age_adjust(features, ages_full, nm, method="ols")
    assert abs(r_loess.loc["A1", "C"]) < abs(r_ols.loc["A1", "C"])
    assert abs(r_loess.loc["A1", "C"]) < 5.0


def test_age_adjustment_unknown_method_falls_back_to_ols():
    # An unrecognised method name is treated the same as the default OLS path
    # (linear regression), so a typo in the config degrades gracefully rather
    # than raising.
    ages = pd.Series(np.linspace(1, 80, 40), index=[f"N{i}" for i in range(40)],
                     name="age")
    feat = pd.DataFrame({
        "C": ages.to_numpy() * 2.0 + 0.1 * np.random.RandomState(3).randn(40),
    }, index=ages.index)
    feat.index.name = "sample"
    abnormal = pd.DataFrame({"C": [2.0]}, index=["A1"])
    features = pd.concat([feat, abnormal])
    features.index.name = "sample"
    ages_full = pd.concat([ages, pd.Series([1.0], index=["A1"], name="age")])
    normal_mask = pd.Series([True] * 40 + [False], index=features.index)
    z_default = compute_metabolite_zscores(
        features, normal_mask, iqr_scale=True,
        ages=ages_full.reindex(features.index))
    z_unknown = compute_metabolite_zscores(
        features, normal_mask, iqr_scale=True,
        ages=ages_full.reindex(features.index),
        age_adjustment_method="not-a-real-method")
    # The unknown method path is the linear OLS branch, matching the default.
    assert np.allclose(z_unknown.to_numpy(), z_default.to_numpy(), equal_nan=True)


def test_flag_metabolites_handles_duplicate_sample_ids():
    # The real dataset carries duplicate sample IDs in the row index. Label-
    # based access (``col_z.loc[sid]``) returns a Series for a duplicate label
    # and used to crash the float() cast, so flag_metabolites must iterate by
    # integer position and emit one flag per row that trips the override.
    idx = ["S1", "S1", "S2", "S2", "S3"]
    feat = pd.DataFrame({
        "Cortisol": [1.0, 1.1, 50.0, 51.0, 2.0],
        "Fumarate": [2.0, 2.1, 3.0, 3.1, 2.5],
    }, index=idx)
    feat.index.name = "sample"
    normal_mask = pd.Series([True, True, False, False, True], index=idx)
    z = compute_metabolite_zscores(feat, normal_mask, iqr_scale=True)
    flags = flag_metabolites(z, normal_mask=normal_mask, override_threshold=4.0,
                            flag_percentile=99)
    # Both duplicate-S2 rows are extreme on Cortisol and each get its own flag.
    s2 = flags[flags["sample_id"] == "S2"]
    assert len(s2) == 2
    assert set(s2["metabolite"]) == {"Cortisol"}
    assert s2["z"].to_numpy().dtype == float


def test_flag_metabolites_atomic_override():
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    flags = flag_metabolites(z, normal_mask, override_threshold=4.0,
                            flag_percentile=99)
    # The abnormal A1 has |z| > 4 on Cortisol and Fumarate.
    a1 = flags[flags["sample_id"] == "A1"]
    flagged_mets = set(a1["metabolite"])
    assert "Cortisol" in flagged_mets
    assert "Fumarate" in flagged_mets
    # Normals should not trigger the 4.0 override.
    assert set(flags[flags["sample_id"].str.startswith("N")]["sample_id"]) == set()


# ---------------------------------------------------------------------------
# Layer 2: pathway severity tiers
# ---------------------------------------------------------------------------

def _toy_stats_one_pathway():
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    f2p = pd.DataFrame([
        {"feature": "Cortisol", "hmdb_id": "HMDB0000063", "smp_id": "SMP1",
         "pathway_name": "Pathway 1", "n_compounds": 4},
        {"feature": "Fumarate", "hmdb_id": "HMDB0000148", "smp_id": "SMP1",
         "pathway_name": "Pathway 1", "n_compounds": 4},
        {"feature": "Aspartate", "hmdb_id": "HMDB0000902", "smp_id": "SMP1",
         "pathway_name": "Pathway 1", "n_compounds": 4},
        {"feature": "Glutamate", "hmdb_id": "HMDB0000216", "smp_id": "SMP1",
         "pathway_name": "Pathway 1", "n_compounds": 4},
    ])
    stats = compute_pathway_statistics(z, f2p, normal_mask, flag_percentile=99,
                                       min_pathway_size=3)
    return stats


def test_flag_pathways_severity_tiers():
    stats = _toy_stats_one_pathway()
    flagged = flag_pathways(stats, zmed_threshold=2.0,
                            flagged_fraction_threshold=0.5,
                            signed_extreme_threshold=2.5,
                            severe_zmed_threshold=3.0,
                            severe_flagged_fraction_threshold=0.7,
                            severe_signed_extreme_threshold=4.0)
    a1 = flagged[flagged["sample_id"] == "A1"].iloc[0]
    # A1 has huge signed extremes (|Z_up| and |Z_down| >> 4) -> severe.
    assert a1["flagged"] is True or a1["flagged"] == True
    assert a1["severity"] == "severe"
    # A normal sample should be neither flagged nor severe.
    n0 = flagged[flagged["sample_id"] == "N0"].iloc[0]
    assert bool(n0["flagged"]) is False
    assert n0["severity"] == "none"


def test_flag_pathways_moderate_when_below_severe():
    # Lower the severe thresholds so high that only the moderate tier can fire,
    # confirming a row can be moderate-but-not-severe.
    stats = _toy_stats_one_pathway()
    flagged = flag_pathways(stats, zmed_threshold=0.5,
                            flagged_fraction_threshold=0.1,
                            signed_extreme_threshold=0.5,
                            severe_zmed_threshold=100.0,
                            severe_flagged_fraction_threshold=100.0,
                            severe_signed_extreme_threshold=100.0)
    a1 = flagged[flagged["sample_id"] == "A1"].iloc[0]
    assert a1["flagged"] is True or a1["flagged"] == True
    assert a1["severity"] == "moderate"


# ---------------------------------------------------------------------------
# Layer 4: global anomaly score
# ---------------------------------------------------------------------------

def test_global_anomaly_score_higher_for_odd_sample():
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    scores = compute_global_anomaly_score(z, top_k=3)
    # A1 (two extreme metabolites) should have a higher score than any normal.
    assert scores.loc["A1"] > scores.loc["N0"]
    assert scores.loc["A1"] > 2.0


# ---------------------------------------------------------------------------
# Layer 3: sample decision rule
# ---------------------------------------------------------------------------

def test_decision_rule_single_severe_flag():
    stats = _toy_stats_one_pathway()
    flagged = flag_pathways(stats, zmed_threshold=2.0,
                            flagged_fraction_threshold=0.5,
                            signed_extreme_threshold=2.5,
                            severe_zmed_threshold=3.0,
                            severe_flagged_fraction_threshold=0.7,
                            severe_signed_extreme_threshold=4.0)
    met_flags = pd.DataFrame(columns=["sample_id", "metabolite"])
    decision = decide_samples(flagged, met_flags, global_scores=None,
                              min_moderate=2, min_severe=1)
    # A1 has 1 severe pathway flag -> flagged.
    assert bool(decision.loc["A1", "flagged"]) is True
    # Normals: no severe, no moderate, no overrides -> not flagged.
    assert bool(decision.loc["N0", "flagged"]) is False


def test_decision_rule_metabolite_override_flags_sample():
    # Even with NO pathway flags, a single-metabolite override flags the sample.
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    met_flags = flag_metabolites(z, normal_mask, override_threshold=4.0,
                                flag_percentile=99)
    # Empty pathway flags table (no pathway crosses anything).
    empty_pw = pd.DataFrame(columns=["sample_id", "smp_id", "pathway_name",
                                     "n_metabolites", "z_med",
                                     "flagged_fraction", "z_up", "z_down",
                                     "threshold_percentile", "flagged",
                                     "severity", "flag_reason"])
    decision = decide_samples(empty_pw, met_flags, global_scores=None,
                              min_moderate=2, min_severe=1)
    assert bool(decision.loc["A1", "flagged"]) is True
    assert decision.loc["A1", "n_metabolite_overrides"] >= 1


def test_decision_rule_two_moderate_flags():
    # Two separate moderate pathway flags should flag a sample even with no
    # severe flag and no metabolite overrides.
    pw = pd.DataFrame([
        {"sample_id": "A1", "smp_id": "SMP1", "pathway_name": "P1",
         "n_metabolites": 3, "z_med": 2.5, "flagged_fraction": 0.6,
         "z_up": float("nan"), "z_down": float("nan"),
         "threshold_percentile": 99.0, "flagged": True,
         "severity": "moderate", "flag_reason": "mod"},
        {"sample_id": "A1", "smp_id": "SMP2", "pathway_name": "P2",
         "n_metabolites": 3, "z_med": 2.2, "flagged_fraction": 0.6,
         "z_up": float("nan"), "z_down": float("nan"),
         "threshold_percentile": 99.0, "flagged": True,
         "severity": "moderate", "flag_reason": "mod"},
    ])
    met = pd.DataFrame(columns=["sample_id", "metabolite"])
    decision = decide_samples(pw, met, global_scores=None,
                              min_moderate=2, min_severe=1)
    assert bool(decision.loc["A1", "flagged"]) is True
    assert "moderate pathway flag" in decision.loc["A1", "decision_reason"]


def test_decision_rule_one_moderate_not_flagged():
    # A single moderate pathway flag alone (no severe, no overrides) must NOT
    # flag the sample -- the breadth rule needs >=2.
    pw = pd.DataFrame([
        {"sample_id": "A1", "smp_id": "SMP1", "pathway_name": "P1",
         "n_metabolites": 3, "z_med": 2.5, "flagged_fraction": 0.6,
         "z_up": float("nan"), "z_down": float("nan"),
         "threshold_percentile": 99.0, "flagged": True,
         "severity": "moderate", "flag_reason": "mod"},
    ])
    met = pd.DataFrame(columns=["sample_id", "metabolite"])
    decision = decide_samples(pw, met, global_scores=None,
                              min_moderate=2, min_severe=1)
    assert bool(decision.loc["A1", "flagged"]) is False


def test_decision_rule_global_threshold_lights_up():
    # No pathway / metabolite flags, but the global score exceeds its
    # threshold -> flagged (the safety light).
    pw = pd.DataFrame(columns=["sample_id", "smp_id", "pathway_name",
                               "n_metabolites", "z_med", "flagged_fraction",
                               "z_up", "z_down", "threshold_percentile",
                               "flagged", "severity", "flag_reason"])
    met = pd.DataFrame(columns=["sample_id", "metabolite"])
    gs = pd.Series([0.1, 5.0], index=["N0", "A1"], name="global_anomaly_score")
    decision = decide_samples(pw, met, global_scores=gs,
                              min_moderate=2, min_severe=1,
                              global_threshold=3.0)
    assert bool(decision.loc["A1", "flagged"]) is True
    assert bool(decision.loc["N0", "flagged"]) is False


# ---------------------------------------------------------------------------
# Threshold tuning on the inner IMD split
# ---------------------------------------------------------------------------

def test_tune_decision_thresholds_runs_and_ranks():
    features, normal_mask = _toy_zscores_and_mask()
    z = compute_metabolite_zscores(features, normal_mask, iqr_scale=True)
    f2p = pd.DataFrame([
        {"feature": "Cortisol", "hmdb_id": "HMDB0000063", "smp_id": "SMP1",
         "pathway_name": "P1", "n_compounds": 4},
        {"feature": "Fumarate", "hmdb_id": "HMDB0000148", "smp_id": "SMP1",
         "pathway_name": "P1", "n_compounds": 4},
        {"feature": "Aspartate", "hmdb_id": "HMDB0000902", "smp_id": "SMP1",
         "pathway_name": "P1", "n_compounds": 4},
        {"feature": "Glutamate", "hmdb_id": "HMDB0000216", "smp_id": "SMP1",
         "pathway_name": "P1", "n_compounds": 4},
    ])
    stats = compute_pathway_statistics(z, f2p, normal_mask, flag_percentile=99,
                                       min_pathway_size=3)
    labels = pd.Series([0] * 40 + [1], index=z.index, name="label")
    sweep = tune_decision_thresholds(
        pathway_stats=stats, zscores=z, normal_mask=normal_mask, labels=labels,
        metabolite_override_grid=[4.0, 8.0],
        moderate_zmed_grid=[1.5, 3.0],
        severe_zmed_grid=[2.5, 4.0],
        flag_percentile=99, signed_extreme_grid=[2.5, 4.0],
        prevalence=0.02, metric="detection",
    )
    # The sweep should find at least one setting that detects A1 (detection 1.0).
    assert "detection_rate" in sweep.columns
    assert sweep["detection_rate"].max() == pytest.approx(1.0)
    # Sorted by score (detection) descending.
    assert sweep["score"].iloc[0] >= sweep["score"].iloc[1]


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
