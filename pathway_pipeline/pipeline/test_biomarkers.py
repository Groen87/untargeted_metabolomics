#!/usr/bin/env python3
"""Unit tests for the biomarker attachment channel (STEP 8d)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.biomarkers import (
    load_biomarker_attachments,
    resolve_biomarker_attachments,
    aggregate_metabolite_zscores,
    flag_biomarker_attachments,
)


def _attachments(path, rows):
    df = pd.DataFrame(rows, columns=["smp_id", "pathway_name",
                                     "hmdb_id", "source"])
    df.to_csv(path, index=False)
    return str(path)


def _coverage():
    return pd.DataFrame([
        {"smp_id": "SMP0000055", "pathway_name": "Alanine Metabolism"},
        {"smp_id": "SMP0000056", "pathway_name": "MCAD Deficiency"},
    ])


def _f2h():
    return pd.DataFrame([
        {"feature": "Alanine", "hmdb_id": "HMDB0000161"},
        {"feature": "Alanine.HMDB0000161", "hmdb_id": "HMDB0000161"},
        {"feature": "Octanoylcarnitine", "hmdb_id": "HMDB0000215"},
        {"feature": "Unmatched", "hmdb_id": None},
    ])


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def test_load_attachments_missing_file_disables_channel(tmp_path):
    table = load_biomarker_attachments(str(tmp_path / "missing.csv"))
    assert table.empty
    assert list(table.columns) == ["smp_id", "pathway_name", "hmdb_id",
                                   "source"]


def test_load_attachments_normalizes_and_drops_invalid_rows(tmp_path):
    csv = _attachments(tmp_path / "att.csv", [
        {"smp_id": "SMP0000056", "pathway_name": "", "hmdb_id": " hmdb0000215 ",
         "source": "ref1"},
        {"smp_id": "", "pathway_name": "Alanine Metabolism",
         "hmdb_id": "HMDB0000161", "source": "ref2"},
        # No pathway reference at all -> dropped.
        {"smp_id": "", "pathway_name": "", "hmdb_id": "HMDB0009999",
         "source": "ref3"},
        # No hmdb_id -> dropped.
        {"smp_id": "SMP0000056", "pathway_name": "", "hmdb_id": "",
         "source": "ref4"},
    ])
    table = load_biomarker_attachments(csv)
    assert len(table) == 2
    assert set(table["hmdb_id"]) == {"HMDB0000215", "HMDB0000161"}
    # Case-normalized accession.
    assert "hmdb0000215" not in set(table["hmdb_id"])


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------

def test_resolve_by_smp_id_and_by_name():
    attachments = pd.DataFrame([
        {"smp_id": "SMP0000056", "pathway_name": None, "hmdb_id": "HMDB0000215",
         "source": "ref1"},
        {"smp_id": None, "pathway_name": "alanine metabolism",
         "hmdb_id": "HMDB0000161", "source": "ref2"},
    ])
    resolved, features = resolve_biomarker_attachments(
        attachments, _f2h(), _coverage())
    assert set(resolved["smp_id"]) == {"SMP0000056", "SMP0000055"}
    assert set(features) == {"Alanine", "Alanine.HMDB0000161",
                             "Octanoylcarnitine"}
    mcad = resolved[resolved["smp_id"] == "SMP0000056"].iloc[0]
    assert mcad["hmdb_id"] == "HMDB0000215"
    assert mcad["features"] == "Octanoylcarnitine"


def test_resolve_drops_unkept_pathway_and_unmatched_biomarker():
    attachments = pd.DataFrame([
        # Pathway outside the kept set.
        {"smp_id": "SMP9999999", "pathway_name": None, "hmdb_id": "HMDB0000215",
         "source": "ref1"},
        # Biomarker with no dataset feature.
        {"smp_id": "SMP0000056", "pathway_name": None, "hmdb_id": "HMDB0008888",
         "source": "ref2"},
    ])
    resolved, features = resolve_biomarker_attachments(
        attachments, _f2h(), _coverage())
    assert resolved.empty
    assert features == []


def test_resolve_ambiguous_name_dropped():
    coverage = pd.DataFrame([
        {"smp_id": "SMP0000055", "pathway_name": "Dual Name"},
        {"smp_id": "SMP0000056", "pathway_name": "dual name"},
    ])
    attachments = pd.DataFrame([
        {"smp_id": None, "pathway_name": "Dual Name", "hmdb_id": "HMDB0000215",
         "source": "ref1"},
    ])
    resolved, _ = resolve_biomarker_attachments(attachments, _f2h(), coverage)
    assert resolved.empty


# ---------------------------------------------------------------------------
# Z-score aggregation
# ---------------------------------------------------------------------------

def test_aggregate_combines_duplicate_features_weighted():
    rng = np.random.default_rng(5)
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(6, 3)),
        index=[f"s{i}" for i in range(6)],
        columns=["Alanine", "Alanine.HMDB0000161", "Octanoylcarnitine"])
    weights = {"Alanine": 0.25, "Alanine.HMDB0000161": 4.0}
    combined = aggregate_metabolite_zscores(
        zscores, _f2h(), ["HMDB0000161", "HMDB0000215"],
        feature_scale_weights=weights)
    assert list(combined.columns) == ["HMDB0000161", "HMDB0000215"]
    # The wide-scale twin (weight 4) dominates the metabolite mean.
    expected = (zscores["Alanine"] * 0.25
                + zscores["Alanine.HMDB0000161"] * 4.0) / 4.25
    pd.testing.assert_series_equal(combined["HMDB0000161"], expected,
                                   check_names=False)
    pd.testing.assert_series_equal(combined["HMDB0000215"],
                                   zscores["Octanoylcarnitine"],
                                   check_names=False)


def test_aggregate_skips_biomarker_without_zscored_feature():
    rng = np.random.default_rng(6)
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(6, 2)),
        index=[f"s{i}" for i in range(6)],
        columns=["Alanine", "Octanoylcarnitine"])
    combined = aggregate_metabolite_zscores(
        zscores, _f2h(), ["HMDB0008888"])
    assert combined.empty


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def _resolved():
    return pd.DataFrame([
        {"smp_id": "SMP0000056", "pathway_name": "MCAD Deficiency",
         "hmdb_id": "HMDB0000215", "source": "ref1",
         "features": "Octanoylcarnitine", "n_features": 1},
        {"smp_id": "SMP0000055", "pathway_name": "Alanine Metabolism",
         "hmdb_id": "HMDB0000161", "source": "ref2",
         "features": "Alanine;Alanine.HMDB0000161", "n_features": 2},
    ])


def test_flag_biomarker_channel_flags_spike_and_ors():
    rng = np.random.default_rng(7)
    n = 40
    z = rng.normal(0, 1, size=(n, 3))
    zscores = pd.DataFrame(
        z, index=[f"s{i}" for i in range(n)],
        columns=["Alanine", "Alanine.HMDB0000161", "Octanoylcarnitine"])
    # Normal mask: all but the last two samples; the last one spikes
    # octanoylcarnitine.
    normal_mask = pd.Series(
        [True] * (n - 2) + [False, False], index=zscores.index)
    zscores.loc[zscores.index[-1], "Octanoylcarnitine"] = 12.0
    zscores.loc[zscores.index[-2], "Octanoylcarnitine"] = 30.0

    flags, summary = flag_biomarker_attachments(
        zscores, _resolved(), _f2h(), normal_mask=normal_mask,
        threshold_percentile=99.0)
    assert not flags.empty
    assert set(flags.columns) == {"sample_id", "smp_id", "pathway_name",
                                  "hmdb_id", "abs_z", "threshold", "excess",
                                  "flagged"}
    flagged = flags[flags["flagged"]]
    # Both spiked samples flag through the MCAD pathway attachment (a few
    # normal pairs may exceed their own 99th-percentile threshold -- the
    # depth null, not the pair flag, drives the sample decision).
    assert set(zscores.index[-2:]) <= set(flagged.loc[
        flagged["smp_id"] == "SMP0000056", "sample_id"])
    row = summary[summary["sample_id"] == zscores.index[-1]].iloc[0]
    assert row["top_biomarker"] == "HMDB0000215"
    assert row["max_biomarker_z"] == pytest.approx(12.0)
    assert row["biomarker_depth_p"] == 0.0
    # Normal samples rarely flag through the channel (pair flags at the
    # 99th percentile are ~1% per biomarker; the depth null filters the
    # rest at the sample level).
    normals = summary[summary["sample_id"].isin(zscores.index[:-2])]
    assert int(normals["n_flagged_biomarkers"].sum()) <= 3


def test_flag_biomarker_channel_empty_inputs():
    flags, summary = flag_biomarker_attachments(
        pd.DataFrame(), _resolved(), _f2h(), normal_mask=pd.Series(dtype=bool))
    assert flags.empty
    assert summary.empty


# ---------------------------------------------------------------------------
# IEMbase-style disease biomarker table
# ---------------------------------------------------------------------------

from pathway_pipeline.pipeline.biomarkers import (
    load_disease_biomarker_table,
    resolve_disease_biomarkers,
    flag_disease_biomarkers,
    audit_unlinked_disease_markers,
)


def _write_disease_table(path):
    df = pd.DataFrame([
        {"Disease": "MCAD deficiency", "OMIM": "603361",
         "Abbreviation IEMbase": "MCADD",
         "Biochemical_Markers": "Octanoylcarnitine \u2191; Glucose \u2193",
         "PathBank disease pathway": "MCAD (PathBank PW000216)",
         "SMPDB code (SMP)": "SMP0000555",
         "HMDB codes of named metabolites":
             "Octanoylcarnitine (HMDB0000215); Glucose (HMDB0000122)"},
        {"Disease": "MTHFR deficiency", "OMIM": "236250",
         "Abbreviation IEMbase": "MTHFR",
         "Biochemical_Markers": "Methionine \u2193; Homocysteine \u2191",
         "PathBank disease pathway": "Not found in PathBank",
         "SMPDB code (SMP)": "",
         "HMDB codes of named metabolites":
             "Methionine (HMDB0000696); Homocysteine (no HMDB entry found)"},
    ])
    df.to_csv(path, index=False)
    return str(path)


def test_load_disease_table_parses_rows_directions_and_codes(tmp_path):
    table = load_disease_biomarker_table(
        _write_disease_table(tmp_path / "disease.csv"))
    assert len(table) == 3  # homocysteine has no code -> skipped
    mcad = table[table["disease"] == "MCAD deficiency"]
    assert set(mcad["hmdb_id"]) == {"HMDB0000215", "HMDB0000122"}
    assert dict(zip(mcad["hmdb_id"], mcad["direction"])) == {
        "HMDB0000215": "up", "HMDB0000122": "down"}
    assert set(mcad["biomarker"]) == {"Octanoylcarnitine", "Glucose"}
    assert mcad["smp_id"].iloc[0] == "SMP0000555"
    mthfr = table[table["disease"] == "MTHFR deficiency"]
    assert len(mthfr) == 1
    assert mthfr["direction"].iloc[0] == "down"
    assert pd.isna(mthfr["smp_id"].iloc[0])
    assert mthfr["pathway_name"].iloc[0] == "Not found in PathBank"


def test_audit_unlinked_disease_markers_matches_names(tmp_path):
    """The unlinked-marker audit reports no-HMDB-code markers with
    their dataset name matches and name-index accessions, without
    touching scoring."""
    table_path = _write_disease_table(tmp_path / "disease.csv")
    audit = audit_unlinked_disease_markers(
        table_path,
        feature_columns=["Homocysteine.HMDB0000157", "Other feature"],
        name_index={"HOMOCROSTEINE": {"HMDB0001111"}})
    # Homocysteine is the single no-code marker in the fixture.
    assert list(audit["marker"]) == ["Homocysteine"]
    row = audit.iloc[0]
    assert row["disease"] == "MTHFR deficiency"
    assert row["direction"] == "up"
    # Exact name matching finds the tagged feature column.
    assert row["matched_features"] == "Homocysteine.HMDB0000157"
    # No name-index hit for the marker itself.
    assert row["name_index_accessions"] == ""
    # A marker with no feature match but a name-index accession is
    # reported for potential table enrichment.
    audit = audit_unlinked_disease_markers(
        table_path,
        feature_columns=["Unrelated column"],
        name_index={"HOMOCYSTEINE": {"HMDB0000157"}})
    row = audit.iloc[0]
    assert row["matched_features"] == ""
    assert row["name_index_accessions"] == "HMDB0000157"


def test_load_disease_table_missing_file(tmp_path):
    table = load_disease_biomarker_table(str(tmp_path / "missing.xlsx"))
    assert table.empty


def test_resolve_disease_biomarkers_reports_and_filters(tmp_path):
    table = load_disease_biomarker_table(
        _write_disease_table(tmp_path / "disease.csv"))
    f2h = pd.DataFrame([
        {"feature": "Octanoylcarnitine.HMDB0000215", "hmdb_id": "HMDB0000215"},
        {"feature": "Glucose.HMDB0000122", "hmdb_id": "HMDB0000122"},
        {"feature": "Methionine", "hmdb_id": "HMDB0000696"},
    ])
    coverage = pd.DataFrame([
        {"smp_id": "SMP0000555", "pathway_name": "MCAD"}])
    resolved, features, audit = resolve_disease_biomarkers(
        table, f2h, coverage)
    # All three codes resolve to dataset features; MCAD's SMP is in the
    # kept set, MTHFR's disease scores without a kept pathway.
    assert set(resolved["hmdb_id"]) == {"HMDB0000215", "HMDB0000122",
                                       "HMDB0000696"}
    assert set(features) == {"Octanoylcarnitine.HMDB0000215",
                             "Glucose.HMDB0000122", "Methionine"}
    assert len(audit) == 3
    # Pathway link reported only when the SMP is kept.
    mcad_row = audit[audit["disease"] == "MCAD deficiency"]
    assert set(mcad_row["pathway_status"]) == {"kept"}
    mthfr_row = audit[audit["disease"] == "MTHFR deficiency"]
    assert set(mthfr_row["pathway_status"]) == {"no_pathbank_pathway"}


def test_flag_disease_biomarkers_direction_and_disease_groups(tmp_path):
    table = load_disease_biomarker_table(
        _write_disease_table(tmp_path / "disease.csv"))
    f2h = pd.DataFrame([
        {"feature": "Octanoylcarnitine.HMDB0000215", "hmdb_id": "HMDB0000215"},
        {"feature": "Glucose.HMDB0000122", "hmdb_id": "HMDB0000122"},
        {"feature": "Methionine", "hmdb_id": "HMDB0000696"},
    ])
    coverage = pd.DataFrame([
        {"smp_id": "SMP0000555", "pathway_name": "MCAD"}])
    resolved, _, _ = resolve_disease_biomarkers(table, f2h, coverage)

    rng = np.random.default_rng(3)
    n = 43
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(n, 3)),
        index=[f"s{i}" for i in range(n)],
        columns=["Octanoylcarnitine.HMDB0000215", "Glucose.HMDB0000122",
                 "Methionine"])
    # The last three samples are 'patients' (excluded from the reference).
    normal_mask = pd.Series(
        [True] * (n - 3) + [False, False, False], index=zscores.index)
    # One sample: methionine drops hard (MTHFR 'down' biomarker).
    zscores.loc["s40", "Methionine"] = -14.0
    # Another: octanoylcarnitine rises hard (MCAD 'up' biomarker).
    zscores.loc["s41", "Octanoylcarnitine.HMDB0000215"] = 14.0
    # And a sample with methionine RISING: must NOT flag MTHFR ('down').
    zscores.loc["s42", "Methionine"] = 14.0

    flags, summary = flag_disease_biomarkers(
        zscores, resolved, f2h, normal_mask=normal_mask,
        threshold_percentile=99.0, max_sample_p=0.05)

    flagged_diseases = flags[flags["flagged"]].groupby(
        "sample_id")["disease"].apply(set).to_dict()
    assert "MTHFR deficiency" in flagged_diseases.get("s40", set())
    assert "MCAD deficiency" in flagged_diseases.get("s41", set())
    assert "MTHFR deficiency" not in flagged_diseases.get("s42", set())
    row = summary[summary["sample_id"] == "s40"].iloc[0]
    assert bool(row["biomarker_flagged"])
    assert row["top_disease"] == "MTHFR deficiency"
    # Normals rarely flag (threshold at the 99th percentile).
    normal_rows = summary[summary["sample_id"].isin(zscores.index[:-3])]
    assert int(normal_rows["n_flagged_biomarkers"].sum()) <= 3


def test_flag_disease_biomarkers_ratio_tests():
    """Declared ratio tests join the channel: direction-aware, in the
    same global depth null, and attributed to their disease."""
    rng = np.random.default_rng(5)
    n = 40
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(n, 2)),
        index=[f"s{i}" for i in range(n)],
        columns=["C8/C2", "Glucose.HMDB0000122"])
    normal_mask = pd.Series(
        [True] * (n - 2) + [False, False], index=zscores.index)
    zscores.loc["s38", "C8/C2"] = 12.0
    # Wrong direction: C8/C2 DOWN must not flag the 'up' disease.
    zscores.loc["s39", "C8/C2"] = -12.0
    resolved = pd.DataFrame(columns=[
        "disease", "biomarker", "hmdb_id", "direction", "omim",
        "smp_id", "pathway_name", "source", "features", "n_features"])
    ratio_tests = pd.DataFrame([
        {"disease": "MCAD deficiency", "ratio": "C8/C2", "direction": "up"},
        {"disease": "MCAD deficiency", "ratio": "C8/C10", "direction": "up"},
    ])
    # C8/C10 has no z-scored column: the test must be skipped loudly,
    # not silently scored as zeros.
    f2h = pd.DataFrame({"feature": ["C8/C2"], "hmdb_id": [None]})
    flags, summary = flag_disease_biomarkers(
        zscores, resolved, f2h, normal_mask=normal_mask,
        threshold_percentile=99.0, max_sample_p=0.05,
        ratio_tests=ratio_tests)
    flagged = flags[flags["flagged"]]
    assert "C8/C2" in set(flagged.loc[flagged["sample_id"] == "s38",
                                      "biomarker"])
    assert flags.loc[(flags["sample_id"] == "s38")
                     & (flags["biomarker"] == "C8/C2"), "disease"].iloc[0] \
        == "MCAD deficiency"
    assert "s39" not in set(flagged["sample_id"])
    row = summary[summary["sample_id"] == "s38"].iloc[0]
    assert bool(row["biomarker_flagged"])
    assert row["top_disease"] == "MCAD deficiency"
    assert row["top_biomarker"] == "C8/C2"
    assert flags[flags["biomarker"] == "C8/C10"].empty


def test_flag_disease_biomarkers_ratio_disease_spelling_warning(caplog):
    """A ratio-test disease name absent from the IEMbase table warns:
    the test still scores, but its evidence is attributed to the
    unmatched spelling instead of merging with the disease's
    metabolite tests."""
    import logging
    rng = np.random.default_rng(7)
    n = 40
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(n, 2)),
        index=[f"s{i}" for i in range(n)],
        columns=["C8/C2", "Glucose.HMDB0000122"])
    normal_mask = pd.Series(
        [True] * (n - 1) + [False], index=zscores.index)
    zscores.loc["s39", "C8/C2"] = 12.0
    resolved = pd.DataFrame([{
        "disease": "Medium-chain acyl-CoA dehydrogenase deficiency",
        "biomarker": "Octanoylcarnitine", "hmdb_id": "HMDB0000791",
        "direction": "up", "omim": "", "smp_id": "SMP0000055",
        "pathway_name": "Alanine Metabolism", "source": "IEMbase",
        "features": "Octanoylcarnitine.HMDB0000791", "n_features": 1,
    }])
    ratio_tests = pd.DataFrame([
        {"disease": "Medium-chain acyl-CoA dehydrogenase deficiency",
         "ratio": "C8/C2", "direction": "up"},
        {"disease": "MCAD deficiency (misspelled)",
         "ratio": "C8/C2", "direction": "up"},
    ])
    f2h = pd.DataFrame(
        {"feature": ["Octanoylcarnitine.HMDB0000791"],
         "hmdb_id": ["HMDB0000791"]})
    with caplog.at_level(logging.WARNING,
                          logger="pathway_pipeline.pipeline.biomarkers"):
        flags, summary = flag_disease_biomarkers(
            zscores, resolved, f2h, normal_mask=normal_mask,
            threshold_percentile=99.0, max_sample_p=0.05,
            ratio_tests=ratio_tests)
    spelling = [r for r in caplog.records
                if "ratio-test disease name" in r.getMessage()]
    assert len(spelling) == 1
    assert "MCAD deficiency (misspelled)" in spelling[0].getMessage()
    # The matched spelling raises no warning of its own.
    assert "Medium-chain acyl-CoA" not in spelling[0].getMessage()
    # Both tests still score under their own disease names.
    diseases = set(flags.loc[flags["biomarker"] == "C8/C2", "disease"])
    assert "MCAD deficiency (misspelled)" in diseases
    assert "Medium-chain acyl-CoA dehydrogenase deficiency" in diseases


def test_flag_disease_biomarkers_empty_inputs():
    flags, summary = flag_disease_biomarkers(
        pd.DataFrame(), pd.DataFrame(columns=["disease", "hmdb_id"]),
        _f2h(), normal_mask=pd.Series(dtype=bool))
    assert flags.empty
    assert summary.empty
