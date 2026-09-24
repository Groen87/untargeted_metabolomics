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
    load_disease_biomarker_table,
    match_biomarker_names,
    match_diseases_to_pathways,
    resolve_disease_biomarker_table,
)
from pathway_pipeline.pipeline.name_utils import normalize_name


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

def _resolved(direction=""):
    return pd.DataFrame([
        {"smp_id": "SMP0000056", "pathway_name": "MCAD Deficiency",
         "hmdb_id": "HMDB0000215", "source": "ref1", "direction": direction,
         "features": "Octanoylcarnitine", "n_features": 1},
        {"smp_id": "SMP0000055", "pathway_name": "Alanine Metabolism",
         "hmdb_id": "HMDB0000161", "source": "ref2", "direction": direction,
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
                                  "hmdb_id", "direction", "abs_z",
                                  "threshold", "excess", "flagged"}
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
# Direction handling
# ---------------------------------------------------------------------------

def test_flag_direction_up_ignores_decrease():
    rng = np.random.default_rng(8)
    n = 40
    z = rng.normal(0, 1, size=(n, 2))
    zscores = pd.DataFrame(
        z, index=[f"s{i}" for i in range(n)],
        columns=["Alanine", "Octanoylcarnitine"])
    normal_mask = pd.Series([True] * (n - 2) + [False, False],
                            index=zscores.index)
    # One sample drops octanoylcarnitine hard (z = -12): for an 'up'
    # biomarker this must NOT flag; for a 'down' biomarker it must.
    zscores.loc[zscores.index[-2], "Octanoylcarnitine"] = -12.0
    # Another sample spikes it up (z = +12): must flag under 'up'.
    zscores.loc[zscores.index[-1], "Octanoylcarnitine"] = 12.0

    up_flags, up_summary = flag_biomarker_attachments(
        zscores, _resolved(direction="up"), _f2h(),
        normal_mask=normal_mask, threshold_percentile=99.0)
    down_flags, down_summary = flag_biomarker_attachments(
        zscores, _resolved(direction="down"), _f2h(),
        normal_mask=normal_mask, threshold_percentile=99.0)

    up_bio = up_flags[(up_flags["hmdb_id"] == "HMDB0000215")
                      & up_flags["flagged"]]["sample_id"]
    down_bio = down_flags[(down_flags["hmdb_id"] == "HMDB0000215")
                          & down_flags["flagged"]]["sample_id"]
    # 'up': only the spiked sample flags; the dropped sample does not.
    assert zscores.index[-1] in set(up_bio)
    assert zscores.index[-2] not in set(up_bio)
    # 'down': only the dropped sample flags.
    assert zscores.index[-2] in set(down_bio)
    assert zscores.index[-1] not in set(down_bio)


# ---------------------------------------------------------------------------
# Disease biomarker table (Excel/CSV source)
# ---------------------------------------------------------------------------

def test_load_disease_table_csv_and_directions(tmp_path):
    csv = tmp_path / "disease_table.csv"
    csv.write_text(
        "Disease,Biomarker,Direction,Source\n"
        "MCAD deficiency,Octanoylcarnitine,up,ref1\n"
        "MCAD deficiency,Alanine,Elevated,ref1\n"
        "Some disease,Alanine,reduced,ref2\n"
        "Some disease,Alanine,weird-label,ref2\n"
        ",Alanine,up,ref3\n"
        "Some disease,,up,ref4\n",
        encoding="utf-8")
    table = load_disease_biomarker_table(str(csv))
    assert len(table) == 4
    assert set(table["direction"]) == {"up", "down", ""}


def test_load_disease_table_missing_file(tmp_path):
    table = load_disease_biomarker_table(str(tmp_path / "missing.xlsx"))
    assert table.empty
    assert list(table.columns) == ["disease", "biomarker", "direction",
                                   "source"]


def test_match_biomarker_names_chain():
    table = pd.DataFrame([
        {"disease": "MCAD deficiency", "biomarker": "octanoyl-carnitine",
         "direction": "up", "source": "ref1"},
        {"disease": "MCAD deficiency", "biomarker": "Unknown metabolite",
         "direction": "up", "source": "ref1"},
    ])
    name_index = {
        normalize_name("Octanoylcarnitine"): {"HMDB0000215"},
        normalize_name("Octanoyl-L-carnitine"): {"HMDB0000215"},
        normalize_name("L-Alanine"): {"HMDB0000161"},
    }
    matched = match_biomarker_names(
        table, name_index,
        overrides={"Unknown metabolite": "HMDB0000161"})
    by_name = {r["biomarker"]: r for _, r in matched.iterrows()}
    # 'octanoyl-carnitine' resolves loosely to the accession.
    assert by_name["octanoyl-carnitine"]["hmdb_id"] == "HMDB0000215"
    assert by_name["octanoyl-carnitine"]["match_method"] in ("name_exact",
                                                             "name_loose")
    # The override wins over the failed name chain.
    assert by_name["Unknown metabolite"]["hmdb_id"] == "HMDB0000161"
    assert by_name["Unknown metabolite"]["match_method"] == "override"


def test_match_diseases_to_pathways_exact_substring_ambiguous():
    coverage = pd.DataFrame([
        {"smp_id": "SMP0000056", "pathway_name": "Medium-Chain Acyl-CoA "
         "Dehydrogenase Deficiency"},
        {"smp_id": "SMP0000057", "pathway_name": "Short-Chain Acyl-CoA "
         "Dehydrogenase Deficiency"},
        {"smp_id": "SMP0000055", "pathway_name": "Alanine Metabolism"},
    ])
    table = pd.DataFrame([
        {"disease": "Medium-Chain Acyl-CoA Dehydrogenase Deficiency",
         "biomarker": "Octanoylcarnitine", "direction": "up",
         "source": "ref1"},
        {"disease": "alanine metabolism", "biomarker": "Alanine",
         "direction": "", "source": "ref2"},
        {"disease": "Acyl-CoA Dehydrogenase", "biomarker": "Alanine",
         "direction": "", "source": "ref3"},
        {"disease": "Not A Real Disease", "biomarker": "Alanine",
         "direction": "", "source": "ref4"},
    ])
    matched = match_diseases_to_pathways(table, coverage)
    by_disease = matched.groupby("disease")["match_method"].agg(set)
    assert by_disease["Medium-Chain Acyl-CoA Dehydrogenase Deficiency"] \
        == {"exact"}
    assert by_disease["alanine metabolism"] == {"exact"}
    # 'Acyl-CoA Dehydrogenase' is a substring of two pathway names ->
    # ambiguous, never silently matched.
    assert by_disease["Acyl-CoA Dehydrogenase"] == {"ambiguous"}
    assert by_disease["Not A Real Disease"] == {"unmatched"}
    assert (matched.loc[matched["match_method"] == "exact",
                        "smp_id"].notna()).all()


def test_resolve_disease_table_joins_to_directional_attachments():
    table = pd.DataFrame([
        {"disease": "MCAD deficiency", "biomarker": "Octanoylcarnitine",
         "direction": "up", "source": "ref1"},
        {"disease": "Not A Real Disease", "biomarker": "Octanoylcarnitine",
         "direction": "up", "source": "ref1"},
    ])
    disease_matches = pd.DataFrame([
        {"disease": "MCAD deficiency", "smp_id": "SMP0000056",
         "pathway_name": "MCAD Deficiency", "match_method": "exact"},
        {"disease": "Not A Real Disease", "smp_id": None,
         "pathway_name": None, "match_method": "unmatched"},
    ])
    biomarker_matches = pd.DataFrame([
        {"biomarker": "Octanoylcarnitine", "hmdb_id": "HMDB0000215",
         "match_method": "name_exact", "n_hmdb_ids": 1},
    ])
    attachments = resolve_disease_biomarker_table(
        table, disease_matches, biomarker_matches)
    assert len(attachments) == 1
    row = attachments.iloc[0]
    assert row["smp_id"] == "SMP0000056"
    assert row["hmdb_id"] == "HMDB0000215"
    assert row["direction"] == "up"
