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
