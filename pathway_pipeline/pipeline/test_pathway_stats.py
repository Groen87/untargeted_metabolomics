#!/usr/bin/env python3
"""Unit tests for the z-score stage (pathway_stats).

All inputs are synthetic; no external data files are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.pathway_stats import (
    classify_samples,
    compute_metabolite_zscores,
    filter_pathways_for_scoring,
)


# ---------------------------------------------------------------------------
# Sample classification
# ---------------------------------------------------------------------------

def test_classify_samples_normal_definition():
    metadata = pd.DataFrame({
        "Classification": [0, 0, 1, 1, 2],
        "Oordeel targeted": [0, 1, 1, 0, 0],
    }, index=["a", "b", "c", "d", "e"])
    mask = classify_samples(metadata)
    # Normals = Classification 0 AND Oordeel 0 -> only 'a'.
    assert mask.tolist() == [True, False, False, False, False]


def test_classify_samples_missing_columns():
    metadata = pd.DataFrame({"foo": [1, 2]}, index=["a", "b"])
    mask = classify_samples(metadata)
    assert not mask.any()


# ---------------------------------------------------------------------------
# Z-scores
# ---------------------------------------------------------------------------

def _features():
    rng = np.random.default_rng(42)
    n = 50
    normal_vals = rng.normal(0.0, 1.0, size=(n, 3))
    df = pd.DataFrame(
        normal_vals + np.array([0.0, 0.0, 0.0]),
        columns=["A", "B", "C"],
        index=[f"s{i}" for i in range(n)],
    )
    df["A_flat"] = 1.234  # zero IQR -> dropped
    df["B_all_nan_normals"] = np.nan  # handled via mask below
    return df


def test_compute_metabolite_zscores_robust():
    df = _features()
    normal_mask = pd.Series([i < 40 for i in range(len(df))], index=df.index)

    # Make the 'B_all_nan_normals' column NaN for normals but valued elsewhere.
    df.loc[df.index[:40], "B_all_nan_normals"] = np.nan
    df.loc[df.index[40:], "B_all_nan_normals"] = 5.0

    zscores, reference_stats, dropped = compute_metabolite_zscores(
        df, normal_mask=normal_mask, iqr_scale=True)

    # Flat feature and no-normal-values feature are dropped.
    assert set(dropped["feature"]) == {"A_flat", "B_all_nan_normals"}
    assert set(dropped["reason"]) == {"zero_scale", "no_normal_values"}
    assert list(zscores.columns) == ["A", "B", "C"]

    # Normals' z-scores are centered: median ~ 0, IQR ~ 1.
    zn = zscores.loc[normal_mask]
    assert np.allclose(zn.median(), 0.0, atol=1e-9)
    assert np.allclose(
        zn.quantile(0.75) - zn.quantile(0.25), 1.0, atol=1e-9)

    # reference_stats has one row per kept feature.
    assert len(reference_stats) == 3
    assert set(reference_stats["feature"]) == {"A", "B", "C"}
    assert (reference_stats["n_normal_values"] == 40).all()


def test_compute_metabolite_zscores_no_normals():
    df = _features()
    mask = pd.Series(False, index=df.index)
    zscores, reference_stats, dropped = compute_metabolite_zscores(df, mask)
    assert zscores.empty or zscores.shape[1] == 0
    assert len(dropped) == df.shape[1]
    assert (dropped["reason"] == "no_normal_values").all()


def test_compute_metabolite_zscores_std_scale():
    df = _features().drop(columns=["A_flat", "B_all_nan_normals"])
    mask = pd.Series([i < 40 for i in range(len(df))], index=df.index)
    zscores, _, _ = compute_metabolite_zscores(df, mask, iqr_scale=False)
    # std scaling: normals have std ~ 1.
    assert np.allclose(zscores.loc[mask].std(ddof=0), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# Pathway restriction after z-scores
# ---------------------------------------------------------------------------

def _coverage():
    return pd.DataFrame([
        {"smp_id": "SMP1", "pathway_name": "Alpha Pathway", "n_metabolites": 10,
         "n_matched_metabolites": 5, "matched_metabolites": "H1;H2;H3;H4;H5",
         "n_matched_features": 5, "matched_features": "f1;f2;f3;f4;f5",
         "coverage": 0.5},
        {"smp_id": "SMP2", "pathway_name": "Beta Pathway", "n_metabolites": 10,
         "n_matched_metabolites": 3, "matched_metabolites": "H6;H7;H8",
         "n_matched_features": 3, "matched_features": "f6;f7;f8",
         "coverage": 0.3},
    ])


def _links():
    return pd.DataFrame([
        {"feature": "f1", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f2", "hmdb_id": "H2", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M2", "metabolite_name": "m2"},
        {"feature": "f9", "hmdb_id": "H3", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M3", "metabolite_name": "m3"},
        {"feature": "f6", "hmdb_id": "H6", "smp_id": "SMP2",
         "pathway_name": "Beta Pathway", "metabolite_id": "M6", "metabolite_name": "m6"},
        {"feature": "f7", "hmdb_id": "H7", "smp_id": "SMP2",
         "pathway_name": "Beta Pathway", "metabolite_id": "M7", "metabolite_name": "m7"},
    ])


def test_filter_pathways_for_scoring():
    coverage = _coverage()
    links = _links()
    # Only f1, f2 (SMP1) and f6 (SMP2) survive the z-score stage.
    available = ["f1", "f2", "f6"]

    scored = filter_pathways_for_scoring(coverage, links,
                                         available_features=available,
                                         min_pathway_features=3)
    # SMP1: H1, H2 -> 2 usable metabolites < 3 -> dropped.
    # SMP2: H6 -> 1 usable metabolite < 3 -> dropped.
    assert scored.empty

    scored = filter_pathways_for_scoring(coverage, links,
                                         available_features=["f1", "f2", "f6", "f7"],
                                         min_pathway_features=2)
    # SMP1: H1, H2 (2 >= 2) kept; SMP2: H6, H7 (2 >= 2) kept.
    assert set(scored["smp_id"]) == {"SMP1", "SMP2"}
    smp1 = scored[scored["smp_id"] == "SMP1"].iloc[0]
    assert smp1["n_matched_metabolites"] == 2
    assert smp1["matched_features"] == "f1;f2"
    assert smp1["coverage"] == pytest.approx(2 / 10)


def test_filter_pathways_empty_coverage():
    empty = pd.DataFrame(columns=["smp_id", "pathway_name", "n_metabolites",
                                  "n_matched_metabolites", "matched_metabolites",
                                  "n_matched_features", "matched_features", "coverage"])
    scored = filter_pathways_for_scoring(empty, pd.DataFrame(),
                                         available_features=["f1"],
                                         min_pathway_features=3)
    assert scored.empty
