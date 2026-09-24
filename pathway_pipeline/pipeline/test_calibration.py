#!/usr/bin/env python3
"""Unit tests for the label-blind calibration and evaluation modules.

All inputs are synthetic; no external data files are needed.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.calibration import (
    assign_groups,
    leave_one_out_hygiene,
    stratified_split,
)
from pathway_pipeline.pipeline.develop import (
    diverging_duplicates,
    noise_floor_features,
    redundant_pathways,
    verify_calibration,
)
from pathway_pipeline.pipeline.evaluate import (
    bootstrap_auc,
    exact_binomial_ci,
    summarize_evaluation,
)


# ---------------------------------------------------------------------------
# Cohort split
# ---------------------------------------------------------------------------

def test_stratified_split_is_deterministic_and_stratified():
    idx = [f"s{i}" for i in range(100)]
    groups = pd.Series(["normal"] * 60 + ["imd"] * 25 + ["other"] * 15,
                        index=idx)
    split1 = stratified_split(groups, seed=7, validation_fraction=0.5)
    split2 = stratified_split(groups, seed=7, validation_fraction=0.5)
    assert split1.equals(split2), "the split must be reproducible from the seed"
    # Stratification: each group keeps its proportion in the validation half.
    for g in ("normal", "imd", "other"):
        n_g = int((groups == g).sum())
        n_g_val = int(split1[groups == g].sum())
        assert abs(n_g_val - n_g / 2) <= 1
    # Every sample assigned exactly once (boolean series, no NaN).
    assert split1.dtype == bool
    assert not split1.isna().any()


def test_split_differs_for_different_seeds():
    idx = [f"s{i}" for i in range(50)]
    groups = pd.Series(["normal"] * 50, index=idx)
    a = stratified_split(groups, seed=1)
    b = stratified_split(groups, seed=2)
    assert not a.equals(b)


# ---------------------------------------------------------------------------
# Reference hygiene
# ---------------------------------------------------------------------------

def test_leave_one_out_hygiene_excludes_deeply_disturbed_normal():
    rng = np.random.default_rng(5)
    n = 30
    df = pd.DataFrame(
        rng.normal(5.0, 0.2, size=(n, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=[f"n{i}" for i in range(n)],
    )
    # One candidate normal is a gross outlier on f0 (z >> 10 vs peers).
    df.loc["n7", "f0"] = df["f0"].median() + 30 * 0.2
    mask = pd.Series(True, index=df.index)
    clean, report = leave_one_out_hygiene(df, mask, max_depth=10.0)
    assert not clean.loc["n7"]
    assert clean.drop("n7").all()
    assert report.loc[report["sample_id"] == "n7", "excluded"].iloc[0]
    assert report["loo_max_z"].max() > 10.0


def test_leave_one_out_hygiene_self_masking_is_prevented():
    """A disturbed normal cannot dilate the peer reference enough to hide."""
    rng = np.random.default_rng(6)
    n = 12
    df = pd.DataFrame(
        rng.normal(5.0, 0.1, size=(n, 3)),
        columns=["f0", "f1", "f2"],
        index=[f"n{i}" for i in range(n)],
    )
    # Two candidates are shifted far on every feature: with a plain
    # (non-LOO) reference they would inflate the IQR and mask themselves;
    # LOO they must be caught.
    shift = df.loc["n2":].median() + 3.0
    for sid in ("n0", "n1"):
        df.loc[sid, ["f0", "f1", "f2"]] = shift.to_numpy()
    assert df.loc[["n0", "n1"]].notna().all().all()
    mask = pd.Series(True, index=df.index)
    clean, report = leave_one_out_hygiene(df, mask, max_depth=10.0)
    assert not clean.loc[["n0", "n1"]].any()
    assert clean.drop(["n0", "n1"]).all()


def test_leave_one_out_hygiene_keeps_clean_reference():
    rng = np.random.default_rng(7)
    n = 40
    df = pd.DataFrame(
        rng.normal(5.0, 0.3, size=(n, 5)),
        columns=[f"f{i}" for i in range(5)],
        index=[f"n{i}" for i in range(n)],
    )
    mask = pd.Series(True, index=df.index)
    clean, report = leave_one_out_hygiene(df, mask, max_depth=10.0)
    assert clean.all()
    assert not report["excluded"].any()


# ---------------------------------------------------------------------------
# Development QC (label-blind checks)
# ---------------------------------------------------------------------------

def test_verify_calibration_flags_unstandardized_features():
    n = 200
    rng = np.random.default_rng(8)
    # Exactly standardized 'good' features: median 0, IQR 1 by construction,
    # so any flag on them is a false alarm of the check itself.
    good = pd.DataFrame({
        "good_a": rng.normal(0.0, 1.0, size=n),
        "good_b": rng.normal(0.0, 1.0, size=n),
    }, index=[f"s{i}" for i in range(n)])
    for col in ("good_a", "good_b"):
        med = good[col].median()
        iqr = good[col].quantile(0.75) - good[col].quantile(0.25)
        good[col] = (good[col] - med) / iqr
    bad = pd.DataFrame({"bad_shift": rng.normal(4.0, 1.0, size=n)},
                      index=[f"s{i}" for i in range(n)])
    z = pd.concat([good, bad], axis=1)
    mask = pd.Series(True, index=z.index)
    report = verify_calibration(z, mask)
    flagged = report[report["median_abs_z"] > 0.05]["feature"].tolist()
    assert "bad_shift" in flagged
    assert not set(["good_a", "good_b"]) & set(flagged)


def test_noise_floor_features_detects_spike_and_thin_iqr():
    rng = np.random.default_rng(9)
    n = 40
    df = pd.DataFrame({
        "clean": rng.normal(5.0, 0.4, size=n),
        "spiky": [4.2] * n,
    }, index=[f"s{i}" for i in range(n)])
    mask = pd.Series(True, index=df.index)
    report = noise_floor_features(df, mask)
    assert "spiky" in set(report["feature"])
    assert "clean" not in set(report["feature"])


def test_diverging_duplicates_flags_iqr_ratio():
    reference_stats = pd.DataFrame({
        "feature": ["twin_a", "twin_b", "solo"],
        "scale": [0.05, 0.30, 0.20],
    })
    feature_to_hmdb = pd.DataFrame([
        {"feature": "twin_a", "hmdb_id": "H1"},
        {"feature": "twin_b", "hmdb_id": "H1"},
        {"feature": "solo", "hmdb_id": "H2"},
    ])
    report = diverging_duplicates(reference_stats, feature_to_hmdb)
    assert set(report["hmdb_id"]) == {"H1"}
    assert "solo" not in set(report["hmdb_id"])


def test_redundant_pathways_finds_near_duplicates():
    feature_to_pathway = pd.DataFrame([
        {"feature": "f1", "hmdb_id": "H1", "smp_id": "SMP_A"},
        {"feature": "f2", "hmdb_id": "H2", "smp_id": "SMP_A"},
        {"feature": "f3", "hmdb_id": "H3", "smp_id": "SMP_A"},
        {"feature": "f1", "hmdb_id": "H1", "smp_id": "SMP_B"},
        {"feature": "f2", "hmdb_id": "H2", "smp_id": "SMP_B"},
        {"feature": "f3", "hmdb_id": "H3", "smp_id": "SMP_B"},
        {"feature": "f4", "hmdb_id": "H4", "smp_id": "SMP_C"},
        {"feature": "f5", "hmdb_id": "H5", "smp_id": "SMP_C"},
    ])
    flags = pd.DataFrame([{"smp_id": "SMP_A", "flagged": True},
                          {"smp_id": "SMP_B", "flagged": True}])
    report = redundant_pathways(flags, feature_to_pathway, min_jaccard=0.8)
    pairs = set(zip(report["pathway_a"], report["pathway_b"]))
    assert ("SMP_A", "SMP_B") in pairs
    assert all("SMP_C" not in p for p in pairs)


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def test_exact_binomial_ci_known_values():
    # For k=8, n=10: the 95% CI must bracket 0.8 and be wider than 0.1.
    lo, hi = exact_binomial_ci(8, 10)
    assert lo < 0.8 < hi
    assert 0.44 < lo < 0.52   # Clopper-Pearson for 8/10
    assert 0.94 < hi < 0.99
    # Degenerate cases.
    lo0, hi0 = exact_binomial_ci(0, 5)
    assert lo0 == 0.0
    assert 0.0 < hi0 < 1.0
    lo1, hi1 = exact_binomial_ci(5, 5)
    assert 0.0 < lo1 < 1.0
    assert hi1 == 1.0


def test_bootstrap_auc_separates_groups():
    rng = np.random.default_rng(10)
    labelled = pd.DataFrame({
        "score": list(rng.normal(0.0, 1.0, size=30))
                 + list(rng.normal(4.0, 1.0, size=20)),
        "group": ["normal"] * 30 + ["imd"] * 20,
    })
    auc, lo, hi = bootstrap_auc(labelled, "score", n_bootstrap=200)
    assert auc > 0.95
    assert lo <= auc <= hi


def test_summarize_evaluation_metrics_and_halves():
    idx = [f"n{i}" for i in range(20)] + [f"i{i}" for i in range(10)]
    decisions = pd.DataFrame({
        "sample_id": idx,
        "flagged": [False] * 18 + [True, True]      # 18/20 normals clean
                 + [True] * 8 + [False, False],      # 8/10 IMD flagged
        "top_excess": [0.1] * 18 + [2.0, 3.0] + [4.0] * 8 + [0.2, 0.3],
        "group": ["normal"] * 20 + ["imd"] * 10,
    })
    flags = pd.DataFrame({
        "sample_id": [f"i{i}" for i in range(8)],
        "smp_id": ["SMP_A"] * 8,
        "flagged": [True] * 8,
    })
    metrics = summarize_evaluation(decisions, flags, half="validation")
    assert metrics["sensitivity"] == pytest.approx(0.8)
    assert metrics["specificity"] == pytest.approx(0.9)
    assert metrics["sensitivity_n"] == 8
    assert metrics["specificity_n"] == 18
    # CI brackets the point estimates.
    assert metrics["sensitivity_ci"][0] <= 0.8 <= metrics["sensitivity_ci"][1]
    assert metrics["specificity_ci"][0] <= 0.9 <= metrics["specificity_ci"][1]
    assert metrics["auc"] > 0.9


def test_summarize_evaluation_respects_half_mask():
    idx = [f"n{i}" for i in range(10)] + [f"i{i}" for i in range(10)]
    decisions = pd.DataFrame({
        "sample_id": idx,
        "flagged": [True] * 20,
        "top_excess": [1.0] * 20,
        "group": ["normal"] * 10 + ["imd"] * 10,
    })
    flags = pd.DataFrame({
        "sample_id": idx, "smp_id": "SMP_A", "flagged": [True] * 20})
    half = pd.Series([True] * 5 + [False] * 5 + [True] * 5 + [False] * 5,
                     index=idx)
    metrics = summarize_evaluation(decisions, flags, half="validation",
                                  validation_mask=half)
    assert metrics["n_normal"] == 5
    assert metrics["n_imd"] == 5


def test_assign_groups_labels():
    meta = pd.DataFrame({
        "Classification": [0, 0, 1, 2],
        "Oordeel targeted": [0, 1, 1, 0],
    }, index=["a", "b", "c", "d"])
    groups = assign_groups(meta)
    assert groups.tolist() == ["normal", "other", "imd", "other"]


def test_leave_one_out_hygiene_exclusion_cap_stops_cascade():
    """The pre-declared exclusion cap stops a miscalibrated max_depth cascade."""
    rng = np.random.default_rng(12)
    n = 20
    df = pd.DataFrame(
        rng.normal(5.0, 0.1, size=(n, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=[f"n{i}" for i in range(n)],
    )
    # A far-gross outlier would, in iterative rounds, tighten the peer IQRs
    # and pull many borderline candidates past a too-low max_depth; the cap
    # must stop the cascade at the declared fraction.
    df.loc["n5", :] = df.median() + 50.0
    mask = pd.Series(True, index=df.index)
    clean, report = leave_one_out_hygiene(
        df, mask, max_depth=1.5, max_excluded_fraction=0.10)
    n_excluded = int(report["excluded"].sum())
    assert n_excluded <= 2  # 10% of 20 candidates
    assert int((~clean).sum()) == n_excluded
    # The deepest candidate is excluded first even under the cap.
    assert not clean.loc["n5"]
    # Without the cap the same settings would cascade further.
    clean_uncapped, report_uncapped = leave_one_out_hygiene(
        df, mask, max_depth=1.5)
    assert int(report_uncapped["excluded"].sum()) > n_excluded


def test_leave_one_out_hygiene_cap_disabled_by_default():
    """max_excluded_fraction=None runs the full iterative rounds."""
    rng = np.random.default_rng(13)
    n = 30
    df = pd.DataFrame(
        rng.normal(5.0, 0.3, size=(n, 4)),
        columns=[f"f{i}" for i in range(4)],
        index=[f"n{i}" for i in range(n)],
    )
    df.loc["n9", :] = df.median() + 30.0
    mask = pd.Series(True, index=df.index)
    clean, report = leave_one_out_hygiene(df, mask, max_depth=20.0)
    assert not clean.loc["n9"]
    assert int(report["excluded"].sum()) >= 1
