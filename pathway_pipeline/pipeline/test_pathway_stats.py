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
    compute_stouffer_scores,
    flag_pathway_scores,
    summarize_sample_flags,
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


# ---------------------------------------------------------------------------
# Stouffer scores
# ---------------------------------------------------------------------------

def _stouffer_inputs():
    samples = ["s1", "s2", "s3", "s4"]
    zscores = pd.DataFrame({
        "f1": [1.0, 0.0, 0.5, 0.0],
        "f2": [1.0, 0.0, 0.5, 0.0],
        "f3": [1.0, 0.0, 0.5, 0.0],
        "f4": [np.nan, 0.0, 0.0, 0.0],
    }, index=samples)
    feature_to_pathway = pd.DataFrame([
        {"feature": "f1", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f2", "hmdb_id": "H2", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M2", "metabolite_name": "m2"},
        {"feature": "f3", "hmdb_id": "H3", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M3", "metabolite_name": "m3"},
        # f4 is a second feature for metabolite H1 (dedup: averaged with f1).
        {"feature": "f4", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "Alpha Pathway", "metabolite_id": "M1", "metabolite_name": "m1"},
    ])
    scored_coverage = pd.DataFrame([
        {"smp_id": "SMP1", "pathway_name": "Alpha Pathway", "n_metabolites": 3,
         "n_matched_metabolites": 3, "matched_metabolites": "H1;H2;H3",
         "n_matched_features": 4, "matched_features": "f1;f2;f3;f4", "coverage": 0.75},
    ])
    normal_mask = pd.Series([True, True, False, False], index=samples)
    return zscores, feature_to_pathway, scored_coverage, normal_mask


def test_stouffer_scores_and_reference():
    zscores, links, coverage, normal_mask = _stouffer_inputs()
    scores, reference = compute_stouffer_scores(zscores, links, coverage, normal_mask,
                                               min_metabolites=3)

    assert set(scores["smp_id"]) == {"SMP1"}
    assert scores["sample_id"].nunique() == 4

    s1 = scores[scores["sample_id"] == "s1"].iloc[0]
    # s1: H1 = mean(f1, f4) = mean(1.0, nan) = 1.0; H2 = 1.0; H3 = 1.0
    # signed = 3/sqrt(3); abs same (all positive)
    assert s1["n_metabolites_used"] == 3
    assert s1["z_stouffer"] == pytest.approx(3.0 / np.sqrt(3))
    assert s1["z_stouffer_abs"] == pytest.approx(3.0 / np.sqrt(3))

    # s2: all zeros -> scores 0.
    s2 = scores[scores["sample_id"] == "s2"].iloc[0]
    assert s2["z_stouffer"] == pytest.approx(0.0)
    assert s2["z_stouffer_abs"] == pytest.approx(0.0)

    # Reference percentiles come from the two normal samples (s1, s2).
    row = reference.iloc[0]
    assert row["n_metabolites"] == 3
    assert row["normal_z_stouffer_abs_p50"] == pytest.approx(
        np.mean([3.0 / np.sqrt(3), 0.0]))


def test_stouffer_min_metabolites_nan():
    zscores, links, coverage, normal_mask = _stouffer_inputs()
    # Only f4 present for H1 -> 1 usable metabolite in s1 -> NaN, not shrunken.
    zscores2 = zscores.drop(columns=["f2", "f3"])
    scores, reference = compute_stouffer_scores(zscores2, links, coverage,
                                                normal_mask, min_metabolites=3)
    assert scores.empty  # no pathway reaches 3 metabolites at all


def test_stouffer_signed_vs_absolute():
    samples = ["s1", "s2"]
    zscores = pd.DataFrame({
        "f1": [2.0, -2.0],
        "f2": [-2.0, -2.0],
        "f3": [2.0, -2.0],
    }, index=samples)
    links = pd.DataFrame([
        {"feature": "f1", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f2", "hmdb_id": "H2", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M2", "metabolite_name": "m2"},
        {"feature": "f3", "hmdb_id": "H3", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M3", "metabolite_name": "m3"},
    ])
    coverage = pd.DataFrame([
        {"smp_id": "SMP1", "pathway_name": "P", "n_metabolites": 3,
         "n_matched_metabolites": 3, "matched_metabolites": "H1;H2;H3",
         "n_matched_features": 3, "matched_features": "f1;f2;f3", "coverage": 1.0},
    ])
    normal_mask = pd.Series([True, False], index=samples)
    scores, _ = compute_stouffer_scores(zscores, links, coverage, normal_mask,
                                        min_metabolites=3)
    s1 = scores[scores["sample_id"] == "s1"].iloc[0]
    # Mixed directions: signed cancels, absolute does not.
    assert s1["z_stouffer"] == pytest.approx(2.0 / np.sqrt(3))
    assert s1["z_stouffer_abs"] == pytest.approx(6.0 / np.sqrt(3))


# ---------------------------------------------------------------------------
# Flagging
# ---------------------------------------------------------------------------

def _flags_inputs(n_normal=20):
    """Synthetic pathway scores: two pathways, n_normal normal samples with
    well-behaved absolute scores, plus two IMD samples with extreme ones."""
    rng = np.random.default_rng(7)
    samples = [f"n{i}" for i in range(n_normal)] + ["imd1", "imd2"]
    normal_mask = pd.Series([s.startswith("n") for s in samples], index=samples)
    rows = []
    for smp in ("SMP_A", "SMP_B"):
        for s in samples:
            base = rng.uniform(0.1, 2.0)
            if s.startswith("n"):
                z_abs = float(base)
            elif smp == "SMP_A":
                z_abs = 6.0 if s == "imd1" else 1.0
            else:
                z_abs = 1.0 if s == "imd1" else 6.0
            rows.append({
                "sample_id": s, "smp_id": smp,
                "pathway_name": {"SMP_A": "Pathway A", "SMP_B": "Pathway B"}[smp],
                "n_metabolites_used": 5,
                "z_stouffer": z_abs if smp == "SMP_A" else -z_abs,
                "z_stouffer_abs": z_abs,
            })
    return pd.DataFrame(rows), normal_mask


def test_flag_pathway_scores_threshold_from_normals_only():
    scores, normal_mask = _flags_inputs()
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=99.0)

    assert list(flags.columns[-3:]) == ["threshold", "excess", "flagged"]
    # Thresholds come from the normals' distribution (uniform 0.1-2.0).
    for smp in ("SMP_A", "SMP_B"):
        thr = flags.loc[flags["smp_id"] == smp, "threshold"].iloc[0]
        assert 0.1 <= thr <= 2.0
    # The 6.0 scores exceed every threshold; the 1.0 scores never do.
    extreme = flags[flags["z_stouffer_abs"] == 6.0]
    assert extreme["flagged"].all()
    mild = flags[flags["z_stouffer_abs"] == 1.0]
    assert not mild["flagged"].any()
    # Empirical calibration: the very top of the normal distribution can
    # exceed the interpolated percentile, but only a small tail does
    # (<= 10% of normal rows at the 99th percentile).
    normals = flags[~flags["sample_id"].str.startswith("imd")]
    assert normals["flagged"].mean() <= 0.10


def test_flag_pathway_scores_excess_and_empty():
    scores, normal_mask = _flags_inputs(n_normal=4)
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=90.0)
    row = flags[flags["sample_id"] == "imd1"].iloc[0]
    assert row["excess"] == pytest.approx(row["z_stouffer_abs"] / row["threshold"])

    empty = flag_pathway_scores(pd.DataFrame(), normal_mask)
    assert empty.empty
    assert "flagged" in empty.columns


def test_flag_pathway_scores_nan_scores():
    """NaN z_stouffer_abs rows (insufficient metabolites) are never flagged."""
    scores, normal_mask = _flags_inputs(n_normal=10)
    scores.loc[scores["sample_id"] == "imd1", "z_stouffer_abs"] = np.nan
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=99.0)
    imd1 = flags[flags["sample_id"] == "imd1"]
    # NaN scores: excess is NaN, flagged is False.
    assert not bool(imd1["flagged"].any())


def test_summarize_sample_flags_basic():
    scores, normal_mask = _flags_inputs()
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=99.0)
    summary = summarize_sample_flags(flags, min_flagged_pathways=1)

    assert set(summary.columns) == {
        "sample_id", "n_flagged_pathways", "n_scored_pathways", "flagged",
        "top_pathway_name", "top_z_stouffer_abs", "top_excess"}

    imd1 = summary[summary["sample_id"] == "imd1"].iloc[0]
    # imd1: flagged in A only (6.0); its 1.0 in B stays below the threshold.
    assert imd1["n_flagged_pathways"] == 1
    assert imd1["n_scored_pathways"] == 2
    assert bool(imd1["flagged"])
    assert imd1["top_pathway_name"] == "Pathway A"
    assert imd1["top_z_stouffer_abs"] == pytest.approx(6.0)

    normals = summary[summary["sample_id"].str.startswith("n")]
    # Small normal tail can flag (interpolated percentile) but stays rare.
    assert normals["n_flagged_pathways"].max() <= 1
    assert normals["flagged"].mean() <= 0.10


def test_summarize_sample_flags_min_flagged_pathways():
    scores, normal_mask = _flags_inputs()
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=99.0)

    strict = summarize_sample_flags(flags, min_flagged_pathways=2)
    for s in ("imd1", "imd2"):
        row = strict[strict["sample_id"] == s].iloc[0]
        # Each IMD sample is flagged in exactly one pathway -> min 2 rejects.
        assert row["n_flagged_pathways"] == 1
        assert not bool(row["flagged"])

    loose = summarize_sample_flags(flags, min_flagged_pathways=1)
    flagged = loose[loose["sample_id"].str.startswith("imd")]
    assert flagged["flagged"].all()


def test_summarize_sample_flags_empty_and_all_nan():
    empty = summarize_sample_flags(pd.DataFrame(), min_flagged_pathways=1)
    assert empty.empty
    assert "flagged" in empty.columns

    scores, normal_mask = _flags_inputs(n_normal=5)
    scores["z_stouffer_abs"] = np.nan
    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=99.0)
    summary = summarize_sample_flags(flags, min_flagged_pathways=1)
    # All-NaN: no flags, top evidence columns are NaN-safe.
    assert (summary["n_flagged_pathways"] == 0).all()
    assert (~summary["flagged"].astype(bool)).all()
    assert summary["top_excess"].isna().all()


# ---------------------------------------------------------------------------
# Duplicate sample IDs (regression: real-data failure in STEP 7)
# ---------------------------------------------------------------------------

def test_stouffer_and_flagging_with_duplicate_sample_ids():
    """Duplicated sample IDs must not break scalar lookups in the loops."""
    samples = ["n1", "n2", "n3", "imd1", "imd1", "n4"]
    zscores = pd.DataFrame({
        "f1": [0.1, 0.2, -0.1, 4.0, 4.1, 0.0],
        "f2": [0.0, -0.2, 0.1, 3.8, 3.9, 0.3],
        "f3": [-0.1, 0.1, 0.0, 4.2, 4.0, -0.2],
    }, index=samples)
    links = pd.DataFrame([
        {"feature": f"f{i}", "hmdb_id": f"H{i}", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": f"M{i}", "metabolite_name": f"m{i}"}
        for i in (1, 2, 3)
    ])
    coverage = pd.DataFrame([{
        "smp_id": "SMP1", "pathway_name": "P", "n_metabolites": 3,
        "n_matched_metabolites": 3, "matched_metabolites": "H1;H2;H3",
        "n_matched_features": 3, "matched_features": "f1;f2;f3", "coverage": 1.0,
    }])
    normal_mask = pd.Series([True, True, True, False, False, True], index=samples)

    scores, reference = compute_stouffer_scores(zscores, links, coverage,
                                                normal_mask, min_metabolites=3)
    assert len(scores) == 6  # every row kept, duplicates included
    # Normal |sum(z)| values: 0.2, 0.5, 0.2, 0.5 -> p50 of 0.2/sqrt(3) and
    # 0.5/sqrt(3) interpolated = 0.35/sqrt(3).
    assert reference["normal_z_stouffer_abs_p50"].iloc[0] == pytest.approx(
        0.35 / np.sqrt(3), abs=1e-9)

    flags = flag_pathway_scores(scores, normal_mask, threshold_percentile=50.0)
    assert len(flags) == 6
    # Both imd1 rows exceed the normals' p50 threshold.
    assert flags[flags["sample_id"] == "imd1"]["flagged"].all()

    summary = summarize_sample_flags(flags, min_flagged_pathways=1)
    # Duplicated imd1 rows land in one decision row (grouped by sample_id).
    assert len(summary) == 5  # n1, n2, n3, imd1 (merged), n4
    imd_row = summary[summary["sample_id"] == "imd1"].iloc[0]
    assert int(imd_row["n_scored_pathways"]) == 2  # both duplicate rows counted
    assert bool(imd_row["flagged"])
