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

from pathway_pipeline.main import load_feature_matrix
from pathway_pipeline.pipeline.pathway_stats import (
    _binomial_sf,
    classify_samples,
    flag_metabolite_scores,
    summarize_metabolite_flags,
    compute_metabolite_zscores,
    filter_pathways_for_scoring,
    prune_redundant_pathways,
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


def _scored_row(smp_id, name, metabolites):
    metabolites = sorted(metabolites)
    return {"smp_id": smp_id, "pathway_name": name,
            "n_metabolites": len(metabolites),
            "n_matched_metabolites": len(metabolites),
            "matched_metabolites": ";".join(metabolites),
            "n_matched_features": len(metabolites),
            "matched_features": ";".join(metabolites),
            "coverage": 1.0}


def test_prune_redundant_general_beats_disease_cartoon():
    metabolites = ["H1", "H2", "H3", "H4"]
    coverage = pd.DataFrame([
        _scored_row("SMP1", "Urea Cycle", metabolites),
        _scored_row("SMP2", "Argininosuccinic Aciduria", metabolites),
        _scored_row("SMP3", "Citrullinemia Type I Disease", metabolites),
    ])
    pruned, dropped = prune_redundant_pathways(coverage, min_jaccard=0.8)
    assert set(pruned["smp_id"]) == {"SMP1"}
    assert set(dropped["smp_id"]) == {"SMP2", "SMP3"}
    assert (dropped["represented_by"] == "SMP1").all()


def test_prune_redundant_larger_set_breaks_ties():
    coverage = pd.DataFrame([
        _scored_row("SMP1", "Beta Deficiency", ["H1", "H2", "H3"]),
        _scored_row("SMP2", "Alpha Deficiency", ["H1", "H2", "H3", "H4"]),
    ])
    # Jaccard 3/4 = 0.75 < 0.8 -> both kept at the default threshold.
    pruned, dropped = prune_redundant_pathways(coverage, min_jaccard=0.8)
    assert set(pruned["smp_id"]) == {"SMP1", "SMP2"}
    assert dropped.empty

    # At 0.7 the group collapses and the larger scored set wins the tie.
    pruned, dropped = prune_redundant_pathways(coverage, min_jaccard=0.7)
    assert set(pruned["smp_id"]) == {"SMP2"}
    assert dropped["smp_id"].tolist() == ["SMP1"]
    assert dropped["represented_by"].tolist() == ["SMP2"]


def test_prune_redundant_jaccard_threshold_respected():
    coverage = pd.DataFrame([
        _scored_row("SMP1", "A Pathway", ["H1", "H2", "H3"]),
        _scored_row("SMP2", "B Pathway", ["H1", "H2", "H4"]),
        # Jaccard 3/5 = 0.6 < 0.8 -> both kept.
        _scored_row("SMP3", "C Pathway", ["H1", "H3", "H4", "H5"]),
    ])
    pruned, dropped = prune_redundant_pathways(coverage, min_jaccard=0.8)
    # SMP1 vs SMP2: Jaccard 2/4 = 0.5 -> kept; all pairs below threshold.
    assert set(pruned["smp_id"]) == {"SMP1", "SMP2", "SMP3"}
    assert dropped.empty


def test_prune_redundant_connected_components_collapse_transitively():
    # A-B overlap 0.9, B-C overlap 0.9, A-C overlap 0.5 -> one group of three.
    coverage = pd.DataFrame([
        _scored_row("SMPA", "Alpha Pathway",
                    ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H10"]),
        _scored_row("SMPB", "Beta Pathway",
                    ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H9", "H11"]),
        _scored_row("SMPC", "Gamma Pathway",
                    ["H1", "H2", "H3", "H4", "H5", "H6", "H7", "H8", "H11", "H12"]),
    ])
    pruned, dropped = prune_redundant_pathways(coverage, min_jaccard=0.8)
    assert len(pruned) == 1
    assert len(dropped) == 2
    assert (dropped["represented_by"] == pruned["smp_id"].iloc[0]).all()


def test_prune_redundant_deterministic_and_preserves_rows():
    coverage = pd.DataFrame([
        _scored_row("SMP1", "Urea Cycle", ["H1", "H2", "H3"]),
        _scored_row("SMP2", "Argininosuccinic Aciduria", ["H1", "H2", "H3"]),
        _scored_row("SMP3", "Unrelated Pathway", ["H4", "H5", "H6"]),
    ])
    pruned1, dropped1 = prune_redundant_pathways(coverage)
    pruned2, dropped2 = prune_redundant_pathways(coverage)
    pd.testing.assert_frame_equal(pruned1, pruned2)
    pd.testing.assert_frame_equal(dropped1, dropped2)
    assert set(pruned1["smp_id"]) == {"SMP1", "SMP3"}
    assert list(pruned1.columns) == list(coverage.columns)


def test_prune_redundant_empty_input():
    empty = pd.DataFrame(columns=["smp_id", "pathway_name", "n_metabolites",
                                  "n_matched_metabolites", "matched_metabolites",
                                  "n_matched_features", "matched_features", "coverage"])
    pruned, dropped = prune_redundant_pathways(empty)
    assert pruned.empty
    assert dropped.empty
    assert list(dropped.columns) == ["smp_id", "pathway_name", "represented_by"]


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
        "sample_p_value", "top_pathway_name", "top_z_stouffer_abs",
        "top_excess"}

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
    # Duplicate imd1 rows merge into one decision row (grouped by sample_id).
    assert len(summary) == 5  # n1, n2, n3, imd1 (merged), n4
    imd_row = summary[summary["sample_id"] == "imd1"].iloc[0]
    assert int(imd_row["n_scored_pathways"]) == 2  # both duplicate rows counted
    assert bool(imd_row["flagged"])


def test_load_feature_matrix_drops_duplicate_rows(tmp_path):
    """True duplicates: only the first occurrence of each sample ID is kept."""
    df = pd.DataFrame({
        "Classification": [0, 0, 1, 1],
        "Oordeel targeted": [0, 0, 1, 1],
        "A": [1.0, 1.0, 9.0, 8.8],
    }, index=["s1", "s1", "imd1", "imd1"])
    csv = tmp_path / "dups.csv"
    df.to_csv(csv)

    features, metadata, ages = load_feature_matrix(
        str(csv), non_feature_columns=["Oordeel targeted", "Classification"])
    assert features.index.tolist() == ["s1", "imd1"]
    assert features["A"].tolist() == [1.0, 9.0]  # first occurrence kept
    assert metadata.index.tolist() == ["s1", "imd1"]


# ---------------------------------------------------------------------------
# Label-NaN filter at load time
# ---------------------------------------------------------------------------

def test_load_feature_matrix_drops_unlabeled_samples(tmp_path):
    df = pd.DataFrame({
        "Classification": [0, 0, None, 1, 0],
        "Oordeel targeted": [0, None, 0, 1, 0],
        "A": [1.0, 2.0, 3.0, 9.0, 2.5],
        "B": [1.1, 2.2, 3.3, 9.9, 2.6],
    }, index=["s1", "s2", "s3", "s4", "s5"])
    csv = tmp_path / "input.csv"
    df.to_csv(csv)

    features, metadata, ages = load_feature_matrix(
        str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
    )
    # s2 (NaN oordeel) and s3 (NaN classification) are dropped.
    assert features.index.tolist() == ["s1", "s4", "s5"]
    assert metadata.index.tolist() == ["s1", "s4", "s5"]
    assert features.columns.tolist() == ["A", "B"]
    assert metadata["Classification"].tolist() == [0, 1, 0]



# ---------------------------------------------------------------------------
# Binomial sample rule and z-cap
# ---------------------------------------------------------------------------

def test_binomial_sf_known_values():
    # P(X >= 2) for Binomial(4, 0.25) = 1 - P(0) - P(1)
    expected = 1.0 - 0.75 ** 4 - 4 * 0.25 * 0.75 ** 3
    assert _binomial_sf(2, 4, 0.25) == pytest.approx(expected)
    assert _binomial_sf(0, 4, 0.25) == 1.0
    assert _binomial_sf(5, 4, 0.25) == 0.0
    assert _binomial_sf(1, 234, 0.01) == pytest.approx(
        1.0 - 0.99 ** 234, abs=1e-6)


def _many_pathway_flags(n_pathways=234, n_noise=20):
    """Fixture: noise samples flag 2 pathways each, one sample flags 20."""
    rows = []
    for i in range(n_noise):
        flagged_set = {i % n_pathways, (i + 117) % n_pathways}
        for pw in range(n_pathways):
            flagged = pw in flagged_set
            rows.append({
                "sample_id": f"noise_{i}", "smp_id": f"SMP{pw}",
                "pathway_name": f"P{pw}", "n_metabolites_used": 5,
                "z_stouffer": 1.0, "z_stouffer_abs": 1.0,
                "threshold": 0.9,
                "excess": 1.5 if flagged else 0.5,
                "flagged": flagged,
            })
    for pw in range(20):
        rows.append({
            "sample_id": "shifted", "smp_id": f"SMP{pw}",
            "pathway_name": f"P{pw}", "n_metabolites_used": 5,
            "z_stouffer": 1.5, "z_stouffer_abs": 1.5,
            "threshold": 0.9, "excess": 1.7, "flagged": True,
        })
    return pd.DataFrame(rows)


def test_summarize_sample_flags_binomial_rule():
    """A sample flagging 2 of 234 pathways at p99 is chance, not a decision."""
    flags = _many_pathway_flags()

    without_rule = summarize_sample_flags(flags, min_flagged_pathways=1,
                                          per_pathway_flag_rate=None,
                                          sample_rule="none")
    with_rule = summarize_sample_flags(flags, min_flagged_pathways=1,
                                       per_pathway_flag_rate=0.01,
                                       max_sample_p=0.05,
                                       sample_rule="binomial")

    assert without_rule["flagged"].sum() == 21
    assert not with_rule.loc[
        with_rule["sample_id"].str.startswith("noise"), "flagged"].any()
    shifted = with_rule[with_rule["sample_id"] == "shifted"].iloc[0]
    assert bool(shifted["flagged"])
    assert shifted["sample_p_value"] == pytest.approx(
        _binomial_sf(20, 234, 0.01), rel=1e-9)
    assert np.isnan(without_rule["sample_p_value"].iloc[0])


def test_summarize_sample_flags_empirical_rule():
    """The empirical null absorbs correlated flags among normals."""
    flags = _many_pathway_flags()
    correlated = flags.copy()
    correlated.loc[correlated["sample_id"].str.startswith("noise")
                   & correlated["flagged"], "flagged"] = False
    for i in range(20):
        for pw in range(10, 18):
            sel = ((correlated["sample_id"] == f"noise_{i}")
                   & (correlated["smp_id"] == f"SMP{pw}"))
            correlated.loc[sel, "flagged"] = True
            correlated.loc[sel, "excess"] = 1.4

    normal_mask = pd.Series(
        [s.startswith("noise") for s in correlated["sample_id"].unique()],
        index=correlated["sample_id"].unique())

    empirical = summarize_sample_flags(correlated, min_flagged_pathways=1,
                                       max_sample_p=0.05,
                                       normal_mask=normal_mask,
                                       sample_rule="empirical")
    binomial = summarize_sample_flags(correlated, min_flagged_pathways=1,
                                      per_pathway_flag_rate=0.01,
                                      max_sample_p=0.05,
                                      normal_mask=normal_mask,
                                      sample_rule="binomial")

    # Every normal flags the same 8 pathways: binomial p(8 of 234 @1%) ~
    # 0.006 flags them, but the empirical p is 1.0 (all normals do it).
    assert not empirical.loc[
        empirical["sample_id"].str.startswith("noise"),
        "flagged"].any()
    assert binomial.loc[
        binomial["sample_id"].str.startswith("noise"),
        "flagged"].all()
    shifted = empirical[empirical["sample_id"] == "shifted"].iloc[0]
    assert bool(shifted["flagged"])







def _depth_vs_breadth_flags(n_pathways=40, n_breadth=20):
    """Breadth normals flag many pathways at low excess; a depth sample flags
    few pathways at high excess."""
    rows = []
    for i in range(n_breadth):
        for pw in range(12):
            rows.append({
                "sample_id": f"breadth_{i}", "smp_id": f"SMP{pw}",
                "pathway_name": f"P{pw}", "n_metabolites_used": 5,
                "z_stouffer": 1.0, "z_stouffer_abs": 1.0,
                "threshold": 0.9, "excess": 1.3, "flagged": True,
            })
        for pw in range(12, n_pathways):
            rows.append({
                "sample_id": f"breadth_{i}", "smp_id": f"SMP{pw}",
                "pathway_name": f"P{pw}", "n_metabolites_used": 5,
                "z_stouffer": 0.5, "z_stouffer_abs": 0.5,
                "threshold": 0.9, "excess": 0.5, "flagged": False,
            })
    for pw in (3, 7):
        rows.append({
            "sample_id": "depth", "smp_id": f"SMP{pw}",
            "pathway_name": f"P{pw}", "n_metabolites_used": 5,
            "z_stouffer": 4.0, "z_stouffer_abs": 4.0,
            "threshold": 0.9, "excess": 4.5, "flagged": True,
        })
    return pd.DataFrame(rows)


def test_summarize_sample_flags_max_excess_rule():
    """The depth rule flags profound few-pathway shifts, not mild breadth."""
    flags = _depth_vs_breadth_flags()
    normal_mask = pd.Series(
        [True] * 20 + [False],
        index=[f"breadth_{i}" for i in range(20)] + ["depth"])

    count_rule = summarize_sample_flags(flags, min_flagged_pathways=1,
                                        normal_mask=normal_mask,
                                        sample_rule="empirical")
    depth_rule = summarize_sample_flags(flags, min_flagged_pathways=1,
                                        normal_mask=normal_mask,
                                        sample_rule="max_excess")

    # The count rule is blind both ways: the breadth normals ARE the null,
    # and the depth sample flags fewer pathways than all of them (p = 1.0).
    assert not count_rule["flagged"].any()
    depth_count = count_rule.loc[count_rule["sample_id"] == "depth"].iloc[0]
    assert depth_count["sample_p_value"] == pytest.approx(1.0)
    # The depth rule flags the depth sample; no breadth normal clears it.
    assert bool(depth_rule.loc[depth_rule["sample_id"] == "depth",
                               "flagged"].iloc[0])
    assert not depth_rule.loc[
        depth_rule["sample_id"].str.startswith("breadth"), "flagged"].any()
    depth_row = depth_rule.loc[depth_rule["sample_id"] == "depth"].iloc[0]
    assert depth_row["sample_p_value"] == 0.0
    assert depth_row["top_excess"] == pytest.approx(4.5)


def test_flag_metabolite_scores_per_metabolite_thresholds():
    """Each metabolite gets its own normal-calibrated threshold."""
    rng = np.random.default_rng(3)
    normals = [f"n{i}" for i in range(20)]
    zscores = pd.DataFrame({
        "m1": np.concatenate([np.abs(rng.normal(0, 0.5, 20)), [12.0]]),
        "m2": np.concatenate([np.abs(rng.normal(0, 0.2, 20)), [0.25]]),
    }, index=normals + ["p1"])
    normal_mask = pd.Series([True] * 20 + [False], index=zscores.index)
    flags = flag_metabolite_scores(zscores, normal_mask,
                                   threshold_percentile=99.0)
    # m1: noisy metabolite; the patient at |z| 12 flags with a huge excess
    m1 = flags[flags["metabolite"] == "m1"]
    p1_m1 = m1[m1["sample_id"] == "p1"].iloc[0]
    assert bool(p1_m1["flagged"])
    assert p1_m1["excess"] > 5.0
    # with 20 normals the p99 sits just below the sample maximum, so at
    # most the single most extreme normal can cross it
    normal_m1 = m1[m1["sample_id"] != "p1"]
    assert int(normal_m1["flagged"].sum()) <= 1
    # m2: quiet metabolite; the patient's mild value stays under its p99
    m2 = flags[flags["metabolite"] == "m2"]
    p1_m2 = m2[m2["sample_id"] == "p1"].iloc[0]
    assert not bool(p1_m2["flagged"])
    # thresholds differ per metabolite (noisy m1 gets a wider range)
    t1 = m1["threshold"].iloc[0]
    t2 = m2["threshold"].iloc[0]
    assert t1 > t2


def test_summarize_metabolite_flags_depth():
    """One grossly elevated metabolite beats the normals' max-|z| null."""
    rows = []
    for i in range(10):
        for m in ("m1", "m2", "m3"):
            rows.append({"sample_id": f"n{i}", "metabolite": m,
                         "abs_z": 0.5 + 0.1 * i, "threshold": 1.0,
                         "excess": 0.5 + 0.1 * i, "flagged": False})
    # depth sample: one metabolite at |z| 12, rest quiet
    rows += [
        {"sample_id": "depth", "metabolite": "m1", "abs_z": 12.0,
         "threshold": 1.0, "excess": 12.0, "flagged": True},
        {"sample_id": "depth", "metabolite": "m2", "abs_z": 0.4,
         "threshold": 1.0, "excess": 0.4, "flagged": False},
        {"sample_id": "depth", "metabolite": "m3", "abs_z": 0.3,
         "threshold": 1.0, "excess": 0.3, "flagged": False},
    ]
    flags = pd.DataFrame(rows)
    normal_mask = pd.Series([True] * 10 + [False],
                            index=[f"n{i}" for i in range(10)] + ["depth"])
    summary = summarize_metabolite_flags(flags, normal_mask)
    depth_row = summary[summary["sample_id"] == "depth"].iloc[0]
    assert depth_row["max_metabolite_z"] == pytest.approx(12.0)
    assert depth_row["n_flagged_metabolites"] == 1
    assert depth_row["metabolite_depth_p"] == 0.0
    assert depth_row["top_metabolite"] == "m1"
    # no normal is flagged at sample level (their max |z| ~1.4 < null)
    normals = summary[summary["sample_id"].str.startswith("n")]
    assert (normals["metabolite_depth_p"] >= 0.1).all()


def test_stouffer_max_abs_z_cap():
    """A single extreme feature cannot dominate the pathway sum."""
    samples = ["s1", "s2"]
    zscores = pd.DataFrame({
        "f1": [20.0, 0.1],
        "f2": [0.2, 0.2],
        "f3": [0.1, 0.1],
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
    normal_mask = pd.Series([True, False], index=samples)

    capped, _ = compute_stouffer_scores(zscores, links, coverage, normal_mask,
                                        min_metabolites=3, max_abs_z=10.0)
    uncapped, _ = compute_stouffer_scores(zscores, links, coverage, normal_mask,
                                          min_metabolites=3, max_abs_z=None)
    s1_capped = capped[capped["sample_id"] == "s1"].iloc[0]["z_stouffer"]
    s1_uncapped = uncapped[uncapped["sample_id"] == "s1"].iloc[0]["z_stouffer"]
    assert s1_capped == pytest.approx((10.0 + 0.2 + 0.1) / np.sqrt(3))
    assert s1_uncapped == pytest.approx((20.0 + 0.2 + 0.1) / np.sqrt(3))
    assert s1_capped < s1_uncapped


# ---------------------------------------------------------------------------
# Noise-floor scale floor, scale^2 metabolite weighting, demoted features
# ---------------------------------------------------------------------------

def test_compute_metabolite_zscores_scale_floor():
    """Features with a reference IQR below the floor are dropped."""
    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "wide": rng.normal(5.0, 0.5, size=n),
        "thin": rng.normal(5.0, 0.01, size=n),  # IQR ~ 0.013 < 0.08
    }, index=[f"s{i}" for i in range(n)])
    mask = pd.Series([True] * n, index=df.index)
    zscores, reference_stats, dropped = compute_metabolite_zscores(
        df, mask, iqr_scale=True, min_reference_scale=0.08)
    assert list(dropped.loc[dropped["feature"] == "thin", "reason"]) == ["small_scale"]
    assert list(zscores.columns) == ["wide"]
    assert set(reference_stats["feature"]) == {"wide"}
    # Floor disabled keeps the thin feature.
    zscores2, _, dropped2 = compute_metabolite_zscores(
        df, mask, iqr_scale=True, min_reference_scale=None)
    assert list(zscores2.columns) == ["wide", "thin"]
    assert dropped2.empty


def test_stouffer_scale_squared_weights_duplicate_features():
    """A razor-thin duplicate feature cannot dominate the metabolite's z:
    scale^2 weights make the wide feature carry the metabolite."""
    samples = ["s1", "s2"]
    zscores = pd.DataFrame({
        "f_wide": [1.0, -1.0],
        "f_thin": [15.0, -15.0],  # noise-inflated twin
        "f3": [0.5, -0.5],
        "f4": [-0.5, 0.5],
    }, index=samples)
    links = pd.DataFrame([
        {"feature": "f_wide", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f_thin", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f3", "hmdb_id": "H2", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M2", "metabolite_name": "m2"},
        {"feature": "f4", "hmdb_id": "H3", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M3", "metabolite_name": "m3"},
    ])
    coverage = pd.DataFrame([{
        "smp_id": "SMP1", "pathway_name": "P", "n_metabolites": 3,
        "n_matched_metabolites": 3, "matched_metabolites": "H1;H2;H3",
        "n_matched_features": 4, "matched_features": "f_wide;f_thin;f3;f4",
        "coverage": 1.0,
    }])
    normal_mask = pd.Series([True, False], index=samples)
    weights = {"f_wide": 0.3 ** 2, "f_thin": 0.05 ** 2, "f3": 1.0, "f4": 1.0}
    weighted, _ = compute_stouffer_scores(
        zscores, links, coverage, normal_mask, min_metabolites=3,
        feature_scale_weights=weights)
    plain, _ = compute_stouffer_scores(
        zscores, links, coverage, normal_mask, min_metabolites=3)
    s1_w = weighted[weighted["sample_id"] == "s1"].iloc[0]["z_stouffer"]
    s1_p = plain[plain["sample_id"] == "s1"].iloc[0]["z_stouffer"]
    # Weighted H1 z is dominated by the wide feature (weight ratio 36:1).
    h1_weighted = (0.09 * 1.0 + 0.0025 * 15.0) / (0.09 + 0.0025)
    assert s1_w == pytest.approx((h1_weighted + 0.5 - 0.5) / np.sqrt(3))
    # Plain mean drags H1 halfway to the noise twin.
    h1_plain = (1.0 + 15.0) / 2
    assert s1_p == pytest.approx((h1_plain + 0.5 - 0.5) / np.sqrt(3))
    assert abs(s1_w) < abs(s1_p)


def test_stouffer_weighted_handles_nan_in_one_duplicate():
    """A missing value in one duplicate uses the remaining ones only."""
    zscores = pd.DataFrame({
        "f_a": [1.0, 2.0],
        "f_b": [np.nan, -2.0],
        "f3": [0.5, 0.5],
        "f4": [0.5, 0.5],
    }, index=["s1", "s2"])
    links = pd.DataFrame([
        {"feature": "f_a", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f_b", "hmdb_id": "H1", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M1", "metabolite_name": "m1"},
        {"feature": "f3", "hmdb_id": "H2", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M2", "metabolite_name": "m2"},
        {"feature": "f4", "hmdb_id": "H3", "smp_id": "SMP1",
         "pathway_name": "P", "metabolite_id": "M3", "metabolite_name": "m3"},
    ])
    coverage = pd.DataFrame([{
        "smp_id": "SMP1", "pathway_name": "P", "n_metabolites": 3,
        "n_matched_metabolites": 3, "matched_metabolites": "H1;H2;H3",
        "n_matched_features": 4, "matched_features": "f_a;f_b;f3;f4",
        "coverage": 1.0,
    }])
    normal_mask = pd.Series([True, False], index=zscores.index)
    scores, _ = compute_stouffer_scores(
        zscores, links, coverage, normal_mask, min_metabolites=3,
        feature_scale_weights={"f_a": 1.0, "f_b": 1.0})
    s1 = scores[scores["sample_id"] == "s1"].iloc[0]
    # s1's H1 comes only from f_a (f_b is NaN): z = 1.0.
    assert s1["n_metabolites_used"] == 3
    assert s1["z_stouffer"] == pytest.approx((1.0 + 0.5 + 0.5) / np.sqrt(3))


def test_flag_metabolite_scores_ignores_demoted_features():
    """Demoted artifact features never produce metabolite flags."""
    rng = np.random.default_rng(3)
    n = 40
    df = pd.DataFrame({
        "good": rng.normal(0.0, 1.0, size=n),
        "artifact": np.concatenate([rng.normal(0.0, 0.05, size=n - 1), [50.0]]),
    }, index=[f"s{i}" for i in range(n)])
    mask = pd.Series([True] * n, index=df.index)
    flags_all = flag_metabolite_scores(df, mask)
    flags_scored = flag_metabolite_scores(df[["good"]], mask)
    assert "artifact" in set(flags_all["metabolite"])
    assert set(flags_scored["metabolite"]) == {"good"}
