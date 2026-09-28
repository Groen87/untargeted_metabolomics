"""Tests for the report-only visualization module.

The figures are report-only: these tests check that every function
produces its PNG files from realistic pipeline-style frames, stays
label-blind by default (no IMD-group coloring unless asked), and never
raises on empty or missing inputs.
"""

import numpy as np
import pandas as pd
import pytest

from pathway_pipeline.pipeline.visualize import (
    overview_figures, sample_overview_figure, sample_report_figures,
    top_pathway_waterfall_figure, multisplit_figures)


@pytest.fixture
def frames():
    rng = np.random.default_rng(11)
    n_samples = 40
    samples = [f"S{i}" for i in range(n_samples)]
    normals = pd.Series([i < 25 for i in range(n_samples)], index=samples)

    decisions = pd.DataFrame({
        "sample_id": samples,
        "flagged": [i >= 30 for i in range(n_samples)],
        "top_excess": np.concatenate([
            rng.uniform(0.2, 0.9, 30), rng.uniform(1.2, 4.0, 10)]),
        "sample_p_value": np.concatenate([
            rng.uniform(0.2, 1.0, 30), rng.uniform(0.0, 0.04, 10)]),
        "group": ["normal"] * 25 + ["other"] * 5 + ["imd"] * 10,
    })

    pathways = ["PW_A", "PW_B", "PW_C", "PW_D"]
    rows = []
    for s in samples:
        for pw in pathways:
            z = rng.normal(0.3, 0.5) if s not in samples[30:] else \
                rng.normal(2.0, 1.0)
            rows.append({"sample_id": s, "smp_id": pw, "pathway_name": pw,
                         "n_metabolites_used": 5,
                         "z_stouffer": z, "z_stouffer_abs": abs(z)})
    pathway_scores = pd.DataFrame(rows)
    pathway_scores["threshold"] = 1.5
    pathway_scores["excess"] = pathway_scores["z_stouffer_abs"] / 1.5
    pathway_scores["flagged"] = pathway_scores["excess"] > 1.0
    pathway_flags = pathway_scores

    metab_rows = []
    for s in samples:
        for m in ("M1", "M2", "M3", "M4"):
            az = abs(rng.normal(0.5, 0.6))
            metab_rows.append({"sample_id": s, "metabolite": m,
                               "abs_z": az, "threshold": 2.5,
                               "excess": az / 2.5, "flagged": az / 2.5 > 1.0})
    metabolite_flags = pd.DataFrame(metab_rows)

    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(n_samples, 4)),
        index=samples, columns=["M1", "M2", "M3", "M4"])

    return {"decisions": decisions, "pathway_flags": pathway_flags,
            "pathway_scores": pathway_scores, "metabolite_flags": metabolite_flags,
            "zscores": zscores, "normal_mask": normals}


def test_overview_figures_written(tmp_path, frames):
    written = overview_figures(
        decisions=frames["decisions"],
        pathway_flags=frames["pathway_flags"],
        pathway_scores=frames["pathway_scores"],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert written, "at least one overview figure must be written"
    for name, ok in written.items():
        assert ok, f"{name} failed to write"
        assert (tmp_path / name).exists()
        assert (tmp_path / name).stat().st_size > 0


def test_overview_figures_label_blind_by_default(tmp_path, frames):
    """Default figures must not read the IMD group column at all."""
    decisions = frames["decisions"].drop(columns=["group"])
    written = overview_figures(
        decisions=decisions,
        pathway_flags=frames["pathway_flags"],
        pathway_scores=frames["pathway_scores"],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert written and all(written.values())


def test_overview_figures_empty_inputs(tmp_path):
    empty = pd.DataFrame()
    written = overview_figures(decisions=empty, pathway_flags=empty,
                               pathway_scores=empty, normal_mask=None,
                               out_dir=tmp_path)
    assert written == {}


def test_sample_report_written(tmp_path, frames):
    written = sample_report_figures(
        sample_id="S35",
        zscores=frames["zscores"],
        pathway_scores=frames["pathway_scores"],
        pathway_flags=frames["pathway_flags"],
        metabolite_flags=frames["metabolite_flags"],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert written, "at least one panel must be written for a scored sample"
    for name, ok in written.items():
        assert ok
        assert (tmp_path / name).exists()


def test_sample_report_unknown_sample(tmp_path, frames):
    written = sample_report_figures(
        sample_id="DOES_NOT_EXIST",
        zscores=frames["zscores"],
        pathway_scores=frames["pathway_scores"],
        pathway_flags=frames["pathway_flags"],
        metabolite_flags=frames["metabolite_flags"],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert written == {}


def test_sample_overview_figure_written(tmp_path, frames):
    """The 3-panel per-sample overview: disease waterfall, PathBank
    waterfall, top signed metabolite z-scores."""
    flags = frames["pathway_flags"].copy()
    # Promote two of the four pathways into IEMbase disease panels.
    flags["pathway_name"] = flags["pathway_name"].replace(
        {"PW_A": "Disease A panel", "PW_B": "Disease B panel"})
    promoted = ["Disease A panel", "Disease B panel"]
    ok = sample_overview_figure(
        sample_id="S35",
        zscores=frames["zscores"],
        pathway_scores=frames["pathway_scores"],
        pathway_flags=flags,
        promoted_diseases=promoted,
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert ok
    out = tmp_path / "S35_overview.png"
    assert out.exists() and out.stat().st_size > 0


def test_sample_overview_figure_unknown_sample(tmp_path, frames):
    ok = sample_overview_figure(
        sample_id="DOES_NOT_EXIST",
        zscores=frames["zscores"],
        pathway_scores=frames["pathway_scores"],
        pathway_flags=frames["pathway_flags"],
        promoted_diseases=[],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert ok is False
    assert list(tmp_path.glob("*.png")) == []


def test_sample_overview_figure_no_panels(tmp_path, frames):
    """With no promoted panels the disease axis degrades gracefully."""
    ok = sample_overview_figure(
        sample_id="S35",
        zscores=frames["zscores"],
        pathway_scores=frames["pathway_scores"],
        pathway_flags=frames["pathway_flags"],
        promoted_diseases=[],
        normal_mask=frames["normal_mask"],
        out_dir=tmp_path)
    assert ok
    assert (tmp_path / "S35_overview.png").stat().st_size > 0


def test_top_pathway_waterfall_flagged_sample(tmp_path, frames):
    """Flagged sample: member z waterfall for the top flagging pathway."""
    flags = frames["pathway_flags"].copy()
    # Force one pathway to flag for S35 with a big excess.
    sel = (flags["sample_id"] == "S35") & (flags["pathway_name"] == "PW_C")
    flags.loc[sel, ["flagged", "excess"]] = [True, 2.5]
    f2p = pd.DataFrame([
        {"feature": "M1", "hmdb_id": "HMDB0000001", "smp_id": "PW_C",
         "pathway_name": "PW_C", "metabolite_id": "PW_C1",
         "metabolite_name": "Metabolite One"},
        {"feature": "M2", "hmdb_id": "HMDB0000002", "smp_id": "PW_C",
         "pathway_name": "PW_C", "metabolite_id": "PW_C2",
         "metabolite_name": "Metabolite Two"},
        {"feature": "M3", "hmdb_id": "HMDB0000003", "smp_id": "PW_C",
         "pathway_name": "PW_C", "metabolite_id": "PW_C3",
         "metabolite_name": "Metabolite Three"},
    ])
    ok = top_pathway_waterfall_figure(
        sample_id="S35",
        zscores=frames["zscores"],
        feature_to_pathway=f2p,
        disease_resolved=None,
        pathway_scores=frames["pathway_scores"],
        pathway_flags=flags,
        out_dir=tmp_path)
    assert ok
    out = tmp_path / "S35_top_pathway_waterfall.png"
    assert out.exists() and out.stat().st_size > 0


def test_top_pathway_waterfall_disease_panel_directions(tmp_path, frames):
    """Disease-panel top pathway: literature direction arrows on the bars."""
    flags = frames["pathway_flags"].copy()
    flags["pathway_name"] = flags["pathway_name"].replace(
        {"PW_A": "MCADD"})
    sel = (flags["sample_id"] == "S35") & (flags["pathway_name"] == "MCADD")
    flags.loc[sel, ["flagged", "excess"]] = [True, 2.1]
    resolved = pd.DataFrame([
        {"disease": "MCADD", "biomarker": "Octanoylcarnitine",
         "hmdb_id": "HMDB0000001", "direction": "up",
         "features": "M1"},
        {"disease": "MCADD", "biomarker": "Acetylcarnitine",
         "hmdb_id": "HMDB0000002", "direction": "down",
         "features": "M2"},
        {"disease": "MCADD", "biomarker": "Hexanoylcarnitine",
         "hmdb_id": "HMDB0000003", "direction": "up",
         "features": "M3"},
    ])
    ok = top_pathway_waterfall_figure(
        sample_id="S35",
        zscores=frames["zscores"],
        feature_to_pathway=pd.DataFrame(),
        disease_resolved=resolved,
        pathway_scores=frames["pathway_scores"],
        pathway_flags=flags,
        out_dir=tmp_path)
    assert ok
    assert (tmp_path / "S35_top_pathway_waterfall.png").stat().st_size > 0


def test_top_pathway_waterfall_unflagged_sample(tmp_path, frames):
    """Unflagged sample: no figure, no crash."""
    ok = top_pathway_waterfall_figure(
        sample_id="S5",
        zscores=frames["zscores"],
        feature_to_pathway=pd.DataFrame(),
        disease_resolved=None,
        pathway_scores=frames["pathway_scores"],
        pathway_flags=frames["pathway_flags"],
        out_dir=tmp_path)
    assert ok is False
    assert list(tmp_path.glob("*.png")) == []


def test_multisplit_figures_written(tmp_path):
    runs = pd.DataFrame({
        "seed": [1, 2, 3],
        "sensitivity": [1.0, 0.917, 1.0],
        "specificity": [0.85, 0.72, 0.94],
        "auc": [0.999, 0.98, 1.0],
        "dev_flag_rate": [0.09, 0.11, 0.08],
        "val_flag_rate": [0.19, 0.28, 0.06],
    })
    stability = pd.DataFrame({
        "sample_id": [f"S{i}" for i in range(6)],
        "group": ["imd", "imd", "imd", "normal", "normal", "other"],
        "flag_rate": [1.0, 1.0, 0.5, 0.1, 0.0, 0.3],
    })
    written = multisplit_figures(runs=runs, stability=stability,
                                 out_dir=tmp_path)
    assert written and all(written.values())
    for name in written:
        assert (tmp_path / name).exists()


def test_multisplit_figures_empty(tmp_path):
    assert multisplit_figures(runs=pd.DataFrame(),
                              stability=pd.DataFrame(),
                              out_dir=tmp_path) == {}
