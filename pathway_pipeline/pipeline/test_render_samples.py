"""Tests for the report-only sample figure renderer (render_samples).

The helper must rebuild the STEP 8e figure inputs purely from saved
CSVs (including re-merging promoted disease panels the way main.py
does), never recompute scores, and produce the same PNGs the pipeline's
own STEP 8e block writes.
"""
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pathway_pipeline.render_samples import (
    load_run_outputs, main, render_samples)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_run_outputs(run_dir: Path, promoted: bool = True) -> dict:
    """Write a synthetic saved run: normal-masked z-scores, PathBank
    pathways plus one promoted IEMbase disease panel with directions."""
    run_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    samples = [f"S{i}" for i in range(30)]
    feature_to_hmdb = pd.DataFrame({
        "feature": ["F1", "F2", "F3", "F4"],
        "hmdb_id": ["HMDB00001", "HMDB00002", "HMDB00003", "HMDB00004"],
    })
    feature_to_pathway = pd.DataFrame({
        "feature": ["F1", "F2", "F3", "F4"],
        "smp_id": ["SMP00001"] * 4,
        "pathway_name": ["Fatty acid metabolism"] * 4,
        "metabolite_name": ["MetA", "MetB", "MetC", "MetD"],
        "hmdb_id": feature_to_hmdb["hmdb_id"],
    })
    coverage = feature_to_pathway[["smp_id", "pathway_name"]].drop_duplicates()
    zscores = pd.DataFrame(
        rng.normal(0, 1, size=(30, 4)), index=pd.Index(samples,
                                                    name="sample_id"),
        columns=["F1", "F2", "F3", "F4"])
    zscores.loc["S29", ["F1", "F2"]] = 6.0
    zscores.loc["S29", ["F3", "F4"]] = -4.0
    zscores.to_csv(run_dir / "metabolite_zscores.csv")
    feature_to_hmdb.to_csv(run_dir / "feature_to_hmdb.csv", index=False)
    feature_to_pathway.to_csv(run_dir / "feature_to_pathway.csv",
                              index=False)
    coverage.to_csv(run_dir / "pathway_coverage.csv", index=False)
    pd.DataFrame({
        "feature": ["F1", "F2", "F3", "F4"],
        "scale": [1.0, 1.0, 1.0, 1.0],
    }).to_csv(run_dir / "reference_stats.csv", index=False)
    split = pd.DataFrame({
        "sample_id": samples,
        "group": ["normal"] * 25 + ["other"] * 3 + ["imd"] * 2,
        "validation": [True] * 15 + [False] * 15,
    })
    split.to_csv(run_dir / "cohort_split.csv", index=False)

    scores = []
    for s in samples:
        z = 5.0 if s == "S29" else rng.normal(0.2, 0.4)
        scores.append({
            "sample_id": s, "smp_id": "SMP00001",
            "pathway_name": "Fatty acid metabolism",
            "n_metabolites_used": 4,
            "z_stouffer": z, "z_stouffer_abs": abs(z),
        })
    pathway_scores = pd.DataFrame(scores)
    pathway_scores.to_csv(run_dir / "pathway_stouffer_scores.csv",
                          index=False)
    pathway_flags = pathway_scores.copy()
    pathway_flags["threshold"] = 3.0
    pathway_flags["excess"] = pathway_flags["z_stouffer_abs"] / 3.0
    pathway_flags["flagged"] = pathway_flags["excess"] > 1.0
    metabolite_flags = pd.DataFrame({
        "sample_id": np.repeat(samples, 4),
        "metabolite": ["F1", "F2", "F3", "F4"] * 30,
        "abs_z": np.abs(zscores.to_numpy()).flatten(),
        "threshold": 2.5,
    })
    metabolite_flags["excess"] = metabolite_flags["abs_z"] / 2.5
    metabolite_flags["flagged"] = metabolite_flags["excess"] > 1.0

    decisions = pd.DataFrame({
        "sample_id": samples,
        "flagged": [s == "S29" for s in samples],
        "top_excess": [1.6 if s == "S29" else rng.uniform(0.1, 0.8)
                       for s in samples],
    })
    decisions.to_csv(run_dir / "sample_decisions.csv", index=False)

    promoted_diseases = []
    if promoted:
        panel_scores = []
        for s in samples:
            z = 4.5 if s == "S29" else rng.normal(0.1, 0.3)
            panel_scores.append({
                "sample_id": s, "smp_id": "DISEASE-pde",
                "pathway_name": "Pyridoxine-dependent epilepsy",
                "n_metabolites_used": 2,
                "z_stouffer": z, "z_stouffer_abs": abs(z),
            })
        panel_scores = pd.DataFrame(panel_scores)
        panel_scores.to_csv(run_dir / "disease_panel_scores.csv",
                            index=False)
        promoted_diseases = ["Pyridoxine-dependent epilepsy"]
        flags = panel_scores.copy()
        flags["threshold"] = 3.0
        flags["excess"] = flags["z_stouffer_abs"] / 3.0
        flags["flagged"] = flags["excess"] > 1.0
        pathway_flags = pd.concat([pathway_flags, flags],
                                  ignore_index=True)
        decisions = decisions.copy()
        decisions["flagged"] = [True] * 30
        decisions["top_excess"] = [2.0 if s == "S29" else 0.5
                                   for s in samples]
        decisions.to_csv(run_dir / "sample_decisions.csv", index=False)
    pathway_flags.to_csv(run_dir / "pathway_flags.csv", index=False)
    metabolite_flags.to_csv(run_dir / "metabolite_flags.csv", index=False)

    config = {
        "demoted_features": [],
        "scale_weighted_metabolites": True,
        "biomarker_channel": {
            "disease_table_file": str(
                REPO_ROOT / "pathway_pipeline"
                / "iembase-diseases-with-pathbank-pathways-and-hmdb-codes.xlsx"),
        },
    }
    import yaml
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f)
    return {"samples": samples,
            "promoted": promoted_diseases}


def test_load_run_outputs_rebuilds_promoted_panels(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    frames = load_run_outputs(run_dir)
    assert frames is not None
    assert "Pyridoxine-dependent epilepsy" in frames["promoted_diseases"]
    names = set(frames["pathway_scores"]["pathway_name"])
    assert "Pyridoxine-dependent epilepsy" in names
    assert "Fatty acid metabolism" in names
    assert frames["normal_mask"].sum() == 25
    assert frames["feature_scale_weights"] is not None


def test_load_run_outputs_numeric_ids_unnamed_index(tmp_path):
    """Real runs write metabolite_zscores.csv with the patient-ID column
    as the (possibly unnamed or differently named) index and numeric IDs;
    the loader must still expose a string sample_id index."""
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    raw = pd.read_csv(run_dir / "metabolite_zscores.csv", index_col=0)
    raw.index.name = None
    raw.index = raw.index.astype(str).map(lambda s: s.replace("S", ""))
    raw.to_csv(run_dir / "metabolite_zscores.csv")
    for name in ("cohort_split.csv", "sample_decisions.csv"):
        f = run_dir / name
        df = pd.read_csv(f)
        df["sample_id"] = df["sample_id"].astype(str).str.replace("S", "")
        df.to_csv(f, index=False)
    frames = load_run_outputs(run_dir)
    assert frames is not None
    assert "26150973733" not in frames["zscores_scored"].index
    assert "0" in frames["zscores_scored"].index
    assert str(raw.index[0]) in frames["zscores_scored"].index
    code = render_samples(run_dir, ["29"], out_dir=tmp_path / "figs")
    assert code == 0
    assert any("29_" in p.name for p in (tmp_path / "figs").glob("*.png"))


def test_render_samples_writes_figures(tmp_path):
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    out_dir = tmp_path / "figs"
    code = render_samples(run_dir, ["S29"], out_dir=out_dir)
    assert code == 0
    pngs = sorted(p.name for p in out_dir.glob("S29_*.png"))
    assert any("overview" in n for n in pngs)
    assert any("top_pathway_waterfall" in n or "waterfall" in n
               for n in pngs)
    assert any("panelA" in n for n in pngs)


def test_waterfall_members_from_saved_audit(tmp_path):
    """A run's own disease_table_audit.csv must resolve promoted-panel
    members for the waterfall even when the Excel disease table file is
    absent -- scoring used the run's saved mapping, not a re-resolution."""
    import yaml
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    config = {
        "demoted_features": [],
        "scale_weighted_metabolites": True,
        "biomarker_channel": {
            "disease_table_file": str(tmp_path / "missing.xlsx")},
    }
    with open(run_dir / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f)
    audit = pd.DataFrame({
        "disease": ["Pyridoxine-dependent epilepsy"],
        "biomarker": ["MetA"],
        "hmdb_id": ["HMDB00001"],
        "direction": ["up"],
        "features": ["F1"],
    })
    audit.to_csv(run_dir / "disease_table_audit.csv", index=False)
    frames = load_run_outputs(run_dir)
    assert frames is not None
    resolved = frames["disease_resolved"]
    assert resolved is not None and not resolved.empty
    assert "Pyridoxine-dependent epilepsy" in set(resolved["disease"])
    code = render_samples(run_dir, ["S29"], out_dir=tmp_path / "figs")
    assert code == 0
    assert any("top_pathway_waterfall" in p.name
               for p in (tmp_path / "figs").glob("S29_*.png"))


def test_render_samples_unknown_ids(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    out_dir = tmp_path / "figs"
    code = render_samples(run_dir, ["NOSUCH"], out_dir=out_dir)
    assert code == 1


def test_render_samples_without_disease_table(tmp_path):
    """A run without promoted panels (no disease_panel_scores.csv) must
    still render; disease panels are simply absent."""
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=False)
    out_dir = tmp_path / "figs"
    code = render_samples(run_dir, ["S29"], out_dir=out_dir)
    assert code == 0
    assert (out_dir / "S29_overview.png").exists()


def test_main_top_flagged(tmp_path, capsys):
    run_dir = tmp_path / "run"
    _write_run_outputs(run_dir, promoted=True)
    code = main([str(run_dir), "--top-flagged", "1"])
    assert code == 0
    default_out = run_dir / "figures" / "sample_reports"
    assert any(default_out.glob("*.png"))


def test_main_seed_layout(tmp_path):
    """The multi-split per_seed/seed_<seed> layout resolves and renders
    into a seed-specific figures directory."""
    run_dir = tmp_path / "multi" / "per_seed" / "seed_20260923"
    _write_run_outputs(run_dir, promoted=True)
    code = main([str(tmp_path / "multi"), "--seed", "20260923",
                 "--ids", "S29"])
    assert code == 0
    out = tmp_path / "multi" / "figures" / "seed_20260923" / "sample_reports"
    assert any(out.glob("S29_*.png"))


def test_main_list_samples(tmp_path, capsys):
    run_dir = tmp_path / "run"
    meta = _write_run_outputs(run_dir, promoted=False)
    code = main([str(run_dir), "--list-samples"])
    assert code == 0
    out = capsys.readouterr().out.strip().splitlines()
    assert meta["samples"][29] in out
