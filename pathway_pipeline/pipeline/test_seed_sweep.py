#!/usr/bin/env python3
"""Tests for the label-blind split-stability seed sweep."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.test_pathway_pipeline import (
    _write_hmdb_xml, _write_pathbank_csv, _write_pathway_names_csv)
from pathway_pipeline.seed_sweep import run_seed_sweep


def _write_sweep_inputs(tmp_path):
    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)
    rng = np.random.default_rng(5)
    n_normal, n_untreated, n_treated = 30, 4, 2
    n = n_normal + n_untreated + n_treated
    df = pd.DataFrame({
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * (n_untreated + n_treated),
        "Oordeel targeted": [0] * n_normal + [1] * (n_untreated + n_treated),
        "Non-treated": [None] * n_normal + [1] * n_untreated
        + [0] * n_treated,
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
    })
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)
    config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification",
                                "Non-treated"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_coverage": 0.10,
        "min_pathway_features": 2,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.5,
        "untreated_imd_only": True,
        "run_evaluation": True,
        "run_development_qc": True,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(input_csv), str(config_path)


def test_seed_sweep_runs_label_blind_and_reports_normal_stats(tmp_path):
    """The sweep forces run_evaluation off and returns per-seed rows."""
    input_csv, config_path = _write_sweep_inputs(tmp_path)
    runs = run_seed_sweep(
        input_file=input_csv,
        config_path=config_path,
        output_dir=str(tmp_path / "sweep"),
        seeds=[11, 12],
    )
    assert list(runs["seed"]) == [11, 12]
    for _, row in runs.iterrows():
        assert row["dev_n_normals"] + row["val_n_normals"] == 30
        assert row["n_normal"] == 30
        assert row["n_imd"] == 4
        assert row["n_other"] == 2
        assert 0 <= row["dev_flagged"] <= row["dev_n_normals"]
        assert 0 <= row["val_flagged"] <= row["val_n_normals"]
    seed_dir = tmp_path / "sweep" / "seed_11"
    used = yaml.safe_load(
        (seed_dir / "config_used.yaml").read_text())
    assert used["run_evaluation"] is False
    assert used["run_development_qc"] is False
    assert used["validation_split"]["seed"] == 11
    assert not (seed_dir / "evaluation_summary.csv").exists()
