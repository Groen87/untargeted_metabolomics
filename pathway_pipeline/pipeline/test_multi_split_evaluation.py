#!/usr/bin/env python3
"""Tests for the pre-declared multi-split evaluation driver."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.test_pathway_pipeline import (
    _write_hmdb_xml, _write_pathbank_csv, _write_pathway_names_csv)
from pathway_pipeline.multi_split_evaluation import (
    run_multi_split_evaluation, _aggregate, _write_report)


def _write_inputs(tmp_path):
    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)
    rng = np.random.default_rng(9)
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
    df.to_csv(tmp_path / "input.csv", index=False)
    config = {
        "input_file": str(tmp_path / "input.csv"),
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
        "run_evaluation": False,
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return str(tmp_path / "input.csv"), str(config_path)


def test_multi_split_evaluation_writes_runs_and_report(tmp_path):
    input_csv, config_path = _write_inputs(tmp_path)
    out_root = tmp_path / "ms"
    runs, missed, stability = run_multi_split_evaluation(
        input_file=input_csv,
        config_path=config_path,
        output_dir=str(out_root),
        seeds=[21, 22],
    )
    assert list(runs["seed"]) == [21, 22]
    for key in ("sensitivity", "specificity", "auc", "n_normal", "n_imd",
                "sensitivity_ci_low", "sensitivity_ci_high", "auc_ci_low",
                "auc_ci_high", "dev_flag_rate", "val_flag_rate",
                "n_missed_imd"):
        assert key in runs.columns, key
    assert runs["n_normal"].eq(30).all()
    assert runs["n_imd"].eq(4).all()
    assert runs["n_other"].eq(2).all()
    assert runs["n_missed_imd"].eq(len(missed[missed["seed"] == 21])
                                   if not missed.empty else 0).iloc[0] \
        or runs["n_missed_imd"].iloc[0] == 0
    for seed in (21, 22):
        seed_dir = out_root / "per_seed" / f"seed_{seed}"
        used = yaml.safe_load((seed_dir / "config_used.yaml").read_text())
        assert used["run_evaluation"] is True
        assert used["evaluate_dev_half"] is False
        assert used["validation_split"]["seed"] == seed
        assert (seed_dir / "evaluation_summary.csv").exists()
    agg = _aggregate(runs)
    for metric in ("sensitivity", "specificity", "auc",
                   "dev_flag_rate", "val_flag_rate"):
        assert metric in set(agg["metric"])
        sel = agg[agg["metric"] == metric].iloc[0]
        assert sel["n_seeds"] == 2
        assert sel["min"] <= sel["mean"] <= sel["max"]
    assert list(stability["n_splits"].unique()) == [2]
    assert (stability["flag_rate"] <= 1.0).all()
    assert (stability["flag_rate"] >= 0.0).all()
    assert stability["flag_rate"].is_monotonic_decreasing
    n_samples = runs["n_normal"].iloc[0] + runs["n_imd"].iloc[0] \
        + runs["n_other"].iloc[0]
    assert len(stability) == n_samples
    _write_report(out_root, [21, 22], runs, agg, missed, stability)
    report = (out_root / "MULTI_SPLIT_REPORT.md").read_text()
    assert "Multi-Split Evaluation Report" in report
    assert "sensitivity" in report
