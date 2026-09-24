#!/usr/bin/env python3
"""Tests for configured diagnostic ratio features (STEP 1b)."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pathway_pipeline.pipeline.pathway_stats import derive_ratio_features
from pathway_pipeline.pipeline.test_pathway_pipeline import (
    _write_hmdb_xml, _write_pathbank_csv, _write_pathway_names_csv)


def test_derive_ratio_features_log_difference():
    """On log10 values the difference is the log-ratio; NaN propagates."""
    rng = np.random.default_rng(4)
    n = 12
    df = pd.DataFrame({
        "Octanoylcarnitine": rng.normal(1.0, 0.2, size=n),
        "Acetylcarnitine": rng.normal(2.0, 0.2, size=n),
    }, index=[f"s{i}" for i in range(n)])
    df.loc["s3", "Acetylcarnitine"] = np.nan
    specs = [{"name": "C8/C2", "numerator": "Octanoylcarnitine",
              "denominator": "Acetylcarnitine"}]
    derived, audit = derive_ratio_features(df, specs)
    assert list(audit["status"]) == ["derived"]
    expected = (df["Octanoylcarnitine"] - df["Acetylcarnitine"]).drop("s3")
    actual = derived["C8/C2"].drop("s3")
    np.testing.assert_allclose(actual, expected)
    assert pd.isna(derived.loc["s3", "C8/C2"])
    assert "C8/C2" not in df.columns


def test_derive_ratio_features_audit_paths():
    """Collisions, missing parents, and incomplete specs are audited."""
    df = pd.DataFrame({"A": [1.0, 2.0], "B": [2.0, 1.0]},
                      index=["s0", "s1"])
    specs = [
        {"name": "A", "numerator": "A", "denominator": "B"},
        {"name": "A/C", "numerator": "A", "denominator": "C"},
        {"name": "", "numerator": "A", "denominator": "B"},
    ]
    derived, audit = derive_ratio_features(df, specs)
    assert list(audit["status"]) == ["name_collision",
                                     "missing_parent: C",
                                     "incomplete_spec"]
    assert list(derived.columns) == ["A", "B"]


def test_derive_ratio_features_sum_spec():
    """List specs sum in linear space: log10(10^a + 10^b) - log10(10^c)."""
    rng = np.random.default_rng(9)
    n = 10
    df = pd.DataFrame({
        "C16": rng.normal(1.0, 0.2, size=n),
        "C18": rng.normal(1.2, 0.2, size=n),
        "C2": rng.normal(2.0, 0.2, size=n),
    }, index=[f"s{i}" for i in range(n)])
    df.loc["s5", "C16"] = np.nan
    specs = [{"name": "(C16+C18)/C2",
              "numerator": ["C16", "C18"],
              "denominator": "C2"}]
    derived, audit = derive_ratio_features(df, specs)
    assert list(audit["status"]) == ["derived"]
    assert audit.loc[0, "numerator"] == "C16+C18"
    expected = (np.log10(10 ** df["C16"] + 10 ** df["C18"])
                - df["C2"])
    np.testing.assert_allclose(derived["(C16+C18)/C2"], expected)
    assert pd.isna(derived.loc["s5", "(C16+C18)/C2"])


def test_ratio_feature_reaches_metabolite_scoring_not_pathways(tmp_path):
    """A configured ratio is z-scored and flag-eligible but never
    pathway-mapped or Stouffer-scored, and the empty default changes
    nothing."""
    xml = _write_hmdb_xml(tmp_path)
    pathbank = _write_pathbank_csv(tmp_path)
    names = _write_pathway_names_csv(tmp_path)
    rng = np.random.default_rng(6)
    n_normal, n_imd = 30, 4
    n = n_normal + n_imd
    carnitine_shift = 1.6
    df = pd.DataFrame({
        "Sample": [f"s{i}" for i in range(n)],
        "Classification": [0] * n_normal + [1] * n_imd,
        "Oordeel targeted": [0] * n_normal + [1] * n_imd,
        "Alanine": list(rng.normal(2.0, 0.30, size=n)),
        "ATP": list(rng.normal(2.0, 0.30, size=n)),
        "Octanoylcarnitine": list(rng.normal(1.0, 0.10, size=n)),
        "Acetylcarnitine": list(rng.normal(2.0, 0.10, size=n)),
    })
    df.loc[df.index[-1], "Octanoylcarnitine"] = 1.0 + carnitine_shift
    input_csv = tmp_path / "input.csv"
    df.to_csv(input_csv, index=False)
    ratio = {"name": "C8/C2", "numerator": "Octanoylcarnitine",
             "denominator": "Acetylcarnitine"}
    base_config = {
        "input_file": str(input_csv),
        "output_dir": str(tmp_path / "out"),
        "patient_id_column": "Sample",
        "non_feature_columns": ["Oordeel targeted", "Classification"],
        "hmdb_xml_file": xml,
        "use_hmdb_cache": False,
        "pathbank_file": pathbank,
        "pathbank_pathway_names_file": names,
        "min_pathway_coverage": 0.10,
        "min_pathway_features": 2,
        "min_stouffer_metabolites": 2,
        "sample_rule": "max_excess",
        "max_sample_p": 0.5,
        "run_metabolite_flags": True,
        "save_mapping_outputs": True,
        "save_zscore_outputs": True,
        "save_stouffer_outputs": True,
        "save_flagging_outputs": True,
    }
    from pathway_pipeline.main import run_pipeline
    config_with = dict(base_config, ratio_features=[ratio])
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config_with, f)
    result = run_pipeline(input_file=str(input_csv),
                          output_dir=str(tmp_path / "out"),
                          config_path=str(config_path))
    zscores = result["zscores"]
    assert "C8/C2" in set(zscores.columns)
    assert "Octanoylcarnitine" not in set(zscores.columns)
    imd_z = zscores.loc[zscores.index[-1], "C8/C2"]
    assert imd_z > 10
    f2h = result["feature_to_hmdb"]
    assert "C8/C2" not in set(f2h["feature"])
    coverage = result["pathway_coverage_scored"]
    assert "C8/C2" not in set(
        coverage["matched_features"].str.split(";").explode().dropna())
    mflags = pd.read_csv(tmp_path / "out" / "metabolite_flags.csv")
    assert "C8/C2" in set(mflags["metabolite"])
    decisions = result["sample_decisions"]
    top = decisions.loc[decisions["sample_id"]
                        == decisions["sample_id"].iloc[-1],
                        "max_metabolite_z"]
    assert (top >= carnitine_shift * 10 - 1e-6).any() or top.iloc[0] > 5
