#!/usr/bin/env python3
"""Unit tests for the transformations-CSV feature filter (offline)."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outlier_detection_pipeline.pipeline import data_loader


def _write_csv(path: Path, rows: list) -> Path:
    cols = [
        "predecessor", "predecessorcid", "transformation", "successor",
        "successorcid", "evidencedoi", "evidenceref", "sourcecomment",
        "sourcecommentfull", "datasetref", "enzyme", "datasetdoi", "biosystem",
    ]
    df = pd.DataFrame(rows, columns=cols)
    df.to_csv(path, index=False)
    return path


def _make_input_csv(path: Path) -> Path:
    df = pd.DataFrame({
        "Caffeine": [1.0, 2.0, 3.0],
        "Paraxanthine": [4.0, 5.0, 6.0],
        "Glucose": [7.0, 8.0, 9.0],
        "Coproporphyrin-III": [1.0, 1.0, 1.0],
        "Oordeel targeted": [0, 0, 1],
        "Classification": [0, 0, 1],
    }, index=["P1", "P2", "P3"])
    df.index.name = "Sample"
    df.to_csv(path)
    return path


def test_drops_predecessor_and_successor(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    tcsv = _write_csv(tmp_path / "transformations.csv", [
        {"predecessor": "Caffeine", "successor": "Paraxanthine"},
    ])
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(tcsv),
        output_dir=str(tmp_path),
    )
    assert "Caffeine" not in features.columns
    assert "Paraxanthine" not in features.columns
    assert "Glucose" in features.columns
    assert "Coproporphyrin-III" in features.columns
    assert (tmp_path / "transformation_compounds_dropped.csv").exists()


def test_case_insensitive_and_loose_match(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    # 'caffeine' lowercase + 'Coproporphyrin III' (space, no hyphen) -> should
    # match 'Coproporphyrin-III' via the loose alphanumeric fallback.
    tcsv = _write_csv(tmp_path / "transformations.csv", [
        {"predecessor": "caffeine", "successor": "Coproporphyrin III"},
    ])
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(tcsv),
    )
    assert "Caffeine" not in features.columns
    assert "Coproporphyrin-III" not in features.columns
    assert "Glucose" in features.columns


def test_disabled_when_null(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=None,
    )
    assert set(features.columns) == {"Caffeine", "Paraxanthine", "Glucose", "Coproporphyrin-III"}


def test_missing_file_keeps_features(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(tmp_path / "does_not_exist.csv"),
    )
    assert "Glucose" in features.columns
    assert "Caffeine" in features.columns


def test_missing_columns_keeps_features(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    bad = tmp_path / "bad.csv"
    bad.write_text("foo,bar\n1,2\n")
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(bad),
    )
    assert "Glucose" in features.columns
    assert "Caffeine" in features.columns


def test_no_matches_keeps_features(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    tcsv = _write_csv(tmp_path / "transformations.csv", [
        {"predecessor": "Aspirin", "successor": "Salicylic acid"},
    ])
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(tcsv),
    )
    assert set(features.columns) == {"Caffeine", "Paraxanthine", "Glucose", "Coproporphyrin-III"}


def test_dedups_repeated_compounds(tmp_path):
    csv = _make_input_csv(tmp_path / "data.csv")
    tcsv = _write_csv(tmp_path / "transformations.csv", [
        {"predecessor": "Caffeine", "successor": "Paraxanthine"},
        {"predecessor": "Caffeine", "successor": "Theobromine"},
        {"predecessor": "Caffeine", "successor": "Paraxanthine"},
    ])
    features, _, _, _ = data_loader.load_data(
        input_file=str(csv),
        non_feature_columns=["Oordeel targeted", "Classification"],
        transformations_file=str(tcsv),
    )
    assert "Caffeine" not in features.columns
    assert "Paraxanthine" not in features.columns
    assert "Glucose" in features.columns


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
