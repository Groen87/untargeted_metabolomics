#!/usr/bin/env python3
"""Unit tests for the lab-protocol ground-truth role masks."""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outlier_detection_pipeline.pipeline.roles import lab_protocol_role_masks


def _gm(rows):
    return pd.DataFrame(rows, columns=['raw_classification', 'oordeel', 'non_treated'])


def test_non_treated_imd_is_true_outlier():
    gm = _gm([(1, 1, 1)])
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [True]
    assert tio.tolist() == [False]


def test_treated_imd_is_gray():
    gm = _gm([(1, 1, 0), (1, 1, None)])
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [False, False]
    assert tio.tolist() == [False, False]


def test_confident_normal_is_true_inlier_regardless_of_non_treated():
    gm = _gm([(0, 0, 1), (0, 0, 0), (0, 0, None)])
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [False, False, False]
    assert tio.tolist() == [True, True, True]


def test_other_combinations_are_gray():
    gm = _gm([(2, 1, 1), (3, 0, 0), (0, 1, 1), (1, 0, 1)])
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [False, False, False, False]
    assert tio.tolist() == [False, False, False, False]


def test_nan_label_columns_are_neither():
    gm = _gm([(None, 1, 1), (1, None, 1), (None, None, 1)])
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [False, False, False]
    assert tio.tolist() == [False, False, False]


def test_missing_non_treated_column_falls_back_to_legacy():
    gm = pd.DataFrame({'raw_classification': [1, 1, 0],
                       'oordeel': [1, 0, 0]})
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [True, False, False]
    assert tio.tolist() == [False, False, True]


def test_string_values_are_coerced():
    gm = pd.DataFrame({'raw_classification': ['1', '0', 'x'],
                      'oordeel': ['1', '0', '1'],
                      'non_treated': ['1', '0', '0']})
    to, tio = lab_protocol_role_masks(gm)
    assert to.tolist() == [True, False, False]
    assert tio.tolist() == [False, True, False]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
