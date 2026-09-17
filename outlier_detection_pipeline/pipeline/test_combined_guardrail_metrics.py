#!/usr/bin/env python3
"""Unit tests for the combined (model OR guardrail) metrics computation."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outlier_detection_pipeline.main import _compute_combined_guardrail_metrics


def _per_sample(rows):
    return [{'sample_id': s, 'true_label': tl, 'score': sc, 'flagged': f}
            for s, tl, sc, f in rows]


def test_guardrail_rescue_counts_as_detected():
    # Model misses A2; guardrail catches it -> combined detection = 1.0
    per_sample = _per_sample([
        ('N1', 0, 0.9, 0), ('N2', 0, 0.95, 0), ('N3', 0, 0.1, 1),
        ('A1', 1, 0.05, 1), ('A2', 1, 0.8, 0), ('A3', 1, 0.07, 1),
    ])
    guardrail = {'A2': ['rescue']}
    m = _compute_combined_guardrail_metrics(per_sample, guardrail, None, 0.02)
    assert m['n_detected'] == 3
    assert m['detection_rate'] == pytest.approx(1.0)
    a2 = [r for r in m['per_sample'] if r['sample_id'] == 'A2'][0]
    assert a2['flagged'] == 1 and a2['model_flagged'] == 0 and a2['guardrail_flagged'] == 1


def test_model_only_when_no_guardrail():
    per_sample = _per_sample([
        ('N1', 0, 0.9, 0), ('N2', 0, 0.95, 0), ('N3', 0, 0.1, 1),
        ('A1', 1, 0.05, 1), ('A2', 1, 0.8, 0),
    ])
    m = _compute_combined_guardrail_metrics(per_sample, {}, None, 0.02)
    assert m['n_detected'] == 1  # only A1
    assert m['detection_rate'] == pytest.approx(0.5)


def test_combined_fpr_can_rise_from_guardrail_fp():
    # Guardrail flags a normal the model did not -> combined FPR rises.
    per_sample = _per_sample([
        ('N1', 0, 0.9, 0), ('N2', 0, 0.95, 0), ('N3', 0, 0.5, 0),
        ('A1', 1, 0.05, 1),
    ])
    guardrail = {'N3': ['guardrail FP']}
    m = _compute_combined_guardrail_metrics(per_sample, guardrail, None, 0.02)
    assert m['false_positive_rate'] == pytest.approx(1/3)
    assert m['n_false_positives'] == 1


def test_combined_uses_union_not_intersection():
    per_sample = _per_sample([
        ('N1', 0, 0.9, 0), ('N2', 0, 0.95, 0),
        ('A1', 1, 0.05, 1), ('A2', 1, 0.8, 0),
    ])
    # Both model AND guardrail flag A1; guardrail also flags A2 -> all detected
    guardrail = {'A1': ['x'], 'A2': ['y']}
    m = _compute_combined_guardrail_metrics(per_sample, guardrail, None, 0.02)
    assert m['detection_rate'] == 1.0
    assert m['n_detected'] == 2


def test_empty_returns_empty():
    assert _compute_combined_guardrail_metrics([], {}, None, 0.02) == {}


def test_group_map_restricts_headline_roles():
    # gray samples (Class/Oordeel not clean roles) excluded from headline
    idx = ['TO', 'TI', 'GRAY']
    gm = pd.DataFrame({
        'raw_classification': [1, 0, 2],
        'oordeel': [1, 0, 1],
    }, index=idx)
    # TO=true outlier, TI=true inlier, GRAY=excluded
    per_sample = _per_sample([
        ('TO', 1, 0.05, 1),    # detected
        ('TI', 0, 0.9, 0),
        ('GRAY', 1, 0.8, 0),   # model missed, guardrail catches but gray -> not headline
    ])
    guardrail = {'GRAY': ['rescue']}
    m = _compute_combined_guardrail_metrics(per_sample, guardrail, gm, 0.02)
    assert m['n_true_outlier'] == 1
    assert m['n_true_inlier'] == 1
    assert m['detection_rate'] == 1.0  # TO detected
    # GRAY rescue not counted in headline


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
