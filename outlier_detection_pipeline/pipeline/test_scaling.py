#!/usr/bin/env python3
"""Unit tests for the configurable scaler (robust / standard / none)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import RobustScaler, StandardScaler, FunctionTransformer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from outlier_detection_pipeline.pipeline.scaling import make_scaler
from outlier_detection_pipeline.pipeline.model import ExtendedIsolationForestModel


def test_make_scaler_types():
    assert isinstance(make_scaler('robust'), RobustScaler)
    assert isinstance(make_scaler('standard'), StandardScaler)
    assert isinstance(make_scaler('none'), FunctionTransformer)
    assert isinstance(make_scaler('bogus'), RobustScaler)


def test_none_scaler_is_identity():
    rng = np.random.RandomState(0)
    X = rng.normal(size=(20, 3))
    s = make_scaler('none')
    s.fit(X)
    np.testing.assert_array_equal(s.transform(X), X)


def test_model_uses_configured_scaler():
    X = pd.DataFrame(np.random.RandomState(1).normal(size=(30, 4)),
                     columns=[f'f{i}' for i in range(4)])
    y = pd.Series([0] * 20 + [1] * 10)
    for name, expected_type in [('robust', RobustScaler),
                                ('standard', StandardScaler),
                                ('none', FunctionTransformer)]:
        model = ExtendedIsolationForestModel(scorer_name='lof', scaler_name=name)
        model.fit(X, y, normal_classification=0)
        assert isinstance(model.scaler, expected_type)


def test_none_scaler_scores_on_raw_scale():
    # With scaler 'none' the scorer must see the raw (log-scale) values:
    # fit on data where anomaly depends on raw magnitude and verify
    # transform is a no-op through the model's scaler attribute.
    X = pd.DataFrame(np.random.RandomState(2).normal(6.0, 0.2, size=(30, 3)),
                     columns=[f'f{i}' for i in range(3)])
    y = pd.Series([0] * 25 + [1] * 5)
    model = ExtendedIsolationForestModel(scorer_name='lof', scaler_name='none')
    model.fit(X, y, normal_classification=0)
    X_tr = model.scaler.transform(X)
    np.testing.assert_array_equal(np.asarray(X_tr), X.values)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
