import numpy as np
import pandas as pd

from combined_batch_pipeline.pipeline.feature_ratios import add_feature_ratios


def _sample_data(columns):
    rng = np.random.default_rng(0)
    rows = ["Propionylcarnitine.HMDB0000842", "L-Acetylcarnitine.HMDB0000201"]
    return pd.DataFrame(rng.normal(2.0, 0.5, size=(len(rows), len(columns))),
                       index=rows, columns=columns)


def test_adds_ratio_feature():
    df = _sample_data(["s1", "s2", "s3"])
    ratios = [{
        "name": "ratio_C3_C2",
        "numerator": ["Propionylcarnitine.HMDB0000842"],
        "denominator": ["L-Acetylcarnitine.HMDB0000201"],
    }]
    out = add_feature_ratios(df, ratios)
    assert "ratio_C3_C2" in out.index
    assert out.shape[0] == df.shape[0] + 1


def test_duplicate_sample_columns():
    # Reproduces the original crash: duplicate injection sample IDs.
    df = _sample_data(["s1", "s1", "s2"])
    ratios = [{
        "name": "ratio_C3_C2",
        "numerator": ["Propionylcarnitine.HMDB0000842"],
        "denominator": ["L-Acetylcarnitine.HMDB0000201"],
    }]
    out = add_feature_ratios(df, ratios)
    assert list(out.columns) == ["s1", "s1", "s2"]
    assert "ratio_C3_C2" in out.index
    assert out.loc["ratio_C3_C2"].notna().all()


def test_missing_features_skipped():
    df = _sample_data(["s1", "s2"])
    ratios = [{
        "name": "ratio_C3_C2",
        "numerator": ["DoesNotExist"],
        "denominator": ["L-Acetylcarnitine.HMDB0000201"],
    }]
    out = add_feature_ratios(df, ratios)
    assert "ratio_C3_C2" not in out.index
    assert out.shape[0] == df.shape[0]


def test_empty_ratios_returns_unchanged():
    df = _sample_data(["s1", "s2"])
    out = add_feature_ratios(df, [])
    pd.testing.assert_frame_equal(out, df)


def test_ratio_value_correct():
    # ratio = 10**num / 10**den on the raw scale, then log10 -> num - den.
    df = pd.DataFrame(
        {"s1": [np.log10(100.0), np.log10(10.0)],
         "s2": [np.log10(50.0), np.log10(5.0)]},
        index=["Propionylcarnitine.HMDB0000842", "L-Acetylcarnitine.HMDB0000201"],
    )
    out = add_feature_ratios(df, [{
        "name": "ratio_C3_C2",
        "numerator": ["Propionylcarnitine.HMDB0000842"],
        "denominator": ["L-Acetylcarnitine.HMDB0000201"],
    }])
    np.testing.assert_allclose(out.loc["ratio_C3_C2", "s1"], np.log10(100.0 / 10.0))
    np.testing.assert_allclose(out.loc["ratio_C3_C2", "s2"], np.log10(50.0 / 5.0))
