import numpy as np

from fraud_detection.preprocessing import (
    FEATURE_COLUMNS,
    fit_scaler,
    load_scaler,
    save_scaler,
    transform,
)


def test_transform_produces_expected_shape(sample_dataframe):
    scaler = fit_scaler(sample_dataframe)
    result = transform(sample_dataframe, scaler)
    assert result.shape == (len(sample_dataframe), len(FEATURE_COLUMNS))


def test_transform_is_deterministic(sample_dataframe):
    scaler = fit_scaler(sample_dataframe)
    first = transform(sample_dataframe, scaler)
    second = transform(sample_dataframe, scaler)
    np.testing.assert_array_equal(first, second)


def test_save_and_load_scaler_round_trip(sample_dataframe, tmp_path):
    scaler = fit_scaler(sample_dataframe)
    path = tmp_path / "scaler.joblib"
    save_scaler(scaler, path)
    loaded = load_scaler(path)
    original = transform(sample_dataframe, scaler)
    reloaded = transform(sample_dataframe, loaded)
    np.testing.assert_array_equal(original, reloaded)
