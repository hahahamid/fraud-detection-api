import numpy as np

from fraud_detection.models.autoencoder import AutoencoderModel


def _make_data():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 1, size=(100, 5)).astype(np.float32)
    outliers = rng.normal(8, 1, size=(10, 5)).astype(np.float32)
    return inliers, outliers


def test_outliers_score_higher_than_inliers():
    inliers, outliers = _make_data()
    model = AutoencoderModel(input_dim=5, epochs=50)
    model.fit(inliers)
    assert model.score(outliers).mean() > model.score(inliers).mean()


def test_save_and_load_round_trip(tmp_path):
    inliers, outliers = _make_data()
    model = AutoencoderModel(input_dim=5, epochs=50)
    model.fit(inliers)
    model.set_threshold(0.6)
    path = tmp_path / "autoencoder.joblib"
    model.save(path)
    loaded = AutoencoderModel.load(path)
    np.testing.assert_allclose(model.score(outliers), loaded.score(outliers), rtol=1e-5)
    assert loaded.threshold == 0.6
