import numpy as np

from fraud_detection.models.isolation_forest import IsolationForestModel


def _make_data():
    rng = np.random.default_rng(0)
    inliers = rng.normal(0, 1, size=(100, 5))
    outliers = rng.normal(8, 1, size=(10, 5))
    return inliers, outliers


def test_outliers_score_higher_than_inliers():
    inliers, outliers = _make_data()
    model = IsolationForestModel()
    model.fit(inliers)
    assert model.score(outliers).mean() > model.score(inliers).mean()


def test_save_and_load_round_trip(tmp_path):
    inliers, outliers = _make_data()
    model = IsolationForestModel()
    model.fit(inliers)
    model.set_threshold(0.7)
    path = tmp_path / "isolation_forest.joblib"
    model.save(path)
    loaded = IsolationForestModel.load(path)
    np.testing.assert_array_equal(model.score(outliers), loaded.score(outliers))
    assert loaded.threshold == 0.7
