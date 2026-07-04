import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sample_dataframe() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n_rows = 200
    data = {f"V{i}": rng.normal(0, 1, n_rows) for i in range(1, 29)}
    class_labels = np.array([1 if i % 10 == 0 else 0 for i in range(n_rows)])
    for i in range(1, 6):
        data[f"V{i}"] = data[f"V{i}"] + class_labels * 6.0
    data["Time"] = np.arange(n_rows, dtype=float)
    data["Amount"] = rng.uniform(1, 500, n_rows)
    data["Class"] = class_labels
    return pd.DataFrame(data)


@pytest.fixture
def sample_csv_path(tmp_path, sample_dataframe):
    path = tmp_path / "sample_transactions.csv"
    sample_dataframe.to_csv(path, index=False)
    return path
