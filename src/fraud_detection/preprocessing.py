import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

from fraud_detection.config import FEATURE_COLUMNS, PCA_FEATURE_COLUMNS

__all__ = ["FEATURE_COLUMNS", "fit_scaler", "transform", "save_scaler", "load_scaler"]


def fit_scaler(train_df: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(train_df[["Amount", "Time"]])
    return scaler


def transform(df: pd.DataFrame, scaler: StandardScaler) -> np.ndarray:
    scaled = scaler.transform(df[["Amount", "Time"]])
    pca_part = df[PCA_FEATURE_COLUMNS].to_numpy()
    return np.hstack([pca_part, scaled])


def save_scaler(scaler: StandardScaler, path) -> None:
    joblib.dump(scaler, path)


def load_scaler(path) -> StandardScaler:
    return joblib.load(path)
