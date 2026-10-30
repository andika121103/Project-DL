"""Baseline: seasonal naive — prediksi = revenue jam yang sama 7 hari sebelumnya."""
import pandas as pd


def predict_seasonal_naive(test: pd.DataFrame) -> pd.Series:
    return test["lag_168"]  # butuh kolom dari data.add_lag_features
