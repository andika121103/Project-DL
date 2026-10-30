"""Evaluasi: gabungkan prediksi ketiga model di data test, lalu hitung MAE/RMSE/MAPE."""
import numpy as np
import pandas as pd

from .baseline import predict_seasonal_naive
from .dl_lstm import predict_lstm
from .ml_random_forest import predict_rf

MODELS = {"seasonal_naive": "Seasonal Naive", "random_forest": "Random Forest", "lstm": "LSTM"}


def build_results(h, test, rf, lstm, scaler) -> pd.DataFrame:
    """Aktual vs prediksi ketiga model di data test (prediksi 1 jam ke depan, riwayat = data aktual)."""
    res = test[["ts", "store_location", "jam"]].assign(
        tanggal=test["ts"].dt.normalize(), aktual=test["revenue"],
        seasonal_naive=predict_seasonal_naive(test),
        random_forest=predict_rf(rf, test),
        lstm=predict_lstm(lstm, h, scaler).reindex(test.index))
    assert res[list(MODELS)].notna().all().all()
    return res.reset_index(drop=True)


def scores(a: np.ndarray, p: np.ndarray) -> dict:
    nz = a > 0  # MAPE mengabaikan jam dengan aktual 0
    return {"MAE": np.mean(np.abs(a - p)), "RMSE": np.sqrt(np.mean((a - p) ** 2)),
            "MAPE": np.mean(np.abs((a[nz] - p[nz]) / a[nz])) * 100}


def daily(res: pd.DataFrame) -> pd.DataFrame:
    """Prediksi per jam dijumlahkan menjadi per hari per cabang."""
    return res.groupby(["store_location", "tanggal"], as_index=False)[["aktual", *MODELS]].sum()


def evaluate(res: pd.DataFrame) -> pd.DataFrame:
    """MAE/RMSE/MAPE per model × cabang (+ 'Semua'), di level jam dan level hari."""
    rows = []
    for level, d in (("jam", res), ("hari", daily(res))):
        for cabang, g in [("Semua", d), *d.groupby("store_location")]:
            for col, name in MODELS.items():
                rows.append({"level": level, "cabang": cabang, "model": name,
                             **scores(g["aktual"].to_numpy(), g[col].to_numpy())})
    return pd.DataFrame(rows)
