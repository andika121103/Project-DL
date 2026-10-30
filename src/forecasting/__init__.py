"""Forecasting revenue per jam per cabang: seasonal naive, Random Forest, dan LSTM.

    data.py              data & fitur bersama, split train/test
    baseline.py          seasonal naive
    ml_random_forest.py  Machine Learning (scikit-learn)
    dl_lstm.py           Deep Learning (PyTorch)
    evaluate.py          gabung prediksi & metrik

Dipakai oleh train_models.py dan notebooks/modeling.ipynb. Dashboard (app.py) TIDAK mengimpor modul ini
supaya Streamlit Cloud tidak butuh torch/sklearn; dashboard hanya membaca file hasil di results/.
"""
import json

import joblib
import torch

from src import processing as P

from .baseline import predict_seasonal_naive
from .data import SEED, STORES, TEST_START, VAL_START, add_lag_features, hourly_series, split
from .dl_lstm import WINDOW, LSTMForecaster, lstm_dataset, predict_lstm, save_loss_plot, train_lstm
from .evaluate import MODELS, build_results, daily, evaluate, scores
from .ml_random_forest import RF_DEFAULT, RF_FEATURES, RF_GRID, predict_rf, train_rf, tune_rf

ML_DIR, DL_DIR = P.ROOT / "models" / "ml", P.ROOT / "models" / "dl"
RESULTS = P.RESULTS_DIR / "forecast_results.parquet"
METRICS = P.RESULTS_DIR / "model_metrics.csv"
LOSS = P.RESULTS_DIR / "lstm_loss.csv"
TUNING = P.RESULTS_DIR / "rf_tuning.json"


def run(save: bool = True) -> dict:
    """Latih ulang semua model dari awal, evaluasi di Juni, dan (opsional) simpan model & hasil."""
    h = add_lag_features(hourly_series(P.add_features(P.load_data())))
    train, test = split(h)
    assert train["ts"].max() < TEST_START <= test["ts"].min()
    gs = tune_rf(train)
    rf = gs.best_estimator_
    rf_default = train_rf(train)  # pembanding: parameter sebelum tuning
    tuning = {"best_params": gs.best_params_, "cv_rmse": -gs.best_score_, "default_params": RF_DEFAULT,
              "test_rmse_default": scores(test["revenue"].to_numpy(), predict_rf(rf_default, test))["RMSE"]}
    lstm, loss, scaler = train_lstm(h)
    res = build_results(h, test, rf, lstm, scaler)
    metrics = evaluate(res)
    if save:
        for d in (ML_DIR, DL_DIR, P.RESULTS_DIR):
            d.mkdir(parents=True, exist_ok=True)
        joblib.dump(rf, ML_DIR / "random_forest.joblib", compress=3)
        torch.save(lstm.state_dict(), DL_DIR / "lstm.pt")
        (DL_DIR / "lstm_config.json").write_text(json.dumps({"scaler": scaler, "window": WINDOW, "stores": STORES}))
        save_loss_plot(loss, P.RESULTS_DIR / "lstm_loss.png")
        loss.to_csv(LOSS, index=False)
        res.to_parquet(RESULTS, index=False)
        metrics.to_csv(METRICS, index=False)
        TUNING.write_text(json.dumps(tuning, indent=2))
    return {"h": h, "train": train, "test": test, "rf": rf, "tuning": tuning, "lstm": lstm, "loss": loss,
            "scaler": scaler, "results": res, "metrics": metrics}
