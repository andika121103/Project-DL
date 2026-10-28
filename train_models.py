"""Latih ulang semua model forecasting dari awal: pip install -r requirements-train.txt && python train_models.py"""
import time

from src.forecasting import run

if __name__ == "__main__":
    t0 = time.time()
    out = run()
    m = out["metrics"]
    print(f"Selesai dalam {time.time() - t0:.0f} detik. LSTM berhenti di epoch {len(out['loss'])}.")
    t = out["tuning"]
    print(f"RF terbaik (TimeSeriesSplit): {t['best_params']}, CV RMSE {t['cv_rmse']:.2f}; "
          f"RMSE test parameter default {t['test_rmse_default']:.2f}")
    for level in ("jam", "hari"):
        print(f"\nMetrik test Juni (level {level}):")
        print(m[m["level"] == level].drop(columns="level").to_string(index=False, float_format="%.2f"))
