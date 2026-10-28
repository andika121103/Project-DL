"""Forecasting revenue per jam per cabang: seasonal naive, Random Forest, dan LSTM.

Dipakai oleh train_models.py dan notebooks/modeling.ipynb. Dashboard (app.py) TIDAK mengimpor modul ini
supaya Streamlit Cloud tidak butuh torch/sklearn; dashboard hanya membaca file hasil di data/.
"""
import copy
import json

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from torch import nn

from src import processing as P

MODELS_DIR = P.DATA_DIR.parent / "models"
RESULTS = P.DATA_DIR / "forecast_results.parquet"
METRICS = P.DATA_DIR / "model_metrics.csv"
LOSS = P.DATA_DIR / "lstm_loss.csv"
TUNING = P.DATA_DIR / "rf_tuning.json"

VAL_START = pd.Timestamp("2023-05-18")   # 2 minggu terakhir data train → validation LSTM
TEST_START = pd.Timestamp("2023-06-01")  # Juni = data test
SEED, WINDOW = 42, 48
STORES = ["Astoria", "Hell's Kitchen", "Lower Manhattan"]
MODELS = {"seasonal_naive": "Seasonal Naive", "random_forest": "Random Forest", "lstm": "LSTM"}
RF_DEFAULT = {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 3}
RF_GRID = {"n_estimators": [100, 300], "max_depth": [None, 10, 20], "min_samples_leaf": [1, 3, 5, 10]}
RF_FEATURES = ["jam", "hari_ke", *[f"cabang_{s}" for s in STORES], "lag_1", "lag_24", "lag_168", "roll_mean_24"]


# ---------------- Data & fitur ----------------

def hourly_series(df: pd.DataFrame) -> pd.DataFrame:
    """Revenue per jam per cabang pada grid 24 jam penuh (jam tanpa transaksi = 0).

    Kolom `buka` menandai jam buka tiap cabang (jam pertama–terakhir yang pernah ada transaksi di data train).
    Model hanya dilatih & dievaluasi di jam buka; jam tutup tetap ada di grid agar lag 24/168 = jam yang sama
    kemarin/minggu lalu.
    """
    assert sorted(df["store_location"].unique()) == STORES
    ts = (df["tanggal"] + pd.to_timedelta(df["jam"], unit="h")).rename("ts")
    rev = df.groupby([df["store_location"], ts])["revenue"].sum()
    hours = pd.date_range(df["tanggal"].min(), df["tanggal"].max() + pd.Timedelta(hours=23), freq="h")
    idx = pd.MultiIndex.from_product([STORES, hours], names=["store_location", "ts"])
    h = rev.reindex(idx, fill_value=0.0).rename("revenue").reset_index()
    h["jam"], h["hari_ke"] = h["ts"].dt.hour, h["ts"].dt.dayofweek
    jam_buka = df[df["tanggal"] < TEST_START].groupby("store_location")["jam"].agg(["min", "max"])
    h["buka"] = h["jam"].between(h["store_location"].map(jam_buka["min"]), h["store_location"].map(jam_buka["max"]))
    return h


def add_lag_features(h: pd.DataFrame) -> pd.DataFrame:
    """Fitur Random Forest. Semua fitur revenue di-shift minimal 1 jam, jadi hanya memakai masa lalu (tanpa leakage)."""
    h = h.copy()
    g = h.groupby("store_location")["revenue"]
    for k in (1, 24, 168):
        h[f"lag_{k}"] = g.shift(k)
    h["roll_mean_24"] = g.transform(lambda s: s.shift(1).rolling(24).mean())
    for s in STORES:
        h[f"cabang_{s}"] = (h["store_location"] == s).astype(int)
    return h


def split(h: pd.DataFrame):
    """Split berdasarkan waktu (bukan acak): train Januari–Mei, test Juni. Hanya jam buka."""
    o = h[h["buka"]]
    return o[o["ts"] < TEST_START], o[o["ts"] >= TEST_START]


# ---------------- Random Forest ----------------

def _rf_rows(train: pd.DataFrame) -> pd.DataFrame:
    # minggu pertama Januari belum punya lag 168; urut waktu agar TimeSeriesSplit memotong berdasarkan waktu
    return train.dropna(subset=RF_FEATURES).sort_values(["ts", "store_location"])


def train_rf(train: pd.DataFrame, **params) -> RandomForestRegressor:
    # n_jobs=1: urutan penjumlahan antar-pohon tetap, jadi hasil identik persis antar-run
    tr = _rf_rows(train)
    rf = RandomForestRegressor(**{**RF_DEFAULT, **params}, n_jobs=1, random_state=SEED)
    return rf.fit(tr[RF_FEATURES], tr["revenue"])


def tune_rf(train: pd.DataFrame, n_splits: int = 5) -> GridSearchCV:
    """Grid search RF_GRID dengan TimeSeriesSplit di data train saja (fold validasi selalu sesudah fold latih).

    Model terbaik di-refit ke seluruh data train → `best_estimator_`.
    """
    tr = _rf_rows(train)
    gs = GridSearchCV(RandomForestRegressor(n_jobs=1, random_state=SEED), RF_GRID, cv=TimeSeriesSplit(n_splits),
                      scoring="neg_root_mean_squared_error", n_jobs=-1)  # paralel antar-kombinasi, tiap fit tetap deterministik
    return gs.fit(tr[RF_FEATURES], tr["revenue"])


# ---------------- LSTM ----------------

def _time_feats(ts: pd.Series) -> np.ndarray:
    """sin/cos jam & hari, supaya jam 23→0 dan Minggu→Senin dianggap berdekatan."""
    jam, hari = 2 * np.pi * ts.dt.hour / 24, 2 * np.pi * ts.dt.dayofweek / 7
    return np.column_stack([np.sin(jam), np.cos(jam), np.sin(hari), np.cos(hari)]).astype(np.float32)


def lstm_dataset(h: pd.DataFrame, scaler: dict):
    """Satu sampel per jam buka yang punya 48 jam riwayat.

    X: window 48 jam terakhir (revenue ter-normalisasi + sin/cos jam) → (n, 48, 3)
    S: fitur jam target (sin/cos jam & hari) + one-hot cabang → (n, 7)
    y: revenue ter-normalisasi jam target; keys: index baris `h` untuk tiap sampel.
    """
    X, S, y, keys = [], [], [], []
    for k, s in enumerate(STORES):
        g = h[h["store_location"] == s]
        r = ((g["revenue"].to_numpy() - scaler["mean"]) / scaler["std"]).astype(np.float32)
        tf = _time_feats(g["ts"])
        step = np.column_stack([r, tf[:, :2]])
        win = np.lib.stride_tricks.sliding_window_view(step, (WINDOW, step.shape[1]))[:, 0]  # win[j] = step[j:j+48]
        idx = np.flatnonzero(g["buka"].to_numpy())
        idx = idx[idx >= WINDOW]
        X.append(win[idx - WINDOW])  # hanya jam sebelum target → tanpa leakage
        S.append(np.column_stack([tf[idx], np.tile(np.eye(len(STORES), dtype=np.float32)[k], (len(idx), 1))]))
        y.append(r[idx])
        keys.append(g.index[idx])
    return (torch.from_numpy(np.concatenate(X)), torch.from_numpy(np.concatenate(S)),
            torch.from_numpy(np.concatenate(y)), np.concatenate(keys))


class LSTMForecaster(nn.Module):
    def __init__(self, n_step: int = 3, n_static: int = 7, hidden: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(n_step, hidden, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden + n_static, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x, s):
        _, (hn, _) = self.lstm(x)  # hidden state terakhir = ringkasan 48 jam
        return self.head(torch.cat([hn[-1], s], dim=1)).squeeze(1)


def train_lstm(h: pd.DataFrame, max_epochs: int = 100, patience: int = 10, batch: int = 128, lr: float = 1e-3):
    """Latih LSTM dengan early stopping pada validation loss → (model terbaik, riwayat loss, scaler)."""
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    rng = np.random.default_rng(SEED)

    fit = h.loc[h["ts"] < VAL_START, "revenue"]  # scaler di-fit hanya di data train (tanpa validation & test)
    scaler = {"mean": float(fit.mean()), "std": float(fit.std())}
    X, S, y, keys = lstm_dataset(h, scaler)
    ts = h.loc[keys, "ts"].to_numpy()
    tr = torch.from_numpy(ts < VAL_START)
    va = torch.from_numpy((ts >= VAL_START) & (ts < TEST_START))
    tr_idx = np.flatnonzero(tr.numpy())

    model = LSTMForecaster()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    best, best_state, wait, hist = np.inf, None, 0, []
    for epoch in range(1, max_epochs + 1):
        model.train()
        perm = rng.permutation(tr_idx)
        for i in range(0, len(perm), batch):
            b = torch.from_numpy(perm[i:i + batch])
            opt.zero_grad()
            loss_fn(model(X[b], S[b]), y[b]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            lt, lv = (loss_fn(model(X[m], S[m]), y[m]).item() for m in (tr, va))
        hist.append({"epoch": epoch, "train_loss": lt, "val_loss": lv})
        if lv < best:
            best, best_state, wait = lv, copy.deepcopy(model.state_dict()), 0
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    loss = pd.DataFrame(hist)
    loss["best"] = loss["val_loss"].eq(best)
    return model, loss, scaler


def predict_lstm(model: LSTMForecaster, h: pd.DataFrame, scaler: dict) -> pd.Series:
    X, S, _, keys = lstm_dataset(h, scaler)
    model.eval()
    with torch.no_grad():
        p = model(X, S).numpy() * scaler["std"] + scaler["mean"]
    return pd.Series(np.clip(p, 0, None), index=keys)  # revenue tidak mungkin negatif


# ---------------- Evaluasi ----------------

def build_results(h, test, rf, lstm, scaler) -> pd.DataFrame:
    """Aktual vs prediksi ketiga model di data test (prediksi 1 jam ke depan, riwayat = data aktual)."""
    res = test[["ts", "store_location", "jam"]].assign(
        tanggal=test["ts"].dt.normalize(), aktual=test["revenue"],
        seasonal_naive=test["lag_168"],  # nilai jam yang sama 7 hari sebelumnya
        random_forest=rf.predict(test[RF_FEATURES]),
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


# ---------------- Pipeline lengkap ----------------

def save_loss_plot(loss: pd.DataFrame, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(loss["epoch"], loss["train_loss"], label="Train")
    ax.plot(loss["epoch"], loss["val_loss"], label="Validation")
    ax.axvline(loss.loc[loss["best"], "epoch"].iloc[0], color="gray", ls="--", label="Epoch terbaik")
    ax.set(xlabel="Epoch", ylabel="MSE (skala ter-normalisasi)", title="Loss LSTM: train vs validation")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def run(save: bool = True) -> dict:
    """Latih ulang semua model dari awal, evaluasi di Juni, dan (opsional) simpan model & hasil."""
    h = add_lag_features(hourly_series(P.add_features(P.load_data())))
    train, test = split(h)
    assert train["ts"].max() < TEST_START <= test["ts"].min()
    gs = tune_rf(train)
    rf = gs.best_estimator_
    rf_default = train_rf(train)  # pembanding: parameter sebelum tuning
    tuning = {"best_params": gs.best_params_, "cv_rmse": -gs.best_score_, "default_params": RF_DEFAULT,
              "test_rmse_default": scores(test["revenue"].to_numpy(), rf_default.predict(test[RF_FEATURES]))["RMSE"]}
    lstm, loss, scaler = train_lstm(h)
    res = build_results(h, test, rf, lstm, scaler)
    metrics = evaluate(res)
    if save:
        MODELS_DIR.mkdir(exist_ok=True)
        joblib.dump(rf, MODELS_DIR / "random_forest.joblib", compress=3)
        torch.save(lstm.state_dict(), MODELS_DIR / "lstm.pt")
        (MODELS_DIR / "lstm_config.json").write_text(json.dumps({"scaler": scaler, "window": WINDOW, "stores": STORES}))
        save_loss_plot(loss, MODELS_DIR / "lstm_loss.png")
        loss.to_csv(LOSS, index=False)
        res.to_parquet(RESULTS, index=False)
        metrics.to_csv(METRICS, index=False)
        TUNING.write_text(json.dumps(tuning, indent=2))
    return {"h": h, "train": train, "test": test, "rf": rf, "tuning": tuning, "lstm": lstm, "loss": loss,
            "scaler": scaler, "results": res, "metrics": metrics}
