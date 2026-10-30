"""Deep Learning: LSTM (PyTorch, CPU) dengan window 48 jam dan early stopping."""
import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from .data import SEED, STORES, TEST_START, VAL_START

WINDOW = 48


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
