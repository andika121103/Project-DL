"""Data & fitur bersama semua model: deret revenue per jam per cabang, fitur lag, dan split train/test."""
import pandas as pd

VAL_START = pd.Timestamp("2023-05-18")   # 2 minggu terakhir data train → validation LSTM
TEST_START = pd.Timestamp("2023-06-01")  # Juni = data test
SEED = 42
STORES = ["Astoria", "Hell's Kitchen", "Lower Manhattan"]


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
