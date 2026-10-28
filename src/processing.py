"""Pengolahan & agregasi data Coffee Shop Sales (Maven Roasters)."""
from pathlib import Path

import pandas as pd
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
HARI = ["Senin", "Selasa", "Rabu", "Kamis", "Jumat", "Sabtu", "Minggu"]
BULAN = ["Januari", "Februari", "Maret", "April", "Mei", "Juni",
         "Juli", "Agustus", "September", "Oktober", "November", "Desember"]


PARQUET = DATA_DIR / "coffee_sales.parquet"


def _find_file() -> Path:
    """Parquet (cepat) diutamakan; fallback ke Excel/CSV mentah."""
    if PARQUET.exists():
        return PARQUET
    return _find_raw_file()


def _find_raw_file() -> Path:
    files = sorted(DATA_DIR.glob("*.xlsx")) + sorted(DATA_DIR.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"Tidak ada file .parquet/.xlsx/.csv di {DATA_DIR}")
    return files[0]


def _parse_date(s: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):  # serial Excel (hari sejak 1899-12-30)
        return pd.to_datetime(s, unit="D", origin="1899-12-30")
    return pd.to_datetime(s.astype(str).str.strip(), dayfirst=True, format="mixed")


def _parse_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_timedelta64_dtype(s):
        return s
    if pd.api.types.is_numeric_dtype(s):  # pecahan hari dari Excel
        return pd.to_timedelta(s, unit="D")
    return pd.to_timedelta(s.astype(str).str.strip())


def _parse_price(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(r"[$,\s]", "", regex=True), errors="raise")


def load_data(path: str | Path | None = None) -> pd.DataFrame:
    """Baca file, parsing tanggal & waktu, bersihkan unit_price."""
    path = Path(path) if path else _find_file()
    readers = {".parquet": pd.read_parquet, ".xlsx": pd.read_excel, ".csv": pd.read_csv}
    df = readers[path.suffix](path)
    df["transaction_date"] = _parse_date(df["transaction_date"])
    df["transaction_time"] = _parse_time(df["transaction_time"])
    df["unit_price"] = _parse_price(df["unit_price"])
    df["transaction_qty"] = pd.to_numeric(df["transaction_qty"], errors="raise")
    return df


def convert_to_parquet() -> Path:
    """Simpan data mentah yang sudah diparsing ke data/coffee_sales.parquet."""
    load_data(_find_raw_file()).to_parquet(PARQUET, index=False)
    return PARQUET


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Tambah revenue dan fitur waktu."""
    df = df.copy()
    df["revenue"] = df["transaction_qty"] * df["unit_price"]
    df["tanggal"] = df["transaction_date"].dt.normalize()
    df["bulan"] = df["tanggal"].dt.month
    df["nama_bulan"] = pd.Categorical(df["bulan"].map(lambda m: BULAN[m - 1]),
                                      categories=BULAN, ordered=True)
    df["minggu"] = df["tanggal"].dt.to_period("W-SUN").dt.start_time  # minggu dimulai Senin
    df["hari"] = pd.Categorical(df["tanggal"].dt.dayofweek.map(lambda d: HARI[d]),
                                categories=HARI, ordered=True)
    df["jam"] = (df["transaction_time"].dt.total_seconds() // 3600).astype(int)
    return df


@st.cache_data(show_spinner="Memuat data...")
def get_data() -> pd.DataFrame:
    return add_features(load_data())


# ---------------- Agregasi ----------------

def _agg(df: pd.DataFrame, by) -> pd.DataFrame:
    return (df.groupby(by, observed=True)
              .agg(revenue=("revenue", "sum"),
                   transaksi=("transaction_id", "count"),
                   item=("transaction_qty", "sum"))
              .reset_index())


def per_hari(df):
    out = _agg(df, "tanggal")
    out["ma7"] = moving_average(out)
    return out


def per_minggu(df):
    return _agg(df, "minggu")


def per_bulan(df):
    out = _agg(df, ["bulan", "nama_bulan"]).sort_values("bulan")
    out["mom_pct"] = mom_growth(out)
    return out


def mom_growth(bulanan: pd.DataFrame) -> pd.Series:
    """Pertumbuhan revenue bulan ke bulan (%)."""
    return bulanan["revenue"].pct_change() * 100


def bulan_lengkap(start, end) -> list[int]:
    """Nomor bulan yang seluruh harinya tercakup dalam rentang [start, end]."""
    start, end = pd.Timestamp(start), pd.Timestamp(end)
    return [p.month for p in pd.period_range(start, end, freq="M")
            if p.start_time >= start and p.end_time.normalize() <= end]


def mom_bulan_lengkap(bulanan: pd.DataFrame, start, end):
    """MoM % antara dua bulan lengkap terakhir → (pct, bulan, bulan_sebelumnya), atau None."""
    full = bulanan[bulanan["bulan"].isin(bulan_lengkap(start, end))]
    if len(full) < 2:
        return None
    pct = (full["revenue"].iloc[-1] / full["revenue"].iloc[-2] - 1) * 100
    return pct, full["nama_bulan"].iloc[-1], full["nama_bulan"].iloc[-2]


def mom_pct_lengkap(bulanan: pd.DataFrame, start, end) -> pd.Series:
    """mom_pct hanya untuk bulan lengkap yang bulan sebelumnya juga lengkap; selainnya NaN."""
    full = bulanan["bulan"].isin(bulan_lengkap(start, end))
    return bulanan["mom_pct"].where(full & full.shift(fill_value=False) & bulanan["bulan"].diff().eq(1))


def moving_average(harian: pd.DataFrame, window: int = 7) -> pd.Series:
    """Moving average revenue harian (min_periods=1 agar awal periode tetap terisi)."""
    return harian["revenue"].rolling(window, min_periods=1).mean()


def per_cabang_bulan(df):
    return _agg(df, ["store_location", "bulan", "nama_bulan"]).sort_values(["store_location", "bulan"])


def per_cabang(df):
    return _agg(df, "store_location").sort_values("revenue", ascending=False)


def per_kategori(df):
    return _agg(df, "product_category").sort_values("revenue", ascending=False)


def top_produk(df, n: int = 10):
    return _agg(df, ["product_detail", "product_category"]).nlargest(n, "revenue")


def per_kategori_tipe(df):
    return _agg(df, ["product_category", "product_type"])


def per_hari_nama(df):
    return _agg(df, "hari")


def per_jam(df):
    return _agg(df, "jam")


def heatmap_hari_jam(df) -> pd.DataFrame:
    """Pivot total revenue: baris = hari (Senin–Minggu), kolom = jam."""
    return df.pivot_table(index="hari", columns="jam", values="revenue",
                          aggfunc="sum", fill_value=0, observed=False)


def avg_transaction_value(df) -> float:
    return df["revenue"].sum() / len(df) if len(df) else 0.0


if __name__ == "__main__":
    import sys
    if "--parquet" in sys.argv:
        print("Parquet ditulis ke", convert_to_parquet())

    # Self-check: parquet (jika ada) identik dengan Excel, dan total agregasi = sum(qty * unit_price) mentah
    raw = load_data()
    if PARQUET.exists() and any(DATA_DIR.glob("*.xlsx")) | any(DATA_DIR.glob("*.csv")):  # clone dari git tanpa Excel
        pd.testing.assert_frame_equal(raw, load_data(_find_raw_file()), check_dtype=False)
    data = add_features(raw)
    expected = (raw["transaction_qty"] * raw["unit_price"]).sum()
    for name, agg in [("hari", per_hari), ("minggu", per_minggu), ("bulan", per_bulan),
                      ("cabang", per_cabang_bulan), ("kategori", per_kategori),
                      ("hari_nama", per_hari_nama), ("jam", per_jam)]:
        got = agg(data)["revenue"].sum()
        assert abs(got - expected) < 1e-6, (name, got, expected)
    assert abs(heatmap_hari_jam(data).to_numpy().sum() - expected) < 1e-6
    assert per_hari(data)["transaksi"].sum() == len(raw)
    assert _parse_price(pd.Series(["$3.00", "$1,234.50"])).tolist() == [3.0, 1234.5]
    assert _parse_date(pd.Series(["13/01/2023"])).iloc[0] == pd.Timestamp("2023-01-13")
    assert _parse_date(pd.Series([44927])).iloc[0] == pd.Timestamp("2023-01-01")
    assert bulan_lengkap("2023-01-01", "2023-06-30") == [1, 2, 3, 4, 5, 6]
    assert bulan_lengkap("2023-01-15", "2023-03-31") == [2, 3]
    assert bulan_lengkap("2023-03-02", "2023-04-29") == []
    b = per_bulan(data)
    assert mom_bulan_lengkap(b, "2023-01-15", "2023-03-31")[1:] == ("Maret", "Februari")
    assert mom_bulan_lengkap(b, "2023-02-10", "2023-04-15") is None  # hanya Maret yang lengkap
    b = per_bulan(data[data["tanggal"].between("2023-01-15", "2023-03-31")])
    assert mom_pct_lengkap(b, "2023-01-15", "2023-03-31").notna().tolist() == [False, False, True]
    b = per_bulan(data[data["tanggal"].between("2023-02-10", "2023-04-15")])
    assert mom_pct_lengkap(b, "2023-02-10", "2023-04-15").notna().tolist() == [False, False, False]
    print(f"OK — total revenue ${expected:,.2f}, {len(raw):,} transaksi")
