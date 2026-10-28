# Dashboard Tren Penjualan Coffee Shop — Kelompok 04

Tugas kelompok mata kuliah **Analisis Data**.

| | |
|---|---|
| **Materi** | Agregasi data tabular historis |
| **Aplikasi** | Dashboard analitik visual interaktif |
| **Proyek** | Visualisasi grafik tren penjualan |

## Deskripsi

Proyek ini menganalisis dataset **Coffee Shop Sales (Maven Roasters)** dari Kaggle: 149.116 transaksi dari 3 cabang
coffee shop di New York City (Astoria, Hell's Kitchen, Lower Manhattan) selama Januari–Juni 2023.

Data mentah diagregasi per hari, minggu, bulan, cabang, kategori produk, serta pola hari × jam. Hasilnya ditampilkan
dalam dashboard interaktif berbasis **Streamlit + Plotly**, yang berisi:

- Filter rentang tanggal, cabang, dan kategori produk. Semua grafik dan KPI mengikuti filter.
- KPI: total revenue, jumlah transaksi, rata-rata per transaksi, total item, dan pertumbuhan MoM.
- 5 tab: **Tren Penjualan**, **Perbandingan Cabang**, **Pola Waktu**, **Produk**, **Data** (tabel + download CSV).
- Insight otomatis di setiap tab yang dihitung langsung dari data.

## Struktur folder

```
analisis-data-kelompok04/
├── data/
│   ├── coffee_sales.parquet     # dataset siap pakai (dibaca dashboard, cepat)
│   ├── Coffee Shop Sales.xlsx   # dataset asli dari Kaggle (fallback, tidak di-commit)
│   ├── forecast_results.parquet # prediksi test Juni ketiga model (dibaca tab Prediksi)
│   ├── model_metrics.csv        # MAE/RMSE/MAPE per model × cabang
│   ├── lstm_loss.csv            # riwayat loss train vs validation LSTM
│   └── rf_tuning.json           # parameter terbaik Random Forest hasil TimeSeriesSplit
├── models/                      # random_forest.joblib, lstm.pt, lstm_config.json, lstm_loss.png
├── src/
│   ├── processing.py            # load, cleaning, feature engineering, agregasi
│   └── forecasting.py           # forecasting per jam: seasonal naive, Random Forest, LSTM
├── notebooks/
│   ├── eda.ipynb                # analisis langkah demi langkah (bahan laporan)
│   └── modeling.ipynb           # penjelasan & hasil Machine Learning / Deep Learning
├── .streamlit/config.toml       # warna tema dashboard
├── app.py                       # dashboard Streamlit
├── train_models.py              # latih ulang semua model dari awal
├── requirements.txt             # untuk dashboard / deploy (tanpa torch)
├── requirements-train.txt       # untuk training (torch CPU, scikit-learn)
└── README.md
```

## Instalasi

Membutuhkan Python 3.11 atau lebih baru (diuji dengan 3.13).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

Untuk menjalankan notebook, tambahkan: `pip install jupyter`.

## Menjalankan

```bash
streamlit run app.py
```

Dashboard akan terbuka di `http://localhost:8501`. Pemuatan data pertama sekitar 1,5 detik dari file parquet
(sekitar 9,5 detik jika hanya ada file Excel). Setelah itu data di-cache (`@st.cache_data`) sehingga perubahan filter terasa cepat.

`load_data()` membaca `data/coffee_sales.parquet` jika ada, dan otomatis kembali ke file Excel/CSV jika tidak ada.
Untuk membuat ulang file parquet dari Excel (misalnya setelah dataset diganti), jalankan:

```bash
python -m src.processing --parquet
```

Untuk memeriksa bahwa isi parquet identik dengan Excel dan total agregasi sama dengan `sum(qty × unit_price)` pada data
mentah, jalankan:

```bash
python -m src.processing
```

Notebook analisis: buka `notebooks/eda.ipynb` dengan Jupyter / VS Code, lalu jalankan semua sel (*Run All*).

## Machine Learning & Deep Learning: forecasting revenue per jam

Tugas: memprediksi revenue **1 jam ke depan per cabang**. Train Januari–Mei 2023 (2 minggu terakhir = validation LSTM),
test Juni 2023, split berdasarkan waktu.

| Model | Jenis | MAE/jam | RMSE/jam | MAPE/hari |
|---|---|---|---|---|
| Seasonal naive (jam yang sama 7 hari lalu) | baseline | \$70,60 | \$99,63 | 17,9% |
| **Random Forest** (scikit-learn, tuned) | Machine Learning | **\$40,03** | **\$58,62** | **9,5%** |
| LSTM (PyTorch, CPU) | Deep Learning | \$41,22 | \$59,55 | 10,4% |

Random Forest sedikit lebih baik dari LSTM, dan keduanya jauh lebih baik dari baseline. Hyperparameter Random Forest
di-*tuning* dengan grid search + `TimeSeriesSplit` (5 fold) di data train saja; terpilih `n_estimators=300, max_depth=10,
min_samples_leaf=5`, tapi RMSE test per jam hanya turun dari \$58,69 ke \$58,62. Penjelasan lengkap ada di
`notebooks/modeling.ipynb`. Training deterministik (seed tetap, `n_jobs=1` di Random Forest), jadi hasilnya identik setiap dijalankan.

Melatih ulang, termasuk tuning (±80 detik di CPU):

```bash
pip install -r requirements-train.txt
python train_models.py
```

Hasilnya ditulis ke `data/forecast_results.parquet`, `data/model_metrics.csv`, `data/lstm_loss.csv`, `data/rf_tuning.json`, dan folder `models/`.
Tab **Prediksi** di dashboard hanya membaca file-file tersebut, jadi deploy tidak membutuhkan torch/scikit-learn.

## Deploy ke Streamlit Community Cloud

1. Upload folder proyek ke repository GitHub (publik atau privat), **termasuk** file `data/coffee_sales.parquet`
   (±2 MB). File ini sengaja tidak di-ignore oleh `.gitignore` karena dibutuhkan saat deploy. File Excel (±9 MB)
   di-ignore; unduh dari Kaggle ke `data/` jika ingin membuat ulang parquet.
2. Buka [share.streamlit.io](https://share.streamlit.io) dan login dengan akun GitHub.
3. Klik **Create app** → **Deploy a public app from GitHub**.
4. Pilih repository dan branch, lalu isi *Main file path* dengan `app.py`. Di *Advanced settings* pilih Python 3.13.
5. Klik **Deploy**. Streamlit akan otomatis meng-install paket dari `requirements.txt`.
6. Setiap `git push` ke branch tersebut akan memperbarui aplikasi secara otomatis.

## Catatan data

- Tidak ada missing value maupun baris duplikat.
- `transaction_id` unik per baris, sehingga **1 baris = 1 transaksi**.
- `revenue = transaction_qty × unit_price`.
- KPI pertumbuhan MoM hanya membandingkan dua bulan terakhir yang **seluruh harinya** tercakup dalam rentang filter.
  Jika belum ada dua bulan penuh, KPI menampilkan "—".
- Parsing di `load_data()` aman untuk tanggal format Excel, serial Excel, maupun `dd/mm/yyyy`, serta `unit_price`
  yang mengandung simbol `$`.
