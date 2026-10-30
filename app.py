import json

import pandas as pd
import plotly.express as px
import streamlit as st

from src import processing as P

st.set_page_config(page_title="Dashboard Tren Penjualan Coffee Shop", page_icon="☕", layout="wide")

BLUE = "#2a78d6"
STORE_COLORS = {"Astoria": "#2a78d6", "Hell's Kitchen": "#eb6834", "Lower Manhattan": "#1baf7a"}
MUTED = "#8a8985"
MODEL_COLORS = {"Aktual": BLUE, "Seasonal Naive": MUTED, "Random Forest": "#eb6834", "LSTM": "#1baf7a"}
SEQ = "Blues"


def usd(x: float) -> str:
    return f"${x:,.2f}"


def tgl(d) -> str:
    return f"{d.day} {P.BULAN[d.month - 1]} {d.year}"


def style(fig, y_money=True, height=380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=90, b=10), title_y=0.97,
                      font_family="Inter, sans-serif", title_font=dict(size=16, weight=600),
                      paper_bgcolor="rgba(0,0,0,0)",
                      plot_bgcolor="rgba(0,0,0,0)", hovermode="x unified" if fig.data and fig.data[0].type == "scatter" else "closest",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None))
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.2)", zeroline=False)
    if y_money:
        fig.update_yaxes(tickprefix="$", tickformat=",.0f")
    return fig


def show(fig):
    with st.container(border=True):
        st.plotly_chart(fig, width="stretch")


def insight(*lines, saran=None):
    text = "**Insight**\n\n" + "\n".join(f"- {l}" for l in lines if l) + (f"\n\n**💡 Saran:** {saran}" if saran else "")
    st.info(text.replace("$", r"\$"))  # cegah $...$ dibaca sebagai LaTeX


def pct(a, b) -> float:
    return (a / b - 1) * 100


# ---------------- Data & filter ----------------
df = P.get_data()

st.sidebar.header("Filter")
dmin, dmax = df["tanggal"].min().date(), df["tanggal"].max().date()
rng = st.sidebar.date_input("Rentang tanggal", (dmin, dmax), min_value=dmin, max_value=dmax)
start, end = (rng[0], rng[-1]) if isinstance(rng, (tuple, list)) and rng else (dmin, dmax)
stores = st.sidebar.multiselect("Cabang", sorted(df["store_location"].unique()),
                                default=sorted(df["store_location"].unique()))
cats = st.sidebar.multiselect("Kategori produk", sorted(df["product_category"].unique()),
                              default=sorted(df["product_category"].unique()))

f = df[df["tanggal"].between(pd.Timestamp(start), pd.Timestamp(end))
       & df["store_location"].isin(stores) & df["product_category"].isin(cats)]

st.title("☕ Dashboard Tren Penjualan Coffee Shop")
st.caption(f"Maven Roasters · 3 cabang di New York City · {tgl(start)} – {tgl(end)}")

if f.empty:
    st.warning("Tidak ada data untuk kombinasi filter ini. Silakan ubah rentang tanggal, cabang, atau kategori.")
    st.stop()

harian, mingguan, bulanan = P.per_hari(f), P.per_minggu(f), P.per_bulan(f)
cabang, cabang_bulan = P.per_cabang(f), P.per_cabang_bulan(f)
kategori, top10 = P.per_kategori(f), P.top_produk(f)
by_hari, by_jam, heat = P.per_hari_nama(f), P.per_jam(f), P.heatmap_hari_jam(f)

# ---------------- KPI ----------------
mom = P.mom_bulan_lengkap(bulanan, start, end)
k = st.columns(5)
spark = dict(border=True, chart_type="area")  # sparkline = tren mingguan dalam rentang filter
rev = f["revenue"].sum()
k[0].metric("Total Revenue", f"${rev / 1e3:,.1f}K" if rev >= 1e5 else usd(rev), help=usd(rev),
            chart_data=mingguan["revenue"], **spark)
k[1].metric("Transaksi", f"{len(f):,}", chart_data=mingguan["transaksi"], **spark)
k[2].metric("Nilai/Transaksi", usd(P.avg_transaction_value(f)), help="Rata-rata revenue per transaksi",
            chart_data=mingguan["revenue"] / mingguan["transaksi"], **spark)
k[3].metric("Item Terjual", f"{int(f['transaction_qty'].sum()):,}", chart_data=mingguan["item"], **spark)
mom_spark = dict(border=True, chart_type="bar", chart_data=bulanan["revenue"])
if mom:
    k[4].metric("MoM", f"{mom[0]:+.1f}%", **mom_spark,
                help=f"{mom[1]} vs {mom[2]}. Hanya membandingkan bulan yang seluruh harinya tercakup dalam rentang filter.")
else:
    k[4].metric("MoM", "—", **mom_spark, help="Butuh minimal 2 bulan penuh dalam rentang tanggal.")

tabs = st.tabs(["📈 Tren Penjualan", "🏪 Perbandingan Cabang", "🕒 Pola Waktu", "🥐 Produk", "📋 Data", "🔮 Prediksi"])

# ---------------- Tab 1 ----------------
with tabs[0]:
    gran = st.radio("Granularitas", ["Harian", "Mingguan", "Bulanan"], horizontal=True)
    if gran == "Harian":
        fig = px.line(harian, x="tanggal", y="revenue", title="Revenue Harian & Moving Average 7 Hari",
                      color_discrete_sequence=[BLUE], labels={"tanggal": "Tanggal", "revenue": "Revenue"})
        fig.data[0].name, fig.data[0].showlegend = "Revenue harian", True
        fig.data[0].line.width = 1
        fig.data[0].opacity = 0.5
        fig.add_scatter(x=harian["tanggal"], y=harian["ma7"], name="Moving average 7 hari",
                        line=dict(color=BLUE, width=2.5))
    elif gran == "Mingguan":
        fig = px.line(mingguan, x="minggu", y="revenue", markers=True, title="Revenue Mingguan",
                      color_discrete_sequence=[BLUE], labels={"minggu": "Minggu mulai", "revenue": "Revenue"})
    else:
        fig = px.line(bulanan, x="nama_bulan", y="revenue", markers=True, title="Revenue Bulanan",
                      color_discrete_sequence=[BLUE], labels={"nama_bulan": "Bulan", "revenue": "Revenue"})
    fig.update_traces(hovertemplate="%{y:$,.2f}")
    show(style(fig))

    full = bulanan["bulan"].isin(P.bulan_lengkap(start, end))
    mom_ok = P.mom_pct_lengkap(bulanan, start, end)
    lbl = [usd(r) + ("" if pd.isna(m) else f"<br>{m:+.1f}% MoM") for r, m in zip(bulanan["revenue"], mom_ok)]
    xb = bulanan.assign(nama_bulan=bulanan["nama_bulan"].astype(str) + full.map({True: "", False: " (sebagian)"}))
    fig = px.bar(xb, x="nama_bulan", y="revenue", text=lbl, title="Revenue per Bulan (dengan pertumbuhan MoM)",
                 color_discrete_sequence=[BLUE], labels={"nama_bulan": "Bulan", "revenue": "Revenue"})
    fig.update_traces(textposition="outside", hovertemplate="%{x}: %{y:$,.2f}<extra></extra>", cliponaxis=False,
                      marker_opacity=full.map({True: 1.0, False: 0.35}).tolist())
    show(style(fig, height=420))
    if not full.all():
        st.caption("Bulan bertanda *(sebagian)* hanya tercakup sebagian oleh rentang tanggal: batangnya dipudarkan "
                   "dan MoM hanya ditampilkan untuk bulan lengkap yang bulan sebelumnya juga lengkap.")

    # dinormalisasi per hari aktif, supaya bulan pendek (Februari) atau bulan sebagian tidak terlihat turun
    pb = bulanan.set_index("nama_bulan")[["revenue", "transaksi"]].div(f.groupby("nama_bulan", observed=True)["tanggal"].nunique(), axis=0)
    pb["atv"] = pb["revenue"] / pb["transaksi"]
    best_d, worst_d = harian.loc[harian["revenue"].idxmax()], harian.loc[harian["revenue"].idxmin()]
    lines, saran = [], None
    if len(pb) > 1:
        a, b = pb.iloc[0], pb.iloc[-1]
        lines.append(f"Revenue per hari naik dari {usd(a['revenue'])} ({pb.index[0]}) ke {usd(b['revenue'])} ({pb.index[-1]}), "
                     f"**{pct(b['revenue'], a['revenue']):+.1f}%**.")
        lines.append(f"Penyebabnya adalah **jumlah transaksi**: {a['transaksi']:,.0f} → {b['transaksi']:,.0f} per hari "
                     f"({pct(b['transaksi'], a['transaksi']):+.1f}%), sedangkan nilai per transaksi hampir tetap "
                     f"({usd(a['atv'])} → {usd(b['atv'])}, {pct(b['atv'], a['atv']):+.1f}%).")
        for (m0, r0), (m1, r1) in zip(bulanan[["nama_bulan", "revenue"]].values, bulanan[["nama_bulan", "revenue"]].values[1:]):
            if r1 < r0 and pb.loc[m1, "revenue"] > pb.loc[m0, "revenue"]:
                lines.append(f"Total {m1} turun {pct(r1, r0):.1f}% dari {m0} hanya karena jumlah harinya lebih sedikit; "
                             f"per hari, revenue {m1} justru naik {pct(pb.loc[m1, 'revenue'], pb.loc[m0, 'revenue']):+.1f}%.")
        if abs(pct(b["atv"], a["atv"])) < 5:
            saran = ("Pertumbuhan datang dari pelanggan yang lebih banyak, bukan belanja yang lebih besar. Ruang berikutnya "
                     "adalah menaikkan nilai per transaksi, misalnya paket kopi + bakery atau upsell ukuran Large.")
    lines.append(f"Hari tertinggi **{tgl(best_d['tanggal'])}** ({usd(best_d['revenue'])}), terendah {tgl(worst_d['tanggal'])} "
                 f"({usd(worst_d['revenue'])}).")
    insight(*lines, saran=saran)

# ---------------- Tab 2 ----------------
with tabs[1]:
    c1, c2 = st.columns([3, 2])
    fig = px.line(cabang_bulan, x="nama_bulan", y="revenue", color="store_location", markers=True,
                  color_discrete_map=STORE_COLORS, title="Tren Revenue Bulanan per Cabang",
                  labels={"nama_bulan": "Bulan", "revenue": "Revenue", "store_location": "Cabang"})
    fig.update_traces(hovertemplate="%{y:$,.2f}")
    with c1:
        show(style(fig))
    fig = px.bar(cabang, x="store_location", y="revenue", color="store_location", text=cabang["revenue"].map(usd),
                 color_discrete_map=STORE_COLORS, title="Total Revenue per Cabang",
                 labels={"store_location": "Cabang", "revenue": "Revenue"})
    fig.update_traces(textposition="outside", cliponaxis=False, hovertemplate="%{x}: %{y:$,.2f}<extra></extra>")
    with c2:
        show(style(fig).update_layout(showlegend=False))

    top, bot = cabang.iloc[0], cabang.iloc[-1]
    atv = (cabang.assign(atv=cabang["revenue"] / cabang["transaksi"]).sort_values("atv", ascending=False).iloc[0])
    lines = [f"Cabang dengan revenue tertinggi adalah **{top['store_location']}** ({usd(top['revenue'])}, "
             f"{top['revenue'] / cabang['revenue'].sum() * 100:.1f}% dari total).",
             f"Rata-rata nilai per transaksi tertinggi ada di **{atv['store_location']}** ({usd(atv['atv'])})."]
    saran = None
    if len(cabang) > 1:
        gap = pct(top["revenue"], bot["revenue"])
        lines.insert(1, f"Selisih dengan cabang terendah ({bot['store_location']}) hanya {usd(top['revenue'] - bot['revenue'])} "
                        f"({gap:.1f}%), jadi performa ketiga cabang **{'relatif seimbang' if gap < 10 else 'cukup timpang'}**.")
        cb = cabang_bulan.pivot(index="bulan", columns="store_location", values="revenue")
        if len(cb) > 1:
            g = pct(cb.iloc[-1], cb.iloc[0]).sort_values()
            lines.append(f"Pertumbuhan {cabang_bulan['nama_bulan'].iloc[0]}–{cabang_bulan['nama_bulan'].iloc[-1]} tercepat di "
                         f"**{g.index[-1]}** ({g.iloc[-1]:+.1f}%), paling lambat {g.index[0]} ({g.iloc[0]:+.1f}%).")
        pagi = (f[f["jam"].between(7, 10)].groupby("store_location")["revenue"].sum()
                / f.groupby("store_location")["revenue"].sum() * 100).sort_values()
        lines.append(f"**{pagi.index[-1]}** paling bergantung pada jam pagi 07–10 ({pagi.iloc[-1]:.1f}% revenue), "
                     f"sedangkan {pagi.index[0]} lebih merata sepanjang hari ({pagi.iloc[0]:.1f}%).")
        saran = (f"Prioritaskan staf dan stok pagi di {pagi.index[-1]}; untuk cabang ini, promo siang/sore adalah cara "
                 f"paling mudah menambah revenue di jam yang masih sepi.")
    insight(*lines, saran=saran)

# ---------------- Tab 3 ----------------
with tabs[2]:
    fig = px.imshow(heat, color_continuous_scale=SEQ, aspect="auto", title="Heatmap Revenue: Hari × Jam",
                    labels=dict(x="Jam", y="Hari", color="Revenue"))
    fig.update_traces(hovertemplate="%{y}, jam %{x}:00<br>%{z:$,.2f}<extra></extra>", xgap=2, ygap=2)
    fig.update_xaxes(dtick=1)
    show(style(fig, y_money=False, height=420).update_coloraxes(colorbar_tickprefix="$", colorbar_tickformat=",.0f"))

    c1, c2 = st.columns(2)
    fig = px.bar(by_hari, x="hari", y="revenue", title="Revenue per Hari", color_discrete_sequence=[BLUE],
                 labels={"hari": "Hari", "revenue": "Revenue"})
    fig.update_traces(hovertemplate="%{x}: %{y:$,.2f}<extra></extra>")
    with c1:
        show(style(fig))
    fig = px.bar(by_jam, x="jam", y="revenue", title="Revenue per Jam", color_discrete_sequence=[BLUE],
                 labels={"jam": "Jam", "revenue": "Revenue"})
    fig.update_traces(hovertemplate="Jam %{x}:00: %{y:$,.2f}<extra></extra>")
    fig.update_xaxes(dtick=1)
    with c2:
        show(style(fig))

    bj, bh = by_jam.loc[by_jam["revenue"].idxmax()], by_hari.loc[by_hari["revenue"].idxmax()]
    lh = by_hari.loc[by_hari["revenue"].idxmin()]
    pagi = f.loc[f["jam"].between(7, 10), "revenue"].sum() / f["revenue"].sum() * 100
    lines = [f"Jam tersibuk adalah **{int(bj['jam']):02d}:00–{int(bj['jam']) + 1:02d}:00** ({usd(bj['revenue'])}).",
             f"Sekitar **{pagi:.1f}%** revenue terjadi hanya dalam 4 jam pagi (07:00–10:59)."]
    akhir_pekan = harian.groupby(harian["tanggal"].dt.dayofweek >= 5)["revenue"].mean()
    datar = False
    if len(akhir_pekan) == 2:
        d = pct(akhir_pekan[True], akhir_pekan[False])
        datar = abs(d) < 5
        lines.append(f"Rata-rata revenue per hari akhir pekan {usd(akhir_pekan[True])} vs hari kerja {usd(akhir_pekan[False])} "
                     f"({d:+.1f}%)" + (": **hampir sama**, jadi pola mingguan relatif datar." if datar else "."))
    lines.append(f"Hari dengan total revenue tertinggi **{bh['hari']}**, terendah **{lh['hari']}**.")
    insight(*lines, saran=("Yang perlu diatur adalah jam, bukan hari. " if datar else "")
                          + f"Tambah kasir dan persiapan bahan sekitar jam {int(bj['jam']):02d}:00, "
                            "dan pakai jam sepi untuk promo atau persiapan stok.")

# ---------------- Tab 4 ----------------
with tabs[3]:
    c1, c2 = st.columns(2)
    fig = px.bar(kategori.sort_values("revenue"), x="revenue", y="product_category", orientation="h",
                 title="Revenue per Kategori Produk", color_discrete_sequence=[BLUE],
                 labels={"product_category": "Kategori", "revenue": "Revenue"})
    fig.update_traces(hovertemplate="%{y}: %{x:$,.2f}<extra></extra>")
    fig = style(fig, y_money=False).update_xaxes(tickprefix="$", tickformat=",.0f", showgrid=True,
                                                gridcolor="rgba(128,128,128,0.2)")
    with c1:
        show(fig)
    fig = px.bar(top10.sort_values("revenue"), x="revenue", y="product_detail", orientation="h",
                 title="Top 10 Produk (Revenue)", color_discrete_sequence=[BLUE], hover_data=["product_category"],
                 labels={"product_detail": "Produk", "revenue": "Revenue", "product_category": "Kategori"})
    fig = style(fig, y_money=False).update_xaxes(tickprefix="$", tickformat=",.0f", showgrid=True,
                                                gridcolor="rgba(128,128,128,0.2)")
    with c2:
        show(fig)

    tm = P.per_kategori_tipe(f)
    fig = px.treemap(tm, path=[px.Constant("Semua"), "product_category", "product_type"], values="revenue",
                     color="revenue", color_continuous_scale=SEQ, title="Treemap Revenue: Kategori → Tipe Produk")
    fig.update_traces(hovertemplate="%{label}<br>%{value:$,.2f}<br>%{percentRoot:.1%} dari total<extra></extra>",
                      texttemplate="%{label}<br>%{value:$,.0f}")
    show(style(fig, y_money=False, height=520).update_coloraxes(showscale=False))

    kc, tp = kategori.iloc[0], top10.iloc[0]
    share = kategori.set_index("product_category")["revenue"] / kategori["revenue"].sum() * 100
    lines = [f"Kategori **{kc['product_category']}** menyumbang revenue terbesar "
             f"({usd(kc['revenue'])}, {share.iloc[0]:.1f}%)."
             + (f" Bersama **{share.index[1]}**, dua kategori ini mencakup **{share.iloc[:2].sum():.1f}%** revenue."
                if len(share) > 2 else "")]
    mix = f.groupby(["bulan", "product_category"])["revenue"].sum().unstack(fill_value=0)
    if len(mix) > 1 and mix.shape[1] > 1:
        lines.append(f"Komposisi kategori **stabil** dari bulan ke bulan: porsi tiap kategori bergeser paling banyak "
                     f"{(mix.div(mix.sum(axis=1), axis=0) * 100).agg(lambda s: s.max() - s.min()).max():.1f} poin persen, "
                     f"jadi kenaikan revenue terjadi merata di semua kategori.")
    lines += [f"Produk terlaris berdasarkan revenue adalah **{tp['product_detail']}** ({usd(tp['revenue'])}).",
              f"Top 10 dari {f['product_detail'].nunique()} produk menyumbang {top10['revenue'].sum() / f['revenue'].sum() * 100:.1f}% "
              f"revenue, jadi penjualan tidak bergantung pada segelintir produk."]
    insight(*lines, saran=f"Jaga ketersediaan {share.index[0]}" + (f" dan {share.index[1]}" if len(share) > 1 else "")
                          + "; bundling dengan Bakery bisa menaikkan nilai per transaksi.")

# ---------------- Tab 5 ----------------
with tabs[4]:
    tables = {
        "Revenue harian": harian, "Revenue mingguan": mingguan, "Revenue bulanan (MoM)": bulanan,
        "Cabang per bulan": cabang_bulan, "Total per cabang": cabang, "Per kategori": kategori,
        "Top 10 produk": top10, "Per hari": by_hari, "Per jam": by_jam,
        "Heatmap hari × jam": heat.rename(columns=lambda j: f"{j:02d}:00").reset_index(),
    }
    name = st.selectbox("Pilih tabel agregasi", list(tables))
    t = tables[name]
    money = {c: st.column_config.NumberColumn(format="dollar") for c in ["revenue", "ma7"] if c in t}
    if "mom_pct" in t:
        money["mom_pct"] = st.column_config.NumberColumn("MoM %", format="%.1f%%")
    st.dataframe(t, width="stretch", hide_index=True, column_config=money)
    st.download_button("⬇️ Download CSV", t.to_csv(index=False).encode("utf-8"),
                       file_name=f"{name.lower().replace(' ', '_').replace('×', 'x')}.csv", mime="text/csv")
    insight(f"Data terfilter berisi {len(f):,} transaksi pada {f['tanggal'].nunique()} hari aktif.",
            f"Rata-rata revenue harian {usd(harian['revenue'].mean())} dan rata-rata {harian['transaksi'].mean():,.0f} transaksi per hari.")

# ---------------- Tab 6 ----------------
# Hanya membaca file hasil train_models.py, tanpa torch/sklearn, supaya ringan di Streamlit Cloud.
@st.cache_data
def load_forecast():
    return (pd.read_parquet(P.RESULTS_DIR / "forecast_results.parquet"), pd.read_csv(P.RESULTS_DIR / "model_metrics.csv"),
            pd.read_csv(P.RESULTS_DIR / "lstm_loss.csv"), json.loads((P.RESULTS_DIR / "rf_tuning.json").read_text()))


with tabs[5]:
    if not (P.RESULTS_DIR / "forecast_results.parquet").exists():
        st.info("Belum ada hasil model. Jalankan `python train_models.py` terlebih dahulu.")
    else:
        res, met, loss, tun = load_forecast()
        names = {"aktual": "Aktual", "seasonal_naive": "Seasonal Naive", "random_forest": "Random Forest", "lstm": "LSTM"}
        st.caption("Forecasting revenue **per jam per cabang**, 1 jam ke depan. Model dilatih pada Januari–Mei 2023 "
                   "dan diuji pada Juni 2023. Tab ini tidak terpengaruh filter di sidebar.")

        c1, c2 = st.columns(2)
        cab = c1.selectbox("Cabang", ["Semua"] + sorted(res["store_location"].unique()))
        models = c2.multiselect("Model", list(names.values())[1:], default=list(names.values())[1:])

        st.subheader("Metrik di data test (Juni)")
        lvl = st.radio("Level evaluasi", ["Per jam", "Per hari (prediksi jam dijumlahkan)"], horizontal=True)
        m = met[(met["level"] == ("jam" if lvl == "Per jam" else "hari")) & (met["cabang"] == cab)].drop(columns=["level", "cabang"])
        st.dataframe(m, hide_index=True, width="stretch", column_config={
            "MAE": st.column_config.NumberColumn(format="dollar"), "RMSE": st.column_config.NumberColumn(format="dollar"),
            "MAPE": st.column_config.NumberColumn("MAPE", format="%.1f%%")})
        st.caption("MAE/RMSE dalam dolar (makin kecil makin baik). MAPE mengabaikan jam dengan revenue aktual 0. "
                   "Untuk cabang \"Semua\", metrik dihitung atas seluruh baris jam/hari dari ketiga cabang.")

        cols = ["aktual"] + [k for k, v in names.items() if v in models]
        r = res if cab == "Semua" else res[res["store_location"] == cab]
        mulai = pd.date_range("2023-06-01", "2023-06-30", freq="7D")
        w0 = st.selectbox("Minggu di bulan Juni", mulai, format_func=lambda d: f"{tgl(d)} – {tgl(min(d + pd.Timedelta(days=6), mulai[-1] + pd.Timedelta(days=1)))}")
        w = r[r["ts"].between(w0, w0 + pd.Timedelta(days=7), inclusive="left")].groupby("ts")[cols].sum()
        w = w.reindex(pd.date_range(w0, w0 + pd.Timedelta(days=7), freq="h", inclusive="left"))  # jam tutup = celah di garis
        fig = px.line(w.rename(columns=names).rename_axis("ts").reset_index().melt("ts", var_name="Model", value_name="Revenue"),
                      x="ts", y="Revenue", color="Model", color_discrete_map=MODEL_COLORS,
                      title=f"Aktual vs Prediksi per Jam · {cab}", labels={"ts": "Waktu"})
        fig.update_traces(hovertemplate="%{y:$,.2f}")
        show(style(fig, height=420))

        d = r.groupby("tanggal")[cols].sum().rename(columns=names).reset_index().melt("tanggal", var_name="Model", value_name="Revenue")
        fig = px.line(d, x="tanggal", y="Revenue", color="Model", markers=True, color_discrete_map=MODEL_COLORS,
                      title=f"Aktual vs Prediksi per Hari (Juni) · {cab}", labels={"tanggal": "Tanggal"})
        fig.update_traces(hovertemplate="%{y:$,.2f}")
        show(style(fig))

        lp = loss.melt("epoch", ["train_loss", "val_loss"], var_name="Data", value_name="Loss")
        fig = px.line(lp.replace({"train_loss": "Train", "val_loss": "Validation"}), x="epoch", y="Loss", color="Data",
                      color_discrete_map={"Train": BLUE, "Validation": "#eb6834"},
                      title="Loss LSTM: Train vs Validation (MSE, skala ter-normalisasi)", labels={"epoch": "Epoch"})
        best_ep = int(loss.loc[loss["best"], "epoch"].iloc[0])
        fig.add_vline(x=best_ep, line_dash="dash", line_color=MUTED, annotation_text=f"Epoch terbaik ({best_ep})")
        show(style(fig, y_money=False, height=340))

        mj = met[(met["level"] == "jam") & (met["cabang"] == "Semua")].set_index("model")
        mh = met[(met["level"] == "hari") & (met["cabang"] == "Semua")].set_index("model")
        best = mj["RMSE"].idxmin()
        bp = ", ".join(f"{k}={v}" for k, v in tun["best_params"].items())
        runner = mj.drop(index=[best, "Seasonal Naive"])["RMSE"].idxmin()
        gain = (1 - mj.loc[best, "RMSE"] / mj.loc["Seasonal Naive", "RMSE"]) * 100
        hb = mh["RMSE"].idxmin()
        insight(f"Model terbaik per jam adalah **{best}** (RMSE {usd(mj.loc[best, 'RMSE'])}, MAE {usd(mj.loc[best, 'MAE'])}), "
                f"RMSE-nya {gain:.1f}% lebih kecil dibanding baseline seasonal naive.",
                f"Selisihnya dengan {runner} hanya {usd(mj.loc[runner, 'RMSE'] - mj.loc[best, 'RMSE'])} RMSE, jadi keduanya hampir setara.",
                f"Di level harian, model terbaik adalah **{hb}**: rata-rata meleset {mh.loc[hb, 'MAPE']:.1f}% dari revenue "
                f"aktual per hari (MAPE).",
                f"Tuning Random Forest dengan TimeSeriesSplit di data train memilih {bp}; RMSE test per jam berubah dari "
                f"{usd(tun['test_rmse_default'])} (parameter awal) menjadi {usd(mj.loc['Random Forest', 'RMSE'])} "
                f"({(mj.loc['Random Forest', 'RMSE'] / tun['test_rmse_default'] - 1) * 100:+.1f}%).",
                "Seasonal naive menyalin satu jam dari minggu lalu apa adanya, termasuk hari yang tidak biasa, dan tidak mengikuti "
                "tren naik. Model ML/DL memakai kondisi terbaru (lag 1 jam, rata-rata 24 jam) dan pola jam sibuk dari 5 bulan data.",
                saran=f"Pakai {best} untuk memperkirakan kebutuhan staf dan stok per jam."
                      + (" Model yang lebih sederhana dan cepat ini sudah sedikit lebih akurat dari LSTM, jadi Deep Learning "
                         "belum memberi keuntungan di data sebesar ini." if best == "Random Forest" else ""))
