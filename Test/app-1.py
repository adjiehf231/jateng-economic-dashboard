# app_full_final.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# =============================
# Load Dataset
# =============================
df = pd.read_excel("Data_Jateng_35KabKota_2021_2025.xlsx")

st.set_page_config(page_title="Dashboard Jateng", layout="wide")
st.title("📊 Dashboard Analisis Kab/Kota Di Jawa Tengah (2021–2025)")

# =============================
# Sidebar Filter
# =============================
st.sidebar.header("Filter Data")

# Pilih Semua / Manual
select_all = st.sidebar.checkbox("Pilih Semua Kabupaten/Kota", value=True)

if select_all:
    kabupaten = df["Kabupaten/Kota"].unique().tolist()
else:
    kabupaten = st.sidebar.multiselect(
        "Pilih Kabupaten/Kota",
        df["Kabupaten/Kota"].unique(),
        default=[]
    )

# Slider Tahun
tahun = st.sidebar.slider(
    "Pilih Rentang Tahun",
    int(df["Tahun"].min()),
    int(df["Tahun"].max()),
    (int(df["Tahun"].min()), int(df["Tahun"].max()))
)

# =============================
# Filter Dataset
# =============================
df_f = df[
    (df["Kabupaten/Kota"].isin(kabupaten)) &
    (df["Tahun"].between(tahun[0], tahun[1]))
]

# =============================
# Handler jika tidak ada data
# =============================
if df_f.empty:
    st.warning("⚠️ Silakan pilih minimal satu Kabupaten/Kota untuk menampilkan analisis.")
    st.stop()


# =============================
# 1️⃣ Tren UMK & PDRB
# =============================
st.subheader("📈 Tren Populasi → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="Populasi", color="Kabupaten/Kota", markers=True),
    use_container_width=True,
)

st.subheader("📈 Tren UMK → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="UMK (Rp)", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)

st.subheader("📈 Tren Kenaikkan UMK → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="Kenaikan UMK (%)", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)

st.subheader("📈 Tren Pertumbuhan PRDB → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="Pertumbuhan PDRB (%)", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)

st.subheader("📈 Tren PRDB → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="PDRB (Rp)", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)

st.subheader("📈 Tren Jumlah Pengangguran → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="Jumlah Pengangguran", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)

st.subheader("📈 Tren TPT% → Tahun")
st.plotly_chart(
    px.line(df_f, x="Tahun", y="TPT (%)", color="Kabupaten/Kota", markers=True),
    use_container_width=True
)


# =============================
# 2️⃣ Analisis Hubungan
# =============================
st.subheader("📊 Analisis Hubungan")

col1, col2, col3 = st.columns(3)

with col1:
    st.plotly_chart(
        px.scatter(
            df_f, x="UMK (Rp)", y="TPT (%)",
            color="Tahun", hover_data=["Kabupaten/Kota"],
            title="UMK → TPT"
        ),
        use_container_width=True
    )

with col2:
    st.plotly_chart(
        px.scatter(
            df_f, x="PDRB (Rp)", y="TPT (%)",
            color="Tahun", hover_data=["Kabupaten/Kota"],
            title="PDRB → TPT"
        ),
        use_container_width=True
    )

with col3:
    st.plotly_chart(
        px.scatter(
            df_f, x="UMK (Rp)", y="PDRB (Rp)",
            color="Tahun", hover_data=["Kabupaten/Kota"],
            title="UMK → PDRB"
        ),
        use_container_width=True
    )

palette = px.colors.qualitative.Set2

col4, col5 = st.columns(2)

with col4:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="UMK (Rp)", y="Nilai Investasi (Rp)",
            color="Tahun",
            size="Populasi",
            hover_data=["Kabupaten/Kota"],
            trendline="ols",
            color_discrete_sequence=palette,
            title="UMK → Investasi"
        ),
        use_container_width=True
    )

with col5:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="TPT (%)", y="Nilai Investasi (Rp)",
            color="Tahun",
            size="Populasi",
            hover_data=["Kabupaten/Kota"],
            trendline="ols",
            color_discrete_sequence=palette,
            title="TPT → Investasi"
        ),
        use_container_width=True
    )

col6, col7 = st.columns(2)

with col6:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="PDRB (Rp)", y="Nilai Investasi (Rp)",
            color="Tahun",
            size="Populasi",
            hover_data=["Kabupaten/Kota"],
            trendline="ols",
            color_discrete_sequence=palette,
            title="PDRB → Investasi"
        ),
        use_container_width=True
    )

with col7:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="Populasi", y="Nilai Investasi (Rp)",
            color="Tahun",
            hover_data=["Kabupaten/Kota"],
            trendline="ols",
            color_discrete_sequence=palette,
            title="Populasi → Investasi"
        ),
        use_container_width=True
    )

# =============================
# Analisis Populasi terhadap Indikator Ekonomi
# =============================
col8, col9, col10, col11 = st.columns(4)

with col8:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="Populasi", y="UMK (Rp)",
            color="Tahun",
            trendline="ols",
            hover_data=["Kabupaten/Kota"],
            color_discrete_sequence=palette,
            title="Populasi → UMK"
        ),
        use_container_width=True
    )

with col9:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="Populasi", y="PDRB (Rp)",
            color="Tahun",
            trendline="ols",
            hover_data=["Kabupaten/Kota"],
            color_discrete_sequence=palette,
            title="Populasi → PDRB"
        ),
        use_container_width=True
    )

with col10:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="Populasi", y="TPT (%)",
            color="Tahun",
            trendline="ols",
            hover_data=["Kabupaten/Kota"],
            color_discrete_sequence=palette,
            title="Populasi → TPT"
        ),
        use_container_width=True
    )
    
with col11:
    st.plotly_chart(
        px.scatter(
            df_f,
            x="Populasi", y="Nilai Investasi (Rp)",
            color="Tahun",
            trendline="ols",
            hover_data=["Kabupaten/Kota"],
            color_discrete_sequence=palette,
            title="Populasi → Nilai Investasi (Rp)"
        ),
        use_container_width=True
    )

st.subheader("📐 Analisis Korelasi Statistik")

corr_features = [
    "UMK (Rp)", "PDRB (Rp)", "TPT (%)",
    "Populasi", "Nilai Investasi (Rp)"
]

corr_df = df_f[corr_features].dropna()

tab1, tab2 = st.tabs(["Pearson", "Spearman"])

with tab1:
    pearson_corr = corr_df.corr(method="pearson")
    fig_p = px.imshow(
        pearson_corr,
        text_auto=".2f",
        title="Heatmap Korelasi Pearson",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_p, use_container_width=True)

with tab2:
    spearman_corr = corr_df.corr(method="spearman")
    fig_s = px.imshow(
        spearman_corr,
        text_auto=".2f",
        title="Heatmap Korelasi Spearman",
        color_continuous_scale="RdBu",
        aspect="auto"
    )
    st.plotly_chart(fig_s, use_container_width=True)


st.subheader("📉 Uji Signifikansi Regresi")

def regression_test(x, y, x_label):
    X = sm.add_constant(x)
    model = sm.OLS(y, X).fit()
    return model

# UMK → TPT
model_umk = regression_test(df_f["UMK (Rp)"], df_f["TPT (%)"], "UMK")

# PDRB → TPT
model_pdrb = regression_test(df_f["PDRB (Rp)"], df_f["TPT (%)"], "PDRB")

# Populasi → TPT
model_pop = regression_test(df_f["Populasi"], df_f["TPT (%)"], "Populasi")

# INVESTASI → POPULASI
model_investasi = regression_test(df_f["Nilai Investasi (Rp)"], df_f["Populasi"], "Investasi")

# UMK → POPULASI
model_umk_pop = regression_test(df_f["UMK (Rp)"], df_f["Populasi"], "UMK")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### UMK → TPT")
    st.write(f"R²: **{model_umk.rsquared:.3f}**")
    st.write(f"p-value: **{model_umk.pvalues[1]:.4f}**")

with col2:
    st.markdown("### PDRB → TPT")
    st.write(f"R²: **{model_pdrb.rsquared:.3f}**")
    st.write(f"p-value: **{model_pdrb.pvalues[1]:.4f}**")

with col3:
    st.markdown("### POPULASI → TPT")
    st.write(f"R²: **{model_pop.rsquared:.3f}**")
    st.write(f"p-value: **{model_pop.pvalues[1]:.4f}**")
    
with col4:
    st.markdown("### INVESTASI → POPULASI")
    st.write(f"R²: **{model_investasi.rsquared:.3f}**")
    st.write(f"p-value: **{model_investasi.pvalues[1]:.4f}**")
    
with col5:
    st.markdown("### UMK → POPULASI")
    st.write(f"R²: **{model_umk_pop.rsquared:.3f}**")
    st.write(f"p-value: **{model_umk_pop.pvalues[1]:.4f}**")

# =============================
# 3️⃣ Clustering Kabupaten/Kota
# =============================
st.subheader("🧩 Clustering Kabupaten/Kota")

features = [
    "UMK (Rp)", "Kenaikan UMK (%)", "PDRB (Rp)",
    "Pertumbuhan PDRB (%)", "Populasi",
    "TPT (%)", "Nilai Investasi (Rp)"
]

X = df_f[features].dropna()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🔒 Proteksi jumlah cluster
if X_scaled.shape[0] < 3:
    st.warning("⚠️ Data terlalu sedikit untuk clustering (minimal 3 observasi).")
    st.stop()

n_clusters = min(3, X_scaled.shape[0] - 1)
kmeans = KMeans(n_clusters=n_clusters, random_state=42)

df_f.loc[X.index, "Cluster"] = kmeans.fit_predict(X_scaled)

st.plotly_chart(
    px.scatter(
        df_f, x="UMK (Rp)", y="PDRB (Rp)",
        color="Cluster", symbol="Tahun",
        hover_data=["Kabupaten/Kota"],
        title="Cluster UMK–PDRB–TPT"
    ),
    use_container_width=True
)

st.subheader("📊 Validasi Clustering")

n_samples = X_scaled.shape[0]

if n_samples < 3:
    st.warning("⚠️ Data terlalu sedikit untuk validasi clustering (minimal 3 data).")
else:
    max_k = min(6, n_samples - 1)
    range_k = range(2, max_k + 1)

    silhouette_scores = []

    for k in range_k:
        km = KMeans(n_clusters=k, random_state=42)
        labels = km.fit_predict(X_scaled)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    fig_sil = px.line(
        x=list(range_k),
        y=silhouette_scores,
        markers=True,
        title="Silhouette Score vs Jumlah Cluster",
        labels={"x": "Jumlah Cluster", "y": "Silhouette Score"}
    )
    st.plotly_chart(fig_sil, use_container_width=True)


cluster_summary = (
    df_f.groupby("Cluster")[features]
    .mean()
    .reset_index()
)

cluster_summary["Label"] = cluster_summary["Cluster"].map({
    0: "Maju",
    1: "Berkembang",
    2: "Rentan"
})

df_f = df_f.merge(
    cluster_summary[["Cluster", "Label"]],
    on="Cluster",
    how="left"
)

st.subheader("🏷️ Klasifikasi Daerah")
st.dataframe(cluster_summary)

st.subheader("🚨 Deteksi Outlier & Anomali")

outlier_umk = df_f[
    (df_f["UMK (Rp)"] > df_f["UMK (Rp)"].quantile(0.75)) &
    (df_f["TPT (%)"] > df_f["TPT (%)"].quantile(0.75))
]

outlier_pdrb = df_f[
    (df_f["PDRB (Rp)"] > df_f["PDRB (Rp)"].quantile(0.75)) &
    (df_f["Nilai Investasi (Rp)"] < df_f["Nilai Investasi (Rp)"].quantile(0.25))
]

st.markdown("### UMK Tinggi tapi TPT Tinggi")
st.dataframe(outlier_umk[["Kabupaten/Kota", "UMK (Rp)", "TPT (%)"]])

st.markdown("### PDRB Tinggi tapi Investasi Rendah")
st.dataframe(outlier_pdrb[["Kabupaten/Kota", "PDRB (Rp)", "Nilai Investasi (Rp)"]])


st.subheader("🤖 Evaluasi Model Prediksi TPT")

X = df_f[["UMK (Rp)", "PDRB (Rp)", "Nilai Investasi (Rp)"]]
y = df_f["TPT (%)"]

lr = LinearRegression().fit(X, y)
rf = RandomForestRegressor(random_state=42).fit(X, y)

pred_lr = lr.predict(X)
pred_rf = rf.predict(X)

result = pd.DataFrame({
    "Model": ["Linear Regression", "Random Forest"],
    "MAE": [
        mean_absolute_error(y, pred_lr),
        mean_absolute_error(y, pred_rf)
    ],
    "RMSE": [
        np.sqrt(mean_squared_error(y, pred_lr)),
        np.sqrt(mean_squared_error(y, pred_rf))
    ]
})


st.dataframe(result)

st.subheader("🧪 Simulasi What-if Kebijakan")

delta_umk = st.slider("Kenaikan UMK (%)", -20, 30, 10)
delta_inv = st.slider("Kenaikan Investasi (%)", -20, 30, 15)

X_sim = X.copy()
X_sim["UMK (Rp)"] *= (1 + delta_umk / 100)
X_sim["Nilai Investasi (Rp)"] *= (1 + delta_inv / 100)

sim_tpt = lr.predict(X_sim).mean()

st.metric(
    "Prediksi Rata-rata TPT (%)",
    f"{sim_tpt:.2f}"
)


# =============================
# 4️⃣ Prediksi 1 Tahun ke Depan
# =============================
st.subheader("🔮 Prediksi Indikator Ekonomi Tahun Berikutnya")

target_vars = {
    "UMK (Rp)": "Prediksi UMK",
    "PDRB (Rp)": "Prediksi PDRB",
    "TPT (%)": "Prediksi TPT",
    "Kenaikan UMK (%)": "Prediksi Kenaikan UMK",
    "Pertumbuhan PDRB (%)": "Prediksi Pertumbuhan PDRB",
    "Nilai Investasi (Rp)": "Prediksi Investasi"
}

next_year = df_f["Tahun"].max() + 1

for col, label in target_vars.items():
    model = LinearRegression()
    X_year = df_f[["Tahun"]]
    y_val = df_f[col]
    model.fit(X_year, y_val)
    pred = model.predict([[next_year]])[0]
    st.write(f"**{label} {next_year}:** {pred:,.2f}")

st.info(
    "ℹ️ Catatan Metodologi: "
    "Analisis ini bersifat asosiatif berbasis data historis. "
    "Hasil tidak dapat ditafsirkan sebagai hubungan kausal langsung."
)


# =============================
# 5️⃣ Insight & Storytelling
# =============================
st.subheader("🧠 Temuan Utama Tahun Ini")

# =============================
# UMK → TPT
# =============================
coef_umk = model_umk.params[1]
pval_umk = model_umk.pvalues[1]

if pval_umk < 0.05:
    if coef_umk > 0:
        st.info(
            "📌 **UMK berpengaruh signifikan dan positif terhadap TPT**. "
            "Kenaikan UMK cenderung diikuti kenaikan TPT → indikasi *mismatch tenaga kerja*."
        )
    else:
        st.success(
            "📌 **UMK berpengaruh signifikan dalam menurunkan TPT**. "
            "Kebijakan UMK berpotensi mendukung penyerapan tenaga kerja."
        )
else:
    st.warning(
        "📌 **UMK tidak berpengaruh signifikan terhadap TPT**. "
        "Artinya, perubahan TPT lebih dipengaruhi faktor lain seperti investasi dan struktur industri."
    )

# =============================
# PDRB → TPT
# =============================
coef_pdrb = model_pdrb.params[1]
pval_pdrb = model_pdrb.pvalues[1]

if pval_pdrb < 0.05:
    if coef_pdrb > 0:
        st.warning(
            "📌 **PDRB meningkat namun TPT juga meningkat secara signifikan**. "
            "Pertumbuhan ekonomi belum bersifat inklusif."
        )
    else:
        st.success(
            "📌 **PDRB berpengaruh signifikan dalam menurunkan TPT**. "
            "Pertumbuhan ekonomi mendorong penyerapan tenaga kerja."
        )
else:
    st.info(
        "📌 **PDRB tidak menunjukkan pengaruh signifikan terhadap TPT**. "
        "Pertumbuhan ekonomi belum secara langsung berdampak pada pasar tenaga kerja."
    )



st.subheader("📝 Insight & Rekomendasi")
st.markdown("""
- UMK **tidak selalu menurunkan TPT**, perlu dukungan investasi & industri.
- Daerah dengan **PDRB tinggi tapi TPT tinggi** indikasi mismatch tenaga kerja.
- Clustering membantu klasifikasi daerah maju, berkembang, dan tertinggal.
- Model prediksi ini cocok sebagai **baseline analisis kebijakan ekonomi daerah**.
""")

# =============================
# 6️⃣ Data Table
# =============================
st.subheader("📋 Data Akhir")
st.dataframe(df_f.reset_index(drop=True))

st.subheader("🏆 Peringkat Daerah")

# =============================
# Filter Tahun
# =============================
tahun = st.selectbox("Pilih Tahun:", options=sorted(df_f["Tahun"].unique()))
df_filtered = df_f[df_f["Tahun"] == tahun]

# =============================
# Kolom peringkat
# =============================
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("### 🔺 TPT (%)")
    df_tpt = df_filtered.groupby("Kabupaten/Kota")["TPT (%)"].mean().reset_index()
    st.dataframe(df_tpt.sort_values("TPT (%)", ascending=False))

with col2:
    st.markdown("### 💰 Investasi (Rp)")
    df_invest = df_filtered.groupby("Kabupaten/Kota")["Nilai Investasi (Rp)"].mean().reset_index()
    st.dataframe(df_invest.sort_values("Nilai Investasi (Rp)", ascending=False))

with col3:
    st.markdown("### 💵 UMK (Rp)")
    df_umk = df_filtered.groupby("Kabupaten/Kota")["UMK (Rp)"].mean().reset_index()
    st.dataframe(df_umk.sort_values("UMK (Rp)", ascending=False))

with col4:
    st.markdown("### 📈 PDRB (Rp)")
    df_pdrb = df_filtered.groupby("Kabupaten/Kota")["PDRB (Rp)"].mean().reset_index()
    st.dataframe(df_pdrb.sort_values("PDRB (Rp)", ascending=False))

with col5:
    st.markdown("### 📊 Populasi")
    df_pop = df_filtered.groupby("Kabupaten/Kota")["Populasi"].mean().reset_index()
    st.dataframe(df_pop.sort_values("Populasi", ascending=False))