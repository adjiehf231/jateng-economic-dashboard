import streamlit as st
from src.preprocessing import load_data, filter_data
from src.visualization import (
    plot_trends, plot_scatter_relationships, plot_correlation_heatmap,
    display_rank_tables
)
from src.modeling import (
    run_regression_tests, predict_next_year, simulate_policy, evaluate_models_with_insight, evaluate_models_with_outliers
)

# =============================
# Load Dataset
# =============================
df = load_data("data/data_jateng.xlsx")

# =============================
# Streamlit Config
# =============================
st.set_page_config(page_title="Dashboard Jateng", layout="wide")
st.title("📊 Dashboard Analisis Kab/Kota Di Jawa Tengah (2021–2025)")

# =============================
# Sidebar Filter
# =============================
st.sidebar.header("Filter Data")

# Pilih Kabupaten/Kota
select_all = st.sidebar.checkbox("Pilih Semua Kabupaten/Kota", value=True)
if select_all:
    kabupaten = df["Kabupaten/Kota"].unique().tolist()
else:
    kabupaten = st.sidebar.multiselect(
        "Pilih Kabupaten/Kota", df["Kabupaten/Kota"].unique(), default=[]
    )

# Pilih Tahun
tahun_range = st.sidebar.slider(
    "Pilih Rentang Tahun",
    int(df["Tahun"].min()), int(df["Tahun"].max()),
    (int(df["Tahun"].min()), int(df["Tahun"].max()))
)

# =============================
# Filter Data
# =============================
df_f = filter_data(df, kabupaten, tahun_range)

if df_f.empty:
    st.warning("⚠️ Silakan pilih minimal satu Kabupaten/Kota untuk menampilkan analisis.")
    st.stop()

# =============================
# 1️⃣ Tren & Grafik
# =============================
plot_trends(df_f)

# =============================
# 2️⃣ Analisis Hubungan
# =============================
plot_scatter_relationships(df_f)

# =============================
# 3️⃣ Korelasi Statistik
# =============================
plot_correlation_heatmap(df_f)

# =============================
# 4️⃣ Uji Regresi
# =============================
models = run_regression_tests(df_f)

# =============================
# 5️⃣ Clustering, Outlier & Insight
# =============================
# Bisa ditambahkan fungsi dari src.visualization.py untuk clustering & insight

# =============================
# 6️⃣ Evaluasi Model Prediksi
# =============================
evaluate_models_with_insight(df_f)

# Deteksi Outlier Dan Anomali
# =============================
evaluate_models_with_outliers(df_f)

# =============================
# 7️⃣ Simulasi What-if Kebijakan
# =============================
simulate_policy(df_f)


# =============================
# 8️⃣ Prediksi Tahun Berikutnya
# =============================
predict_next_year(df_f)

# =============================
# 9️⃣ Peringkat Daerah
# =============================
display_rank_tables(df_f)
