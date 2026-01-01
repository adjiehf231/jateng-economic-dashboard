import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Load dataset
df = pd.read_excel("Data_Jateng_35KabKota_2021_2025.xlsx")

st.set_page_config(page_title="Dashboard UMK vs PDRB Jateng", layout="wide")

st.title("Dashboard Analisis UMK dan Pertumbuhan PDRB - Jawa Tengah 2021-2025")

# Sidebar: Filter kabupaten/kota dan tahun
kabupaten = st.sidebar.multiselect(
    "Pilih Kabupaten/Kota:",
    options=df['Kabupaten/Kota'].unique(),
    default=df['Kabupaten/Kota'].unique()
)

tahun = st.sidebar.slider(
    "Pilih Tahun:",
    min_value=int(df['Tahun'].min()),
    max_value=int(df['Tahun'].max()),
    value=(int(df['Tahun'].min()), int(df['Tahun'].max()))
)

# Filter data
df_filtered = df[(df['Kabupaten/Kota'].isin(kabupaten)) &
                 (df['Tahun'] >= tahun[0]) & (df['Tahun'] <= tahun[1])]

# 1️⃣ Line chart UMK
fig_umk = px.line(
    df_filtered,
    x="Tahun",
    y="UMK (Rp)",
    color="Kabupaten/Kota",
    markers=True,
    title="Tren UMK 2021-2025"
)
st.plotly_chart(fig_umk, use_container_width=True)

# 2️⃣ Line chart PDRB
fig_pdrb = px.line(
    df_filtered,
    x="Tahun",
    y="PDRB (Rp)",
    color="Kabupaten/Kota",
    markers=True,
    title="Tren PDRB 2021-2025"
)
st.plotly_chart(fig_pdrb, use_container_width=True)

# 3️⃣ Scatter Plot Kenaikan UMK vs Pertumbuhan PDRB
fig_scatter = px.scatter(
    df_filtered,
    x="Kenaikan UMK (%)",
    y="Pertumbuhan PDRB (%)",
    color="Kabupaten/Kota",
    size="Populasi",
    hover_data=["TPT (%)"],
    title="Kenaikan UMK (%) vs Pertumbuhan PDRB (%)"
)
st.plotly_chart(fig_scatter, use_container_width=True)

# 4️⃣ Heatmap Korelasi Variabel
corr = df_filtered[['UMK (Rp)','Kenaikan UMK (%)','PDRB (Rp)','Pertumbuhan PDRB (%)',
                    'Populasi','Nilai Investasi (Rp)','Jumlah Pengangguran','TPT (%)']].corr()

fig_heatmap = go.Figure(data=go.Heatmap(
    z=corr.values,
    x=corr.columns,
    y=corr.columns,
    colorscale='RdBu',
    zmin=-1, zmax=1
))
fig_heatmap.update_layout(title="Heatmap Korelasi Variabel")
st.plotly_chart(fig_heatmap, use_container_width=True)

# 5️⃣ Data Table
st.subheader("Data Filtered")
st.dataframe(df_filtered)
