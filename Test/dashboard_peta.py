import streamlit as st
import pandas as pd
import geopandas as gpd
import plotly.express as px

# -----------------------------
# Load Dataset dan GeoJSON
# -----------------------------
df = pd.read_excel("Data_Jateng_35KabKota_2021_2025.xlsx")
geo_path = "jawa_tengah_kabkota.geojson"  # FILE GEOJSON
gdf = gpd.read_file(geo_path)

st.set_page_config(layout="wide")
st.title("📌 Dashboard Interaktif Peta UMK & PDRB - Jawa Tengah")

# -----------------------------
# Sidebar Filters
# -----------------------------
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

# Filter data berdasarkan input
df_filtered = df[(df['Kabupaten/Kota'].isin(kabupaten)) &
                 (df['Tahun'] >= tahun[0]) & (df['Tahun'] <= tahun[1])]

# -----------------------------
# Peta Interaktif
# -----------------------------
st.subheader("📍 Peta Interaktif UMK, PDRB, & TPT")

# Gabungkan dataset dengan geojson
df_map = df_filtered.groupby("Kabupaten/Kota").mean().reset_index()
gdf_map = gdf.merge(df_map, left_on="nama_kabkot", right_on="Kabupaten/Kota", how="left")

# Pilihan variabel peta
var_peta = st.selectbox(
    "Pilih Variabel untuk Peta:",
    ["UMK (Rp)", "PDRB (Rp)", "TPT (%)"]
)

fig_map = px.choropleth_mapbox(
    gdf_map,
    geojson=gdf_map.geometry,
    locations=gdf_map.index,
    color=var_peta,
    hover_name="Kabupaten/Kota",
    mapbox_style="carto-positron",
    center={"lat": -7.0, "lon": 110.0},
    zoom=6,
    opacity=0.7,
    color_continuous_scale="Viridis"
)

fig_map.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)

# -----------------------------
# Tren UMK & PDRB
# -----------------------------
st.subheader("📈 Tren UMK & PDRB")

fig_umk = px.line(
    df_filtered,
    x="Tahun",
    y="UMK (Rp)",
    color="Kabupaten/Kota",
    markers=True,
    title="Tren UMK 2021–2025"
)
st.plotly_chart(fig_umk, use_container_width=True)

fig_pdrb = px.line(
    df_filtered,
    x="Tahun",
    y="PDRB (Rp)",
    color="Kabupaten/Kota",
    markers=True,
    title="Tren PDRB 2021–2025"
)
st.plotly_chart(fig_pdrb, use_container_width=True)

# -----------------------------
# Scatter Kenaikan UMK vs Pertumbuhan PDRB
# -----------------------------
st.subheader("📊 UMK vs Pertumbuhan PDRB")

fig_scatter = px.scatter(
    df_filtered,
    x="Kenaikan UMK (%)",
    y="Pertumbuhan PDRB (%)",
    color="Kabupaten/Kota",
    size="Populasi",
    hover_data=["Nilai Investasi (Rp)", "TPT (%)"]
)
st.plotly_chart(fig_scatter, use_container_width=True)

# -----------------------------
# Heatmap Korelasi Variabel
# -----------------------------
st.subheader("📌 Heatmap Korelasi Variabel")

corr = df_filtered[['UMK (Rp)','Kenaikan UMK (%)','PDRB (Rp)',
                    'Pertumbuhan PDRB (%)','Populasi','Inflasi (%)',
                    'Nilai Investasi (Rp)','Jumlah Pengangguran','TPT (%)']].corr()

fig_heatmap = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r")
fig_heatmap.update_layout(title="Heatmap Korelasi Antar Variabel")
st.plotly_chart(fig_heatmap, use_container_width=True)

# -----------------------------
# Tabel Data
# -----------------------------
st.subheader("📋 Tabel Data Filtered")
st.dataframe(df_filtered.reset_index(drop=True))
