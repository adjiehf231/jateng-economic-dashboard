import streamlit as st
import plotly.express as px

def plot_trends(df):
    st.subheader("📈 Tren Populasi → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="Populasi", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren UMK → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="UMK (Rp)", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren Kenaikan UMK → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="Kenaikan UMK (%)", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren PDRB → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="PDRB (Rp)", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren Pertumbuhan PDRB → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="Pertumbuhan PDRB (%)", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren Jumlah Pengangguran → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="Jumlah Pengangguran", color="Kabupaten/Kota", markers=True), use_container_width=True)
    st.subheader("📈 Tren TPT → Tahun")
    st.plotly_chart(px.line(df, x="Tahun", y="TPT (%)", color="Kabupaten/Kota", markers=True), use_container_width=True)


def plot_scatter_relationships(df):
    st.subheader("📊 Analisis Hubungan Antar Variabel")

    # List semua kolom numerik yang ingin dianalisis
    indicators = [
        "Populasi",
        "UMK (Rp)",
        "Kenaikan UMK (%)",
        "PDRB (Rp)",
        "Pertumbuhan PDRB (%)",
        "Nilai Investasi (Rp)",
        "Jumlah Pengangguran",
        "TPT (%)"
    ]

    # Buat dictionary tab: key = nama tab, value = list tuple (x, y, title)
    tabs_dict = {}
    for ind in indicators:
        # Target = semua kolom numerik kecuali dirinya sendiri
        targets = [col for col in indicators if col != ind]
        tabs_dict[ind] = [(ind, target, f"{ind} → {target}") for target in targets]

    # Buat tab Streamlit
    tab_list = st.tabs(list(tabs_dict.keys()))

    for tab_obj, (tab_name, chart_list) in zip(tab_list, tabs_dict.items()):
        col1, col2 = tab_obj.columns(2)
        for i, (x, y, title) in enumerate(chart_list):
            fig = px.scatter(
                df,
                x=x,
                y=y,
                color="Tahun",
                hover_data=["Kabupaten/Kota"],
                title=title
            )
            target_col = col1 if i % 2 == 0 else col2
            with target_col:
                st.plotly_chart(fig, use_container_width=True)



def plot_correlation_heatmap(df):
    st.subheader("📐 Analisis Korelasi Statistik")

    # Kolom numerik yang dianalisis
    features = ["UMK (Rp)", "PDRB (Rp)", "TPT (%)", "Populasi", "Nilai Investasi (Rp)"]
    corr_df = df[features].dropna()

    # Buat 3 tab
    tab1, tab2, tab3 = st.tabs(["Pearson", "Spearman", "Kendall"])

    with tab1:
        pearson_corr = corr_df.corr(method="pearson")
        fig_p = px.imshow(
            pearson_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Heatmap Korelasi Pearson"
        )
        st.plotly_chart(fig_p, use_container_width=True)

    with tab2:
        spearman_corr = corr_df.corr(method="spearman")
        fig_s = px.imshow(
            spearman_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Heatmap Korelasi Spearman"
        )
        st.plotly_chart(fig_s, use_container_width=True)

    with tab3:
        kendall_corr = corr_df.corr(method="kendall")
        fig_k = px.imshow(
            kendall_corr,
            text_auto=".2f",
            color_continuous_scale="RdBu",
            aspect="auto",
            title="Heatmap Korelasi Kendall"
        )
        st.plotly_chart(fig_k, use_container_width=True)
    

def display_rank_tables(df):
    st.subheader("🏆 Peringkat Daerah")
    tahun = st.selectbox("Pilih Tahun:", options=sorted(df["Tahun"].unique()))
    df_filtered = df[df["Tahun"] == tahun]

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.markdown("### 📊 Populasi")
        st.dataframe(df_filtered.groupby("Kabupaten/Kota")["Populasi"].mean().reset_index().sort_values("Populasi", ascending=False))

    with col2:
        st.markdown("### 💵 UMK (Rp)")
        st.dataframe(df_filtered.groupby("Kabupaten/Kota")["UMK (Rp)"].mean().reset_index().sort_values("UMK (Rp)", ascending=False))

    with col3:
        st.markdown("### 📈 PDRB (Rp)")
        st.dataframe(df_filtered.groupby("Kabupaten/Kota")["PDRB (Rp)"].mean().reset_index().sort_values("PDRB (Rp)", ascending=False))

    with col4:
        st.markdown("### 💰 Investasi (Rp)")
        st.dataframe(df_filtered.groupby("Kabupaten/Kota")["Nilai Investasi (Rp)"].mean().reset_index().sort_values("Nilai Investasi (Rp)", ascending=False))

    with col5:
        st.markdown("### 🔺 TPT (%)")
        st.dataframe(df_filtered.groupby("Kabupaten/Kota")["TPT (%)"].mean().reset_index().sort_values("TPT (%)", ascending=False))
