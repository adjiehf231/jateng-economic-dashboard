from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import statsmodels.api as sm
from scipy.stats import zscore

def run_regression_tests(df):
    st.subheader("📉 Uji Signifikansi Regresi Antar Variabel")

    # Semua kolom numerik utama
    variables = [
        "Populasi",
        "UMK (Rp)",
        "Kenaikan UMK (%)",
        "PDRB (Rp)",
        "Pertumbuhan PDRB (%)",
        "Nilai Investasi (Rp)",
        "Jumlah Pengangguran",
        "TPT (%)"
    ]

    # Fungsi regresi OLS
    def regression_test(x, y):
        X = sm.add_constant(x)
        return sm.OLS(y, X).fit()

    models = {}
    results = []

    # Buat kombinasi x → y tanpa duplikat terbalik
    for i, x in enumerate(variables):
        for j, y in enumerate(variables):
            if i < j:  # hanya buat kombinasi di satu arah (x sebelum y)
                name = f"{x} → {y}"
                model = regression_test(df[x], df[y])
                models[name] = model
                results.append({
                    "Model": name,
                    "R²": round(model.rsquared, 3),
                    "p-value": round(model.pvalues[1], 4)
                })

    results_df = pd.DataFrame(results)

    # Tampilkan tabel interaktif
    st.dataframe(results_df.sort_values("R²", ascending=False).reset_index(drop=True), use_container_width=True)

    return models, results_df

def evaluate_models_with_insight(df):
    st.subheader("🤖 Evaluasi Model Prediksi Antar Variabel dengan Insight")

    # Semua kolom numerik utama
    variables = [
        "Populasi",
        "UMK (Rp)",
        "Kenaikan UMK (%)",
        "PDRB (Rp)",
        "Pertumbuhan PDRB (%)",
        "Nilai Investasi (Rp)",
        "Jumlah Pengangguran",
        "TPT (%)"
    ]

    # -------------------------------
    # 1️⃣ Evaluasi model sebelum outlier
    # -------------------------------
    all_results = []
    feature_importances = {var: 0 for var in variables}

    for target in variables:
        features = [v for v in variables if v != target]
        X = df[features]
        y = df[target]

        # Linear Regression
        lr = LinearRegression().fit(X, y)
        pred_lr = lr.predict(X)
        mae_lr = mean_absolute_error(y, pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y, pred_lr))

        # Random Forest
        rf = RandomForestRegressor(random_state=42).fit(X, y)
        pred_rf = rf.predict(X)
        mae_rf = mean_absolute_error(y, pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y, pred_rf))

        # Simpan hasil
        all_results.append({
            "Target": target,
            "Model": "Linear Regression",
            "MAE": round(mae_lr, 3),
            "RMSE": round(rmse_lr, 3)
        })
        all_results.append({
            "Target": target,
            "Model": "Random Forest",
            "MAE": round(mae_rf, 3),
            "RMSE": round(rmse_rf, 3)
        })

        # Random Forest importances
        for f, imp in zip(features, rf.feature_importances_):
            feature_importances[f] += imp

    results_df = pd.DataFrame(all_results)
    st.markdown("### 📊 Hasil Evaluasi Model (Sebelum Deteksi Outlier)")
    st.dataframe(results_df.sort_values(["Target", "RMSE"]), use_container_width=True)

    # Insight dari evaluasi awal
    st.markdown("### 💡 Insight Sebelum Outlier")
    best_target = results_df.groupby("Target")["RMSE"].mean().idxmin()
    worst_target = results_df.groupby("Target")["RMSE"].mean().idxmax()
    st.markdown(f"- Target **paling mudah diprediksi**: `{best_target}` (RMSE rendah)")
    st.markdown(f"- Target **paling sulit diprediksi**: `{worst_target}` (RMSE tinggi)")

    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    st.markdown(f"- Fitur **paling berpengaruh** secara keseluruhan: `{sorted_features[0][0]}`")
    st.markdown(f"- Fitur **kurang berpengaruh**: `{sorted_features[-1][0]}`")

    # -------------------------------
    # 2️⃣ Deteksi outlier setelah evaluasi
    # -------------------------------
    st.markdown("### ⚠️ Deteksi Outlier (Z-score)")
    outlier_info = {}
    for col in variables:
        z_scores = np.abs(zscore(df[col].dropna()))
        n_outliers = (z_scores > 3).sum()
        outlier_info[col] = n_outliers
    outlier_df = pd.DataFrame(list(outlier_info.items()), columns=["Variabel", "Jumlah Outlier (Z>3)"])
    st.dataframe(outlier_df, use_container_width=True)

    # Insight outlier
    st.markdown("### 💡 Insight Berdasarkan Outlier")
    most_outlier_var = outlier_df.sort_values("Jumlah Outlier (Z>3)", ascending=False).iloc[0]
    st.markdown(f"- Variabel dengan outlier terbanyak: `{most_outlier_var['Variabel']}` ({most_outlier_var['Jumlah Outlier (Z>3)']} nilai)")

    return results_df, outlier_df


def evaluate_models_with_outliers(df):
    st.subheader("🤖 Evaluasi Model Prediksi Antar Variabel dengan Deteksi Outlier")

    # =============================
    # Catatan metodologi
    # =============================
    st.info(
        "ℹ️ Catatan Metodologi: "
        "Analisis ini bersifat asosiatif berbasis data historis. "
        "Hasil tidak dapat ditafsirkan sebagai hubungan kausal langsung."
    )

    # Semua kolom numerik utama
    variables = [
        "Populasi",
        "UMK (Rp)",
        "Kenaikan UMK (%)",
        "PDRB (Rp)",
        "Pertumbuhan PDRB (%)",
        "Nilai Investasi (Rp)",
        "Jumlah Pengangguran",
        "TPT (%)"
    ]

    # =============================
    # Deteksi outlier menggunakan Z-score
    # =============================
    st.markdown("### ⚠️ Deteksi Outlier (Z-score)")
    outlier_info = {}
    for col in variables:
        z_scores = np.abs(zscore(df[col].dropna()))
        n_outliers = (z_scores > 3).sum()
        outlier_info[col] = n_outliers
    outlier_df = pd.DataFrame(list(outlier_info.items()), columns=["Variabel", "Jumlah Outlier (Z>3)"])
    st.dataframe(outlier_df, use_container_width=True)

    # =============================
    # Loop semua variabel sebagai target
    # =============================
    all_results = []
    feature_importances = {var: 0 for var in variables}  # untuk insight fitur penting
    for target in variables:
        features = [v for v in variables if v != target]
        X = df[features]
        y = df[target]

        # Linear Regression
        lr = LinearRegression().fit(X, y)
        pred_lr = lr.predict(X)
        mae_lr = mean_absolute_error(y, pred_lr)
        rmse_lr = np.sqrt(mean_squared_error(y, pred_lr))

        # Random Forest
        rf = RandomForestRegressor(random_state=42).fit(X, y)
        pred_rf = rf.predict(X)
        mae_rf = mean_absolute_error(y, pred_rf)
        rmse_rf = np.sqrt(mean_squared_error(y, pred_rf))

        # Simpan hasil
        all_results.append({
            "Target": target,
            "Model": "Linear Regression",
            "MAE": round(mae_lr, 3),
            "RMSE": round(rmse_lr, 3)
        })
        all_results.append({
            "Target": target,
            "Model": "Random Forest",
            "MAE": round(mae_rf, 3),
            "RMSE": round(rmse_rf, 3)
        })

        # Simpan importances dari Random Forest untuk insight
        for f, imp in zip(features, rf.feature_importances_):
            feature_importances[f] += imp

    results_df = pd.DataFrame(all_results)
    st.markdown("### 📊 Hasil Evaluasi Model")
    st.dataframe(results_df.sort_values(["Target", "RMSE"]), use_container_width=True)

    # =============================
    # Insight sederhana
    # =============================
    st.markdown("### 💡 Insight Berdasarkan Analisis")
    best_target = results_df.groupby("Target")["RMSE"].mean().idxmin()
    worst_target = results_df.groupby("Target")["RMSE"].mean().idxmax()
    st.markdown(f"- Target **paling mudah diprediksi**: `{best_target}` (RMSE rendah)")
    st.markdown(f"- Target **paling sulit diprediksi**: `{worst_target}` (RMSE tinggi)")

    # Fitur paling berpengaruh
    sorted_features = sorted(feature_importances.items(), key=lambda x: x[1], reverse=True)
    st.markdown(f"- Fitur **paling berpengaruh secara keseluruhan**: `{sorted_features[0][0]}`")
    st.markdown(f"- Fitur **kurang berpengaruh**: `{sorted_features[-1][0]}`")

    # =============================
    # Storytelling UMK → TPT
    # =============================
    st.subheader("🧠 Temuan Utama Tahun Ini")

    # Hitung model regresi UMK → TPT
    X_umk = sm.add_constant(df["UMK (Rp)"])
    model_umk = sm.OLS(df["TPT (%)"], X_umk).fit()

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

    return results_df, outlier_df, model_umk

def simulate_policy(df):
    st.subheader("🧪 Simulasi What-if Kebijakan")
    delta_umk = st.slider("Kenaikan UMK (%)", -20, 30, 10)
    delta_inv = st.slider("Kenaikan Investasi (%)", -20, 30, 15)
    X = df[["UMK (Rp)", "PDRB (Rp)", "Nilai Investasi (Rp)"]]
    lr = LinearRegression().fit(X, df["TPT (%)"])
    X_sim = X.copy()
    X_sim["UMK (Rp)"] *= (1 + delta_umk / 100)
    X_sim["Nilai Investasi (Rp)"] *= (1 + delta_inv / 100)
    sim_tpt = lr.predict(X_sim).mean()
    st.metric("Prediksi Rata-rata TPT (%)", f"{sim_tpt:.2f}")

def predict_next_year(df):
    st.subheader("🔮 Prediksi Indikator Ekonomi Tahun Berikutnya")
    next_year = df["Tahun"].max() + 1
    target_vars = ["UMK (Rp)", "PDRB (Rp)", "TPT (%)", "Kenaikan UMK (%)", "Pertumbuhan PDRB (%)", "Nilai Investasi (Rp)"]
    for var in target_vars:
        model = LinearRegression().fit(df[["Tahun"]], df[var])
        pred = model.predict([[next_year]])[0]
        st.write(f"**Prediksi {var} {next_year}:** {pred:,.2f}")