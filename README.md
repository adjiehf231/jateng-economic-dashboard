
# 📊 Jateng Economic Dashboard (2021–2025)

### Dashboard Analisis Ekonomi Kabupaten/Kota Provinsi Jawa Tengah

Dashboard interaktif berbasis **Streamlit** yang menyajikan analisis komprehensif indikator ekonomi kabupaten/kota di Provinsi Jawa Tengah periode **2021–2025**.
Dashboard ini dirancang untuk mendukung **analisis kebijakan ekonomi daerah berbasis data** melalui visualisasi, analisis statistik, pemodelan prediktif, dan simulasi kebijakan.

---

## 🎯 Tujuan Proyek

- Menganalisis tren indikator ekonomi kabupaten/kota di Jawa Tengah
- Mengkaji hubungan antar variabel ekonomi utama
- Mengevaluasi signifikansi statistik antar indikator
- Mengidentifikasi pola, outlier, dan anomali data
- Melakukan simulasi kebijakan (*what-if analysis*)
- Memprediksi indikator ekonomi tahun berikutnya
- Menyediakan dashboard interaktif sebagai alat bantu analisis kebijakan

---

## 📂 Dataset

- **Sumber**: Dummy
- **Periode**: 2021–2025
- **Unit Analisis**: Kabupaten/Kota di Provinsi Jawa Tengah

### Variabel Utama:

- Populasi
- UMK (Upah Minimum Kabupaten/Kota) – Rp
- Kenaikan UMK (%)
- PDRB – Rp
- Pertumbuhan PDRB (%)
- Nilai Investasi – Rp
- Jumlah Pengangguran
- Tingkat Pengangguran Terbuka (TPT %)

---

## 📈 Fitur Dashboard

### 1️⃣ Analisis Tren Waktu

- Tren Populasi
- Tren UMK & Kenaikan UMK
- Tren PDRB & Pertumbuhan PDRB
- Tren Jumlah Pengangguran
- Tren TPT (%)

Semua tren dapat difilter berdasarkan:

- Kabupaten/Kota
- Rentang tahun

---

### 2️⃣ Analisis Hubungan Antar Variabel

Visualisasi hubungan antar indikator ekonomi, antara lain:

- Populasi → UMK
- Populasi → PDRB
- Populasi → Investasi
- Populasi → TPT
- UMK → PDRB
- UMK → Investasi
- PDRB → TPT
- TPT → Investasi

---

### 3️⃣ Analisis Korelasi Statistik

- Korelasi **Pearson**
- Korelasi **Spearman**
- Korelasi **Kendall**
- Heatmap korelasi untuk seluruh variabel utama

---

### 4️⃣ Uji Signifikansi & Evaluasi Model

- Analisis regresi antar variabel
- Evaluasi performa model prediksi (RMSE)
- Insight fitur paling berpengaruh & paling lemah
- Perbandingan model sebelum & sesudah deteksi outlier

---

### 5️⃣ Deteksi Outlier & Anomali

- Metode **Z-Score**
- Identifikasi variabel dengan outlier terbanyak
- Evaluasi dampak outlier terhadap performa model

---

### 6️⃣ Simulasi Kebijakan (*What-if Analysis*)

Simulasi interaktif kebijakan ekonomi:

- Kenaikan UMK (%)
- Kenaikan Investasi (%)

Output:

- Prediksi rata-rata TPT (%)

---

### 7️⃣ Prediksi Indikator Ekonomi Tahun Berikutnya (2026)

- Prediksi UMK
- Prediksi PDRB
- Prediksi TPT
- Prediksi Kenaikan UMK (%)
- Prediksi Pertumbuhan PDRB (%)
- Prediksi Nilai Investasi

---

### 8️⃣ Peringkat Daerah

- Ranking kabupaten/kota berdasarkan:
  - Populasi
  - UMK
  - PDRB
  - Nilai Investasi
  - TPT
- Dapat difilter per tahun

---

## 🧠 Insight Utama

- UMK **tidak berpengaruh signifikan terhadap TPT**, menunjukkan bahwa pengangguran dipengaruhi faktor struktural lain
- PDRB tinggi **tidak selalu berkorelasi dengan TPT rendah**
- Investasi memiliki peran lebih kuat terhadap penyerapan tenaga kerja
- Jumlah pengangguran merupakan fitur paling berpengaruh dalam model
- Pertumbuhan PDRB merupakan fitur dengan pengaruh paling lemah
- Ditemukan daerah dengan **PDRB tinggi namun TPT juga tinggi** (indikasi mismatch tenaga kerja)

---

## 🛠️ Tech Stack

- **Python**
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn, Plotly
- Streamlit

---

## 🗂️ Struktur Direktori


Jateng Economic Dashboard/

│

├── data/

│   └── jateng_clean.csv

│

├── src/

│   ├── preprocessing.py

│   ├── modeling.py

│   └── visualization.py

│

├── app.py

├── requirements.txt

├── README.md

└── .gitignore



---
## 🚀 Live Dashboard
https://jateng-economic-dashboard.streamlit.app/](https://jateng-economic-dashboard-by-adjiehf231.streamlit.app/)
---
## ⚠️ Catatan Metodologi

Analisis ini bersifat **asosiatif** dan berbasis data historis.
Hasil tidak dapat ditafsirkan sebagai hubungan kausal langsung, melainkan sebagai **insight pendukung pengambilan keputusan**.

---

## 👤 Author

Nama Anda : Adjie Hari Fajar
Data Scientist
