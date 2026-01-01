import pandas as pd

def load_data(path):
    if path.endswith(".csv"):
        df = pd.read_csv(path, encoding='utf-8', errors='ignore')  # fallback ignore karakter yang tidak bisa dibaca
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        df = pd.read_excel(path)
    else:
        raise ValueError("Format file tidak didukung. Gunakan CSV atau Excel.")
    return df

def filter_data(df, kabupaten, tahun_range):
    return df[
        (df["Kabupaten/Kota"].isin(kabupaten)) &
        (df["Tahun"].between(tahun_range[0], tahun_range[1]))
    ]
