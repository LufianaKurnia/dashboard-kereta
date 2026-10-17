import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Dashboard Kereta Api 2026", layout="wide")
st.title("🚆 Dashboard Penumpang Kereta Api 2026")

# 1. Memuat Data
@st.cache_data
def load_data():
    # Melewati 3 baris pertama karena biasanya berisi judul tabel
    df = pd.read_csv("Jumlah Penumpang Kereta Api, 2026.csv", skiprows=3)
    
    # Mengganti nama kolom pertama agar lebih jelas
    df.rename(columns={'Unnamed: 0': 'Wilayah/Jenis'}, inplace=True)
    
    # Mengubah data dari format lebar (wide) ke panjang (long) agar mudah diplot dengan Seaborn
    # Kita hanya ambil bulan Januari - Agustus karena bulan selanjutnya mungkin masih kosong/berisi "-"
    bulan = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus']
    
    # Mengonversi nilai bulan menjadi numerik dan menghapus spasi/karakter tak terduga
    for col in bulan:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    df_melted = df.melt(id_vars=['Wilayah/Jenis'], value_vars=bulan, 
                        var_name='Bulan', value_name='Jumlah Penumpang (Ribu)')
    return df, df_melted

df_raw, df_plot = load_data()

# Menampilkan tabel mentah
with st.expander("Lihat Tabel Data Asli"):
    st.dataframe(df_raw)

# 2. Visualisasi dengan Matplotlib & Seaborn
st.header("Tren Jumlah Penumpang (Januari - Agustus 2026)")

# Filter wilayah yang ingin dibandingkan (misal Jabodetabek vs Non Jabodetabek)
wilayah_pilihan = st.multiselect(
    "Pilih Wilayah/Jenis Kereta:", 
    options=df_raw['Wilayah/Jenis'].dropna().unique(),
    default=['Jabodetabek', 'Non Jabodetabek (Jawa)']
)

df_filtered = df_plot[df_plot['Wilayah/Jenis'].isin(wilayah_pilihan)]

# Membuat Figure Matplotlib
fig, ax = plt.subplots(figsize=(10, 5))
sns.lineplot(data=df_filtered, x='Bulan', y='Jumlah Penumpang (Ribu)', 
             hue='Wilayah/Jenis', marker='o', ax=ax)

ax.set_title("Tren Penumpang Kereta Api 2026")
ax.grid(True, linestyle='--', alpha=0.6)

# Menampilkan ke Streamlit
st.pyplot(fig)