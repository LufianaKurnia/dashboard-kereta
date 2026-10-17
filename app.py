import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

# --- PALET WARNA CUSTOM ---
# Diambil dari gambar referensi
COLOR_1 = '#F8B195' # Peach/Light Coral
COLOR_2 = '#F67280' # Salmon Pink
COLOR_3 = '#C06C84' # Mauve/Rose
COLOR_4 = '#6C5B7B' # Muted Purple
COLOR_5 = '#355C7D' # Dark Blue

# 1. SETUP HALAMAN
st.set_page_config(page_title="Dashboard Kereta Api", layout="wide")
st.title("Dashboard & Prediksi Penumpang Kereta Api")
st.markdown("Analisis data historis dan proyeksi jumlah penumpang menggunakan Decision Tree Regression.")

# 2. FUNGSI MEMUAT & MENGGABUNGKAN DATA
@st.cache_data
def load_data():
    tahun_list = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    df_all = pd.DataFrame()
    bulan_str = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    for thn in tahun_list:
        if thn == 2026:
            nama_file = "Jumlah Penumpang Kereta Api, 2026_2.csv"
        else:
            nama_file = f"Jumlah Penumpang Kereta Api, {thn}.csv"

        try:
            df = pd.read_csv(nama_file, skiprows=3)
            df.rename(columns={'Unnamed: 0': 'Wilayah/Jenis'}, inplace=True)
            df_melted = df.melt(id_vars=['Wilayah/Jenis'], value_vars=bulan_str,
                                var_name='Bulan', value_name='Jumlah Penumpang')
            df_melted['Tahun'] = thn
            df_all = pd.concat([df_all, df_melted], ignore_index=True)
        except Exception as e:
            pass

    df_all['Jumlah Penumpang'] = pd.to_numeric(df_all['Jumlah Penumpang'], errors='coerce')
    bulan_map = {b: i+1 for i, b in enumerate(bulan_str)}
    df_all['Bulan_Angka'] = df_all['Bulan'].map(bulan_map)

    return df_all, bulan_str

df_gabungan, daftar_bulan = load_data()

if df_gabungan.empty:
    st.error("Data tidak ditemukan! Pastikan nama file CSV sesuai dan berada di folder yang sama.")
    st.stop()

# 3. SIDEBAR: FITUR FILTER
st.sidebar.header("Pengaturan Data")
daftar_wilayah = df_gabungan['Wilayah/Jenis'].dropna().unique()
wilayah_pilihan = st.sidebar.selectbox("Wilayah / Jenis Kereta:", daftar_wilayah)

tahun_pilihan = st.sidebar.slider("Rentang Tahun Historis:", min_value=2020, max_value=2026, value=(2020, 2026))

df_wilayah = df_gabungan[df_gabungan['Wilayah/Jenis'] == wilayah_pilihan].copy()
df_historis = df_wilayah[(df_wilayah['Tahun'] >= tahun_pilihan[0]) & (df_wilayah['Tahun'] <= tahun_pilihan[1])]


# 4. TAMPILAN UTAMA
tab1, tab2, tab3 = st.tabs(["Data Historis", "Prediksi 2026", "Proyeksi 2027"])

# --- TAB 1: DATA HISTORIS ---
with tab1:
    st.subheader(f"Tren Penumpang: {wilayah_pilihan} ({tahun_pilihan[0]} - {tahun_pilihan[1]})")

    df_historis_plot = df_historis.dropna(subset=['Jumlah Penumpang']).copy()
    if not df_historis_plot.empty:
        df_historis_plot['Tanggal'] = pd.to_datetime(df_historis_plot['Tahun'].astype(str) + '-' + df_historis_plot['Bulan_Angka'].astype(str) + '-01')
        df_historis_plot = df_historis_plot.sort_values('Tanggal')

        # Gaya Plot Minimalis
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_historis_plot, x='Tanggal', y='Jumlah Penumpang', marker='o', color=COLOR_5, linewidth=2, ax=ax) # Memakai warna Dark Blue

        ax.set_title("Data Historis Penumpang", fontsize=14, pad=15)
        ax.set_ylabel("Jumlah (Ribu Orang)", color='grey')
        ax.set_xlabel("") # Kosongkan label X untuk estetika minimalis
        ax.grid(axis='y', linestyle='--', alpha=0.4) # Grid hanya horizontal dan transparan
        sns.despine() # Menghilangkan kotak batas plot

        st.pyplot(fig)
    else:
        st.warning("Tidak ada data historis yang valid.")

    with st.expander("Lihat Tabel Data Asli"):
        st.dataframe(df_historis)

# --- PROSES MACHINE LEARNING ---
df_train = df_wilayah.dropna(subset=['Jumlah Penumpang'])

if not df_train.empty:
    X_train = df_train[['Tahun', 'Bulan_Angka']]
    y_train = df_train['Jumlah Penumpang']

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # --- TAB 2: PREDIKSI 2026 ---
    with tab2:
        st.subheader(f"Estimasi Sisa Tahun 2026")

        df_2026 = df_wilayah[df_wilayah['Tahun'] == 2026]
        bulan_kosong = df_2026[df_2026['Jumlah Penumpang'].isna()]['Bulan_Angka'].tolist()

        if bulan_kosong:
            X_pred_2026 = pd.DataFrame({'Tahun': [2026]*len(bulan_kosong), 'Bulan_Angka': bulan_kosong})
            pred_2026 = model.predict(X_pred_2026)

            df_hasil_2026 = pd.DataFrame({
                'Bulan': [daftar_bulan[i-1] for i in bulan_kosong],
                'Estimasi (Ribu)': np.round(pred_2026, 2)
            })

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(df_hasil_2026, hide_index=True)

            with col_b:
                df_aktual_2026 = df_2026.dropna(subset=['Jumlah Penumpang']).copy()
                df_aktual_2026['Status'] = 'Aktual'

                df_pred_plot_2026 = pd.DataFrame({
                    'Bulan': df_hasil_2026['Bulan'],
                    'Jumlah Penumpang': df_hasil_2026['Estimasi (Ribu)'],
                    'Bulan_Angka': bulan_kosong,
                    'Status': 'Estimasi'
                })

                df_gabung_2026 = pd.concat([df_aktual_2026, df_pred_plot_2026]).sort_values('Bulan_Angka')

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                # Menggunakan palet: Dark Blue untuk aktual, Salmon Pink untuk estimasi
                sns.barplot(data=df_gabung_2026, x='Bulan', y='Jumlah Penumpang', hue='Status',
                            palette={'Aktual': COLOR_5, 'Estimasi': COLOR_2}, ax=ax2)

                ax2.set_title("Perbandingan Aktual vs Estimasi", pad=15)
                ax2.set_ylabel("")
                ax2.set_xlabel("")
                ax2.grid(axis='y', linestyle='--', alpha=0.4)
                sns.despine()
                plt.xticks(rotation=45)
                st.pyplot(fig2)
        else:
            st.info("Data tahun 2026 telah lengkap.")

    # --- TAB 3: PREDIKSI 2027 ---
    with tab3:
        st.subheader("Proyeksi Tahunan 2027")

        X_pred_2027 = pd.DataFrame({'Tahun': [2027]*12, 'Bulan_Angka': list(range(1, 13))})
        pred_2027 = model.predict(X_pred_2027)

        df_hasil_2027 = pd.DataFrame({
            'Bulan': daftar_bulan,
            'Proyeksi (Ribu)': np.round(pred_2027, 2)
        })

        total_2027 = np.sum(pred_2027)

        st.metric(label="Total Penumpang 2027 (Proyeksi)", value=f"{total_2027:,.2f} Ribu")

        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.dataframe(df_hasil_2027, hide_index=True)

        with col_d:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            # Menggunakan warna Mauve/Rose untuk garis proyeksi
            sns.lineplot(data=df_hasil_2027, x='Bulan', y='Proyeksi (Ribu)', marker='s', color=COLOR_3, ax=ax3, linewidth=2)

            ax3.set_title("Tren Bulanan 2027", pad=15)
            ax3.set_ylabel("")
            ax3.set_xlabel("")
            ax3.grid(axis='y', linestyle='--', alpha=0.4)
            sns.despine()
            plt.xticks(rotation=45)
            st.pyplot(fig3)
else:
    st.warning("Data tidak mencukupi untuk memproses model.")