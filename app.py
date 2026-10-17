import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

# Pengaturan palet warna kustom dari gambar
COLOR_PRIMARY = "#355C7D"    # Biru Gelap
COLOR_SECONDARY = "#6C5B7B"  # Ungu Redup
COLOR_TERTIARY = "#C06C84"   # Merah Muda Gelap
COLOR_ACCENT_1 = "#F67280"   # Coral
COLOR_ACCENT_2 = "#F8B195"   # Peach

# 1. SETUP HALAMAN
st.set_page_config(page_title="Dashboard Kereta Api", layout="wide")
st.title("Dashboard Prediksi Penumpang Kereta Api")
st.markdown("Analisis data historis dan prediksi jumlah penumpang menggunakan model Decision Tree.")

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
    st.error("Data tidak ditemukan. Pastikan file CSV tersedia.")
    st.stop()

# 3. SIDEBAR: FITUR FILTER
st.sidebar.header("Filter Data")
daftar_wilayah = df_gabungan['Wilayah/Jenis'].dropna().unique()
wilayah_pilihan = st.sidebar.selectbox("Wilayah / Jenis Kereta:", daftar_wilayah)

tahun_pilihan = st.sidebar.slider("Rentang Tahun Historis:", min_value=2020, max_value=2026, value=(2020, 2026))

df_wilayah = df_gabungan[df_gabungan['Wilayah/Jenis'] == wilayah_pilihan].copy()
df_historis = df_wilayah[(df_wilayah['Tahun'] >= tahun_pilihan[0]) & (df_wilayah['Tahun'] <= tahun_pilihan[1])]

# 4. TAMPILAN UTAMA
tab1, tab2, tab3 = st.tabs(["Data Historis", "Prediksi 2026", "Proyeksi 2027"])

# Fungsi styling plot minimalis
def set_minimalist_style(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    ax.grid(axis='y', linestyle='-', alpha=0.3, color='#DDDDDD')

# --- TAB 1: DATA HISTORIS ---
with tab1:
    st.subheader(f"Tren Penumpang: {wilayah_pilihan} ({tahun_pilihan[0]} - {tahun_pilihan[1]})")

    df_historis_plot = df_historis.dropna(subset=['Jumlah Penumpang']).copy()
    if not df_historis_plot.empty:
        df_historis_plot['Tanggal'] = pd.to_datetime(df_historis_plot['Tahun'].astype(str) + '-' + df_historis_plot['Bulan_Angka'].astype(str) + '-01')
        df_historis_plot = df_historis_plot.sort_values('Tanggal')

        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_historis_plot, x='Tanggal', y='Jumlah Penumpang', color=COLOR_PRIMARY, linewidth=2, ax=ax)

        ax.set_title("Grafik Historis Penumpang", fontsize=12, pad=15, color='#333333')
        ax.set_ylabel("Jumlah Penumpang (Ribu Orang)", fontsize=10, color='#666666')
        ax.set_xlabel("", fontsize=10)
        set_minimalist_style(ax)

        st.pyplot(fig)
    else:
        st.warning("Tidak ada data historis yang valid.")

    with st.expander("Tabel Data Asli"):
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
        st.subheader("Prediksi Kekosongan Data (2026)")

        df_2026 = df_wilayah[df_wilayah['Tahun'] == 2026]
        bulan_kosong = df_2026[df_2026['Jumlah Penumpang'].isna()]['Bulan_Angka'].tolist()

        if bulan_kosong:
            X_pred_2026 = pd.DataFrame({'Tahun': [2026]*len(bulan_kosong), 'Bulan_Angka': bulan_kosong})
            pred_2026 = model.predict(X_pred_2026)

            df_hasil_2026 = pd.DataFrame({
                'Bulan': [daftar_bulan[i-1] for i in bulan_kosong],
                'Prediksi (Ribu)': np.round(pred_2026, 2)
            })

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.dataframe(df_hasil_2026, hide_index=True)

            with col_b:
                df_aktual_2026 = df_2026.dropna(subset=['Jumlah Penumpang']).copy()
                df_aktual_2026['Kategori'] = 'Aktual'

                df_pred_plot_2026 = pd.DataFrame({
                    'Bulan': df_hasil_2026['Bulan'],
                    'Jumlah Penumpang': df_hasil_2026['Prediksi (Ribu)'],
                    'Bulan_Angka': bulan_kosong,
                    'Kategori': 'Prediksi'
                })

                df_gabung_2026 = pd.concat([df_aktual_2026, df_pred_plot_2026]).sort_values('Bulan_Angka')

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                sns.barplot(data=df_gabung_2026, x='Bulan', y='Jumlah Penumpang', hue='Kategori',
                            palette={'Aktual': COLOR_SECONDARY, 'Prediksi': COLOR_ACCENT_1}, ax=ax2)

                ax2.set_title("Perbandingan Aktual vs Prediksi", fontsize=12, pad=15, color='#333333')
                ax2.set_ylabel("")
                ax2.set_xlabel("")
                set_minimalist_style(ax2)
                plt.xticks(rotation=45)

                st.pyplot(fig2)
        else:
            st.success("Semua data bulan di tahun 2026 sudah terisi penuh.")

    # --- TAB 3: PREDIKSI 2027 ---
    with tab3:
        st.subheader("Proyeksi Total (2027)")

        X_pred_2027 = pd.DataFrame({'Tahun': [2027]*12, 'Bulan_Angka': list(range(1, 13))})
        pred_2027 = model.predict(X_pred_2027)

        df_hasil_2027 = pd.DataFrame({
            'Bulan': daftar_bulan,
            'Proyeksi (Ribu Orang)': np.round(pred_2027, 2)
        })

        total_2027 = np.sum(pred_2027)
        st.metric(label="Estimasi Total Penumpang 2027", value=f"{total_2027:,.2f} Ribu Orang")

        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.dataframe(df_hasil_2027, hide_index=True)

        with col_d:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df_hasil_2027, x='Bulan', y='Proyeksi (Ribu Orang)', color=COLOR_TERTIARY, ax=ax3, linewidth=2)

            ax3.set_title("Tren Prediksi Sepanjang 2027", fontsize=12, pad=15, color='#333333')
            ax3.set_ylabel("")
            ax3.set_xlabel("")
            set_minimalist_style(ax3)
            plt.xticks(rotation=45)

            st.pyplot(fig3)
else:
    st.warning("Data historis tidak mencukupi untuk melatih model.")