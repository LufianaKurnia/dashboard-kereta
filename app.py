import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeRegressor

# 1. SETUP HALAMAN
st.set_page_config(page_title="Dashboard Kereta Api", page_icon="🚆", layout="wide")
st.title("🚆 Dashboard & Prediksi Penumpang Kereta Api (2020-2027)")
st.markdown("Dashboard ini menganalisis data historis penumpang kereta api dan memprediksi jumlah penumpang untuk bulan-bulan mendatang menggunakan Machine Learning (Decision Tree).")

# 2. FUNGSI MEMUAT & MENGGABUNGKAN DATA
@st.cache_data
def load_data():
    tahun_list = [2020, 2021, 2022, 2023, 2024, 2025, 2026]
    df_all = pd.DataFrame()
    bulan_str = ['Januari', 'Februari', 'Maret', 'April', 'Mei', 'Juni', 'Juli', 'Agustus', 'September', 'Oktober', 'November', 'Desember']

    for thn in tahun_list:
        # Penyesuaian nama file khusus 2026 sesuai datamu
        if thn == 2026:
            nama_file = "Jumlah Penumpang Kereta Api, 2026_2.csv"
        else:
            nama_file = f"Jumlah Penumpang Kereta Api, {thn}.csv"

        try:
            # Baca file dan lewati baris judul BPS
            df = pd.read_csv(nama_file, skiprows=3)
            df.rename(columns={'Unnamed: 0': 'Wilayah/Jenis'}, inplace=True)

            # Ubah data lebar ke panjang (Melt)
            df_melted = df.melt(id_vars=['Wilayah/Jenis'], value_vars=bulan_str,
                                var_name='Bulan', value_name='Jumlah Penumpang')
            df_melted['Tahun'] = thn

            # Gabungkan dengan dataframe utama
            df_all = pd.concat([df_all, df_melted], ignore_index=True)
        except Exception as e:
            # Lewati jika ada file yang tidak terbaca
            pass

    # Pembersihan Data
    df_all['Jumlah Penumpang'] = pd.to_numeric(df_all['Jumlah Penumpang'], errors='coerce')

    # Mapping Bulan ke Angka untuk Machine Learning
    bulan_map = {b: i+1 for i, b in enumerate(bulan_str)}
    df_all['Bulan_Angka'] = df_all['Bulan'].map(bulan_map)

    return df_all, bulan_str

# Panggil fungsi load data
df_gabungan, daftar_bulan = load_data()

# Pastikan data ada sebelum lanjut
if df_gabungan.empty:
    st.error("Data tidak ditemukan! Pastikan nama file CSV sesuai (2020.csv hingga 2026_2.csv) dan berada di folder yang sama.")
    st.stop()

# 3. SIDEBAR: FITUR FILTER & PENCARIAN
st.sidebar.header("🔍 Filter Data")
# Ambil daftar wilayah yang unik
daftar_wilayah = df_gabungan['Wilayah/Jenis'].dropna().unique()
wilayah_pilihan = st.sidebar.selectbox("Pilih Wilayah/Jenis Kereta:", daftar_wilayah)

# Slider Rentang Tahun
tahun_pilihan = st.sidebar.slider("Rentang Tahun Historis:", min_value=2020, max_value=2026, value=(2020, 2026))

# Memfilter data berdasarkan input user
df_wilayah = df_gabungan[df_gabungan['Wilayah/Jenis'] == wilayah_pilihan].copy()
df_historis = df_wilayah[(df_wilayah['Tahun'] >= tahun_pilihan[0]) & (df_wilayah['Tahun'] <= tahun_pilihan[1])]


# 4. TAMPILAN UTAMA (MENGGUNAKAN TAB)
tab1, tab2, tab3 = st.tabs(["📊 Data & Visualisasi Historis", "🔮 Prediksi Sisa 2026", "🚀 Prediksi Total 2027"])

# --- TAB 1: DATA HISTORIS ---
with tab1:
    st.subheader(f"Tren Penumpang: {wilayah_pilihan} ({tahun_pilihan[0]} - {tahun_pilihan[1]})")

    # Menyatukan Tahun dan Bulan menjadi format Datetime untuk Line Chart
    df_historis_plot = df_historis.dropna(subset=['Jumlah Penumpang']).copy()
    if not df_historis_plot.empty:
        df_historis_plot['Tanggal'] = pd.to_datetime(df_historis_plot['Tahun'].astype(str) + '-' + df_historis_plot['Bulan_Angka'].astype(str) + '-01')
        df_historis_plot = df_historis_plot.sort_values('Tanggal')

        # Plot Seaborn
        fig, ax = plt.subplots(figsize=(12, 5))
        sns.lineplot(data=df_historis_plot, x='Tanggal', y='Jumlah Penumpang', marker='o', color='#1f77b4', linewidth=2, ax=ax)
        ax.set_title(f"Grafik Historis Penumpang - {wilayah_pilihan}", fontsize=14)
        ax.set_ylabel("Jumlah Penumpang (Ribu Orang)")
        ax.set_xlabel("Waktu (Bulan-Tahun)")
        ax.grid(True, linestyle='--', alpha=0.6)
        st.pyplot(fig)
    else:
        st.warning("Tidak ada data historis yang valid untuk dirender.")

    with st.expander("Klik untuk melihat Tabel Data Asli"):
        st.dataframe(df_historis)

# --- PROSES MACHINE LEARNING (DECISION TREE) ---
# Kita buang data yang kosong/NaN agar model bisa belajar dari data yang ada angkanya
df_train = df_wilayah.dropna(subset=['Jumlah Penumpang'])

if not df_train.empty:
    # Fitur (X) dan Target (y)
    X_train = df_train[['Tahun', 'Bulan_Angka']]
    y_train = df_train['Jumlah Penumpang']

    # Latih Model
    model = DecisionTreeRegressor(random_state=42)
    model.fit(X_train, y_train)

    # --- TAB 2: PREDIKSI 2026 ---
    with tab2:
        st.subheader(f"Prediksi Kekosongan Data Tahun 2026 ({wilayah_pilihan})")

        # Cari data 2026 yang kolom penumpangnya NaN (Kosong)
        df_2026 = df_wilayah[df_wilayah['Tahun'] == 2026]
        bulan_kosong = df_2026[df_2026['Jumlah Penumpang'].isna()]['Bulan_Angka'].tolist()

        if bulan_kosong:
            # Lakukan Prediksi
            X_pred_2026 = pd.DataFrame({'Tahun': [2026]*len(bulan_kosong), 'Bulan_Angka': bulan_kosong})
            pred_2026 = model.predict(X_pred_2026)

            df_hasil_2026 = pd.DataFrame({
                'Bulan': [daftar_bulan[i-1] for i in bulan_kosong],
                'Prediksi Penumpang (Ribu)': np.round(pred_2026, 2)
            })

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write("Perkiraan untuk sisa bulan:")
                st.dataframe(df_hasil_2026, hide_index=True)

            with col_b:
                # Gabungkan data aktual (sudah lewat) dan prediksi untuk bar chart 2026
                df_aktual_2026 = df_2026.dropna(subset=['Jumlah Penumpang']).copy()
                df_aktual_2026['Status'] = 'Aktual'

                df_pred_plot_2026 = pd.DataFrame({
                    'Bulan': df_hasil_2026['Bulan'],
                    'Jumlah Penumpang': df_hasil_2026['Prediksi Penumpang (Ribu)'],
                    'Bulan_Angka': bulan_kosong,
                    'Status': 'Prediksi'
                })

                df_gabung_2026 = pd.concat([df_aktual_2026, df_pred_plot_2026]).sort_values('Bulan_Angka')

                fig2, ax2 = plt.subplots(figsize=(10, 4))
                sns.barplot(data=df_gabung_2026, x='Bulan', y='Jumlah Penumpang', hue='Status',
                            palette={'Aktual':'#1f77b4', 'Prediksi':'#ff7f0e'}, ax=ax2)
                ax2.set_title("Perbandingan Aktual vs Prediksi (2026)")
                plt.xticks(rotation=45)
                st.pyplot(fig2)
        else:
            st.success("Semua data bulan di tahun 2026 sudah terisi penuh! Tidak perlu prediksi sisa bulan.")

    # --- TAB 3: PREDIKSI 2027 ---
    with tab3:
        st.subheader(f"Proyeksi Total Tahun Depan (2027) untuk {wilayah_pilihan}")

        # Buat dummy data untuk 12 Bulan di 2027
        X_pred_2027 = pd.DataFrame({'Tahun': [2027]*12, 'Bulan_Angka': list(range(1, 13))})
        pred_2027 = model.predict(X_pred_2027)

        df_hasil_2027 = pd.DataFrame({
            'Bulan': daftar_bulan,
            'Proyeksi (Ribu Orang)': np.round(pred_2027, 2)
        })

        total_2027 = np.sum(pred_2027)

        # Tampilkan Angka Total yang besar (KPI Metric)
        st.metric(label="🌟 Estimasi Total Penumpang Selama 2027", value=f"{total_2027:,.2f} Ribu Orang")

        col_c, col_d = st.columns([1, 2])
        with col_c:
            st.dataframe(df_hasil_2027, hide_index=True)

        with col_d:
            fig3, ax3 = plt.subplots(figsize=(8, 4))
            sns.lineplot(data=df_hasil_2027, x='Bulan', y='Proyeksi (Ribu Orang)', marker='s', color='#2ca02c', ax=ax3, linewidth=2)
            ax3.set_title("Tren Prediksi Sepanjang 2027")
            ax3.grid(True, linestyle=':', alpha=0.7)
            plt.xticks(rotation=45)
            st.pyplot(fig3)
else:
    st.warning("Data historis tidak mencukupi untuk melatih model Machine Learning.")