import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Forecasting Kualanamu🌧️",
    page_icon="🔮",
    layout="wide"
)

# --- 2. FUNGSI UNTUK MEMUAT ASET FORECASTING ---
@st.cache_resource
def load_assets():
    """Memuat model dan scaler forecasting."""
    try:
        model = joblib.load('rf_forecast_model.pkl')
        scaler_X = joblib.load('scaler_X_forecast.pkl')
        scaler_y = joblib.load('scaler_y_forecast.pkl')
    except FileNotFoundError:
        st.error("File model/scaler FORECASTING tidak ditemukan. Jalankan skrip `train_forecast_model.py` terlebih dahulu.")
        return None, None, None, None

    try:
        logo = Image.open('Stmkg.png') # Ganti jika nama file logo Anda berbeda
    except FileNotFoundError:
        logo = None
            
    return model, scaler_X, scaler_y, logo

# --- 3. MEMUAT ASET ---
model, scaler_X, scaler_y, logo = load_assets()

# --- 4. TAMPILAN HEADER ---
col_logo1, col_logo2 = st.columns([1, 6])
with col_logo1:
    if logo:
        st.image(logo, width=150)
with col_logo2:
    st.title("Forecasting Curah Hujan Kualanamu")
    st.markdown("*Implementasi Model **Random Forest** untuk Peramalan Curah Hujan 3 Jam ke Depan*")

st.markdown("---")

# --- 5. TAMPILAN UTAMA ---
if model:
    st.subheader("Input Parameter Cuaca Saat Ini")

    with st.expander("Parameter Awan & Kondisi Cuaca", expanded=True):
        col1, col2, col3 = st.columns(3)
        # ... (Kode input widget sama seperti sebelumnya) ...
        with col1:
            cl = st.number_input("Jenis Awan Rendah (CL)", 0, 9, 5, help="Kode 0-9")
            cla = st.number_input("Jml Awan Rendah/Menengah (Oktas)", 0, 8, 3)
            cm = st.number_input("Jenis Awan Menengah (CM)", 0, 9, 7, help="Kode 0-9")
        with col2:
            ch = st.number_input("Jenis Awan Tinggi (CH)", 0, 9, 6, help="Kode 0-9")
            cc = st.number_input("Tutupan Awan Total (Oktas)", 0, 8, 7)
            ww = st.number_input("Cuaca Saat Pengamatan (WW)", 0, 99, 10, help="Kode 0-99")
        with col3:
            land_cond_map = {0: "0 - Kering", 1: "1 - Basah", 2: "2 - Tergenang Air"}
            lc = st.selectbox("Kondisi Tanah (LAND_COND)", options=list(land_cond_map.keys()), format_func=lambda x: land_cond_map[x], index=1)

    with st.expander("Parameter Suhu, Kelembapan, Angin & Tekanan"):
        # ... (Kode input widget sama seperti sebelumnya) ...
        colA, colB, colC = st.columns(3)
        with colA:
            tdry = st.slider("Suhu Udara (°C)", 21.0, 36.0, 27.0, 0.1)
            td = st.slider("Suhu Titik Embun (°C)", 20.0, 30.0, 24.5, 0.1)
            twet = st.slider("Suhu Bola Basah (°C)", 20.0, 30.0, 25.5, 0.1)
        with colB:
            rh = st.slider("Kelembapan Relatif (%)", 40, 100, 85)
            ws = st.slider("Kecepatan Angin (m/s)", 0.0, 15.0, 4.0, 0.1)
        with colC:
            qff = st.slider("Tekanan QFF (mb)", 1000.0, 1020.0, 1009.5, 0.1)
            qfe = st.slider("Tekanan QFE (mb)", 1000.0, 1020.0, 1008.6, 0.1)
    
    st.markdown("---")

    if st.button("Jalankan Peramalan", type="primary", use_container_width=True):
        features_order = scaler_X.feature_names_in_
        user_inputs = {
            "CLOUD_LOW_TYPE_CL": cl, "CLOUD_LOW_MED_AMT_OKTAS": cla, "CLOUD_MED_TYPE_CM": cm, 
            "CLOUD_HIGH_TYPE_CH": ch, "CLOUD_COVER_OKTAS_M": cc, "LAND_COND": lc,
            "PRESENT_WEATHER_WW": ww, "TEMP_DEWPOINT_C_TDTDTD": td, "TEMP_DRYBULB_C_TTTTTT": tdry,
            "TEMP_WETBULB_C": twet, "WIND_SPEED_FF": ws, "RELATIVE_HUMIDITY_PC": rh,
            "PRESSURE_QFF_MB_DERIVED": qff, "PRESSURE_QFE_MB_DERIVED": qfe
        }
        input_list = [user_inputs[feature] for feature in features_order]
        
        input_array = np.array(input_list).reshape(1, -1)
        input_scaled = scaler_X.transform(input_array)
        prediction_scaled = model.predict(input_scaled)
        prediction_inv = scaler_y.inverse_transform(prediction_scaled.reshape(-1, 1))
        predicted_value = max(0, prediction_inv[0][0])

        if predicted_value < 0.5: kategori, delta_color = "Tidak Hujan", "off"
        elif 0.5 <= predicted_value <= 5: kategori, delta_color = "Hujan Ringan", "normal"
        elif 5 < predicted_value <= 10: kategori, delta_color = "Hujan Sedang", "normal"
        else: kategori, delta_color = "Hujan Lebat", "inverse"
            
        st.subheader("Hasil Peramalan untuk 3 Jam ke Depan")
        col_res1, col_res2 = st.columns(2)
        with col_res1:
            st.metric(label="Peramalan Curah Hujan", value=f"{predicted_value:.2f} mm")
        with col_res2:
            st.metric(label="Kategori", value=kategori, delta=kategori, delta_color=delta_color)
else:
    st.warning("Memuat model gagal. Silakan jalankan skrip `train_forecast_model.py` terlebih dahulu.")

# --- SIDEBAR ---
st.sidebar.header("Tentang Proyek")
st.sidebar.info("Aplikasi ini adalah implementasi model Random Forest untuk forecasting curah hujan 3 jam ke depan di Bandara Kualanamu.")
st.sidebar.write("---")
st.sidebar.write("Dibuat oleh: Samuel F. Pardede")
st.sidebar.header("Hubungi Kreator")
st.sidebar.markdown("👨‍💻 [GitHub](https://github.com/samuelpardede)")
st.sidebar.markdown("📧 [Email](mailto:fernandezsamuel041@gmail.com)")
