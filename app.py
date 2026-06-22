import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. KONFIGURASI HALAMAN & CSS
# ==========================================
# Konfigurasi harus selalu berada paling atas
st.set_page_config(page_title="Sylvair", page_icon="🍃", layout="centered", initial_sidebar_state="collapsed")

# Custom CSS untuk tampilan Minimalis Modern
st.markdown("""
    <style>
    /* Menyembunyikan elemen default Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mengubah jarak atas agar lebih proporsional */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Styling tombol utama */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    /* Styling container kartu */
    div[data-testid="stVerticalBlock"] div[data-testid="stVerticalBlock"] {
        background-color: rgba(255, 255, 255, 0.02);
        border-radius: 10px;
        padding: 1rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. INISIALISASI & LOAD RESOURCE
# ==========================================
@st.cache_resource
def load_resources():
    if not os.path.exists('models/sylvair_best_model.pkl'):
        st.error("Model AI tidak ditemukan! Silakan jalankan 'python train.py' terlebih dahulu.")
        st.stop()
        
    model = joblib.load('models/sylvair_best_model.pkl')
    encoders = joblib.load('models/sylvair_encoders.pkl')
    feature_names = joblib.load('models/sylvair_features.pkl')
    
    return model, encoders, feature_names

# ==========================================
# 3. HEADER & HERO SECTION
# ==========================================
st.markdown("<h1 style='text-align: center; color: #2E86C1; margin-bottom: 0;'>🍃 Sylvair</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: gray; font-size: 1.1rem;'>Sistem Prediksi Risiko Penyakit Paru-Paru</p>", unsafe_allow_html=True)
st.divider()

with st.spinner("Menyiapkan engine..."):
    model, encoders, feature_names = load_resources()

# ==========================================
# 4. ANTARMUKA INPUT PENGGUNA (FORM)
# ==========================================
st.markdown("### 📋 Profil Pasien")
st.caption("Lengkapi formulir di bawah ini untuk memulai analisis.")

# Menggunakan form agar web tidak ter-refresh sebelum tombol ditekan
with st.form(key='sylvair_form', clear_on_submit=False):
    col1, col2 = st.columns(2)
    
    with col1:
        usia = st.selectbox("Usia", ['Tua', 'Muda'])
        jk = st.selectbox("Jenis Kelamin", ['Pria', 'Wanita'])
        merokok = st.selectbox("Status Merokok", ['Aktif', 'Pasif'])
        bekerja = st.selectbox("Status Pekerjaan", ['Ya', 'Tidak'])
        rt = st.selectbox("Aktivitas Rumah Tangga", ['Ya', 'Tidak'])

    with col2:
        begadang = st.selectbox("Aktivitas Begadang", ['Ya', 'Tidak'])
        olahraga = st.selectbox("Intensitas Olahraga", ['Sering', 'Jarang'])
        asuransi = st.selectbox("Kepemilikan Asuransi", ['Ada', 'Tidak'])
        penyakit = st.selectbox("Penyakit Bawaan", ['Ada', 'Tidak'])
        
    st.markdown("<br>", unsafe_allow_html=True)
    submit_button = st.form_submit_button(label="🔍 Jalankan Analisis Sylvair", type="primary")

# ==========================================
# 5. PROSES PREDIKSI & OUTPUT HASIL
# ==========================================
if submit_button:
    # Memetakan input
    input_df = pd.DataFrame([[usia, jk, merokok, bekerja, rt, begadang, olahraga, asuransi, penyakit]], 
                              columns=feature_names)
    
    # Transformasi data
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
        
    # Prediksi
    pred_class = model.predict(input_df)
    pred_proba = model.predict_proba(input_df)
    
    result_label = encoders['Hasil'].inverse_transform(pred_class)[0]
    max_proba = pred_proba[0].max()
    
    # Tampilan Hasil
    st.divider()
    st.markdown("### 📊 Hasil Analisis")
    
    res_col1, res_col2 = st.columns([2, 1])
    
    with res_col1:
        if result_label == 'Ya':
            st.error("#### ⚠️ Berisiko Tinggi\nBerdasarkan pola data gaya hidup dan riwayat kesehatan, pasien memiliki **indikasi kuat** risiko penyakit paru-paru. Disarankan untuk segera melakukan konsultasi medis.")
        else:
            st.success("#### ✅ Status Aman\nPasien saat ini diprediksi **tidak memiliki risiko signifikan** terhadap penyakit paru-paru. Pertahankan pola hidup sehat!")
            
    with res_col2:
        # Menampilkan probabilitas dalam format metrik yang elegan
        st.metric(label="Confidence Score", value=f"{max_proba:.2f}")