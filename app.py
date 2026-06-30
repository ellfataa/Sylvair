import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Sylvair - Dashboard", 
    page_icon="logo-sylvair.webp", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Injeksi CSS
st.markdown("""
    <style>
    /* Mengubah tema utama menjadi latar belakang gelap formal */
    .stApp {
        background-color: #0b0f19;
        color: #f1f5f9;
        font-family: 'Inter', -apple-system, sans-serif;
    }
    
    /* Mengatur estetika panel Sidebar Gelap */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1f2937 !important;
    }
    
    /* Memastikan teks heading di sidebar berwarna terang */
    [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3, [data-testid="stSidebar"] p {
        color: #f1f5f9 !important;
    }
    
    /* Optimalisasi ruang padding utama */
    .main .block-container {
        padding-top: 3rem;
        padding-bottom: 3rem;
        padding-left: 5rem;
        padding-right: 5rem;
    }
    
    /* Memaksa warna label selectbox teks menjadi putih/terang */
    label, .stMarkdown p {
        color: #cbd5e1 !important;
    }
    
    /* Tombol Eksekusi Premium Blue */
    .stButton>button {
        width: 100%;
        border-radius: 6px;
        height: 3.4rem;
        font-size: 1.05rem;
        font-weight: 600;
        background-color: #0284c7 !important;
        color: #ffffff !important;
        border: none !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        transition: background-color 0.2s ease;
    }
    .stButton>button:hover {
        background-color: #0369a1 !important;
        color: #ffffff !important;
    }
    
    /* Pembungkus Formulir Elegan dalam Mode Gelap */
    div[data-testid="stForm"] {
        border: 1px solid #1f2937 !important;
        border-radius: 8px !important;
        padding: 2.5rem !important;
        background-color: #1f2937 !important;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3) !important;
    }
    
    /* --- KARTU DIAGNOSTIK ELEGAN UNTUK TEMA GELAP --- */
    .report-card {
        padding: 30px;
        border-radius: 8px;
        margin-top: 25px;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.4);
        border: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    .report-danger {
        background: linear-gradient(135deg, #7f1d1d 0%, #450a0a 100%);
        border-left: 8px solid #f87171;
        color: #fca5a5;
    }
    
    .report-success {
        background: linear-gradient(135deg, #064e3b 0%, #022c22 100%);
        border-left: 8px solid #4ade80;
        color: #86efac;
    }
    
    .metric-container {
        background-color: rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 12px 20px;
        border-radius: 6px;
        display: inline-block;
        margin-top: 20px;
        font-size: 1.1rem;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_resources():
    if not os.path.exists('models/sylvair_best_model.pkl'):
        st.error("Model AI tidak ditemukan! Silakan jalankan 'python train.py' terlebih dahulu.")
        st.stop()
        
    model = joblib.load('models/sylvair_best_model.pkl')
    encoders = joblib.load('models/sylvair_encoders.pkl')
    feature_names = joblib.load('models/sylvair_features.pkl')
    
    return model, encoders, feature_names

with st.spinner("Menghubungkan ke engine Sylvair..."):
    model, encoders, feature_names = load_resources()

st.sidebar.markdown("<h2 style='text-align: center; color: #38bdf8;'>Sistem Analisis Risiko Penyakit Paru-Paru</h2>", unsafe_allow_html=True)

logo_path = "logo-sylvair.webp"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width='stretch')
else:
    st.sidebar.caption("Tempatkan file 'logo-sylvair.webp' di direktori utama proyek.")

#FORMULIR EVALUASI
st.markdown("<h1 style='color: #f8fafc; font-weight: 800; margin-bottom: 5px; margin-top: 0px;'>Dashboard Analisis Risiko Paru-Paru</h1>", unsafe_allow_html=True)
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem; margin-bottom: 0px;'>Sistem evaluasi prediktif klinis berbasis kecerdasan buatan untuk deteksi dini parameter kesehatan harian menggunakan metode Random Forest dan XGBoost.</p>", unsafe_allow_html=True)
st.divider()

st.markdown("<h4 style='color: #e2e8f0; margin-bottom: 15px;'>Formulir Penginputan Parameter Pasien</h4>", unsafe_allow_html=True)

with st.form(key='sylvair_dark_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        usia = st.selectbox("Klasifikasi Usia Pasien", ['Tua', 'Muda'])
        jk = st.selectbox("Jenis Kelamin", ['Pria', 'Wanita'])
        merokok = st.selectbox("Paparan Asap/Status Merokok", ['Aktif', 'Pasif'])
        
    with col2:
        bekerja = st.selectbox("Status Aktivitas Pekerjaan", ['Ya', 'Tidak'])
        rt = st.selectbox("Aktivitas Urusan Rumah Tangga", ['Ya', 'Tidak'])
        begadang = st.selectbox("Kebiasaan/Aktivitas Begadang", ['Ya', 'Tidak'])
        
    with col3:
        olahraga = st.selectbox("Intensitas Olahraga", ['Sering', 'Jarang'])
        asuransi = st.selectbox("Kepemilikan Jaminan Asuransi", ['Ada', 'Tidak'])
        penyakit = st.selectbox("Riwayat Komorbid/Penyakit Bawaan", ['Ada', 'Tidak'])
        
    submit_button = st.form_submit_button(label="⚡ Jalankan Diagnostik Prediktif")

if submit_button:
    # Transformasi array ke DataFrame
    input_df = pd.DataFrame([[usia, jk, merokok, bekerja, rt, begadang, olahraga, asuransi, penyakit]], 
                              columns=feature_names)
    
    # Label encoding input secara sekuensial
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
        
    # Inferensi model
    pred_class = model.predict(input_df)[0]  
    pred_proba = model.predict_proba(input_df)[0] 
    
    # Dekode label balik
    result_label = encoders['Hasil'].inverse_transform([pred_class])[0]
    selected_proba = pred_proba[pred_class]
    
    st.markdown("<br><h4 style='color: #e2e8f0; margin-bottom: 0px'>Hasil Penilaian Klinis</h4>", unsafe_allow_html=True)
    
    # Kondisi 1: Pasien Terindikasi Risiko Tinggi (Ya)
    if result_label == 'Ya':
        card_html = f"""
        <div class="report-card report-danger" style="margin-top:0px">
            <h3 style="margin-top: 0; font-weight: 700; color: #fca5a5;">⚠️ Hasil Analisis: Terindikasi Risiko Tinggi</h3>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #fca5a5; margin-bottom: 0;">
                Berdasarkan hasil komparasi parameter gaya hidup dan riwayat kondisi komorbid, profil pasien menunjukkan kecenderungan pola yang selaras dengan kelompok riwayat penyakit paru-paru berat. Direkomendasikan untuk segera melakukan skrining penunjang klinis atau melakukan tindakan konsultasi preventif secara langsung kepada dokter spesialis.
            </p>
            <div class="metric-container" style="color: #fca5a5; margin-bottom: 0">
                Tingkat Keyakinan Prediksi (Confidence Score): <b>{selected_proba:.2f}</b>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
    # Kondisi 2: Pasien Berada dalam Kondisi Aman (Tidak)
    else:
        card_html = f"""
        <div class="report-card report-success">
            <h3 style="margin-top: 0; font-weight: 700; color: #86efac;">✅ Hasil Analisis: Profil Kontrol Aman</h3>
            <p style="font-size: 1.05rem; line-height: 1.6; color: #86efac; margin-bottom: 0;">
                Evaluasi komparatif matematis terhadap parameter aktivitas harian menunjukkan bahwa kondisi fisik dan lingkungan pasien saat ini berada dalam koridor risiko rendah dari indikasi penyakit paru-paru. Tetap pelihara pola hidup sehat Anda dan lakukan olahraga secara teratur!
            </p>
            <div class="metric-container" style="color: #86efac; margin-bottom: 0">
                Tingkat Keyakinan Prediksi (Confidence Score): <b>{selected_proba:.2f}</b>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        st.balloons()