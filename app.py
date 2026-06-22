import streamlit as st
import pandas as pd
import joblib
import os

# ==========================================
# 1. KONFIGURASI LAYOUT (FULL WIDTH & SIDEBAR EXPANDED)
# ==========================================
st.set_page_config(
    page_title="Sylvair", 
    page_icon="🍃", 
    layout="wide", 
    initial_sidebar_state="expanded"
)

# Custom CSS untuk estetika modern minimalis & DYNAMIC CARDS
st.markdown("""
    <style>
    /* Menyembunyikan elemen default Streamlit */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Mengoptimalkan padding halaman */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Styling tombol submisi */
    .stButton>button {
        width: 100%;
        border-radius: 8px;
        height: 3rem;
        font-weight: 600;
        margin-top: 1rem;
        border: none;
        transition: all 0.3s ease-in-out;
    }
    
    /* --- CSS UNTUK KARTU HASIL (DYNAMIC CARDS) --- */
    .result-card {
        padding: 25px;
        border-radius: 12px;
        margin-top: 20px;
        color: white;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        animation: fadeIn 0.8s ease-in-out;
    }
    
    .card-danger {
        background: linear-gradient(135deg, #e74c3c 0%, #c0392b 100%);
        border-left: 8px solid #922b21;
    }
    
    .card-success {
        background: linear-gradient(135deg, #2ecc71 0%, #27ae60 100%);
        border-left: 8px solid #1e8449;
    }
    
    .score-box {
        background-color: rgba(255, 255, 255, 0.2);
        padding: 10px 20px;
        border-radius: 8px;
        display: inline-block;
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 15px;
        backdrop-filter: blur(5px);
    }
    
    /* Efek animasi kemunculan card */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(20px); }
        to { opacity: 1; transform: translateY(0); }
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

with st.spinner("Menghubungkan ke engine Sylvair..."):
    model, encoders, feature_names = load_resources()

# ==========================================
# 3. STRUKTUR SIDEBAR (LOGO & BRANDING)
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: #2E86C1; margin-bottom: 10px;'>🍃 Sylvair</h2>", unsafe_allow_html=True)

logo_path = "Gemini_Generated_Image_3enx763enx763enx.png"
if os.path.exists(logo_path):
    st.sidebar.image(logo_path, width='stretch')
else:
    st.sidebar.caption("💡 Tempatkan file logo Anda di direktori utama proyek.")

st.sidebar.divider()

st.sidebar.markdown("### 🧠 Engine Status")
st.sidebar.info(
    "**Model:** Random Forest Classifier\n\n"
    "**Akurasi Sistem:** 94.44%\n\n"
    "**Metode Tuning:** RandomizedSearchCV"
)
st.sidebar.caption("Sylvair Decision Support System v1.0")

# ==========================================
# 4. MAIN CONTENT (FORMULIR MULTI-KOLOM)
# ==========================================
st.title("🫁 Dasbor Analisis Risiko Paru-Paru")
st.markdown("Sistem berbasis kecerdasan buatan untuk deteksi dini prediktif berdasarkan profil klinis dan aktivitas harian.")
st.divider()

st.subheader("📋 Form Data Pasien")

with st.form(key='sylvair_wide_form'):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        usia = st.selectbox("Usia Pasien", ['Tua', 'Muda'])
        jk = st.selectbox("Jenis Kelamin", ['Pria', 'Wanita'])
        merokok = st.selectbox("Status Merokok", ['Aktif', 'Pasif'])
        
    with col2:
        bekerja = st.selectbox("Status Bekerja", ['Ya', 'Tidak'])
        rt = st.selectbox("Aktivitas Rumah Tangga", ['Ya', 'Tidak'])
        begadang = st.selectbox("Aktivitas Begadang", ['Ya', 'Tidak'])
        
    with col3:
        olahraga = st.selectbox("Intensitas Olahraga", ['Sering', 'Jarang'])
        asuransi = st.selectbox("Kepemilikan Asuransi", ['Ada', 'Tidak'])
        penyakit = st.selectbox("Penyakit Bawaan", ['Ada', 'Tidak'])
        
    submit_button = st.form_submit_button(label="⚡ Jalankan Diagnostik Prediktif", type="primary")

# ==========================================
# 5. PROSES EVALUASI & OUTPUT HASIL CARDS
# ==========================================
if submit_button:
    # Ekstraksi input ke DataFrame
    input_df = pd.DataFrame([[usia, jk, merokok, bekerja, rt, begadang, olahraga, asuransi, penyakit]], 
                              columns=feature_names)
    
    # Transformasi nilai input menggunakan encoder yang tersimpan
    for col in input_df.columns:
        input_df[col] = encoders[col].transform(input_df[col])
        
    # Inferensi klasifikasi
    pred_class = model.predict(input_df)[0]  # Mengambil nilai kelas prediksi (0 atau 1)
    pred_proba = model.predict_proba(input_df)[0] # Mengambil array probabilitas untuk kelas 0 dan 1
    
    # Menerjemahkan angka kelas kembali ke label ('Ya' atau 'Tidak')
    result_label = encoders['Hasil'].inverse_transform([pred_class])[0]
    
    # Mengambil persentase probabilitas untuk kelas yang TERPILIH oleh model
    # Jika model memprediksi 0, ambil probabilitas indeks 0. Jika memprediksi 1, ambil indeks 1.
    selected_proba = pred_proba[pred_class]
    
    # Rendering area hasil analisis
    st.divider()
    st.subheader("📊 Hasil Diagnostik Sylvair")
    
    # ----------------------------------------------------
    # LOGIKA DYNAMIC CARDS (BERUBAH WARNA SESUAI HASIL)
    # ----------------------------------------------------
    if result_label == 'Ya':
        card_html = f"""
        <div class="result-card card-danger">
            <h2 style="color: white; margin-top: 0;">⚠️ Peringatan: Terindikasi Berisiko Tinggi</h2>
            <p style="font-size: 1.1rem; opacity: 0.9;">
                Profil data pasien menunjukkan kecenderungan pola yang selaras dengan kelompok risiko gangguan paru-paru. Direkomendasikan untuk segera melakukan tindakan preventif atau konsultasi medis lanjutan ke dokter spesialis.
            </p>
            <div class="score-box">
                🎯 Confidence Score Model: <b>{selected_proba:.2f}</b>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
    else:
        card_html = f"""
        <div class="result-card card-success">
            <h2 style="color: white; margin-top: 0;">✅ Status: Kontrol Aman</h2>
            <p style="font-size: 1.1rem; opacity: 0.9;">
                Evaluasi komparatif terhadap parameter gaya hidup menunjukkan profil kesehatan pasien berada dalam kategori aman dari risiko signifikan penyakit paru-paru. Tetap pertahankan pola hidup sehat Anda!
            </p>
            <div class="score-box">
                🎯 Confidence Score Model: <b>{selected_proba:.2f}</b>
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)
        
        # Tambahan sedikit efek balon untuk hasil yang baik agar lebih interaktif
        st.balloons()