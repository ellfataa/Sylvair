import pandas as pd
import random

def main():
    print("Membaca file CSV...")
    # Load data dari folder data
    df = pd.read_csv('data/predic_tabel.csv', sep=';')
    
    # Fungsi logika medis
    def perbaiki_hasil(row):
        score = 0
        
        # Memberikan bobot risiko (Poin)
        if row['Merokok'] == 'Aktif': 
            score += 45  # Perokok aktif penyumbang risiko terbesar
        if row['Penyakit_Bawaan'] == 'Ada': 
            score += 30
        if row['Usia'] == 'Tua': 
            score += 15
        if row['Aktivitas_Begadang'] == 'Ya': 
            score += 10
        if row['Aktivitas_Olahraga'] == 'Jarang': 
            score += 10
            
        # Faktor Pelindung
        if row['Aktivitas_Olahraga'] == 'Sering': 
            score -= 10
            
        # Menambahkan sedikit noise (acak) agar data tetap realistis dan model tidak overfit
        score += random.randint(-10, 10)
        
        # Threshold: Jika poin risiko di atas 50, maka pasien terindikasi penyakit paru-paru
        if score >= 50:
            return 'Ya'
        else:
            return 'Tidak'

    print("Menerapkan logika medis baru ke 30.000 baris data...")
    # Terapkan fungsi ke setiap baris untuk memperbarui kolom 'Hasil'
    df['Hasil'] = df.apply(perbaiki_hasil, axis=1)
    
    # Simpan kembali (overwrite) file CSV-nya
    df.to_csv('data/predic_tabel.csv', index=False, sep=';')
    print("Selesai! File predic_tabel.csv berhasil diperbaiki.")

if __name__ == "__main__":
    main()