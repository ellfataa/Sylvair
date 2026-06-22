import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("Memulai proses pembangunan model Sylvair yang Robust...")
    
    # ==========================================
    # 1. LOAD DATA
    # ==========================================
    print("-> Memuat dataset...")
    df = pd.read_csv('data/predic_tabel.csv', sep=';')
    
    # ==========================================
    # 2. PREPROCESSING LANJUTAN
    # ==========================================
    print("-> Melakukan preprocessing data...")
    # A. Drop kolom yang tidak relevan
    if 'No' in df.columns:
        df = df.drop(columns=['No'])
        
    # B. Handling Missing Values
    # Karena data kita kategorikal teks, kita isi nilai kosong dengan nilai yang paling sering muncul (modus)
    print("-> Mengecek dan menangani missing values...")
    imputer = SimpleImputer(strategy='most_frequent')
    df_imputed = pd.DataFrame(imputer.fit_transform(df), columns=df.columns)
        
    # C. Encoding fitur kategorikal menjadi numerik
    encoders = {}
    for col in df_imputed.columns:
        le = LabelEncoder()
        df_imputed[col] = le.fit_transform(df_imputed[col])
        encoders[col] = le
        
    # Memisahkan Fitur (X) dan Target (y)
    X = df_imputed.drop(columns=['Hasil'])
    y = df_imputed['Hasil']
    feature_names = X.columns.tolist()

    # ==========================================
    # 3. DATA SPLITTING (70% Train, 30% Test)
    # ==========================================
    print("-> Membagi data: 70% Training dan 30% Testing...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    # ==========================================
    # 4. MODELING & HYPERPARAMETER TUNING
    # ==========================================
    print("-> Melatih Model 1: Random Forest...")
    rf_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    rf = RandomForestClassifier(random_state=42)
    rf_random = RandomizedSearchCV(estimator=rf, param_distributions=rf_param_dist, 
                                   n_iter=10, cv=3, verbose=0, random_state=42, n_jobs=-1)
    rf_random.fit(X_train, y_train)
    rf_best = rf_random.best_estimator_

    print("-> Melatih Model 2: XGBoost...")
    # XGBoost membutuhkan input numerik dan target mulai dari 0
    xgb_param_dist = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb_random = RandomizedSearchCV(estimator=xgb, param_distributions=xgb_param_dist, 
                                    n_iter=10, cv=3, verbose=0, random_state=42, n_jobs=-1)
    xgb_random.fit(X_train, y_train)
    xgb_best = xgb_random.best_estimator_

    # ==========================================
    # 5. EVALUASI DAN KOMPARASI MODEL
    # ==========================================
    print("-> Mengevaluasi performa model...")
    rf_pred = rf_best.predict(X_test)
    xgb_pred = xgb_best.predict(X_test)
    
    rf_acc = accuracy_score(y_test, rf_pred)
    xgb_acc = accuracy_score(y_test, xgb_pred)
    
    print(f"\nAkurasi Random Forest: {rf_acc * 100:.2f}%")
    print(f"Akurasi XGBoost:     {xgb_acc * 100:.2f}%")
    
    # Memilih model terbaik untuk diekspor
    if xgb_acc > rf_acc:
        print("\n🏆 XGBoost terpilih sebagai model terbaik!")
        best_model = xgb_best
        print("Laporan Klasifikasi XGBoost:\n", classification_report(y_test, xgb_pred))
    else:
        print("\n🏆 Random Forest terpilih sebagai model terbaik!")
        best_model = rf_best
        print("Laporan Klasifikasi Random Forest:\n", classification_report(y_test, rf_pred))

    # ==========================================
    # 6. EXPORT (PENYIMPANAN MODEL)
    # ==========================================
    print("-> Menyimpan model terbaik dan arsitektur ke folder 'models'...")
    if not os.path.exists('models'):
        os.makedirs('models')
        
    # Menyimpan model (apa pun yang menang, file-nya akan bernama sylvair_best_model.pkl)
    joblib.dump(best_model, 'models/sylvair_best_model.pkl')
    joblib.dump(encoders, 'models/sylvair_encoders.pkl')
    joblib.dump(feature_names, 'models/sylvair_features.pkl')
    # Kita juga perlu menyimpan imputer agar jika ada input kosong di web, tidak error
    joblib.dump(imputer, 'models/sylvair_imputer.pkl') 
    
    print("Selesai! Engine Sylvair siap digunakan.")

if __name__ == "__main__":
    main()