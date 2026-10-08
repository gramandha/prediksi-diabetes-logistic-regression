# Evaluasi Model Klasifikasi — Prediksi Risiko Diabetes

> **Machine Learning Training**
> Penulis: **Gramandha Wega Intyanto**

## Teknologi dan Library

[![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)](https://numpy.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-4C72B0?style=for-the-badge)](https://seaborn.pydata.org/)
[![Joblib](https://img.shields.io/badge/Joblib-333333?style=for-the-badge)](https://joblib.readthedocs.io/)
[![Flask](https://img.shields.io/badge/Flask-000000?style=for-the-badge&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)

---

## Deskripsi Proyek

Proyek ini bertujuan untuk membangun, mengevaluasi, dan menyajikan model klasifikasi **Logistic Regression** yang memprediksi risiko diabetes berdasarkan data klinis pasien. Proyek mencakup seluruh alur kerja *machine learning*, mulai dari eksplorasi data, pelatihan model, evaluasi komprehensif, hingga deployment sebagai aplikasi web interaktif menggunakan **Flask**.

Dataset yang digunakan adalah **Pima Indians Diabetes Dataset**, yang berisi data klinis wanita berusia ≥ 18 tahun.

---

## Struktur Direktori

```
1_evaluasi_model_klasifikasi_gramandha_wega_intyanto/
│
├── 📓 membuat_model_logistic_regression_diabetes.ipynb   # Notebook utama (EDA + Training + Evaluasi)
├── 🌐 app.py                                             # Aplikasi web Flask
├── 📄 requirements.txt                                   # Dependensi Python
│
├── data/
│   ├── diabetes_dataset.csv        # Dataset asli
│   ├── train.csv                   # Data latih (hasil split)
│   ├── test.csv                    # Data uji (hasil split)
│   ├── hasil_prediksi_test.csv     # Output prediksi pada data uji
│   └── metrik_evaluasi.csv         # Ringkasan metrik evaluasi
│
├── models/
│   ├── model_diabetes.pkl          # Model Logistic Regression tersimpan
│   ├── scaler_diabetes.pkl         # StandardScaler tersimpan
│   └── info_model.json             # Metadata model & metrik performa
│
├── gambar/                         # 15 visualisasi hasil EDA & evaluasi
│
└── templates/
    └── index.html                  # Tampilan antarmuka web
```

---

## Detail Model

| Atribut | Nilai |
|---|---|
| **Algoritma** | Logistic Regression |
| **Fitur Input** | Pregnancies, Glucose, Blood Pressure, BMI, Age |
| **Target** | Diabetes / Tidak Diabetes |
| **Threshold Default** | 0.5 |
| **Scikit-learn** | v1.7.2 |
| **Preprocessing** | StandardScaler |

---

## Performa Model

| Metrik | Nilai |
|---|---|
| **Accuracy** | 68.33% |
| **Precision** | 68.85% |
| **Recall** | 68.85% |
| **F1-Score** | 68.85% |
| **ROC-AUC** | **79.36%** |
| **Specificity** | 67.80% |
| **Balanced Accuracy** | 68.32% |
| **MCC** | 0.3665 |
| **CV Accuracy (Mean ± Std)** | 74.58% ± 4.30% |

> ROC-AUC sebesar **0.7936** menunjukkan kemampuan diskriminasi model yang cukup baik untuk membedakan kelas diabetes dan non-diabetes.

---

## Alur Kerja (Notebook)

Notebook `membuat_model_logistic_regression_diabetes.ipynb` mencakup:

1. **Eksplorasi Data (EDA)**
   - Distribusi kelas target
   - Histogram & boxplot fitur
   - Heatmap korelasi antar fitur

2. **Preprocessing**
   - Pembagian data latih / uji
   - Standarisasi fitur dengan StandardScaler

3. **Pelatihan Model**
   - Logistic Regression dengan scikit-learn
   - Learning curve untuk mendeteksi underfitting/overfitting

4. **Evaluasi Model**
   - Confusion Matrix
   - Kurva ROC & Precision-Recall
   - Cross-Validation (5-Fold)
   - Analisis Threshold
   - Koefisien model (interpretasi fitur)

5. **Prediksi**
   - Prediksi pada data uji
   - Contoh prediksi pasien baru

---

## Aplikasi Web (Flask)

Aplikasi web interaktif memungkinkan pengguna memasukkan data klinis secara langsung dan mendapatkan prediksi risiko diabetes secara real-time.

### Fitur Aplikasi
- Form input data klinis (Jumlah Kehamilan, Glukosa, Tekanan Darah, BMI, Usia)
- Tampilan probabilitas risiko beserta kategori (Rendah / Sedang / Tinggi)
- Visualisasi kontribusi tiap fitur terhadap prediksi
- Validasi input di sisi server

### Kategori Risiko
| Probabilitas | Kategori |
|---|---|
| < 35% | Rendah |
| 35% – 60% | Sedang |
| > 60% | Tinggi |

### Endpoint API
| Method | Endpoint | Deskripsi |
|---|---|---|
| `GET` | `/` | Halaman utama dengan form input |
| `POST` | `/predict` | Endpoint prediksi (JSON) |

---

## Cara Menjalankan

### 1. Persiapan Lingkungan

```bash
python -m venv venv-regresi
# Windows
venv-regresi\Scripts\activate
```

### 2. Instalasi Dependensi

```bash
pip install -r requirements.txt
```

### 3. Jalankan Notebook (Opsional — jika ingin melatih ulang model)

Buka dan jalankan seluruh sel pada:
```
membuat_model_logistic_regression_diabetes.ipynb
```
> Model akan tersimpan otomatis ke folder `models/`.

### 4. Jalankan Aplikasi Web

```bash
python app.py
```

Buka browser dan akses: http://127.0.0.1:5000

---

## Dependensi

```
numpy
pandas
scikit-learn
matplotlib
seaborn
joblib
flask
```

---

## Catatan

- Model dibaca langsung dari file `models/model_diabetes.pkl` — pastikan file ini ada sebelum menjalankan `app.py`.
- Jika model belum tersedia, jalankan terlebih dahulu notebook untuk menghasilkannya.
- Dataset yang digunakan adalah **Pima Indians Diabetes Dataset** (sumber: UCI Machine Learning Repository / Kaggle).

---

## Informasi Penulis

| Atribut | Detail |
|---|---|
| **Nama** | Gramandha Wega Intyanto |
| **Konteks** | Tugas 2 — Day 2, Machine Learning Training |
| **Topik** | Evaluasi Model Klasifikasi (Logistic Regression) |

## Lisensi

Proyek ini menggunakan lisensi **Apache License 2.0**. Lihat file [LICENSE](LICENSE) untuk ketentuan lengkap.

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
