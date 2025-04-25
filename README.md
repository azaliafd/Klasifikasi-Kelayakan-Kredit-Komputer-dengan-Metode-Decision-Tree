# Klasifikasi Kelayakan Kredit Komputer dengan Decision Tree

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model klasifikasi menggunakan algoritma **Decision Tree** guna memprediksi apakah seseorang layak mendapatkan kredit komputer atau tidak, berdasarkan data dummy yang disediakan. Proses ini melibatkan berbagai tahapan dari pra-pemrosesan data hingga evaluasi performa model.

## Tahapan Pembuatan Model

### 1. Import Library
Library yang digunakan dalam proyek ini meliputi:
- `pandas` dan `numpy` untuk manipulasi data
- `scikit-learn` untuk machine learning dan evaluasi model
- `matplotlib` dan `seaborn` untuk visualisasi

### 2. Load Dataset
Dataset diunduh dari Google Drive dan dimuat menggunakan pandas. Dataset diasumsikan dalam format .csv.

### 3. Eksplorasi Data Awal (EDA)
- Mengecek dimensi data, tipe data, dan jumlah missing values
- Menampilkan statistik deskriptif
- Visualisasi distribusi dan korelasi awal jika diperlukan

### 4. Pra-pemrosesan Data
Beberapa langkah yang dilakukan:
Menghapus atau mengisi missing values (jika ada)
Encoding: Mengubah data kategorikal ke numerikal (menggunakan One-Hot Encoding atau Label Encoding)
Normalisasi/Standarisasi: Tidak wajib untuk Decision Tree

### 5. Pemisahan Fitur dan Target
Fitur (X) adalah variabel independen
Target (y) adalah label: apakah layak atau tidak

### 6. Split Data Latih dan Uji
Membagi dataset menjadi data training dan testing dengan proporsi 80:20.

### 7. Pembuatan dan Pelatihan Model
Menggunakan algoritma Decision Tree dari sklearn.

### 8. Prediksi
Menggunakan model yang telah dilatih untuk memprediksi data uji.

### 9. Evaluasi Model
Beberapa metrik evaluasi yang digunakan:
- Confusion Matrix
- Classification Report (Precision, Recall, F1-score)
- Accuracy Score
