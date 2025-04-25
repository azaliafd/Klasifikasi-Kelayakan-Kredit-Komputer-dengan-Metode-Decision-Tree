# Klasifikasi Kelayakan Kredit Komputer dengan Decision Tree

## Deskripsi Proyek
Proyek ini bertujuan untuk membangun model klasifikasi menggunakan algoritma **Decision Tree** guna memprediksi apakah seseorang layak mendapatkan kredit komputer atau tidak, berdasarkan data dummy yang disediakan. Proses ini melibatkan berbagai tahapan dari pra-pemrosesan data hingga evaluasi performa model.

## Langkah-langkah Implementasi Model Decision Tree

### 1. Import Libraries yang Dibutuhkan dan Membaca Dataset
Dataset yang digunakan adalah `dataset_buys_comp.csv`. Dataset ini dibaca menggunakan library `pandas`.

### 2. Menampilkan Data Awal
Menampilkan 5 data pertama untuk melihat seperti apa bentuk dataset yang digunakan.

### 3. Memeriksa Data yang Hilang
Memeriksa apakah ada data yang hilang pada dataset.

### 4. Menampilkan Nilai Unik dari Setiap Kolom
Menampilkan nilai-nilai unik yang ada pada setiap kolom dataset.

### 5. Melakukan Encoding pada Kolom Kategorikal
Menggunakan `LabelEncoder` untuk mengubah data kategorikal menjadi numerik pada setiap kolom, kecuali kolom target.

### 6. Membagi Data Menjadi Fitur dan Target
Fitur (`X`) dan target (`y`) dipisahkan, dengan target berupa kolom `Buys_Computer`.

### 7. Membagi Data Menjadi Training dan Testing Set
Data dibagi menjadi dua bagian: data pelatihan (80%) dan data pengujian (20%).

### 8. Membuat dan Melatih Model Decision Tree
Dua model Decision Tree dibuat: model default dan model dengan kriteria `entropy`.

### 9. Evaluasi Model
Evaluasi dilakukan dengan menggunakan metrik seperti akurasi, confusion matrix, dan classification report untuk model dengan kriteria `entropy`.

#### 10. Visualisasi Pohon Keputusan
Pohon keputusan divisualisasikan untuk kedua model, yaitu model default dan model dengan kriteria `entropy`.

#### 11. Menampilkan Feature Importance
Menampilkan tingkat pentingnya fitur untuk masing-masing model.

#### 12. Menguji dengan Data Baru
Prediksi dilakukan menggunakan model dengan data baru yang dimasukkan ke dalam format yang sesuai.

#### 13. Hasil Prediksi
Setelah dilakukan prediksi terhadap data baru, hasilnya akan menunjukkan apakah seseorang dianggap "Layak" atau "Tidak Layak" untuk membeli komputer, berdasarkan dua model yang berbeda.
