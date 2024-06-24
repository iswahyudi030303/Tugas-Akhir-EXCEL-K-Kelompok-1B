### Tugas Akhir Kelompok 1B AI PT Stechoq Robotika Indonesia
Nama Kelompok
1.Abdan Syakuroh	
2.Jilan Alhafizh	
3.Rizky Syaifurrahman	
4.Wildan Iswahyudi	
5.Zulfa Aulia	

Mentor
Tsamara Hanifah
## Deteksi Keadaan Paru-paru Menggunakan Metode MobilenetV2

Proyek ini bertujuan untuk mendeteksi keadaan paru-paru pada gambar sinar-X (X-Ray) yang dibagi menjadi 3 kelas: Normal, Pneumonia, dan Pneumothorax. Metode yang digunakan adalah MobilenetV2, dengan menggunakan framework Machine Learning (ML), OpenCV, dan Google Colab.

### Dataset
Dataset yang digunakan berasal dari National Institutes of Health Chest X-Ray Dataset dan Society for Imaging Informatics in Medicine (SIIM). Dataset ini mencakup berbagai gambar sinar-X paru-paru yang sudah dilabeli sesuai dengan keadaan paru-paru yang terdapat dalam gambar tersebut.

### Langkah-langkah Proses
1. **Pemrosesan Data**: Dataset akan diimpor dan diproses untuk persiapan pelatihan model.
2. **Pembagian Data**: Data akan dibagi menjadi data pelatihan dan data pengujian untuk evaluasi model.
3. **Pembuatan Model**: Model MobilenetV2 akan dibangun dan dilatih menggunakan data pelatihan.
4. **Evaluasi Model**: Model akan dievaluasi menggunakan data pengujian untuk melihat tingkat akurasi dalam mendeteksi keadaan paru-paru.
5. **Pengujian**: Model akan diuji menggunakan gambar sinar-X baru untuk melihat performanya dalam situasi nyata.

### Requirements
- Python 3.x
- TensorFlow
- Keras
- OpenCV
- Jupyter Notebook (untuk penggunaan Google Colab)

### Cara Penggunaan
1. Clone repository ini ke lokal Anda.
2. Instal semua requirements dengan menjalankan `pip install -r requirements.txt`.
3. Jalankan notebook `main.ipynb` di Google Colab atau Jupyter Notebook untuk melihat proses pembuatan model dan evaluasi.

### Referensi
- National Institutes of Health Chest X-Ray Dataset: [link](https://www.nih.gov/)
- Society for Imaging Informatics in Medicine (SIIM): [link](https://siim.org/)

### Catatan
Pastikan untuk selalu menggunakan dataset dengan etika dan aturan yang berlaku, serta mematuhi lisensi dataset yang digunakan.

---
