### **Skenario 1: Melatih Model Dari Awal (Train from Scratch)**
1. **Ekstrak Proyek:** *Unzip* file `cleanProjek.zip` ke folder kerja.
2. **Persiapan Dataset:** Masukkan video ke `data/NamaDataset/raw/`. Jika ada penambahan data, sesuaikan nama dataset pada skrip terkait agar *path* direktori cocok.
3. **Ekstraksi Frame:** Jalankan skrip ekstraksi frame di folder `scripts`.
4. **Analisis Data (EDA):** Jalankan skrip EDA di folder `utils` untuk memeriksa kualitas data.
5. **Split Dataset:** Bagi data (training/val/test) dengan menjalankan skrip di folder `scripts`.
6. **Konfigurasi Augmentasi:** Atur parameter augmentasi di `utils/data_loader.py` jika diperlukan.
7. **Proses Training:** Mulai pelatihan model melalui skrip di folder `train`.
8. **Monitoring Log:** Ketik `tensorboard --logdir=outputs/logs` di terminal, lalu buka `http://localhost:6006` di browser.
9. **Cek Hasil Evaluasi:** Buka file `Outputs/logs/model/runX/evaluation.json` untuk melihat detail metrik model.
10. **Uji Real-time:** Jalankan `realtime.py` di folder `inference` untuk tes model menggunakan kamera.

---

### **Skenario 2: Menggunakan Model Pre-trained (Siap Pakai)**
1. **Persiapan:** Gunakan repository lengkap (bukan versi *clean project*).
2. **Akses Log & Evaluasi:** Langsung cek performa model lama melalui Tensorboard atau file `.json` di folder `Outputs`.
3. **Uji Real-time:** Jalankan `realtime.py` di folder `inference` untuk mencoba model yang sudah terlatih.