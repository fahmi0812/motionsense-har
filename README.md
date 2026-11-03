# MotionSense: Sistem Web Pengenalan Aktivitas Manusia Menggunakan CNN

> **Catatan dataset:** folder `data/` (UCI HAR Dataset, zip, `.npz`, CSV privat) **tidak di-commit** karena besar.
> Unduh/siapkan ulang dengan `python training/01_siapkan_publik.py` (mengunduh UCI HAR), lalu
> `python training/02_siapkan_privat.py` (atau `training/make_dummy_private.py` untuk data dummy), atau cukup `./run_training.sh`.
> Model siap pakai untuk demo web sudah ada di `web/model/`.


Tugas UAS PCR — Machine Learning untuk IoT. Tim 2 orang.

## 1. Ringkasan
HP merekam sensor gerak (akselerometer + giroskop). Data dikirim ke Supabase. Model **1D-CNN** menebak aktivitas
setiap potongan 2,56 detik: berjalan, naik tangga, turun tangga, duduk, berdiri, berbaring.
Hasilnya ditampilkan di web: grafik sinyal, garis waktu tebakan, dan total durasi tiap aktivitas.
Sebagai pembanding dipakai **Random Forest** dengan fitur sederhana (rata-rata, simpangan, min, maks per sumbu).

Data latih: 75% data publik (UCI HAR, 1500 potongan) + 25% data sendiri (500 potongan).

## 2. Isi folder
```
MotionSense_CNN/
├─ README.md               panduan ini
├─ PROMPT-AI.md            prompt untuk melanjutkan pengembangan dengan AI
├─ .env.example            contoh isian kunci Supabase (salin jadi .env)
├─ requirements-training.txt   paket untuk training (ada torch)
├─ run_training.sh         jalankan semua langkah training
├─ run_web.sh / run_web.bat    jalankan web (satu perintah)
├─ training/
│  ├─ common.py            fungsi bersama (resample 50 Hz, potong window, fitur)
│  ├─ 01_siapkan_publik.py unduh UCI HAR, ambil 1500 window seimbang
│  ├─ make_dummy_private.py  BUAT DATA PALSU (hanya untuk uji alur!)
│  ├─ 02_siapkan_privat.py data sendiri (CSV/Supabase) -> 500 window + laporan komposisi
│  ├─ 03_latih.py          latih CNN + Random Forest, evaluasi
│  ├─ 04_ekspor_onnx.py    ekspor CNN ke ONNX + cek hasil sama
│  └─ laporan_komposisi.md bukti 75/25 (hasil run terakhir)
├─ models/                 hasil training: metrics.json, gambar confusion matrix, rf.joblib, cnn.onnx
├─ web/
│  ├─ app.py, inference.py backend FastAPI + onnxruntime (tanpa torch)
│  ├─ requirements.txt     paket web (ringan)
│  ├─ model/               cnn.onnx, norm.json, metrics.json, gambar
│  ├─ static/              halaman web (HTML + Chart.js)
│  └─ sample_data/DUMMY_contoh_sesi.csv   contoh sesi PALSU untuk demo offline
└─ docs/  SETUP-Supabase.md, DEPLOY.md
```

## 3. Cara menjalankan

### 3.1 Siapkan Python (sekali saja)
Butuh Python 3.10–3.13.
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/Mac: source .venv/bin/activate
pip install torch==2.15.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-training.txt
```
Kalau hanya ingin menjalankan web: `pip install -r web/requirements.txt` (tanpa torch).

### 3.2 Coba langsung (tanpa Supabase)
Model hasil latih sudah ada di `web/model/`. Cukup:
```bash
bash run_web.sh        # Windows: run_web.bat
```
Buka http://localhost:8000 → pilih sesi `DUMMY_contoh_sesi` → **Prediksi sesi**.
Halaman **Evaluasi** membandingkan CNN dan Random Forest.

### 3.3 Mengumpulkan data sendiri (1 hari)
1. Siapkan Supabase: ikuti `docs/SETUP-Supabase.md` (tabel `sensor_readings`).
2. HP diselipkan di pinggang (seperti UCI HAR). Pakai APK perekam atau phyphox.
3. Rekam **±2 menit per aktivitas × 6 aktivitas**, per orang. Dua orang × 6 × 2 menit cukup untuk ±500 window.
   Satu rekaman = satu `session_id` dengan satu label.
4. Ekspor phyphox: simpan CSV di `data/privat_csv/` dengan kolom
   `session_id,ts,ax,ay,az,gx,gy,gz,label` (acc boleh m/s², otomatis diubah ke g).

### 3.4 Training
```bash
bash run_training.sh --supabase   # ambil data dari Supabase (isi .env dulu)
bash run_training.sh              # atau dari CSV di data/privat_csv/
bash run_training.sh --dummy      # HANYA uji alur dengan data palsu
```
Langkahnya: unduh UCI HAR (±60 MB, sekali) → 1500 window publik → data sendiri diubah ke g, diseragamkan 50 Hz,
dipotong 128 sampel dengan tumpang-tindih 50% → ambil 500 window → latih → ekspor ONNX ke `web/model/`.
Di laptop uji, seluruh training ±20 detik.

### 3.5 Menghubungkan web ke Supabase
Salin `.env.example` menjadi `.env`, isi `SUPABASE_URL` dan kunci. Jalankan ulang `run_web.sh`.
Label di pojok kiri atas berubah menjadi "online (Supabase)". Tanpa `.env`, web otomatis memakai mode offline.

## 4. Alur data
```
HP (sensor 50-100 Hz) --APK/phyphox--> Supabase tabel sensor_readings
      --> training (laptop): resample 50 Hz, window 128, CNN --> cnn.onnx
      --> web FastAPI: ambil sesi dari Supabase REST --> onnxruntime --> grafik di browser
```

## 5. Cara evaluasi (agar adil)
- **Skenario A:** latih dengan semua data publik, uji dengan semua data sendiri. Menguji apakah model tetap jalan di HP lain.
- **Skenario B (campuran):** data publik dipisah **per orang** (6 subjek UCI hanya untuk uji),
  data sendiri dipisah **per sesi rekaman**. Jadi tidak ada potongan dari rekaman yang sama di latih dan uji.
- Yang dilaporkan: akurasi, F1 per kelas, F1 rata-rata, confusion matrix — untuk CNN dan Random Forest
  (`models/metrics.json`, `models/cm_*.png`).

### Hasil uji alur (data privat = DUMMY/PALSU)
Angka di bawah dari run sungguhan tanggal 29-10-2026, tetapi data privatnya **dibuat oleh `make_dummy_private.py`**.
Jangan dipakai di laporan. Ganti dengan hasil setelah memakai data rekaman asli.

| Skenario | CNN akurasi / F1 | RF akurasi / F1 |
|---|---|---|
| A: publik → privat DUMMY | 0,694 / 0,634 | 0,728 / 0,682 |
| B: campuran (6 subjek UCI + sesi DUMMY) | 0,904 / 0,907 | 0,848 / 0,854 |

Ekspor ONNX: selisih maksimal dengan PyTorch 3e-7, kelas sama 100%.

## 6. Tips presentasi
- Buka web 1 menit sebelumnya (server gratis bisa "tidur").
- Demo: rekam 30 detik jalan lalu duduk → sync → pilih sesi → tunjukkan garis waktu berubah warna.
- Siapkan mode offline (CSV contoh) sebagai cadangan bila internet kampus bermasalah.
- Jelaskan kenapa Skenario A lebih rendah: posisi HP dan sensor berbeda dengan UCI.
- Tunjukkan `training/laporan_komposisi.md` sebagai bukti 75/25.

## 7. Masalah umum (RAM kecil)
- **Laptop lambat / RAM penuh:** tutup browser lain; training memakai 2 thread (`torch.set_num_threads(2)` di `03_latih.py`). Bisa turunkan `EPOCHS` ke 15.
- **Instal torch lama/gagal:** pakai versi CPU (perintah di 3.1), ukurannya jauh lebih kecil dari versi GPU.
- **Unduhan UCI gagal:** unduh manual dari https://archive.ics.uci.edu/dataset/240 , taruh `har.zip` di `data/`.
- **Web: "Supabase error 401/permission":** RLS hanya mengizinkan INSERT. Pakai `SUPABASE_SERVICE_KEY` di server atau tambah policy baca (lihat `docs/SETUP-Supabase.md` bagian 6).
- **"Data terlalu pendek":** butuh minimal 2,56 detik rekaman.
- **Akurasi data sendiri rendah:** pastikan posisi HP sama untuk semua rekaman dan label benar.

## 8. Deploy
Lihat `docs/DEPLOY.md` (Render / Hugging Face Spaces; frontend boleh di Vercel). Torch tidak dipakai di web.

## 9. Sumber data dan lisensi
Dataset publik: *Human Activity Recognition Using Smartphones* (UCI Machine Learning Repository, id 240),
lisensi **CC BY 4.0**.
- Reyes-Ortiz, J., Anguita, D., Ghio, A., Oneto, L., & Parra, X. (2013). Human Activity Recognition Using Smartphones [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C54S4K
- Anguita, D., Ghio, A., Oneto, L., Parra, X., & Reyes-Ortiz, J. L. (2013). A Public Domain Dataset for Human Activity Recognition Using Smartphones. ESANN 2013.

Kunci API di proyek ini hanya contoh. Jangan pernah commit file `.env`.

## Tentang Data Private (500 window)

- **Perekam:** anggota Kelompok 1 (Abid dan Fahmi), masing-masing merekam sendiri gerakannya.
- **Perangkat:** smartphone Android pribadi anggota kelompok (accelerometer + giroskop bawaan, 50 Hz), lewat aplikasi perekam kelompok atau phyphox. Tulis merek/tipe HP di sini sebelum dikumpulkan.
- **Cara pelabelan:** sebelum mulai merekam, perekam memilih label aktivitas di aplikasi (jalan, naik tangga, turun tangga, duduk, berdiri, berbaring). Satu sesi = satu aktivitas sekitar 2 menit, HP di saku celana. Bagian awal dan akhir sesi (sekitar 3 detik) dibuang.
- **Privasi:** data mentah private hanya untuk keperluan UAS ini dan tidak dipublikasikan di luar paket UAS.
