# Prompt untuk Melanjutkan Pengembangan (salin ke asisten AI)

Saya mengerjakan proyek "MotionSense: Sistem Web Pengenalan Aktivitas Manusia Menggunakan CNN" (UAS ML untuk IoT, tim 2 orang, laptop RAM kecil).
Struktur: `training/` (PyTorch CPU, 1D-CNN Conv32k5-BN-ReLU → Conv64k5-ReLU → MaxPool2 → Conv64k3-ReLU → GAP → Dropout0.3 → Dense6),
`web/` (FastAPI + onnxruntime, tanpa torch; HTML + Chart.js), data di Supabase tabel `sensor_readings`
(session_id, user_id, label, ts, ax..gz, sumber, UNIQUE(session_id, ts)).
Input model: window 128×6 (acc dalam g, gyro rad/s, 50 Hz, overlap 50%), dinormalisasi per kanal dengan `web/model/norm.json`.
Data: 1500 window UCI HAR + 500 window privat (75/25). Evaluasi: pisah per subjek/sesi, CNN vs Random Forest.

Aturan: semua komentar & dokumen Bahasa Indonesia; tetap ringan (tanpa GPU); jangan menaruh kunci asli di kode;
jangan mengarang angka metrik — hanya dari `models/metrics.json` hasil run.

Tugas berikutnya: [tulis di sini, mis. "tambahkan filter smoothing mayoritas 3 window di timeline" /
"tambahkan halaman rekam langsung via WebSocket" / "tambah augmentasi rotasi untuk Skenario A"].
