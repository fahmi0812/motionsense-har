# Catatan Deploy

Prinsip: **torch hanya untuk training di laptop.** Web cukup `onnxruntime` (model `web/model/cnn.onnx`, ±100 KB).

## Opsi 1 — Render (gratis)
1. Upload folder proyek ke repo GitHub pribadi Anda (tanpa `.env`, tanpa `data/`).
2. Render → New → Web Service → pilih repo.
3. Root Directory: `web` · Build: `pip install -r requirements.txt` · Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
4. Environment: `SUPABASE_URL`, `SUPABASE_SERVICE_KEY` (atau ANON + policy baca). Tanpa itu web jalan mode offline.
5. Paket gratis "tidur" bila tak dipakai; buka 1 menit sebelum presentasi.

## Opsi 2 — Hugging Face Spaces (Docker)
Buat Space tipe Docker, isi folder `web/` dan `Dockerfile` berikut, port 7860:
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir -r requirements.txt
CMD ["uvicorn","app:app","--host","0.0.0.0","--port","7860"]
```
Isi secret `SUPABASE_URL` dan `SUPABASE_SERVICE_KEY` di Settings → Variables and secrets.

## Opsi 3 — Frontend di Vercel, backend di Render/HF
- Upload isi `web/static/` saja ke Vercel (situs statis). **Jangan** taruh torch/backend Python berat di Vercel.
- Di `index.html` dan `evaluasi.html`, sebelum `<script src="app.js">`, tambahkan:
  `<script>window.API_BASE="https://nama-backend.onrender.com"</script>`
- Gambar confusion matrix diambil dari backend (`/model/...`).
- Di backend set `CORS_ORIGINS=https://nama-anda.vercel.app`.
