"""MotionSense - backend FastAPI. Jalankan: python -m uvicorn app:app --port 8000 (dari folder web/)
Mode ONLINE  : isi SUPABASE_URL + SUPABASE_ANON_KEY (atau SUPABASE_SERVICE_KEY, khusus server) di .env
Mode OFFLINE : tanpa .env -> daftar sesi diambil dari web/sample_data/*.csv"""
import os, io, glob, json, pandas as pd, httpx
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from inference import predict

HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(HERE, "..", ".env")); load_dotenv(os.path.join(HERE, ".env"))
URL = os.getenv("SUPABASE_URL", "").rstrip("/")
KEY = os.getenv("SUPABASE_SERVICE_KEY") or os.getenv("SUPABASE_ANON_KEY", "")
ONLINE = bool(URL and KEY and "xxxx" not in URL)
H = {"apikey": KEY, "Authorization": f"Bearer {KEY}"}
SAMPLE = os.path.join(HERE, "sample_data")

app = FastAPI(title="MotionSense")
# izinkan frontend di domain lain (mis. Vercel). Ganti "*" dengan domain Anda saat produksi.
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","), allow_methods=["*"], allow_headers=["*"])

@app.get("/api/health")
def health():
    return {"status": "ok", "mode": "online (Supabase)" if ONLINE else "offline (CSV contoh)"}

@app.get("/api/sessions")
def sessions():
    if not ONLINE:
        return [{"session_id": os.path.basename(f)[:-4], "user_id": "contoh", "label": "-", "sumber": "csv"}
                for f in sorted(glob.glob(os.path.join(SAMPLE, "*.csv")))]
    # ambil kolom ringan saja lalu ringkas per sesi
    r = httpx.get(f"{URL}/rest/v1/sensor_readings", headers=H, timeout=30,
                  params={"select": "session_id,user_id,label,ts", "order": "ts.desc", "limit": 20000})
    if r.status_code != 200:
        raise HTTPException(502, f"Supabase error {r.status_code}: {r.text[:200]}")
    df = pd.DataFrame(r.json())
    if df.empty:
        return []
    g = df.groupby("session_id").agg(user_id=("user_id", "first"), label=("label", "first"),
                                     mulai=("ts", "min"), n=("ts", "size")).reset_index()
    return g.sort_values("mulai", ascending=False).to_dict("records")

def load_session(sid: str) -> pd.DataFrame:
    if not ONLINE:
        p = os.path.join(SAMPLE, os.path.basename(sid) + ".csv")
        if not os.path.exists(p):
            raise HTTPException(404, "Sesi tidak ditemukan")
        return pd.read_csv(p)
    rows, off = [], 0
    while True:   # Supabase membatasi 1000 baris per permintaan
        r = httpx.get(f"{URL}/rest/v1/sensor_readings", headers=H, timeout=60, params={
            "select": "ts,ax,ay,az,gx,gy,gz", "session_id": f"eq.{sid}", "order": "ts", "offset": off, "limit": 1000})
        if r.status_code != 200:
            raise HTTPException(502, f"Supabase error {r.status_code}: {r.text[:200]}")
        rows += r.json()
        if len(r.json()) < 1000: break
        off += 1000
    if not rows:
        raise HTTPException(404, "Sesi kosong")
    return pd.DataFrame(rows)

@app.get("/api/sessions/{sid}/predict")
def predict_session(sid: str):
    try:
        return predict(load_session(sid))
    except ValueError as e:
        raise HTTPException(400, str(e))

@app.post("/api/predict_csv")
async def predict_csv(file: UploadFile = File(...)):
    raw = await file.read()
    if len(raw) > 20_000_000:
        raise HTTPException(413, "File terlalu besar (maks 20 MB)")
    try:
        return predict(pd.read_csv(io.BytesIO(raw)))
    except (ValueError, KeyError, IndexError) as e:
        raise HTTPException(400, f"Format CSV tidak dikenali: {e}. Wajib kolom ts atau time, plus ax,ay,az,gx,gy,gz")

@app.get("/api/metrics")
def metrics():
    return json.load(open(os.path.join(HERE, "model", "metrics.json")))

app.mount("/model", StaticFiles(directory=os.path.join(HERE, "model")), name="model")
app.mount("/", StaticFiles(directory=os.path.join(HERE, "static"), html=True), name="static")
