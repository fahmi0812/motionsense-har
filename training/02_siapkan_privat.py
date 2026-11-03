"""Langkah 2: data privat (Supabase atau CSV) -> window 128x6, 50 Hz, overlap 50%.
- acc m/s² -> g (/9.80665), gyro tetap rad/s
- target 500 window (seimbang per kelas bila cukup)
- gabung dengan publik, tiap baris diberi sumber=publik/privat, laporan komposisi 75/25
Pakai:  python 02_siapkan_privat.py --csv ../data/privat_csv
        python 02_siapkan_privat.py --supabase   (butuh .env: SUPABASE_URL, SUPABASE_ANON_KEY/SERVICE KEY)"""
import os, sys, glob, json, argparse, numpy as np, pandas as pd
from common import LABELS, LABEL_ID, CHANNELS, G, resample_50hz, make_windows

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
N_TARGET = 500

def from_csv(folder):
    files = glob.glob(os.path.join(folder, "*.csv"))
    if not files:
        sys.exit(f"Tidak ada CSV di {folder}")
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

def from_supabase():
    import httpx
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))
    url = os.environ["SUPABASE_URL"].rstrip("/")
    key = os.environ.get("SUPABASE_SERVICE_KEY") or os.environ["SUPABASE_ANON_KEY"]
    h = {"apikey": key, "Authorization": f"Bearer {key}"}
    rows, off = [], 0
    while True:   # ambil per 1000 baris (batas default Supabase)
        r = httpx.get(f"{url}/rest/v1/sensor_readings", headers=h, timeout=60, params={
            "select": "session_id,user_id,label,ts,ax,ay,az,gx,gy,gz", "order": "session_id,ts",
            "offset": off, "limit": 1000})
        r.raise_for_status()
        b = r.json()
        rows += b
        if len(b) < 1000:
            break
        off += 1000
    return pd.DataFrame(rows)

def to_windows(df):
    df = df.copy()
    dt = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
    df["t"] = (dt - pd.Timestamp("1970-01-01", tz="UTC")).dt.total_seconds()
    Xs, ys, sess = [], [], []
    for (sid, lab), g in df.groupby(["session_id", "label"]):
        g = g[["t"] + CHANNELS].astype(float)
        if g[["ax", "ay", "az"]].abs().mean().max() > 3:     # deteksi m/s² -> ubah ke g
            g[["ax", "ay", "az"]] /= G
        r = resample_50hz(g)
        W, _ = make_windows(r[CHANNELS].to_numpy())
        Xs.append(W); ys += [LABEL_ID[lab]] * len(W); sess += [sid] * len(W)
    return np.concatenate(Xs), np.array(ys), np.array(sess)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    ap.add_argument("--supabase", action="store_true")
    a = ap.parse_args()
    df = from_supabase() if a.supabase else from_csv(a.csv or os.path.join(ROOT, "privat_csv"))
    X, y, sess = to_windows(df)
    print("Window privat tersedia:", len(X))
    rng = np.random.default_rng(42)
    per = N_TARGET // len(LABELS)
    idx = []
    for k in range(len(LABELS)):
        cand = np.where(y == k)[0]
        idx += list(rng.choice(cand, min(per, len(cand)), replace=False))
    sisa = np.setdiff1d(np.arange(len(X)), idx)        # isi kekurangan sampai 500
    need = N_TARGET - len(idx)
    if need > 0 and len(sisa):
        idx += list(rng.choice(sisa, min(need, len(sisa)), replace=False))
    idx = np.array(idx)
    np.savez_compressed(os.path.join(ROOT, "privat.npz"), X=X[idx], y=y[idx], session=sess[idx])

    pub = np.load(os.path.join(ROOT, "publik.npz"))
    meta = pd.DataFrame({"sumber": ["publik"] * len(pub["y"]) + ["privat"] * len(idx),
                         "label": [LABELS[i] for i in np.concatenate([pub["y"], y[idx]])]})
    meta.to_csv(os.path.join(ROOT, "komposisi_window.csv"), index_label="window_id")
    tab = pd.crosstab(meta["label"], meta["sumber"], margins=True, margins_name="TOTAL")
    pct = meta["sumber"].value_counts(normalize=True).mul(100).round(2).to_dict()
    rep = ["# Laporan Komposisi Data", "", "```\n" + tab.to_string() + "\n```", "",
           f"Persentase: publik {pct.get('publik',0)}% | privat {pct.get('privat',0)}% (target 75/25)"]
    open(os.path.join(ROOT, "..", "training", "laporan_komposisi.md"), "w").write("\n".join(rep))
    json.dump({"publik": int(len(pub['y'])), "privat": int(len(idx)), "persen": pct},
              open(os.path.join(ROOT, "komposisi.json"), "w"), indent=2)
    print("\n".join(rep))
