"""*** DATA DUMMY / SINTETIS — BUKAN DATA ASLI ***
Membuat CSV palsu mirip ekspor phyphox/APK (acc m/s², gyro rad/s, 100 Hz) untuk menguji pipeline
tanpa merekam. Hasil akurasi dengan data ini TIDAK boleh dilaporkan sebagai hasil penelitian.
Output: data/privat_csv/dummy_<user>_<label>.csv (kolom: session_id,ts,ax,ay,az,gx,gy,gz,label,user_id)"""
import os, uuid, numpy as np, pandas as pd
from datetime import datetime, timedelta, timezone
from common import LABELS, G

OUT = os.path.join(os.path.dirname(__file__), "..", "data", "privat_csv")
FS_HP = 100      # HP merekam 100 Hz -> nanti di-resample ke 50 Hz
DUR = 120        # detik per aktivitas (~2 menit)

# arah gravitasi (dalam g) kira-kira untuk HP di pinggang, mirip UCI (sumbu x vertikal)
GRAV = {"WALKING": (1, -0.2, 0.1), "WALKING_UPSTAIRS": (0.95, -0.25, 0.2), "WALKING_DOWNSTAIRS": (1, -0.15, 0.05),
        "SITTING": (0.75, 0.05, 0.6), "STANDING": (1, -0.1, 0.05), "LAYING": (0.05, 0.6, 0.8)}
AMP = {"WALKING": 0.25, "WALKING_UPSTAIRS": 0.2, "WALKING_DOWNSTAIRS": 0.35,
       "SITTING": 0.01, "STANDING": 0.015, "LAYING": 0.01}
FREQ = {"WALKING": 1.8, "WALKING_UPSTAIRS": 1.5, "WALKING_DOWNSTAIRS": 2.1}

def gen(label, user, rng, t0):
    n = DUR * FS_HP
    t = np.arange(n) / FS_HP
    gvec = np.array(GRAV[label]) + rng.normal(0, 0.05, 3)
    f = FREQ.get(label, 0.3) * rng.uniform(0.9, 1.1)
    a = AMP[label]
    acc = np.stack([gvec[i] + a * np.sin(2 * np.pi * f * t + i) * (1.0 if i == 0 else 0.5) for i in range(3)], 1)
    acc += rng.normal(0, 0.03 + a * 0.2, acc.shape)
    gyr = np.stack([a * 2 * np.sin(2 * np.pi * f * t + i * 0.7) for i in range(3)], 1) + rng.normal(0, 0.02 + a * 0.3, (n, 3))
    acc_ms2 = acc * G     # HP memberi m/s²
    sid = str(uuid.uuid4())
    ts = [(t0 + timedelta(seconds=float(x))).isoformat() for x in t]
    df = pd.DataFrame({"session_id": sid, "ts": ts, "ax": acc_ms2[:, 0], "ay": acc_ms2[:, 1], "az": acc_ms2[:, 2],
                       "gx": gyr[:, 0], "gy": gyr[:, 1], "gz": gyr[:, 2], "label": label, "user_id": user})
    return df

if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    rng = np.random.default_rng(7)
    t0 = datetime(2026, 10, 28, 8, 0, tzinfo=timezone.utc)
    for user in ("dummy_A", "dummy_B"):
        for lab in LABELS:
            df = gen(lab, user, rng, t0)
            t0 += timedelta(minutes=5)
            df.to_csv(os.path.join(OUT, f"dummy_{user}_{lab}.csv"), index=False)
    print("CSV DUMMY dibuat di", OUT)
