"""Pra-proses + prediksi ONNX (TANPA torch). Logika sama dengan training/common.py."""
import os, json, numpy as np, pandas as pd, onnxruntime as ort

HERE = os.path.dirname(__file__)
NORM = json.load(open(os.path.join(HERE, "model", "norm.json")))
LABELS, CH = NORM["labels"], NORM["channels"]
FS, WIN, STEP, G = NORM["fs"], NORM["win"], NORM["step"], 9.80665
MEAN, STD = np.array(NORM["mean"], np.float32), np.array(NORM["std"], np.float32)
SESS = ort.InferenceSession(os.path.join(HERE, "model", "cnn.onnx"), providers=["CPUExecutionProvider"])

def prepare(df: pd.DataFrame) -> pd.DataFrame:
    """Terima kolom ts (ISO) atau t/time (detik) + ax..gz. Hasil: 50 Hz, acc dalam g."""
    df = df.rename(columns={c: c.strip().lower() for c in df.columns})
    if "ts" in df:
        dt = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        t = (dt - dt.min()).dt.total_seconds()
    else:
        t = df[[c for c in df.columns if c in ("t", "time", "time (s)")][0]].astype(float)
    df = df.assign(t=t.to_numpy())[["t"] + CH].astype(float).dropna().sort_values("t").drop_duplicates("t")
    if df[["ax", "ay", "az"]].abs().mean().max() > 3:      # m/s² -> g
        df[["ax", "ay", "az"]] /= G
    grid = np.arange(df["t"].iloc[0], df["t"].iloc[-1], 1 / FS)
    return pd.DataFrame({"t": grid - grid[0], **{c: np.interp(grid, df["t"], df[c]) for c in CH}})

def predict(df: pd.DataFrame) -> dict:
    r = prepare(df)
    arr = r[CH].to_numpy(np.float32)
    starts = list(range(0, len(arr) - WIN + 1, STEP))
    if not starts:
        raise ValueError(f"Data terlalu pendek: butuh minimal {WIN} sampel (~{WIN/FS:.1f} detik) setelah resample 50 Hz")
    X = (np.stack([arr[s:s + WIN] for s in starts]) - MEAN) / STD
    prob = SESS.run(None, {"input": X.astype(np.float32)})[0]
    idx = prob.argmax(1)
    windows = [{"mulai_s": round(s / FS, 2), "label": LABELS[k], "conf": round(float(p[k]), 3)}
               for s, k, p in zip(starts, idx, prob)]
    durasi = {l: 0.0 for l in LABELS}
    for w in windows:                      # tiap window "mewakili" STEP sampel baru (1,28 s)
        durasi[w["label"]] += STEP / FS
    durasi[windows[-1]["label"]] += (WIN - STEP) / FS
    step = max(1, len(r) // 3000)          # kirim maks ~3000 titik ke grafik
    sig = r.iloc[::step]
    return {"n_sampel": len(r), "durasi_total_s": round(len(r) / FS, 1), "windows": windows,
            "ringkasan_durasi_s": {k: round(v, 1) for k, v in durasi.items() if v > 0},
            "sinyal": {"t": sig["t"].round(2).tolist(), **{c: sig[c].round(4).tolist() for c in CH}}}
