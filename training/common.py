"""Fungsi bersama: label, windowing, resample, fitur statistik."""
import numpy as np, pandas as pd

LABELS = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
LABEL_ID = {l: i for i, l in enumerate(LABELS)}
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]   # acc dalam g, gyro dalam rad/s
FS = 50          # Hz
WIN = 128        # sampel per window (2,56 detik)
STEP = 64        # 50% overlap
G = 9.80665

def resample_50hz(df: pd.DataFrame) -> pd.DataFrame:
    """df punya kolom t (detik) + 6 kanal. Interpolasi linear ke grid 50 Hz."""
    df = df.sort_values("t").drop_duplicates("t")
    t = df["t"].to_numpy(float)
    if len(t) < 2:
        return df.iloc[:0]
    grid = np.arange(t[0], t[-1], 1.0 / FS)
    out = {"t": grid}
    for c in CHANNELS:
        out[c] = np.interp(grid, t, df[c].to_numpy(float))
    return pd.DataFrame(out)

def make_windows(arr: np.ndarray, win=WIN, step=STEP):
    """arr (N,6) -> (n_win, 128, 6) dan indeks awal tiap window."""
    starts = list(range(0, len(arr) - win + 1, step))
    if not starts:
        return np.zeros((0, win, arr.shape[1]), np.float32), []
    return np.stack([arr[s:s + win] for s in starts]).astype(np.float32), starts

def stat_features(X: np.ndarray) -> np.ndarray:
    """X (n,128,6) -> (n,24): mean, std, min, max per sumbu (untuk Random Forest)."""
    return np.concatenate([X.mean(1), X.std(1), X.min(1), X.max(1)], axis=1)
