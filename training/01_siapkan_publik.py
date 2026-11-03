"""Langkah 1: ambil UCI HAR (id 240) 'Inertial Signals' mentah -> 1500 window seimbang.
Kanal: total_acc xyz (g) + body_gyro xyz (rad/s), 128x6 per window (50 Hz).
Output: data/publik.npz (X, y, subject)."""
import os, sys, zipfile, urllib.request, numpy as np
from common import LABELS

ROOT = os.path.join(os.path.dirname(__file__), "..", "data")
URL = "https://archive.ics.uci.edu/static/public/240/human+activity+recognition+using+smartphones.zip"
N_TARGET = 1500
SEED = 42

def ensure_dataset():
    base = os.path.join(ROOT, "UCI HAR Dataset")
    if os.path.isdir(base):
        return base
    os.makedirs(ROOT, exist_ok=True)
    z1 = os.path.join(ROOT, "har.zip")
    if not os.path.exists(z1):
        print("Mengunduh UCI HAR (~60 MB)...")
        urllib.request.urlretrieve(URL, z1)
    zipfile.ZipFile(z1).extractall(ROOT)
    inner = os.path.join(ROOT, "UCI HAR Dataset.zip")
    if os.path.exists(inner):
        zipfile.ZipFile(inner).extractall(ROOT)
    return base

def load_split(base, split):
    d = os.path.join(base, split, "Inertial Signals")
    names = [f"total_acc_{a}" for a in "xyz"] + [f"body_gyro_{a}" for a in "xyz"]
    X = np.stack([np.loadtxt(os.path.join(d, f"{n}_{split}.txt"), dtype=np.float32) for n in names], axis=-1)
    y = np.loadtxt(os.path.join(base, split, f"y_{split}.txt"), dtype=int) - 1   # 0..5
    s = np.loadtxt(os.path.join(base, split, f"subject_{split}.txt"), dtype=int)
    return X, y, s

if __name__ == "__main__":
    base = ensure_dataset()
    Xs, ys, ss = zip(*(load_split(base, sp) for sp in ("train", "test")))
    X, y, s = np.concatenate(Xs), np.concatenate(ys), np.concatenate(ss)
    print("Total window UCI:", X.shape)
    rng = np.random.default_rng(SEED)
    per = N_TARGET // len(LABELS)
    idx = np.concatenate([rng.choice(np.where(y == k)[0], per, replace=False) for k in range(len(LABELS))])
    rng.shuffle(idx)
    os.makedirs(ROOT, exist_ok=True)
    np.savez_compressed(os.path.join(ROOT, "publik.npz"), X=X[idx], y=y[idx], subject=s[idx])
    print(f"Disimpan {len(idx)} window publik ({per}/kelas), subjek unik: {len(set(s[idx]))}")
