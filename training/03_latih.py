"""Langkah 3: latih 1D-CNN (PyTorch CPU) + baseline Random Forest, evaluasi 2 skenario.
Skenario A "publik->privat": latih SEMUA window publik, uji di SEMUA window privat.
Skenario B "campuran"     : publik dipisah per SUBJEK (24 latih / 6 uji),
                            privat dipisah per SESI (tiap aktivitas: sebagian sesi latih, sisanya uji).
Model yang diekspor ke web = model skenario B (melihat data publik + privat).
Output: models/cnn.pt, models/norm.json, models/rf.joblib, models/metrics.json, models/*.png"""
import os, json, time, numpy as np, torch, torch.nn as nn, joblib
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from common import LABELS, CHANNELS, stat_features

torch.set_num_threads(2)            # hemat RAM/CPU laptop
torch.manual_seed(0); np.random.seed(0)
D = os.path.join(os.path.dirname(__file__), "..", "data")
M = os.path.join(os.path.dirname(__file__), "..", "models"); os.makedirs(M, exist_ok=True)
EPOCHS, BATCH, LR = 30, 64, 1e-3

class CNN1D(nn.Module):
    def __init__(self, n_ch=6, n_cls=6):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 32, 5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(),      # Global Average Pooling
            nn.Dropout(0.3), nn.Linear(64, n_cls))      # softmax dipasang saat ekspor
    def forward(self, x):          # x: (batch, 128, 6) -> diputar ke (batch, 6, 128)
        return self.net(x.transpose(1, 2))

def fit_norm(X):
    return X.reshape(-1, 6).mean(0), X.reshape(-1, 6).std(0) + 1e-8

def train_cnn(X, y, mean, std):
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32); yt = torch.tensor(y)
    model = CNN1D(); opt = torch.optim.Adam(model.parameters(), lr=LR); lossf = nn.CrossEntropyLoss()
    for ep in range(EPOCHS):
        model.train(); perm = torch.randperm(len(Xt)); tot = 0
        for i in range(0, len(Xt), BATCH):
            b = perm[i:i + BATCH]; opt.zero_grad()
            loss = lossf(model(Xt[b]), yt[b]); loss.backward(); opt.step(); tot += loss.item() * len(b)
        if ep % 5 == 4: print(f"  epoch {ep+1}/{EPOCHS} loss={tot/len(Xt):.4f}")
    return model.eval()

def predict_cnn(model, X, mean, std):
    with torch.no_grad():
        return model(torch.tensor((X - mean) / std, dtype=torch.float32)).argmax(1).numpy()

def score(y, p):
    return {"accuracy": round(float(accuracy_score(y, p)), 4),
            "f1_macro": round(float(f1_score(y, p, average="macro", labels=range(6), zero_division=0)), 4),
            "f1_per_kelas": dict(zip(LABELS, [round(float(v), 4) for v in f1_score(y, p, average=None, labels=range(6), zero_division=0)])),
            "confusion_matrix": confusion_matrix(y, p, labels=range(6)).tolist(), "n_uji": int(len(y))}

def plot_cm(cm, title, path):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    ConfusionMatrixDisplay(np.array(cm), display_labels=[l[:10] for l in LABELS]).plot(ax=ax, xticks_rotation=45, colorbar=False)
    ax.set_title(title); fig.tight_layout(); fig.savefig(path, dpi=90); plt.close(fig)

def run(name, Xtr, ytr, Xte, yte, save=False):
    print(f"== Skenario {name}: latih {len(Xtr)} / uji {len(Xte)}")
    mean, std = fit_norm(Xtr)
    t = time.time(); cnn = train_cnn(Xtr, ytr, mean, std); tc = time.time() - t
    rf = RandomForestClassifier(n_estimators=150, max_depth=15, random_state=0, n_jobs=2)
    rf.fit(stat_features(Xtr), ytr)
    res = {"cnn": score(yte, predict_cnn(cnn, Xte, mean, std)),
           "rf": score(yte, rf.predict(stat_features(Xte))), "n_latih": int(len(Xtr)), "detik_latih_cnn": round(tc, 1)}
    for k in ("cnn", "rf"):
        plot_cm(res[k]["confusion_matrix"], f"{k.upper()} - {name} (acc {res[k]['accuracy']:.3f})", os.path.join(M, f"cm_{name}_{k}.png"))
        print(f"  {k.upper()}: acc={res[k]['accuracy']} f1_macro={res[k]['f1_macro']}")
    if save:
        torch.save(cnn.state_dict(), os.path.join(M, "cnn.pt"))
        json.dump({"channels": CHANNELS, "mean": mean.tolist(), "std": std.tolist(), "labels": LABELS,
                   "fs": 50, "win": 128, "step": 64}, open(os.path.join(M, "norm.json"), "w"), indent=2)
        joblib.dump(rf, os.path.join(M, "rf.joblib"), compress=3)
    return res

if __name__ == "__main__":
    pub, prv = np.load(os.path.join(D, "publik.npz")), np.load(os.path.join(D, "privat.npz"))
    komp = json.load(open(os.path.join(D, "komposisi.json")))
    csvdir = os.path.join(D, "privat_csv")
    dummy = os.path.isdir(csvdir) and any(f.startswith("dummy_") for f in os.listdir(csvdir))
    out = {"catatan": "PERHATIAN: data privat = DUMMY/SINTETIS (make_dummy_private.py), angka uji privat BUKAN hasil asli"
           if dummy else "Data privat: rekaman asli tim", "komposisi": komp}
    out["A_publik_ke_privat"] = run("A_publik_ke_privat", pub["X"], pub["y"], prv["X"], prv["y"])
    # --- skenario B: pisah per subjek & per sesi (cegah kebocoran) ---
    subj = np.unique(pub["subject"]); rng = np.random.default_rng(1)
    test_subj = rng.choice(subj, 6, replace=False)
    pm = np.isin(pub["subject"], test_subj)
    test_sess = []
    for k in range(6):
        ss = sorted(set(prv["session"][prv["y"] == k]))
        test_sess += ss[len(ss) // 2:] if len(ss) > 1 else []   # sesi kedua dst -> uji
    qm = np.isin(prv["session"], test_sess)
    Xtr = np.concatenate([pub["X"][~pm], prv["X"][~qm]]); ytr = np.concatenate([pub["y"][~pm], prv["y"][~qm]])
    Xte = np.concatenate([pub["X"][pm], prv["X"][qm]]); yte = np.concatenate([pub["y"][pm], prv["y"][qm]])
    out["B_campuran"] = run("B_campuran", Xtr, ytr, Xte, yte, save=True)
    out["B_campuran"]["subjek_uji_uci"] = sorted(int(s) for s in test_subj)
    out["B_campuran"]["jumlah_sesi_privat_uji"] = len(test_sess)
    json.dump(out, open(os.path.join(M, "metrics.json"), "w"), indent=2)
    print("Selesai -> models/metrics.json")
