"""Langkah 4: ekspor CNN ke ONNX (+softmax) dan cek kesamaan hasil dengan onnxruntime.
Input ONNX: 'input' (batch,128,6) SUDAH dinormalisasi; output 'prob' (batch,6).
Hasil disalin ke web/model/ (cnn.onnx, norm.json, metrics.json)."""
import os, json, shutil, numpy as np, torch, onnxruntime as ort
from importlib import import_module
CNN1D = import_module("03_latih").CNN1D

M = os.path.join(os.path.dirname(__file__), "..", "models")
W = os.path.join(os.path.dirname(__file__), "..", "web", "model"); os.makedirs(W, exist_ok=True)

class WithSoftmax(torch.nn.Module):
    def __init__(self, m): super().__init__(); self.m = m
    def forward(self, x): return torch.softmax(self.m(x), dim=1)

if __name__ == "__main__":
    net = CNN1D(); net.load_state_dict(torch.load(os.path.join(M, "cnn.pt"))); model = WithSoftmax(net).eval()
    dummy = torch.randn(1, 128, 6)
    path = os.path.join(M, "cnn.onnx")
    torch.onnx.export(model, dummy, path, input_names=["input"], output_names=["prob"],
                      dynamic_axes={"input": {0: "batch"}, "prob": {0: "batch"}}, opset_version=17, dynamo=False)
    # uji kesamaan pada data privat sungguhan (ternormalisasi)
    norm = json.load(open(os.path.join(M, "norm.json")))
    X = np.load(os.path.join(os.path.dirname(__file__), "..", "data", "privat.npz"))["X"][:64]
    X = ((X - np.array(norm["mean"])) / np.array(norm["std"])).astype(np.float32)
    with torch.no_grad(): pt = model(torch.from_numpy(X)).numpy()
    ox = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).run(None, {"input": X})[0]
    diff = float(np.abs(pt - ox).max()); same = float((pt.argmax(1) == ox.argmax(1)).mean())
    print(f"Paritas PyTorch vs ONNX: selisih maks={diff:.2e}, kelas sama={same*100:.1f}%")
    assert diff < 1e-4 and same == 1.0, "ONNX tidak sama dengan PyTorch!"
    for f in ("cnn.onnx", "norm.json", "metrics.json"):
        shutil.copy(os.path.join(M, f), W)
    json.dump({"max_abs_diff": diff, "argmax_match": same}, open(os.path.join(M, "onnx_parity.json"), "w"))
    print("Disalin ke web/model/  ukuran onnx:", os.path.getsize(path), "byte")
