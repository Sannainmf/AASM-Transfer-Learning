# train_cf_cnn_model_b.py
# 1D-CNN Model B: train on RANS Cf from scratch

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import json
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [1, 2, 3, 4, 5]


class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(12, 128), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(128, 256), nn.ReLU(), nn.Dropout(0.05),
        )
        self.expand = nn.Linear(256, 512)
        self.convs = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=5, padding=2), nn.BatchNorm1d(64), nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=5, padding=2), nn.BatchNorm1d(32), nn.ReLU(),
            nn.Conv1d(32, 1, kernel_size=5, padding=2),
        )
    def forward(self, x):
        latent = self.encoder(x)
        spatial = self.expand(latent).unsqueeze(1)
        return self.convs(spatial).squeeze(1)


class Normalizer:
    def __init__(self):
        self.in_mean = self.in_std = self.out_mean = self.out_std = None
    def fit(self, X, y):
        self.in_mean, self.in_std = X.mean(0), X.std(0)
        self.out_mean, self.out_std = y.mean(0), y.std(0)
        self.in_std[self.in_std < 1e-10] = 1.0
        self.out_std[self.out_std < 1e-10] = 1.0
    def norm_in(self, X): return (X - self.in_mean) / self.in_std
    def norm_out(self, y): return (y - self.out_mean) / self.out_std
    def denorm_out(self, y): return y * self.out_std + self.out_mean


def set_seed(s):
    torch.manual_seed(s); torch.cuda.manual_seed_all(s)
    np.random.seed(s); torch.backends.cudnn.deterministic = True

def make_loaders(X, y, bs, val_frac=0.1):
    n = len(X); nv = max(1, int(n * val_frac)); idx = np.random.permutation(n)
    tl = DataLoader(TensorDataset(torch.tensor(X[idx[nv:]], dtype=torch.float32),
         torch.tensor(y[idx[nv:]], dtype=torch.float32)), batch_size=bs, shuffle=True)
    vl = DataLoader(TensorDataset(torch.tensor(X[idx[:nv]], dtype=torch.float32),
         torch.tensor(y[idx[:nv]], dtype=torch.float32)), batch_size=bs)
    return tl, vl

def train(model, tl, vl, epochs, lr, patience, label=""):
    opt = optim.Adam(model.parameters(), lr=lr)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, patience=15, factor=0.5)
    loss_fn = nn.MSELoss()
    best_loss, best_state, wait = float('inf'), None, 0
    for ep in range(epochs):
        model.train()
        tl_sum, nb = 0, 0
        for xb, yb in tl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad(); loss = loss_fn(model(xb), yb); loss.backward(); opt.step()
            tl_sum += loss.item(); nb += 1
        model.eval()
        vl_sum, nv = 0, 0
        with torch.no_grad():
            for xb, yb in vl:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                vl_sum += loss_fn(model(xb), yb).item(); nv += 1
        vl_avg = vl_sum / nv; sched.step(vl_avg)
        if vl_avg < best_loss:
            best_loss = vl_avg; best_state = copy.deepcopy(model.state_dict()); wait = 0
        else: wait += 1
        if (ep+1) % 50 == 0 or ep == 0:
            print(f"  [{label}] Epoch {ep+1}/{epochs} | Train: {tl_sum/nb:.6f} | Val: {vl_avg:.6f}")
        if wait >= patience:
            print(f"  [{label}] Early stop at epoch {ep+1} (best val: {best_loss:.6f})"); break
    model.load_state_dict(best_state)
    return model

def r2_score(true, pred):
    return 1 - np.sum((true - pred)**2) / np.sum((true - true.mean())**2)


# ---- load data ----

rans_tr = np.load(os.path.join(PROJECT_DIR, "Data", "case3_train_v2.npz"))
rans_te = np.load(os.path.join(PROJECT_DIR, "Data", "case3_test_v2.npz"))
train_X, train_cf = rans_tr['inputs'], rans_tr['cf_distributions']
test_X, test_cf = rans_te['inputs'], rans_te['cf_distributions']
print(f"RANS train: {len(train_X)}, test: {len(test_X)}")

norm = Normalizer()
norm.fit(train_X, train_cf)
train_X_n = norm.norm_in(train_X)
train_cf_n = norm.norm_out(train_cf)

def test_model(model):
    model.eval()
    X_t = torch.tensor(norm.norm_in(test_X), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = norm.denorm_out(model(X_t).cpu().numpy())
    return np.sqrt(np.mean((pred - test_cf)**2)), r2_score(test_cf, pred)

print("\n=== Cf CNN Model B: RANS scratch ===")
rmses, r2s = [], []
for s in seeds:
    set_seed(s)
    tl, vl = make_loaders(train_X_n, train_cf_n, 32)
    m = CNN1D().to(DEVICE)
    m = train(m, tl, vl, 500, 1e-3, 30, label=f"B-s{s}")
    rmse, r2 = test_model(m)
    rmses.append(rmse); r2s.append(r2)
    print(f"  seed {s}: RMSE={rmse:.6f}, R2={r2:.4f}")

print(f"\n  RMSE = {np.mean(rmses):.6f} +- {np.std(rmses):.6f}")
print(f"  R2   = {np.mean(r2s):.4f} +- {np.std(r2s):.4f}")

with open(os.path.join(PROJECT_DIR, "Results", "cf_cnn_model_b.json"), 'w') as f:
    json.dump({'rmse': [float(x) for x in rmses], 'r2': [float(x) for x in r2s],
               'rmse_mean': float(np.mean(rmses)), 'rmse_std': float(np.std(rmses)),
               'r2_mean': float(np.mean(r2s)), 'r2_std': float(np.std(r2s))}, f, indent=2)
print("Done.")