# train_cp_budget.py
# Data budget study for Cp: transfer vs scratch at different RANS fractions

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
import copy
import json
import os

PROJECT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # repo root
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seeds = [1, 2, 3, 4, 5]
fractions = [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(12, 256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256, 256), nn.ReLU(), nn.Dropout(0.05),
            nn.Linear(256, 512),
        )
    def forward(self, x):
        return self.network(x)


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

xfoil = np.load(os.path.join(PROJECT_DIR, "Data", "xfoil_dataset_v2.npz"))
xfoil_X = xfoil['inputs']
xfoil_cp = xfoil['cp_distributions']
xfoil_cf = xfoil['cf_distributions']

bad = np.any((xfoil_cf > 0.1) | (xfoil_cf < -0.05), axis=1)
xfoil_X = xfoil_X[~bad]
xfoil_cp = xfoil_cp[~bad]

rans_tr = np.load(os.path.join(PROJECT_DIR, "Data", "case3_train_v2.npz"))
rans_te = np.load(os.path.join(PROJECT_DIR, "Data", "case3_test_v2.npz"))
train_X, train_cp = rans_tr['inputs'], rans_tr['cp_distributions']
test_X, test_cp = rans_te['inputs'], rans_te['cp_distributions']
print(f"XFOIL: {len(xfoil_X)}, RANS train: {len(train_X)}, test: {len(test_X)}")

shared_norm = Normalizer()
shared_norm.fit(np.vstack([xfoil_X, train_X]), np.vstack([xfoil_cp, train_cp]))
xfoil_X_n = shared_norm.norm_in(xfoil_X)
xfoil_cp_n = shared_norm.norm_out(xfoil_cp)


def test_model(model, norm):
    model.eval()
    X_t = torch.tensor(norm.norm_in(test_X), dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        pred = norm.denorm_out(model(X_t).cpu().numpy())
    return np.sqrt(np.mean((pred - test_cp)**2)), r2_score(test_cp, pred)


print("\n=== Cp Data Budget Study ===")
budget = {}

for frac in fractions:
    n = max(10, int(len(train_X) * frac))
    print(f"\n--- {frac:.0%} ({n} samples) ---")
    tf_rmses, tf_r2s, sc_rmses, sc_r2s = [], [], [], []

    for s in seeds:
        set_seed(s)
        idx = np.random.RandomState(s).permutation(len(train_X))[:n]
        sub_X, sub_cp = train_X[idx], train_cp[idx]

        # transfer
        tl, vl = make_loaders(xfoil_X_n, xfoil_cp_n, 128)
        m = MLP().to(DEVICE)
        m = train(m, tl, vl, 300, 1e-3, 30, label=f"TF-{frac:.0%}-pre-s{s}")
        tl, vl = make_loaders(shared_norm.norm_in(sub_X), shared_norm.norm_out(sub_cp), 32)
        m = train(m, tl, vl, 500, 5e-4, 30, label=f"TF-{frac:.0%}-ft-s{s}")
        rmse, r2 = test_model(m, shared_norm)
        tf_rmses.append(rmse); tf_r2s.append(r2)

        # scratch
        sc_norm = Normalizer()
        sc_norm.fit(sub_X, sub_cp)
        tl, vl = make_loaders(sc_norm.norm_in(sub_X), sc_norm.norm_out(sub_cp), 32)
        m = MLP().to(DEVICE)
        m = train(m, tl, vl, 500, 1e-3, 30, label=f"SC-{frac:.0%}-s{s}")
        rmse, r2 = test_model(m, sc_norm)
        sc_rmses.append(rmse); sc_r2s.append(r2)

    t_stat, p_val = stats.ttest_rel(sc_rmses, tf_rmses)
    diffs = [a - b for a, b in zip(sc_rmses, tf_rmses)]
    d = np.mean(diffs) / np.std(diffs) if np.std(diffs) > 0 else 0
    improv = (np.mean(sc_rmses) - np.mean(tf_rmses)) / np.mean(sc_rmses) * 100

    budget[f"{frac:.0%}"] = {
        'n': n, 'tf_mean': float(np.mean(tf_rmses)), 'tf_std': float(np.std(tf_rmses)),
        'tf_r2': float(np.mean(tf_r2s)), 'sc_mean': float(np.mean(sc_rmses)),
        'sc_std': float(np.std(sc_rmses)), 'sc_r2': float(np.mean(sc_r2s)),
        'improvement': float(improv), 'p_value': float(p_val), 'cohens_d': float(d),
        'significant': bool(p_val < 0.05),
        'tf_seeds': [float(x) for x in tf_rmses], 'sc_seeds': [float(x) for x in sc_rmses],
    }
    print(f"  transfer: {np.mean(tf_rmses):.6f}+-{np.std(tf_rmses):.6f} (R2={np.mean(tf_r2s):.4f})")
    print(f"  scratch:  {np.mean(sc_rmses):.6f}+-{np.std(sc_rmses):.6f} (R2={np.mean(sc_r2s):.4f})")
    print(f"  improvement: {improv:.1f}%, p={p_val:.4f}, d={d:.2f}")

with open(os.path.join(PROJECT_DIR, "Results", "cp_mlp_budget.json"), 'w') as f:
    json.dump(budget, f, indent=2)
print("Done.")