# ====== transformer.py (single-seed=23, add checkpointing & resume) ======
import re, math, numpy as np, pandas as pd
import numpy as np
from pathlib import Path
import warnings
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.feature_selection import mutual_info_regression
from sklearn.cross_decomposition import PLSRegression
from scipy.signal import savgol_filter
import math, time, numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

# ========================
# 新增：是否继续训练 & 权重路径（无需命令行）
# ========================
CONTINUE_TRAINING = True           # False=从头训练；True=基于已保存权重继续训练（refit阶段）
WEIGHTS_PATH      = "model23_weights.pt"  # 仅单seed=23，使用固定文件名

#Config
CSV_PATH   = "input.csv"
TARGET_COL = "Nmass_O"
WAVELEN_MIN = 400.0
WAVELEN_MAX = 2400.0

#Remove strong water absorption windows (nanometres)
WATER_BANDS = [(1350, 1460), (1790, 1960), (2470, 2560)]

#Savitzky–Golay (for 1st derivative)
#Must be odd; try 11, 15, 21
SG_WINDOW   = 15
SG_POLY     = 2
SG_DERIV    = 1

#Selection limits
K_MIN = 8
K_MAX = 500

#For inner CV when choosing k / components
INNER_KFOLDS = 5
MIN_SPACING_NM = 12.0
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE  = 0.15

CFG = dict(
    d_model=64,
    nhead=4,
    #Encoder depth
    num_layers=6,          
    dim_ff=256,
    dropout=0.10,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    max_epochs=10000,
    #Early stopping on val
    patience=10000,           
    grad_clip=1.0,
    huber_beta=0.5,
    warmup_epochs=10,
    cosine_min_lr=1e-6,
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ====== 新增：保存/加载权重的辅助函数（不改动训练/数据逻辑） ======
# --- replace this whole function ---
def save_model_weights(path: str | Path, model: nn.Module, extra: dict | None = None):
    """
    Save only tensors + basic python types to keep it loadable with weights_only=True.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    def _clean(o):
        import numpy as _np
        import torch as _torch
        if isinstance(o, _np.ndarray):
            # turn into list to avoid numpy pickling
            return o.tolist()
        if isinstance(o, _torch.Tensor):
            return o  # tensors are fine
        if isinstance(o, (float, int, str, bool)) or o is None:
            return o
        if isinstance(o, (list, tuple)):
            return type(o)(_clean(x) for x in o)
        if isinstance(o, dict):
            return {k: _clean(v) for k, v in o.items()}
        # fallback: stringify to be safe
        return str(o)

    pkg = {"model_state_dict": model.state_dict()}
    if extra:
        pkg["meta"] = _clean(extra)

    torch.save(pkg, str(path))
    print(f"[Checkpoint] Saved weights to: {path}")

# --- replace this whole function ---
def try_load_model_weights(path: str | Path, model: nn.Module) -> bool:
    """
    Try safe loading first (PyTorch 2.6: weights_only=True).
    If it fails, allowlist numpy reconstruct and retry.
    Finally, fall back to weights_only=False (ONLY IF YOU TRUST THE FILE).
    """
    path = Path(path)
    if not path.exists():
        print(f"[Checkpoint] Not found, skip load: {path}")
        return False

    # 1) Safe attempt
    try:
        pkg = torch.load(str(path), map_location="cpu", weights_only=True)
        state = pkg.get("model_state_dict", pkg)
        model.load_state_dict(state, strict=False)
        print(f"[Checkpoint] Loaded (safe) from: {path}")
        return True
    except Exception as e_safe:
        print(f"[Checkpoint] Safe load failed: {e_safe}")

    # 2) Safe attempt with allowlisted numpy reconstruct (still weights_only=True)
    try:
        import numpy as _np
        import torch.serialization as ts
        # allowlist the numpy reconstruct used by older numpy pickles
        ts.add_safe_globals([_np.core.multiarray._reconstruct])
        pkg = torch.load(str(path), map_location="cpu", weights_only=True)
        state = pkg.get("model_state_dict", pkg)
        model.load_state_dict(state, strict=False)
        print(f"[Checkpoint] Loaded (safe+allowlist) from: {path}")
        return True
    except Exception as e_allow:
        print(f"[Checkpoint] Safe+allowlist load failed: {e_allow}")

    # 3) LAST RESORT: weights_only=False  (⚠️ 仅在你信任该文件来源时使用)
    try:
        pkg = torch.load(str(path), map_location="cpu", weights_only=False)
        state = pkg.get("model_state_dict", pkg)
        model.load_state_dict(state, strict=False)
        print(f"[Checkpoint] Loaded (weights_only=False) from: {path}")
        return True
    except Exception as e_unsafe:
        print(f"[Checkpoint] Load failed ({path}): {e_unsafe}")
        return False




#Utils
def nm_from_col(name: str):
    m = re.findall(r"(\d+(?:\.\d+)?)", str(name))
    return float(m[-1]) if m else np.nan

def snv_transform(X):
    mu = X.mean(axis=1, keepdims=True)
    sd = X.std(axis=1, keepdims=True) + 1e-12
    return (X - mu) / sd

def remove_water_bands(wls_nm, X):
    mask = np.ones_like(wls_nm, dtype=bool)
    for lo, hi in WATER_BANDS:
        mask &= ~((wls_nm >= lo) & (wls_nm <= hi))
    return wls_nm[mask], X[:, mask], mask

def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

#Preprocess: band windowing (400–2400), water windows, SG deriv + SNV
def spectral_preprocess(X_raw, wls_nm_raw):
    mask = (wls_nm_raw >= WAVELEN_MIN) & (wls_nm_raw <= WAVELEN_MAX) & np.isfinite(wls_nm_raw)
    wls = wls_nm_raw[mask]
    X = X_raw[:, mask]
    window = SG_WINDOW if SG_WINDOW < X.shape[1] else (X.shape[1] - (1 - X.shape[1] % 2))
    window = max(5, window if window % 2 == 1 else window - 1)
    X_sg = savgol_filter(X, window_length=window, polyorder=SG_POLY, deriv=SG_DERIV, axis=1)
    wls2, X2, mask2 = remove_water_bands(wls, X_sg)
    X2 = snv_transform(X2)
    return X2.astype(np.float32), wls2

#sSPA (supervised) with train-only fit + min spacing
def corr_abs(a, b):
    a = a - a.mean(); b = b - b.mean()
    sa, sb = a.std() + 1e-12, b.std() + 1e-12
    return abs(np.dot(a/sa, b/sb) / len(a))

def supervised_spa_order_trainonly(Xtr, ytr, wls_nm, max_k=40, first="maxcorr", min_spacing_nm=None, verbose=False):
    N, P = Xtr.shape
    if first == "maxcorr":
        cors = [corr_abs(Xtr[:, j], ytr) for j in range(P)]
        j0 = int(np.argmax(cors))
    elif first == "maxnorm":
        j0 = int(np.argmax(np.linalg.norm(Xtr, axis=0)))
    else:
        rng = np.random.default_rng(RANDOM_STATE); j0 = int(rng.integers(0, P))
    chosen = [j0]
    if verbose:
        print(f"[sSPA] m=1 → idx={j0} (~{wls_nm[j0]:.0f} nm)")
    def too_close(j):
        if min_spacing_nm is None or min_spacing_nm <= 0: return False
        for c in chosen:
            if abs(wls_nm[j] - wls_nm[c]) < min_spacing_nm:
                return True
        return False
    for m in range(2, max_k+1):
        Xk = Xtr[:, chosen]
        best_j, best_score, best_norm = None, -1.0, None
        for j in range(P):
            if j in chosen: continue
            if too_close(j): continue
            xj = Xtr[:, j]
            if Xk.ndim == 1 or Xk.shape[1] == 0:
                rj = xj
            else:
                beta, *_ = np.linalg.lstsq(Xk, xj, rcond=None)
                rj = xj - Xk @ beta
            rn = np.linalg.norm(rj)
            sc = 0.0 if rn < 1e-12 else rn * corr_abs(rj, ytr)
            if sc > best_score:
                best_score, best_j, best_norm = sc, j, rn
        if best_j is None:
            if verbose: print(f"[sSPA] stopped early at m={m-1} (spacing/exhausted)")
            break
        chosen.append(best_j)
        if verbose and (m <= 5 or m % 5 == 0):
            print(f"[sSPA] m={m} → idx={best_j} (~{wls_nm[best_j]:.0f} nm)  score={best_score:.4g}  ||res||={best_norm:.4g}")
    return chosen

def choose_k_via_inner_cv(Xtr, ytr, order, k_min=8, k_max=40, inner_folds=5, base_model="ridge"):
    k_max = min(k_max, len(order))
    ks = list(range(k_min, k_max+1))
    kf = KFold(n_splits=inner_folds, shuffle=True, random_state=RANDOM_STATE)
    mean_rmse = []
    for k in ks:
        cols = order[:k]
        fold_err = []
        for tr, va in kf.split(Xtr):
            X_tr, X_va = Xtr[tr][:, cols], Xtr[va][:, cols]
            y_tr, y_va = ytr[tr], ytr[va]
            sc = StandardScaler().fit(X_tr)
            X_tr_s = sc.transform(X_tr)
            X_va_s = sc.transform(X_va)
            if base_model == "linreg":
                mdl = LinearRegression()
            else:
                mdl = RidgeCV(alphas=np.logspace(-4, 3, 20))
            mdl.fit(X_tr_s, y_tr)
            y_hat = mdl.predict(X_va_s)
            fold_err.append(rmse(y_va, y_hat))
        mean_rmse.append(np.mean(fold_err))
    k_best = ks[int(np.argmin(mean_rmse))]
    return k_best, mean_rmse

#Alternative selectors (MI, LassoCV)
def mi_rank(Xtr, ytr):
    mi = mutual_info_regression(Xtr, ytr, random_state=RANDOM_STATE)
    return list(np.argsort(mi)[::-1])  # descending

def lasso_select(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    lcv = LassoCV(alphas=None, cv=INNER_KFOLDS, random_state=RANDOM_STATE, max_iter=20000).fit(Xtr_s, ytr)
    coef = np.abs(lcv.coef_)
    order = list(np.argsort(coef)[::-1])
    sel = [j for j in order if coef[j] > 1e-9]
    return sel, lcv.alpha_

#Baselines on selected bands (PLSR, Ridge)
def run_plsr_trainval_test(X_tr, y_tr, X_va, y_va, X_te, y_te, ncomp_min=4, ncomp_max=24):
    kf = KFold(n_splits=INNER_KFOLDS, shuffle=True, random_state=RANDOM_STATE)
    comps = list(range(ncomp_min, ncomp_max+1))
    mean_rmse = []
    for c in comps:
        fold_err = []
        for tr, va in kf.split(X_tr):
            sc = StandardScaler().fit(X_tr[tr])
            Xtr_s = sc.transform(X_tr[tr]); Xva_s = sc.transform(X_tr[va])
            pls = PLSRegression(n_components=c)
            pls.fit(Xtr_s, y_tr[tr])
            y_hat = pls.predict(Xva_s).ravel()
            fold_err.append(rmse(y_tr[va], y_hat))
        mean_rmse.append(np.mean(fold_err))
    c_best = comps[int(np.argmin(mean_rmse))]
    X_tv = np.vstack([X_tr, X_va]); y_tv = np.concatenate([y_tr, y_va])
    sc2 = StandardScaler().fit(X_tv)
    pls2 = PLSRegression(n_components=c_best).fit(sc2.transform(X_tv), y_tv)
    y_pred = pls2.predict(sc2.transform(X_te)).ravel()
    return {"n_components": c_best,
            "rmse": rmse(y_te, y_pred),
            "r2": r2_score(y_te, y_pred),
            "pred": y_pred}

def run_ridge_trainval_test(X_tr, y_tr, X_va, y_va, X_te, y_te):
    alphas = np.logspace(-4, 3, 60)
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr); Xva_s = sc.transform(X_va)
    ridge = RidgeCV(alphas=alphas, cv=INNER_KFOLDS).fit(np.vstack([Xtr_s, Xva_s]),
                                                        np.concatenate([y_tr, y_va]))
    sc_all = StandardScaler().fit(np.vstack([X_tr, X_va]))
    y_pred = ridge.predict(sc_all.transform(X_te))
    return {"alpha": float(ridge.alpha_),
            "rmse": rmse(y_te, y_pred),
            "r2": r2_score(y_te, y_pred),
            "pred": y_pred}

# Load data, preprocess, split
df = pd.read_csv(CSV_PATH)
all_band_cols = [c for c in df.columns if "wave" in c.lower()]
wls_nm_all = np.array([nm_from_col(c) for c in all_band_cols], dtype=float)
df = df[[TARGET_COL] + all_band_cols].dropna().reset_index(drop=True)
X_raw = df[all_band_cols].to_numpy(dtype=float)
y = df[TARGET_COL].to_numpy(dtype=float)

#Preprocess spectra (SG deriv + water removal + SNV)
X_pp, wls_pp = spectral_preprocess(X_raw, wls_nm_all)

#Consistent split: 70/15/15 (train/val/test)
X_tv, X_te, y_tv, y_te = train_test_split(X_pp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
val_ratio_in_tv = VAL_SIZE / (1.0 - TEST_SIZE)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=val_ratio_in_tv, random_state=RANDOM_STATE, shuffle=True)
print(f"Split → train={len(y_tr)} | valid={len(y_va)} | test={len(y_te)} | bands={X_pp.shape[1]}")

#Train-only sSPA with spacing -> choose k by inner CV -> lock bands
order_sspa = supervised_spa_order_trainonly(
    X_tr, y_tr, wls_pp,
    max_k=K_MAX, first="maxcorr", min_spacing_nm=MIN_SPACING_NM, verbose=True
)
k_best, curve = choose_k_via_inner_cv(
    X_tr, y_tr, order_sspa,
    k_min=K_MIN, k_max=K_MAX, inner_folds=INNER_KFOLDS, base_model="ridge"
)
sel_cols = order_sspa[:k_best]
sel_wls  = [float(wls_pp[j]) for j in sel_cols]
print(f"\n[sSPA] Inner-CV best k = {k_best}")
print("Selected wavelengths (nm):", [int(round(w)) for w in sel_wls])

# Slice datasets to selected bands
Xtr_sel = X_tr[:, sel_cols]; Xva_sel = X_va[:, sel_cols]; Xte_sel = X_te[:, sel_cols]

#Baselines on selected bands
res_plsr  = run_plsr_trainval_test(Xtr_sel, y_tr, Xva_sel, y_va, Xte_sel, y_te, ncomp_min=4, ncomp_max=min(24, k_best))
res_ridge = run_ridge_trainval_test(Xtr_sel, y_tr, Xva_sel, y_va, Xte_sel, y_te)

print("\n=== Baselines on train-only selected bands ===")
print(f"PLSR  → comps={res_plsr['n_components']:>2} | Test RMSE={res_plsr['rmse']:.3f} | Test R²={res_plsr['r2']:.3f}")
print(f"Ridge → alpha={res_ridge['alpha']:.4g} | Test RMSE={res_ridge['rmse']:.3f} | Test R²={res_ridge['r2']:.3f}")

#Alternative selectors for comparison
mi_order = mi_rank(Xtr_sel if Xtr_sel.shape[1] > 0 else X_tr, y_tr)
if len(mi_order) > 0:
    base_X = Xtr_sel if Xtr_sel.shape[1] > 0 else X_tr
    k_best_mi, _ = choose_k_via_inner_cv(base_X, y_tr, mi_order, k_min=min(K_MIN, len(mi_order)), k_max=min(K_MAX, len(mi_order)))
    cols_mi = mi_order[:k_best_mi]
    if base_X is X_tr:
        wls_mi = [int(round(wls_pp[j])) for j in cols_mi]
        print(f"\n[MI] best k={k_best_mi} | wls(nm)={wls_mi}")
        Xtr_mi, Xva_mi, Xte_mi = X_tr[:, cols_mi], X_va[:, cols_mi], X_te[:, cols_mi]
    else:
        Xtr_mi, Xva_mi, Xte_mi = Xtr_sel[:, cols_mi], Xva_sel[:, cols_mi], Xte_sel[:, cols_mi]
    res_plsr_mi  = run_plsr_trainval_test(Xtr_mi, y_tr, Xva_mi, y_va, Xte_mi, y_te, ncomp_min=2, ncomp_max=min(20, Xtr_mi.shape[1]))
    res_ridge_mi = run_ridge_trainval_test(Xtr_mi, y_tr, Xva_mi, y_va, Xte_mi, y_te)
    print(f"[MI]  PLSR  Test R²={res_plsr_mi['r2']:.3f} | RMSE={res_plsr_mi['rmse']:.3f}")
    print(f"[MI]  Ridge Test R²={res_ridge_mi['r2']:.3f} | RMSE={res_ridge_mi['rmse']:.3f}")

order_lasso, alpha_lasso = lasso_select(X_tr, y_tr)
if len(order_lasso) > 0:
    k_lasso = min(K_MAX, max(K_MIN, len(order_lasso)))
    cols_lasso = order_lasso[:k_lasso]
    print(f"\n[LassoCV] α={alpha_lasso:.4g} | selected={len(order_lasso)} → using first {k_lasso}")
    print("[LassoCV] wls(nm)=", [int(round(wls_pp[j])) for j in cols_lasso])
    Xtr_la, Xva_la, Xte_la = X_tr[:, cols_lasso], X_va[:, cols_lasso], X_te[:, cols_lasso]
    res_plsr_la  = run_plsr_trainval_test(Xtr_la, y_tr, Xva_la, y_va, Xte_la, y_te, ncomp_min=2, ncomp_max=min(20, Xtr_la.shape[1]))
    res_ridge_la = run_ridge_trainval_test(Xtr_la, y_tr, Xva_la, y_va, Xte_la, y_te)
    print(f"[Lasso] PLSR  Test R²={res_plsr_la['r2']:.3f} | RMSE={res_plsr_la['rmse']:.3f}")
    print(f"[Lasso] Ridge Test R²={res_ridge_la['r2']:.3f} | RMSE={res_ridge_la['rmse']:.3f}")

#Export table of selected bands (sSPA)
sel_table = pd.DataFrame({
    "rank": np.arange(1, len(sel_cols)+1),
    "col_index": sel_cols,
    "wavelength_nm": sel_wls
})
print("\nTrain-only sSPA selected bands:")
print(sel_table.to_string(index=False))

# Data scaling (fit on TRAIN ONLY for model selection; for final training after early-stop, refit on TRAIN+VAL)
sc_tr = StandardScaler().fit(Xtr_sel)
Xtr_s = sc_tr.transform(Xtr_sel).astype(np.float32)
Xva_s = sc_tr.transform(Xva_sel).astype(np.float32)
Xte_s = sc_tr.transform(Xte_sel).astype(np.float32)

class SpectraDS(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]

tr_loader = DataLoader(SpectraDS(Xtr_s, y_tr), batch_size=CFG["batch_size"], shuffle=True, drop_last=False)
va_loader = DataLoader(SpectraDS(Xva_s, y_va), batch_size=CFG["batch_size"], shuffle=False,  drop_last=False)
te_loader = DataLoader(SpectraDS(Xte_s, y_te), batch_size=CFG["batch_size"], shuffle=False,  drop_last=False)

#Model
class PosEnc1D(nn.Module):
    def __init__(self, d_model, max_len=4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0)/d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)
    def forward(self, x):  # x: (B,L,D)
        return x + self.pe[:x.size(1)]

class EncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_ff, dropout):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop  = nn.Dropout(dropout)
    def forward(self, x):
        y, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + self.drop(y))
        y = self.ffn(x)
        x = self.norm2(x + self.drop(y))
        return x

class SelfTransformerRegressor(nn.Module):
    def __init__(self, n_bands, d_model=32, nhead=2, num_layers=2, dim_ff=96, dropout=0.2):
        super().__init__()
        self.embed = nn.Linear(1, d_model)
        self.pos = PosEnc1D(d_model, max_len=n_bands)
        self.blocks = nn.ModuleList([EncoderBlock(d_model, nhead, dim_ff, dropout) for _ in range(num_layers)])
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(d_model, 1)
        )
    def forward(self, x):
        z = self.embed(x.unsqueeze(-1))
        z = self.pos(z)
        for blk in self.blocks:
            z = blk(z)
        z = z.mean(dim=1)               
        return self.head(z).squeeze(-1)

#Schedulers & training utils
def cosine_with_warmup(epoch, base_lr, warmup, max_epochs, min_lr):
    if epoch < warmup:
        return base_lr * (epoch + 1) / max(1, warmup)
    t = (epoch - warmup) / max(1, (max_epochs - warmup))
    return min_lr + 0.5*(base_lr - min_lr)*(1 + math.cos(math.pi * t))

@torch.no_grad()
def eval_loader(model, loader, crit):
    model.eval()
    losses, yT, yP = [], [], []
    for xb, yb in loader:
        xb = xb.to(device); yb = yb.to(device)
        pred = model(xb)
        losses.append(crit(pred, yb).item())
        yT.append(yb.detach().cpu().numpy())
        yP.append(pred.detach().cpu().numpy())
    yT = np.concatenate(yT); yP = np.concatenate(yP)
    rmse = float(np.sqrt(np.mean((yT - yP)**2)))
    ss_res = float(np.sum((yT - yP)**2))
    ss_tot = float(np.sum((yT - yT.mean())**2) + 1e-12)
    r2 = 1.0 - ss_res/ss_tot
    return rmse, r2, yP

def train_one_seed(seed, verbose_every=5):
    """
    当 CONTINUE_TRAINING=True：
      - 直接加载 WEIGHTS_PATH 的已训练权重
      - 在 Train/Val 上继续训练满 CFG["max_epochs"] 轮，并打印日志
      - 选择验证集最佳权重后，进入原有 TRAIN+VAL 的 refit 阶段，保存权重并评估

    当 CONTINUE_TRAINING=False：
      - 保持原流程：从零训练+早停 → refit（TRAIN+VAL）→ 保存权重 → 评估
    """
    torch.manual_seed(seed); np.random.seed(seed)

    # 先准备 Train/Val/Test 的缩放和 DataLoader（两条分支都会用到）
    sc_tr = StandardScaler().fit(Xtr_sel)
    Xtr_s = sc_tr.transform(Xtr_sel).astype(np.float32)
    Xva_s = sc_tr.transform(Xva_sel).astype(np.float32)
    Xte_s = sc_tr.transform(Xte_sel).astype(np.float32)

    tr_loader = DataLoader(SpectraDS(Xtr_s, y_tr), batch_size=CFG["batch_size"], shuffle=True, drop_last=False)
    va_loader = DataLoader(SpectraDS(Xva_s, y_va), batch_size=CFG["batch_size"], shuffle=False, drop_last=False)

    # === 分支 A：继续训练（加载后在 Train/Val 上跑满 max_epochs，并打印日志） ===
    if CONTINUE_TRAINING:
        # 1) 构建与保存时一致的模型结构（band 数需一致）
        model = SelfTransformerRegressor(
            n_bands=Xtr_s.shape[1],
            d_model=CFG["d_model"], nhead=CFG["nhead"],
            num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
        ).to(device)

        # 2) 载入已有权重；失败则报错，避免意外从零训
        ok = try_load_model_weights(WEIGHTS_PATH, model)
        if not ok:
            raise FileNotFoundError(
                f"[ResumeError] CONTINUE_TRAINING=True 但无法加载权重：{WEIGHTS_PATH}。"
                " 请确认文件存在且可信，或将 CONTINUE_TRAINING=False 先从头训练一次。"
            )

        # 3) 继续训练：在 Train 上优化，在 Val 上评估，打印与原样式一致的日志
        opt  = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        crit = nn.SmoothL1Loss(beta=CFG["huber_beta"])
        best_val = float("inf")
        best_state = None

        for ep in range(1, CFG["max_epochs"] + 1):
            # 余弦退火 + warmup（这里继续训练阶段长度为 max_epochs）
            for pg in opt.param_groups:
                pg["lr"] = cosine_with_warmup(ep-1, CFG["lr"], CFG["warmup_epochs"], CFG["max_epochs"], CFG["cosine_min_lr"])

            model.train()
            train_mse_batches = []
            for xb, yb in tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt.zero_grad(set_to_none=True)
                pred = model(xb)
                loss = crit(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
                opt.step()
                with torch.no_grad():
                    train_mse_batches.append(torch.mean((pred - yb)**2).item())

            tr_rmse = float(np.sqrt(np.mean(train_mse_batches)))
            val_rmse, val_r2, _ = eval_loader(model, va_loader, crit)

            # 按原来格式打印；可用 verbose_every 控制频率
            if ep == 1 or ep % verbose_every == 0:
                print(f"Epoch {ep:03d} | trainRMSE={tr_rmse:.3f} | valRMSE={val_rmse:.3f} | valR²={val_r2:.3f}")

            # 记录“最好验证表现”的权重
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse
                best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

        # 用最佳验证权重（如有）
        if best_state is not None:
            model.load_state_dict(best_state)

        # 4) 进入原有的 TRAIN+VAL refit 阶段，并保存可供下次继续训练的权重
        sc_tv = StandardScaler().fit(np.vstack([Xtr_sel, Xva_sel]))
        Xtv_s = sc_tv.transform(np.vstack([Xtr_sel, Xva_sel])).astype(np.float32)
        y_tv  = np.concatenate([y_tr, y_va]).astype(np.float32)
        tv_loader = DataLoader(SpectraDS(Xtv_s, y_tv), batch_size=CFG["batch_size"], shuffle=True, drop_last=False)
        te_loader_refit = DataLoader(
            SpectraDS(sc_tv.transform(Xte_sel).astype(np.float32), y_te.astype(np.float32)),
            batch_size=CFG["batch_size"], shuffle=False, drop_last=False
        )

        model2 = SelfTransformerRegressor(
            n_bands=Xtv_s.shape[1],
            d_model=CFG["d_model"], nhead=CFG["nhead"],
            num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
        ).to(device)
        model2.load_state_dict(model.state_dict(), strict=False)

        opt2  = torch.optim.AdamW(model2.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        crit2 = nn.SmoothL1Loss(beta=CFG["huber_beta"])
        refit_epochs = min(int(1.2*CFG["patience"]), int(0.6*CFG["max_epochs"]))
        for ep in range(1, refit_epochs+1):
            for pg in opt2.param_groups:
                pg["lr"] = cosine_with_warmup(ep-1, CFG["lr"], CFG["warmup_epochs"], refit_epochs, CFG["cosine_min_lr"])
            model2.train()
            for xb, yb in tv_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt2.zero_grad(set_to_none=True)
                pred = model2(xb)
                loss = crit2(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model2.parameters(), CFG["grad_clip"])
                opt2.step()

        save_model_weights(
            WEIGHTS_PATH,
            model2,
            extra={
                "cfg": CFG,
                "sel_cols": sel_cols,
                "scaler_mean": sc_tv.mean_,
                "scaler_scale": sc_tv.scale_,
            },
        )

        te_rmse, te_r2, te_pred = eval_loader(model2, te_loader_refit, crit2)
        return te_pred, te_rmse, te_r2

    # === 分支 B：从头训练 + 早停（保持原逻辑不变） ===
    model = SelfTransformerRegressor(
        n_bands=Xtr_s.shape[1],
        d_model=CFG["d_model"], nhead=CFG["nhead"],
        num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
    ).to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    crit = nn.SmoothL1Loss(beta=CFG["huber_beta"])  # Huber
    best_val = float("inf")
    best_state = None
    wait = 0

    for ep in range(1, CFG["max_epochs"]+1):
        for pg in opt.param_groups:
            pg["lr"] = cosine_with_warmup(ep-1, CFG["lr"], CFG["warmup_epochs"], CFG["max_epochs"], CFG["cosine_min_lr"])
        model.train()
        train_mse_batches = []
        for xb, yb in tr_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            loss = crit(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            opt.step()
            with torch.no_grad():
                train_mse_batches.append(torch.mean((pred - yb)**2).item())
        tr_rmse = float(np.sqrt(np.mean(train_mse_batches)))
        val_rmse, val_r2, _ = eval_loader(model, va_loader, crit)
        if ep == 1 or ep % verbose_every == 0:
            print(f"Epoch {ep:03d} | trainRMSE={tr_rmse:.3f} | valRMSE={val_rmse:.3f} | valR²={val_r2:.3f}")
        if val_rmse < best_val - 1e-6:
            best_val = val_rmse; wait = 0
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= CFG["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)

    # Refit on TRAIN+VAL（保持原逻辑）
    sc_tv = StandardScaler().fit(np.vstack([Xtr_sel, Xva_sel]))
    Xtv_s = sc_tv.transform(np.vstack([Xtr_sel, Xva_sel])).astype(np.float32)
    y_tv  = np.concatenate([y_tr, y_va]).astype(np.float32)
    tv_loader = DataLoader(SpectraDS(Xtv_s, y_tv), batch_size=CFG["batch_size"], shuffle=True, drop_last=False)
    te_loader_refit = DataLoader(
        SpectraDS(sc_tv.transform(Xte_sel).astype(np.float32), y_te.astype(np.float32)),
        batch_size=CFG["batch_size"], shuffle=False, drop_last=False
    )

    torch.manual_seed(seed)
    model2 = SelfTransformerRegressor(
        n_bands=Xtv_s.shape[1],
        d_model=CFG["d_model"], nhead=CFG["nhead"],
        num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
    ).to(device)
    opt2  = torch.optim.AdamW(model2.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    crit2 = nn.SmoothL1Loss(beta=CFG["huber_beta"])
    refit_epochs = min(int(1.2*CFG["patience"]), int(0.6*CFG["max_epochs"]))
    for ep in range(1, refit_epochs+1):
        for pg in opt2.param_groups:
            pg["lr"] = cosine_with_warmup(ep-1, CFG["lr"], CFG["warmup_epochs"], refit_epochs, CFG["cosine_min_lr"])
        model2.train()
        for xb, yb in tv_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad(set_to_none=True)
            pred = model2(xb)
            loss = crit2(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model2.parameters(), CFG["grad_clip"])
            opt2.step()

    save_model_weights(
        WEIGHTS_PATH,
        model2,
        extra={
            "cfg": CFG,
            "sel_cols": sel_cols,
            "scaler_mean": sc_tv.mean_,
            "scaler_scale": sc_tv.scale_,
        },
    )

    te_rmse, te_r2, te_pred = eval_loader(model2, te_loader_refit, crit2)
    return te_pred, te_rmse, te_r2


# ====== 单seed=23 训练与评估 ======
SEED = 23
y_pred, test_rmse, test_r2 = train_one_seed(SEED)
print(f"\nSeed {SEED} → Test RMSE={test_rmse:.3f} | Test R²={test_r2:.3f}")

#Optional: scatter plot
plt.figure(figsize=(4.8,4.2))
plt.scatter(y_te, y_pred, s=16)
mn, mx = min(y_te.min(), y_pred.min()), max(y_te.max(), y_pred.max())
plt.plot([mn,mx],[mn,mx],'k--',lw=1)
plt.xlabel("True Nmass_O"); plt.ylabel("Predicted Nmass_O"); plt.title(f"Self-Transformer — Test (seed={SEED})")
plt.tight_layout(); plt.show()
# ====== end ======

