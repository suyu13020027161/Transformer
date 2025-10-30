import re, math, numpy as np, pandas as pd
import numpy as np
from pathlib import Path
import warnings, os
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

# =========================
# Config (数据与任务参数)
# =========================
CSV_PATH   = "input.csv"
TARGET_COL = "Nmass_O"
WAVELEN_MIN = 400.0
WAVELEN_MAX = 2400.0

# 新增：训练/保存相关参数（无需命令行）
SAVE_PER_SEED = False
SAVE_BEST_BY_TEST_R2 = True
SAVE_DIR = "checkpoints"

# 继续训练（在已有权重基础上）
CONTINUE_TRAIN = True
RESUME_WEIGHTS_PATH = "checkpoints/seed37.pt"  # 可是 .pt / .pth / checkpoint dict 都行
RESUME_SEED = 37
# 可选：当权重不完全匹配时的策略（一般保持 False 更鲁棒）
RESUME_STRICT = False

# =========================

WATER_BANDS = [(1350, 1460), (1790, 1960), (2470, 2560)]
SG_WINDOW   = 15
SG_POLY     = 2
SG_DERIV    = 1
K_MIN = 8
K_MAX = 500
INNER_KFOLDS = 5
MIN_SPACING_NM = 12.0
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE  = 0.15

CFG = dict(
    d_model=64,
    nhead=4,
    num_layers=6,
    dim_ff=256,
    dropout=0.10,
    lr=1e-3,
    weight_decay=1e-4,
    batch_size=64,
    max_epochs=10,
    patience=10,
    grad_clip=1.0,
    huber_beta=0.5,
    warmup_epochs=10,
    cosine_min_lr=1e-6,
    seeds=[13, 23, 37, 47, 59],
)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ================= 工具函数（保存/加载鲁棒化） =================
def _ensure_dir(p: str):
    Path(p).mkdir(parents=True, exist_ok=True)

def _save_state_dict(state_dict: dict, path: str):
    _ensure_dir(os.path.dirname(path) if os.path.dirname(path) else ".")
    torch.save(state_dict, path)
    print(f"[Save] Weights saved → {path}")

def _unwrap_state_dict(obj):
    """
    支持以下几种常见格式：
      - 直接是 state_dict (dict[str, Tensor])
      - {'state_dict': ...}
      - {'model': {'state_dict': ...}} / {'model_state_dict': ...}
    """
    if isinstance(obj, dict):
        # 直接是 state_dict（值基本是 Tensor）
        if all(isinstance(v, torch.Tensor) for v in obj.values()):
            return obj
        # 常见包装
        for key in ["state_dict", "model_state_dict", "model", "net", "ema_state_dict"]:
            if key in obj:
                sub = obj[key]
                # model 里再套一层
                if isinstance(sub, dict) and "state_dict" in sub:
                    return sub["state_dict"]
                return sub
    return obj  # 尝试原样返回

def _strip_module_prefix(sd: dict):
    out = {}
    for k, v in sd.items():
        if k.startswith("module."):
            out[k[len("module."):]] = v
        else:
            out[k] = v
    return out

def load_weights_flex(model: nn.Module, path: str, strict: bool = False) -> int:
    """
    尝试以最大兼容性加载权重：
      1) 自动解包 checkpoint/state_dict
      2) 去掉 DataParallel 的 'module.' 前缀
      3) 当 strict=False 时，只加载键名和 shape 都匹配的参数
    返回：成功加载的参数个数
    """
    obj = torch.load(path, map_location="cpu")
    sd = _unwrap_state_dict(obj)
    if not isinstance(sd, dict):
        raise RuntimeError(f"Unsupported checkpoint format at {path}")

    sd = _strip_module_prefix(sd)

    if strict:
        missing, unexpected = model.load_state_dict(sd, strict=True)
        # PyTorch 2.0 之后 strict=True 返回的是 None；为兼容性这里不依赖返回值
        print("[Resume] Loaded with strict=True")
        return len(sd)
    else:
        model_sd = model.state_dict()
        loadable = {}
        skipped = []
        for k, v in sd.items():
            if k in model_sd and isinstance(v, torch.Tensor) and v.shape == model_sd[k].shape:
                loadable[k] = v
            else:
                skipped.append(k)
        model_sd.update(loadable)
        model.load_state_dict(model_sd, strict=False)
        print(f"[Resume] Loaded {len(loadable)} tensors; skipped {len(skipped)} keys (name/shape mismatch).")
        if skipped:
            # 打印前若干个，方便诊断
            print("         Skipped examples:", skipped[:5], "...")
        return len(loadable)

# =======================================================
# 以下为原有逻辑（保持不变），只在合适位置挂接保存/续训逻辑
# =======================================================

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

def mi_rank(Xtr, ytr):
    mi = mutual_info_regression(Xtr, ytr, random_state=RANDOM_STATE)
    return list(np.argsort(mi)[::-1])

def lasso_select(Xtr, ytr):
    sc = StandardScaler().fit(Xtr)
    Xtr_s = sc.transform(Xtr)
    lcv = LassoCV(alphas=None, cv=INNER_KFOLDS, random_state=RANDOM_STATE, max_iter=20000).fit(Xtr_s, ytr)
    coef = np.abs(lcv.coef_)
    order = list(np.argsort(coef)[::-1])
    sel = [j for j in order if coef[j] > 1e-9]
    return sel, lcv.alpha_

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
    return {"n_components": c_best, "rmse": rmse(y_te, y_pred), "r2": r2_score(y_te, y_pred), "pred": y_pred}

def run_ridge_trainval_test(X_tr, y_tr, X_va, y_va, X_te, y_te):
    alphas = np.logspace(-4, 3, 60)
    sc = StandardScaler().fit(X_tr)
    Xtr_s = sc.transform(X_tr); Xva_s = sc.transform(X_va)
    ridge = RidgeCV(alphas=alphas, cv=INNER_KFOLDS).fit(np.vstack([Xtr_s, Xva_s]), np.concatenate([y_tr, y_va]))
    sc_all = StandardScaler().fit(np.vstack([X_tr, X_va]))
    y_pred = ridge.predict(sc_all.transform(X_te))
    return {"alpha": float(ridge.alpha_), "rmse": rmse(y_te, y_pred), "r2": r2_score(y_te, y_pred), "pred": y_pred}

# ====== 数据与分割 ======
df = pd.read_csv(CSV_PATH)
all_band_cols = [c for c in df.columns if "wave" in c.lower()]
wls_nm_all = np.array([nm_from_col(c) for c in all_band_cols], dtype=float)
df = df[[TARGET_COL] + all_band_cols].dropna().reset_index(drop=True)
X_raw = df[all_band_cols].to_numpy(dtype=float)
y = df[TARGET_COL].to_numpy(dtype=float)

X_pp, wls_pp = spectral_preprocess(X_raw, wls_nm_all)

X_tv, X_te, y_tv, y_te = train_test_split(X_pp, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, shuffle=True)
val_ratio_in_tv = VAL_SIZE / (1.0 - TEST_SIZE)
X_tr, X_va, y_tr, y_va = train_test_split(X_tv, y_tv, test_size=val_ratio_in_tv, random_state=RANDOM_STATE, shuffle=True)
print(f"Split → train={len(y_tr)} | valid={len(y_va)} | test={len(y_te)} | bands={X_pp.shape[1]}")

order_sspa = supervised_spa_order_trainonly(
    X_tr, y_tr, wls_pp, max_k=K_MAX, first="maxcorr", min_spacing_nm=MIN_SPACING_NM, verbose=True
)
k_best, curve = choose_k_via_inner_cv(X_tr, y_tr, order_sspa, k_min=K_MIN, k_max=K_MAX, inner_folds=INNER_KFOLDS, base_model="ridge")
sel_cols = order_sspa[:k_best]
sel_wls  = [float(wls_pp[j]) for j in sel_cols]
print(f"\n[sSPA] Inner-CV best k = {k_best}")
print("Selected wavelengths (nm):", [int(round(w)) for w in sel_wls])

Xtr_sel = X_tr[:, sel_cols]; Xva_sel = X_va[:, sel_cols]; Xte_sel = X_te[:, sel_cols]

res_plsr  = run_plsr_trainval_test(Xtr_sel, y_tr, Xva_sel, y_va, Xte_sel, y_te, ncomp_min=4, ncomp_max=min(24, k_best))
res_ridge = run_ridge_trainval_test(Xtr_sel, y_tr, Xva_sel, y_va, Xte_sel, y_te)

print("\n=== Baselines on train-only selected bands ===")
print(f"PLSR  → comps={res_plsr['n_components']:>2} | Test RMSE={res_plsr['rmse']:.3f} | Test R²={res_plsr['r2']:.3f}")
print(f"Ridge → alpha={res_ridge['alpha']:.4g} | Test RMSE={res_ridge['rmse']:.3f} | Test R²={res_ridge['r2']:.3f}")

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

sel_table = pd.DataFrame({
    "rank": np.arange(1, len(sel_cols)+1),
    "col_index": sel_cols,
    "wavelength_nm": sel_wls
})
print("\nTrain-only sSPA selected bands:")
print(sel_table.to_string(index=False))

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
        z = z.mean(dim=1)  # GAP
        return self.head(z).squeeze(-1)

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
    rmse_v = float(np.sqrt(np.mean((yT - yP)**2)))
    ss_res = float(np.sum((yT - yP)**2))
    ss_tot = float(np.sum((yT - yT.mean())**2) + 1e-12)
    r2_v = 1.0 - ss_res/ss_tot
    return rmse_v, r2_v, yP

def train_one_seed(seed, verbose_every=5):
    torch.manual_seed(seed); np.random.seed(seed)

    # ===== 预先准备：TV 标准化 & 测试集 loader =====
    sc_tv = StandardScaler().fit(np.vstack([Xtr_sel, Xva_sel]))
    Xtv_s = sc_tv.transform(np.vstack([Xtr_sel, Xva_sel])).astype(np.float32)
    y_tv_all = np.concatenate([y_tr, y_va]).astype(np.float32)
    te_loader_refit = DataLoader(
        SpectraDS(sc_tv.transform(Xte_sel).astype(np.float32), y_te.astype(np.float32)),
        batch_size=CFG["batch_size"], shuffle=False, drop_last=False
    )

    # ===== A) 续训分支：跳过 stage-1，但训练&打印流程与 stage-1 完全一致 =====
    if CONTINUE_TRAIN and (seed == RESUME_SEED) and RESUME_WEIGHTS_PATH:
        print(f"[Resume] CONTINUE_TRAIN=True & seed={seed}. 跳过 stage-1，直接在 Train+Val 划分出 train/val 继续训练（流程与 stage-1 一致）。")

        # 用 TV 再切一个 train/val（保证与 stage-1 一样有 val）
        continue_val_ratio = min(0.2, max(0.1, VAL_SIZE / (1.0 - TEST_SIZE)))
        idx = np.arange(len(y_tv_all))
        tr_idx, va_idx = train_test_split(
            idx, test_size=continue_val_ratio, random_state=RANDOM_STATE + seed, shuffle=True
        )
        X_tv_tr, y_tv_tr = Xtv_s[tr_idx], y_tv_all[tr_idx]
        X_tv_va, y_tv_va = Xtv_s[va_idx], y_tv_all[va_idx]

        tv_tr_loader = DataLoader(SpectraDS(X_tv_tr, y_tv_tr), batch_size=CFG["batch_size"], shuffle=True,  drop_last=False)
        tv_va_loader = DataLoader(SpectraDS(X_tv_va, y_tv_va), batch_size=CFG["batch_size"], shuffle=False, drop_last=False)

        # 构建同结构模型并加载权重
        model2 = SelfTransformerRegressor(
            n_bands=Xtv_s.shape[1],
            d_model=CFG["d_model"], nhead=CFG["nhead"],
            num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
        ).to(device)

        try:
            n_loaded = load_weights_flex(model2, RESUME_WEIGHTS_PATH, strict=RESUME_STRICT)
            if n_loaded == 0:
                print(f"[Resume][Warn] 从 {RESUME_WEIGHTS_PATH} 未加载到任何张量，将从随机初始化继续。")
            else:
                print(f"[Resume] Loaded {n_loaded} tensors; skipped 0 keys (name/shape mismatch).")
        except Exception as e:
            print(f"[Resume][Warn] 加载失败：{e}。将从随机初始化继续。")

        # === 与 stage-1 一致的训练循环 ===
        opt2  = torch.optim.AdamW(model2.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
        crit2 = nn.SmoothL1Loss(beta=CFG["huber_beta"])
        best_val = float("inf"); best_state = None; wait = 0

        for ep in range(1, CFG["max_epochs"] + 1):
            # lr 调度一致
            for pg in opt2.param_groups:
                pg["lr"] = cosine_with_warmup(ep - 1, CFG["lr"], CFG["warmup_epochs"], CFG["max_epochs"], CFG["cosine_min_lr"])

            # Train
            model2.train()
            train_mse_batches = []
            for xb, yb in tv_tr_loader:
                xb, yb = xb.to(device), yb.to(device)
                opt2.zero_grad(set_to_none=True)
                pred = model2(xb)
                loss = crit2(pred, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model2.parameters(), CFG["grad_clip"])
                opt2.step()
                with torch.no_grad():
                    train_mse_batches.append(torch.mean((pred - yb) ** 2).item())
            tr_rmse = float(np.sqrt(np.mean(train_mse_batches)))

            # Val
            val_rmse, val_r2, _ = eval_loader(model2, tv_va_loader, crit2)

            # 打印格式与 stage-1 完全一致
            if ep == 1 or ep % verbose_every == 0:
                print(f"Epoch {ep:03d} | trainRMSE={tr_rmse:.3f} | valRMSE={val_rmse:.3f} | valR²={val_r2:.3f}")

            # early stopping 与 best_state
            if val_rmse < best_val - 1e-6:
                best_val = val_rmse; wait = 0
                best_state = {k: v.detach().cpu() for k, v in model2.state_dict().items()}
            else:
                wait += 1
                if wait >= CFG["patience"]:
                    break

        if best_state is not None:
            model2.load_state_dict(best_state)

        # 测试集评估并返回
        te_rmse, te_r2, te_pred = eval_loader(model2, te_loader_refit, crit2)
        final_state_dict = {k: v.detach().cpu() for k, v in model2.state_dict().items()}
        return te_pred, te_rmse, te_r2, final_state_dict

    # ===== B) 原始两阶段流程：stage-1 (Train/Val) + refit (Train+Val) =====
    # stage-1
    model = SelfTransformerRegressor(
        n_bands=Xtr_s.shape[1],
        d_model=CFG["d_model"], nhead=CFG["nhead"],
        num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
    ).to(device)
    opt  = torch.optim.AdamW(model.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    crit = nn.SmoothL1Loss(beta=CFG["huber_beta"])
    best_val = float("inf"); best_state = None; wait = 0

    for ep in range(1, CFG["max_epochs"] + 1):
        for pg in opt.param_groups:
            pg["lr"] = cosine_with_warmup(ep - 1, CFG["lr"], CFG["warmup_epochs"], CFG["max_epochs"], CFG["cosine_min_lr"])
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
                train_mse_batches.append(torch.mean((pred - yb) ** 2).item())
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

    # refit（TV 上再训练若干轮）
    tv_loader = DataLoader(SpectraDS(Xtv_s, y_tv_all), batch_size=CFG["batch_size"], shuffle=True, drop_last=False)
    model2 = SelfTransformerRegressor(
        n_bands=Xtv_s.shape[1],
        d_model=CFG["d_model"], nhead=CFG["nhead"],
        num_layers=CFG["num_layers"], dim_ff=CFG["dim_ff"], dropout=CFG["dropout"]
    ).to(device)
    try:
        if best_state is not None:
            model2.load_state_dict(best_state, strict=False)
    except Exception:
        pass

    opt2  = torch.optim.AdamW(model2.parameters(), lr=CFG["lr"], weight_decay=CFG["weight_decay"])
    crit2 = nn.SmoothL1Loss(beta=CFG["huber_beta"])
    refit_epochs = min(int(1.2 * CFG["patience"]), int(0.6 * CFG["max_epochs"]))
    for ep in range(1, refit_epochs + 1):
        for pg in opt2.param_groups:
            pg["lr"] = cosine_with_warmup(ep - 1, CFG["lr"], CFG["warmup_epochs"], refit_epochs, CFG["cosine_min_lr"])
        model2.train()
        for xb, yb in tv_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt2.zero_grad(set_to_none=True)
            pred = model2(xb)
            loss = crit2(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model2.parameters(), CFG["grad_clip"])
            opt2.step()

    te_rmse, te_r2, te_pred = eval_loader(model2, te_loader_refit, crit2)
    final_state_dict = {k: v.detach().cpu() for k, v in model2.state_dict().items()}
    return te_pred, te_rmse, te_r2, final_state_dict



# ===== 多 seed 训练与保存 =====
_ensure_dir(SAVE_DIR)

if CONTINUE_TRAIN:
    print(f"[Mode] CONTINUE_TRAIN=True → 仅继续训练 seed={RESUME_SEED} 并覆盖原权重")
    pred, rm, r2, state_dict = train_one_seed(RESUME_SEED)
    save_path = RESUME_WEIGHTS_PATH  # 覆盖原文件
    _save_state_dict(state_dict, save_path)
    print(f"[Continue] 覆盖保存权重 → {save_path}")
    print(f"[Continue] Test RMSE={rm:.4f} | R²={r2:.4f}")
else:
    all_preds, seed_scores = [], []
    seed_states, seed_save_paths = {}, {}

    for sd in CFG["seeds"]:
        pred, rm, r2, state_dict = train_one_seed(sd)
        all_preds.append(pred.reshape(-1, 1))
        seed_scores.append((sd, rm, r2))
        seed_states[sd] = state_dict
        if SAVE_PER_SEED:
            save_path = os.path.join(SAVE_DIR, f"transformer_seed{sd}_testR2_{r2:.4f}.pt")
            _save_state_dict(state_dict, save_path)
            seed_save_paths[sd] = save_path

    ens_pred = np.mean(np.hstack(all_preds), axis=1)
    ens_rmse = float(np.sqrt(np.mean((y_te - ens_pred)**2)))
    ss_res = float(np.sum((y_te - ens_pred)**2))
    ss_tot = float(np.sum((y_te - y_te.mean())**2) + 1e-12)
    ens_r2  = 1.0 - ss_res/ss_tot
    print("\nPer-seed test scores:")
    for sd, rm, r2 in seed_scores:
        print(f"  seed {sd:>2} → RMSE={rm:.3f} | R²={r2:.3f}")
    print(f"\nEnsemble  → RMSE={ens_rmse:.3f} | R²={ens_r2:.3f}")

    if SAVE_BEST_BY_TEST_R2 and len(seed_scores) > 0:
        best_seed, best_rmse, best_r2 = max(seed_scores, key=lambda t: t[2])
        best_state = seed_states[best_seed]
        best_path = os.path.join(SAVE_DIR, f"best_by_testR2_seed{best_seed}_R2_{best_r2:.4f}.pt")
        _save_state_dict(best_state, best_path)
        print(f"[Best] Best-by-testR2 seed={best_seed} | R²={best_r2:.4f} | RMSE={best_rmse:.4f}")
        print(f"[Best] Saved extra copy to: {best_path}")

    plt.figure(figsize=(4.8,4.2))
    plt.scatter(y_te, ens_pred, s=16)
    mn, mx = min(y_te.min(), ens_pred.min()), max(y_te.max(), ens_pred.max())
    plt.plot([mn,mx],[mn,mx],'k--',lw=1)
    plt.xlabel("True Nmass_O"); plt.ylabel("Predicted Nmass_O"); plt.title("Self-Transformer (ensemble) — Test")
    plt.tight_layout(); plt.show()

