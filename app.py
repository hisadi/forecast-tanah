# app.py — Halaman Training/Evaluasi + Pilih Algoritma + Outlier + Diagnostik + Save Bundle
# Jalankan: streamlit run app.py

import io, os, re, json, unicodedata, pickle
from typing import Optional, List, Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Algo umum (selalu ada)
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor

# Custom encoders
from custom_transformers import AddressTopTokens, FrequencyEncoder

# Algo opsional
XGB_OK = LGBM_OK = CAT_OK = False
try:
    from xgboost import XGBRegressor
    XGB_OK = True
except Exception:
    pass
try:
    from lightgbm import LGBMRegressor
    LGBM_OK = True
except Exception:
    pass
try:
    from catboost import CatBoostRegressor
    CAT_OK = True
except Exception:
    pass

# --- (opsional) SciPy untuk fit distribusi ---
try:
    import scipy.stats as ss
    SCIPY_OK = True
except Exception:
    SCIPY_OK = False


# =========================
# Util parsing angka lokal
# =========================
def _to_float_local(num_str: str) -> Optional[float]:
    if not isinstance(num_str, str):
        return None
    s = re.sub(r"[^0-9\-,\.]", "", num_str.strip())
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except:
        return None

def rupiah_to_number(text: str) -> Optional[float]:
    if not isinstance(text, str):
        return None
    s = text.lower()
    m = re.search(r"([0-9\.\,]+)", s)
    if not m:
        return None
    val = _to_float_local(m.group(1))
    if val is None:
        return None
    if   "triliun" in s: val *= 1_000_000_000_000
    elif "milyar" in s or "miliar" in s: val *= 1_000_000_000
    elif "juta"  in s: val *= 1_000_000
    elif "ribu"  in s: val *= 1_000
    return float(val)

def try_convert_numeric_series(ser: pd.Series, force_currency=False) -> pd.Series:
    s = ser.copy()
    if pd.api.types.is_numeric_dtype(s):
        return s
    if force_currency:
        conv = s.apply(rupiah_to_number)
        if pd.notna(conv).mean() >= 0.5:
            return pd.to_numeric(conv, errors="coerce")
    conv = s.apply(_to_float_local)
    if pd.notna(conv).mean() >= 0.7:
        return pd.to_numeric(conv, errors="coerce")
    return s

def compute_rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


# =========================
# Encoder & helper
# =========================
def make_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def slugify_name(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r'[^0-9A-Za-z_]+', '_', text)
    text = re.sub(r'_+', '_', text).strip('_')
    text = text.replace('__', '_')
    return text or "col"


# =========================
# Outlier helpers
# =========================
def _numeric_copy(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    s1 = try_convert_numeric_series(s, force_currency=True)
    return pd.to_numeric(s1, errors="coerce")

def _bounds_iqr(x: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    return (q1 - k * iqr, q3 + k * iqr)

def _bounds_zscore(x: pd.Series, z: float = 3.0) -> Tuple[float, float]:
    mu, sd = x.mean(), x.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return (x.min(), x.max())
    return (mu - z * sd, mu + z * sd)

def _bounds_log_iqr(x: pd.Series, k: float = 1.5) -> Tuple[float, float]:
    pos = x[x > 0]
    if pos.empty:
        return _bounds_iqr(x, k)
    min_pos = float(pos.min())
    x_clip = x.clip(lower=max(min_pos * 0.1, 1e-9))
    lx = np.log10(x_clip)
    llo, lhi = _bounds_iqr(lx, k)
    return (10 ** llo, 10 ** lhi)

def _bounds_quantile(x: pd.Series, q_low: float = 0.01, q_high: float = 0.99) -> Tuple[float, float]:
    return (x.quantile(q_low), x.quantile(q_high))

def detect_outlier_mask_series(
    s_raw: pd.Series,
    method: str = "auto",
    iqr_k: float = 1.5,
    z_k: float = 3.0,
    q_low: float = 0.01,
    q_high: float = 0.99,
) -> Tuple[pd.Series, Tuple[float, float]]:
    x = _numeric_copy(s_raw).dropna()
    if x.empty:
        return pd.Series(False, index=s_raw.index), (np.nan, np.nan)
    if method == "auto":
        sk = float(x.skew())
        method_eff = "log_iqr" if sk > 1.0 else "iqr"
    else:
        method_eff = method
    if method_eff == "iqr":
        lo, hi = _bounds_iqr(x, iqr_k)
    elif method_eff == "zscore":
        lo, hi = _bounds_zscore(x, z_k)
    elif method_eff == "log_iqr":
        lo, hi = _bounds_log_iqr(x, iqr_k)
    elif method_eff == "quantile":
        lo, hi = _bounds_quantile(x, q_low, q_high)
    else:
        lo, hi = _bounds_iqr(x, iqr_k)
    xx = pd.to_numeric(s_raw, errors="coerce")
    mask = (xx < lo) | (xx > hi)
    return mask.fillna(False), (lo, hi)

def winsorize_series(s_raw: pd.Series, lower: float, upper: float) -> pd.Series:
    x = pd.to_numeric(s_raw, errors="coerce")
    return x.clip(lower=lower, upper=upper)


# =========================
# Diagnostik Distribusi (fit + rekomendasi)
# =========================
def fit_distributions(x: np.ndarray) -> pd.DataFrame:
    results = []
    if not SCIPY_OK:
        return pd.DataFrame(results)
    candidates = [
        ("Normal", ss.norm, False),
        ("Lognormal", ss.lognorm, True),
        ("Exponential", ss.expon, False),
        ("Gamma", ss.gamma, False),
        ("Weibull", ss.weibull_min, False),
        ("Laplace", ss.laplace, False),
        ("StudentT", ss.t, False),
    ]
    for name, dist, needs_pos in candidates:
        xi = x.copy()
        if needs_pos:
            xi = xi[xi > 0]
            if len(xi) < max(30, int(0.2 * len(x))):
                continue
        try:
            params = dist.fit(xi)
            ll = np.sum(dist.logpdf(xi, *params))
            k = len(params)
            aic = 2 * k - 2 * ll
            ks_stat, ks_p = ss.kstest(xi, dist.name, args=params)
            results.append({
                "distribution": name,
                "aic": aic,
                "ks_pvalue": ks_p,
                "params": params,
            })
        except Exception:
            continue
    if not results:
        return pd.DataFrame(results)
    return pd.DataFrame(results).sort_values(["aic", "ks_pvalue"], ascending=[True, False]).reset_index(drop=True)

def recommend_outlier_method(skew: float, best_dist: Optional[str]) -> str:
    if best_dist is None:
        return "IQR (umum). Jika sangat skewed, coba log-IQR."
    if best_dist == "Lognormal" or skew > 1.0:
        return "log-IQR (data cenderung lognormal / miring kanan)."
    if best_dist in ["Normal", "Laplace"] and abs(skew) < 0.5:
        return "z-score atau IQR."
    if best_dist in ["Gamma", "Weibull", "Exponential"]:
        return "Quantile/IQR (heavy tail). Pertimbangkan winsorize."
    if best_dist == "StudentT":
        return "IQR atau Quantile (heavy tail)."
    return "IQR (aman)."


# =========================
# APP — TRAINING
# =========================
st.set_page_config(page_title="Training Model • Tanah", layout="wide")
st.title("Training & Evaluasi Model (Regresi)")

uploaded = st.file_uploader("Upload Excel/CSV", type=["xlsx", "xls", "csv"])
if uploaded is None:
    st.info("Silakan upload data dulu.")
    st.stop()

# baca file
df0 = pd.read_csv(uploaded) if uploaded.name.lower().endswith(".csv") else pd.read_excel(uploaded, sheet_name=0)
st.success(f"Data dimuat: {df0.shape[0]} baris × {df0.shape[1]} kolom")
st.dataframe(df0.head(15), use_container_width=True)

# target
numeric_cols_all = [c for c in df0.columns if pd.api.types.is_numeric_dtype(df0[c])]
default_target = None
for cand in ["harga_total_num", "harga total", "Harga Total", "harga", "Harga", "price", "Price", "Nilai Tanah", "nilai_tanah"]:
    if cand in df0.columns and pd.api.types.is_numeric_dtype(df0[cand]):
        default_target = cand
        break
if default_target is None and numeric_cols_all:
    default_target = numeric_cols_all[0] if numeric_cols_all else df0.columns[0]

st.subheader("1) Pilih Target (y)")
target_col = st.selectbox(
    "Target (harus numerik)",
    options=numeric_cols_all if numeric_cols_all else df0.columns,
    index=(numeric_cols_all.index(default_target) if (default_target in numeric_cols_all) else 0)
)

st.subheader("Target Type")
target_is_per_m2 = st.toggle("Target adalah Rp/m²?", value=False,
                             help="Centang jika target (y) memang berisi harga per m².")

# fitur
st.subheader("2) Pilih Kolom Fitur")
candidate_feats = [c for c in df0.columns if c != target_col]
bad_keywords = ["url", "link", "foto", "image", "gambar", "whatsapp", "telepon", "phone", "kontak", "agent", "agen", "refurl", "photo", "id"]
def is_bad(c: str) -> bool:
    return any(k in c.lower() for k in bad_keywords)
default_feats = [c for c in candidate_feats if not is_bad(c)]
chosen_feats = st.multiselect("Fitur yang dipakai", options=candidate_feats, default=default_feats)
if not chosen_feats:
    st.warning("Pilih minimal 1 fitur.")
    st.stop()

# encoding
st.subheader("3) Encoding per Kolom")
num_guess = [c for c in chosen_feats if pd.api.types.is_numeric_dtype(df0[c])]
c1, c2 = st.columns(2)
with c1:
    numeric_feats = st.multiselect("Fitur numerik (langsung)", options=chosen_feats, default=num_guess)
with c2:
    remaining = [c for c in chosen_feats if c not in numeric_feats]
    st.caption("Fitur kategori kandidat (One-Hot / Frequency):")
    st.write(remaining)

cat_remaining = [c for c in remaining]
low_card_default, high_card_default = [], []
for c in cat_remaining:
    nunique = df0[c].astype(str).nunique(dropna=True)
    (low_card_default if nunique <= 50 else high_card_default).append(c)

c3, c4 = st.columns(2)
with c3:
    onehot_feats = st.multiselect("Kategori (One-Hot)", options=cat_remaining, default=low_card_default)
with c4:
    freq_default = [c for c in cat_remaining if c not in onehot_feats] or high_card_default
    freq_feats = st.multiselect("Kategori (Frequency Encoding)", options=[c for c in cat_remaining if c not in onehot_feats], default=freq_default)

# alamat (opsional)
addr_candidates = [c for c in cat_remaining if any(k in c.lower() for k in ["alamat", "address", "lokasi", "jalan"])]
addr_feats = st.multiselect("Kolom Alamat (opsional: Top-N token + Freq)", options=cat_remaining, default=addr_candidates)
top_n_addr = st.slider("Top-N token alamat", 10, 120, 40, 5) if addr_feats else 0
onehot_feats = [c for c in onehot_feats if c not in addr_feats]
freq_feats   = [c for c in freq_feats   if c not in addr_feats]

# konversi angka
st.subheader("4) Opsi Konversi Angka")
force_currency_cols = st.multiselect("Kolom rupiah (paksa parsing Rp→angka)", options=[c for c in numeric_feats if c in df0.columns], default=[])
auto_num_convert = st.checkbox("Coba konversi string→angka (≥70% sukses)", value=True)

# diagnostik distribusi
st.subheader("4.2) Diagnostik Distribusi Target/Kolom Numerik")
num_like_cols = []
for c in [target_col] + chosen_feats:
    ser = try_convert_numeric_series(df0[c], force_currency=True)
    if pd.notna(ser).mean() >= 0.7:
        num_like_cols.append(c)
num_like_cols = list(dict.fromkeys(num_like_cols))

dist_col = st.selectbox("Pilih kolom uji distribusi", options=num_like_cols, index=0)
if st.button("🔍 Diagnosa Distribusi"):
    s_num = try_convert_numeric_series(df0[dist_col], force_currency=True)
    x = pd.to_numeric(s_num, errors="coerce").dropna().values.astype(float)
    st.write(f"N sampel valid: {len(x)}")
    if len(x) < 30:
        st.warning("Sampel < 30; hasil uji distribusi bisa tidak stabil.")
    skew = float(pd.Series(x).skew()); kurt = float(pd.Series(x).kurt())
    st.write(pd.DataFrame([{"skew": skew, "excess_kurtosis": kurt}]))

    if SCIPY_OK:
        res = fit_distributions(x)
        if not res.empty:
            st.markdown("**Fit Distribusi (AIC terkecil terbaik):**")
            st.dataframe(res[["distribution", "aic", "ks_pvalue"]])
            best = res.iloc[0]; best_name = best["distribution"]
            st.success(f"Terbaik: **{best_name}** | KS p-value: {best['ks_pvalue']:.4f}")
            st.info(f"Saran outlier: {recommend_outlier_method(skew, best_name)}")
            try:
                name_map = {"Normal": ss.norm, "Lognormal": ss.lognorm, "Exponential": ss.expon,
                            "Gamma": ss.gamma, "Weibull": ss.weibull_min, "Laplace": ss.laplace, "StudentT": ss.t}
                dist = name_map[best_name]; params = best["params"]
                xs = np.linspace(np.nanmin(x), np.nanmax(x), 400); pdf = dist.pdf(xs, *params)
                fig = plt.figure(); plt.hist(x, bins=40, density=True, alpha=0.6); plt.plot(xs, pdf)
                plt.title(f"Histogram + PDF {best_name}"); st.pyplot(fig)
            except Exception:
                pass
    else:
        st.warning("SciPy tidak tersedia. Menampilkan heuristik (skew/kurtosis) saja.")

# outlier handling
st.subheader("4.5) Outlier Handling (opsional)")
out_cols = st.multiselect("Kolom untuk deteksi outlier", options=num_like_cols,
                          default=[target_col] if target_col in num_like_cols else [])
method = st.selectbox("Metode", ["auto", "iqr", "zscore", "log_iqr", "quantile"], 0)
c5, c6, c7 = st.columns(3)
with c5: iqr_k = st.number_input("IQR k", 0.5, 5.0, 1.5, 0.1)
with c6: z_k   = st.number_input("Z-score k", 1.0, 10.0, 3.0, 0.5)
with c7:
    q_low  = st.number_input("Quantile low", 0.0, 0.49, 0.01, 0.01, format="%.2f")
    q_high = st.number_input("Quantile high", 0.51, 1.0, 0.99, 0.01, format="%.2f")
action = st.selectbox("Aksi", ["Drop rows outlier", "Winsorize (clip)"], 0)

# siapkan X,y
df = df0.copy()
y = pd.to_numeric(df[target_col], errors="coerce"); mask = y.notna()
df = df.loc[mask].reset_index(drop=True); y = y.loc[mask].reset_index(drop=True)

for c in numeric_feats:
    if c in df.columns and (auto_num_convert or (c in force_currency_cols)):
        df[c] = try_convert_numeric_series(df[c], force_currency=(c in force_currency_cols))

if out_cols:
    full_mask_out = pd.Series(False, index=df.index); bounds_info = {}
    for c in out_cols:
        ser_num = try_convert_numeric_series(df[c], force_currency=True); df[c] = ser_num
        m_out, (lo, hi) = detect_outlier_mask_series(ser_num, method=method, iqr_k=iqr_k, z_k=z_k, q_low=q_low, q_high=q_high)
        bounds_info[c] = (lo, hi, int(m_out.sum()))
        full_mask_out = full_mask_out | m_out.reindex(df.index, fill_value=False)
    st.caption("Ringkasan batas outlier (skala asli):")
    st.write(pd.DataFrame([{"kolom": k, "lower": v[0], "upper": v[1], "n_outlier": v[2]} for k, v in bounds_info.items()]))
    if action == "Drop rows outlier":
        keep_mask = ~full_mask_out
        df = df.loc[keep_mask].reset_index(drop=True)
        y  = y.loc[keep_mask].reset_index(drop=True)
    else:
        for c in out_cols:
            lo, hi, _ = bounds_info[c]
            if np.isfinite(lo) and np.isfinite(hi):
                df[c] = winsorize_series(df[c], lo, hi)

if len(df) < 10:
    st.error("Data terlalu sedikit (≥ 10 baris).")
    st.stop()


# =========================
# 5) Pilih Algoritma & Hyperparameter
# =========================
st.subheader("5) Pilih Algoritma & Hyperparameter")

algo_options = ["RandomForest", "LinearRegression", "ElasticNet", "SVR", "KNN"]
if XGB_OK:  algo_options.append("XGBoost")
if LGBM_OK: algo_options.append("LightGBM")
if CAT_OK:  algo_options.append("CatBoost")

algo = st.selectbox("Algoritma", options=algo_options, index=0)

hp_cols = st.columns(3)
params = {}

if algo == "RandomForest":
    with hp_cols[0]:
        params["n_estimators"] = st.slider("n_estimators", 100, 1500, 600, 50)
        max_depth_opt = st.selectbox("max_depth", ["None", "10", "20", "30", "50"], 0)
        params["max_depth"] = None if max_depth_opt == "None" else int(max_depth_opt)
    with hp_cols[1]:
        params["max_features"] = st.selectbox("max_features", ["sqrt", "log2", "1.0"], 0)
        params["min_samples_split"] = st.selectbox("min_samples_split", [2, 5, 10, 20], 0)
    with hp_cols[2]:
        params["min_samples_leaf"] = st.selectbox("min_samples_leaf", [1, 2, 4, 8], 0)

elif algo == "LinearRegression":
    st.caption("Linear Regression tidak punya hyperparameter penting. Disarankan scaling fitur.")

elif algo == "ElasticNet":
    with hp_cols[0]:
        params["alpha"] = st.number_input("alpha", 0.0001, 100.0, 1.0, 0.1)
    with hp_cols[1]:
        params["l1_ratio"] = st.slider("l1_ratio (0=L2, 1=L1)", 0.0, 1.0, 0.5, 0.05)
    with hp_cols[2]:
        params["max_iter"] = st.number_input("max_iter", 100, 20000, 5000, 100)

elif algo == "SVR":
    with hp_cols[0]:
        params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly"], 0)
    with hp_cols[1]:
        params["C"] = st.number_input("C", 0.01, 10000.0, 10.0, 0.5)
    with hp_cols[2]:
        params["epsilon"] = st.number_input("epsilon", 0.0, 10.0, 0.2, 0.05)

elif algo == "KNN":
    with hp_cols[0]:
        params["n_neighbors"] = st.slider("n_neighbors", 1, 50, 7, 1)
    with hp_cols[1]:
        params["weights"] = st.selectbox("weights", ["uniform", "distance"], 1)
    with hp_cols[2]:
        params["p"] = st.selectbox("p (1=Manhattan, 2=Euclidean)", [1, 2], 1)

elif algo == "XGBoost" and XGB_OK:
    with hp_cols[0]:
        params["n_estimators"] = st.slider("n_estimators", 100, 3000, 800, 50)
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.05, 0.01)
    with hp_cols[1]:
        params["max_depth"] = st.slider("max_depth", 1, 20, 8, 1)
        params["subsample"] = st.number_input("subsample", 0.5, 1.0, 0.9, 0.05)
    with hp_cols[2]:
        params["colsample_bytree"] = st.number_input("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
        params["reg_lambda"] = st.number_input("reg_lambda", 0.0, 100.0, 1.0, 0.5)
        params["min_child_weight"] = st.number_input("min_child_weight", 0.0, 20.0, 1.0, 0.5)

elif algo == "LightGBM" and LGBM_OK:
    with hp_cols[0]:
        params["n_estimators"] = st.slider("n_estimators", 100, 5000, 1000, 100)
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.05, 0.01)
    with hp_cols[1]:
        params["num_leaves"] = st.slider("num_leaves", 8, 512, 64, 4)
        params["max_depth"] = st.slider("max_depth (-1=auto)", -1, 32, -1, 1)
    with hp_cols[2]:
        params["subsample"] = st.number_input("subsample", 0.5, 1.0, 0.9, 0.05)
        params["colsample_bytree"] = st.number_input("colsample_bytree", 0.5, 1.0, 0.9, 0.05)
        params["reg_lambda"] = st.number_input("reg_lambda", 0.0, 100.0, 1.0, 0.5)
        params["min_child_samples"] = st.slider("min_child_samples", 1, 200, 20, 1)

elif algo == "CatBoost" and CAT_OK:
    with hp_cols[0]:
        params["iterations"] = st.slider("iterations", 200, 5000, 1000, 100)
        params["learning_rate"] = st.number_input("learning_rate", 0.001, 1.0, 0.05, 0.01)
    with hp_cols[1]:
        params["depth"] = st.slider("depth", 2, 12, 8, 1)
    with hp_cols[2]:
        params["l2_leaf_reg"] = st.number_input("l2_leaf_reg", 0.0, 50.0, 3.0, 0.5)

# split set (PASTIKAN hanya fitur terpilih)
c8, c9 = st.columns(2)
with c8:
    test_size    = st.slider("Test size", 0.1, 0.5, 0.2, 0.05)
with c9:
    random_state = st.number_input("random_state", 0, 9999, 42, 1)

X_train, X_test, y_train, y_test = train_test_split(
    df[chosen_feats], y, test_size=float(test_size), random_state=int(random_state)
)

# =========================
# ColumnTransformer
# =========================
transformers = []
freq_step_names: Dict[str, str] = {}
addr_tok_step_names: Dict[str, str] = {}

if numeric_feats:
    transformers.append(("num", SimpleImputer(strategy="median"), numeric_feats))

if onehot_feats:
    transformers.append((
        "catOneHot",
        Pipeline(steps=[("imp", SimpleImputer(strategy="most_frequent")),
                        ("ohe", make_ohe())]),
        onehot_feats
    ))

for c in freq_feats:
    safe = slugify_name(c)
    name = f"freq_{safe}"
    transformers.append((name, FrequencyEncoder(c), [c]))
    freq_step_names[c] = name

for c in addr_feats:
    safe = slugify_name(c)
    tok_name  = f"addrTok_{safe}"
    freq_name = f"addrFreq_{safe}"
    transformers.append((tok_name,  AddressTopTokens(c, top_n=int(top_n_addr)), [c]))
    transformers.append((freq_name, FrequencyEncoder(c), [c]))
    addr_tok_step_names[c]  = tok_name

preprocess = ColumnTransformer(transformers=transformers, remainder="drop")

# =========================
# Build estimator sesuai pilihan
# =========================
def build_estimator(name: str, params: Dict, random_state: int):
    if name == "RandomForest":
        max_features = 1.0 if params.get("max_features") == "1.0" else params.get("max_features", "sqrt")
        return RandomForestRegressor(
            n_estimators=int(params.get("n_estimators", 600)),
            max_depth=params.get("max_depth", None),
            max_features=max_features,
            min_samples_split=int(params.get("min_samples_split", 2)),
            min_samples_leaf=int(params.get("min_samples_leaf", 1)),
            n_jobs=-1,
            random_state=int(random_state),
        )
    if name == "LinearRegression":
        return LinearRegression()
    if name == "ElasticNet":
        return ElasticNet(alpha=float(params.get("alpha", 1.0)),
                          l1_ratio=float(params.get("l1_ratio", 0.5)),
                          max_iter=int(params.get("max_iter", 5000)),
                          random_state=int(random_state))
    if name == "SVR":
        return SVR(kernel=params.get("kernel", "rbf"),
                   C=float(params.get("C", 10.0)),
                   epsilon=float(params.get("epsilon", 0.2)))
    if name == "KNN":
        return KNeighborsRegressor(n_neighbors=int(params.get("n_neighbors", 7)),
                                   weights=params.get("weights", "distance"),
                                   p=int(params.get("p", 2)))
    if name == "XGBoost" and XGB_OK:
        return XGBRegressor(
            n_estimators=int(params.get("n_estimators", 800)),
            max_depth=int(params.get("max_depth", 8)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            min_child_weight=float(params.get("min_child_weight", 1.0)),
            n_jobs=-1,
            random_state=int(random_state),
            tree_method="hist",
            objective="reg:squarederror",
        )
    if name == "LightGBM" and LGBM_OK:
        return LGBMRegressor(
            n_estimators=int(params.get("n_estimators", 1000)),
            num_leaves=int(params.get("num_leaves", 64)),
            max_depth=int(params.get("max_depth", -1)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            subsample=float(params.get("subsample", 0.9)),
            colsample_bytree=float(params.get("colsample_bytree", 0.9)),
            reg_lambda=float(params.get("reg_lambda", 1.0)),
            min_child_samples=int(params.get("min_child_samples", 20)),
            random_state=int(random_state),
            n_jobs=-1,
        )
    if name == "CatBoost" and CAT_OK:
        return CatBoostRegressor(
            iterations=int(params.get("iterations", 1000)),
            learning_rate=float(params.get("learning_rate", 0.05)),
            depth=int(params.get("depth", 8)),
            l2_leaf_reg=float(params.get("l2_leaf_reg", 3.0)),
            random_seed=int(random_state),
            loss_function="RMSE",
            verbose=False
        )
    raise ValueError("Algoritma tidak tersedia atau dependency belum terpasang.")

# Apakah butuh scaling?
ALGOS_NEED_SCALING = {"LinearRegression", "ElasticNet", "SVR", "KNN"}
scaler_step = ("scale", StandardScaler(with_mean=False)) if algo in ALGOS_NEED_SCALING else None

est = build_estimator(algo, params, random_state=int(random_state))
steps = [("prep", preprocess)]
if scaler_step is not None:
    steps.append(scaler_step)
steps.append(("reg", est))
model = Pipeline(steps=steps)

# =========================
# Train & Evaluasi
# =========================
if st.button("🚀 Latih Model"):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    r2  = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = compute_rmse(y_test, y_pred)

    st.subheader("📊 Hasil Evaluasi (Test Set)")
    a, b, c = st.columns(3)
    a.metric("R²", f"{r2:.4f}")
    b.metric("MAE", f"{mae:,.0f}")
    c.metric("RMSE", f"{rmse:,.0f}")

    # Scatter Prediksi vs Aktual
    st.markdown("#### Prediksi vs Aktual")
    fig = plt.figure()
    plt.scatter(y_test, y_pred, alpha=0.6)
    lo = float(min(np.min(y_test), np.min(y_pred)))
    hi = float(max(np.max(y_test), np.max(y_pred)))
    plt.plot([lo, hi], [lo, hi])
    plt.xlabel("Aktual"); plt.ylabel("Prediksi"); plt.title("Prediksi vs Aktual")
    st.pyplot(fig)

    # Feature Importance / Coefficients (jika ada)
    st.subheader("🔎 Pentingnya Fitur (jika tersedia)")
    fi = None
    try:
        reg = model.named_steps["reg"]
        feat_names: List[str] = []
        if numeric_feats:
            feat_names.extend(numeric_feats)
        if onehot_feats:
            ohe = model.named_steps["prep"].named_transformers_["catOneHot"].named_steps["ohe"]
            try:
                ohe_names = list(ohe.get_feature_names_out(onehot_feats))
            except Exception:
                cats = sum([list(cat) for cat in getattr(ohe, "categories_", [])], [])
                ohe_names = [f"ohe_{i}_{v}" for i, v in enumerate(cats)]
            feat_names.extend(ohe_names)
        for c_ in freq_feats:
            feat_names.append(f"{c_}__freq")
        for c_ in addr_feats:
            tok_name = f"addrTok_{slugify_name(c_)}"
            if tok_name in model.named_steps["prep"].named_transformers_:
                tok_tr: AddressTopTokens = model.named_steps["prep"].named_transformers_[tok_name]
                try:
                    feat_names.extend(list(tok_tr.get_feature_names_out()))
                except Exception:
                    pass
            feat_names.append(f"{c_}__freq")

        if hasattr(reg, "feature_importances_"):
            importances = reg.feature_importances_
            if len(feat_names) != len(importances):
                feat_names = [f"feat_{i}" for i in range(len(importances))]
            fi = pd.DataFrame({"feature": feat_names, "importance": importances}).sort_values(
                "importance", ascending=False).reset_index(drop=True)
            st.dataframe(fi.head(100), use_container_width=True)
            fig2 = plt.figure()
            top = fi.head(25).iloc[::-1]
            plt.barh(top["feature"], top["importance"])
            plt.title("Top-25 Feature Importance")
            st.pyplot(fig2)

        elif hasattr(reg, "coef_"):
            coefs = np.ravel(reg.coef_)
            absw = np.abs(coefs)
            if len(feat_names) != len(coefs):
                feat_names = [f"feat_{i}" for i in range(len(coefs))]
            fi = pd.DataFrame({"feature": feat_names, "weight": coefs, "abs_weight": absw}) \
                   .sort_values("abs_weight", ascending=False).reset_index(drop=True)
            st.dataframe(fi.head(100), use_container_width=True)
            fig2 = plt.figure()
            top = fi.head(25).iloc[::-1]
            plt.barh(top["feature"], top["abs_weight"])
            plt.title("Top-25 | |coef|")
            st.pyplot(fig2)
        else:
            st.info("Algoritma ini tidak menyediakan importance/coef (mis. KNN).")

    except Exception as e:
        st.info(f"Tidak bisa menghitung importance/coef: {e}")

    # ================= SAVE MODEL + CONFIG + BUNDLE =================
    os.makedirs("models", exist_ok=True)

    # Ambil kategori OHE & nilai populer untuk form end-user
    ohe_cats_map = {}
    if onehot_feats:
        try:
            ohe = model.named_steps["prep"].named_transformers_["catOneHot"].named_steps["ohe"]
            for col, cats in zip(onehot_feats, ohe.categories_):
                ohe_cats_map[col] = [str(c) for c in list(cats)[:200]]
        except Exception:
            pass

    freq_top_map = {}
    prep = model.named_steps["prep"].named_transformers_
    for c_ in freq_feats:
        step_name = f"freq_{slugify_name(c_)}"
        if step_name in prep:
            enc: FrequencyEncoder = prep[step_name]
            if hasattr(enc, "freq_map_") and enc.freq_map_ is not None:
                freq_top_map[c_] = [str(v) for v in list(enc.freq_map_.index[:200])]

    # daftar fitur mentah (input) untuk end-user
    features_in = list(dict.fromkeys(list(numeric_feats) + list(onehot_feats) + list(freq_feats) + list(addr_feats)))

    # kolom teks lokasi umum (opsional, untuk normalisasi ringan di end-user app)
    canon_text_cols = [c for c in features_in if any(x in c.lower() for x in
                        ["alamat","provinsi","kota","kota_kab","kota_kabupaten","kecamatan","kelurahan","kode_pos"])]

    # kolom numerik yang boleh dipaksa numeric saat inferensi
    force_numeric_cols = [c for c in features_in if c.lower() in ["luas","jarak_cbd","latitude","longitude"]]

    # simpan model & config (legacy terpisah)
    import joblib
    joblib.dump(model, "models/model_latest.pkl")
    feature_config = {
        "target_col": target_col,
        "numeric_feats": numeric_feats,
        "onehot_feats": onehot_feats,
        "freq_feats": freq_feats,
        "addr_feats": addr_feats,
        "top_n_addr": int(top_n_addr),
        "ohe_categories": ohe_cats_map,
        "freq_top_values": freq_top_map,
        "algo": algo,
        "params": params
    }
    with open("models/config_latest.json", "w", encoding="utf-8") as f:
        json.dump(feature_config, f, ensure_ascii=False, indent=2)

    # bundle untuk end-user (single file)
    bundle = {
        "pipeline": model,
        "config": {
            "target_col": target_col,
            "target_is_per_m2": bool(target_is_per_m2),
            "features_in": features_in,
            "numeric_feats": numeric_feats,
            "onehot_feats": onehot_feats,
            "freq_feats": freq_feats,
            "addr_feats": addr_feats,
            "ohe_categories": ohe_cats_map,
            "freq_top_values": freq_top_map,
            "canon_text_cols": canon_text_cols,
            "force_numeric_cols": force_numeric_cols,
            "synonyms": {}
        }
    }
    joblib.dump(bundle, "models/model_bundle_latest.pkl")
    st.success("✅ Disimpan: models/model_latest.pkl, models/config_latest.json, dan models/model_bundle_latest.pkl")

    # tombol unduhan
    out = pd.DataFrame({"y_true": y_test.values, "y_pred": y_pred, "residual": y_test.values - y_pred})
    try:
        import openpyxl
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            out.to_excel(w, sheet_name="Prediksi_Test", index=False)
            if fi is not None:
                fi.to_excel(w, sheet_name="Feature_Importance", index=False)
        st.download_button(
            "⬇️ Download (Excel Hasil Test)",
            data=buf.getvalue(),
            file_name="training_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
    except Exception:
        st.download_button("⬇️ Prediksi (CSV)", data=out.to_csv(index=False).encode("utf-8"),
                           file_name="prediksi_test.csv", mime="text/csv")

    # download bundle pkl langsung
    try:
        with open("models/model_bundle_latest.pkl", "rb") as f:
            st.download_button("⬇️ Download model_bundle_latest.pkl", data=f.read(),
                               file_name="model_bundle_latest.pkl", mime="application/octet-stream")
    except Exception:
        pass

st.caption("Catatan: Outlier & diagnostik distribusi tetap seperti versi Anda. "
           "App ini juga menyimpan single-file bundle untuk dipakai end-user.")
