"""
Investigación + saneamiento de features (v2)
- Detecta features binarias y NO las escala
- Detecta features con IQR ~ 0 (scale muy pequeña)
- Opcional: log1p + clip para features con outliers
- Recompone tensores: [continuas_escaladas + binarias_sin_escalar]
"""

from datetime import datetime
import numpy as np

from mimo_old.data_pipeline_v4 import DataPipeline
from mimo_old.model_builder import Config, ModelConfig
from mimo_old.feature_builder import FeatureConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.data_manager import DataManager
from mimo_old.databases import Database


# -----------------------------
# Utils
# -----------------------------
def safe_feature_names(pipeline, side: str, block: str, n_features: int):
    """
    Intenta obtener nombres reales de features.
    Ajusta aquí si tu pipeline expone otra propiedad.
    """
    # 1) pipeline.feature_names[side][block]
    try:
        names = pipeline.feature_names[side][block]
        if names and len(names) == n_features:
            return list(names)
    except Exception:
        pass

    # 2) pipeline.get_feature_names(side, block)
    try:
        names = pipeline.get_feature_names(side=side, block=block)
        if names and len(names) == n_features:
            return list(names)
    except Exception:
        pass

    # 3) feature_config quizá tenga algo
    # (no asumo nada: fallback)
    return [f"feat_{i:02d}" for i in range(n_features)]


def is_binary_feature(flat_values: np.ndarray, tol=1e-12) -> bool:
    """
    Considera binaria si solo toma valores en {0,1} (con tolerancia).
    """
    if flat_values.size == 0:
        return False
    vals = flat_values[np.isfinite(flat_values)]
    if vals.size == 0:
        return False
    # redondeo suave por si hay floats 0.0/1.0 con ruido numérico mínimo
    v = np.unique(np.round(vals, 6))
    if v.size <= 2 and np.all(np.isin(v, [0.0, 1.0])):
        return True
    return False


def robust_iqr(flat_values: np.ndarray, q_low=25.0, q_high=75.0):
    vals = flat_values[np.isfinite(flat_values)]
    if vals.size == 0:
        return np.nan
    q1 = np.percentile(vals, q_low)
    q3 = np.percentile(vals, q_high)
    return float(q3 - q1)


def summarize_feature(flat_values: np.ndarray):
    vals = flat_values[np.isfinite(flat_values)]
    if vals.size == 0:
        return dict(mean=np.nan, std=np.nan, min=np.nan, max=np.nan)
    return dict(
        mean=float(vals.mean()),
        std=float(vals.std()),
        min=float(vals.min()),
        max=float(vals.max()),
    )


def apply_clip_and_log1p(X: np.ndarray, idxs, clip_abs=None, do_log1p=False):
    """
    X: (samples, seq_len, features)
    idxs: list of feature indices to transform
    """
    if not idxs:
        return X

    X = X.copy()
    for j in idxs:
        v = X[:, :, j]
        if clip_abs is not None:
            v = np.clip(v, -clip_abs, clip_abs)
        if do_log1p:
            # log1p preservando signo
            v = np.sign(v) * np.log1p(np.abs(v))
        X[:, :, j] = v
    return X


# -----------------------------
# Config
# -----------------------------
print("\n" + "=" * 70)
print("INVESTIGACIÓN: Stats + Saneamiento de Features (v2)")
print("=" * 70)

general = Config(release='test', use_oof=False)

feature_config = FeatureConfig(
    ema_periods=[9, 21, 50],
    label_horizon=5,
    tp_barrier=2.5,
    sl_barrier=1.0,
    feature_masks={
        'long': {'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True,
                 'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False},
        'short': {'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True,
                  'ema_bull': False, 'rsi_oversold': False, 'macd_positive': False},
    },
    label_method='adaptive'
)

model_config = ModelConfig(seq_len_short=64, seq_len_long=256)
regime_config = RegimeConfig()

pipeline = DataPipeline(general, feature_config, model_config, regime_config)

# -----------------------------
# Load data
# -----------------------------
print("\n1️⃣ Cargando datos...")
db = Database()
dm = DataManager.from_database_historical_2(db, datetime(2025, 12, 1), datetime(2025, 12, 7))
df = dm.df.head(5000)
print(f"   Cargados: {len(df):,} filas")

# -----------------------------
# Prepare sequences + fit scalers
# -----------------------------
print("\n2️⃣ Preparando datos y creando secuencias (fit_scalers=True)...")
df_prep = pipeline.prepare_data(df.copy(), labels=False, side='long')
sequences = pipeline.create_sequences_by_side(df_prep, sides=('long',), fit_scalers=True, train=False)

# Cogemos seq_short (long) como en tu script
X = sequences['long']['seq_short']  # (samples, seq_len, features)
print(f"   X original: shape={X.shape}")

n_samples, seq_len, n_features = X.shape
feat_names = safe_feature_names(pipeline, side='long', block='seq_short', n_features=n_features)

# -----------------------------
# 3) Diagnóstico previo: binarias, IQR, outliers
# -----------------------------
print("\n3️⃣ Diagnóstico previo (antes de saneamiento/reescalado)")
print("=" * 70)

iqr_eps = 1e-6            # umbral para considerar IQR ~ 0
extreme_abs = 1000.0      # umbral de "outlier gigante" en datos ya escalados (ajusta si quieres)
raw_extreme_abs = 5e6     # umbral “monstruo” típico si te entra algo como 50M

binary_idx = []
tiny_iqr_idx = []
huge_range_idx = []

for j in range(n_features):
    flat = X[:, :, j].reshape(-1)
    stats = summarize_feature(flat)
    iqr = robust_iqr(flat)

    # binaria por valores (0/1)
    if is_binary_feature(flat):
        binary_idx.append(j)

    # IQR casi cero
    if np.isfinite(iqr) and iqr < iqr_eps:
        tiny_iqr_idx.append(j)

    # rangos enormes
    if np.isfinite(stats["max"]) and (abs(stats["max"]) > raw_extreme_abs or abs(stats["min"]) > raw_extreme_abs):
        huge_range_idx.append(j)

print(f"   Binarias detectadas: {len(binary_idx)}")
if binary_idx:
    print("   Ejemplos binarias:", [(i, feat_names[i]) for i in binary_idx[:10]])

print(f"   IQR ~ 0 (scale casi cero): {len(tiny_iqr_idx)}")
if tiny_iqr_idx:
    print("   Ejemplos IQR~0:", [(i, feat_names[i]) for i in tiny_iqr_idx[:10]])

print(f"   Rangos monstruo (>|{raw_extreme_abs:g}|): {len(huge_range_idx)}")
if huge_range_idx:
    print("   Ejemplos rango monstruo:", [(i, feat_names[i]) for i in huge_range_idx[:10]])

# Índices continuos = todos menos binarias
cont_idx = [i for i in range(n_features) if i not in set(binary_idx)]
print(f"\n   Continuas: {len(cont_idx)}  |  Binarias: {len(binary_idx)}")

# -----------------------------
# 4) Transformaciones defensivas (opcionales)
# -----------------------------
print("\n4️⃣ Transformaciones defensivas (clip/log1p) para features problemáticas")
print("=" * 70)

# Heurística: si hay rangos monstruo, aplicar log1p + clip a esas features (solo continuas)
problem_idx = sorted(set(huge_range_idx + tiny_iqr_idx) - set(binary_idx))
print(f"   Features problemáticas (no binarias): {len(problem_idx)}")
if problem_idx:
    print("   Ejemplos:", [(i, feat_names[i]) for i in problem_idx[:10]])

# Ajusta estos flags a tu gusto:
DO_CLIP = True
CLIP_ABS = 1e6       # si te entran 50M, esto evita saturaciones brutales antes del log
DO_LOG1P = True      # log1p reduce órdenes de magnitud manteniendo signo

X_sane = X
if problem_idx:
    X_sane = apply_clip_and_log1p(X_sane, idxs=problem_idx, clip_abs=(CLIP_ABS if DO_CLIP else None), do_log1p=DO_LOG1P)

# -----------------------------
# 5) Reescalado: SOLO continuas (binarias se quedan tal cual)
# -----------------------------
print("\n5️⃣ Reescalado: solo continuas, binarios sin escalar")
print("=" * 70)

# Tomamos el scaler seq_short ya fitted del pipeline
scaler_seq_short = pipeline.scalers.get("seq_short", None)
if scaler_seq_short is None:
    raise RuntimeError("No encuentro pipeline.scalers['seq_short']")

X_cont = X_sane[:, :, cont_idx]
X_bin = X_sane[:, :, binary_idx] if binary_idx else None

# Reescalar continuas usando el scaler existente
# Nota: asumo que tu scaler acepta tensores 3D o al menos 2D.
# Si tu scaler espera 2D, hacemos reshape.
try:
    X_cont_scaled = scaler_seq_short.transform(X_cont)
except Exception:
    # fallback 2D
    X2 = X_cont.reshape(-1, X_cont.shape[-1])
    X2s = scaler_seq_short.transform(X2)
    X_cont_scaled = X2s.reshape(n_samples, seq_len, -1)

# Recomponer con orden: [continuas_escaladas + binarias_originales]
if X_bin is not None:
    X_final = np.concatenate([X_cont_scaled, X_bin], axis=-1)
    final_names = [feat_names[i] for i in cont_idx] + [feat_names[i] for i in binary_idx]
else:
    X_final = X_cont_scaled
    final_names = [feat_names[i] for i in cont_idx]

print(f"   X_final: shape={X_final.shape}")
print(f"   Orden final: continuas({len(cont_idx)}) + binarias({len(binary_idx)})")

# -----------------------------
# 6) Stats finales
# -----------------------------
print("\n6️⃣ Stats finales (X_final)")
print("=" * 70)

def check_feature(j, name, flat):
    s = summarize_feature(flat)
    warnings = []
    if np.isfinite(s["mean"]) and abs(s["mean"]) > 0.5:
        warnings.append(f"mean desviada ({s['mean']:.3f})")
    if np.isfinite(s["std"]) and s["std"] < 0.1:
        warnings.append(f"std muy pequeña ({s['std']:.3f})")
    if np.isfinite(s["min"]) and (s["min"] < -5 or s["max"] > 5):
        warnings.append(f"valores extremos [{s['min']:.2f}, {s['max']:.2f}]")

    status = "⚠️ " if warnings else "✅"
    msg = f"{status} {j:2d} {name:30s} | mean={s['mean']:7.3f} std={s['std']:6.3f} range=[{s['min']:7.2f},{s['max']:7.2f}]"
    if warnings:
        msg += "  -> " + ", ".join(warnings)
    print(msg)

# Muestra todas o solo las problemáticas:
SHOW_ALL = True

bad_after = []
for j in range(X_final.shape[-1]):
    flat = X_final[:, :, j].reshape(-1)
    s = summarize_feature(flat)
    is_bad = (
        (np.isfinite(s["mean"]) and abs(s["mean"]) > 0.5) or
        (np.isfinite(s["std"]) and s["std"] < 0.1) or
        (np.isfinite(s["min"]) and (s["min"] < -5 or s["max"] > 5))
    )
    if is_bad:
        bad_after.append(j)

    if SHOW_ALL or is_bad:
        check_feature(j, final_names[j], flat)

print("\n" + "=" * 70)
print("RESUMEN")
print("=" * 70)
print(f"- Binarias detectadas (sin escalar): {len(binary_idx)}")
print(f"- Continuas escaladas: {len(cont_idx)}")
print(f"- Features problemáticas pre-saneamiento: {len(problem_idx)}")
print(f"- Features con warnings tras recomposición: {len(bad_after)}")
if bad_after:
    print("  Ejemplos warnings post:", [(i, final_names[i]) for i in bad_after[:15]])

print("\n✅ Fin.\n")
