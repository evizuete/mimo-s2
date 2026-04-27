"""
Diagnóstico de umbrales de régimen de volatilidad.

Compara la distribución actual de atr_norm con los umbrales fijados
durante el entrenamiento del clasificador. Si la distribución del mercado
actual se ha desplazado significativamente respecto a la de entrenamiento,
los regímenes se clasificarán incorrectamente y el modelo tomará
decisiones con un contexto desfasado.

Uso:
    python check_regime_thresholds.py
"""

import pandas as pd
import numpy as np
from mimo.data_managers.databases import Database
from mimo.helpers.helper import Helper
from mimo.models.model_builder import Config

# ============================================================================
# CONFIGURACIÓN — ajustar según tu pipeline
# ============================================================================

# Periodo ATR: debe coincidir EXACTAMENTE con el que usa feature_builder.
# Si feature_builder usa SMA en lugar de EWM, cambiar el método abajo.
ATR_PERIOD = 14
ATR_METHOD = "ewm"   # "ewm" o "sma" — verificar contra feature_builder

# Umbrales de entrenamiento (del log del test / artifacts)
REGIME_THRESHOLDS = {
    "vol_low": 0.00028049118277818577,
    "vol_high": 0.0006162369578756273,
    "bb_p20": 0.0008682620379952953,
    "bb_p35": 0.0011726969841940234,
    "bb_p70": 0.0022248104016177298,
    "rexp_p80": 1.3066699153970067
  }

# Distribución esperada de regímenes si los umbrales son correctos.
# low_vol: atr_norm < vol_low  → esperado ~30% (por debajo de p30)
# normal:  vol_low <= atr_norm <= vol_high → esperado ~50%
# volatile: atr_norm > vol_high → esperado ~20% (por encima de p80)
EXPECTED_PCT_LOW_VOL  = 0.30
EXPECTED_PCT_VOLATILE = 0.20

# Margen de tolerancia: si la distribución real se desvía más de esto
# respecto a la esperada, los umbrales están desactualizados.
DRIFT_TOLERANCE = 0.15   # ±15 puntos porcentuales

LAST_N_RATES = 2048
RECENT_WINDOW = 200      # ~3h de barras M1

# ============================================================================
# CARGA DE DATOS
# ============================================================================

db = Database()
db.connect()

helper = Helper(general_config=Config(), path='.')
df = helper.load_from_database_real(db, last_n_rates=LAST_N_RATES)
df = df.sort_values('time').reset_index(drop=True)

if len(df) < RECENT_WINDOW:
    print(f"⚠️  Solo {len(df)} barras disponibles (mínimo recomendado: {RECENT_WINDOW})")

# ============================================================================
# CÁLCULO DE ATR_NORM
# ============================================================================

high = df['high']
low  = df['low']
close_prev = df['close'].shift(1)

tr = pd.concat([
    high - low,
    (high - close_prev).abs(),
    (low  - close_prev).abs(),
], axis=1).max(axis=1)

if ATR_METHOD == "ewm":
    df['atr'] = tr.ewm(span=ATR_PERIOD, adjust=False).mean()
elif ATR_METHOD == "sma":
    df['atr'] = tr.rolling(window=ATR_PERIOD, min_periods=1).mean()
else:
    raise ValueError(f"ATR_METHOD no soportado: {ATR_METHOD}")

df['atr_norm'] = df['atr'] / df['close']

# Eliminar las primeras filas donde ATR no es estable
df = df.iloc[ATR_PERIOD:].reset_index(drop=True)

vol_low  = REGIME_THRESHOLDS["vol_low"]
vol_high = REGIME_THRESHOLDS["vol_high"]

# ============================================================================
# ANÁLISIS
# ============================================================================

full    = df['atr_norm']
recent  = df['atr_norm'].tail(RECENT_WINDOW)

# Contexto temporal
time_col = 'time'
t_first  = df[time_col].iloc[0]
t_last   = df[time_col].iloc[-1]

print(f"{'='*70}")
print(f" DIAGNÓSTICO DE UMBRALES DE RÉGIMEN — XAUUSD M1")
print(f"{'='*70}")
print(f"  Rango:       {t_first}  →  {t_last}")
print(f"  Barras:      {len(df)} (total)  /  {len(recent)} (recientes)")
print(f"  ATR method:  {ATR_METHOD}(period={ATR_PERIOD})")
print()

# ── Estadísticas descriptivas ──────────────────────────────────────────
print(f"{'─'*70}")
print(f" ATR_NORM — distribución actual")
print(f"{'─'*70}")
for label, series in [("Ventana completa", full), (f"Últimas {RECENT_WINDOW}", recent)]:
    print(f"\n  {label}:")
    print(f"    min    : {series.min():.8f}")
    print(f"    p10    : {series.quantile(0.10):.8f}")
    print(f"    p30    : {series.quantile(0.30):.8f}   ← vol_low  train: {vol_low:.8f}")
    print(f"    media  : {series.mean():.8f}")
    print(f"    p80    : {series.quantile(0.80):.8f}   ← vol_high train: {vol_high:.8f}")
    print(f"    p95    : {series.quantile(0.95):.8f}")
    print(f"    max    : {series.max():.8f}")

# ── Distribución de regímenes ──────────────────────────────────────────
print(f"\n{'─'*70}")
print(f" DISTRIBUCIÓN DE REGÍMENES (según umbrales de entrenamiento)")
print(f"{'─'*70}")

results = {}
for label, series in [("Ventana completa", full), (f"Últimas {RECENT_WINDOW}", recent)]:
    pct_low  = (series < vol_low).mean()
    pct_mid  = ((series >= vol_low) & (series <= vol_high)).mean()
    pct_high = (series > vol_high).mean()
    results[label] = {"low_vol": pct_low, "normal": pct_mid, "volatile": pct_high}

    print(f"\n  {label}:")
    print(f"    low_vol  : {pct_low*100:5.1f}%   (esperado ~{EXPECTED_PCT_LOW_VOL*100:.0f}%)")
    print(f"    normal   : {pct_mid*100:5.1f}%   (esperado ~{(1 - EXPECTED_PCT_LOW_VOL - EXPECTED_PCT_VOLATILE)*100:.0f}%)")
    print(f"    volatile : {pct_high*100:5.1f}%   (esperado ~{EXPECTED_PCT_VOLATILE*100:.0f}%)")

# ── Diagnóstico ────────────────────────────────────────────────────────
print(f"\n{'─'*70}")
print(f" DIAGNÓSTICO")
print(f"{'─'*70}")

r = results["Ventana completa"]
warnings = []

# Caso 1: demasiadas barras clasificadas como volatile
drift_high = r["volatile"] - EXPECTED_PCT_VOLATILE
if drift_high > DRIFT_TOLERANCE:
    warnings.append(
        f"⚠️  VOLATILE inflado: {r['volatile']*100:.1f}% vs esperado {EXPECTED_PCT_VOLATILE*100:.0f}%"
        f" (drift +{drift_high*100:.1f}pp)\n"
        f"   → vol_high posiblemente demasiado bajo. Sugerido: {full.quantile(1 - EXPECTED_PCT_VOLATILE):.8f}"
    )

# Caso 2: demasiadas barras clasificadas como low_vol
drift_low = r["low_vol"] - EXPECTED_PCT_LOW_VOL
if drift_low > DRIFT_TOLERANCE:
    warnings.append(
        f"⚠️  LOW_VOL inflado: {r['low_vol']*100:.1f}% vs esperado {EXPECTED_PCT_LOW_VOL*100:.0f}%"
        f" (drift +{drift_low*100:.1f}pp)\n"
        f"   → vol_low posiblemente demasiado alto. Sugerido: {full.quantile(EXPECTED_PCT_LOW_VOL):.8f}"
    )

# Caso 3: casi nada cae en volatile (mercado demasiado tranquilo para los umbrales)
if r["volatile"] < 0.05:
    warnings.append(
        f"⚠️  VOLATILE casi vacío: {r['volatile']*100:.1f}% — "
        f"el modelo casi nunca verá régimen 'volatile' con estos umbrales.\n"
        f"   Posible mercado en baja volatilidad sostenida, o vol_high demasiado alto."
    )

# Caso 4: casi nada cae en low_vol
if r["low_vol"] < 0.05:
    warnings.append(
        f"⚠️  LOW_VOL casi vacío: {r['low_vol']*100:.1f}% — "
        f"el modelo casi nunca verá régimen 'low_vol' con estos umbrales.\n"
        f"   Posible mercado en alta volatilidad sostenida, o vol_low demasiado bajo."
    )

# Caso 5: divergencia fuerte entre ventana completa y reciente
r_recent = results[f"Últimas {RECENT_WINDOW}"]
for regime in ("low_vol", "volatile"):
    delta = abs(r[regime] - r_recent[regime])
    if delta > 0.25:
        warnings.append(
            f"⚠️  DIVERGENCIA temporal en '{regime}': "
            f"completa={r[regime]*100:.1f}% vs reciente={r_recent[regime]*100:.1f}% "
            f"(Δ={delta*100:.1f}pp)\n"
            f"   La volatilidad reciente difiere mucho del promedio — "
            f"posible cambio de régimen en curso."
        )

if warnings:
    for w in warnings:
        print(f"\n{w}")
    print(f"\n{'─'*70}")
    print("  ACCIÓN: considerar recalcular umbrales con datos recientes")
    print(f"          o verificar que el modelo fue entrenado con datos representativos.")
else:
    print("\n✅  Umbrales dentro de rango — distribución de regímenes coherente")
    print("    con las proporciones esperadas del entrenamiento.")

print()