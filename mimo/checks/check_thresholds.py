import pandas as pd
import numpy as np
from mimo.data_managers.databases import Database
from mimo.helpers.helper import Helper
from mimo.models.model_builder import Config

# ── 1. Cargar datos recientes ──────────────────────────────────────────
db = Database()
db.connect()

# Últimas 2048 barras (mismo lookback que en producción)
helper = Helper(general_config=Config(), path='.')
df = helper.load_from_database_real(db, last_n_rates=2048)
df = df.sort_values('time').reset_index(drop=True)

# ── 2. Calcular atr_norm (ATR / close, igual que en feature_builder) ──
# Ajusta el periodo si tu feature_builder usa uno distinto
high = df['high']
low  = df['low']
close_prev = df['close'].shift(1)

tr = pd.concat([
    high - low,
    (high - close_prev).abs(),
    (low  - close_prev).abs()
], axis=1).max(axis=1)

atr_period = 14  # ajusta si usas otro periodo
df['atr']      = tr.ewm(span=atr_period, adjust=False).mean()
df['atr_norm'] = df['atr'] / df['close']

# ── 3. Umbrales guardados en artifacts (del log del test) ─────────────
vol_low_train  = 0.00033640798037065894
vol_high_train = 0.0007045275531554512

'''
  "regime_thresholds": {
    "vol_low": 0.00033640798037065894,
    "vol_high": 0.0007045275531554512,
    "bb_p20": 0.0010437690854299328,
    "bb_p35": 0.0013935764503969223,
    "bb_p70": 0.002604547857030304,
    "rexp_p80": 1.301592092356215
  },

'''

# ── 4. Diagnóstico ────────────────────────────────────────────────────
recent_200  = df['atr_norm'].tail(200)   # últimas ~3h de mercado
recent_2048 = df['atr_norm']             # ventana completa de producción

print("=== ATR_NORM — ventana actual (2048 barras) ===")
print(f"  p30  : {recent_2048.quantile(0.30):.6f}  (vol_low  entrenamiento: {vol_low_train:.6f})")
print(f"  p80  : {recent_2048.quantile(0.80):.6f}  (vol_high entrenamiento: {vol_high_train:.6f})")
print(f"  media: {recent_2048.mean():.6f}")
print(f"  p95  : {recent_2048.quantile(0.95):.6f}")

print("\n=== ATR_NORM — últimas 200 barras (~3h) ===")
print(f"  media: {recent_200.mean():.6f}")
print(f"  p80  : {recent_200.quantile(0.80):.6f}")

pct_volatile_now = (recent_2048 > vol_high_train).mean()
pct_volatile_3h  = (recent_200  > vol_high_train).mean()
print(f"\n=== % barras clasificadas como VOLATILE ===")
print(f"  Ventana 2048: {pct_volatile_now*100:.1f}%  (esperado ~20% si umbrales correctos)")
print(f"  Últimas 200 : {pct_volatile_3h*100:.1f}%")

print("\n=== Diagnóstico ===")
if recent_2048.quantile(0.80) > vol_high_train * 1.20:
    print("⚠️  UMBRALES DESACTUALIZADOS: el p80 actual supera el vol_high de entrenamiento")
    print(f"   vol_high nuevo sugerido: {recent_2048.quantile(0.80):.6f}")
else:
    print("✅  Umbrales dentro de rango — la volatilidad actual es genuinamente alta")