"""
Script para ver stats de scalers - VERSIÓN CORREGIDA
"""
from mimo_old.data_pipeline_v4 import DataPipeline
from mimo_old.model_builder import Config, ModelConfig
from mimo_old.feature_builder import FeatureConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.data_manager import DataManager
from mimo_old.databases import Database
from datetime import datetime
import numpy as np

print("\n" + "=" * 70)
print("INVESTIGACIÓN: Stats de Scalers")
print("=" * 70)

# Tu config
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

# Cargar datos
print("\n1️⃣ Cargando datos...")
db = Database()
dm = DataManager.from_database_historical_2(db, datetime(2025, 12, 1), datetime(2025, 12, 7))
df = dm.df.head(5000)
print(f"   Cargados: {len(df):,} filas")

# Preparar y escalar
print("\n2️⃣ Preparando datos y escalando...")
df_prep = pipeline.prepare_data(df.copy(), labels=False, side='long')
sequences = pipeline.create_sequences_by_side(df_prep, sides=('long',), fit_scalers=True, train=False)

# Ver stats de scalers
print("\n3️⃣ Stats de scalers:")
print("=" * 70)

for name, scaler in pipeline.scalers.items():
    print(f"\n📊 {name.upper()}")
    print("-" * 70)

    # Ver tipo
    print(f"   Tipo: {type(scaler).__name__}")

    # Ver atributos disponibles
    if hasattr(scaler, 'get_stats'):
        stats = scaler.get_stats()
        print(f"   Keys disponibles en stats: {list(stats.keys())}")
        print(f"   Stats completos:")
        for key, value in stats.items():
            if isinstance(value, (list, np.ndarray)):
                if len(value) <= 5:
                    print(f"      {key}: {value}")
                else:
                    print(f"      {key}: array de {len(value)} elementos")
                    # Mostrar estadísticas
                    arr = np.array(value)
                    print(f"         min={arr.min():.4f}, max={arr.max():.4f}, "
                          f"mean={arr.mean():.4f}, std={arr.std():.4f}")
            else:
                print(f"      {key}: {value}")

    # Verificar si tiene median_ y scale_ (sklearn style)
    if hasattr(scaler, 'center_'):
        print(f"   Center_ (median): shape {scaler.center_.shape}")
        print(f"      min={scaler.center_.min():.4f}, max={scaler.center_.max():.4f}, "
              f"mean={scaler.center_.mean():.4f}")

    if hasattr(scaler, 'scale_'):
        print(f"   Scale_ (IQR): shape {scaler.scale_.shape}")
        print(f"      min={scaler.scale_.min():.4f}, max={scaler.scale_.max():.4f}, "
              f"mean={scaler.scale_.mean():.4f}")

        # Features con scale pequeña
        small_scale_idx = np.where(scaler.scale_ < 0.1)[0]
        if len(small_scale_idx) > 0:
            print(f"   ⚠️  {len(small_scale_idx)} features con scale < 0.1:")
            print(f"      Indices: {small_scale_idx[:10]}...")  # Mostrar primeros 10
            print(f"      Valores: {scaler.scale_[small_scale_idx[:10]]}")

    # Ver warmup status
    if hasattr(scaler, 'warmup_status'):
        print(f"   Warmup: {scaler.warmup_status():.1%}")

# Ver datos escalados
print("\n4️⃣ Análisis de datos escalados:")
print("=" * 70)

X_long = sequences['long']['seq_short']  # shape: (samples, seq_len, features)

print(f"\nseq_short shape: {X_long.shape}")
print(f"   samples: {X_long.shape[0]}")
print(f"   seq_len: {X_long.shape[1]}")
print(f"   features: {X_long.shape[2]}")

# Estadísticas por feature (sobre todos los samples y timesteps)
for feat_idx in range(X_long.shape[2]):
    feat_data = X_long[:, :, feat_idx].flatten()

    mean = feat_data.mean()
    std = feat_data.std()
    min_val = feat_data.min()
    max_val = feat_data.max()

    # Detectar anomalías
    warnings = []
    if abs(mean) > 0.5:
        warnings.append(f"mean desviada ({mean:.3f})")
    if std < 0.1:
        warnings.append(f"std muy pequeña ({std:.3f})")
    if min_val < -5 or max_val > 5:
        warnings.append(f"valores extremos [{min_val:.2f}, {max_val:.2f}]")

    status = "⚠️ " if warnings else "✅"

    print(f"{status} Feature {feat_idx:2d}: "
          f"mean={mean:6.3f}, std={std:.3f}, range=[{min_val:6.2f}, {max_val:6.2f}]",
          end="")

    if warnings:
        print(f" → {', '.join(warnings)}")
    else:
        print()

print("\n" + "=" * 70)
print("INTERPRETACIÓN")
print("=" * 70)

print("""
✅ Normal:
   - Mean entre -0.3 y 0.3
   - Std entre 0.5 y 2.0
   - Range entre -4 y 4

⚠️  Revisar:
   - Mean > 0.5 o < -0.5 (distribución asimétrica)
   - Std < 0.1 (poca variabilidad, puede ser feature constante)
   - Range > 5 o < -5 (outliers extremos)

📌 Recordar:
   - Features binarias (0/1) tendrán stats diferentes
   - Features ya normalizadas pueden tener scale pequeña
   - Outliers son normales en RobustScaler
""")

print("=" * 70 + "\n")