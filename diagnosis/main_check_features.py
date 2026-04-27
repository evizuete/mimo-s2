"""
Test rápido de features - VERSIÓN CORREGIDA
"""
from mimo_old.data_pipeline_v4 import DataPipeline
from mimo_old.feature_builder import FeatureEngineer, FeatureConfig
from mimo_old.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.data_manager import DataManager
from mimo_old.databases import Database
from datetime import datetime
import pandas as pd

print("\n" + "="*70)
print("TEST: ¿Por qué solo 23 features?")
print("="*70)

# Tu configuración exacta
general = Config(release='test', use_oof=False)

feature_config = FeatureConfig(
    ema_periods=[9, 21, 50],
    label_horizon=5,
    tp_barrier=2.5,
    sl_barrier=1.0,
    feature_masks={
        'long': {
            'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True,
            'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False
        },
        'short': {
            'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True,
            'ema_bull': False, 'rsi_oversold': False, 'macd_positive': False
        },
    },
    label_method='adaptive'
)

model_config = ModelConfig(seq_len_short=64, seq_len_long=256)
regime_config = RegimeConfig(adx_trend_threshold=25.0)

# Crear pipeline
pipeline = DataPipeline(general, feature_config, model_config, regime_config)

# Cargar datos pequeños para test
print("\n1️⃣ Cargando datos de prueba...")
try:
    db = Database()
    dm = DataManager.from_database_historical_2(
        db,
        from_date=datetime(2025, 12, 1),
        to_date=datetime(2025, 12, 7)
    )
    df = dm.df.head(2000)
    print(f"   ✅ Cargados {len(df):,} filas")
except Exception as e:
    print(f"   ⚠️  Error cargando datos: {e}")
    print("   Creando datos sintéticos...")
    import numpy as np
    dates = pd.date_range('2025-01-01', periods=2000, freq='1min')
    df = pd.DataFrame({
        'time': dates,
        'open': np.random.randn(2000).cumsum() + 100,
        'high': np.random.randn(2000).cumsum() + 101,
        'low': np.random.randn(2000).cumsum() + 99,
        'close': np.random.randn(2000).cumsum() + 100,
        'volume': np.random.randint(1000, 10000, 2000)
    })

# Test LONG
print("\n2️⃣ Preparando datos LONG...")
df_long = pipeline.prepare_data(df.copy(), labels=False, side='long')
print(f"   DataFrame shape: {df_long.shape}")
print(f"   Columnas totales: {len(df_long.columns)}")

# Ver qué columnas tiene
print(f"\n   Primeras columnas: {list(df_long.columns[:20])}")

# Crear secuencias LONG
print("\n3️⃣ Creando secuencias LONG...")
try:
    sequences_long = pipeline.create_sequences_by_side(
        df_long, sides=('long',), fit_scalers=True, train=False
    )

    shape_seq_short = sequences_long['long']['seq_short'].shape
    shape_seq_long = sequences_long['long']['seq_long'].shape
    shape_context = sequences_long['long']['context'].shape
    shape_time = sequences_long['long']['time'].shape

    print(f"   ✅ Secuencias creadas:")
    print(f"      seq_short: {shape_seq_short} → {shape_seq_short[2]} features")
    print(f"      seq_long:  {shape_seq_long} → {shape_seq_long[2]} features")
    print(f"      context:   {shape_context} → {shape_context[1]} features")
    print(f"      time:      {shape_time} → {shape_time[1]} features")

    n_features_long = shape_seq_short[2]

except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
    n_features_long = None

# Test SHORT
print("\n4️⃣ Preparando datos SHORT...")
df_short = pipeline.prepare_data(df.copy(), labels=False, side='short')

print("\n5️⃣ Creando secuencias SHORT...")
try:
    sequences_short = pipeline.create_sequences_by_side(
        df_short, sides=('short',), fit_scalers=False, train=False
    )

    shape_seq_short_s = sequences_short['short']['seq_short'].shape

    print(f"   ✅ Secuencias creadas:")
    print(f"      seq_short: {shape_seq_short_s} → {shape_seq_short_s[2]} features")

    n_features_short = shape_seq_short_s[2]

except Exception as e:
    print(f"   ❌ Error: {e}")
    n_features_short = None

# Verificar feature_engineer
print("\n6️⃣ Verificando FeatureEngineer...")
if hasattr(pipeline, 'feature_engineer'):
    fe = pipeline.feature_engineer

    if hasattr(fe, 'feature_columns'):
        print(f"   ✅ feature_columns disponible")
        print(f"   Lado actual: {fe.side}")

        fc = fe.feature_columns
        print(f"\n   Categorías de features:")
        for key, cols in fc.items():
            print(f"      {key}: {len(cols)} features")
            if len(cols) <= 10:
                print(f"         {cols}")
    else:
        print(f"   ⚠️  No hay feature_columns")
else:
    print(f"   ⚠️  No hay feature_engineer")

# Resultado
print("\n" + "="*70)
print("RESULTADO")
print("="*70)

if n_features_long is not None and n_features_short is not None:
    print(f"\n📊 Features en seq_short:")
    print(f"   LONG:  {n_features_long} features")
    print(f"   SHORT: {n_features_short} features")

    if n_features_long == n_features_short:
        print(f"\n❌ PROBLEMA: Máscaras NO se aplican")
        print(f"   Ambos lados tienen {n_features_long} features")
    else:
        diff = abs(n_features_long - n_features_short)
        print(f"\n✅ Máscaras funcionan correctamente")
        print(f"   Diferencia: {diff} features")

    if n_features_long < 30:
        print(f"\n⚠️  ADVERTENCIA CRÍTICA: {n_features_long} features es MUY BAJO")
        print(f"   Esperado: 60-80 features")
        print(f"   Actual:   {n_features_long} features")
        print(f"\n   Causas posibles:")
        print(f"   1. FeatureEngineer genera muy pocas features base")
        print(f"   2. Máscaras filtran demasiado agresivamente")
        print(f"   3. prepare_data elimina muchas columnas")
    else:
        print(f"\n✅ Número de features es razonable")

print("\n" + "="*70 + "\n")