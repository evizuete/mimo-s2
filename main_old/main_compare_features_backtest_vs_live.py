#!/usr/bin/env python3
"""
Script para comparar features entre backtest y producción
para diagnosticar por qué las predicciones son diferentes
"""
import numpy as np

from mimo_old.data_manager import DataManager
from mimo_old.databases import Database
from mimo_old.feature_builder import FeatureConfig
from mimo_old.helper import Helper
from mimo_old.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.data_pipeline import DataPipeline
from mimo_old.regime_state_machine import add_mimo_state


def compare_prepared_data():
    """
    Compara cómo se preparan los datos en backtest vs live
    """
    print("=" * 80)
    print("COMPARACIÓN: BACKTEST vs PRODUCCIÓN")
    print("=" * 80)

    # Configuración
    general_config = Config(
        release='200312',
        oof_splits=3,
        oof_epochs=20
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        batch_size=8912
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        label_method='adaptive'
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    artifacts_path = f'./artifacts/200312/oof/train_only'

    # Cargar datos
    db = Database()
    db.connect()

    # Cargar últimas 2048 barras (como en producción)
    print("\n1. Cargando datos de la base de datos...")
    helper = Helper(general_config=general_config, path=artifacts_path)
    df_rates = helper.load_from_database_real(db, last_n_rates=2048)

    print(f"   Datos cargados: {len(df_rates)} filas")
    print(f"   Rango temporal: {df_rates['time'].min()} a {df_rates['time'].max()}")

    # Pipeline
    pipeline = DataPipeline(
        general_config=general_config,
        feature_config=feature_config,
        model_config=model_config,
        regime_config=regime_config,
    )

    # Cargar scalers
    print("\n2. Cargando scalers...")
    try:
        pipeline.load_scalers(artifacts_path)
        print("   ✓ Scalers cargados correctamente")
    except Exception as e:
        print(f"   ❌ ERROR cargando scalers: {e}")
        return

    # MÉTODO 1: Como en backtest/training
    print("\n" + "=" * 80)
    print("MÉTODO 1: BACKTEST (prepare_data + create_sequences)")
    print("=" * 80)

    df_prep_backtest = pipeline.prepare_data(
        df_rates,
        labels=False,
        side='long',
        set_market_condition=False,
        ensure_regime=True
    )

    print(f"Filas después de prepare_data: {len(df_prep_backtest)}")

    # Agregar state (como en backtest)
    df_prep_backtest = add_mimo_state(
        df_prep_backtest,
        set_market_condition=True,
        ensure_regime=True,
        use_regime_prior=True
    )

    data_backtest = pipeline.create_sequences(
        df_prep_backtest,
        fit_scalers=False,
        train=False
    )

    print(f"Secuencias creadas: {data_backtest['seq_short'].shape}")

    # MÉTODO 2: Como en producción (main_trading_s2.py)
    print("\n" + "=" * 80)
    print("MÉTODO 2: PRODUCCIÓN (_prepare_live)")
    print("=" * 80)

    # Simular el método _prepare_live del TradingSimulator
    df_live = helper.load_from_dataframe(df_rates)

    df_prep_live = pipeline.prepare_data(
        df_live,
        labels=False,
        side=None,  # ⬅️ DIFERENCIA: None vs 'long'
        set_market_condition=True,  # ⬅️ DIFERENCIA
        ensure_regime=True
    )

    print(f"Filas después de prepare_data: {len(df_prep_live)}")

    df_prep_live = add_mimo_state(
        df_prep_live,
        set_market_condition=True,
        ensure_regime=True,
        use_regime_prior=True
    )

    data_live = pipeline.create_sequences(
        df_prep_live,
        fit_scalers=False,
        train=False
    )

    print(f"Secuencias creadas: {data_live['seq_short'].shape}")

    # COMPARACIÓN
    print("\n" + "=" * 80)
    print("COMPARACIÓN DE RESULTADOS")
    print("=" * 80)

    # Comparar shapes
    print("\n📊 SHAPES:")
    for key in ['seq_short', 'seq_long', 'context', 'time']:
        if key in data_backtest and key in data_live:
            shape_bt = data_backtest[key].shape
            shape_live = data_live[key].shape
            match = "✓" if shape_bt == shape_live else "❌"
            print(f"  {key:12s}: Backtest={shape_bt}, Live={shape_live} {match}")

    # Comparar última fila de features
    print("\n📊 ÚLTIMA FILA DE FEATURES (seq_short):")

    last_bt = data_backtest['seq_short'][-1, -1, :]  # última secuencia, último timestep
    last_live = data_live['seq_short'][-1, -1, :]

    print(f"\n  Backtest:")
    print(f"    Min:  {last_bt.min():.6f}")
    print(f"    Max:  {last_bt.max():.6f}")
    print(f"    Mean: {last_bt.mean():.6f}")
    print(f"    Std:  {last_bt.std():.6f}")

    print(f"\n  Live:")
    print(f"    Min:  {last_live.min():.6f}")
    print(f"    Max:  {last_live.max():.6f}")
    print(f"    Mean: {last_live.mean():.6f}")
    print(f"    Std:  {last_live.std():.6f}")

    # Diferencias
    diff = np.abs(last_bt - last_live)
    max_diff = diff.max()
    mean_diff = diff.mean()

    print(f"\n  Diferencias:")
    print(f"    Max:  {max_diff:.6f}")
    print(f"    Mean: {mean_diff:.6f}")

    if max_diff > 0.01:
        print(f"    ❌ PROBLEMA: Diferencias significativas!")
        print(f"       → Las features NO son iguales entre backtest y live")

        # Encontrar las features con mayor diferencia
        top_diff_idx = np.argsort(diff)[-5:][::-1]
        print(f"\n  Top 5 features con mayor diferencia:")
        for i, idx in enumerate(top_diff_idx):
            print(
                f"    {i + 1}. Feature {idx}: diff={diff[idx]:.6f} (BT={last_bt[idx]:.4f}, Live={last_live[idx]:.4f})")
    else:
        print(f"    ✓ OK: Features similares")

    # Comparar context
    print("\n📊 CONTEXT FEATURES:")
    ctx_bt = data_backtest['context'][-1, :]
    ctx_live = data_live['context'][-1, :]

    ctx_diff = np.abs(ctx_bt - ctx_live)
    print(f"  Max diff: {ctx_diff.max():.6f}")
    print(f"  Mean diff: {ctx_diff.mean():.6f}")

    if ctx_diff.max() > 0.01:
        print(f"  ❌ PROBLEMA en context features")
    else:
        print(f"  ✓ OK")

    # Comparar time features
    print("\n📊 TIME FEATURES:")
    time_bt = data_backtest['time'][-1, :]
    time_live = data_live['time'][-1, :]

    time_diff = np.abs(time_bt - time_live)
    print(f"  Max diff: {time_diff.max():.6f}")
    print(f"  Mean diff: {time_diff.mean():.6f}")

    if time_diff.max() > 0.01:
        print(f"  ❌ PROBLEMA en time features")
    else:
        print(f"  ✓ OK")

    # Comparar DataFrames preparados
    print("\n📊 DATAFRAMES PREPARADOS:")
    print(f"  Backtest: {len(df_prep_backtest)} filas, {len(df_prep_backtest.columns)} columnas")
    print(f"  Live:     {len(df_prep_live)} filas, {len(df_prep_live.columns)} columnas")

    # Comparar última fila
    if 'state' in df_prep_backtest.columns and 'state' in df_prep_live.columns:
        state_bt = df_prep_backtest.iloc[-1]['state']
        state_live = df_prep_live.iloc[-1]['state']
        print(f"\n  State última fila:")
        print(f"    Backtest: {state_bt}")
        print(f"    Live:     {state_live}")
        if state_bt != state_live:
            print(f"    ❌ PROBLEMA: States diferentes!")

    if 'regime' in df_prep_backtest.columns and 'regime' in df_prep_live.columns:
        regime_bt = df_prep_backtest.iloc[-1]['regime']
        regime_live = df_prep_live.iloc[-1]['regime']
        print(f"\n  Regime última fila:")
        print(f"    Backtest: {regime_bt}")
        print(f"    Live:     {regime_live}")
        if regime_bt != regime_live:
            print(f"    ❌ PROBLEMA: Regimes diferentes!")

    # HACER PREDICCIONES
    print("\n" + "=" * 80)
    print("PREDICCIONES CON AMBOS MÉTODOS")
    print("=" * 80)

    # Cargar modelo
    from tensorflow import keras
    model_path = f'{artifacts_path}/model_200312_long.keras'
    print(f"\nCargando modelo: {model_path}")
    model = keras.models.load_model(model_path, compile=False)

    # Predicción método backtest
    X_bt = [
        data_backtest['seq_short'][-1:],
        data_backtest['seq_long'][-1:],
        data_backtest['context'][-1:],
        data_backtest['time'][-1:]
    ]
    pred_bt = model.predict(X_bt, verbose=0)[0][0]

    # Predicción método live
    X_live = [
        data_live['seq_short'][-1:],
        data_live['seq_long'][-1:],
        data_live['context'][-1:],
        data_live['time'][-1:]
    ]
    pred_live = model.predict(X_live, verbose=0)[0][0]

    print(f"\n📊 PREDICCIONES:")
    print(f"  Backtest: {pred_bt:.10f}")
    print(f"  Live:     {pred_live:.10f}")
    print(f"  Diff:     {abs(pred_bt - pred_live):.10f}")
    print(f"  Ratio:    {pred_live / (pred_bt + 1e-10):.4f}x")

    if abs(pred_bt - pred_live) > 0.001:
        print(f"\n  ❌ CRÍTICO: Predicciones MUY diferentes!")
        print(f"     → Hay un problema en el pipeline de producción")
    elif pred_live < 0.001:
        print(f"\n  ❌ PROBLEMA: Predicción live extremadamente baja")
    else:
        print(f"\n  ✓ Predicciones similares")

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 80)

    return df_prep_backtest, df_prep_live, data_backtest, data_live


if __name__ == '__main__':
    compare_prepared_data()
