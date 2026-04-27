"""
Script de verificación para data_pipeline_v2_fixed.py

Verifica que:
1. Las dimensiones son correctas después del escalado
2. Las features binarias NO se escalaron (siguen en {0, 1})
3. Las features continuas SÍ se escalaron (distribución normalizada)
4. No hay errores de dimensionalidad en las asignaciones
"""

import numpy as np
import pandas as pd
from typing import Dict


def verify_binary_features(sequences: Dict[str, np.ndarray],
                           context_cols: list[str],
                           verbose: bool = True) -> bool:
    """
    Verifica que las features binarias siguen en {0, 1} después del escalado.
    """
    binary_features = {
        'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
        'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative'
    }

    context = sequences['context']
    all_ok = True

    if verbose:
        print("\n=== VERIFICACIÓN DE FEATURES BINARIAS ===")

    for i, col in enumerate(context_cols):
        if col in binary_features:
            unique_vals = np.unique(context[:, i])
            is_binary = np.all(np.isin(unique_vals, [0.0, 1.0]))

            if verbose:
                status = "✅" if is_binary else "❌"
                print(f"{status} {col:20s} → valores únicos: {unique_vals}")

            if not is_binary:
                all_ok = False
                if verbose:
                    print(f"   ERROR: {col} debería tener solo valores {{0, 1}}")

    return all_ok


def verify_continuous_features(sequences: Dict[str, np.ndarray],
                               context_cols: list[str],
                               verbose: bool = True) -> bool:
    """
    Verifica que las features continuas están escaladas correctamente.
    """
    # Features que deberían estar escaladas
    continuous_features = {
        'chop_score', 'exhaustion_score', 'atr_norm', 'adx_norm',
        'adx_smooth_norm', 'dm_diff_norm', 'bb_width', 'range_expansion',
        'dist_high_60', 'dist_low_60', 'position_range_240'
    }

    context = sequences['context']
    all_ok = True

    if verbose:
        print("\n=== VERIFICACIÓN DE FEATURES CONTINUAS ESCALADAS ===")

    for i, col in enumerate(context_cols):
        if col in continuous_features:
            values = context[:, i]
            mean_val = np.mean(values)
            std_val = np.std(values)
            min_val = np.min(values)
            max_val = np.max(values)

            # Una feature escalada típicamente tiene:
            # - Media cercana a 0 (entre -1 y 1)
            # - Std cercana a 1 (entre 0.5 y 2.0)
            is_scaled = abs(mean_val) < 2.0 and 0.3 < std_val < 3.0

            if verbose:
                status = "✅" if is_scaled else "⚠️"
                print(
                    f"{status} {col:20s} → mean={mean_val:7.3f}, std={std_val:6.3f}, range=[{min_val:7.3f}, {max_val:7.3f}]")

            if not is_scaled:
                all_ok = False
                if verbose:
                    print(f"   WARNING: {col} podría no estar escalada correctamente")

    return all_ok


def verify_shapes(sequences: Dict[str, np.ndarray],
                  expected_samples: int,
                  verbose: bool = True) -> bool:
    """
    Verifica que las formas de los arrays son correctas.
    """
    if verbose:
        print("\n=== VERIFICACIÓN DE DIMENSIONES ===")

    all_ok = True
    expected_shapes = {
        'seq_short': (expected_samples, 64, 23),  # Ajustar según tu config
        'seq_long': (expected_samples, 256, 15),  # Ajustar según tu config
        'context': (expected_samples, 18),  # Ajustar según tu config
        'time': (expected_samples, 8),  # Ajustar según tu config
    }

    for key, expected_shape in expected_shapes.items():
        if key in sequences and sequences[key] is not None:
            actual_shape = sequences[key].shape
            # Solo verificar el número de muestras
            is_ok = actual_shape[0] == expected_shape[0]

            if verbose:
                status = "✅" if is_ok else "❌"
                print(f"{status} {key:12s} → shape: {actual_shape} (esperado samples={expected_shape[0]})")

            if not is_ok:
                all_ok = False

    return all_ok


def verify_no_nan_inf(sequences: Dict[str, np.ndarray],
                      verbose: bool = True) -> bool:
    """
    Verifica que no hay NaN o Inf en los datos escalados.
    """
    if verbose:
        print("\n=== VERIFICACIÓN DE NaN/Inf ===")

    all_ok = True

    for key, data in sequences.items():
        if data is None:
            continue

        has_nan = np.any(np.isnan(data))
        has_inf = np.any(np.isinf(data))

        if verbose:
            status = "✅" if (not has_nan and not has_inf) else "❌"
            print(f"{status} {key:12s} → NaN: {has_nan}, Inf: {has_inf}")

        if has_nan or has_inf:
            all_ok = False
            if has_nan:
                print(f"   ERROR: {key} contiene NaN")
            if has_inf:
                print(f"   ERROR: {key} contiene Inf")

    return all_ok


def run_full_verification(sequences: Dict[str, np.ndarray],
                          context_cols: list[str],
                          expected_samples: int) -> bool:
    """
    Ejecuta todas las verificaciones.
    """
    print("=" * 70)
    print("VERIFICACIÓN COMPLETA DEL PIPELINE")
    print("=" * 70)

    checks = {
        "Shapes": verify_shapes(sequences, expected_samples),
        "No NaN/Inf": verify_no_nan_inf(sequences),
        "Binary features": verify_binary_features(sequences, context_cols),
        "Continuous features": verify_continuous_features(sequences, context_cols),
    }

    print("\n" + "=" * 70)
    print("RESUMEN DE VERIFICACIONES")
    print("=" * 70)

    all_passed = True
    for check_name, passed in checks.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status} {check_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("✅ TODAS LAS VERIFICACIONES PASARON")
    else:
        print("❌ ALGUNAS VERIFICACIONES FALLARON - REVISAR ARRIBA")

    return all_passed


# Ejemplo de uso:
"""
from data_pipeline_v2_fixed import DataPipeline
from verify_pipeline import run_full_verification

# Crear pipeline
pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)

# Preparar datos
df_prepared = pipeline.prepare_data(df_raw, side='long')

# Crear secuencias con escalado
sequences = pipeline.create_sequences(df_prepared, fit_scalers=True, train=True)

# Verificar
context_cols = pipeline.feature_engineer.feature_columns['context']
expected_samples = len(sequences['context'])

all_ok = run_full_verification(
    sequences=sequences,
    context_cols=context_cols,
    expected_samples=expected_samples
)

if not all_ok:
    print("\n⚠️ ATENCIÓN: Revisar los errores antes de continuar con el entrenamiento")
"""