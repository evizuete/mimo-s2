#!/usr/bin/env python3
"""
Script de diagnóstico para predicciones extremadamente bajas
"""
import json
import os
import joblib
import numpy as np
import pandas as pd
from pathlib import Path


def check_model_architecture(model_path: str):
    """Verificar la arquitectura del modelo, especialmente la última capa"""
    try:
        import tensorflow as tf
        from tensorflow import keras

        print("=" * 80)
        print("1. VERIFICACIÓN DE ARQUITECTURA DEL MODELO")
        print("=" * 80)

        # Intentar cargar sin compilar (más seguro)
        model = keras.models.load_model(model_path, compile=False)

        print(f"\nModelo cargado desde: {model_path}")
        print(f"Tipo de modelo: {type(model).__name__}")
        print(f"Número de capas: {len(model.layers)}")

        # Última capa
        last_layer = model.layers[-1]
        print(f"\n✓ ÚLTIMA CAPA (CRÍTICA):")
        print(f"  Tipo: {type(last_layer).__name__}")
        print(f"  Nombre: {last_layer.name}")

        # Verificar activación
        activation_ok = False
        if hasattr(last_layer, 'activation'):
            activation = last_layer.activation
            activation_name = activation.__name__ if hasattr(activation, '__name__') else str(activation)
            print(f"  Activación: {activation_name}")

            if 'sigmoid' in activation_name.lower():
                print(f"  ✅ OK: Activación sigmoid presente")
                activation_ok = True
            else:
                print(f"  ❌ PROBLEMA: La última capa NO tiene activación sigmoid!")
                print(f"     → Las predicciones NO estarán en rango [0,1]")
                print(f"     → SOLUCIÓN: Añadir activation='sigmoid' en la última capa")
        else:
            print(f"  ⚠️  No se puede verificar activación")

        if hasattr(last_layer, 'units'):
            print(f"  Units: {last_layer.units}")
            if last_layer.units != 1:
                print(f"  ⚠️  WARNING: Se espera 1 unit para clasificación binaria")

        # Verificar bias
        if hasattr(last_layer, 'bias') and last_layer.bias is not None:
            try:
                bias = last_layer.bias.numpy()
                print(f"\n✓ BIAS INICIAL:")
                print(f"  Valor: {bias[0]:.6f}")

                if bias[0] < -4.0:
                    print(f"  ❌ CRÍTICO: Bias muy negativo ({bias[0]:.2f})!")
                    print(f"     → Esto causa predicciones extremadamente bajas")
                    print(f"     → Incluso con sigmoid: sigmoid({bias[0]:.2f}) ≈ {1 / (1 + np.exp(-bias[0])):.6f}")
                    print(f"     → SOLUCIÓN: Limitar init_bias entre [-2, 2] en el entrenamiento")
                elif bias[0] < -2.0:
                    print(f"  ⚠️  WARNING: Bias negativo ({bias[0]:.2f})")
                    print(f"     → sigmoid({bias[0]:.2f}) ≈ {1 / (1 + np.exp(-bias[0])):.6f}")
                else:
                    print(f"  ✓ Bias en rango razonable")
            except Exception as e:
                print(f"  ⚠️  No se pudo leer bias: {e}")

        # Mostrar summary resumido
        print(f"\n✓ RESUMEN DE ARQUITECTURA:")
        try:
            # Obtener info de input shapes
            if hasattr(model, 'input_shape'):
                print(f"  Input shape: {model.input_shape}")
            elif hasattr(model, 'inputs'):
                print(f"  Número de inputs: {len(model.inputs)}")
                for i, inp in enumerate(model.inputs):
                    print(f"    Input {i}: {inp.shape}")

            # Output shape
            if hasattr(model, 'output_shape'):
                print(f"  Output shape: {model.output_shape}")

            # Total params
            total_params = sum([tf.size(w).numpy() for w in model.trainable_weights])
            print(f"  Total parámetros entrenables: {total_params:,}")

        except Exception as e:
            print(f"  ⚠️  No se pudo obtener info de arquitectura: {e}")

        # Compilación info
        print(f"\n✓ Configuración de compilación:")
        try:
            if hasattr(model, 'loss'):
                loss = model.loss
                if loss:
                    print(f"  Loss: {loss}")
                else:
                    print(f"  Loss: N/A (modelo guardado sin compilar)")

            if hasattr(model, 'optimizer') and model.optimizer:
                print(f"  Optimizer: {model.optimizer.__class__.__name__}")
            else:
                print(f"  Optimizer: N/A (modelo guardado sin compilar)")
                print(f"  ℹ️  Nota: Es normal guardar modelos sin compilar para inferencia")
        except:
            print(f"  ⚠️  Modelo guardado sin compilar (esto es normal para predicción)")

        # Test de predicción simple
        print(f"\n✓ TEST DE PREDICCIÓN:")
        try:
            # Inferir dimensiones de los inputs del modelo
            seq_len_short = 64
            seq_len_long = 256
            n_features_seq_short = 23  # por defecto
            n_features_seq_long = 14  # por defecto
            n_features_context = 20
            n_features_time = 10

            # Intentar extraer las dimensiones reales del modelo
            if hasattr(model, 'inputs') and len(model.inputs) >= 4:
                try:
                    # Input 0: seq_short
                    seq_short_shape = model.inputs[0].shape
                    if seq_short_shape[1] is not None:
                        seq_len_short = int(seq_short_shape[1])
                    if seq_short_shape[2] is not None:
                        n_features_seq_short = int(seq_short_shape[2])

                    # Input 1: seq_long
                    seq_long_shape = model.inputs[1].shape
                    if seq_long_shape[1] is not None:
                        seq_len_long = int(seq_long_shape[1])
                    if seq_long_shape[2] is not None:
                        n_features_seq_long = int(seq_long_shape[2])

                    # Input 2: context
                    context_shape = model.inputs[2].shape
                    if context_shape[1] is not None:
                        n_features_context = int(context_shape[1])

                    # Input 3: time
                    time_shape = model.inputs[3].shape
                    if time_shape[1] is not None:
                        n_features_time = int(time_shape[1])

                    print(f"  Dimensiones extraídas del modelo:")
                    print(f"    Input 0 - seq_short: ({seq_len_short}, {n_features_seq_short})")
                    print(f"    Input 1 - seq_long: ({seq_len_long}, {n_features_seq_long})")
                    print(f"    Input 2 - context: ({n_features_context},)")
                    print(f"    Input 3 - time: ({n_features_time},)")
                except Exception as e:
                    print(f"  ⚠️  No se pudo extraer dimensiones exactas: {e}")
                    print(f"  Usando valores por defecto...")

            print(f"\n  Creando inputs dummy...")

            dummy_input = [
                np.random.randn(1, seq_len_short, n_features_seq_short).astype(np.float32),
                np.random.randn(1, seq_len_long, n_features_seq_long).astype(np.float32),
                np.random.randn(1, n_features_context).astype(np.float32),
                np.random.randn(1, n_features_time).astype(np.float32),
            ]

            pred = model.predict(dummy_input, verbose=0)
            print(f"  ✓ Predicción exitosa!")
            print(f"  Shape predicción: {pred.shape}")
            print(f"  Valor ejemplo: {pred[0][0]:.10f}")

            # Análisis del resultado
            if pred[0][0] < 0.0 or pred[0][0] > 1.0:
                print(f"  ❌ CRÍTICO: Predicción fuera de [0,1]! Falta sigmoid en última capa")
            elif pred[0][0] < 0.0001:
                print(f"  ❌ PROBLEMA: Predicción ejemplo muy baja!")
                print(f"     → El modelo tiende a predecir valores cercanos a 0")
                print(f"     → Probablemente por bias inicial muy negativo")
            elif pred[0][0] < 0.01:
                print(f"  ⚠️  WARNING: Predicción ejemplo baja (< 0.01)")
            else:
                print(f"  ✓ Predicción en rango razonable")

            # Hacer varias predicciones para ver varianza
            print(f"\n  Haciendo 10 predicciones con inputs aleatorios...")
            preds = []
            for _ in range(10):
                dummy = [
                    np.random.randn(1, seq_len_short, n_features_seq_short).astype(np.float32),
                    np.random.randn(1, seq_len_long, n_features_seq_long).astype(np.float32),
                    np.random.randn(1, n_features_context).astype(np.float32),
                    np.random.randn(1, n_features_time).astype(np.float32),
                ]
                p = model.predict(dummy, verbose=0)[0][0]
                preds.append(p)

            preds = np.array(preds)
            print(f"  Min: {preds.min():.6f}")
            print(f"  Max: {preds.max():.6f}")
            print(f"  Mean: {preds.mean():.6f}")
            print(f"  Std: {preds.std():.6f}")

            if preds.max() < 0.01:
                print(f"  ❌ CRÍTICO: TODAS las predicciones < 0.01")
                print(f"     → El modelo está roto, predicciones colapsadas")
            elif preds.std() < 0.001:
                print(f"  ⚠️  WARNING: Muy poca varianza en predicciones")

        except Exception as e:
            print(f"  ⚠️  No se pudo hacer test de predicción: {e}")
            import traceback
            traceback.print_exc()

        return model

    except Exception as e:
        print(f"❌ Error al cargar modelo: {e}")
        import traceback
        traceback.print_exc()
        return None


def check_calibrator(calibrator_path: str):
    """Verificar el calibrador isotónico"""
    print("\n" + "=" * 80)
    print("2. VERIFICACIÓN DEL CALIBRADOR")
    print("=" * 80)

    try:
        cal = joblib.load(calibrator_path)
        print(f"\nCalibrador cargado desde: {calibrator_path}")
        print(f"Tipo: {type(cal).__name__}")

        if hasattr(cal, 'X_min_'):
            print(f"\n✓ Rango de entrada (X):")
            print(f"  X_min: {cal.X_min_}")
            print(f"  X_max: {cal.X_max_}")

            if cal.X_max_ < 0.1:
                print(f"  ⚠️  WARNING: X_max muy bajo! Las predicciones del modelo son muy bajas.")

        if hasattr(cal, 'y_min_'):
            print(f"\n✓ Rango de salida (y):")
            print(f"  y_min: {cal.y_min_}")
            print(f"  y_max: {cal.y_max_}")

            if cal.y_max_ < 0.2:
                print(f"  ⚠️  WARNING: y_max muy bajo! La calibración es demasiado conservadora.")

        if hasattr(cal, 'f_'):
            print(f"\n✓ Función de mapeo (primeros 10 puntos):")
            x_points = cal.X_thresholds_[:10] if hasattr(cal, 'X_thresholds_') else []
            y_points = cal.y_thresholds_[:10] if hasattr(cal, 'y_thresholds_') else []

            for x, y in zip(x_points, y_points):
                print(f"  X={x:.6f} -> Y={y:.6f}")

        # Test con valores de ejemplo
        print(f"\n✓ Test de calibración con valores de ejemplo:")
        test_values = [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5]
        for val in test_values:
            try:
                cal_val = cal.predict([val])[0]
                print(f"  Input={val:.4f} -> Output={cal_val:.4f}")
            except:
                try:
                    cal_val = cal.predict([[val]])[0]
                    print(f"  Input={val:.4f} -> Output={cal_val:.4f}")
                except Exception as e:
                    print(f"  Input={val:.4f} -> ERROR: {e}")

        return cal

    except Exception as e:
        print(f"❌ Error al cargar calibrador: {e}")
        return None


def check_oof_artifacts(oof_df_path: str):
    """Verificar artefactos OOF"""
    print("\n" + "=" * 80)
    print("3. VERIFICACIÓN DE ARTEFACTOS OOF")
    print("=" * 80)

    try:
        df_oof = pd.read_parquet(oof_df_path)
        print(f"\nOOF DataFrame cargado desde: {oof_df_path}")
        print(f"Filas: {len(df_oof)}")

        if 'signal' in df_oof.columns:
            pos_rate = df_oof['signal'].mean()
            print(f"\n✓ Positive rate (señales): {pos_rate:.4f} ({pos_rate * 100:.2f}%)")

            if pos_rate < 0.01:
                print(f"  ⚠️  WARNING: Positive rate muy bajo! Clase muy desbalanceada.")
            elif pos_rate > 0.20:
                print(f"  ⚠️  WARNING: Positive rate alto! Puede indicar problemas en labels.")

        # Verificar predicciones OOF raw
        if 'oof_proba_raw' in df_oof.columns:
            print(f"\n✓ Predicciones OOF RAW:")
            print(df_oof['oof_proba_raw'].describe())

            median_raw = df_oof['oof_proba_raw'].median()
            if median_raw < 0.001:
                print(f"  ⚠️  WARNING: Mediana muy baja! El modelo predice valores extremadamente bajos.")

        # Verificar predicciones OOF calibradas
        if 'oof_proba_cal' in df_oof.columns:
            print(f"\n✓ Predicciones OOF CALIBRADAS:")
            print(df_oof['oof_proba_cal'].describe())

            zeros = (df_oof['oof_proba_cal'] == 0).sum()
            pct_zeros = zeros / len(df_oof) * 100
            print(f"\n  Valores exactamente 0: {zeros} ({pct_zeros:.2f}%)")

            if pct_zeros > 90:
                print(f"  ⚠️  CRITICAL: >90% de predicciones son 0! Calibración fallida.")

        return df_oof

    except Exception as e:
        print(f"❌ Error al cargar OOF artifacts: {e}")
        return None


def check_percentiles(percentiles_path: str):
    """Verificar percentiles por régimen"""
    print("\n" + "=" * 80)
    print("4. VERIFICACIÓN DE PERCENTILES")
    print("=" * 80)

    try:
        with open(percentiles_path, 'r') as f:
            percentiles = json.load(f)

        print(f"\nPercentiles cargados desde: {percentiles_path}")
        print(json.dumps(percentiles, indent=2))

        # Verificar si hay regímenes sin datos suficientes
        if '_meta' in percentiles:
            meta = percentiles['_meta']
            print(f"\n✓ Metadata:")
            for regime, info in meta.items():
                if isinstance(info, dict):
                    n_samples = info.get('n_samples', 0)
                    print(f"  {regime}: {n_samples} muestras")
                    if n_samples < 800:
                        print(f"    ⚠️  WARNING: Pocas muestras para {regime}")

        return percentiles

    except Exception as e:
        print(f"❌ Error al cargar percentiles: {e}")
        return None


def check_scalers(scalers_path: str):
    """Verificar scalers"""
    print("\n" + "=" * 80)
    print("5. VERIFICACIÓN DE SCALERS")
    print("=" * 80)

    try:
        scalers = joblib.load(scalers_path)
        print(f"\nScalers cargados desde: {scalers_path}")
        print(f"Tipo: {type(scalers)}")

        if isinstance(scalers, dict):
            for key, scaler in scalers.items():
                print(f"\n✓ Scaler '{key}':")
                print(f"  Tipo: {type(scaler).__name__}")

                if hasattr(scaler, 'mean_'):
                    print(f"  Media: {scaler.mean_[:5]}...")
                if hasattr(scaler, 'scale_'):
                    print(f"  Escala: {scaler.scale_[:5]}...")

        return scalers

    except Exception as e:
        print(f"❌ Error al cargar scalers: {e}")
        return None


def main():
    # Rutas por defecto
    release = '200316'
    base_path = f'./artifacts/{release}/oof/train_only'

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO DE PREDICCIONES EXTREMADAMENTE BAJAS")
    print("=" * 80)
    print(f"\nRelease: {release}")
    print(f"Base path: {base_path}")

    # Verificar que exista el directorio
    if not os.path.exists(base_path):
        print(f"\n❌ ERROR: No existe el directorio {base_path}")
        return

    # Verificar modelo LONG
    print("\n" + "#" * 80)
    print("# ANÁLISIS MODELO LONG")
    print("#" * 80)

    model_long_path = f'{base_path}/model_{release}_long.keras'
    calibrator_long_path = f'{base_path}/oof_calibrator_{release}_long.joblib'
    oof_long_path = f'{base_path}/oof_{release}_long.parquet'
    percentiles_long_path = f'{base_path}/percentiles_{release}_long.json'

    if os.path.exists(model_long_path):
        check_model_architecture(model_long_path)
    else:
        print(f"\n❌ No se encuentra el modelo: {model_long_path}")

    if os.path.exists(calibrator_long_path):
        check_calibrator(calibrator_long_path)
    else:
        print(f"\n❌ No se encuentra el calibrador: {calibrator_long_path}")

    if os.path.exists(oof_long_path):
        check_oof_artifacts(oof_long_path)
    else:
        print(f"\n⚠️  No se encuentra OOF artifacts: {oof_long_path}")

    if os.path.exists(percentiles_long_path):
        check_percentiles(percentiles_long_path)
    else:
        print(f"\n⚠️  No se encuentra percentiles: {percentiles_long_path}")

    # Verificar scalers (común para ambos)
    scalers_path = f'{base_path}/scalers_{release}.joblib'
    if os.path.exists(scalers_path):
        check_scalers(scalers_path)
    else:
        print(f"\n⚠️  No se encuentra scalers: {scalers_path}")

    # Verificar modelo SHORT
    print("\n" + "#" * 80)
    print("# ANÁLISIS MODELO SHORT")
    print("#" * 80)

    model_short_path = f'{base_path}/model_{release}_short.keras'
    calibrator_short_path = f'{base_path}/oof_calibrator_{release}_short.joblib'
    oof_short_path = f'{base_path}/oof_{release}_short.parquet'
    percentiles_short_path = f'{base_path}/percentiles_{release}_short.json'

    if os.path.exists(model_short_path):
        check_model_architecture(model_short_path)
    else:
        print(f"\n❌ No se encuentra el modelo: {model_short_path}")

    if os.path.exists(calibrator_short_path):
        check_calibrator(calibrator_short_path)
    else:
        print(f"\n❌ No se encuentra el calibrador: {calibrator_short_path}")

    if os.path.exists(oof_short_path):
        check_oof_artifacts(oof_short_path)
    else:
        print(f"\n⚠️  No se encuentra OOF artifacts: {oof_short_path}")

    if os.path.exists(percentiles_short_path):
        check_percentiles(percentiles_short_path)
    else:
        print(f"\n⚠️  No se encuentra percentiles: {percentiles_short_path}")

    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETADO")
    print("=" * 80)


if __name__ == '__main__':
    main()
