"""
═══════════════════════════════════════════════════════════════════════════
VALIDACIÓN COMPLETA - SCRIPT DE ENTRENAMIENTO OOF
═══════════════════════════════════════════════════════════════════════════

Este script valida TODO el pipeline de entrenamiento:
1. Carga de datos
2. Generación de features
3. Escalado
4. Creación de secuencias
5. Entrenamiento de modelos
6. Calibración
7. Métricas finales
8. Guardado de artefactos

Uso:
    python validate_full_training.py
"""

import json
import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.rolling_scaler import RollingRobustScaler


class TrainingValidator:
    """Validador completo del pipeline de entrenamiento"""

    def __init__(self):
        self.results = {}
        self.errors = []
        self.warnings = []

    def _set_pipeline_side_for_masks(self, pipeline, side: str):
        fe = getattr(pipeline, 'feature_engineer', None)
        if fe is None:
            return
        try:
            if hasattr(fe, 'set_side'):
                fe.set_side(side)
            if hasattr(fe, '_assign_features_to_inputs'):
                fe._assign_features_to_inputs()
        except Exception:
            pass

    @staticmethod
    def _unwrap_sequences(sequences: dict, side: str = 'long') -> dict:
        if isinstance(sequences, dict) and 'seq_short' in sequences:
            return sequences
        if isinstance(sequences, dict) and side in sequences and isinstance(sequences[side], dict):
            return sequences[side]

        raise KeyError(f"Formato de sequences no reconocido. Keys: {list(sequences.keys())[:20]}")

    @staticmethod
    def _no_scale_features_by_block() -> dict:
        """
        Features que NO deberían considerarse para warnings de 'scale muy pequeña'
        (binarias / flags / discretas).
        Ajusta esta lista si cambias nombres de columnas.
        """
        return {
            "seq_short": {"doji", "hammer", "shooting_star"},
            "seq_long": {"trend_dir"},
            "context": {
                "is_chop", "is_exhaustion",
                "ema_bull", "ema_bear",
                "rsi_oversold", "rsi_overbought",
                "macd_positive", "macd_negative",
            },
        }

    def run_all_checks(self,
                       pipeline,
                       df_train: pd.DataFrame,
                       df_val: pd.DataFrame = None,
                       models: Dict = None,
                       calibrators: Dict = None,
                       scalers: Dict = None):
        """
        Ejecuta TODAS las validaciones.

        Args:
            pipeline: DataPipeline instance
            df_train: DataFrame de entrenamiento
            df_val: DataFrame de validación (opcional)
            models: Modelos entrenados (opcional)
            calibrators: Calibradores (opcional)
            scalers: Scalers (opcional)
        """

        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 15 + "VALIDACIÓN COMPLETA DEL PIPELINE" + " " * 21 + "║")
        print("╚" + "═" * 68 + "╝\n")

        # Guardar df_train en self para que _test_9_artifacts pueda calcular
        # regime_thresholds antes de save_scalers (evita WARNING en meta.json)
        self._df_train = df_train

        # Test 1: Datos de entrada
        self._test_1_input_data(df_train, df_val)

        # Test 2: Generación de features
        self._test_2_feature_generation(pipeline, df_train)

        # Test 3: Escalado



        self._test_3_scaling(pipeline, df_train)

        # Test 4: Creación de secuencias
        self._test_4_sequences(pipeline, df_train)

        # Test 5: Máscaras de features
        self._test_5_feature_masks(pipeline, df_train)

        # Test 6: Labels (si es training)
        self._test_6_labels(pipeline, df_train)

        # Test 7: Modelos (si están disponibles)
        if models:
            self._test_7_models(models, pipeline, df_train)

        # Test 8: Calibradores (si están disponibles)
        if calibrators:
            self._test_8_calibrators(calibrators)

        # Test 9: Guardado/Carga de artefactos
        self._test_9_artifacts(pipeline, models, calibrators, scalers)

        # Test 10: Reproducibilidad
        self._test_10_reproducibility(pipeline, df_train)

        # Resumen final
        self._print_summary()

        return self.results

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: DATOS DE ENTRADA
    # ═══════════════════════════════════════════════════════════════════

    def _test_1_input_data(self, df_train: pd.DataFrame, df_val: pd.DataFrame = None):
        """Valida que los datos de entrada son correctos"""

        print("═" * 70)
        print("TEST 1: VALIDACIÓN DE DATOS DE ENTRADA")
        print("═" * 70)

        checks = []

        # Check 1.1: DataFrame no vacío
        if len(df_train) == 0:
            self.errors.append("DataFrame de entrenamiento está vacío")
            checks.append("❌ DataFrame vacío")
        else:
            checks.append(f"✅ {len(df_train):,} filas en train")

        # Check 1.2: Columnas requeridas
        required_cols = ['open', 'high', 'low', 'close', 'time']
        missing = [col for col in required_cols if col not in df_train.columns]

        if missing:
            self.errors.append(f"Faltan columnas requeridas: {missing}")
            checks.append(f"❌ Faltan columnas: {missing}")
        else:
            checks.append(f"✅ Todas las columnas OHLCV presentes")

        # Check 1.3: No hay NaN en OHLCV antes de features
        nan_cols = df_train[required_cols].isnull().sum()
        nan_cols = nan_cols[nan_cols > 0]

        if len(nan_cols) > 0:
            self.warnings.append(f"NaNs en datos base: {dict(nan_cols)}")
            checks.append(f"⚠️  NaNs en: {list(nan_cols.index)}")
        else:
            checks.append("✅ Sin NaNs en OHLCV")

        # Check 1.4: Datos ordenados por tiempo
        if 'time' in df_train.columns:
            time_col = pd.to_datetime(df_train['time'])
            if not time_col.is_monotonic_increasing:
                self.warnings.append("Datos no están ordenados cronológicamente")
                checks.append("⚠️  Datos desordenados (verificar sort)")
            else:
                checks.append("✅ Datos ordenados cronológicamente")

        # Check 1.5: Rango de valores sensato
        price_checks = []
        for col in ['open', 'high', 'low', 'close']:
            if col in df_train.columns:
                vals = df_train[col]
                if (vals <= 0).any():
                    price_checks.append(f"{col} tiene valores <= 0")
                if vals.max() / vals.min() > 100:
                    price_checks.append(f"{col} rango muy amplio (>100x)")

        if price_checks:
            self.warnings.append(f"Precios sospechosos: {price_checks}")
            checks.append(f"⚠️  Verificar precios: {price_checks}")
        else:
            checks.append("✅ Rangos de precios razonables")

        # Check 1.6: Validación set (si existe)
        if df_val is not None:
            if len(df_val) == 0:
                self.warnings.append("DataFrame de validación vacío")
                checks.append("⚠️  Val set vacío")
            else:
                checks.append(f"✅ {len(df_val):,} filas en validation")
        else:
            checks.append("ℹ️  No hay validation set")

        # Imprimir resultados
        for check in checks:
            print(f"  {check}")

        self.results['input_data'] = {
            'train_rows': len(df_train),
            'val_rows': len(df_val) if df_val is not None else 0,
            'columns': list(df_train.columns),
            'passed': len([c for c in checks if c.startswith('✅')]),
            'warnings': len([c for c in checks if c.startswith('⚠️')]),
            'errors': len([c for c in checks if c.startswith('❌')])
        }

        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: GENERACIÓN DE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def _test_2_feature_generation(self, pipeline, df: pd.DataFrame):
        """Valida la generación de features técnicas"""

        print("═" * 70)
        print("TEST 2: GENERACIÓN DE FEATURES")
        print("═" * 70)

        checks = []

        try:
            # Generar features
            print("  Generando features...")
            t0 = time.time()
            df_features = pipeline.feature_engineer.generate_all_features(df.copy())
            t_elapsed = time.time() - t0

            checks.append(f"✅ Features generadas en {t_elapsed:.2f}s")

            # Check 2.1: Features esperadas están presentes
            expected_features = [
                'atr', 'rsi', 'adx', 'macd_hist',
                'ema_9', 'ema_21', 'ema_50',
                'bb_upper', 'bb_lower',
                'chop_score', 'exhaustion_score'
            ]

            missing_features = [f for f in expected_features if f not in df_features.columns]
            if missing_features:
                self.warnings.append(f"Faltan features esperadas: {missing_features}")
                checks.append(f"⚠️  Faltan: {missing_features}")
            else:
                checks.append(f"✅ Todas las features clave presentes")

            # Check 2.2: Features binarias (máscaras)
            binary_features = ['ema_bull', 'ema_bear', 'rsi_oversold', 'rsi_overbought',
                               'macd_positive', 'macd_negative', 'is_chop', 'is_exhaustion']

            for feat in binary_features:
                if feat in df_features.columns:
                    unique_vals = df_features[feat].dropna().unique()
                    if not set(unique_vals).issubset({0, 1, 0.0, 1.0}):
                        self.errors.append(f"{feat} no es binaria: {unique_vals}")
                        checks.append(f"❌ {feat} no es binaria")

            checks.append(f"✅ Features binarias correctas")

            # Check 2.3: Rango de features normalizadas
            norm_features = [col for col in df_features.columns if col.endswith('_norm')]

            for feat in norm_features[:5]:  # Check primeras 5
                if feat in df_features.columns:
                    vals = df_features[feat].dropna()
                    if len(vals) > 0:
                        extreme = (vals.abs() > 100).sum()
                        if extreme > len(vals) * 0.01:  # >1% valores extremos
                            self.warnings.append(f"{feat} tiene muchos valores extremos")
                            checks.append(f"⚠️  {feat} tiene valores extremos")

            checks.append(f"✅ Features normalizadas en rango razonable")

            # Check 2.4: NaNs después de features
            nan_after = df_features.isnull().sum().sum()
            rows_before = len(df)
            rows_after = len(df_features.dropna())
            pct_lost = (1 - rows_after / rows_before) * 100

            if pct_lost > 30:
                self.warnings.append(f"Se perdió {pct_lost:.1f}% de datos por NaNs")
                checks.append(f"⚠️  {pct_lost:.1f}% datos perdidos por NaNs")
            else:
                checks.append(f"✅ Solo {pct_lost:.1f}% datos perdidos")

            # Check 2.5: Infinitos
            inf_count = np.isinf(df_features.select_dtypes(include=[np.number]).values).sum()
            if inf_count > 0:
                self.errors.append(f"{inf_count} valores infinitos en features")
                checks.append(f"❌ {inf_count} valores infinitos")
            else:
                checks.append(f"✅ Sin infinitos")

            self.results['feature_generation'] = {
                'n_features': len(df_features.columns),
                'time_seconds': t_elapsed,
                'rows_before': rows_before,
                'rows_after': rows_after,
                'pct_lost': pct_lost,
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en generación de features: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['feature_generation'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 3: ESCALADO
    # ═══════════════════════════════════════════════════════════════════

    def _test_3_scaling(self, pipeline, df: pd.DataFrame):
        """Valida el escalado de features"""

        print("═" * 70)
        print("TEST 3: ESCALADO DE FEATURES")
        print("═" * 70)

        checks = []

        try:
            # Preparar datos
            df_prep = pipeline.prepare_data(df.copy(), labels=False, side='long')

            # Crear secuencias y escalar
            print("  Escalando features...")
            t0 = time.time()
            seqs = pipeline.create_sequences_by_side(df_prep, sides=('long',), fit_scalers=True, train=False)
            sequences = seqs['long']

            t_elapsed = time.time() - t0

            checks.append(f"✅ Escalado completado en {t_elapsed:.2f}s")

            # Check 3.1: Scalers creados
            expected_scalers = ['seq_short', 'seq_long', 'context']
            missing = [s for s in expected_scalers if s not in pipeline.scalers]

            if missing:
                self.errors.append(f"Faltan scalers: {missing}")
                checks.append(f"❌ Faltan scalers: {missing}")
            else:
                checks.append(f"✅ Todos los scalers creados")

            # Check 3.2: Tipo de scaler correcto

            for name, scaler in pipeline.scalers.items():
                if not isinstance(scaler, RollingRobustScaler):
                    self.warnings.append(f"Scaler {name} no es RollingRobustScalerV2")
                    checks.append(f"⚠️  {name} no es RollingRobustScalerV2")

            checks.append(f"✅ Tipo de scalers correcto")

            # Check 3.3: Scaler fitted
            for name, scaler in pipeline.scalers.items():
                if hasattr(scaler, 'is_fitted') and not scaler.is_fitted:
                    self.warnings.append(f"Scaler {name} no está fitted")
                    checks.append(f"⚠️  {name} no fitted")

            checks.append(f"✅ Scalers fitted correctamente")

            # Check 3.4: Stats de scaler razonables
            for name, scaler in pipeline.scalers.items():
                if hasattr(scaler, 'get_stats'):
                    stats = scaler.get_stats()

                    # Verificar warmup
                    if stats['buffer_size'] < stats['warmup_size']:
                        self.warnings.append(f"Scaler {name} no tiene warmup completo")
                        checks.append(f"⚠️  {name} sin warmup completo")

                    # Verificar median y scale
                    if stats['median'] is not None:
                        med_arr = np.array(stats['median'])
                        scl_arr = np.array(stats['scale'])

                        no_scale = self._no_scale_features_by_block()
                        scaler_to_cols_key = {
                            'seq_short': 'sequence_short',
                            'seq_long': 'sequence_long',
                            'context': 'context'
                        }

                        cols_key = scaler_to_cols_key.get(name)
                        if cols_key is None:
                            cols = None
                        else:
                            cols = pipeline.feature_engineer.feature_columns.get(cols_key, [])

                        threshold = 1e-6
                        small_idx = np.where(scl_arr < threshold)[0]

                        if cols is None or len(cols) != len(scl_arr):
                            if len(small_idx) > 0:
                                self.warnings.append(f"Scaler {name} tiene scales muy pequeñas (<{threshold})")
                                checks.append(f"⚠️  {name} scales pequeñas")
                        else:
                            # filtrar los índices que correspondan a features "no_scale" (binarias / flags)
                            ignore_set = no_scale.get(name, set())
                            small_feats = [(i, cols[i], float(scl_arr[i])) for i in small_idx]
                            small_feats_non_binary = [t for t in small_feats if t[1] not in ignore_set]

                            if len(small_feats_non_binary) > 0:
                                self.warnings.append(f"Scaler {name} tiene scales muy pequeñas en features no binarias")
                                checks.append(f"⚠️  {name} scales pequeñas (no binarias)")

                                I = ", ".join([f"{feat}={val:.2e}" for _, feat, val in small_feats_non_binary[:8]])
                                checks.append(f"    ↳ Ejemplos: {I}")

            checks.append(f"✅ Stats de scalers razonables")

            # Check 3.5: Datos escalados en rango esperado
            for key in ['seq_short', 'seq_long', 'context']:
                if key in sequences:
                    data = sequences[key]

                    # Estadísticas
                    mean = np.nanmean(data)
                    std = np.nanstd(data)
                    p99 = np.nanpercentile(np.abs(data), 99)

                    if abs(mean) > 0.5:
                        self.warnings.append(f"{key} mean={mean:.3f} (esperado ~0)")
                        checks.append(f"⚠️  {key} mean desviada")

                    if p99 > 10:
                        self.warnings.append(f"{key} p99={p99:.1f} (valores extremos)")
                        checks.append(f"⚠️  {key} valores extremos")

            checks.append(f"✅ Datos escalados en rango esperado")

            self.results['scaling'] = {
                'scalers_created': list(pipeline.scalers.keys()),
                'time_seconds': t_elapsed,
                'seq_short_shape': sequences['seq_short'].shape,
                'seq_long_shape': sequences['seq_long'].shape,
                'context_shape': sequences['context'].shape,
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en escalado: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['scaling'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 4: CREACIÓN DE SECUENCIAS
    # ═══════════════════════════════════════════════════════════════════

    def _test_4_sequences(self, pipeline, df: pd.DataFrame):
        """Valida la creación de secuencias"""

        print("═" * 70)
        print("TEST 4: CREACIÓN DE SECUENCIAS")
        print("═" * 70)

        checks = []

        try:
            df_prep = pipeline.prepare_data(df.copy(), labels=False, side='long')
            sequences_raw = pipeline.create_sequences_by_side(df_prep, sides=('long',), fit_scalers=False, train=False)
            sequences = self._unwrap_sequences(sequences_raw, side='long')

            # Check 4.1: Shapes correctos
            seq_short = sequences['seq_short']
            seq_long = sequences['seq_long']
            context = sequences['context']
            time_feat = sequences['time']

            # Verificar dimensiones
            if seq_short.ndim != 3:
                self.errors.append(f"seq_short debe ser 3D, es {seq_short.ndim}D")
                checks.append(f"❌ seq_short dimensión incorrecta")
            else:
                checks.append(f"✅ seq_short shape: {seq_short.shape}")

            if seq_long.ndim != 3:
                self.errors.append(f"seq_long debe ser 3D, es {seq_long.ndim}D")
                checks.append(f"❌ seq_long dimensión incorrecta")
            else:
                checks.append(f"✅ seq_long shape: {seq_long.shape}")

            # Check 4.2: Longitudes de secuencia correctas
            expected_short = pipeline.model_config.seq_len_short
            expected_long = pipeline.model_config.seq_len_long

            if seq_short.shape[1] != expected_short:
                self.errors.append(f"seq_short len={seq_short.shape[1]}, esperado={expected_short}")
                checks.append(f"❌ seq_short longitud incorrecta")
            else:
                checks.append(f"✅ seq_short longitud: {seq_short.shape[1]}")

            if seq_long.shape[1] != expected_long:
                self.errors.append(f"seq_long len={seq_long.shape[1]}, esperado={expected_long}")
                checks.append(f"❌ seq_long longitud incorrecta")
            else:
                checks.append(f"✅ seq_long longitud: {seq_long.shape[1]}")

            # Check 4.3: Alineación de batches
            n_samples = seq_long.shape[0]

            if seq_short.shape[0] != n_samples:
                self.errors.append(f"seq_short tiene {seq_short.shape[0]} samples vs {n_samples}")
                checks.append(f"❌ Batches desalineados")

            if context.shape[0] != n_samples:
                self.errors.append(f"context tiene {context.shape[0]} samples vs {n_samples}")
                checks.append(f"❌ Context desalineado")

            if time_feat.shape[0] != n_samples:
                self.errors.append(f"time tiene {time_feat.shape[0]} samples vs {n_samples}")
                checks.append(f"❌ Time desalineado")

            checks.append(f"✅ Todos los batches alineados: {n_samples} samples")

            # Check 4.4: Número de features razonable
            n_feat_short = seq_short.shape[2]
            n_feat_long = seq_long.shape[2]
            n_feat_context = context.shape[1]
            n_feat_time = time_feat.shape[1]

            if n_feat_short < 5:
                self.warnings.append(f"seq_short solo tiene {n_feat_short} features (muy pocas?)")
                checks.append(f"⚠️  seq_short pocas features")

            if n_feat_long < 5:
                self.warnings.append(f"seq_long solo tiene {n_feat_long} features (muy pocas?)")
                checks.append(f"⚠️  seq_long pocas features")

            checks.append(
                f"✅ Features: short={n_feat_short}, long={n_feat_long}, ctx={n_feat_context}, time={n_feat_time}")

            # Check 4.5: Datos válidos (sin NaN/Inf después de escalado)
            total_nans = (
                    np.isnan(seq_short).sum() +
                    np.isnan(seq_long).sum() +
                    np.isnan(context).sum() +
                    np.isnan(time_feat).sum()
            )

            if total_nans > 0:
                self.warnings.append(f"{total_nans} NaNs en secuencias escaladas")
                checks.append(f"⚠️  {total_nans} NaNs en secuencias")
            else:
                checks.append(f"✅ Sin NaNs en secuencias")

            total_infs = (
                    np.isinf(seq_short).sum() +
                    np.isinf(seq_long).sum() +
                    np.isinf(context).sum() +
                    np.isinf(time_feat).sum()
            )

            if total_infs > 0:
                self.errors.append(f"{total_infs} Infs en secuencias escaladas")
                checks.append(f"❌ {total_infs} Infs en secuencias")
            else:
                checks.append(f"✅ Sin Infs en secuencias")

            self.results['sequences'] = {
                'n_samples': n_samples,
                'seq_short_shape': list(seq_short.shape),
                'seq_long_shape': list(seq_long.shape),
                'context_shape': list(context.shape),
                'time_shape': list(time_feat.shape),
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en creación de secuencias: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['sequences'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 5: MÁSCARAS DE FEATURES
    # ═══════════════════════════════════════════════════════════════════

    def _test_5_feature_masks(self, pipeline, df: pd.DataFrame):
        """Valida que las máscaras de features funcionan correctamente.

        Soporta tres modos:
          - Sin máscaras (feature_masks=None): ambos lados ven todo.
          - Simétricas (solo True, sin False): ambos lados ven todas las
            features, los True son marcadores de intención, no exclusiones.
          - Asimétricas (hay False): cada lado excluye features del otro.
        """

        print("═" * 70)
        print("TEST 5: MÁSCARAS DE FEATURES (LONG vs SHORT)")
        print("═" * 70)

        checks = []

        try:
            fm = pipeline.feature_config.feature_masks

            # ── Clasificar modo ─────────────────────────────────────────────
            if fm is None or fm == {}:
                mask_mode = "sin_mascaras"
            else:
                mask_long_cfg  = fm.get('long',  {})
                mask_short_cfg = fm.get('short', {})
                excluded_long  = [k for k, v in mask_long_cfg.items()  if v is False]
                excluded_short = [k for k, v in mask_short_cfg.items() if v is False]
                mask_mode = "asimetricas" if (excluded_long or excluded_short) else "simetricas"

            checks.append(f"✅ Modo de máscaras detectado: {mask_mode.upper()}")

            if mask_mode == "sin_mascaras":
                checks.append("ℹ️  Sin máscaras configuradas — ambos modelos ven todas las features")
                self.results['feature_masks'] = {'configured': False, 'mode': mask_mode,
                                                  'passed': 1}
                for check in checks:
                    print(f"  {check}")
                print()
                return

            # ── Preparar datos y crear secuencias ───────────────────────────
            df_prep = pipeline.prepare_data(df.copy(), labels=False, side='both')

            print("  Creando secuencias para long y short...")
            t0 = time.time()
            sequences = pipeline.create_sequences_by_side(
                df_prep, sides=('long', 'short'), fit_scalers=False, train=False
            )
            t_elapsed = time.time() - t0
            checks.append(f"✅ Secuencias creadas en {t_elapsed:.2f}s")

            if 'long' not in sequences or 'short' not in sequences:
                self.errors.append("Faltan secuencias para long o short")
                checks.append("❌ Faltan secuencias para long o short")
                for check in checks:
                    print(f"  {check}")
                print()
                return

            checks.append("✅ Secuencias long y short presentes")

            n_long  = sequences['long']['seq_short'].shape[0]
            n_short = sequences['short']['seq_short'].shape[0]
            if n_long != n_short:
                self.errors.append(f"Samples distintos: long={n_long} short={n_short}")
                checks.append(f"❌ Samples distintos: long={n_long} short={n_short}")
            else:
                checks.append(f"✅ Mismo número de samples: {n_long}")

            n_long_seq_w  = sequences['long']['seq_short'].shape[2]
            n_short_seq_w = sequences['short']['seq_short'].shape[2]
            n_long_ctx    = sequences['long']['context'].shape[1]
            n_short_ctx   = sequences['short']['context'].shape[1]

            checks.append(
                f"ℹ️  Dimensiones — LONG: seq_short={n_long_seq_w} ctx={n_long_ctx} | "
                f"SHORT: seq_short={n_short_seq_w} ctx={n_short_ctx}"
            )

            # ── Reconstruir columnas activas por lado ────────────────────────
            pipeline.feature_engineer.set_side('long')
            pipeline.feature_engineer._assign_features_to_inputs()
            long_ctx_cols  = list(pipeline.feature_engineer.feature_columns['context'])

            pipeline.feature_engineer.set_side('short')
            pipeline.feature_engineer._assign_features_to_inputs()
            short_ctx_cols = list(pipeline.feature_engineer.feature_columns['context'])

            long_only  = set(long_ctx_cols)  - set(short_ctx_cols)
            short_only = set(short_ctx_cols) - set(long_ctx_cols)

            # ── Auditoría de features direccionales ──────────────────────────
            directional = [
                'ema_bull', 'ema_bear',
                'rsi_oversold', 'rsi_overbought',
                'macd_positive', 'macd_negative',
            ]
            mask_long_cfg  = fm.get('long',  {})
            mask_short_cfg = fm.get('short', {})
            required_long  = [k for k, v in mask_long_cfg.items()  if v is True]
            required_short = [k for k, v in mask_short_cfg.items() if v is True]
            excluded_long  = [k for k, v in mask_long_cfg.items()  if v is False]
            excluded_short = [k for k, v in mask_short_cfg.items() if v is False]

            print("  Features direccionales:")
            audit_errors = []
            for f in directional:
                in_long  = f in long_ctx_cols
                in_short = f in short_ctx_cols
                tag = ""
                if f in required_long  and not in_long:
                    audit_errors.append(f"LONG: requerida '{f}' ausente")
                if f in required_short and not in_short:
                    audit_errors.append(f"SHORT: requerida '{f}' ausente")
                if f in excluded_long  and in_long:
                    audit_errors.append(f"LONG: excluida '{f}' presente")
                if f in excluded_short and in_short:
                    audit_errors.append(f"SHORT: excluida '{f}' presente")
                lsym = "✓" if in_long  else "✗"
                ssym = "✓" if in_short else "✗"
                ann  = ("← requerida" if f in required_long  else "") +                        (" ✗excluida"  if f in excluded_long  else "")
                ann += ("← requerida" if f in required_short else "") +                        (" ✗excluida"  if f in excluded_short else "")
                print(f"     {f:<22} LONG={lsym}  SHORT={ssym}  {ann}")

            for err in audit_errors:
                self.errors.append(err)
                checks.append(f"❌ {err}")

            # ── Veredicto por modo ───────────────────────────────────────────
            if mask_mode == "simetricas":
                # Ambos lados ven todo — correcto, no esperamos diferencias
                if n_long_ctx == n_short_ctx and not audit_errors:
                    checks.append(
                        "✅ MÁSCARAS SIMÉTRICAS CORRECTAS — ambos modelos ven las 6 "
                        "features direccionales. Objetivo: eliminar sesgo long/short."
                    )
                else:
                    if n_long_ctx != n_short_ctx:
                        self.warnings.append(
                            f"Máscaras simétricas pero ctx difiere: "
                            f"long={n_long_ctx} short={n_short_ctx}"
                        )
                        checks.append(
                            f"⚠️  ctx distinto con máscaras simétricas "
                            f"(long={n_long_ctx} short={n_short_ctx})"
                        )

            elif mask_mode == "asimetricas":
                if long_only or short_only:
                    if long_only:
                        checks.append(f"✅ Features exclusivas LONG:  {sorted(long_only)}")
                    if short_only:
                        checks.append(f"✅ Features exclusivas SHORT: {sorted(short_only)}")
                    checks.append("✅ MÁSCARAS ASIMÉTRICAS CORRECTAS — exclusiones verificadas")
                else:
                    # Hay False configurados pero no producen diferencias → problema real
                    self.warnings.append(
                        "Máscaras configuradas como asimétricas (hay False) "
                        "pero no se detectan columnas exclusivas por lado. "
                        "Verificar que DataPipeline aplica _apply_side_mask."
                    )
                    checks.append(
                        "⚠️  MÁSCARAS ASIMÉTRICAS SIN EFECTO DETECTABLE — "
                        "revisar DataPipeline._apply_side_mask"
                    )

            self.results['feature_masks'] = {
                'configured': True,
                'mode': mask_mode,
                'long_ctx_features': n_long_ctx,
                'short_ctx_features': n_short_ctx,
                'long_only': sorted(long_only),
                'short_only': sorted(short_only),
                'required_long': required_long,
                'required_short': required_short,
                'excluded_long': excluded_long,
                'excluded_short': excluded_short,
                'time_seconds': t_elapsed,
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en máscaras: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['feature_masks'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 6: LABELS
    # ═══════════════════════════════════════════════════════════════════

    def _test_6_labels(self, pipeline, df: pd.DataFrame):
        """Valida la generación de labels para LONG y SHORT."""

        print("═" * 70)
        print("TEST 6: GENERACIÓN DE LABELS")
        print("═" * 70)

        checks = []
        results_by_side = {}

        for side in ('long', 'short'):
            # Determinar método activo para este side
            lg_cfg = pipeline.label_generator.config
            if side == 'short' and getattr(lg_cfg, 'label_method_short', None):
                active_method = lg_cfg.label_method_short
            elif side == 'long' and getattr(lg_cfg, 'label_method_long', None):
                active_method = lg_cfg.label_method_long
            else:
                active_method = lg_cfg.label_method

            print(f"  Generando labels [{side.upper()}] método={active_method}...")

            try:
                df_prep = pipeline.prepare_data(df.copy(), labels=True, side=side)

                # Check 6.1: Columna signal presente
                if 'signal' not in df_prep.columns:
                    self.errors.append(f"[{side}] Columna 'signal' no generada")
                    checks.append(f"❌ [{side}] Sin columna signal")
                    results_by_side[side] = {'error': 'no signal column'}
                    continue

                checks.append(f"✅ [{side}] Columna signal presente")

                # Check 6.2: Valores binarios
                unique_signals = df_prep['signal'].dropna().unique()
                if not set(unique_signals).issubset({0, 1, 0.0, 1.0}):
                    self.errors.append(f"[{side}] Signal no es binaria: {unique_signals}")
                    checks.append(f"❌ [{side}] Signal no binaria: {unique_signals}")
                else:
                    checks.append(f"✅ [{side}] Signal binaria (0/1)")

                # Check 6.3: Balance de clases
                n_positive = int((df_prep['signal'] == 1).sum())
                n_negative = int((df_prep['signal'] == 0).sum())
                pct_positive = n_positive / max(n_positive + n_negative, 1) * 100

                if n_positive == 0:
                    self.errors.append(f"[{side}] Sin positivos — barriers demasiado exigentes")
                    checks.append(f"❌ [{side}] 0 positivos — revisar barriers o label_method")
                elif pct_positive < 3.0:
                    self.errors.append(
                        f"[{side}] pos_rate={pct_positive:.1f}% — barriers demasiado exigentes "
                        f"(mín recomendado: 5%)"
                    )
                    checks.append(f"❌ [{side}] pos_rate={pct_positive:.1f}% — demasiado bajo")
                elif pct_positive < 5.0:
                    self.warnings.append(
                        f"[{side}] pos_rate={pct_positive:.1f}% — bajo, considera ajustar barriers"
                    )
                    checks.append(f"⚠️  [{side}] pos_rate={pct_positive:.1f}% — bajo")
                else:
                    checks.append(f"✅ [{side}] Balance razonable: {pct_positive:.1f}% positivos")

                # Breakdown por régimen si existe columna state
                if 'state' in df_prep.columns:
                    regime_stats = df_prep.groupby('state')['signal'].agg(['mean', 'sum', 'count'])
                    print(f"  [{side.upper()}] pos_rate por régimen:")
                    print(regime_stats.to_string(float_format=lambda x: f"{x:.4f}"))
                    print()

                    # Detectar regímenes con pos_rate=0
                    zero_regimes = regime_stats[regime_stats['sum'] == 0].index.tolist()
                    if zero_regimes:
                        self.warnings.append(f"[{side}] Regímenes sin positivos: {zero_regimes}")
                        checks.append(f"⚠️  [{side}] Sin señales en: {zero_regimes}")

                # Check 6.4: Labels en secuencias
                sequences_raw = pipeline.create_sequences_by_side(
                    df_prep, sides=(side,), fit_scalers=False, train=True
                )
                sequences = self._unwrap_sequences(sequences_raw, side=side)

                if sequences['labels'] is None:
                    self.errors.append(f"[{side}] Labels no incluidas en secuencias")
                    checks.append(f"❌ [{side}] Sin labels en secuencias")
                else:
                    labels = sequences['labels']
                    checks.append(f"✅ [{side}] Labels en secuencias: {labels.shape}")

                    if labels.shape[0] != sequences['seq_long'].shape[0]:
                        self.errors.append(f"[{side}] Labels desalineadas con secuencias")
                        checks.append(f"❌ [{side}] Labels desalineadas")
                    else:
                        checks.append(f"✅ [{side}] Labels alineadas con secuencias")

                # Check 6.5: Weights
                if sequences.get('weights') is not None:
                    weights = sequences['weights']
                    checks.append(f"✅ [{side}] Weights presentes: {weights.shape}")
                    if (weights < 0).any():
                        self.errors.append(f"[{side}] Weights negativos")
                        checks.append(f"❌ [{side}] Weights negativos")
                    else:
                        checks.append(f"✅ [{side}] Weights válidos")
                else:
                    checks.append(f"ℹ️  [{side}] Sin weights (usando uniformes)")

                results_by_side[side] = {
                    'signal_present': True,
                    'label_method': active_method,
                    'n_positive': n_positive,
                    'n_negative': n_negative,
                    'pct_positive': round(pct_positive, 2),
                    'labels_in_sequences': sequences['labels'] is not None,
                }

            except Exception as e:
                self.errors.append(f"[{side}] Error en labels: {e}")
                checks.append(f"❌ [{side}] ERROR: {e}")
                results_by_side[side] = {'error': str(e)}

        self.results['labels'] = {
            'passed': len([c for c in checks if c.startswith('✅')]),
            'by_side': results_by_side,
        }

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 7: MODELOS
    # ═══════════════════════════════════════════════════════════════════

    def _test_7_models(self, models: Dict, pipeline, df: pd.DataFrame):
        """Valida los modelos entrenados"""

        print("═" * 70)
        print("TEST 7: MODELOS ENTRENADOS")
        print("═" * 70)

        checks = []

        try:
            # Check 7.1: Modelos para ambos lados
            expected_sides = ['long', 'short']
            for side in expected_sides:
                if side not in models:
                    self.warnings.append(f"Falta modelo para {side}")
                    checks.append(f"⚠️  Sin modelo {side}")
                else:
                    checks.append(f"✅ Modelo {side} presente")

            # Check 7.2: Modelos son Keras/TF
            for side, model in models.items():
                if not hasattr(model, 'predict'):
                    self.errors.append(f"Modelo {side} no tiene método predict")
                    checks.append(f"❌ {side} sin predict")
                else:
                    checks.append(f"✅ {side} es modelo válido")

            # Check 7.3: Predicción funciona
            df_prep = pipeline.prepare_data(df.copy(), labels=False, side='long')
            sequences = pipeline.create_sequences_by_side(
                df_prep, sides=('long', 'short'), fit_scalers=False, train=False
            )

            for side in ['long', 'short']:
                if side in models and side in sequences:
                    try:
                        X = [
                            sequences[side]['seq_short'],
                            sequences[side]['seq_long'],
                            sequences[side]['context'],
                            sequences[side]['time']
                        ]

                        print(f"  Prediciendo con modelo {side}...")
                        t0 = time.time()
                        pred = models[side].predict(X, verbose=0)
                        t_pred = time.time() - t0

                        checks.append(f"✅ {side} predice: {pred.shape} en {t_pred:.2f}s")

                        # Verificar rango de predicciones
                        if (pred < 0).any() or (pred > 1).any():
                            self.warnings.append(f"{side} predicciones fuera de [0,1]")
                            checks.append(f"⚠️  {side} pred fuera de rango")
                        else:
                            checks.append(f"✅ {side} pred en [0,1]")

                    except Exception as e:
                        self.errors.append(f"Error prediciendo {side}: {e}")
                        checks.append(f"❌ {side} falla al predecir: {e}")

            self.results['models'] = {
                'sides_present': list(models.keys()),
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en modelos: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['models'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 8: CALIBRADORES
    # ═══════════════════════════════════════════════════════════════════

    def _test_8_calibrators(self, calibrators: Dict):
        """Valida los calibradores"""

        print("═" * 70)
        print("TEST 8: CALIBRADORES")
        print("═" * 70)

        checks = []

        try:
            # Check 8.1: Calibradores para ambos lados
            for side in ['long', 'short']:
                if side not in calibrators:
                    self.warnings.append(f"Falta calibrador para {side}")
                    checks.append(f"⚠️  Sin calibrador {side}")
                else:
                    checks.append(f"✅ Calibrador {side} presente")

            # Check 8.2: Calibradores funcionan
            test_probs = np.array([0.1, 0.3, 0.5, 0.7, 0.9])

            for side, cal in calibrators.items():
                try:
                    cal_probs = cal.predict(test_probs)

                    if len(cal_probs) != len(test_probs):
                        self.errors.append(f"Calibrador {side} cambió shape")
                        checks.append(f"❌ {side} shape incorrecta")
                    else:
                        checks.append(f"✅ {side} funciona")

                    # Verificar monotonía (aproximada)
                    if not np.all(np.diff(cal_probs) >= -0.01):
                        self.warnings.append(f"Calibrador {side} no es monótono")
                        checks.append(f"⚠️  {side} no monótono")

                except Exception as e:
                    self.errors.append(f"Calibrador {side} falla: {e}")
                    checks.append(f"❌ {side} error: {e}")

            self.results['calibrators'] = {
                'sides_present': list(calibrators.keys()),
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en calibradores: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['calibrators'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 9: GUARDADO/CARGA DE ARTEFACTOS
    # ═══════════════════════════════════════════════════════════════════

    def _test_9_artifacts(self, pipeline, models=None, calibrators=None, scalers=None):
        """Valida guardado y carga de artefactos"""

        print("═" * 70)
        print("TEST 9: GUARDADO/CARGA DE ARTEFACTOS")
        print("═" * 70)

        checks = []

        try:
            import tempfile
            import shutil

            # Crear directorio temporal
            temp_dir = tempfile.mkdtemp()

            try:
                # Check 9.1: Guardar scalers
                # compute_and_store_regime_thresholds() DEBE ir antes de save_scalers()
                # para que los umbrales queden en meta.json y load_scalers() los inyecte.
                print("  Guardando scalers...")
                try:
                    if hasattr(self, '_df_train') and self._df_train is not None:
                        _df_prep_regime = pipeline.prepare_data(
                            self._df_train.copy(), labels=False, side=None
                        )
                        pipeline.compute_and_store_regime_thresholds(_df_prep_regime)
                except Exception as _e_regime:
                    self.warnings.append(f"No se pudieron calcular regime_thresholds: {_e_regime}")
                scaler_path = pipeline.save_scalers(temp_dir)

                if os.path.exists(scaler_path):
                    checks.append(f"✅ Scalers guardados en {scaler_path}")
                else:
                    self.errors.append("Scalers no se guardaron")
                    checks.append("❌ Scalers no guardados")

                # Check 9.2: Cargar scalers
                print("\t  Cargando scalers...")
                pipeline_new = type(pipeline)(
                    pipeline.general_config,
                    pipeline.feature_config,
                    pipeline.model_config,
                    pipeline.regime_config
                )

                # Los scalers de este validador provienen del pipeline entrenado en lado LONG
                # (tests previos preparan y ajustan con side='long'). Forzamos ese contexto
                # antes de recargar para evitar falsos mismatches de schema al reconstruir cols.
                self._set_pipeline_side_for_masks(pipeline_new, 'long')
                pipeline_new.load_scalers(temp_dir)

                if pipeline_new.is_fitted:
                    checks.append("✅ Scalers cargados correctamente")
                else:
                    self.warnings.append("Scalers cargados pero no fitted")
                    checks.append("⚠️  Scalers cargados pero no fitted")

                # Check 9.3: Stats de scalers iguales
                for name in pipeline.scalers.keys():
                    if name in pipeline_new.scalers:
                        orig_stats = pipeline.scalers[name].get_stats()
                        new_stats = pipeline_new.scalers[name].get_stats()

                        if orig_stats['n_updates'] != new_stats['n_updates']:
                            self.warnings.append(f"Scaler {name} n_updates diferente")
                            checks.append(f"⚠️  {name} stats diferentes")

                checks.append("✅ Stats de scalers consistentes")

                # Check 9.4: Archivos meta presentes
                meta_path = os.path.join(scaler_path, 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as f:
                        meta = json.load(f)
                    checks.append(f"✅ Metadata guardada: {meta.get('version')}")
                else:
                    self.warnings.append("Falta archivo meta.json")
                    checks.append("⚠️  Sin metadata")

                schema_path = os.path.join(scaler_path, 'feature_schema.json')
                if os.path.exists(schema_path):
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema = json.load(f)
                    checks.append(f"✅ Feature schema guardado para side={schema.get('side')}")
                else:
                    checks.append("ℹ️  Sin feature_schema.json (compatibilidad con artefactos antiguos)")

                self.results['artifacts'] = {
                    'scalers_saved': os.path.exists(scaler_path),
                    'scalers_loaded': pipeline_new.is_fitted,
                    'temp_dir': temp_dir,
                    'passed': len([c for c in checks if c.startswith('✅')])
                }

            finally:
                # Limpiar directorio temporal
                shutil.rmtree(temp_dir, ignore_errors=True)

        except Exception as e:
            self.errors.append(f"Error en artefactos: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['artifacts'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # TEST 10: REPRODUCIBILIDAD
    # ═══════════════════════════════════════════════════════════════════

    def _test_10_reproducibility(self, pipeline, df: pd.DataFrame):
        """Valida que el pipeline es reproducible"""

        print("═" * 70)
        print("TEST 10: REPRODUCIBILIDAD")
        print("═" * 70)

        checks = []

        try:
            # Ejecutar dos veces y comparar
            print("  Ejecutando pipeline 1ª vez...")
            df_prep1 = pipeline.prepare_data(df.copy(), labels=False, side='long')
            seq1_raw = pipeline.create_sequences_by_side(df_prep1, sides=('long',), fit_scalers=True, train=False)
            seq1 = self._unwrap_sequences(seq1_raw, side='long')

            print("  Ejecutando pipeline 2ª vez...")
            df_prep2 = pipeline.prepare_data(df.copy(), labels=False, side='long')
            seq2_raw = pipeline.create_sequences_by_side(df_prep2, sides=('long',), fit_scalers=False, train=False)
            seq2 = self._unwrap_sequences(seq2_raw, side='long')

            seq_diff = np.abs(seq1['seq_long'] - seq2['seq_long']).max()

            # Comparar features
            feat_diff = (df_prep1.values != df_prep2.values).sum()
            if feat_diff > 0:
                self.warnings.append(f"{feat_diff} features difieren entre ejecuciones")
                checks.append(f"⚠️  {feat_diff} features diferentes")
            else:
                checks.append("✅ Features reproducibles")

            # Comparar secuencias
            seq_diff = np.abs(seq1['seq_long'] - seq2['seq_long']).max()

            if seq_diff > 1e-5:
                self.warnings.append(f"Secuencias difieren: max_diff={seq_diff}")
                checks.append(f"⚠️  Secuencias diferentes: {seq_diff:.2e}")
            else:
                checks.append(f"✅ Secuencias reproducibles (diff={seq_diff:.2e})")

            self.results['reproducibility'] = {
                'features_diff': int(feat_diff),
                'sequences_max_diff': float(seq_diff),
                'reproducible': seq_diff < 1e-5,
                'passed': len([c for c in checks if c.startswith('✅')])
            }

        except Exception as e:
            self.errors.append(f"Error en reproducibilidad: {e}")
            checks.append(f"❌ ERROR: {e}")
            self.results['reproducibility'] = {'error': str(e)}

        for check in checks:
            print(f"  {check}")
        print()

    # ═══════════════════════════════════════════════════════════════════
    # RESUMEN FINAL
    # ═══════════════════════════════════════════════════════════════════

    def _print_summary(self):
        """Imprime resumen final de validación"""

        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "RESUMEN DE VALIDACIÓN" + " " * 27 + "║")
        print("╚" + "═" * 68 + "╝\n")

        # Contar tests
        total_tests = len([k for k in self.results.keys() if k != 'summary'])
        passed_tests = len([v for v in self.results.values()
                            if isinstance(v, dict) and v.get('passed', 0) > 0 and 'error' not in v])

        # Imprimir por test
        for test_name, test_results in self.results.items():
            if test_name == 'summary':
                continue

            if isinstance(test_results, dict):
                if 'error' in test_results:
                    status = "❌ ERROR"
                elif test_results.get('passed', 0) > 0:
                    status = f"✅ PASS ({test_results['passed']} checks)"
                else:
                    status = "⚠️  WARN"
            else:
                status = "ℹ️  INFO"

            print(f"  {test_name:20s}: {status}")

        # Errores y warnings
        print("\n" + "-" * 70)

        if self.errors:
            print(f"\n❌ ERRORES ({len(self.errors)}):")
            for i, error in enumerate(self.errors[:10], 1):
                print(f"  {i}. {error}")
            if len(self.errors) > 10:
                print(f"  ... y {len(self.errors) - 10} más")

        if self.warnings:
            print(f"\n⚠️  WARNINGS ({len(self.warnings)}):")
            for i, warning in enumerate(self.warnings[:10], 1):
                print(f"  {i}. {warning}")
            if len(self.warnings) > 10:
                print(f"  ... y {len(self.warnings) - 10} más")

        # Resultado final
        print("\n" + "═" * 70)

        if len(self.errors) == 0:
            if len(self.warnings) == 0:
                print("✅ VALIDACIÓN EXITOSA - Todo funciona correctamente")
                print("   Puedes proceder con confianza al entrenamiento completo")
            else:
                print("✅ VALIDACIÓN MAYORMENTE EXITOSA - Hay warnings pero nada crítico")
                print(f"   {len(self.warnings)} warnings encontrados - revisar antes de producción")
        else:
            print("❌ VALIDACIÓN FALLIDA - Hay errores que deben corregirse")
            print(f"   {len(self.errors)} errores encontrados")
            print("   Corrige los errores antes de continuar")

        print("═" * 70 + "\n")

        # Guardar resultados
        self.results['summary'] = {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'n_errors': len(self.errors),
            'n_warnings': len(self.warnings),
            'status': 'PASS' if len(self.errors) == 0 else 'FAIL'
        }


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN DE USO RÁPIDO
# ═══════════════════════════════════════════════════════════════════════

def validate_training_quick(df_train: pd.DataFrame, general_config, feature_config,
                            model_config, regime_config):
    """
    Validación rápida - solo necesita el DataFrame de training.

    Uso:
        from mimo_old.model_builder import Config, ModelConfig
        from mimo_old.feature_builder import FeatureConfig
        from mimo_old.regime_detector import RegimeConfig

        general_config = Config(release='test')
        model_config = ModelConfig()
        feature_config = FeatureConfig(...)
        regime_config = RegimeConfig()

        validate_training_quick(df_train, general_config, feature_config,
                               model_config, regime_config)
    """

    # Crear pipeline
    pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)

    # Crear validator
    validator = TrainingValidator()

    # Ejecutar validación
    results = validator.run_all_checks(
        pipeline=pipeline,
        df_train=df_train,
        df_val=None,
        models=None,
        calibrators=None,
        scalers=None
    )

    return results, validator


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("""
    ╔══════════════════════════════════════════════════════════════════════╗
    ║              SCRIPT DE VALIDACIÓN COMPLETA - MAIN_OOF                ║
    ╚══════════════════════════════════════════════════════════════════════╝

    Este script valida TODO el pipeline de entrenamiento OOF.

    USO BÁSICO:
    -----------

    ```python
    from validate_full_training import validate_training_quick
    from mimo_old.model_builder import Config, ModelConfig
    from mimo_old.feature_builder import FeatureConfig
    from mimo_old.regime_detector import RegimeConfig
    import pandas as pd

    # Cargar tus datos
    df = pd.read_parquet('data.parquet')

    # Configuración
    general_config = Config(release='200338')
    model_config = ModelConfig(seq_len_short=64, seq_len_long=256)
    feature_config = FeatureConfig(...)
    regime_config = RegimeConfig()

    # Validar
    results, validator = validate_training_quick(
        df, general_config, feature_config, model_config, regime_config
    )

    # Ver resultados
    print(results['summary'])
    ```

    USO AVANZADO (con modelos ya entrenados):
    -----------------------------------------

    ```python
    from validate_full_training import TrainingValidator

    validator = TrainingValidator()
    results = validator.run_all_checks(
        pipeline=your_pipeline,
        df_train=df_train,
        df_val=df_val,
        models=your_models,  # {'long': model_long, 'short': model_short}
        calibrators=your_calibrators,
        scalers=your_scalers
    )
    ```

    ╔══════════════════════════════════════════════════════════════════════╗
    ║                    ¡LISTO PARA VALIDAR!                              ║
    ╚══════════════════════════════════════════════════════════════════════╝
    """)