import json
import os
from typing import Dict, List, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from mimo.data_managers.rolling_scaler import RollingRobustScaler
from mimo.features.feature_builder import FeatureConfig, FeatureEngineer
from mimo.features.label_generator import LabelGenerator
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig, add_mimo_state, StateDetector


def _extract_labels_weights(df: pd.DataFrame, start: int, end: int) -> tuple:
    """
    Extrae (labels, weights) del slice [start:end] del df.

    Detecta automáticamente el modo multitask por la presencia de las columnas
    'signal_long' y 'signal_short' (escritas por _dual_triple_barrier_labels).
    En multitask devuelve labels y weights como (N, 2) — primera columna LONG,
    segunda SHORT — para alimentar al modelo de dos cabezas.

    En modo single-side (legacy) devuelve labels y weights 1D.
    """
    is_multitask = ('signal_long' in df.columns) and ('signal_short' in df.columns)

    if is_multitask:
        sl = df['signal_long'].values[start:end].astype(np.float32)
        ss = df['signal_short'].values[start:end].astype(np.float32)
        labels = np.stack([sl, ss], axis=-1)

        # Weights: por ahora la misma columna 'regime_weight' replicada en
        # ambas dimensiones. Una variante futura permitirá per-side weights
        # vía columnas 'regime_weight_long' / 'regime_weight_short'.
        if 'regime_weight_long' in df.columns and 'regime_weight_short' in df.columns:
            wl = df['regime_weight_long'].values[start:end].astype(np.float32)
            ws = df['regime_weight_short'].values[start:end].astype(np.float32)
            weights = np.stack([wl, ws], axis=-1)
        elif 'regime_weight' in df.columns:
            w = df['regime_weight'].values[start:end].astype(np.float32)
            weights = np.stack([w, w], axis=-1)
        else:
            n = end - start
            weights = np.ones((n, 2), dtype=np.float32)
        return labels, weights

    # Legacy single-side
    labels = df['signal'].values[start:end].astype(np.float32) if 'signal' in df.columns else None
    if 'regime_weight' in df.columns:
        weights = df['regime_weight'].values[start:end].astype(np.float32)
    else:
        n = end - start
        weights = np.ones(n, dtype=np.float32)
    return labels, weights


class DataPipeline:
    """
    Pipeline completo de preparación de datos - VERSIÓN OPTIMIZADA

    OPTIMIZACIONES INTEGRADAS:
    1. create_sequences_by_side() optimizado - calcula features UNA vez
    2. Caché de columnas por side para evitar recálculos
    3. Aplicación vectorizada de máscaras al final

    COMPATIBILIDAD: 100% compatible con versión anterior (drop-in replacement)
    """

    def __init__(self,
                 general_config: Config,
                 feature_config: FeatureConfig,
                 model_config: ModelConfig,
                 regime_config: 'StateConfig | None' = None):

        self.general_config = general_config
        self.feature_config = feature_config
        self.model_config = model_config
        self.regime_config = regime_config

        self.feature_engineer = FeatureEngineer(feature_config)
        self.state_detector = StateDetector(regime_config if isinstance(regime_config, StateConfig) else StateConfig())
        self.label_generator = LabelGenerator(feature_config)

        self.scalers = {}
        self.is_fitted = False

        # ── Configuración del scaler (varios modos seleccionables por env vars) ─
        #
        # MODO 1 (default, LEGACY): use_rolling_scaler=True + causal_scaler_fit=False
        #   El fit muestrea TODO el train con stride (data_pipeline_v2.py:894-896
        #   + rolling_scaler.py:212-216) → look-ahead INTRA-TRAIN. NO afecta al
        #   test directamente, pero contamina el aprendizaje del modelo.
        #
        # MODO 2 (NO_ROLLING_SCALER=1): use_rolling_scaler=False
        #   Usa sklearn RobustScaler estándar fit sobre todo el train sin sampling.
        #   Sin look-ahead, pero MISMATCH con prod (que sí usa rolling+update).
        #
        # MODO 3 (CAUSAL_SCALER=1): use_rolling_scaler=True + causal_scaler_fit=True
        #   Fit causal del rolling scaler: warmup con primeras warmup_size filas
        #   + update secuencial del resto del train en chunks. El scaler queda
        #   al final con stats que reflejan los últimos window_size filas (igual
        #   estado que tendría en prod justo al cierre del train). Sin look-ahead
        #   en el fit, y CONSISTENTE con prod que sigue actualizándose live.
        #   Recomendado para experimentos honest desplegables. (Opción B)
        #
        # CAUSAL_SCALER tiene precedencia sobre NO_ROLLING_SCALER si ambos son 1.
        import os as _os
        _causal     = _os.environ.get("CAUSAL_SCALER", "0") == "1"
        _no_rolling = _os.environ.get("NO_ROLLING_SCALER", "0") == "1"
        if _causal:
            self.use_rolling_scaler = True
            self.causal_scaler_fit  = True
        else:
            self.use_rolling_scaler = not _no_rolling
            self.causal_scaler_fit  = False
        self.scaler_window_size = 2880
        self.scaler_warmup_size = 390
        self.scaler_smooth_alpha = 0.1
        self.no_scale_by_block = {
            'seq_short': { 'doji', 'hammer', 'shooting_star'},
            'seq_long': { 'trend_dir' },
            'context': { 'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
                         'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative',
                         'is_month_end', 'is_friday',
                         'is_eu_first_hour', 'is_us_first_hour', 'is_us_last_hour'}
        }

        self._last_live_time = None

        # Columnas exactas con las que se ajustó cada scaler (se puebla en load_scalers).
        # Permite que live_update_scalers_from_df seleccione siempre las mismas columnas
        # que usó el training, independientemente de cambios posteriores en el pipeline.
        # Formato: {'seq_short': [...], 'seq_long': [...], 'context_scalable': [...]}
        self._scaler_feature_cols: dict = {}

        # Política de uso en inferencia / validacion
        self.rolling_infer_policy = 'transform'

        # ═══════════════════════════════════════════════════════════════
        # OPTIMIZACIONES: Cache para evitar recálculos
        # ═══════════════════════════════════════════════════════════════
        self._all_feature_columns_cache = None
        self._column_indices_cache = {}

    def _skip_indices(self, scaler_name: str, cols: list[str]) -> list[int]:
        no_scale = self.no_scale_by_block.get(scaler_name, set())
        return [i for i, c in enumerate(cols) if c in no_scale]


    # ═══════════════════════════════════════════════════════════════════
    # MÉTODO OPTIMIZADO: create_sequences_by_side
    # ═══════════════════════════════════════════════════════════════════

    def create_sequences_by_side(self, df: pd.DataFrame, sides: tuple[str, ...] = ('long', 'short'),
                                 fit_scalers: bool = False, train: bool = False) -> Dict[str, Dict[str, np.ndarray]]:
        """
        DISEÑO DEL SCALER DE CONTEXT:
        ─────────────────────────────
        El scaler 'context' se fittea y transforma SIEMPRE con la UNIÓN completa de
        columnas de todos los sides (_get_all_feature_columns). Esto garantiza que:
          • El scaler tiene dimensión fija (N_union) independientemente del side.
          • save_scalers/load_scalers puede usar esa misma lista sin ambigüedad.
          • La selección de columnas por side se aplica DESPUÉS del escalado,
            sobre el array ya escalado (indexación vectorizada por columna).

        seq_short y seq_long se fittean con las columnas del side activo porque
        arquitecturalmente ya tienen dimensión distinta por side (las máscaras
        eliminan features binarias side-specific de las secuencias).
        """
        # ══════════════════════════════════════════════════════════════════════════
        # PRINCIPIO: fit + transform con la UNIÓN de columnas, selección por side
        # al final. Aplica igual a context, seq_short y seq_long.
        # Esto garantiza:
        #   • Scaler de dimensión fija (N_union) independientemente del side.
        #   • Un único scaler por bloque, sin sobreescritura entre iteraciones.
        #   • Las features side-specific se seleccionan DESPUÉS del escalado
        #     mediante indexación vectorizada (sin re-escalar).
        # ══════════════════════════════════════════════════════════════════════════

        all_cols = self._get_all_feature_columns()
        all_seq_short_cols = all_cols.get('sequence_short', [])
        all_seq_long_cols  = all_cols.get('sequence_long',  [])
        all_context_cols   = all_cols.get('context',        [])

        seq_len_s = self.model_config.seq_len_short
        seq_len_l = self.model_config.seq_len_long
        context_offset_base = seq_len_l - 1

        # ── Verificar columnas presentes en df ───────────────────────────────────
        for block_name, block_cols in [('seq_short', all_seq_short_cols),
                                       ('seq_long',  all_seq_long_cols),
                                       ('context',   all_context_cols)]:
            missing = [c for c in block_cols if c not in df.columns]
            if missing:
                print(f"  ⚠️  Missing {block_name} columns (union): {missing}")

        # ── Fit de los tres scalers con la unión completa (UNA SOLA VEZ) ─────────
        if fit_scalers:
            self._fit_scaler_from_raw(
                df[all_seq_short_cols].values.astype(np.float32),
                'seq_short', all_seq_short_cols
            )
            self._fit_scaler_from_raw(
                df[all_seq_long_cols].values.astype(np.float32),
                'seq_long', all_seq_long_cols
            )
            self._fit_scaler_from_raw(
                df[all_context_cols].values.astype(np.float32),
                'context', all_context_cols
            )
            self.is_fitted = True

        # ── Construir arrays escalados completos (unión) FUERA del loop ──────────
        n_samples_full = len(df) - context_offset_base
        if n_samples_full <= 0:
            return {}

        # seq_short y seq_long: construir secuencias con la unión, escalar, y luego
        # seleccionar las columnas del side mediante indexación en el eje de features.
        seq_short_all = self._create_sequences(
            df[all_seq_short_cols].values, seq_len_s
        )
        seq_long_all = self._create_sequences(
            df[all_seq_long_cols].values, seq_len_l
        )
        n_samples_from_long = len(seq_long_all)
        long_short_offset = seq_len_l - seq_len_s
        seq_short_all = seq_short_all[long_short_offset:long_short_offset + n_samples_from_long]

        seq_short_all_scaled = self._transform_scaler(seq_short_all, 'seq_short')
        seq_long_all_scaled  = self._transform_scaler(seq_long_all,  'seq_long')
        del seq_short_all, seq_long_all  # liberar memoria

        context_all_raw    = df[all_context_cols].values[context_offset_base:context_offset_base + n_samples_full]
        context_all_scaled = self._transform_scaler(context_all_raw.astype(np.float32), 'context')

        # Mapas columna → índice para selección vectorizada
        seq_short_col_to_idx = {c: i for i, c in enumerate(all_seq_short_cols)}
        seq_long_col_to_idx  = {c: i for i, c in enumerate(all_seq_long_cols)}
        ctx_col_to_idx       = {c: i for i, c in enumerate(all_context_cols)}

        # ── Loop por side: solo selección de columnas, sin re-escalar ────────────
        out = {}
        for side in sides:
            self.feature_engineer.set_side(side)
            self.feature_engineer._assign_features_to_inputs()
            cols = self.feature_engineer.feature_columns

            n_samples = n_samples_from_long  # mismo nº de muestras para todos los sides

            # Índices de las columnas del side dentro de los arrays de la unión
            ss_idx  = np.array([seq_short_col_to_idx[c] for c in cols['sequence_short']
                                 if c in seq_short_col_to_idx], dtype=np.int32)
            sl_idx  = np.array([seq_long_col_to_idx[c]  for c in cols['sequence_long']
                                 if c in seq_long_col_to_idx],  dtype=np.int32)
            ctx_idx = np.array([ctx_col_to_idx[c]        for c in cols['context']
                                 if c in ctx_col_to_idx],        dtype=np.int32)

            # Selección vectorizada: (batch, seq_len, features) → [:, :, side_idx]
            seq_short_side = seq_short_all_scaled[:n_samples, :, :][:, :, ss_idx]
            seq_long_side  = seq_long_all_scaled[:n_samples,  :, :][:, :, sl_idx]
            context_side   = context_all_scaled[:n_samples, :][:, ctx_idx]

            time_vals = df[cols['time']].values[context_offset_base:context_offset_base + n_samples]

            if train:
                if 'signal' not in df.columns and 'signal_long' not in df.columns:
                    raise ValueError(
                        'Training mode requires "signal" or "signal_long"+"signal_short" '
                        'columns in DataFrame'
                    )
                labels, weights = _extract_labels_weights(
                    df, context_offset_base, context_offset_base + n_samples,
                )
            else:
                labels  = None
                weights = None

            out[side] = {
                'seq_short': seq_short_side.astype(np.float32),
                'seq_long':  seq_long_side.astype(np.float32),
                'context':   context_side.astype(np.float32),
                'time':      time_vals,
                'labels':    labels  if labels  is not None else None,
                'weights':   weights if weights is not None else None,
            }

        return out

    def _get_all_feature_columns(self) -> Dict[str, List[str]]:
        """
        Obtiene la UNIÓN de todas las features (long + short).
        Se cachea para evitar recalcular.
        """
        if self._all_feature_columns_cache is not None:
            return self._all_feature_columns_cache

        # Guardar side actual
        original_side = self.feature_engineer.side

        # Obtener features para long
        self.feature_engineer.set_side('long')
        self.feature_engineer._assign_features_to_inputs()
        long_cols = {k: list(v) for k, v in self.feature_engineer.feature_columns.items()}

        # Obtener features para short
        self.feature_engineer.set_side('short')
        self.feature_engineer._assign_features_to_inputs()
        short_cols = {k: list(v) for k, v in self.feature_engineer.feature_columns.items()}

        # Unión (sin duplicados)
        all_cols = {}
        for key in long_cols.keys():
            all_cols[key] = sorted(set(long_cols[key] + short_cols[key]))

        # Restaurar side original
        self.feature_engineer.set_side(original_side)

        # Cachear
        self._all_feature_columns_cache = all_cols

        return all_cols

    def _create_sequences_base(self, df: pd.DataFrame, feature_cols: Dict[str, List[str]],
                               fit_scalers: bool, train: bool) -> Dict[str, np.ndarray]:
        """
        Crea secuencias con TODAS las features (antes de aplicar máscaras).
        Similar a create_sequences() pero con feature_cols completo.
        """

        # 1. Secuencias cortas
        seq_short_data = df[feature_cols['sequence_short']].values
        seq_short = self._create_sequences(seq_short_data, self.model_config.seq_len_short)

        # 2. Secuencias largas
        seq_long_data = df[feature_cols['sequence_long']].values
        seq_long = self._create_sequences(seq_long_data, self.model_config.seq_len_long)

        # 3. Alinear longitudes
        n_samples = len(seq_long)
        offset = self.model_config.seq_len_long - self.model_config.seq_len_short
        seq_short = seq_short[offset:offset + n_samples]

        # 4. Context y time
        context_offset = self.model_config.seq_len_long - 1
        context = df[feature_cols['context']].values[context_offset:context_offset + n_samples]
        time = df[feature_cols['time']].values[context_offset:context_offset + n_samples]

        # 5. Labels (si aplica) — soporta multitask vía _extract_labels_weights
        if train:
            labels, weights = _extract_labels_weights(
                df, context_offset, context_offset + n_samples,
            )
        else:
            labels = None
            weights = None

        # 6. Escalar datos (esto es común para ambos lados)
        context_cols = feature_cols['context']
        if fit_scalers:
            seq_short = self._fit_transform_scaler(seq_short, 'seq_short', feature_cols['sequence_short'])
            seq_long = self._fit_transform_scaler(seq_long, 'seq_long', feature_cols['sequence_long'])
            context = self._fit_transform_scaler(context, 'context', feature_cols['context'])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scalers no han sido ajustados. Ejecuta primero con fit_scalers=True")
            seq_short = self._transform_scaler(seq_short, 'seq_short')
            seq_long = self._transform_scaler(seq_long, 'seq_long')
            context = self._transform_scaler(context, 'context')

        # Retornar con metadatos de columnas
        return {
            'seq_short': seq_short.astype(np.float32),
            'seq_short_cols': feature_cols['sequence_short'],
            'seq_long': seq_long.astype(np.float32),
            'seq_long_cols': feature_cols['sequence_long'],
            'context': context.astype(np.float32),
            'context_cols': feature_cols['context'],
            'time': time.astype(np.float32),
            'time_cols': feature_cols['time'],
            'labels': labels.astype(np.float32) if labels is not None else None,
            'weights': weights.astype(np.float32) if weights is not None else None
        }

    def _apply_side_mask(self, base_data: Dict, side: str,
                         full_feature_cols: Dict[str, List[str]]) -> Dict[str, np.ndarray]:
        """
        Aplica máscara de features para un side específico.
        SOLO selecciona columnas, NO recalcula nada.

        Esta es la clave de la optimización: indexación vectorizada ultrarrápida.
        """

        # Obtener columnas para este side
        side_cols = self._get_side_columns(side)

        result = {}

        for key in ['seq_short', 'seq_long', 'context', 'time']:
            data = base_data[key]
            full_cols = base_data[f'{key}_cols']
            side_specific_cols = side_cols[key.replace('seq_', 'sequence_')]

            # Obtener índices (con caché para no recalcular)
            cache_key = (tuple(full_cols), side, key)
            if cache_key not in self._column_indices_cache:
                indices = [i for i, col in enumerate(full_cols) if col in side_specific_cols]
                self._column_indices_cache[cache_key] = np.array(indices, dtype=np.int32)

            col_indices = self._column_indices_cache[cache_key]

            # Selección vectorizada (MUY rápida)
            if len(data.shape) == 3:  # secuencias (batch, seq_len, features)
                result[key] = data[:, :, col_indices]
            elif len(data.shape) == 2:  # context/time (batch, features)
                result[key] = data[:, col_indices]
            else:
                result[key] = data

        # Labels y weights son iguales para ambos lados
        result['labels'] = base_data.get('labels')
        result['weights'] = base_data.get('weights')

        return result

    def _get_side_columns(self, side: str) -> Dict[str, List[str]]:
        """Obtiene columnas para un side específico"""
        original_side = self.feature_engineer.side
        self.feature_engineer.set_side(side)
        self.feature_engineer._assign_features_to_inputs()
        cols = {k: list(v) for k, v in self.feature_engineer.feature_columns.items()}
        self.feature_engineer.set_side(original_side)
        return cols

    # ═══════════════════════════════════════════════════════════════════
    # MÉTODOS ORIGINALES (sin cambios)
    # ═══════════════════════════════════════════════════════════════════

    def prepare_data(self, df: pd.DataFrame, labels: bool = True, side: str | None = 'long',
                     set_market_condition: bool = False, ensure_regime: bool = True) -> pd.DataFrame:
        """Pipeline completo de preparación de datos."""

        if side is None:
            side = 'both'

        self.feature_engineer.set_side(side)

        # 1. Generar features
        df = self.feature_engineer.generate_all_features(df)

        # 2. Detectar estado de mercado (StateDetector unificado)
        # FIX runtime drift 2026-05-20: usar `self.state_detector.config`
        # directamente en lugar de reconstruir un StateConfig desde el dict
        # `_state_config_overrides`. El attribute path indirecto fallaba:
        # cuando load_scalers invocaba _inject_regime_thresholds, los
        # `fixed_*` se seteaban en `self.state_detector.config` pero
        # `_state_config_overrides` no siempre llegaba a ser leído en
        # add_mimo_state (timing/instance issues). Resultado: cada prepare_data
        # creaba un StateDetector NUEVO sin thresholds inyectados y caía al
        # path DINÁMICO, recomputando vol_high/low sobre el buffer corto del
        # runtime (~1,128 barras) en lugar de usar los persistidos en meta.json.
        #
        # Síntoma: 68% del tiempo etiquetado VOLATILE incluso tras recalibrar
        # thresholds en meta.json. Confirmado por el warning `[StateDetector]
        # Computing thresholds DYNAMICALLY from the current df` apareciendo en
        # cada tick. Ver docs/RUNBOOK.md → Apéndice C.
        #
        # Pasando `self.state_detector.config` garantizamos que el StateDetector
        # interno de add_mimo_state hereda EXACTAMENTE el cfg cuyos `fixed_*`
        # fueron poblados por inject_thresholds. Retrocompat: cuando no se ha
        # inyectado nada (training/Optuna), config.fixed_* son None y el path
        # dinámico se mantiene — mismo comportamiento que antes.
        df = add_mimo_state(
            df,
            cfg=self.state_detector.config,
            set_market_condition=set_market_condition,
        )

        # Alias: state_weight -> regime_weight para compatibilidad con código existente
        if 'state_weight' in df.columns and 'regime_weight' not in df.columns:
            df['regime_weight'] = df['state_weight']

        # 3. Generar etiquetas
        if labels:
            print(f"[DIAG PRE] side={side} | regime_barriers_short set: {self.label_generator.config.regime_barriers_short is not None} | regime_barriers: {self.label_generator.config.regime_barriers}")

            df = self.label_generator.generate_labels(df, side)
            # En modo quantile_return el target es continuo (return en ATR),
            # por lo que pos_rate/positivos no aplican: reportamos mean/std
            # con etiqueta "mean_return" para evitar confusión.
            if self.label_generator.config.label_method == 'quantile_return':
                _s = df['signal']
                print(
                    f"[DIAG {side}] mean_return(ATR)={_s.mean():.4f} "
                    f"std={_s.std():.4f} min={_s.min():.4f} max={_s.max():.4f} "
                    f"n_finite={int(_s.notna().sum())}/{len(df)}"
                )
                print(f"[DIAG {side}] por régimen (mean_return / std / count):")
                print(df.groupby('state')['signal'].agg(['mean', 'std', 'count']))
            else:
                print(f"[DIAG {side}] pos_rate: {df['signal'].mean():.4f} ({df['signal'].sum()} positivos de {len(df)} filas)")
                print(f"[DIAG {side}] por régimen:")
                print(df.groupby('state')['signal'].agg(['mean', 'sum', 'count']))

            df = df.dropna()
        else:
            df = df.ffill().bfill().fillna(0)

        return df

    def _scale_context_with_exclusions(self, context: np.ndarray, context_cols: list[str], fit: bool) -> np.ndarray:
        """Escala solo features continuas, preserva binarias sin modificar (sin cambios)"""

        # Features binarias que NO se escalan
        no_scale = {
            'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
            'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative',
            'is_month_end', 'is_friday',
            'is_eu_first_hour', 'is_us_first_hour', 'is_us_last_hour'
        }

        scale_mask = np.array([c not in no_scale for c in context_cols])

        if not scale_mask.any():
            return context

        out = context.copy()
        context_to_scale = context[:, scale_mask]

        scaler_key = 'context'

        if fit:
            context_scaled = self._fit_transform_scaler(context_to_scale, scaler_key)
        else:
            context_scaled = self._transform_scaler(context_to_scale, scaler_key)

        out[:, scale_mask] = context_scaled

        return out

    def create_last_sample_by_side_from_index(
            self,
            df: pd.DataFrame,
            end_idx: int,
            side: str,
            fit_scalers: bool = False,
            train: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Construye SOLO la última muestra/ventana para un side usando df completo + end_idx,
        evitando crear df_slice = df.iloc[:end_idx] en cada iteración.

        end_idx es EXCLUSIVO, igual que en iloc[:end_idx].
        """

        if end_idx <= 0 or end_idx > len(df):
            raise ValueError(f"end_idx fuera de rango: {end_idx} para len(df)={len(df)}")

        # ---- 1) Columnas base del scaler (preferidas) ----
        union_cols = self._get_all_feature_columns()

        all_seq_short_cols = self._scaler_feature_cols.get("seq_short") \
                             or union_cols.get("sequence_short", [])
        all_seq_long_cols = self._scaler_feature_cols.get("seq_long") \
                            or union_cols.get("sequence_long", [])
        all_context_cols = self._scaler_feature_cols.get("context_all") \
                           or union_cols.get("context", [])

        # ---- 2) Columnas del side activo ----
        self.feature_engineer.set_side(side)
        self.feature_engineer._assign_features_to_inputs()
        side_cols = self.feature_engineer.feature_columns

        seq_len_s = self.model_config.seq_len_short
        seq_len_l = self.model_config.seq_len_long

        if end_idx < seq_len_l:
            raise ValueError(
                f"No hay suficientes filas para construir la última muestra: "
                f"{end_idx} < seq_len_long={seq_len_l}"
            )

        s0 = end_idx - seq_len_s
        l0 = end_idx - seq_len_l
        c0 = end_idx - 1

        # ---- 3) Construir bloques usando las columnas del scaler ----
        seq_short_all = df.iloc[s0:end_idx][all_seq_short_cols].values.astype(np.float32, copy=False)[None, :, :]
        seq_long_all = df.iloc[l0:end_idx][all_seq_long_cols].values.astype(np.float32, copy=False)[None, :, :]
        context_all = df.iloc[c0:c0 + 1][all_context_cols].values.astype(np.float32, copy=False)
        time_vals = df.iloc[c0:c0 + 1][side_cols["time"]].values.astype(np.float32, copy=False)

        # ---- 4) Escalar ----
        if fit_scalers:
            seq_short_all = self._fit_transform_scaler(seq_short_all, "seq_short", all_seq_short_cols)
            seq_long_all = self._fit_transform_scaler(seq_long_all, "seq_long", all_seq_long_cols)
            context_all = self._fit_transform_scaler(context_all, "context", all_context_cols)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError(
                    "Scalers no ajustados. Ejecuta primero load_scalers() o fit_scalers=True."
                )

            seq_short_all = self._transform_scaler(seq_short_all, "seq_short")
            seq_long_all = self._transform_scaler(seq_long_all, "seq_long")
            context_all = self._transform_scaler(context_all, "context")

        # ---- 5) Selección side-specific DESPUÉS del escalado ----
        seq_short_col_to_idx = {c: i for i, c in enumerate(all_seq_short_cols)}
        seq_long_col_to_idx = {c: i for i, c in enumerate(all_seq_long_cols)}
        context_col_to_idx = {c: i for i, c in enumerate(all_context_cols)}

        ss_idx = np.array(
            [seq_short_col_to_idx[c] for c in side_cols["sequence_short"] if c in seq_short_col_to_idx],
            dtype=np.int32
        )
        sl_idx = np.array(
            [seq_long_col_to_idx[c] for c in side_cols["sequence_long"] if c in seq_long_col_to_idx],
            dtype=np.int32
        )
        ctx_idx = np.array(
            [context_col_to_idx[c] for c in side_cols["context"] if c in context_col_to_idx],
            dtype=np.int32
        )

        seq_short = seq_short_all[:, :, ss_idx]
        seq_long = seq_long_all[:, :, sl_idx]
        context = context_all[:, ctx_idx]

        if train:
            labels = df.iloc[c0:c0 + 1]["signal"].values.astype(np.float32)
            weights = (
                df.iloc[c0:c0 + 1]["regime_weight"].values.astype(np.float32)
                if "regime_weight" in df.columns
                else np.ones(1, dtype=np.float32)
            )
        else:
            labels = None
            weights = None

        return {
            "seq_short": seq_short.astype(np.float32),
            "seq_long": seq_long.astype(np.float32),
            "context": context.astype(np.float32),
            "time": time_vals.astype(np.float32),
            "labels": labels,
            "weights": weights,
        }

    def create_last_sample_by_side(
            self,
            df: pd.DataFrame,
            side: str,
            fit_scalers: bool = False,
            train: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Construye SOLO la última muestra/ventana para un side.

        Diseño:
          - escala SIEMPRE con las columnas exactas del scaler cargado
            (_scaler_feature_cols) si están disponibles
          - hace fallback a _get_all_feature_columns() si no lo están
          - selecciona columnas side-specific DESPUÉS del escalado

        Esto evita mismatches tipo:
          context side=16 columnas vs scaler context=19 columnas
        """

        union_cols = self._get_all_feature_columns()

        all_seq_short_cols = self._scaler_feature_cols.get("seq_short") or union_cols.get("sequence_short", [])
        all_seq_long_cols = self._scaler_feature_cols.get("seq_long") or union_cols.get("sequence_long", [])
        all_context_cols = self._scaler_feature_cols.get("context_all") or union_cols.get("context", [])

        self.feature_engineer.set_side(side)
        self.feature_engineer._assign_features_to_inputs()
        side_cols = self.feature_engineer.feature_columns

        seq_len_s = self.model_config.seq_len_short
        seq_len_l = self.model_config.seq_len_long

        if len(df) < seq_len_l:
            raise ValueError(
                f"No hay suficientes filas para construir la última muestra: "
                f"{len(df)} < seq_len_long={seq_len_l}"
            )

        for colset_name, colset in {
            "all_seq_short_cols": all_seq_short_cols,
            "all_seq_long_cols": all_seq_long_cols,
            "all_context_cols": all_context_cols,
        }.items():
            missing = [c for c in colset if c not in df.columns]
            if missing:
                raise ValueError(
                    f"Faltan columnas requeridas en df para {colset_name}: {missing[:10]}"
                    + (" ..." if len(missing) > 10 else "")
                )

        seq_short_all = df[all_seq_short_cols].values[-seq_len_s:].astype(np.float32, copy=False)[None, :, :]
        seq_long_all = df[all_seq_long_cols].values[-seq_len_l:].astype(np.float32, copy=False)[None, :, :]
        context_all = df[all_context_cols].values[-1:].astype(np.float32, copy=False)
        time_vals = df[side_cols["time"]].values[-1:].astype(np.float32, copy=False)

        if fit_scalers:
            seq_short_all = self._fit_transform_scaler(seq_short_all, "seq_short", all_seq_short_cols)
            seq_long_all = self._fit_transform_scaler(seq_long_all, "seq_long", all_seq_long_cols)
            context_all = self._fit_transform_scaler(context_all, "context", all_context_cols)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError(
                    "Scalers no ajustados. Ejecuta primero load_scalers() o fit_scalers=True."
                )
            seq_short_all = self._transform_scaler(seq_short_all, "seq_short")
            seq_long_all = self._transform_scaler(seq_long_all, "seq_long")
            context_all = self._transform_scaler(context_all, "context")

        seq_short_col_to_idx = {c: i for i, c in enumerate(all_seq_short_cols)}
        seq_long_col_to_idx = {c: i for i, c in enumerate(all_seq_long_cols)}
        context_col_to_idx = {c: i for i, c in enumerate(all_context_cols)}

        ss_idx = np.array([seq_short_col_to_idx[c] for c in side_cols["sequence_short"] if c in seq_short_col_to_idx], dtype=np.int32)
        sl_idx = np.array([seq_long_col_to_idx[c] for c in side_cols["sequence_long"] if c in seq_long_col_to_idx], dtype=np.int32)
        ctx_idx = np.array([context_col_to_idx[c] for c in side_cols["context"] if c in context_col_to_idx], dtype=np.int32)

        seq_short = seq_short_all[:, :, ss_idx]
        seq_long = seq_long_all[:, :, sl_idx]
        context = context_all[:, ctx_idx]

        if train:
            labels = df["signal"].values[-1:].astype(np.float32)
            weights = (
                df["regime_weight"].values[-1:].astype(np.float32)
                if "regime_weight" in df.columns
                else np.ones(1, dtype=np.float32)
            )
        else:
            labels = None
            weights = None

        return {
            "seq_short": seq_short.astype(np.float32),
            "seq_long": seq_long.astype(np.float32),
            "context": context.astype(np.float32),
            "time": time_vals.astype(np.float32),
            "labels": labels,
            "weights": weights,
        }

    def create_sequences(self, df: pd.DataFrame, fit_scalers: bool = False, train: bool = False) -> Dict[
        str, np.ndarray]:
        """
        Crea secuencias y prepara datos para el modelo (sin cambios)

        NOTA: Este método se mantiene para compatibilidad hacia atrás.
        create_sequences_by_side() ahora es más eficiente.
        """
        feature_cols = self.feature_engineer.feature_columns

        # Verificar compatibilidad con scalers cargados
        if not fit_scalers and self.is_fitted and 'context' in self.scalers:
            scaler = self.scalers['context']
            if hasattr(scaler, 'stats_'):
                n_features_saved = scaler.stats_['center'].shape[0]
                no_scale = {'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
                            'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative',
                            'is_month_end', 'is_friday',
                            'is_eu_first_hour', 'is_us_first_hour', 'is_us_last_hour'}
                context_cols = feature_cols.get('context', [])
                n_features_to_scale = sum(1 for c in context_cols if c not in no_scale)

                if n_features_saved != n_features_to_scale:
                    raise ValueError(
                        f"Scaler dimension mismatch! "
                        f"Loaded scaler expects {n_features_saved} features, "
                        f"but current config has {n_features_to_scale} scalable features "
                        f"(out of {len(context_cols)} total context features). "
                        f"Context columns: {context_cols}"
                    )

        # Preparar datos
        data = {}

        # 1. Secuencias cortas
        seq_short_data = df[feature_cols['sequence_short']].values
        seq_short = self._create_sequences(seq_short_data, self.model_config.seq_len_short)

        # 2. Secuencias largas
        seq_long_data = df[feature_cols['sequence_long']].values
        seq_long = self._create_sequences(seq_long_data, self.model_config.seq_len_long)

        # 3. Alinear longitudes
        n_samples = len(seq_long)
        offset = self.model_config.seq_len_long - self.model_config.seq_len_short
        seq_short = seq_short[offset:offset + n_samples]

        # 4. Context y time features
        context_offset = self.model_config.seq_len_long - 1
        context = df[feature_cols['context']].values[context_offset:context_offset + n_samples]
        time = df[feature_cols['time']].values[context_offset:context_offset + n_samples]

        # 5. Labels y pesos — soporta multitask vía _extract_labels_weights
        if train:
            labels, weights = _extract_labels_weights(
                df, context_offset, context_offset + n_samples,
            )
        else:
            labels = None
            weights = None

        # 6. Escalar datos
        context_cols = feature_cols['context']
        if fit_scalers:
            seq_short = self._fit_transform_scaler(seq_short, 'seq_short', feature_cols['sequence_short'])
            seq_long = self._fit_transform_scaler(seq_long, 'seq_long', feature_cols['sequence_long'])
            context = self._fit_transform_scaler(context, 'context', feature_cols['context'])
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scalers no han sido ajustados. Ejecuta primero con fit_scalers=True")
            seq_short = self._transform_scaler(seq_short, 'seq_short')
            seq_long = self._transform_scaler(seq_long, 'seq_long')
            context = self._transform_scaler(context, 'context')

        return {
            'seq_short': seq_short.astype(np.float32),
            'seq_long': seq_long.astype(np.float32),
            'context': context.astype(np.float32),
            'time': time.astype(np.float32),
            'labels': labels.astype(np.float32) if labels is not None else None,
            'weights': weights.astype(np.float32) if weights is not None else None
        }

    @staticmethod
    def _create_sequences_v0(data: np.ndarray, seq_len: int) -> np.ndarray:
        """Crea secuencias de longitud fija (sin cambios)"""
        sequences = []
        for i in range(len(data) - seq_len + 1):
            sequences.append(data[i:i + seq_len])
        return np.array(sequences)

    @staticmethod
    def _create_sequences(data: np.ndarray, seq_len: int) -> np.ndarray:
        from numpy.lib.stride_tricks import as_strided
        n = len(data) - seq_len + 1
        if n <= 0:
            return np.empty((0, seq_len, data.shape[1] if data.ndim > 1 else 1), dtype=data.dtype)
        shape = (n, seq_len) + data.shape[1:]
        strides = (data.strides[0],) + data.strides
        return np.ascontiguousarray(as_strided(data, shape=shape, strides=strides))

    def _fit_transform_scaler(self, data: np.ndarray, name: str, feature_cols: list[str] | None = None) -> np.ndarray:
        """Ajusta y transforma con scaler (sin cambios)"""
        original_shape = data.shape

        if len(original_shape) == 3:
            data_2d = data.reshape(-1, original_shape[-1])
            del data
            data = data_2d

        skip = None
        if feature_cols is not None:
            skip = self._skip_indices(name, feature_cols)

        if self.use_rolling_scaler:
            scaler = RollingRobustScaler(
                window_size=self.scaler_window_size,
                warmup_size=self.scaler_warmup_size,
                quantile_range=(25.0, 75.0),
                smooth_alpha=self.scaler_smooth_alpha,
                clip=8.0,
                nan_policy='zero',
                scale_before_warmup=True,
                skip_features=skip,
                min_iqr=1e-4,
                feature_names=list(feature_cols) if feature_cols is not None else None,
                name=str(name),
            )

            data_scaled = scaler.fit_transform(data)
            del data

            progress = scaler.get_warmup_progress() * 100.0
            print(f'\tScaler {name}: warmup {progress:.1f}%')
        else:
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            data_scaled = scaler.fit_transform(data)

        self.scalers[name] = scaler

        if len(original_shape) == 3:
            data_scaled = data_scaled.reshape(original_shape)

        return data_scaled

    def _fit_scaler_from_raw(self, raw_data: np.ndarray, name: str, feature_cols: list[str] | None = None) -> None:
        """
        Fittea el scaler directamente desde datos 2D (sin construir secuencias).
        Evita materializar el array 3D completo solo para submuestrear.
        raw_data: shape (n_rows, n_features) — el df.values antes de crear secuencias
        """
        skip = None
        if feature_cols is not None:
            skip = self._skip_indices(name, feature_cols)

        if self.use_rolling_scaler:
            scaler = RollingRobustScaler(
                window_size=self.scaler_window_size,
                warmup_size=self.scaler_warmup_size,
                quantile_range=(25.0, 75.0),
                smooth_alpha=self.scaler_smooth_alpha,
                clip=8.0,
                nan_policy='zero',
                scale_before_warmup=True,
                skip_features=skip,
                min_iqr=1e-4,
                feature_names=list(feature_cols) if feature_cols is not None else None,
                name=str(name),
            )
            n_rows = raw_data.shape[0]

            if self.causal_scaler_fit:
                # Modo CAUSAL: warmup con primeras warmup_size filas + update
                # secuencial del resto en chunks. Replica el régimen que tendría
                # el scaler en producción justo al final del train (después de
                # haber procesado todas las barras causalmente). No hay stride
                # sobre todo el train → no hay look-ahead intra-train en el fit.
                warmup_n = min(self.scaler_warmup_size, n_rows)
                scaler.fit(raw_data[:warmup_n].astype(np.float32))
                # Resto del train: update secuencial en chunks pequeños
                # (chunk_size=128 mantiene buena dinámica rolling sin matar
                # performance — recompute_every del scaler controla coste real).
                if n_rows > warmup_n:
                    chunk_size = 128
                    for start in range(warmup_n, n_rows, chunk_size):
                        end = min(start + chunk_size, n_rows)
                        scaler.update(raw_data[start:end].astype(np.float32))
                progress = scaler.get_warmup_progress() * 100.0
                print(f'\tScaler {name}: causal fit warmup {progress:.1f}%')
            else:
                # Modo LEGACY: stride sobre todo el train → look-ahead intra-train
                # (la fila 0 se escala con stats de filas posteriores del mismo
                # train al transformarse después). Mantener solo por compatibilidad.
                if n_rows > self.scaler_window_size:
                    stride = max(1, n_rows // self.scaler_window_size)
                    sample = np.ascontiguousarray(raw_data[::stride][:self.scaler_window_size].astype(np.float32))
                else:
                    sample = raw_data.astype(np.float32)

                scaler.fit(sample)
                del sample
                progress = scaler.get_warmup_progress() * 100.0
                print(f'\tScaler {name}: warmup {progress:.1f}%')
        else:
            from sklearn.preprocessing import RobustScaler
            scaler = RobustScaler(quantile_range=(25.0, 75.0))
            scaler.fit(raw_data)

        self.scalers[name] = scaler

    def _transform_scaler(self, data: np.ndarray, name: str) -> np.ndarray:
        """Transforma con scaler existente (sin cambios)"""
        original_shape = data.shape

        if len(original_shape) == 3:
            data = data.reshape(-1, original_shape[-1])

        scaler = self.scalers[name]

        if isinstance(scaler, RollingRobustScaler):
            '''
            if hasattr(scaler, 'scale_') and scaler.scale_ is not None:
                tiny = scaler.scale_ < 1e-3
                if tiny.any():
                    cols = self.feature_engineer.feature_columns
                    col_map = {
                        'seq_short': cols.get('sequence_short', []),
                        'seq_long': cols.get('sequence_long', []),
                        'context': cols.get('context', []),
                    }
                    col_names = col_map.get(name, [])
                    for idx in np.where(tiny)[0]:
                        col_name = col_names[idx] if idx < len(col_names) else '???'
                        print(f"[TINY_SCALE] {name}[{idx}] '{col_name}': scale={scaler.scale_[idx]:.2e}")
            '''

            if self.rolling_infer_policy == 'transform_then_update':
                data_scaled = scaler.transform_then_update(data)
            else:
                data_scaled = scaler.transform(data)
        else:
            data_scaled = scaler.transform(data)

        if len(original_shape) == 3:
            data_scaled = data_scaled.reshape(original_shape)

        return data_scaled

    def update_rolling_scalers(self, updates: Dict[str, np.ndarray]) -> None:
        """Actualiza scaler rolling con filas nuevas (sin cambios)"""
        for name, x_new in updates.items():
            if name not in self.scalers:
                continue
            scaler = self.scalers[name]
            if isinstance(scaler, RollingRobustScaler):
                scaler.update(x_new)

    def scalers_dir(self, base_path: str | None = None) -> str:
        """Directorio de scalers (sin cambios)"""
        if base_path is None:
            base_path = '../../artifacts'
        return os.path.join(base_path, f'scalers_{self.general_config.release}')

    def compute_and_store_regime_thresholds(self, df_prepared: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula los umbrales de estado sobre el DataFrame de entrenamiento y los
        almacena en el pipeline para que save_scalers los persista.

        Llamar UNA VEZ tras prepare_data() antes de save_scalers().
        En producción, load_scalers() los inyecta automáticamente en StateDetector.

        Args:
            df_prepared: DataFrame ya procesado por prepare_data() (con atr_norm,
                         bb_width, range_expansion calculados).

        Returns:
            Diccionario con los 6 umbrales calculados.
        """
        # Delegar en StateDetector (cálculo unificado)
        thresholds = self.state_detector.compute_thresholds(df_prepared)
        self._regime_thresholds = thresholds
        return thresholds

    def _inject_regime_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Inyecta umbrales fijos en StateDetector.
        Se llama automáticamente desde load_scalers().
        """
        self.state_detector.inject_thresholds(thresholds)
        # También guardamos overrides para que prepare_data() los use
        # (compatibilidad con código que lee _state_config_overrides directamente)
        self._state_config_overrides = {
            'fixed_vol_low':              thresholds.get('vol_low'),
            'fixed_vol_high':             thresholds.get('vol_high'),
            'fixed_bb_width_p20':         thresholds.get('bb_p20'),
            'fixed_bb_width_p35':         thresholds.get('bb_p35'),
            'fixed_bb_width_p70':         thresholds.get('bb_p70'),
            'fixed_range_expansion_p80':  thresholds.get('rexp_p80'),
        }

    def save_scalers(self, base_path: str | None = None, *, include_buffer: bool = False) -> str:
        """Guarda scalers (sin cambios)"""
        path = self.scalers_dir(base_path)
        os.makedirs(path, exist_ok=True)

        # IMPORTANTE: usar la UNIÓN completa de columnas (todos los sides), no el side
        # activo en este momento. El scaler de context se fittea con _get_all_feature_columns()
        # (antes de aplicar máscaras por side), así que meta.json debe reflejar eso.
        # Si se usara feature_engineer.feature_columns directamente se capturaría solo el
        # último side activo (p.ej. 'short' con 11 cols) → mismatch vs scaler (16 cols).
        feature_cols   = self._get_all_feature_columns()
        context_cols   = feature_cols.get('context', [])
        seq_short_cols = feature_cols.get('sequence_short', [])
        seq_long_cols  = feature_cols.get('sequence_long', [])
        no_scale_by_block = {
            'seq_short': ['doji', 'hammer', 'shooting_star'],
            'seq_long': ['trend_dir'],
            'context': ['is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear', 'rsi_oversold', 'rsi_overbought',
                        'macd_positive', 'macd_negative',
                        'is_month_end', 'is_friday',
                        'is_eu_first_hour', 'is_us_first_hour', 'is_us_last_hour']
        }

        # Calcular columnas escalables del contexto (sin no_scale) para auditoría
        _no_scale_ctx = set(no_scale_by_block['context'])
        context_scalable_cols = [c for c in context_cols if c not in _no_scale_ctx]

        # Verificación de consistencia: el scaler context recibe TODAS las columnas
        # (incluyendo binarias via skip_features), por tanto n_features == len(context_cols).
        # Comparar contra context_cols total, no contra context_scalable_cols.
        if 'context' in self.scalers:
            _scaler_ctx = self.scalers['context']
            _n_feat = getattr(_scaler_ctx, 'n_features', None) or (
                _scaler_ctx.stats_['center'].shape[0]
                if hasattr(_scaler_ctx, 'stats_') else None
            )
            if _n_feat is not None and _n_feat != len(context_cols):
                print(f"\tWARN save_scalers: context_cols={len(context_cols)} "
                      f"vs scaler.n_features={_n_feat} — revisar _get_all_feature_columns()")
            else:
                print(f"\tSave scalers context OK: {len(context_cols)} cols total "
                      f"({len(context_scalable_cols)} escalables + "
                      f"{len(context_cols) - len(context_scalable_cols)} binarias skip_features)")

        meta = {
            'release': str(self.general_config.release),
            'use_rolling_scaler': bool(self.use_rolling_scaler),
            'scaler_window_size': int(self.scaler_window_size),
            'scaler_warmup_size': int(self.scaler_warmup_size),
            'scaler_smooth_alpha': float(self.scaler_smooth_alpha),
            'rolling_infer_policy': str(self.rolling_infer_policy),
            'scalers': list(self.scalers.keys()),
            # Columnas completas del contexto (total = scalable + no_scale)
            'context_columns': context_cols,
            # Columnas escalables del contexto: las que el scaler realmente vio.
            # live_update_scalers_from_df usa esta lista para ser robusto a cambios
            # posteriores del pipeline (nuevas features, feature_masks distintos, etc.)
            'context_scalable_columns': context_scalable_cols,
            # Columnas de secuencias (para futuro uso en live_update)
            'seq_short_columns': seq_short_cols,
            'seq_long_columns': seq_long_cols,
            'version': 'data_pipeline_v5.scaler_cols',
            'no_scale_by_block': no_scale_by_block,
            # Umbrales de régimen calculados sobre el conjunto de entrenamiento.
            # Presentes solo si se llamó compute_and_store_regime_thresholds() antes.
            'regime_thresholds': getattr(self, '_regime_thresholds', None),
        }

        with open(os.path.join(path, 'meta.json'), 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        for name, scaler in self.scalers.items():
            if isinstance(scaler, RollingRobustScaler):
                fp = os.path.join(path, f'{name}.json')
                scaler.save(fp, include_buffer=include_buffer)
            else:
                fp = os.path.join(path, f'{name}.joblib')
                joblib.dump(scaler, fp, compress=3)

        self.is_fitted = True
        print(f'\tScalers saved to: {path}')
        return path

    def load_scalers(self, base_path: str | None = None, *, strict: bool = True) -> str:
        """Carga scalers (sin cambios)"""
        path = self.scalers_dir(base_path)
        if not os.path.exists(path):
            raise FileNotFoundError(f'Scalers path was not found: {path}')

        meta_path = os.path.join(path, 'meta.json')
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)

            if 'context_scale_mask' in meta:
                current_context_cols = self.feature_engineer.feature_columns.get('context', [])
                saved_context_cols = meta.get('context_columns', [])

                if current_context_cols and current_context_cols != saved_context_cols:
                    print(f"\tWARNING: Context columns differ!")
                    print(f"\t  Saved: {saved_context_cols}")
                    print(f"\t  Current: {current_context_cols}")

        loaded = {}
        missing = []

        for name in ['seq_short', 'seq_long', 'context']:
            json_path = os.path.join(path, f'{name}.json')
            joblib_path = os.path.join(path, f'{name}.joblib')

            if os.path.exists(json_path):
                loaded[name] = RollingRobustScaler.load(json_path)
            elif os.path.exists(joblib_path):
                loaded[name] = joblib.load(joblib_path)
            else:
                missing.append(name)

        if strict and missing:
            raise FileNotFoundError(f'Missing scalers in {path}: {missing}')

        self.scalers = loaded
        self.is_fitted = True

        # Cargar columnas del scaler desde meta.json para que live_update_scalers_from_df
        # pueda filtrar exactamente las mismas columnas que se usaron en el training,
        # sin importar si el pipeline actual tiene más o menos features.
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_cols = json.load(f)

            _saved_no_scale = meta_cols.get('no_scale_by_block', self.no_scale_by_block)
            _no_scale_ctx   = set(_saved_no_scale.get('context', set()))
            _no_scale_seq_s = set(_saved_no_scale.get('seq_short', set()))
            _no_scale_seq_l = set(_saved_no_scale.get('seq_long', set()))

            # El scaler context recibe TODAS las columnas (incluyendo binarias con skip_features).
            # _scaler_feature_cols['context_all'] = todas las cols que el scaler recibe (n_features).
            # _scaler_feature_cols['context_scalable'] = subconjunto escalable (para auditoría).
            # live_update_scalers_from_df debe pasar context_all al scaler.update().
            _ctx_all = meta_cols.get('context_columns', [])
            if 'context_scalable_columns' in meta_cols:
                _ctx_scalable = meta_cols['context_scalable_columns']
            else:
                _ctx_scalable = [c for c in _ctx_all if c not in _no_scale_ctx]

            if _ctx_all:
                _n_feat = loaded['context'].n_features if 'context' in loaded else '?'
                print(f"\tScaler context: {len(_ctx_all)} cols total "
                      f"({len(_ctx_scalable)} escalables + "
                      f"{len(_ctx_all) - len(_ctx_scalable)} binarias skip_features) "
                      f"| scaler.n_features={_n_feat}")
                if _n_feat != '?' and _n_feat != len(_ctx_all):
                    print(f"\tWARN load_scalers: context_cols={len(_ctx_all)} "
                          f"vs scaler.n_features={_n_feat}. "
                          f"Regenerar scalers con el fix en save_scalers().")
                    # Fallback: no podemos reconstruir el orden exacto → live usará columnas actuales
                    self._scaler_feature_cols.pop('context_all', None)
                    self._scaler_feature_cols.pop('context_scalable', None)
                else:
                    self._scaler_feature_cols['context_all']      = _ctx_all
                    self._scaler_feature_cols['context_scalable']  = _ctx_scalable
            else:
                print("\tWARN: context_columns not found in meta.json — "
                      "live_update_scalers_from_df usará columnas actuales del pipeline (puede fallar)")

            # seq_short / seq_long: guardados implícitamente por el número de features del scaler.
            # Si meta.json tiene seq_columns en el futuro, se pueden añadir aquí.
            # Por ahora, para seq el pipeline raramente cambia de dimensión.

        # Cargar e inyectar umbrales de régimen si están presentes en meta.json.
        # Esto hace que detect_regime() y classify_mimo_state() usen los mismos
        # umbrales que durante el entrenamiento, sin importar cuántas velas se carguen.
        if os.path.exists(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta_full = json.load(f)
            regime_thresholds = meta_full.get('regime_thresholds')
            if regime_thresholds:
                self._regime_thresholds = regime_thresholds
                self._inject_regime_thresholds(regime_thresholds)
                print(f'\tRegime thresholds loaded and injected: {regime_thresholds}')
            else:
                print('\tWARNING: No regime_thresholds found in meta.json. '
                      'Regime will be computed dynamically from the inference window. '
                      'Call compute_and_store_regime_thresholds() during training and retrain.')

        self.rolling_infer_policy = meta.get('rolling_infer_policy', 'transform')

        print(f'\tScalers loaded from: {path}')
        return path

    def live_update_scalers_from_df(self, df_prepared: pd.DataFrame) -> None:
        """
        Actualiza los scalers rolling con las filas nuevas del df_prepared en live.

        FIX v2: robusto a cambios en el pipeline después del entrenamiento.

        Problema anterior: se pasaban TODAS las columnas actuales del pipeline al
        scaler.update(). Si el pipeline tiene más o menos features que cuando se
        entrenó el scaler, _ensure_n_features() lanza ValueError y el update falla.
        Esto ocurre con cualquier cambio de config (nuevas features, feature_masks
        distintos, MAX_POSITIONS diferente, etc.).

        Solución: usar self._scaler_feature_cols (poblado en load_scalers desde
        meta.json) para seleccionar exactamente las mismas columnas que usó el
        training — en el mismo orden y con la misma máscara de no_scale.
        Si meta.json no tiene context_columns (scalers antiguos), caer al
        comportamiento anterior con un warning.
        """
        if not self.is_fitted or not self.use_rolling_scaler or 'time' not in df_prepared.columns:
            return

        time = pd.to_datetime(df_prepared['time'])
        if self._last_live_time is None:
            self._last_live_time = time.iloc[-1]
            return

        mask = time > self._last_live_time
        if not mask.any():
            return

        df_new = df_prepared.loc[mask].copy()
        self._last_live_time = time.iloc[-1]

        cols = self.feature_engineer.feature_columns

        # ── seq_short y seq_long: mismas columnas actuales del pipeline ──────────
        # Estas raramente cambian de dimensión. Si cambian, el scaler lo detectará
        # y lanzará ValueError — aceptable, indica que hay que regenerar scalers.
        rows_to_update = {
            'seq_short': df_new[cols['sequence_short']].values.astype(np.float32),
            'seq_long':  df_new[cols['sequence_long']].values.astype(np.float32),
        }

        # ── context: pasar TODAS las columnas que el scaler conoce (context_all) ──
        # El scaler fue fiteado con todas las columnas de context (incluyendo binarias,
        # que maneja internamente con skip_features sin escalarlas). En live debemos
        # pasar exactamente las mismas columnas y en el mismo orden.
        _ctx_all_saved = self._scaler_feature_cols.get('context_all')

        if _ctx_all_saved is not None:
            # Rellenar columnas que ya no existen en el pipeline actual con 0
            _missing = [c for c in _ctx_all_saved if c not in df_new.columns]
            if _missing:
                for c in _missing:
                    df_new[c] = 0.0
            # Orden exacto del training
            rows_to_update['context'] = df_new[_ctx_all_saved].values.astype(np.float32)
        else:
            # Fallback: usar todas las columnas de context del pipeline actual
            # (puede fallar si el pipeline cambió de dimensión respecto al training)
            rows_to_update['context'] = df_new[cols['context']].values.astype(np.float32)

        self.update_rolling_scalers(rows_to_update)

    def create_last_sample_from_state(
            self,
            state: Dict[str, Any],
            end_idx: int,
            fit_scalers: bool = False,
            train: bool = False,
    ) -> Dict[str, np.ndarray]:
        """
        Construye la última muestra desde buffers numpy ya preparados.
        end_idx es exclusivo, igual que iloc[:end_idx].
        """

        seq_len_s = state["seq_len_short"]
        seq_len_l = state["seq_len_long"]

        if end_idx < seq_len_l:
            raise ValueError(
                f"No hay suficientes filas: end_idx={end_idx} < seq_len_long={seq_len_l}"
            )

        s0 = end_idx - seq_len_s
        l0 = end_idx - seq_len_l
        c0 = end_idx - 1

        seq_short_all = state["seq_short_all"][s0:end_idx][None, :, :]
        seq_long_all = state["seq_long_all"][l0:end_idx][None, :, :]
        context_all = state["context_all"][c0:c0 + 1]
        time_vals = state["time_vals"][c0:c0 + 1]

        if fit_scalers:
            raise NotImplementedError("create_last_sample_from_state() está pensado para inferencia/holdout")
        else:
            if not self.is_fitted:
                raise ValueError("Scalers no ajustados. Ejecuta load_scalers() o fit_scalers=True.")

            seq_short_all = self._transform_scaler(seq_short_all, "seq_short")
            seq_long_all = self._transform_scaler(seq_long_all, "seq_long")
            context_all = self._transform_scaler(context_all, "context")

        seq_short = seq_short_all[:, :, state["ss_idx"]]
        seq_long = seq_long_all[:, :, state["sl_idx"]]
        context = context_all[:, state["ctx_idx"]]

        if train:
            labels = state["labels"][c0:c0 + 1]
            weights = state["weights"][c0:c0 + 1]
        else:
            labels = None
            weights = None

        return {
            "seq_short": seq_short.astype(np.float32),
            "seq_long": seq_long.astype(np.float32),
            "context": context.astype(np.float32),
            "time": time_vals.astype(np.float32),
            "labels": labels,
            "weights": weights,
        }

    def prepare_walkforward_state(
            self,
            df: pd.DataFrame,
            side: str,
            train: bool = False,
    ) -> Dict[str, Any]:
        """
        Prepara buffers numpy e índices para evaluación walk-forward rápida.
        Todo lo caro se hace una sola vez fuera del loop.
        """

        union_cols = self._get_all_feature_columns()

        all_seq_short_cols = self._scaler_feature_cols.get("seq_short") \
                             or union_cols.get("sequence_short", [])
        all_seq_long_cols = self._scaler_feature_cols.get("seq_long") \
                            or union_cols.get("sequence_long", [])
        all_context_cols = self._scaler_feature_cols.get("context_all") \
                           or union_cols.get("context", [])

        self.feature_engineer.set_side(side)
        self.feature_engineer._assign_features_to_inputs()
        side_cols = self.feature_engineer.feature_columns

        seq_short_col_to_idx = {c: i for i, c in enumerate(all_seq_short_cols)}
        seq_long_col_to_idx = {c: i for i, c in enumerate(all_seq_long_cols)}
        context_col_to_idx = {c: i for i, c in enumerate(all_context_cols)}

        ss_idx = np.array(
            [seq_short_col_to_idx[c] for c in side_cols["sequence_short"] if c in seq_short_col_to_idx],
            dtype=np.int32
        )
        sl_idx = np.array(
            [seq_long_col_to_idx[c] for c in side_cols["sequence_long"] if c in seq_long_col_to_idx],
            dtype=np.int32
        )
        ctx_idx = np.array(
            [context_col_to_idx[c] for c in side_cols["context"] if c in context_col_to_idx],
            dtype=np.int32
        )

        state = {
            "seq_short_all": df[all_seq_short_cols].values.astype(np.float32, copy=False),
            "seq_long_all": df[all_seq_long_cols].values.astype(np.float32, copy=False),
            "context_all": df[all_context_cols].values.astype(np.float32, copy=False),
            "time_vals": df[side_cols["time"]].values.astype(np.float32, copy=False),
            "ss_idx": ss_idx,
            "sl_idx": sl_idx,
            "ctx_idx": ctx_idx,
            "seq_len_short": int(self.model_config.seq_len_short),
            "seq_len_long": int(self.model_config.seq_len_long),
        }

        if train:
            state["labels"] = df["signal"].values.astype(np.float32, copy=False)
            state["weights"] = (
                df["regime_weight"].values.astype(np.float32, copy=False)
                if "regime_weight" in df.columns
                else np.ones(len(df), dtype=np.float32)
            )
        else:
            state["labels"] = None
            state["weights"] = None

        return state