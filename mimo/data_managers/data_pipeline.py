import json
import os
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler

from mimo.data_managers.rolling_scaler import RollingRobustScaler
from mimo.features.feature_builder import FeatureConfig, FeatureEngineer
from mimo.features.label_generator import LabelGenerator
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig, add_mimo_state, StateDetector


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

        self.use_rolling_scaler = True
        self.scaler_window_size = 2880
        self.scaler_warmup_size = 390
        self.scaler_smooth_alpha = 0.1
        self.no_scale_by_block = {
            'seq_short': { 'doji', 'hammer', 'shooting_star'},
            'seq_long': { 'trend_dir' },
            'context': { 'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
                         'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative'}
        }

        self._last_live_time = None

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
        out = {}
        for side in sides:
            self.feature_engineer.set_side(side)
            self.feature_engineer._assign_features_to_inputs()
            cols = self.feature_engineer.feature_columns

            # Verificar que columnas existen en df
            missing = [c for c in cols['sequence_short'] if c not in df.columns]
            if missing:
                print(f"  ⚠️  Missing columns: {missing}")

            seq_short = self._create_sequences(
                df[cols['sequence_short']].values, self.model_config.seq_len_short
            )

            seq_long = self._create_sequences(
                df[cols['sequence_long']].values, self.model_config.seq_len_long
            )

            n_samples = len(seq_long)
            offset = self.model_config.seq_len_long - self.model_config.seq_len_short
            seq_short = seq_short[offset:offset + n_samples]

            context_offset = self.model_config.seq_len_long - 1
            context = df[cols['context']].values[context_offset:context_offset + n_samples]
            time = df[cols['time']].values[context_offset:context_offset + n_samples]

            if train:
                if 'signal' not in df.columns:
                    raise ValueError('Training mode requires "signal" column in DataFrame')

                labels = df['signal'].values[context_offset:context_offset + n_samples]
                if 'regime_weight' in df.columns:
                    weights = df['regime_weight'].values[context_offset:context_offset + n_samples]
                else:
                    weights = np.ones(n_samples, dtype=np.float32)
            else:
                labels = None
                weights = None

            if fit_scalers:
                self._fit_scaler_from_raw(
                    df[cols['sequence_short']].values.astype(np.float32),
                    'seq_short', cols['sequence_short']
                )
                self._fit_scaler_from_raw(
                    df[cols['sequence_long']].values.astype(np.float32),
                    'seq_long', cols['sequence_long']
                )
                self._fit_scaler_from_raw(
                    df[cols['context']].values.astype(np.float32),
                    'context', cols['context']
                )

                self.is_fitted = True

            seq_short = self._transform_scaler(seq_short, 'seq_short')
            seq_long = self._transform_scaler(seq_long, 'seq_long')
            context = self._transform_scaler(context, 'context')

            out[side] = {
                'seq_short': seq_short.astype(np.float32),
                'seq_long': seq_long.astype(np.float32),
                'context': context.astype(np.float32),
                'time': time,
                'labels': labels.astype(np.float32) if labels is not None else None,
                'weights': weights.astype(np.float32) if weights is not None else None
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

        # 5. Labels (si aplica)
        if train:
            labels = df['signal'].values[context_offset:context_offset + n_samples]
            weights = df.get('regime_weight', pd.Series(1.0, index=df.index)).values[
                context_offset:context_offset + n_samples]
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
        state_cfg_overrides = getattr(self, '_state_config_overrides', {})
        state_cfg = StateConfig(**state_cfg_overrides) if state_cfg_overrides else StateConfig()

        df = add_mimo_state(
            df,
            cfg=state_cfg,
            set_market_condition=set_market_condition,
        )

        # Alias: state_weight -> regime_weight para compatibilidad con código existente
        if 'state_weight' in df.columns and 'regime_weight' not in df.columns:
            df['regime_weight'] = df['state_weight']

        # 3. Generar etiquetas
        if labels:
            df = self.label_generator.generate_labels(df, side)
            df = df.dropna()
        else:
            df = df.ffill().bfill().fillna(0)

        return df

    def _scale_context_with_exclusions(self, context: np.ndarray, context_cols: list[str], fit: bool) -> np.ndarray:
        """Escala solo features continuas, preserva binarias sin modificar (sin cambios)"""

        # Features binarias que NO se escalan
        no_scale = {
            'is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear',
            'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative'
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
                            'rsi_oversold', 'rsi_overbought', 'macd_positive', 'macd_negative'}
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

        # 5. Labels y pesos
        if train:
            labels = df['signal'].values[context_offset:context_offset + n_samples]
            weights = df['regime_weight'].values[
                context_offset:context_offset + n_samples] if 'regime_weight' in df else None
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
                min_iqr=1e-4
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
                min_iqr=1e-4
            )
            # El scaler solo necesita window_size=2880 filas representativas
            # Las cogemos con stride para cubrir toda la distribución temporal
            n_rows = raw_data.shape[0]
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

        context_cols = self.feature_engineer.feature_columns.get('context', [])
        no_scale_by_block = {
            'seq_short': ['doji', 'hammer', 'shooting_star'],
            'seq_long': ['trend_dir'],
            'context': ['is_chop', 'is_exhaustion', 'ema_bull', 'ema_bear', 'rsi_oversold', 'rsi_overbought',
                        'macd_positive', 'macd_negative']
        }

        meta = {
            'release': str(self.general_config.release),
            'use_rolling_scaler': bool(self.use_rolling_scaler),
            'scaler_window_size': int(self.scaler_window_size),
            'scaler_warmup_size': int(self.scaler_warmup_size),
            'scaler_smooth_alpha': float(self.scaler_smooth_alpha),
            'rolling_infer_policy': str(self.rolling_infer_policy),
            'scalers': list(self.scalers.keys()),
            'context_columns': context_cols,
            'version': 'data_pipeline_v4.regime_thresholds',
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
        print(f'Scalers saved to: {path}')
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
                    print(f"WARNING: Context columns differ!")
                    print(f"  Saved: {saved_context_cols}")
                    print(f"  Current: {current_context_cols}")

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

        print(f'\tScalers loaded from: {path}')
        return path

    def live_update_scalers_from_df(self, df_prepared: pd.DataFrame) -> None:
        """Actualiza scalers con filas nuevas en live (sin cambios)"""
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
        rows_to_update = {
            'seq_short': df_new[cols['sequence_short']].values.astype(np.float32),
            'seq_long': df_new[cols['sequence_long']].values.astype(np.float32),
            'context': df_new[cols['context']].values.astype(np.float32)
        }

        self.update_rolling_scalers(rows_to_update)