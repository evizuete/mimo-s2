from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from numpy import clip

from mimo.models.numba_utils import (
    fast_rolling_mean,
    fast_rolling_std,
    fast_rolling_max,
    fast_rolling_min,
    consecutive_runs_numba,
    rolling_autocorr_numba,
    rsi_numba,
    atr_numba
)

@dataclass
class FeatureConfig:
    """Configuración de features e indicadores"""
    # Indicadores técnicos
    ema_periods: List[int] = field(default_factory=lambda: [9, 21, 50])
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    macd_fast: int = 12
    macd_slow: int = 26
    macd_signal: int = 9
    bb_period: int = 20
    bb_std: float = 2.0

    # Ventanas para features
    return_lags: List[int] = field(default_factory=lambda: [1, 3, 5, 10, 20, 60])
    rolling_windows: List[int] = field(default_factory=lambda: [5, 10, 20, 60])

    # Normalización
    price_norm_window: int = 200

    # Labeling
    label_horizon: int = 10  # Horizonte de predicción (5 mins para 1-min data)
    label_method: str = 'triple_barrier'  # 'adaptive', 'fixed', 'triple_barrier'
    label_method_long: str = 'triple_barrier'
    label_method_short: str = 'adaptive'
    tp_barrier: float = 2.5
    sl_barrier: float = 1.5

    feature_masks: Optional[Dict[str, Dict[str, bool]]] = None

    # Barriers adaptativos por régimen.
    # Si None → se usan tp_barrier / sl_barrier como escalares (comportamiento original).
    # Si dict  → cada clave es un nombre de régimen y el valor define 'tp' y/o 'sl'.
    # Regímenes reconocidos: 'trending', 'ranging', 'low_vol', 'high_vol'.
    # Los valores que falten en un régimen se rellenan con tp_barrier / sl_barrier.
    #
    # Ejemplo:
    #   regime_barriers = {
    #       'trending': {'tp': 3.5, 'sl': 1.5},   # RR=2.33 — precio con recorrido
    #       'ranging':  {'tp': 2.0, 'sl': 1.5},   # RR=1.33 — mercado comprimido
    #       'low_vol':  {'tp': 2.5, 'sl': 1.0},   # RR=2.50 — SL estrecho
    #       'high_vol': {'tp': 3.0, 'sl': 2.0},   # RR=1.50 — ampliar para evitar ruido
    #   }
    regime_barriers: Optional[Dict[str, Dict[str, float]]] = None
    regime_barriers_long: Optional[Dict[str, Dict[str, float]]] = None
    regime_barriers_short: Optional[Dict[str, Dict[str, float]]] = None
    tp_barrier_short: Optional[float] = None  # Si None 2192 usa tp_barrier. Barriers de ejecuci00f3n SHORT en producci00f3n.
    sl_barrier_short: Optional[float] = None  # Si None 2192 usa sl_barrier.

class FeatureEngineer:
    """Generador de features con todos los indicadores técnicos relevantes"""

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config
        self.feature_columns = {
            'sequence_short': [],
            'sequence_long': [],
            'context': [],
            'time': []
        }

        self.side = 'both'

    def set_side(self, side: str):
        if side not in ('long', 'short', 'both'):
            raise ValueError(f'side must be long, short or both. Got: {side}')

        self.side = side
        self._assign_features_to_inputs()

    def _add_trend_dir(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        trend_dir: +1 up, -1 down, 0 neutral
        Basado en ADX_smooth + dominancia direccional (dm_diff).
        Requiere columnas: adx_smooth, dm_diff, atr (ya existen en tu pipeline).
        """
        df = df.copy()

        # Umbrales (ajustables)
        adx_thr = 25.0  # 25-30 típico; en tu gate usas 30
        dm_thr = 2.0  # umbral absoluto sobre dm_diff (depende de escala de pandas_ta)
        persist = 3  # histéresis: exige persistencia N barras

        # Señal base por dominancia direccional, sólo si hay tendencia fuerte
        strong = df["adx_smooth"] >= adx_thr
        up_raw = strong & (df["dm_diff"] > +dm_thr)
        dn_raw = strong & (df["dm_diff"] < -dm_thr)

        # Persistencia para evitar flips en M1
        up_ok = up_raw.rolling(persist, min_periods=1).mean() >= 0.67
        dn_ok = dn_raw.rolling(persist, min_periods=1).mean() >= 0.67

        trend_dir = np.where(up_ok, +1, np.where(dn_ok, -1, 0)).astype(np.int8)
        df["trend_dir"] = trend_dir

        return df

    def generate_all_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Pipeline completo de generación de features"""
        df = df.copy()

        # 1. Indicadores base
        df = self._add_atr(df)
        df = self._add_rsi(df)
        df = self._add_adx(df)  # ADX es crucial para detectar tendencias
        df = self._add_macd(df)  # MACD para momentum
        df = self._add_bollinger_bands(df)
        df = self._add_emas(df)
        df = self._add_trend_dir(df)

        # Añadiendo caracteristicas asociadas a los modelos de compra o venta de manera excluyente
        df["ema_bull"] = (df["ema_cross_9_21"] > 0).astype(int)
        df["ema_bear"] = (df["ema_cross_9_21"] < 0).astype(int)

        # RSI thresholds típicos (ajustables)
        df["rsi_oversold"] = (df["rsi"] < 30).astype(int)
        df["rsi_overbought"] = (df["rsi"] > 70).astype(int)

        df["macd_positive"] = (df["macd_hist"] > 0).astype(int)
        df["macd_negative"] = (df["macd_hist"] < 0).astype(int)

        # 2. Price action
        df = self._add_price_action(df)

        # 3. Microestructura (sin volumen)
        df = self._add_microstructure(df)

        # 4. Patrones de velas
        df = self._add_candle_patterns(df)

        # 5. Features temporales
        df = self._add_temporal_features(df)

        # 6. Features de contexto
        df = self._add_context_features(df)

        # 7. Adding chop and exhaustion scoring
        df = self._add_chop_and_exhaustion_features(df)

        df = self.add_price_invariant_features(df, window=200)

        # 8. Multi-timeframe features (5m / 15m / 1h)
        df = self._add_multi_timeframe_features(df)

        # 9. Calendar / session features extendidas
        df = self._add_extended_calendar_features(df)

        # Definir qué features van en cada input del modelo
        self._assign_features_to_inputs()

        return df.dropna()

    def get_adaptive_barriers(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Devuelve df con columnas 'tp_barrier' y 'sl_barrier' calculadas fila a fila
        según el régimen de mercado detectado en esa barra.

        Requiere que df ya contenga las columnas producidas por generate_all_features():
            - trend_dir      (+1 / -1 / 0)
            - is_chop        (0 / 1)
            - atr_norm_bps_z (z-score de volatilidad normalizada)

        Si config.regime_barriers es None (o vacío), simplemente replica los
        escalares tp_barrier / sl_barrier — compatibilidad total con el flujo original.

        Orden de prioridad de aplicación (último aplicado gana):
            1. low_vol   — baja volatilidad
            2. high_vol  — alta volatilidad
            3. ranging   — mercado comprimido / chop
            4. trending  — mercado con tendencia clara (prioridad máxima)
        """
        tp_base = float(self.config.tp_barrier)
        sl_base = float(self.config.sl_barrier)
        rb      = self.config.regime_barriers or {}

        df = df.copy()

        # Inicializar con valores base (comportamiento legacy si rb vacío)
        tp_arr = np.full(len(df), tp_base, dtype=np.float32)
        sl_arr = np.full(len(df), sl_base, dtype=np.float32)

        if not rb:
            df['tp_barrier'] = tp_arr
            df['sl_barrier'] = sl_arr
            return df

        # ── Régimen 1: baja volatilidad ─────────────────────────────────────
        if 'low_vol' in rb and 'atr_norm_bps_z' in df.columns:
            mask = df['atr_norm_bps_z'].values < -0.5
            tp_arr[mask] = rb['low_vol'].get('tp', tp_base)
            sl_arr[mask] = rb['low_vol'].get('sl', sl_base)

        # ── Régimen 2: alta volatilidad ─────────────────────────────────────
        if 'high_vol' in rb and 'atr_norm_bps_z' in df.columns:
            mask = df['atr_norm_bps_z'].values > 1.0
            tp_arr[mask] = rb['high_vol'].get('tp', tp_base)
            sl_arr[mask] = rb['high_vol'].get('sl', sl_base)

        # ── Régimen 3: ranging / chop ────────────────────────────────────────
        if 'ranging' in rb and 'is_chop' in df.columns:
            mask = df['is_chop'].values == 1
            tp_arr[mask] = rb['ranging'].get('tp', tp_base)
            sl_arr[mask] = rb['ranging'].get('sl', sl_base)

        # ── Régimen 4: trending (prioridad máxima) ───────────────────────────
        if 'trending' in rb and 'trend_dir' in df.columns:
            mask = df['trend_dir'].values != 0
            tp_arr[mask] = rb['trending'].get('tp', tp_base)
            sl_arr[mask] = rb['trending'].get('sl', sl_base)

        df['tp_barrier'] = tp_arr
        df['sl_barrier'] = sl_arr
        return df

    def  _apply_side_mask(self, cols: list[str]) -> list[str]:
        masks = self.config.feature_masks or {}

        # Sin máscaras o side='both', no filtrar
        if self.side not in ('long', 'short') or self.side not in masks:
            return cols

        allow = masks[self.side]

        # INCLUIR si:
        # - La columna NO está en la máscara (features básicas siempre incluidas)
        # - O la columna está en la máscara con valor True
        # EXCLUIR solo si está en la máscara con valor False

        feature_cols = [c for c in cols if c not in allow or allow[c]]
        return feature_cols

    def _add_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Average True Range"""
        atr_vals = atr_numba(
            df['high'].values,
            df['low'].values,
            df['close'].values,
            period=self.config.atr_period
        )

        df['atr'] = atr_vals
        df['atr_norm'] = df['atr'] / df['close']
        df['atr_norm_bps'] = df['atr_norm'] * 10_000.0

        df['atr_norm_bps_z'] = (
            df['atr_norm_bps']
            .transform(lambda x: (x - x.rolling(200, min_periods=50).mean())
                                 / (x.rolling(200, min_periods=50).std() + 1e-8))
            .clip(-3, 3)
        )

        return df

    def _add_rsi(self, df: pd.DataFrame) -> pd.DataFrame:
        """Relative Strength Index"""
        rsi_vals = rsi_numba(df['close'].values, period=self.config.rsi_period)

        df['rsi'] = rsi_vals
        df['rsi_norm'] = df['rsi'] / 100.0
        return df

    def _add_adx(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Average Directional Index - MUY IMPORTANTE para trading
        ADX mide la fuerza de la tendencia (no la dirección)
        """
        adx_df = ta.adx(df.high, df.low, df.close, length=self.config.adx_period)
        df['adx'] = adx_df[f'ADX_{self.config.adx_period}']
        df['adx_norm'] = df['adx'] /100.0
        df['dmp'] = adx_df[f'DMP_{self.config.adx_period}']  # Directional Movement Plus
        df['dmn'] = adx_df[f'DMN_{self.config.adx_period}']  # Directional Movement Minus

        # ADX suavizado para reducir ruido
        df['adx_smooth'] = df['adx'].rolling(5, min_periods=1).mean()
        df['adx_smooth_norm'] = df['adx_smooth'] / 100.0

        # Diferencia direccional (quién domina)
        df['dm_diff'] = df['dmp'] - df['dmn']
        df['dm_diff_norm'] = df['dm_diff'] / 100.0
        df['dm_ratio'] = df['dmp'] / (df['dmn'] + 1e-10)

        return df

    def _add_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        MACD - Moving Average Convergence Divergence
        Excelente para detectar cambios de momentum
        """
        macd_df = ta.macd(
            df.close,
            fast=self.config.macd_fast,
            slow=self.config.macd_slow,
            signal=self.config.macd_signal
        )

        df['macd'] = macd_df[f'MACD_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_signal'] = macd_df[f'MACDs_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']
        df['macd_hist'] = macd_df[f'MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}']

        # Normalizar por ATR para comparabilidad
        x = df['macd_hist'] / (df['atr'] + 1e-10)
        df['macd_hist_atr_log'] = np.sign(x) * np.log1p(np.abs(x) * 10_000.0)

        # Divergencia del MACD
        df['macd_divergence'] = df['macd'] - df['macd_signal']

        return df

    def _add_bollinger_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Bollinger Bands para detectar sobrecompra/sobreventa"""
        bb = ta.bbands(df.close, length=self.config.bb_period, std=self.config.bb_std)

        df['bb_upper'] = bb[f'BBU_{self.config.bb_period}_{self.config.bb_std}']
        df['bb_middle'] = bb[f'BBM_{self.config.bb_period}_{self.config.bb_std}']
        df['bb_lower'] = bb[f'BBL_{self.config.bb_period}_{self.config.bb_std}']

        # Ancho de banda (volatilidad)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        df['bb_width_bps'] = df['bb_width'] * 10_000.0

        df['bb_width_bps_z'] = (
            df['bb_width_bps']
            .transform(lambda x: (x - x.rolling(200, min_periods=50).mean())
                                 / (x.rolling(200, min_periods=50).std() + 1e-8))
            .clip(-3, 3)
        )

        # Posición del precio en la banda [0,1]
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)

        return df

    def _add_emas(self, df: pd.DataFrame) -> pd.DataFrame:
        """EMAs múltiples y sus relaciones"""
        for period in self.config.ema_periods:
            ema_col = f'ema_{period}'
            df[ema_col] = ta.ema(df.close, length=period)

            # Distancia del precio a la EMA
            df[f'{ema_col}_dist'] = (df.close - df[ema_col]) / df.close

            # Pendiente de la EMA (momentum)
            #df[f'{ema_col}_slope'] = df[ema_col].diff(3) / (df['atr'] + 1e-10)
            df[f'{ema_col}_slope'] = df[ema_col].pct_change(3)

            BPS = 10_000.0
            df[f'{ema_col}_dist_bps'] = df[f'{ema_col}_dist'] * BPS
            df[f'{ema_col}_slope_bps'] = df[f'{ema_col}_slope'] * BPS

        # Relaciones entre EMAs
        if len(self.config.ema_periods) >= 2:
            fast, slow = self.config.ema_periods[0], self.config.ema_periods[1]
            df[f'ema_cross_{fast}_{slow}'] = df[f'ema_{fast}'] - df[f'ema_{slow}']
            df[f'ema_cross_{fast}_{slow}_norm'] = df[f'ema_cross_{fast}_{slow}'] / (df['atr'] + 1e-10)

        return df

    def _add_price_action(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de acción del precio"""
        # Componentes de velas
        df['body'] = df.close - df.open
        df['upper_wick'] = df.high - df[['open', 'close']].max(axis=1)
        df['lower_wick'] = df[['open', 'close']].min(axis=1) - df.low
        df['range_hl'] = df.high - df.low

        # Normalizar por ATR
        for component in ['body', 'upper_wick', 'lower_wick', 'range_hl']:
            df[f'{component}_rel'] = df[component] / (df['atr'] + 1e-10)

        # Retornos múltiples horizontes
        BPS = 10_000.0
        for lag in self.config.return_lags:
            df[f'ret_{lag}'] = df.close.pct_change(lag)
            df[f'ret_{lag}_bps'] = clip(df[f'ret_{lag}'] * BPS, -150, 150)

        # Velocidad y aceleración del precio
        df['price_velocity'] = df.close.diff() / (df['atr'] + 1e-10)
        df['price_velocity_bps'] = df['price_velocity'] * 10_000.0

        df['price_acceleration'] = df['price_velocity'].diff()
        df['price_acceleration_bps'] = df['price_acceleration'] * 10_000.0

        return df

    EPS = 1e-12

    def _rolling_autocorr_lag1(self, r: pd.Series, w: int) -> pd.Series:
        # Corr( r[t-w+1:t], r[t-w+2:t+1] ) usando sumas rodantes
        x = r
        y = r.shift(-1)

        Sx = x.rolling(w).sum()
        Sy = y.rolling(w).sum()
        Sxx = (x * x).rolling(w).sum()
        Syy = (y * y).rolling(w).sum()
        Sxy = (x * y).rolling(w).sum()

        n = float(w)
        cov = (Sxy - (Sx * Sy) / n) / (n - 1.0)
        varx = (Sxx - (Sx * Sx) / n) / (n - 1.0)
        vary = (Syy - (Sy * Sy) / n) / (n - 1.0)

        out = cov / (np.sqrt(varx * vary) + self.EPS)
        out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

        # por el shift(-1), el último queda mal alineado
        out.iloc[-1] = 0.0
        return out

    def _consecutive_runs(self, cond: np.ndarray) -> np.ndarray:
        # cond: bool array
        idx = np.arange(cond.size, dtype=np.int64)
        last_false = np.maximum.accumulate(np.where(cond, -1, idx))
        return np.where(cond, idx - last_false, 0).astype(np.int32)

    def _add_microstructure(self, df: pd.DataFrame) -> pd.DataFrame:
        close = df["close"]
        range_hl = df["range_hl"]
        ret_1 = df["ret_1"]
        atr = df["atr"]

        diff = close.diff()
        up = (diff > 0)
        down = (diff < 0)

        # Precomputas esto una vez (lo reutilizas en todas las w)
        up_f = up.astype(np.float32)

        for w in self.config.rolling_windows:  # [5,10,20,60]
            range_sum = range_hl.rolling(w).sum()
            range_mean = range_hl.rolling(w).mean()

            df[f"efficiency_{w}"] = close.sub(close.shift(w)).abs() / (range_sum + self.EPS)
            df[f"realized_vol_{w}"] = ret_1.rolling(w).std()
            df[f'realized_vol_{w}_bps'] = df[f'realized_vol_{w}'] * 10_000.0

            df[f'realized_vol_{w}_bps_z'] = (
                df[f'realized_vol_{w}_bps']
                .transform(lambda x: (x - x.rolling(200, min_periods=50).mean())
                                     / (x.rolling(200, min_periods=50).std() + 1e-8))
                .clip(-3, 3)
            )

            df[f"autocorr_{w}"] = self._rolling_autocorr_lag1(ret_1, w)
            df[f"avg_range_{w}"] = range_mean / (atr + self.EPS)

            # sin apply
            df[f"direction_bias_{w}"] = up_f.rolling(w).mean().fillna(0.5)

        df["range_expansion"] = range_hl / (range_hl.rolling(20).mean() + self.EPS)

        # consecutivos sin groupby
        df["consecutive_ups"] = self._consecutive_runs(up.to_numpy(dtype=bool))
        df["consecutive_downs"] = self._consecutive_runs(down.to_numpy(dtype=bool))



        return df

    def _add_candle_patterns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Patrones de velas japonesas"""
        body_abs = abs(df['body'])

        # Doji - indecisión
        df['doji'] = (body_abs < df['atr'] * 0.1).astype(int)

        # Hammer/Hanging man
        df['hammer'] = (
                (df['lower_wick'] > body_abs * 2) &
                (df['upper_wick'] < body_abs * 0.3)
        ).astype(int)

        # Shooting star
        df['shooting_star'] = (
                (df['upper_wick'] > body_abs * 2) &
                (df['lower_wick'] < body_abs * 0.3)
        ).astype(int)

        # Engulfing
        df['bullish_engulfing'] = (
                (df['body'] > 0) &
                (df['body'] > abs(df['body'].shift(1)) * 1.5) &
                (df['body'].shift(1) < 0)
        ).astype(int)

        df['bearish_engulfing'] = (
                (df['body'] < 0) &
                (abs(df['body']) > abs(df['body'].shift(1)) * 1.5) &
                (df['body'].shift(1) > 0)
        ).astype(int)

        # Pin bar
        df['pin_bar'] = (
                ((df['upper_wick'] > df['atr'] * 2) | (df['lower_wick'] > df['atr'] * 2)) &
                (body_abs < df['atr'] * 0.3)
        ).astype(int)

        return df

    def _add_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features temporales y de sesión"""
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'])
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['dayofweek'] = df['time'].dt.dayofweek

            # Codificación cíclica
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['minute_sin'] = np.sin(2 * np.pi * df['minute'] / 60)
            df['minute_cos'] = np.cos(2 * np.pi * df['minute'] / 60)
            df['dow_sin'] = np.sin(2 * np.pi * df['dayofweek'] / 5)
            df['dow_cos'] = np.cos(2 * np.pi * df['dayofweek'] / 5)

            # Sesiones de trading
            df['is_asia'] = ((df['hour'] >= 0) & (df['hour'] < 8)).astype(int)
            df['is_london'] = ((df['hour'] >= 8) & (df['hour'] < 16)).astype(int)
            df['is_ny'] = ((df['hour'] >= 13) & (df['hour'] < 21)).astype(int)
            df['is_overlap'] = ((df['is_london'] == 1) & (df['is_ny'] == 1)).astype(int)

            # Apertura/cierre
            df['is_open'] = ((df['hour'] == 9) & (df['minute'] < 30)).astype(int)
            df['is_close'] = ((df['hour'] == 15) & (df['minute'] > 30)).astype(int)

        return df

    def _add_context_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Features de contexto de mercado"""
        # Distancia a máximos/mínimos
        new_cols = {}
        for window in [60, 240, 480]:
            roll_high = fast_rolling_max(df['high'].values, window)
            roll_low = fast_rolling_min(df['low'].values, window)

            new_cols[f'dist_high_{window}'] = np.clip((df['close'].values - roll_high) / (df['atr'].values + 1e-10), -10, 10)
            new_cols[f'dist_low_{window}'] = np.clip((df['close'].values - roll_low) / (df['atr'].values + 1e-10), -10, 10)
            new_cols[f'position_range_{window}'] = (df['close'].values - roll_low) / ((roll_high - roll_low) + 1e-10)
            df.drop(columns=[f'dist_high_{window}', f'dist_low_{window}', f'position_range_{window}'], inplace=True, errors='ignore')

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    def _normalize_prices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza precios OHLC"""
        new_cols = {}
        for col in ['open', 'high', 'low', 'close']:
            roll_mean = df[col].rolling(self.config.price_norm_window, min_periods=50).mean()
            new_cols[f'{col}_norm'] = df[col] / (roll_mean + 1e-10)
            df.drop(columns=[f'{col}_norm'], inplace=True, errors='ignore')

        df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
        return df

    @staticmethod
    def add_price_invariant_features(df: pd.DataFrame, window: int = 200) -> pd.DataFrame:
        for col in ['close', 'open', 'high', 'low']:
            arr = df[col].values
            rolling_mean = fast_rolling_mean(arr, window)
            rolling_std = fast_rolling_std(arr, window)
            df[f'{col}_norm'] = (arr - rolling_mean) / (rolling_std + 1e-8)

        return df

    @staticmethod
    def _safe_div(a, b, eps=1e-12):
        return a / (b + eps)

    def _compute_chop_score(self,
                           df: pd.DataFrame,
                           n: int = 12,
                           atr_col: str = 'atr',
                           ema_fast_col: str = 'ema_fast',
                           close_col: str = 'close',
                           high_col: str = 'high',
                           low_col: str = 'low' ) -> pd.Series:

        """
        Chop score 0..1: alto => compresión + mechas + precio extendido (zona mala para entradas).
        Requisitos: df[atr_col], df[ema_fast_col], OHLC.
        """

        close = df[close_col]
        high = df[high_col]
        low = df[low_col]
        atr = df[atr_col].astype(float)

        # 1) Compresión: rango N / ATR
        range_n = (high.rolling(n).max() - low.rolling(n).min())
        compression = 1.0 - np.tanh(self._safe_div(range_n, atr * np.sqrt(n)))  # 0..1 aprox (1=alta compresión)

        # 2) Wick ratio: mechas grandes respecto al rango de vela (promedio N)
        candle_range = (high - low).replace(0, np.nan)
        upper_wick = (high - np.maximum(close, df.get("open", close)))
        lower_wick = (np.minimum(close, df.get("open", close)) - low)
        wick_ratio = self._safe_div(upper_wick + lower_wick, candle_range).clip(0, 2)
        wick_ratio_n = wick_ratio.rolling(n).mean().fillna(0.0)
        wick_component = np.tanh(wick_ratio_n)  # 0..~1

        # 3) Extensión: distancia a EMA rápida en ATRs
        if ema_fast_col in df.columns:
            dist = (close - df[ema_fast_col]).abs()
            extension = np.tanh(self._safe_div(dist, atr * 1.5))  # 0..1
        else:
            extension = 0.0

        # Mezcla (pesos conservadores)
        chop = (0.45 * compression + 0.35 * wick_component + 0.20 * extension)
        return pd.Series(chop, index=df.index).clip(0.0, 1.0)

    def _compute_exhaustion_score(self,
                                 df: pd.DataFrame,
                                 n: int = 9,
                                 macd_hist_col: str = "macd_hist",
                                 atr_col: str = "atr",
                                 close_col: str = "close",
                                 high_col: str = "high",
                                 low_col: str = "low") -> pd.Series:
        """
        Exhaustion 0..1: alto => mechas + desaceleración del impulso (MACD hist slope negativa)
        Requisitos: macd_hist, atr, OHLC.
        """
        high = df[high_col]
        low = df[low_col]
        close = df[close_col]
        atr = df[atr_col].astype(float)

        candle_range = (high - low).replace(0, np.nan)
        upper_wick = (high - np.maximum(close, df.get("open", close)))
        wick_top = self._safe_div(upper_wick, candle_range).clip(0, 2)
        wick_top_n = wick_top.rolling(n).mean().fillna(0.0)
        wick_component = np.tanh(wick_top_n)

        if macd_hist_col in df.columns:
            hist = df[macd_hist_col].astype(float)
            slope = (hist - hist.shift(1))
            # desaceleración: slope negativa persistente (promedio N)
            slope_n = slope.rolling(n).mean().fillna(0.0)
            decel = (np.tanh((-slope_n).clip(lower=0) / 0.5))  # 0..1 (depende escala hist)
        else:
            decel = 0.0

        # Extensión (opcional) respecto a ATR: cierre cerca del máximo y muy extendido
        ext = np.tanh(self._safe_div((close - close.rolling(n).min()).abs(), atr * 2.0)).fillna(0.0)

        exhaustion = (0.50 * wick_component + 0.35 * decel + 0.15 * ext)
        return pd.Series(exhaustion, index=df.index).clip(0.0, 1.0)

    def _add_chop_and_exhaustion_features(self,
                                         df: pd.DataFrame,
                                         chop_n: int = 12,
                                         exhaustion_n: int = 9
    ) -> pd.DataFrame:

        df = df.copy()

        df["chop_score"] = self._compute_chop_score(df, n=chop_n)
        score_threshold = df['chop_score'].rolling(200, min_periods=50).quantile(0.85)
        df['is_chop'] = (df['chop_score'] >= score_threshold).astype(int)

        df["exhaustion_score"] = self._compute_exhaustion_score(df, n=exhaustion_n)
        exhaustion_threshold = df['exhaustion_score'].rolling(200, min_periods=50).quantile(0.85)
        df["is_exhaustion"] = (df["exhaustion_score"] >= exhaustion_threshold).astype(int)

        return df

    def _add_multi_timeframe_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derivadas de timeframes agregados (5m, 15m, 1h) reindexadas
        al timeframe base 1m con forward-fill. Cada barra 1m ve los valores
        del último cierre completado en el TF correspondiente.

        Sin lookahead: se usa label='right', closed='right' al resamplear y
        despues reindex con method='ffill'. Para una barra 1m a tiempo T se
        usa el ultimo bar 5m/15m/1h cuyo cierre fue <= T.

        Indicadores por TF:
          - EMA21 distancia (close - ema21) / atr * 10000  → bps por ATR
          - EMA50 distancia
          - EMA21 slope (pct change rolling 5)
          - EMA50 slope
          - RSI(14) normalizado [-1, +1]
          - MACD hist normalizado por ATR del TF
          - BB position [0, 1]
          - ADX(14) normalizado [0, 1]
          - DM diff (dmp - dmn) / 100

        Total: 9 indicadores x 3 TFs = 27 columnas nuevas.
        """
        if 'time' not in df.columns:
            return df

        df = df.copy()
        df_idx = pd.to_datetime(df['time'])

        df_tf_src = df[['open', 'high', 'low', 'close']].copy()
        df_tf_src.index = df_idx
        df_tf_src = df_tf_src[~df_tf_src.index.duplicated(keep='last')]

        tfs = {
            '5m':  '5min',
            '15m': '15min',
            '1h':  '1h',
        }
        new_cols = {}

        for tf_label, tf_rule in tfs.items():
            agg = df_tf_src.resample(tf_rule, label='right', closed='right').agg(
                {'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}
            ).dropna()
            if len(agg) < 60:
                continue

            ema21 = agg['close'].ewm(span=21, adjust=False).mean()
            ema50 = agg['close'].ewm(span=50, adjust=False).mean()

            tr = pd.concat([
                agg['high'] - agg['low'],
                (agg['high'] - agg['close'].shift(1)).abs(),
                (agg['low']  - agg['close'].shift(1)).abs(),
            ], axis=1).max(axis=1)
            atr_tf = tr.rolling(14, min_periods=1).mean()
            atr_tf_safe = atr_tf.replace(0, np.nan)

            ema21_dist_bps = (agg['close'] - ema21) / (atr_tf_safe + 1e-10) * 10_000.0
            ema50_dist_bps = (agg['close'] - ema50) / (atr_tf_safe + 1e-10) * 10_000.0
            ema21_slope_bps = (ema21.diff(5) / (atr_tf_safe + 1e-10)) * 10_000.0
            ema50_slope_bps = (ema50.diff(5) / (atr_tf_safe + 1e-10)) * 10_000.0

            try:
                rsi_tf = ta.rsi(agg['close'], length=14)
            except Exception:
                rsi_tf = pd.Series(50.0, index=agg.index)
            rsi_norm = (rsi_tf - 50.0) / 50.0

            try:
                macd_df_tf = ta.macd(agg['close'], fast=12, slow=26, signal=9)
                macd_hist_tf = macd_df_tf['MACDh_12_26_9']
            except Exception:
                macd_hist_tf = pd.Series(0.0, index=agg.index)
            macd_hist_atr = (macd_hist_tf / (atr_tf_safe + 1e-10)).clip(-10, 10)

            try:
                bb_tf = ta.bbands(agg['close'], length=20, std=2)
                bb_pos = (
                    (agg['close'] - bb_tf['BBL_20_2.0'])
                    / (bb_tf['BBU_20_2.0'] - bb_tf['BBL_20_2.0'] + 1e-10)
                ).clip(-0.5, 1.5)
            except Exception:
                bb_pos = pd.Series(0.5, index=agg.index)

            try:
                adx_df_tf = ta.adx(agg['high'], agg['low'], agg['close'], length=14)
                adx_tf = adx_df_tf['ADX_14'] / 100.0
                dm_diff_tf = (
                    adx_df_tf['DMP_14'] - adx_df_tf['DMN_14']
                ) / 100.0
            except Exception:
                adx_tf = pd.Series(0.0, index=agg.index)
                dm_diff_tf = pd.Series(0.0, index=agg.index)

            tf_feats = pd.DataFrame({
                f'ema21_dist_{tf_label}_bps':  ema21_dist_bps,
                f'ema50_dist_{tf_label}_bps':  ema50_dist_bps,
                f'ema21_slope_{tf_label}_bps': ema21_slope_bps,
                f'ema50_slope_{tf_label}_bps': ema50_slope_bps,
                f'rsi_{tf_label}_norm':        rsi_norm,
                f'macd_hist_{tf_label}_atr':   macd_hist_atr,
                f'bb_position_{tf_label}':     bb_pos,
                f'adx_{tf_label}_norm':        adx_tf,
                f'dm_diff_{tf_label}_norm':    dm_diff_tf,
            })

            tf_feats_1m = tf_feats.reindex(df_idx, method='ffill')
            for col in tf_feats_1m.columns:
                new_cols[col] = tf_feats_1m[col].to_numpy()

        if new_cols:
            df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

        return df

    def _add_extended_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features de calendario adicionales no cubiertas por
        _add_temporal_features. Foco en granularidad sub-hora y posicion
        relativa dentro de la sesion.
        """
        if 'time' not in df.columns:
            return df

        df = df.copy()
        t = pd.to_datetime(df['time'])

        minute_of_day = t.dt.hour * 60 + t.dt.minute
        df['minute_of_day_sin'] = np.sin(2 * np.pi * minute_of_day / 1440.0)
        df['minute_of_day_cos'] = np.cos(2 * np.pi * minute_of_day / 1440.0)

        df['day_of_month'] = t.dt.day
        df['is_month_end'] = (t.dt.day >= 28).astype(int)
        df['is_friday'] = (t.dt.dayofweek == 4).astype(int)

        hour = t.dt.hour
        minute = t.dt.minute
        df['is_eu_first_hour'] = ((hour == 8) & (minute < 60)).astype(int)
        df['is_us_first_hour'] = ((hour == 13) & (minute < 60)).astype(int)
        df['is_us_last_hour']  = ((hour == 20) & (minute < 60)).astype(int)
        df['is_lunch_eu']      = ((hour == 12)).astype(int)

        return df

    def _assign_features_to_inputs(self):
        """Define qué features van a cada input del modelo"""

        # Features para secuencia corta (más reactivas)
        self.feature_columns['sequence_short'] = [
            'close_norm', 'open_norm', 'high_norm', 'low_norm',
            'body_rel', 'upper_wick_rel', 'lower_wick_rel', 'range_hl_rel',
            'ret_1_bps', 'ret_3_bps', 'ret_5_bps', 'ret_10_bps',
            'ema_9_dist_bps', 'ema_21_dist_bps', 'ema_9_slope_bps',
            'rsi_norm', 'macd_hist_atr_log', 'bb_position',
            'price_velocity', 'price_acceleration',
            'doji', 'hammer', 'shooting_star'
        ]

        # Features para secuencia larga (tendencia)
        # Incluye multi-TF (5m, 15m) que aportan contexto a escalas mayores
        # sin que el modelo tenga que inferirlo desde la secuencia 1m.
        self.feature_columns['sequence_long'] = [
            'close_norm', 'range_hl_rel',
            'ema_21_dist_bps', 'ema_50_dist_bps',
            'ema_21_slope_bps', 'ema_50_slope_bps',
            'trend_dir',
            'rsi_norm', 'adx_norm', 'macd_hist_atr_log',
            'ret_20_bps', 'ret_60_bps',
            'realized_vol_20_bps_z', 'efficiency_20',
            'direction_bias_20',
            # Multi-TF 5m
            'ema21_dist_5m_bps', 'ema50_dist_5m_bps',
            'ema21_slope_5m_bps', 'rsi_5m_norm',
            'macd_hist_5m_atr', 'bb_position_5m',
            'adx_5m_norm', 'dm_diff_5m_norm',
            # Multi-TF 15m
            'ema21_dist_15m_bps', 'ema50_dist_15m_bps',
            'ema21_slope_15m_bps', 'rsi_15m_norm',
            'macd_hist_15m_atr', 'adx_15m_norm', 'dm_diff_15m_norm',
        ]

        # Features de contexto (estado actual del mercado)
        # Incluye 1h multi-TF y calendario extendido para macro-context.
        self.feature_columns['context'] = [
            'atr_norm_bps_z', 'adx_norm', 'adx_smooth_norm', 'dm_diff_norm',
            'bb_width_bps_z', 'range_expansion',
            'dist_high_60', 'dist_low_60', 'position_range_240',
            'chop_score', 'exhaustion_score', 'is_chop', 'is_exhaustion',
            'ema_bull', 'ema_bear', 'rsi_oversold', 'rsi_overbought',
            'macd_positive', 'macd_negative',
            # Multi-TF 1h (contexto largo)
            'ema21_dist_1h_bps', 'ema50_dist_1h_bps',
            'rsi_1h_norm', 'adx_1h_norm', 'dm_diff_1h_norm',
            'bb_position_1h',
            # Calendario extendido
            'is_month_end', 'is_friday',
            'is_eu_first_hour', 'is_us_first_hour', 'is_us_last_hour',
        ]

        # Features temporales
        self.feature_columns['time'] = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_asia', 'is_london', 'is_ny', 'is_overlap',
            'minute_of_day_sin', 'minute_of_day_cos',
        ]

        for k, cols in self.feature_columns.items():
            self.feature_columns[k] = self._apply_side_mask(cols)