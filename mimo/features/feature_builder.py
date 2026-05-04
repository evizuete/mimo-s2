from dataclasses import dataclass, field
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from numba import jit
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


@jit(nopython=True, cache=True)
def _volume_profile_numba(high, low, close, vol, window, n_bins, top_k):
    """
    Rolling volume profile sobre window barras.

    Para cada t devuelve (poc_price[t], concentration[t]):
      - poc_price: precio centro del bin con más volumen (Point of Control)
      - concentration: suma de los top_k bins / total volumen ∈ [0, 1]
    """
    n = len(high)
    poc_price = np.full(n, np.nan)
    concentration = np.full(n, np.nan)
    hist = np.empty(n_bins, dtype=np.float64)

    for i in range(window - 1, n):
        s = i - window + 1
        rng_min = low[s]
        rng_max = high[s]
        for j in range(s + 1, i + 1):
            if low[j] < rng_min:
                rng_min = low[j]
            if high[j] > rng_max:
                rng_max = high[j]

        if rng_max <= rng_min:
            continue

        for k in range(n_bins):
            hist[k] = 0.0

        inv_width = n_bins / (rng_max - rng_min)
        for j in range(s, i + 1):
            tp = (high[j] + low[j] + close[j]) / 3.0
            idx = int((tp - rng_min) * inv_width)
            if idx >= n_bins:
                idx = n_bins - 1
            elif idx < 0:
                idx = 0
            hist[idx] += vol[j]

        total = 0.0
        max_idx = 0
        max_val = hist[0]
        for k in range(n_bins):
            total += hist[k]
            if hist[k] > max_val:
                max_val = hist[k]
                max_idx = k

        if total <= 0.0:
            continue

        bin_width = (rng_max - rng_min) / n_bins
        poc_price[i] = rng_min + (max_idx + 0.5) * bin_width

        sorted_hist = np.sort(hist)
        top_sum = 0.0
        for k in range(n_bins - top_k, n_bins):
            top_sum += sorted_hist[k]
        concentration[i] = top_sum / total

    return poc_price, concentration

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

    # Vol-invariant features: si True, usa variantes ATR-normalizadas de
    # retornos y EMA dist/slope en lugar de las _bps. Recomendado cuando el
    # holdout tiene una distribucion de volatilidad distinta a train (ver
    # mimo/oof/shift_analyzer/distribution_shift_analyzer.py — si las
    # features _bps salen con PSI > 0.1 y sigma_ratio > 1.5).
    use_vol_invariant_features: bool = False

    # Reduced features: si True, elimina del input del modelo el set de
    # features identificadas como ruido por permutation importance sobre
    # 202200 (drop_max < 0.0005 en AUC-PR). Reduce 97 -> 72 features (25%
    # menos parametros en el LSTM, menos overfitting). Ver
    # mimo/oof/feature_importance_permutation.py.
    use_reduced_features: bool = False

    # Ultra-reduced features: aplica un segundo recorte sobre el subset ya
    # reducido, identificado por permutation importance sobre 202300 (drop_max
    # < 0.001 — threshold mas conservador en segunda iteracion). Reduce de
    # 71 a ~33 features (>50% menos del baseline 97). Solo se debe activar
    # con use_reduced_features=True (es un superset de eliminaciones).
    use_ultra_reduced_features: bool = False

    # Labeling
    label_horizon: int = 10  # Horizonte de predicción (5 mins para 1-min data)
    label_method: str = 'triple_barrier'  # 'adaptive', 'fixed', 'triple_barrier', 'quantile_return'
    label_method_long: str = 'triple_barrier'
    label_method_short: str = 'adaptive'
    tp_barrier: float = 2.5
    sl_barrier: float = 1.5

    # Quantile regression labeling.
    # Si label_method == 'quantile_return', el target es el forward return
    # normalizado por ATR (continuo, no binario) y el modelo se entrena con
    # pinball loss para predecir múltiples cuantiles simultáneamente.
    quantile_horizon: int = 5  # h en barras para el forward return
    quantile_levels: tuple = (0.25, 0.50, 0.75)

    # Magnitude binary labeling (direction-agnostic).
    # Si label_method == 'magnitude_binary', el label vale 1 si la mayor
    # excursión |precio - close[t]| / ATR[t] sobre las próximas label_horizon
    # barras supera magnitude_threshold (en unidades de ATR), 0 si no.
    # Diseñado como gate del modelo direccional, no como trade per se.
    magnitude_threshold: float = 1.5

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

        # 3b. Volumen (ticks_volume → z, pct, spike, trend)
        df = self._add_volume_features(df)

        # 3c. VWAP (rolling 1h y 4h) — magnet de volumen y desviaciones
        df = self._add_vwap_features(df)

        # 3d. Volume profile (POC y concentración sobre 4h)
        df = self._add_volume_profile_features(df)

        # 4. Patrones de velas
        df = self._add_candle_patterns(df)

        # 5. Features temporales
        df = self._add_temporal_features(df)

        # 6. Features de contexto
        df = self._add_context_features(df)

        # 7. Adding chop and exhaustion scoring
        df = self._add_chop_and_exhaustion_features(df)

        df = self.add_price_invariant_features(df, window=self.config.price_norm_window)

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

        _w = int(self.config.price_norm_window)
        _mp = max(10, _w // 4)
        df['atr_norm_bps_z'] = (
            df['atr_norm_bps']
            .transform(lambda x: (x - x.rolling(_w, min_periods=_mp).mean())
                                 / (x.rolling(_w, min_periods=_mp).std() + 1e-8))
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

        # Ancho de banda (volatilidad). Epsilon para evitar /0 si bb_middle ~ 0
        # (caso teórico en activos con precios cercanos a cero).
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (df['bb_middle'] + 1e-10)
        df['bb_width_bps'] = df['bb_width'] * 10_000.0

        _w = int(self.config.price_norm_window)
        _mp = max(10, _w // 4)
        df['bb_width_bps_z'] = (
            df['bb_width_bps']
            .transform(lambda x: (x - x.rolling(_w, min_periods=_mp).mean())
                                 / (x.rolling(_w, min_periods=_mp).std() + 1e-8))
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

            # Variantes ATR-normalizadas: invariantes al regimen de volatilidad.
            # PSI bajo en holdout cuando la vol cambia (vs _bps que escalan con
            # la dispersion natural de los retornos). Construir SIEMPRE; el
            # uso depende de _assign_features_to_inputs y use_vol_invariant_features.
            atr_safe = df['atr'].replace(0, np.nan).ffill().fillna(1e-8)
            df[f'{ema_col}_dist_atr'] = clip(
                (df.close - df[ema_col]) / (atr_safe + 1e-10), -8, 8
            )
            df[f'{ema_col}_slope_atr'] = clip(
                df[ema_col].diff(3) / (atr_safe + 1e-10), -8, 8
            )

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
        atr_safe = df['atr'].replace(0, np.nan).ffill().fillna(1e-8)
        for lag in self.config.return_lags:
            df[f'ret_{lag}'] = df.close.pct_change(lag)
            df[f'ret_{lag}_bps'] = clip(df[f'ret_{lag}'] * BPS, -150, 150)
            # Variante ATR-normalizada: cuantos ATRs se movio el precio en
            # 'lag' barras. Invariante al regimen de volatilidad — clave para
            # transferibilidad train -> holdout cuando la vol cambia.
            df[f'ret_{lag}_atr'] = clip(
                df.close.diff(lag) / (atr_safe + 1e-10), -10, 10
            )

        # Velocidad y aceleración del precio
        df['price_velocity'] = df.close.diff() / (df['atr'] + 1e-10)
        df['price_velocity_bps'] = df['price_velocity'] * 10_000.0

        df['price_acceleration'] = df['price_velocity'].diff()
        df['price_acceleration_bps'] = df['price_acceleration'] * 10_000.0

        return df

    EPS = 1e-12

    def _rolling_autocorr_lag1(self, r: pd.Series, w: int) -> pd.Series:
        """Autocorrelación causal con lag-1.

        Calcula Corr(r[t-w+1 : t], r[t-w : t-1]) — la serie y su versión
        retrasada un paso. Causal: solo usa información hasta t.

        Antes (buggy): y = r.shift(-1) tomaba r[t+1] (lookahead).
        """
        x = r
        y = r.shift(1)  # lag-1 retrasado, NO adelantado

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
        # Las primeras w filas tienen NaN propagado por el shift inicial
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

            _w = int(self.config.price_norm_window)
            _mp = max(10, _w // 4)
            df[f'realized_vol_{w}_bps_z'] = (
                df[f'realized_vol_{w}_bps']
                .transform(lambda x: (x - x.rolling(_w, min_periods=_mp).mean())
                                     / (x.rolling(_w, min_periods=_mp).std() + 1e-8))
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

    def _add_volume_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume-derived context features. Solo se computan si existe la columna
        'ticks_volume' (presente en histórico y MT5 live tras DataManager).

        Cuatro features escala-invariantes, todas sobre log(1+volume):
          vol_z_1h      z-score rolling 1h
          vol_pct_1h    percentile rank rolling 1h (robusto a colas)
          vol_spike     desviación de la EMA local (aceleración)
          vol_trend_1h  pendiente normalizada en 1h

        n_per_h se detecta del propio df.time (mediana del Δt) para que la
        semántica "1h" se mantenga independientemente del base_tf (1m / 5m / etc).

        Si la columna no existe, se rellenan con valores neutros para
        retro-compatibilidad con datasets sin volumen.
        """
        if "ticks_volume" not in df.columns:
            df["vol_z_1h"] = 0.0
            df["vol_pct_1h"] = 0.5
            df["vol_spike"] = 0.0
            df["vol_trend_1h"] = 0.0
            return df

        # Detecta resolución del df: mediana del Δt en segundos → barras/hora.
        if "time" in df.columns and len(df) > 1:
            dt_med = (
                pd.to_datetime(df["time"]).diff().dropna().dt.total_seconds().median()
            )
            n_per_h = int(round(3600.0 / max(dt_med, 1.0))) if dt_med and dt_med > 0 else 12
        else:
            n_per_h = 12
        n_per_h = max(4, n_per_h)
        ema_span = max(6, n_per_h)  # EMA local: ~1h (en lugar de 12 fijo)

        v = df["ticks_volume"].astype(float).clip(lower=0)
        log_v = np.log1p(v)

        mp = max(8, n_per_h // 2)
        mu_1h = log_v.rolling(n_per_h, min_periods=mp).mean()
        sd_1h = log_v.rolling(n_per_h, min_periods=mp).std()
        df["vol_z_1h"] = ((log_v - mu_1h) / sd_1h.replace(0, np.nan)).clip(-3, 3)

        df["vol_pct_1h"] = (
            log_v.rolling(n_per_h, min_periods=mp).rank(pct=True)
        )

        df["vol_spike"] = (log_v - log_v.ewm(span=ema_span, adjust=False).mean()).clip(-3, 3)

        df["vol_trend_1h"] = (log_v - log_v.shift(n_per_h)) / float(n_per_h)

        return df

    def _detect_n_per_h(self, df: pd.DataFrame, default: int = 12) -> int:
        """Detecta barras/hora del df según mediana del Δt en 'time'."""
        if "time" in df.columns and len(df) > 1:
            dt_med = (
                pd.to_datetime(df["time"]).diff().dropna().dt.total_seconds().median()
            )
            if dt_med and dt_med > 0:
                return max(4, int(round(3600.0 / dt_med)))
        return default

    def _add_vwap_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Features derivadas de VWAP (rolling, no anclado a sesión).

        Cuatro features escala-invariantes:
          vwap_dist_atr      (close - vwap_1h) / atr  — desviación local
          vwap_dist_4h_atr   (close - vwap_4h) / atr  — desviación a contexto largo
          vwap_band_pos      (close - vwap_1h) / vwap_std_1h  — z-score VWAP-relative
          vwap_slope_atr     pendiente VWAP_1h / atr / n_per_h

        VWAP rolling = sum(typical_price * vol) / sum(vol) sobre window.
        Si no hay 'ticks_volume' se usa el bar count (≈ TWAP) para preservar
        retro-compatibilidad con datasets sin volumen.
        """
        n_per_h = self._detect_n_per_h(df)
        w_short = max(6, n_per_h)
        w_long = max(24, n_per_h * 4)

        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        atr = df["atr"].replace(0, np.nan)

        if "ticks_volume" in df.columns:
            v = df["ticks_volume"].astype(float).clip(lower=0).replace(0, np.nan)
        else:
            v = pd.Series(1.0, index=df.index)

        tp_v = tp * v.fillna(0)
        v_filled = v.fillna(0)

        mp_s = max(4, w_short // 2)
        mp_l = max(8, w_long // 4)

        num_s = tp_v.rolling(w_short, min_periods=mp_s).sum()
        den_s = v_filled.rolling(w_short, min_periods=mp_s).sum().replace(0, np.nan)
        vwap_s = num_s / den_s

        num_l = tp_v.rolling(w_long, min_periods=mp_l).sum()
        den_l = v_filled.rolling(w_long, min_periods=mp_l).sum().replace(0, np.nan)
        vwap_l = num_l / den_l

        df["vwap_dist_atr"] = ((df["close"] - vwap_s) / atr).clip(-5, 5)
        df["vwap_dist_4h_atr"] = ((df["close"] - vwap_l) / atr).clip(-5, 5)

        # Banda VWAP: std del precio típico ponderada uniformemente sobre window
        # (proxy estable y rápida de la dispersión vs VWAP).
        tp_std_s = tp.rolling(w_short, min_periods=mp_s).std().replace(0, np.nan)
        df["vwap_band_pos"] = ((df["close"] - vwap_s) / tp_std_s).clip(-3, 3)

        df["vwap_slope_atr"] = (
            (vwap_s - vwap_s.shift(n_per_h)) / atr / float(n_per_h)
        ).clip(-1, 1)

        return df

    def _add_volume_profile_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Volume profile rolling sobre 4h: POC (Point of Control) y concentración.

        Dos features:
          poc_dist_atr       (close - POC_4h) / atr   — distancia al imán de volumen
          vol_concentration  top 5 bins / total       — qué tan concentrado está
                              el volumen (alto = nivel claro; bajo = disperso)

        Implementación numba: rolling histograma de 20 bins sobre 4h × n_per_h.
        Si no hay 'ticks_volume' devuelve neutros.
        """
        if "ticks_volume" not in df.columns:
            df["poc_dist_atr"] = 0.0
            df["vol_concentration"] = 0.25  # 5/20 = uniforme
            return df

        n_per_h = self._detect_n_per_h(df)
        window = max(24, n_per_h * 4)
        n_bins = 20
        top_k = 5

        high = df["high"].to_numpy(np.float64)
        low = df["low"].to_numpy(np.float64)
        close = df["close"].to_numpy(np.float64)
        vol = df["ticks_volume"].astype(float).clip(lower=0).to_numpy(np.float64)
        atr = df["atr"].to_numpy(np.float64)

        poc_price, concentration = _volume_profile_numba(
            high, low, close, vol, window, n_bins, top_k
        )

        atr_safe = np.where(atr > 0, atr, np.nan)
        poc_dist = (close - poc_price) / atr_safe
        df["poc_dist_atr"] = pd.Series(poc_dist, index=df.index).clip(-5, 5)
        df["vol_concentration"] = pd.Series(concentration, index=df.index).clip(0, 1)

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

        # Defragmentar el DataFrame tras los ~20 df['col']=... consecutivos
        # (silencia PerformanceWarning de pandas). Sin impacto en resultados.
        df = df.copy()
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

        _w = int(self.config.price_norm_window)
        _mp = max(10, _w // 4)

        df["chop_score"] = self._compute_chop_score(df, n=chop_n)
        score_threshold = df['chop_score'].rolling(_w, min_periods=_mp).quantile(0.85)
        df['is_chop'] = (df['chop_score'] >= score_threshold).astype(int)

        df["exhaustion_score"] = self._compute_exhaustion_score(df, n=exhaustion_n)
        exhaustion_threshold = df['exhaustion_score'].rolling(_w, min_periods=_mp).quantile(0.85)
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

    # Features identificadas como ruido por permutation importance sobre 202200
    # (drop_max < 0.0005 sobre AUC-PR). Se eliminan cuando
    # use_reduced_features=True. Total: 25 features (~25.8% del input).
    _REDUCED_DROP_SEQ_SHORT = {
        "doji", "hammer", "shooting_star",  # candle patterns: muy raros
        "rsi_norm",                          # duplicado con sequence_long.rsi_norm
        "ret_5_atr",                         # redundante con ret_3/ret_10
    }
    _REDUCED_DROP_SEQ_LONG = {
        "rsi_15m_norm", "macd_hist_15m_atr",  # multi-TF 15m no aporta
        "adx_15m_norm", "ema21_slope_15m_bps",
        "ema50_dist_5m_bps",                  # ema21_5m manda, ema50_5m no
        "direction_bias_20",
    }
    _REDUCED_DROP_CONTEXT = {
        "ema_bull", "ema_bear",               # binarios duplican con macd_pos/neg
        "bb_position_1h",                     # ya tenemos bb_width_bps_z + 1h ema
        "is_eu_first_hour",                   # otros calendar features ganan
        "vol_z_1h", "vol_concentration",      # vol_z_1h ya esta en seq_short
        "vwap_slope_atr",                     # vwap_dist_atr y _4h dominan
        "is_chop",                            # chop_score continuo basta
        "dist_low_60",                        # dist_high y position_range bastan
        "rsi_1h_norm",                        # rsi_norm seq_long manda
        "exhaustion_score",                   # is_exhaustion binario funciona
        "range_expansion",                    # bb_width_bps_z ya cubre
        "ema50_dist_1h_bps",                  # ema21_1h domina
        "rsi_overbought",                     # binario duplica rsi_norm
        "poc_dist_atr",                       # poc_dist en seq_short manda
    }

    # Segunda iteracion (sobre 202300): features con drop_max < 0.001 tras
    # reentrenar con 72 features. Threshold mas conservador (0.001 vs 0.0005).
    # Total: 38 features adicionales (>50% del set ya reducido) — captura las
    # redundancias que se hicieron evidentes tras la primera reduccion.
    _ULTRA_REDUCED_DROP_SEQ_SHORT = {
        "vol_z_1h",                  # ya redundante (vol_spike captura mejor)
        "macd_hist_atr_log",         # seq_long.macd_hist_atr_log ya manda
        "body_rel",                  # range_hl_rel + wicks bastan
        "range_hl_rel",              # ya cubierto por wick_rel
        "ret_3_atr",                 # ret_1 y ret_10 cubren
    }
    _ULTRA_REDUCED_DROP_SEQ_LONG = {
        "dm_diff_15m_norm", "rsi_5m_norm", "rsi_norm",
        "dm_diff_5m_norm", "macd_hist_5m_atr",
        "ema_50_slope_atr", "bb_position_5m", "trend_dir",
        "ema_50_dist_atr", "range_hl_rel", "close_norm",
        "efficiency_20", "realized_vol_20_bps_z",
        "ema_21_dist_atr",            # ema_9_dist_atr seq_short manda
        "adx_5m_norm", "adx_norm",
        "ema21_dist_15m_bps", "ema50_dist_15m_bps",
    }
    _ULTRA_REDUCED_DROP_CONTEXT = {
        "vwap_dist_4h_atr",           # vwap_dist_atr seq_short suficiente
        "macd_positive",              # binarios redundantes
        "bb_width_bps_z",             # atr_norm_bps_z lo cubre
        "adx_norm",                   # adx_1h_norm domina
        "vol_trend_1h",               # vol_pct_1h y vol_spike bastan
        "chop_score",                 # is_chop ya removido en ultra
        "is_us_first_hour",
        "macd_negative",
        "ema21_dist_1h_bps",          # info ya en seq_long
        "adx_smooth_norm",            # adx_1h_norm domina
        "is_us_last_hour",
        "is_exhaustion",
        "dist_high_60",
        "dm_diff_norm",               # dm_diff_1h_norm domina
        "rsi_oversold",
    }

    def _assign_features_to_inputs(self):
        """Define qué features van a cada input del modelo"""

        # Si use_vol_invariant_features=True, sustituimos las _bps que el
        # distribution_shift_analyzer marcó como WARNING (PSI ~ 0.23-0.25 con
        # sigma_ratio ~ 2 en holdout) por sus equivalentes ATR-normalizadas.
        # Bps son sensibles al cambio de regimen de volatilidad; ATR-normalized
        # son invariantes porque el ATR recoge la vol local.
        use_atr = bool(getattr(self.config, "use_vol_invariant_features", False))
        use_reduced = bool(getattr(self.config, "use_reduced_features", False))
        use_ultra = bool(getattr(self.config, "use_ultra_reduced_features", False))

        def _suffix(bps_name: str) -> str:
            """Para nombres tipo 'ret_5_bps' o 'ema_9_dist_bps', devuelve el
            equivalente _atr cuando use_atr=True, si la feature ATR existe.
            """
            if not use_atr:
                return bps_name
            atr_name = bps_name[:-len("_bps")] + "_atr"
            return atr_name

        def _filter_reduced(cols: list, drop_set: set,
                            ultra_drop_set: set | None = None) -> list:
            """Aplica los filtros de reducción según los flags activos.
            ultra es un superset (segundo recorte sobre 202300)."""
            if not use_reduced and not use_ultra:
                return cols
            drops = set(drop_set) if use_reduced else set()
            if use_ultra and ultra_drop_set is not None:
                drops |= set(ultra_drop_set)
            return [c for c in cols if c not in drops]

        # Features para secuencia corta (más reactivas)
        self.feature_columns['sequence_short'] = _filter_reduced([
            'close_norm', 'open_norm', 'high_norm', 'low_norm',
            'body_rel', 'upper_wick_rel', 'lower_wick_rel', 'range_hl_rel',
            _suffix('ret_1_bps'), _suffix('ret_3_bps'),
            _suffix('ret_5_bps'), _suffix('ret_10_bps'),
            _suffix('ema_9_dist_bps'), _suffix('ema_21_dist_bps'),
            _suffix('ema_9_slope_bps'),
            'rsi_norm', 'macd_hist_atr_log', 'bb_position',
            'price_velocity', 'price_acceleration',
            'doji', 'hammer', 'shooting_star',
            # Volumen bar-a-bar: confirmación precio-volumen y bursts locales
            'vol_z_1h', 'vol_spike',
            # VWAP / volume profile bar-a-bar (magnet de volumen)
            'vwap_dist_atr', 'poc_dist_atr',
        ], self._REDUCED_DROP_SEQ_SHORT, self._ULTRA_REDUCED_DROP_SEQ_SHORT)

        # Features para secuencia larga (tendencia)
        # Incluye multi-TF (5m, 15m) que aportan contexto a escalas mayores
        # sin que el modelo tenga que inferirlo desde la secuencia 1m.
        self.feature_columns['sequence_long'] = _filter_reduced([
            'close_norm', 'range_hl_rel',
            _suffix('ema_21_dist_bps'), _suffix('ema_50_dist_bps'),
            _suffix('ema_21_slope_bps'), _suffix('ema_50_slope_bps'),
            'trend_dir',
            'rsi_norm', 'adx_norm', 'macd_hist_atr_log',
            _suffix('ret_20_bps'), _suffix('ret_60_bps'),
            'realized_vol_20_bps_z', 'efficiency_20',
            'direction_bias_20',
            # Multi-TF 5m (PSI bajo en holdout — no necesitan ATR-norm)
            'ema21_dist_5m_bps', 'ema50_dist_5m_bps',
            'ema21_slope_5m_bps', 'rsi_5m_norm',
            'macd_hist_5m_atr', 'bb_position_5m',
            'adx_5m_norm', 'dm_diff_5m_norm',
            # Multi-TF 15m (PSI bajo en holdout — no necesitan ATR-norm)
            'ema21_dist_15m_bps', 'ema50_dist_15m_bps',
            'ema21_slope_15m_bps', 'rsi_15m_norm',
            'macd_hist_15m_atr', 'adx_15m_norm', 'dm_diff_15m_norm',
        ], self._REDUCED_DROP_SEQ_LONG, self._ULTRA_REDUCED_DROP_SEQ_LONG)

        # Features de contexto (estado actual del mercado)
        # Incluye 1h multi-TF y calendario extendido para macro-context.
        self.feature_columns['context'] = _filter_reduced([
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
            # Volumen (ticks_volume — magnitud/convicción del movimiento)
            'vol_z_1h', 'vol_pct_1h', 'vol_spike', 'vol_trend_1h',
            # VWAP rolling (1h corta + 4h larga, banda y pendiente)
            'vwap_dist_atr', 'vwap_dist_4h_atr', 'vwap_band_pos', 'vwap_slope_atr',
            # Volume profile rolling 4h (POC y concentración)
            'poc_dist_atr', 'vol_concentration',
        ], self._REDUCED_DROP_CONTEXT, self._ULTRA_REDUCED_DROP_CONTEXT)

        # Features temporales
        self.feature_columns['time'] = [
            'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
            'is_asia', 'is_london', 'is_ny', 'is_overlap',
            'minute_of_day_sin', 'minute_of_day_cos',
        ]

        for k, cols in self.feature_columns.items():
            self.feature_columns[k] = self._apply_side_mask(cols)