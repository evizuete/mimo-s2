import numpy as np
import pandas as pd

from mimo.features.feature_builder import FeatureConfig
from mimo.models.numba_utils import (
    triple_barrier_fixed_numba,
    triple_barrier_3class_numba,
    triple_barrier_adaptive_numba,
    fast_rolling_max,
    fast_rolling_quantile, fast_rolling_min,
)


class LabelGenerator:
    """Genera etiquetas para entrenamiento"""

    def __init__(self, config: FeatureConfig = FeatureConfig()):
        self.config = config

    # ─────────────────────────────────────────────────────────────────────────
    # Punto de entrada principal
    # ─────────────────────────────────────────────────────────────────────────

    def generate_labels(self, df: pd.DataFrame, side: str = 'long') -> pd.DataFrame:
        """
        Genera etiquetas de trading.

        Si config.regime_barriers está definido y el método es 'triple_barrier'
        o 'fixed', los barriers se calculan fila a fila según el régimen.
        Con 'adaptive' se ajusta el percentil de corte por régimen.

        Args:
            df:   DataFrame con OHLC y features (incluyendo atr, trend_dir,
                  is_chop, atr_norm_bps_z si se usan barriers adaptativos).
            side: 'long' para compra, 'short' para venta.
        """
        df = df.copy()

        original_method = self.config.label_method
        if side == 'short' and self.config.label_method_short is not None:
            active_method = self.config.label_method_short
        elif side == 'long' and self.config.label_method_long is not None:
            active_method = self.config.label_method_long
        else:
            active_method = self.config.label_method
        self.config.label_method = active_method

        original_barriers = self.config.regime_barriers
        if side == 'short' and self.config.regime_barriers_short is not None:
            active_barriers = self.config.regime_barriers_short
        elif side == 'long' and self.config.regime_barriers_long is not None:
            active_barriers = self.config.regime_barriers_long
        else:
            active_barriers = self.config.regime_barriers
        self.config.regime_barriers = active_barriers

        if self.config.label_method == 'adaptive':
            df = self._adaptive_labels(df, side)
        elif self.config.label_method == 'fixed':
            df = self._fixed_labels(df, side)
        elif self.config.label_method == 'triple_barrier':
            df = self._triple_barrier_labels(df, side)
        elif self.config.label_method == 'quantile_return':
            df = self._quantile_return_labels(df, side)
        elif self.config.label_method == 'magnitude_binary':
            df = self._magnitude_binary_labels(df, side)
        elif self.config.label_method == 'triple_class':
            df = self._triple_class_labels(df, side)
        else:
            raise ValueError(f"Método {self.config.label_method} no reconocido")

        # Filtrar señales en condiciones adversas
        _regime_col = 'state' if 'state' in df.columns else (
            'regime' if 'regime' in df.columns else None
        )
        if _regime_col is not None:
            _vals = (
                df[_regime_col].str.upper()
                if df[_regime_col].dtype == object
                else df[_regime_col]
            )

            # HIGH_VOL: reducir peso, NO borrar señal
            mask_high_vol = _vals.isin(['VOLATILE', 'HIGH_VOLATILITY'])
            if 'regime_weight' in df.columns:
                df.loc[mask_high_vol, 'regime_weight'] *= 0.3

            # LOW_VOL: reducir peso
            mask_low_vol = _vals.isin(['LOW_VOL', 'LOW_VOLATILITY'])
            if 'regime_weight' in df.columns:
                df.loc[mask_low_vol, 'regime_weight'] *= 0.5

        self.config.regime_barriers = original_barriers
        self.config.label_method = original_method

        return df

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers internos
    # ─────────────────────────────────────────────────────────────────────────

    def _get_barrier_arrays(self, df: pd.DataFrame):
        """
        Devuelve (tp_arr, sl_arr) como arrays numpy float32 con longitud len(df).

        Si config.regime_barriers está definido, delega en
        FeatureEngineer.get_adaptive_barriers() para calcularlos por fila.
        Si no, replica los escalares tp_barrier / sl_barrier.

        Requiere que FeatureEngineer esté disponible como atributo del pipeline;
        si no lo está, genera los barriers directamente aquí con la misma lógica
        para mantener independencia de módulo.
        """

        rb      = self.config.regime_barriers or {}
        tp_base = float(self.config.tp_barrier)
        sl_base = float(self.config.sl_barrier)
        n       = len(df)

        tp_arr = np.full(n, tp_base, dtype=np.float32)
        sl_arr = np.full(n, sl_base, dtype=np.float32)

        if not rb:
            return tp_arr, sl_arr

        # ── low_vol ──────────────────────────────────────────────────────────
        if 'low_vol' in rb and 'atr_norm_bps_z' in df.columns:
            mask = df['atr_norm_bps_z'].values < -0.5
            tp_arr[mask] = rb['low_vol'].get('tp', tp_base)
            sl_arr[mask] = rb['low_vol'].get('sl', sl_base)

        # ── high_vol ─────────────────────────────────────────────────────────
        if 'high_vol' in rb and 'atr_norm_bps_z' in df.columns:
            mask = df['atr_norm_bps_z'].values > 1.0
            tp_arr[mask] = rb['high_vol'].get('tp', tp_base)
            sl_arr[mask] = rb['high_vol'].get('sl', sl_base)

        # ── ranging / chop ───────────────────────────────────────────────────
        if 'ranging' in rb and 'is_chop' in df.columns:
            mask = df['is_chop'].values == 1
            tp_arr[mask] = rb['ranging'].get('tp', tp_base)
            sl_arr[mask] = rb['ranging'].get('sl', sl_base)

        # ── trending (prioridad máxima) ───────────────────────────────────────
        if 'trending' in rb and 'trend_dir' in df.columns:
            mask = df['trend_dir'].values != 0
            tp_arr[mask] = rb['trending'].get('tp', tp_base)
            sl_arr[mask] = rb['trending'].get('sl', sl_base)

        print(f"[DIAG BARRIERS] tp_base={tp_base} sl_base={sl_base} | rb keys={list(rb.keys())}")
        print(f"[DIAG BARRIERS] tp unique={np.unique(tp_arr)} sl unique={np.unique(sl_arr)}")

        return tp_arr, sl_arr

    # ─────────────────────────────────────────────────────────────────────────
    # Métodos de labeling
    # ─────────────────────────────────────────────────────────────────────────

    def _adaptive_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Etiquetado adaptativo basado en percentiles dinámicos.

        Sin regime_barriers: comportamiento original — percentil 70 global
        sobre ventana de 1000 barras.

        Con regime_barriers: el percentil de corte se ajusta por régimen:
          - trending → percentil 75 (más exigente, explotamos el recorrido)
          - ranging  → percentil 65 (más permisivo, señales más cortas)
          - high_vol → percentil 80 (muy selectivo, evitar ruido)
          - low_vol  → percentil 68 (ligeramente más permisivo)
          - default  → percentil 70 (igual que antes)
        """
        horizon = self.config.label_horizon
        rb      = self.config.regime_barriers or {}
        out     = df.copy()

        close = out['close'].values
        high  = out['high'].values
        low   = out['low'].values
        atr   = out['atr'].values

        side_is_long = (side == 'long')
        potential = triple_barrier_adaptive_numba(
            close, high, low, atr, horizon, side_is_long
        )

        out['potential']       = potential
        out[f'potential_{side}'] = potential

        if not rb:
            # ── Comportamiento original ───────────────────────────────────────
            window    = 1000
            threshold = fast_rolling_quantile(potential, window, 0.7)
            threshold = np.roll(threshold, 1)
            threshold[0] = np.nan
            out['signal'] = (
                (potential > threshold) & (~np.isnan(potential))
            ).astype(int)

        else:
            # ── Percentil adaptativo por régimen ─────────────────────────────
            # Percentiles base por régimen (ajustar según backtest)
            regime_quantiles = {
                'trending': rb.get('trending', {}).get('adaptive_q', 0.75),
                'ranging':  rb.get('ranging',  {}).get('adaptive_q', 0.65),
                'high_vol': rb.get('high_vol', {}).get('adaptive_q', 0.80),
                'low_vol':  rb.get('low_vol',  {}).get('adaptive_q', 0.68),
                'default':  0.70,
            }

            window = 1000

            # Precomputar threshold para cada percentil único (evitar repetir trabajo)
            unique_q = set(regime_quantiles.values())
            thresholds_by_q = {}
            for q in unique_q:
                t = fast_rolling_quantile(potential, window, q)
                t = np.roll(t, 1)
                t[0] = np.nan
                thresholds_by_q[q] = t

            # Construir array de thresholds fila a fila según régimen
            q_default = regime_quantiles['default']
            thr_arr   = thresholds_by_q[q_default].copy()

            # Aplicar por régimen (mismo orden de prioridad que _get_barrier_arrays)
            if 'low_vol' in rb and 'atr_norm_bps_z' in out.columns:
                mask = out['atr_norm_bps_z'].values < -0.5
                thr_arr[mask] = thresholds_by_q[regime_quantiles['low_vol']][mask]

            if 'high_vol' in rb and 'atr_norm_bps_z' in out.columns:
                mask = out['atr_norm_bps_z'].values > 1.0
                thr_arr[mask] = thresholds_by_q[regime_quantiles['high_vol']][mask]

            if 'ranging' in rb and 'is_chop' in out.columns:
                mask = out['is_chop'].values == 1
                thr_arr[mask] = thresholds_by_q[regime_quantiles['ranging']][mask]

            if 'trending' in rb and 'trend_dir' in out.columns:
                mask = out['trend_dir'].values != 0
                thr_arr[mask] = thresholds_by_q[regime_quantiles['trending']][mask]

            out['signal'] = (
                (potential > thr_arr) & (~np.isnan(potential))
            ).astype(int)

        return out

    def _fixed_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Etiquetado con umbrales en ATRs.

        Con regime_barriers: tp y sl varían fila a fila según el régimen.
        Sin ellos: comportamiento original con escalares fijos.
        """
        horizon  = int(self.config.label_horizon)
        out      = df.copy()

        close    = out['close'].values
        high     = out['high'].values
        low      = out['low'].values
        atr_norm = out['atr_norm'].values

        future_close = np.roll(close, -horizon)
        future_close[-horizon:] = np.nan

        if side == 'long':
            future_return = (future_close / close) - 1.0
            future_low    = fast_rolling_min(np.roll(low, -1), horizon)
            future_low    = np.roll(future_low, -horizon + 1)
            adverse       = (close - future_low) / (close + 1e-10)
        else:
            future_return = 1.0 - (future_close / close)
            future_high   = fast_rolling_max(np.roll(high, -1), horizon)
            future_high   = np.roll(future_high, -horizon + 1)
            adverse       = (future_high - close) / (close + 1e-10)

        tp_arr, sl_arr = self._get_barrier_arrays(out)

        out['signal'] = (
            (future_return > tp_arr * atr_norm) &
            (adverse < sl_arr * atr_norm)
        ).astype(int)

        return out

    def _triple_barrier_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Triple barrier labeling (binario).
        label=1 si TP se toca antes que SL dentro del horizonte; 0 en caso contrario.

        Con regime_barriers: tp_mult y sl_mult varían fila a fila.
        Sin ellos: comportamiento original con un único par de escalares,
                   delegando en la función numba optimizada.
        """


        horizon      = int(self.config.label_horizon)
        out          = df.copy()
        close        = out['close'].values
        high         = out['high'].values
        low          = out['low'].values
        atr          = out['atr'].values
        side_is_long = (side == 'long')

        rb = self.config.regime_barriers or {}


        if not rb:
            # ── Camino original: una sola llamada numba, escalar ─────────────
            tp_mult = float(self.config.tp_barrier)
            sl_mult = float(self.config.sl_barrier)
            signal  = triple_barrier_fixed_numba(
                close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long
            )
        else:
            # ── Barriers por fila: llamada vectorizada por régimen ────────────
            # Para evitar un loop Python barra a barra, agrupamos filas que
            # comparten los mismos barriers y llamamos a numba una vez por grupo.
            tp_arr, sl_arr = self._get_barrier_arrays(out)
            print(f"[DEBUG TB] side={side} rb={self.config.regime_barriers} tp_unique={np.unique(tp_arr)}")

            # Identificar grupos únicos de (tp, sl) — normalmente 2-4 combinaciones
            pairs = np.stack([tp_arr, sl_arr], axis=1)
            unique_pairs = np.unique(pairs, axis=0)

            signal = np.zeros(len(out), dtype=np.int32)

            for tp_val, sl_val in unique_pairs:
                group_mask = (tp_arr == tp_val) & (sl_arr == sl_val)
                idx        = np.where(group_mask)[0]

                if len(idx) == 0:
                    continue

                # Necesitamos arrays contiguos para numba; construimos sub-arrays
                # y luego un loop reducido solo sobre índices del grupo.
                # Como triple_barrier_fixed_numba necesita mirar 'horizon' barras
                # hacia adelante desde cada índice, no podemos recortar el array —
                # pasamos el array completo pero solo escribimos el resultado para
                # los índices del grupo.
                partial = triple_barrier_fixed_numba(
                    close, high, low, atr, horizon,
                    float(tp_val), float(sl_val), side_is_long
                )
                signal[idx] = partial[idx]


        out['signal'] = signal
        return out

    def _triple_class_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Triple barrier labeling 3-class. Devuelve clases {0=SL, 1=TIMEOUT, 2=TP}.

        Mismo recorrido temporal que _triple_barrier_labels (binario) pero
        distingue los tres outcomes. La cabeza del modelo correspondiente es
        softmax(3) y la loss SparseCategoricalCrossentropy. Downstream:
        P(TP) = softmax[..., 2] se usa como "signal" para calibración y
        umbrales (compatible con el flujo binario).

        Soporta regime_barriers igual que el binario.
        """
        horizon      = int(self.config.label_horizon)
        out          = df.copy()
        close        = out['close'].values
        high         = out['high'].values
        low          = out['low'].values
        atr          = out['atr'].values
        side_is_long = (side == 'long')

        rb = self.config.regime_barriers or {}

        if not rb:
            tp_mult = float(self.config.tp_barrier)
            sl_mult = float(self.config.sl_barrier)
            signal  = triple_barrier_3class_numba(
                close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long
            )
        else:
            tp_arr, sl_arr = self._get_barrier_arrays(out)
            print(f"[DEBUG TB-3CLASS] side={side} rb={self.config.regime_barriers} "
                  f"tp_unique={np.unique(tp_arr)}")

            pairs = np.stack([tp_arr, sl_arr], axis=1)
            unique_pairs = np.unique(pairs, axis=0)

            signal = np.ones(len(out), dtype=np.int32)  # default TIMEOUT (1)

            for tp_val, sl_val in unique_pairs:
                group_mask = (tp_arr == tp_val) & (sl_arr == sl_val)
                idx        = np.where(group_mask)[0]
                if len(idx) == 0:
                    continue
                partial = triple_barrier_3class_numba(
                    close, high, low, atr, horizon,
                    float(tp_val), float(sl_val), side_is_long
                )
                signal[idx] = partial[idx]

        out['signal'] = signal.astype(np.int32)
        return out

    def _triple_barrier_labels_debug(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Triple barrier labeling (binario).
        label=1 si TP se toca antes que SL dentro del horizonte; 0 en caso contrario.
        Con regime_barriers: tp_mult y sl_mult varían fila a fila.
        Sin ellos: comportamiento original con un único par de escalares,
                   delegando en la función numba optimizada.
        """

        horizon = int(self.config.label_horizon)
        out = df.copy()
        close = out['close'].values
        high = out['high'].values
        low = out['low'].values
        atr = out['atr'].values
        side_is_long = (side == 'long')
        rb = self.config.regime_barriers or {}

        if not rb:
            # ── Camino original: una sola llamada numba, escalar ─────────────
            tp_mult = float(self.config.tp_barrier)
            sl_mult = float(self.config.sl_barrier)

            # ── DIAGNÓSTICO TEMPORAL ─────────────────────────────────────────
            idx_diag = 1000
            entry = close[idx_diag]
            atr_d = atr[idx_diag]
            if side_is_long:
                tp_level = entry + tp_mult * atr_d
                sl_level = entry - sl_mult * atr_d
            else:
                tp_level = entry - tp_mult * atr_d
                sl_level = entry + sl_mult * atr_d

            print(f"[DIAG TB] side={side} side_is_long={side_is_long}")
            print(f"[DIAG TB] tp_mult={tp_mult} sl_mult={sl_mult} horizon={horizon}")
            print(f"[DIAG TB] entry={entry:.4f} atr={atr_d:.4f}")
            print(f"[DIAG TB] tp_level={tp_level:.4f} sl_level={sl_level:.4f}")
            print(f"[DIAG TB] Simulación manual de las primeras {horizon} barras:")
            for k in range(1, horizon + 1):
                j = idx_diag + k
                if j >= len(close):
                    break
                if side_is_long:
                    tp_hit = high[j] >= tp_level
                    sl_hit = low[j] <= sl_level
                else:
                    tp_hit = low[j] <= tp_level
                    sl_hit = high[j] >= sl_level
                print(f"  bar {k}: close={close[j]:.4f} high={high[j]:.4f} low={low[j]:.4f} "
                      f"tp_hit={tp_hit} sl_hit={sl_hit}")

            signal = triple_barrier_fixed_numba(
                close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long
            )

            # Verificar resultado numba vs manual
            sig_diag = int(signal[idx_diag])
            print(f"[DIAG TB] signal[{idx_diag}] = {sig_diag} (1=TP ganó, 0=SL/timeout)")
            n_pos = int(signal.sum())
            print(f"[DIAG TB] Total positivos numba: {n_pos} / {len(signal)} ({100 * n_pos / len(signal):.2f}%)")
            # ── FIN DIAGNÓSTICO ──────────────────────────────────────────────

        else:
            # ── Barriers por fila: llamada vectorizada por régimen ────────────
            tp_arr, sl_arr = self._get_barrier_arrays(out)
            print(f"[DEBUG TB] side={side} rb={self.config.regime_barriers} tp_unique={np.unique(tp_arr)}")

            pairs = np.stack([tp_arr, sl_arr], axis=1)
            unique_pairs = np.unique(pairs, axis=0)
            signal = np.zeros(len(out), dtype=np.int32)

            for tp_val, sl_val in unique_pairs:
                group_mask = (tp_arr == tp_val) & (sl_arr == sl_val)
                idx = np.where(group_mask)[0]
                if len(idx) == 0:
                    continue
                partial = triple_barrier_fixed_numba(
                    close, high, low, atr, horizon,
                    float(tp_val), float(sl_val), side_is_long
                )
                signal[idx] = partial[idx]

        out['signal'] = signal
        return out

    def _quantile_return_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Target continuo: forward return normalizado por ATR.

            target = (close[t+h] - close[t]) / atr[t]   (LONG)
            target = (close[t] - close[t+h]) / atr[t]   (SHORT)

        A diferencia de triple_barrier (binario), 'signal' aquí es float32
        en unidades de ATR (típicamente -3..+3). El modelo se entrena con
        pinball loss y predice múltiples cuantiles en lugar de una probabilidad.

        Las últimas `quantile_horizon` filas no tienen futuro suficiente y se
        marcan con NaN; el pipeline downstream filtra NaN antes de entrenar.
        """
        h = int(self.config.quantile_horizon)
        out = df.copy()
        close = out['close'].values.astype(np.float64)
        atr = out['atr'].values.astype(np.float64)

        fwd_close = np.roll(close, -h).astype(np.float64)
        # Las últimas h filas no tienen futuro: marcar NaN
        fwd_close[-h:] = np.nan

        atr_safe = np.maximum(atr, 1e-10)
        if side == 'long':
            ret = (fwd_close - close) / atr_safe
        else:
            ret = (close - fwd_close) / atr_safe

        out['signal'] = ret.astype(np.float32)
        return out

    def _magnitude_binary_labels(self, df: pd.DataFrame, side: str) -> pd.DataFrame:
        """
        Magnitude binary labeling (direction-agnostic).

            label[t] = 1 si max(|high[t+k]-close[t]|, |close[t]-low[t+k]|) / atr[t]
                          >= magnitude_threshold  para algún k en 1..h
                       0 en caso contrario.

        El mismo label se devuelve para LONG y SHORT — la idea es predecir
        si va a haber un movimiento significativo en cualquier dirección
        (intensidad), no la dirección. Diseñado para usarse como gate del
        modelo direccional existente, no como modelo de trading directo.

        Las últimas `label_horizon` filas no tienen futuro suficiente y se
        marcan con label=0 (descartables vía dropna del fwd_close en pipeline).
        """
        from numpy.lib.stride_tricks import sliding_window_view

        h = int(self.config.label_horizon)
        m_thr = float(self.config.magnitude_threshold)
        out = df.copy()
        n = len(out)
        close = out['close'].values.astype(np.float64)
        high = out['high'].values.astype(np.float64)
        low = out['low'].values.astype(np.float64)
        atr = np.maximum(out['atr'].values.astype(np.float64), 1e-10)

        if n <= h:
            out['signal'] = np.zeros(n, dtype=np.int32)
            return out

        # Sliding windows: row i = high[i:i+h]. Para cada t, queremos
        # max(high[t+1:t+1+h]) → row (t+1). t válido: 0..n-h-1.
        high_windows = sliding_window_view(high, h)
        low_windows = sliding_window_view(low, h)

        fwd_high = np.full(n, np.nan)
        fwd_low = np.full(n, np.nan)
        fwd_high[:n - h] = high_windows[1:].max(axis=1)
        fwd_low[:n - h] = low_windows[1:].min(axis=1)

        excursion_up = (fwd_high - close) / atr
        excursion_down = (close - fwd_low) / atr
        abs_move = np.maximum(excursion_up, excursion_down)

        # NaN >= m_thr → False, así que las últimas h filas quedan label=0.
        label = (abs_move >= m_thr).astype(np.int32)
        out['signal'] = label

        pos_rate = float(label[:n - h].mean()) if n > h else 0.0
        print(f"[MAGNITUDE] h={h} M={m_thr:.2f} ATR | "
              f"pos_rate={pos_rate:.4f} ({label.sum():,}/{n - h:,})")
        return out
