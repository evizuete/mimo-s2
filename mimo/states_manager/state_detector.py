"""
state_detector.py
═══════════════════════════════════════════════════════════════════════════
Fusión de regime_detector.py + regime_state_machine.py en un único módulo.

Produce directamente 9 estados sin capa intermedia de "regime":

    TREND_UP            → ADX alto + dirección alcista confirmada
    TREND_DOWN          → ADX alto + dirección bajista confirmada
    TRANSITION_UP       → ADX medio + presión alcista emergente
    TRANSITION_DOWN     → ADX medio + presión bajista emergente
    BREAKOUT_WAIT_UP    → BB comprimido + energía naciente + dirección alcista
    BREAKOUT_WAIT_DOWN  → BB comprimido + energía naciente + dirección bajista
    RANGE               → ADX bajo + ATR medio + sin señal direccional clara
    VOLATILE            → ATR alto (prioridad máxima, independiente de ADX)
    LOW_VOL             → ATR bajo + ADX bajo (mercado dormido → no operar)

Pesos para entrenamiento OOF:
    LOW_VOL  → 0.0  (excluido del entrenamiento)
    VOLATILE → 0.1
    RANGE    → 0.5
    TRANSITION_* → 0.7
    BREAKOUT_WAIT_* → 0.8
    TREND_*  → 1.0

Umbrales:
    Se calculan UNA VEZ en entrenamiento (compute_and_store_thresholds) y se
    persisten en meta.json. En producción se inyectan via inject_thresholds()
    garantizando consistencia train/inference independientemente del tamaño
    de la ventana cargada.

Uso:
    from mimo_old.state_detector import StateDetector, StateConfig

    detector = StateDetector(StateConfig())
    df = detector.detect(df)          # añade df["state"], df["state_weight"]

    # En DataPipeline (reemplaza add_mimo_state):
    df = detector.detect(df)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import pandas_ta_classic as ta


# ═══════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class StateConfig:
    """
    Configuración unificada para StateDetector.
    Fusiona RegimeConfig + StateConfig anteriores.
    """

    # ── Umbrales ADX ──────────────────────────────────────────────────
    adx_trend_threshold: float = 25.0   # ADX > este → tendencia confirmada
    adx_range_max: float = 18.0         # ADX < este → rango fuerte

    # ── Volatilidad (ATR normalizado) ─────────────────────────────────
    volatility_low_percentile: float = 0.30   # ATR < p30 → low_vol
    volatility_high_percentile: float = 0.80  # ATR > p80 → high_vol / VOLATILE

    # ── EMA lenta (slope de tendencia macro) ──────────────────────────
    ema_slow_period: int = 50
    slope_lookback: int = 5

    # ── BB (compresión / expansión) ───────────────────────────────────
    bb_width_p20: float = 0.20    # compresión extrema
    bb_width_p35: float = 0.35    # baja volatilidad BB
    bb_width_p70: float = 0.70    # BB amplio (volatile local)
    bb_width_slope_min: float = 0.0
    atr_slope_min: float = 0.0

    # ── Breakout confirmation ─────────────────────────────────────────
    bb_pos_breakout_up: float = 0.85
    bb_pos_breakout_dn: float = 0.15
    rsi_breakout_up: float = 55.0
    rsi_breakout_dn: float = 45.0
    breakout_becomes_trend: bool = True

    # ── MACD slope ────────────────────────────────────────────────────
    macd_slope_window: int = 8

    # ── Dirección (trend_dir) ─────────────────────────────────────────
    trend_dir_up_min: float = 0.25
    trend_dir_dn_max: float = -0.25

    # ── Range expansion (volatile local) ─────────────────────────────
    range_expansion_p80: float = 0.80

    # ── Umbrales fijos calculados en entrenamiento ────────────────────
    # Si se especifican, NO se recalculan en inferencia.
    fixed_vol_low: Optional[float] = None
    fixed_vol_high: Optional[float] = None
    fixed_bb_width_p20: Optional[float] = None
    fixed_bb_width_p35: Optional[float] = None
    fixed_bb_width_p70: Optional[float] = None
    fixed_range_expansion_p80: Optional[float] = None


# ═══════════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════════

# Estados válidos
STATES = (
    "TREND_UP",
    "TREND_DOWN",
    "TRANSITION_UP",
    "TRANSITION_DOWN",
    "BREAKOUT_WAIT_UP",
    "BREAKOUT_WAIT_DOWN",
    "RANGE",
    "VOLATILE",
    "LOW_VOL",
)

# Pesos para entrenamiento OOF
# LOW_VOL = 0.0 → excluido del entrenamiento
STATE_WEIGHTS: Dict[str, float] = {
    "TREND_UP":            1.0,
    "TREND_DOWN":          1.0,
    "TRANSITION_UP":       0.7,
    "TRANSITION_DOWN":     0.7,
    "BREAKOUT_WAIT_UP":    0.8,
    "BREAKOUT_WAIT_DOWN":  0.8,
    "RANGE":               0.5,
    "VOLATILE":            0.1,
    "LOW_VOL":             0.0,
}

# Flag: estos estados no deben generar señales operativas
NO_TRADE_STATES = frozenset({"LOW_VOL"})


# ═══════════════════════════════════════════════════════════════════════
# UTILIDADES INTERNAS
# ═══════════════════════════════════════════════════════════════════════

def _safe(df: pd.DataFrame, col: str, default: float = np.nan) -> pd.Series:
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index, dtype=np.float32)


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    """Proxy de pendiente: valor actual − media del período anterior."""
    return x - x.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)


# ═══════════════════════════════════════════════════════════════════════
# DETECTOR UNIFICADO
# ═══════════════════════════════════════════════════════════════════════

class StateDetector:
    """
    Detecta el estado de mercado en un único paso.
    Reemplaza RegimeDetector + StateMachine.

    Prioridad de clasificación (de mayor a menor):
        1. VOLATILE          → ATR muy alto (override global)
        2. LOW_VOL           → ATR muy bajo + ADX bajo
        3. TREND_UP/DOWN     → ADX alto + confirmación DM/slope/breakout
        4. BREAKOUT_WAIT_*   → BB comprimido + energía naciente + dirección
        5. TRANSITION_*      → ADX medio + presión direccional
        6. RANGE             → ADX bajo, ATR medio, sin señal clara (default)
    """

    def __init__(self, config: StateConfig = StateConfig()):
        self.config = config
        self._cache_key: Optional[Tuple] = None
        self._cached_thresholds: Optional[Dict[str, float]] = None

    # ──────────────────────────────────────────────────────────────────
    # API PÚBLICA
    # ──────────────────────────────────────────────────────────────────

    def detect(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Añade al DataFrame:
            df["state"]         → estado de mercado (str, ver STATES)
            df["state_weight"]  → peso para entrenamiento OOF (float32)

        Columnas requeridas en df:
            close, atr, atr_norm, adx, dm_diff,
            bb_width, bb_position, rsi, macd_hist,
            trend_dir, range_expansion

        Columnas opcionales (se calculan si no existen):
            ema_slow, ema_slow_slope
        """
        df = df.copy()

        # 1. EMA lenta + slope (si no existen)
        if "ema_slow" not in df.columns:
            df["ema_slow"] = ta.ema(df["close"], length=self.config.ema_slow_period)

        if "ema_slow_slope" not in df.columns:
            df["ema_slow_slope"] = (
                (df["ema_slow"] - df["ema_slow"].shift(self.config.slope_lookback))
                / (df["atr"] + 1e-10)
            )

        # 2. Umbrales dinámicos (o fijos si están disponibles)
        thr = self._get_thresholds(df)

        # 3. Clasificar
        df["state"] = self._classify(df, thr)

        # 4. Pesos
        df["state_weight"] = (
            df["state"].map(STATE_WEIGHTS).fillna(0.5).astype(np.float32)
        )

        return df

    def inject_thresholds(self, thresholds: Dict[str, float]) -> None:
        """
        Inyecta umbrales fijos calculados en entrenamiento.
        Llamar desde DataPipeline.load_scalers() para garantizar
        consistencia train/inference.
        """
        cfg = self.config
        cfg.fixed_vol_low            = thresholds.get("vol_low")
        cfg.fixed_vol_high           = thresholds.get("vol_high")
        cfg.fixed_bb_width_p20       = thresholds.get("bb_p20")
        cfg.fixed_bb_width_p35       = thresholds.get("bb_p35")
        cfg.fixed_bb_width_p70       = thresholds.get("bb_p70")
        cfg.fixed_range_expansion_p80 = thresholds.get("rexp_p80")
        # Invalidar caché
        self._cache_key = None
        self._cached_thresholds = None

    def compute_thresholds(self, df_prepared: pd.DataFrame) -> Dict[str, float]:
        """
        Calcula los umbrales sobre el DataFrame de entrenamiento.
        Llamar UNA VEZ tras prepare_data() antes de save_scalers().

        Devuelve dict listo para persistir en meta.json:
            vol_low, vol_high, bb_p20, bb_p35, bb_p70, rexp_p80
        """
        cfg = self.config
        thresholds: Dict[str, float] = {}

        if "atr_norm" not in df_prepared.columns:
            raise ValueError("df_prepared must contain 'atr_norm'")

        thresholds["vol_low"]  = float(df_prepared["atr_norm"].quantile(cfg.volatility_low_percentile))
        thresholds["vol_high"] = float(df_prepared["atr_norm"].quantile(cfg.volatility_high_percentile))

        for col, keys, qs in [
            ("bb_width",        ["bb_p20", "bb_p35", "bb_p70"], [0.20, 0.35, 0.70]),
            ("range_expansion", ["rexp_p80"],                    [0.80]),
        ]:
            if col not in df_prepared.columns:
                raise ValueError(f"df_prepared must contain '{col}'")
            vals = df_prepared[col].dropna()
            for key, q in zip(keys, qs):
                thresholds[key] = float(np.nanquantile(vals, q))

        print(f"\t[StateDetector] Thresholds computed from {len(df_prepared):,} rows: {thresholds}")
        return thresholds

    # ──────────────────────────────────────────────────────────────────
    # LÓGICA INTERNA
    # ──────────────────────────────────────────────────────────────────

    def _get_thresholds(self, df: pd.DataFrame) -> Dict[str, float]:
        """Devuelve umbrales fijos (si existen) o los calcula dinámicamente."""
        cfg = self.config

        all_fixed = all(v is not None for v in [
            cfg.fixed_vol_low, cfg.fixed_vol_high,
            cfg.fixed_bb_width_p20, cfg.fixed_bb_width_p35,
            cfg.fixed_bb_width_p70, cfg.fixed_range_expansion_p80,
        ])

        if all_fixed:
            return {
                "vol_low":        cfg.fixed_vol_low,
                "vol_high":       cfg.fixed_vol_high,
                "bb_p20":         cfg.fixed_bb_width_p20,
                "bb_p35":         cfg.fixed_bb_width_p35,
                "bb_p70":         cfg.fixed_bb_width_p70,
                "rexp_p80":       cfg.fixed_range_expansion_p80,
            }

        # Caché dinámica (evita recalcular si el df no cambió)
        cache_key = (len(df), float(df["atr_norm"].mean()), float(df["atr_norm"].std()))
        if self._cache_key == cache_key and self._cached_thresholds is not None:
            return self._cached_thresholds

        bb_width  = _safe(df, "bb_width")
        range_exp = _safe(df, "range_expansion")

        thresholds = {
            "vol_low":  float(df["atr_norm"].quantile(cfg.volatility_low_percentile)),
            "vol_high": float(df["atr_norm"].quantile(cfg.volatility_high_percentile)),
            "bb_p20":   float(np.nanquantile(bb_width,  0.20)) if np.isfinite(bb_width).any() else cfg.bb_width_p20,
            "bb_p35":   float(np.nanquantile(bb_width,  0.35)) if np.isfinite(bb_width).any() else cfg.bb_width_p35,
            "bb_p70":   float(np.nanquantile(bb_width,  0.70)) if np.isfinite(bb_width).any() else cfg.bb_width_p70,
            "rexp_p80": float(np.nanquantile(range_exp, 0.80)) if np.isfinite(range_exp).any() else cfg.range_expansion_p80,
        }

        self._cache_key = cache_key
        self._cached_thresholds = thresholds
        return thresholds

    def _classify(self, df: pd.DataFrame, thr: Dict[str, float]) -> pd.Series:
        """Clasificación vectorizada en un único paso."""
        cfg = self.config

        # ── Señales base ───────────────────────────────────────────────
        adx       = _safe(df, "adx")
        atr_norm  = _safe(df, "atr_norm")
        bb_width  = _safe(df, "bb_width")
        bb_pos    = _safe(df, "bb_position")
        trend_dir = _safe(df, "trend_dir")
        macd_hist = _safe(df, "macd_hist")
        rsi       = _safe(df, "rsi")
        range_exp = _safe(df, "range_expansion")
        dm_diff   = _safe(df, "dm_diff")
        ema_slope = _safe(df, "ema_slow_slope")

        # ── Máscaras ATR ───────────────────────────────────────────────
        atr_high = atr_norm > thr["vol_high"]
        atr_low  = atr_norm < thr["vol_low"]
        atr_mid  = ~atr_high & ~atr_low

        # ── Máscaras ADX ───────────────────────────────────────────────
        adx_trend  = adx >= cfg.adx_trend_threshold
        adx_low    = adx < cfg.adx_range_max
        adx_mid    = ~adx_trend & ~adx_low   # zona de transición

        # ── Dirección ──────────────────────────────────────────────────
        macd_slope = _rolling_slope(macd_hist, cfg.macd_slope_window)
        dir_up = (trend_dir >= cfg.trend_dir_up_min)  | (macd_slope > 0) | (ema_slope > 0) | (dm_diff > 0)
        dir_dn = (trend_dir <= cfg.trend_dir_dn_max)  | (macd_slope < 0) | (ema_slope < 0) | (dm_diff < 0)

        # Dirección fuerte para tendencia (requiere confirmación múltiple)
        strong_up = (
            ((trend_dir >= cfg.trend_dir_up_min).astype(int) +
             (macd_slope > 0).astype(int) +
             (ema_slope > 0).astype(int) +
             (dm_diff > 0).astype(int)) >= 2
        )
        strong_dn = (
            ((trend_dir <= cfg.trend_dir_dn_max).astype(int) +
             (macd_slope < 0).astype(int) +
             (ema_slope < 0).astype(int) +
             (dm_diff < 0).astype(int)) >= 2
        )

        # ── BB ─────────────────────────────────────────────────────────
        bb_compressed  = bb_width <= thr["bb_p20"]
        bb_low         = bb_width <= thr["bb_p35"]
        bb_wide        = bb_width >= thr["bb_p70"]

        bb_width_slope = _rolling_slope(bb_width, 12)
        atr_slope_ser  = _rolling_slope(_safe(df, "atr"), 12)
        is_expanding   = (bb_width_slope >= cfg.bb_width_slope_min) & (atr_slope_ser >= cfg.atr_slope_min)

        # ── Volatile local ─────────────────────────────────────────────
        volatile_local = (range_exp >= thr["rexp_p80"]) | bb_wide

        # ── Breakout ───────────────────────────────────────────────────
        breakout_up = (bb_pos >= cfg.bb_pos_breakout_up) & (rsi >= cfg.rsi_breakout_up) & (macd_slope > 0)
        breakout_dn = (bb_pos <= cfg.bb_pos_breakout_dn) & (rsi <= cfg.rsi_breakout_dn) & (macd_slope < 0)

        # ── Clasificación (orden = prioridad) ──────────────────────────
        # Inicio: todo es RANGE
        state = pd.Series("RANGE", index=df.index, dtype=object)

        # Nivel 5: TRANSITION (ADX medio o presión direccional sin tendencia confirmada)
        in_transition = ~adx_trend & (dir_up | dir_dn)
        state[in_transition & dir_up] = "TRANSITION_UP"
        state[in_transition & dir_dn] = "TRANSITION_DOWN"

        # Nivel 4: BREAKOUT_WAIT (compresión + energía + dirección)
        in_breakout_wait = bb_compressed & is_expanding & (dir_up | dir_dn)
        state[in_breakout_wait & dir_up] = "BREAKOUT_WAIT_UP"
        state[in_breakout_wait & dir_dn] = "BREAKOUT_WAIT_DOWN"

        # Nivel 3: TREND (ADX alto + dirección fuerte)
        state[adx_trend & strong_up] = "TREND_UP"
        state[adx_trend & strong_dn] = "TREND_DOWN"

        # Breakout confirmado → TREND directamente
        if cfg.breakout_becomes_trend:
            state[breakout_up] = "TREND_UP"
            state[breakout_dn] = "TREND_DOWN"

        # Nivel 2: LOW_VOL (ATR bajo + ADX bajo → mercado dormido)
        # Excepción: si hay breakout_wait emergente, mantenemos ese estado
        low_vol_mask = atr_low & adx_low
        state[low_vol_mask] = "LOW_VOL"
        state[low_vol_mask & in_breakout_wait & dir_up] = "BREAKOUT_WAIT_UP"
        state[low_vol_mask & in_breakout_wait & dir_dn] = "BREAKOUT_WAIT_DOWN"

        # Nivel 1: VOLATILE (ATR alto → override máximo, salvo tendencia fuerte)
        # Si hay tendencia fuerte con alta volatilidad, mantenemos la tendencia
        state[atr_high & ~(adx_trend & (strong_up | strong_dn))] = "VOLATILE"

        # Rango fuerte puro (ADX muy bajo, ATR medio, sin volatile local)
        strong_range = adx_low & atr_mid & ~volatile_local & ~in_breakout_wait
        state[strong_range] = "RANGE"

        return state


# ═══════════════════════════════════════════════════════════════════════
# COMPATIBILIDAD: add_mimo_state (reemplaza la función del mismo nombre)
# ═══════════════════════════════════════════════════════════════════════

def add_mimo_state(
    df: pd.DataFrame,
    cfg: Optional[StateConfig] = None,
    *,
    set_market_condition: bool = True,
    ensure_regime: bool = True,         # mantenido por compatibilidad, ignorado
    use_regime_prior: bool = True,      # mantenido por compatibilidad, ignorado
) -> pd.DataFrame:
    """
    Drop-in replacement de add_mimo_state() de regime_state_machine.py.

    Añade:
        df["state"]
        df["state_weight"]
        df["market_condition"] = df["state"]  (si set_market_condition=True)

    Nota: los parámetros ensure_regime y use_regime_prior se mantienen
    por compatibilidad con código existente pero ya no tienen efecto
    (la lógica de regime está integrada en StateDetector).
    """
    detector = StateDetector(cfg or StateConfig())
    out = detector.detect(df)

    if set_market_condition:
        out["market_condition"] = out["state"]

    return out