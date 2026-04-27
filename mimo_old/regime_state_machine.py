"""
mimo_regime_state_machine_v2.py

V2: integrates with your existing RegimeDetector (regime_detector.py) instead of ignoring it.

Design:
- Keep your RegimeDetector (5 regimes) as the "macro" regime:
    high_volatility, trending_up, trending_down, low_volatility, ranging
- Add a higher-resolution "state" layer ONLY where it matters (low_volatility / ranging),
  to fix flip-flop and pre-breakout behavior:
    RANGE
    TRANSITION_UP / TRANSITION_DOWN
    BREAKOUT_WAIT_UP / BREAKOUT_WAIT_DOWN
    TREND_UP / TREND_DOWN
    VOLATILE

Priority rules:
- If RegimeDetector says trending_up/down -> state is TREND_UP/DOWN (hard override).
- If RegimeDetector says high_volatility -> state is VOLATILE (hard override).
- Else (low_volatility or ranging) -> use BB compression/expansion + direction pressure to decide:
    RANGE vs TRANSITION_* vs BREAKOUT_WAIT_*

Assumptions:
- You confirmed bb_* exist. This script expects (at least) these columns:
    adx, atr, bb_width, bb_position, trend_dir, macd_hist, rsi, range_expansion
- If 'regime' doesn't exist, we can run RegimeDetector automatically.

Usage (minimal integration):
1) Add the file to your project (mimo_old/ or root).
2) In your pipeline after feature building:
      from mimo_regime_state_machine_v2 import add_mimo_state_v2
      df = add_mimo_state_v2(df, set_market_condition=True, ensure_regime=True)
   This adds:
      df["state"]
      df["market_condition"] = df["state"]   (optional)
3) In calibration percentiles:
      percentiles = ProbsCalibration().compute_percentiles_by_regime(
          df_with_oof, proba_col="oof_proba_cal", regime_col="state",
          map_to_3_regimes=False, min_n=800
      )
   Save to a new filename (e.g., percentiles_*_state.json) and load it in DecisionEngine.

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd


# -----------------------------
# Configuration
# -----------------------------

@dataclass
class StateConfig:
    # Keep aligned with RegimeConfig defaults (regime_detector.py)
    adx_trend_threshold: float = 25.0  # should match RegimeConfig.adx_trend_threshold

    # For early transition detection inside non-trend macro regimes
    adx_range_max: float = 18.0

    # Volatility / BB compression thresholds (computed from data; these are fallback values)
    bb_width_p20: float = 0.20
    bb_width_p35: float = 0.35

    # "Energy waking up" requirements (slope proxies)
    bb_width_slope_min: float = 0.0
    atr_slope_min: float = 0.0

    # Breakout confirmation thresholds
    bb_pos_breakout_up: float = 0.85
    bb_pos_breakout_dn: float = 0.15
    rsi_breakout_up: float = 55.0
    rsi_breakout_dn: float = 45.0

    # MACD slope window
    macd_slope_window: int = 8

    # Direction thresholds (depends on your trend_dir definition)
    trend_dir_up_min: float = 0.25
    trend_dir_dn_max: float = -0.25

    # Volatile detector (fallback)
    range_expansion_p80: float = 0.80
    volatile_bb_width_p70: float = 0.70

    # Whether to treat breakout confirmation as TREND immediately
    breakout_becomes_trend: bool = True

    # Umbrales fijos calculados en entrenamiento (si se especifican, no se
    # recalculan sobre la ventana de inferencia)
    fixed_bb_width_p20: float | None = None
    fixed_bb_width_p35: float | None = None
    fixed_bb_width_p70: float | None = None
    fixed_range_expansion_p80: float | None = None


# -----------------------------
# Utilities
# -----------------------------

def _safe_series(df: pd.DataFrame, name: str, default: float = np.nan) -> pd.Series:
    if name in df.columns:
        return df[name]
    return pd.Series([default] * len(df), index=df.index)


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    # slope proxy: current - mean(prev window)
    return x - x.rolling(window, min_periods=max(3, window // 2)).mean().shift(1)


def _compute_dynamic_thresholds(df: pd.DataFrame, cfg: StateConfig) -> Dict[str, float]:
    out: Dict[str, float] = {}

    if (cfg.fixed_bb_width_p20 is not None and cfg.fixed_bb_width_p35 is not None and cfg.fixed_bb_width_p70 is not None
        and cfg.fixed_range_expansion_p80 is not None):

        out['bb_width_p20'] = cfg.fixed_bb_width_p20
        out['bb_width_p35'] = cfg.fixed_bb_width_p35
        out['bb_width_p70'] = cfg.fixed_bb_width_p70
        out['range_expansion_p80'] = cfg.fixed_range_expansion_p80
        return out

    bb_width = _safe_series(df, "bb_width")
    range_exp = _safe_series(df, "range_expansion")

    if np.isfinite(bb_width).any():
        out["bb_width_p20"] = float(np.nanquantile(bb_width, 0.20))
        out["bb_width_p35"] = float(np.nanquantile(bb_width, 0.35))
        out["bb_width_p70"] = float(np.nanquantile(bb_width, 0.70))
    else:
        out["bb_width_p20"] = cfg.bb_width_p20
        out["bb_width_p35"] = cfg.bb_width_p35
        out["bb_width_p70"] = cfg.volatile_bb_width_p70

    if np.isfinite(range_exp).any():
        out["range_expansion_p80"] = float(np.nanquantile(range_exp, 0.80))
    else:
        out["range_expansion_p80"] = cfg.range_expansion_p80

    return out


# -----------------------------
# Core classifier
# -----------------------------

def classify_mimo_state(
    df_features: pd.DataFrame,
    cfg: Optional[StateConfig] = None,
    *,
    use_regime_prior: bool = True,
) -> pd.Series:
    """
    Returns per-bar state labels.

    If use_regime_prior=True and df_features has a 'regime' column:
      trending_up/down -> TREND_*
      high_volatility  -> VOLATILE
      low_volatility/ranging -> refined states via BB/ATR/etc.
    """
    cfg = cfg or StateConfig()
    df = df_features

    # Required features (you said bb_* exist)
    adx = _safe_series(df, "adx")
    atr = _safe_series(df, "atr")
    bb_width = _safe_series(df, "bb_width")
    bb_pos = _safe_series(df, "bb_position")
    trend_dir = _safe_series(df, "trend_dir")
    macd_hist = _safe_series(df, "macd_hist")
    rsi = _safe_series(df, "rsi")
    range_exp = _safe_series(df, "range_expansion")

    thr = _compute_dynamic_thresholds(df, cfg)
    bb_p20 = thr["bb_width_p20"]
    bb_p35 = thr["bb_width_p35"]
    bb_p70 = thr["bb_width_p70"]
    rexp_p80 = thr["range_expansion_p80"]

    # Slopes
    bb_width_slope = _rolling_slope(bb_width, window=12)
    atr_slope = _rolling_slope(atr, window=12)
    macd_slope = _rolling_slope(macd_hist, window=max(5, cfg.macd_slope_window))

    # Base masks
    is_compressed = bb_width <= bb_p20
    is_lowvol = bb_width <= bb_p35
    is_expanding = (bb_width_slope >= cfg.bb_width_slope_min) & (atr_slope >= cfg.atr_slope_min)

    # Direction pressure
    dir_up = (trend_dir >= cfg.trend_dir_up_min) | (macd_slope > 0)
    dir_dn = (trend_dir <= cfg.trend_dir_dn_max) | (macd_slope < 0)

    # Volatile local detector (inside non-volatile macro states)
    is_volatile_local = (range_exp >= rexp_p80) | (bb_width >= bb_p70)

    # Breakout confirmation
    breakout_up = (bb_pos >= cfg.bb_pos_breakout_up) & (rsi >= cfg.rsi_breakout_up) & (macd_slope > 0)
    breakout_dn = (bb_pos <= cfg.bb_pos_breakout_dn) & (rsi <= cfg.rsi_breakout_dn) & (macd_slope < 0)

    # Default init
    state = pd.Series(["RANGE"] * len(df), index=df.index, dtype="object")

    # Refined states (only used when not overridden by macro regime)
    # RANGE: low ADX + not volatile
    in_range = (adx <= cfg.adx_range_max) & (~is_volatile_local)

    # TRANSITION: not in range, not trending, but directional pressure exists
    in_transition = (~in_range) & (adx < cfg.adx_trend_threshold) & (dir_up | dir_dn)

    # BREAKOUT_WAIT: compressed + energy waking + directional pressure
    in_breakout_wait = is_compressed & is_expanding & (dir_up | dir_dn)

    state[is_volatile_local] = "VOLATILE"
    state[in_transition & dir_up] = "TRANSITION_UP"
    state[in_transition & dir_dn] = "TRANSITION_DOWN"
    state[in_breakout_wait & dir_up] = "BREAKOUT_WAIT_UP"
    state[in_breakout_wait & dir_dn] = "BREAKOUT_WAIT_DOWN"

    if cfg.breakout_becomes_trend:
        state[breakout_up] = "TREND_UP"
        state[breakout_dn] = "TREND_DOWN"

    # If ADX already above trend threshold, declare trend (even if macro regime missing)
    state[(adx >= cfg.adx_trend_threshold) & dir_up] = "TREND_UP"
    state[(adx >= cfg.adx_trend_threshold) & dir_dn] = "TREND_DOWN"

    # Apply macro regime overrides if requested
    if use_regime_prior and "regime" in df.columns:
        regime = df["regime"].astype(str)

        # Hard overrides:
        state[regime == "trending_up"] = "TREND_UP"
        state[regime == "trending_down"] = "TREND_DOWN"
        state[regime == "high_volatility"] = "VOLATILE"

        # For low_volatility/ranging: keep refined states as computed (including VOLATILE local)

    # Ensure strong range stays range (last pass)
    state[in_range] = "RANGE"

    return state


def add_mimo_state(
    df_features: pd.DataFrame,
    cfg: Optional[StateConfig] = None,
    *,
    set_market_condition: bool = True,
    ensure_regime: bool = True,
    use_regime_prior: bool = True,
) -> pd.DataFrame:
    """
    Adds:
      - df["state"]
      - optionally df["market_condition"] = df["state"]

    If ensure_regime=True and 'regime' not present, attempts to run your RegimeDetector.
    """
    out = df_features.copy()

    if ensure_regime and "regime" not in out.columns:
        try:
            from mimo.strategies.regime_detector import RegimeDetector  # type: ignore
        except Exception:
            # fallback: try local import (if script is in project root)
            from mimo.strategies.regime_detector import RegimeDetector  # type: ignore

        out = RegimeDetector().detect_regime(out)

    out["state"] = classify_mimo_state(out, cfg=cfg, use_regime_prior=use_regime_prior)

    if set_market_condition:
        out["market_condition"] = out["state"]

    return out
