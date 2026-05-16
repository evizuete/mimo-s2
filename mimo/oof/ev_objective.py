#!/usr/bin/env python3
"""
ev_objective.py

EV-based objective for Optuna. Replays the triple barrier on OOF predictions
and reports realised EV per signal at the threshold that maximises EV_net,
optionally penalising large drawdowns.

Used by OptunaOOFTrainer when objective_kind='ev_net'.
"""
from __future__ import annotations

from typing import Dict

import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Triple-barrier replay (per-row outcome + R-multiple)
# -----------------------------------------------------------------------------

def _simulate_outcomes_for_indices(
    indices: np.ndarray,
    high: np.ndarray, low: np.ndarray, close: np.ndarray, atr: np.ndarray,
    *, horizon: int, tp_mult: float, sl_mult: float, side_is_long: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        outcome_code : int8 array (0=TP, 1=SL, 2=EXPIRE, 3=INVALID)
        r_multiple   : float32 array (realised R)
    Tie-break (TP and SL in same bar): TP gana — coincide con
    triple_barrier_fixed_numba (label=1 si hit_tp <= hit_sl).
    """
    n = len(indices)
    outcome = np.full(n, 3, dtype=np.int8)
    r = np.zeros(n, dtype=np.float32)
    last_valid = len(close) - 1
    for ii in range(n):
        idx = int(indices[ii])
        if idx + horizon > last_valid:
            continue
        a = atr[idx]
        if not np.isfinite(a) or a <= 0:
            continue
        entry = close[idx]
        if side_is_long:
            tp_level = entry + tp_mult * a
            sl_level = entry - sl_mult * a
        else:
            tp_level = entry - tp_mult * a
            sl_level = entry + sl_mult * a

        oc = 2  # EXPIRE
        rv = 0.0
        for k in range(1, horizon + 1):
            j = idx + k
            if side_is_long:
                hit_tp = high[j] >= tp_level
                hit_sl = low[j] <= sl_level
            else:
                hit_tp = low[j] <= tp_level
                hit_sl = high[j] >= sl_level
            if hit_tp:
                oc = 0
                rv = tp_mult
                break
            if hit_sl:
                oc = 1
                rv = -sl_mult
                break
        else:
            exit_close = close[idx + horizon]
            raw = (exit_close - entry) / a
            rv = float(raw if side_is_long else -raw)
            oc = 2
        outcome[ii] = oc
        r[ii] = rv
    return outcome, r


# -----------------------------------------------------------------------------
# Drawdown sobre la serie de señales ordenadas por tiempo
# -----------------------------------------------------------------------------

def _max_drawdown_R(r_signal_sorted: np.ndarray) -> float:
    """Drawdown máximo (en R, valor positivo) de la curva acumulada."""
    if r_signal_sorted.size == 0:
        return 0.0
    eq = np.cumsum(r_signal_sorted)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max())


# -----------------------------------------------------------------------------
# API principal — encuentra el threshold que maximiza EV_net y reporta métricas
# -----------------------------------------------------------------------------

def apply_fixed_threshold(
    df_oof: pd.DataFrame,
    *,
    proba_col: str,
    side_is_long: bool,
    thr: float,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    cost_per_signal: float = 0.05,
    min_signals: int = 1,
) -> Dict[str, float]:
    """
    Aplica un threshold FIJO (sin scanning) a df_oof y devuelve las mismas
    métricas que compute_ev_at_best_threshold pero sin look-ahead.

    Usar cuando el threshold se eligió en un periodo PREVIO (val_internal)
    y se quiere evaluar su performance OOS en otro periodo (test).

    Devuelve dict con mismas keys que compute_ev_at_best_threshold.
    Si no hay suficientes señales tras aplicar thr, devuelve _empty_result.
    """
    needed = {"time", "high", "low", "close", "atr", proba_col}
    miss = needed - set(df_oof.columns)
    if miss:
        raise ValueError(f"df_oof falta columnas: {miss}")

    mask = df_oof[proba_col].notna() & df_oof["atr"].notna() & (df_oof["atr"] > 0)
    df = df_oof.loc[mask].reset_index(drop=False)
    if len(df) < max(min_signals, 50):
        return _empty_result(reason="too_few_rows_fixed_thr")

    df = df.sort_values("time").reset_index(drop=True)
    high  = df_oof["high"].to_numpy(dtype=np.float64)
    low   = df_oof["low"].to_numpy(dtype=np.float64)
    close = df_oof["close"].to_numpy(dtype=np.float64)
    atr   = df_oof["atr"].to_numpy(dtype=np.float64)
    orig_idx = df["index"].to_numpy(dtype=np.int64)
    p = df[proba_col].to_numpy(dtype=np.float64)

    outcome, r_real = _simulate_outcomes_for_indices(
        orig_idx, high, low, close, atr,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        side_is_long=side_is_long,
    )
    valid = outcome != 3
    pred = (p >= thr) & valid
    sig = int(pred.sum())
    if sig < min_signals:
        return _empty_result(reason="too_few_signals_fixed_thr")

    sub_r = r_real[pred]
    sub_outcome = outcome[pred]
    ev_gross = float(sub_r.mean())
    ev_net   = ev_gross - cost_per_signal
    mdd      = _max_drawdown_R(sub_r - cost_per_signal)

    n_tp  = int((sub_outcome == 0).sum())
    n_sl  = int((sub_outcome == 1).sum())
    n_exp = int((sub_outcome == 2).sum())
    prec_tp = n_tp / sig if sig > 0 else 0.0

    return {
        "thr":         float(thr),
        "score":       float(ev_net),
        "ev_net":      float(ev_net),
        "ev_gross":    float(ev_gross),
        "mdd_R":       float(mdd),
        "penalty_mdd": 1.0,
        "penalty_R":   0.0,
        "sig_rate":    float(sig / len(p)),
        "n_signals":   sig,
        "n_TP":  n_tp, "n_SL": n_sl, "n_EXPIRE": n_exp,
        "prec_TP":     float(prec_tp),
        "frac_SL":     float(n_sl / sig),
        "frac_EXP":    float(n_exp / sig),
        "total_R_net": float((sub_r - cost_per_signal).sum()),
    }


def compute_ev_at_best_threshold(
    df_oof: pd.DataFrame,
    *,
    proba_col: str,
    side_is_long: bool,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    cost_per_signal: float = 0.05,
    n_thr: int = 60,
    thr_lo: float = 0.10,
    thr_hi: float = 0.40,
    min_signals: int = 100,
    max_drawdown_R: float = 30.0,
    drawdown_softness: float = 0.5,
) -> Dict[str, float]:
    """
    Replays barreras sobre TODAS las filas con OOF, luego barre thresholds.

    Para cada thr:
      EV_net = mean(R) - cost_per_signal
      MDD    = max drawdown de la curva acumulada (señales ordenadas en tiempo)
      penalty_mdd = max(0, 1 - drawdown_softness * (MDD - max_drawdown_R) / max_drawdown_R)
                    cuando MDD > max_drawdown_R, si no, 1.0
      score = EV_net * penalty_mdd

    Devuelve dict del mejor thr (max score) con todos los detalles.
    Requiere df_oof con columnas: time, high, low, close, atr, <proba_col>.
    """
    needed = {"time", "high", "low", "close", "atr", proba_col}
    miss = needed - set(df_oof.columns)
    if miss:
        raise ValueError(f"df_oof falta columnas: {miss}")

    # Filtrar filas con probabilidad calibrada y ATR válido.
    mask = df_oof[proba_col].notna() & df_oof["atr"].notna() & (df_oof["atr"] > 0)
    df = df_oof.loc[mask].reset_index(drop=False)  # 'index' = posición original
    if len(df) < max(min_signals * 2, 200):
        return _empty_result(reason="too_few_rows")

    df = df.sort_values("time").reset_index(drop=True)
    high = df_oof["high"].to_numpy(dtype=np.float64)
    low = df_oof["low"].to_numpy(dtype=np.float64)
    close = df_oof["close"].to_numpy(dtype=np.float64)
    atr = df_oof["atr"].to_numpy(dtype=np.float64)
    orig_idx = df["index"].to_numpy(dtype=np.int64)
    p = df[proba_col].to_numpy(dtype=np.float64)

    # Pre-compute outcomes una sola vez
    outcome, r_real = _simulate_outcomes_for_indices(
        orig_idx, high, low, close, atr,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        side_is_long=side_is_long,
    )
    valid = outcome != 3
    if valid.sum() < min_signals * 2:
        return _empty_result(reason="too_few_valid")

    # Sweep thresholds
    thrs = np.linspace(thr_lo, thr_hi, n_thr)
    best = None
    for thr in thrs:
        pred = (p >= thr) & valid
        sig = int(pred.sum())
        if sig < min_signals:
            continue
        sub_r = r_real[pred]
        sub_outcome = outcome[pred]
        ev_gross = float(sub_r.mean())
        ev_net = ev_gross - cost_per_signal
        mdd = _max_drawdown_R(sub_r - cost_per_signal)
        # Penalización SUSTRACTIVA por exceso de drawdown: cuando MDD > max_drawdown_R
        # restamos `softness * exceso_normalizado` al ev_net. Mantiene el orden
        # natural — config catastrófica nunca es "mejor" que mediocre saneada.
        if mdd > max_drawdown_R:
            mdd_excess = (mdd - max_drawdown_R) / max(max_drawdown_R, 1e-9)
            penalty_R = drawdown_softness * mdd_excess
        else:
            penalty_R = 0.0
        # score = ev_net - penalty_R (no más multiplicación)
        score = ev_net - penalty_R
        # Para reporting: penalty_mdd 1.0 si limpio, fracción restante si penalizado.
        # Lo dejamos sólo informativo — el ranking usa score puro.
        if mdd > max_drawdown_R:
            penalty_info = max(0.0, 1.0 - drawdown_softness * mdd_excess)
        else:
            penalty_info = 1.0

        n_tp = int((sub_outcome == 0).sum())
        n_sl = int((sub_outcome == 1).sum())
        n_exp = int((sub_outcome == 2).sum())
        prec_tp = n_tp / sig
        info = {
            "thr": float(thr),
            "score": float(score),
            "ev_net": float(ev_net),
            "ev_gross": float(ev_gross),
            "mdd_R": float(mdd),
            "penalty_mdd": float(penalty_info),
            "penalty_R": float(penalty_R),
            "sig_rate": float(sig / len(p)),
            "n_signals": sig,
            "n_TP": n_tp, "n_SL": n_sl, "n_EXPIRE": n_exp,
            "prec_TP": float(prec_tp),
            "frac_SL": float(n_sl / sig),
            "frac_EXP": float(n_exp / sig),
            "total_R_net": float((sub_r - cost_per_signal).sum()),
        }
        if best is None or score > best["score"]:
            best = info

    if best is None:
        return _empty_result(reason="no_thr_with_min_signals")
    return best


def compute_balanced_objective(
    df_oof: pd.DataFrame,
    *,
    long_proba_col: str = "oof_proba_long_cal",
    short_proba_col: str = "oof_proba_short_cal",
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    cost_per_signal: float = 0.05,
    n_thr: int = 60,
    thr_lo: float = 0.10,
    thr_hi: float = 0.40,
    min_signals: int = 100,
    max_drawdown_R: float = 30.0,
) -> Dict[str, dict]:
    """
    Calcula EV_net para LONG y SHORT por separado y devuelve un score
    balanceado: mean(ev_net_long * penalty_long, ev_net_short * penalty_short).

    Devuelve {'score', 'long': {...}, 'short': {...}}.
    """
    long_res = compute_ev_at_best_threshold(
        df_oof, proba_col=long_proba_col, side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )
    short_res = compute_ev_at_best_threshold(
        df_oof, proba_col=short_proba_col, side_is_long=False,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )
    score_long = long_res.get("score", float("-inf"))
    score_short = short_res.get("score", float("-inf"))
    if not np.isfinite(score_long):
        score_long = -1.0
    if not np.isfinite(score_short):
        score_short = -1.0
    score = 0.5 * (score_long + score_short)
    return {"score": float(score), "long": long_res, "short": short_res}


# -----------------------------------------------------------------------------

def _empty_result(reason: str) -> Dict[str, float]:
    return {
        "thr": float("nan"),
        "score": float("-inf"),
        "ev_net": float("nan"),
        "ev_gross": float("nan"),
        "mdd_R": float("nan"),
        "penalty_mdd": 0.0,
        "sig_rate": 0.0,
        "n_signals": 0,
        "n_TP": 0, "n_SL": 0, "n_EXPIRE": 0,
        "prec_TP": 0.0,
        "frac_SL": 0.0,
        "frac_EXP": 0.0,
        "total_R_net": 0.0,
        "_reason": reason,
    }
