# mimo_old/rl_thresholds.py
# Reusable utilities for sweeping RL thresholds by state (and optionally by action).
#
# Expected input (trade-level is best):
#   time: datetime-like
#   state: str
#   action: str  (e.g. 'long'/'short' or 'buy'/'sell')
#   rl_score: float  (e.g. p(TAKE) from policy)
#   pnl: float  (realized pnl for that trade IF executed)
#
# You can also use bar-level candidates if you provide a 'trade_id' and aggregate to trades first.

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Iterable

import math
import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Metrics:
    n: int
    net_pnl: float
    gross_profit: float
    gross_loss: float
    profit_factor: float
    win_rate: float
    max_dd: float
    sharpe_daily_ann: float


def max_drawdown(equity: np.ndarray) -> float:
    """Returns the minimum (most negative) drawdown over the equity curve."""
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity - peak
    return float(dd.min())


def sharpe_daily_ann_from_trade_pnl(df: pd.DataFrame, time_col: str, pnl_col: str) -> float:
    """Approximate annualized daily Sharpe from trade PnL aggregated by day."""
    if df.empty:
        return float("nan")
    d = df[[time_col, pnl_col]].copy()
    d[time_col] = pd.to_datetime(d[time_col])
    d["day"] = d[time_col].dt.floor("D")
    daily = d.groupby("day")[pnl_col].sum().sort_index()
    if len(daily) < 10:
        return float("nan")
    mu = float(daily.mean())
    sd = float(daily.std(ddof=1))
    if sd <= 1e-12:
        return float("nan")
    return (mu / sd) * math.sqrt(252.0)


def compute_metrics(trades: pd.DataFrame, time_col: str = "time", pnl_col: str = "pnl") -> Metrics:
    if trades.empty:
        return Metrics(
            n=0,
            net_pnl=0.0,
            gross_profit=0.0,
            gross_loss=0.0,
            profit_factor=float("nan"),
            win_rate=float("nan"),
            max_dd=0.0,
            sharpe_daily_ann=float("nan"),
        )

    pnl = trades[pnl_col].astype(float).to_numpy()
    net = float(pnl.sum())
    gp = float(pnl[pnl > 0].sum())
    gl = float(-pnl[pnl < 0].sum())  # positive number

    if gl > 1e-12:
        pf = gp / gl
    else:
        pf = float("inf") if gp > 0 else float("nan")

    wr = float((pnl > 0).mean())
    equity = np.cumsum(pnl)
    mdd = max_drawdown(equity)
    sharpe = sharpe_daily_ann_from_trade_pnl(trades, time_col=time_col, pnl_col=pnl_col)

    return Metrics(
        n=int(len(trades)),
        net_pnl=net,
        gross_profit=gp,
        gross_loss=gl,
        profit_factor=pf,
        win_rate=wr,
        max_dd=mdd,
        sharpe_daily_ann=sharpe,
    )


def default_objective(m: Metrics, dd_weight: float = 0.25) -> float:
    """
    Robust default objective:
      + net_pnl
      + capped PF term
      - drawdown penalty
    Tune weights to taste.
    """
    if m.n == 0 or np.isnan(m.profit_factor):
        return -1e18

    pf_term = min(float(m.profit_factor), 5.0)
    dd_penalty = dd_weight * abs(float(m.max_dd))
    return float(m.net_pnl) + 200.0 * pf_term - dd_penalty


def quantile_grid(scores: np.ndarray, n_grid: int = 25, q_lo: float = 0.50, q_hi: float = 0.98) -> np.ndarray:
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([float("inf")], dtype=float)
    qs = np.linspace(q_lo, q_hi, n_grid)
    return np.quantile(scores, qs).astype(float)


def linear_grid(scores: np.ndarray, n_grid: int = 25) -> np.ndarray:
    scores = scores[np.isfinite(scores)]
    if len(scores) == 0:
        return np.array([float("inf")], dtype=float)
    lo, hi = float(np.min(scores)), float(np.max(scores))
    if abs(hi - lo) < 1e-12:
        return np.array([lo], dtype=float)
    return np.linspace(lo, hi, n_grid, dtype=float)


def apply_thresholds(
    df: pd.DataFrame,
    thr_by_state: Dict[str, float],
    *,
    state_col: str = "state",
    score_col: str = "rl_score",
    action_col: str = "action",
    time_col: str = "time",
    pnl_col: str = "pnl",
    allowed_actions: Optional[List[str]] = None,
) -> pd.DataFrame:
    d = df.copy()
    if allowed_actions is not None:
        d = d[d[action_col].isin(allowed_actions)].copy()

    thr = d[state_col].map(thr_by_state).astype(float).fillna(float("inf"))
    executed = d[d[score_col].astype(float) >= thr].copy()
    executed.sort_values(time_col, inplace=True)
    return executed


def sweep_thresholds_by_state(
    df_eval: pd.DataFrame,
    *,
    state_col: str = "state",
    score_col: str = "rl_score",
    action_col: str = "action",
    time_col: str = "time",
    pnl_col: str = "pnl",
    allowed_actions: Optional[List[str]] = None,
    grid_n: int = 25,
    grid_method: str = "quantile",
    dd_weight: float = 0.25,
    min_trades: int = 30,
    q_lo: float = 0.50,
    q_hi: float = 0.98,
) -> Tuple[Dict[str, float], pd.DataFrame, Metrics]:
    """
    Returns:
      - thr_by_state: dict[state] -> threshold
      - per_state_summary: DataFrame
      - global_metrics: Metrics (using best thresholds per state)
    """
    if grid_method not in ("quantile", "linear"):
        raise ValueError("grid_method must be 'quantile' or 'linear'")

    thr_by_state: Dict[str, float] = {}
    rows = []

    grouped = df_eval.groupby(state_col)
    for state, g in grouped:
        gg = g.copy()
        if allowed_actions is not None:
            gg = gg[gg[action_col].isin(allowed_actions)].copy()

        scores = gg[score_col].astype(float).to_numpy()
        if grid_method == "quantile":
            grid = quantile_grid(scores, n_grid=grid_n, q_lo=q_lo, q_hi=q_hi)
        else:
            grid = linear_grid(scores, n_grid=grid_n)

        best = None
        for thr in grid:
            executed = gg[gg[score_col].astype(float) >= float(thr)]
            m = compute_metrics(executed, time_col=time_col, pnl_col=pnl_col)
            if m.n < min_trades:
                continue
            obj = default_objective(m, dd_weight=dd_weight)
            if best is None or obj > best["obj"]:
                best = {"thr": float(thr), "obj": float(obj), "m": m}

        if best is None:
            thr_by_state[state] = float("inf")
            rows.append({
                "state": state,
                "thr": float("inf"),
                "n": 0,
                "net_pnl": 0.0,
                "pf": float("nan"),
                "max_dd": 0.0,
                "sharpe_daily_ann": float("nan"),
                "objective": -1e18,
            })
        else:
            thr_by_state[state] = best["thr"]
            m = best["m"]
            rows.append({
                "state": state,
                "thr": best["thr"],
                "n": m.n,
                "net_pnl": m.net_pnl,
                "pf": m.profit_factor,
                "max_dd": m.max_dd,
                "sharpe_daily_ann": m.sharpe_daily_ann,
                "objective": best["obj"],
            })

    summary = pd.DataFrame(rows).sort_values("objective", ascending=False)

    executed_global = apply_thresholds(
        df_eval,
        thr_by_state,
        state_col=state_col,
        score_col=score_col,
        action_col=action_col,
        time_col=time_col,
        pnl_col=pnl_col,
        allowed_actions=allowed_actions,
    )
    global_metrics = compute_metrics(executed_global, time_col=time_col, pnl_col=pnl_col)

    return thr_by_state, summary, global_metrics
