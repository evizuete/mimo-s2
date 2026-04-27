# mimo_old/rl_trade_log.py
"""
Utilities to build a trade-level dataset suitable for sweeping RL thresholds by state.

Why this module exists
----------------------
To sweep thresholds you need rows that look like:
  time,state,action,rl_score,pnl

Your TradingSimulator.backtest() returns a dict with:
  - "trades": a list (or DataFrame) with per-trade info (entry/exit/pnl, etc.)
  - "rl": optional rl_wrapper.logs (useful for debugging, not required)

This module provides two pragmatic paths:

A) FAST (approximate) sweep dataset:
   - Run ONE backtest with RL "scoring" enabled but with take-threshold so low that
     RL does not filter trades (or RL gating disabled).
   - Extract trades and attach rl_score at entry time by recomputing the RL score
     on the entry bar (deterministic forward pass).

   Then you can sweep thresholds offline by filtering that trade log.
   Caveat: if filtering trades changes later availability/positions, this is an approximation.

B) VALIDATE (accurate) top thresholds:
   - Re-run the full backtest for each candidate threshold-map (by state) and compare.
   This is slower but correct.

The FAST path is usually great for selecting a small set of candidate thresholds,
then you VALIDATE the best 3-5.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class TradeLogSpec:
    # Column names expected/produced
    entry_time_col: str = "entry_time"
    exit_time_col: str = "exit_time"
    state_col: str = "state"
    action_col: str = "side"        # e.g. "long"/"short" or "buy"/"sell"
    pnl_col: str = "pnl"
    rl_score_col: str = "rl_score"  # p(TAKE) or other RL gate score
    time_col: str = "time"          # output: entry time


def _as_trades_df(trades: Any) -> pd.DataFrame:
    if trades is None:
        return pd.DataFrame()
    if isinstance(trades, pd.DataFrame):
        return trades.copy()
    # list of dicts?
    if isinstance(trades, list):
        return pd.DataFrame(trades)
    # numpy structured?
    try:
        return pd.DataFrame(trades)
    except Exception:
        raise TypeError(f"Unsupported trades container: {type(trades)}")


def _to_dt(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, utc=False, errors="coerce")


def attach_rl_score_at_entry(
    *,
    trades_df: pd.DataFrame,
    df_bars: pd.DataFrame,
    sim: Any,
    spec: TradeLogSpec = TradeLogSpec(),
    score_field: str = "p_take",
) -> pd.DataFrame:
    """
    Attach rl_score to each trade by recomputing the RL gate score on the entry bar.

    Requirements:
    - df_bars must include a datetime column "time" (or index) matching trades entry_time.
    - df_bars should already include model predictions and state/market_condition columns,
      typically produced by sim.predict(df, simulation=True).

    This function calls:
      sim.decision_engine.decide_at_bar(...)
    which (in your project) returns a DecisionOutput with fields:
      action, rl (dict with p_take), state/market_condition, etc.

    If your DecisionOutput uses a different structure, adjust the extraction below.
    """
    if trades_df.empty:
        return trades_df

    d = trades_df.copy()

    # normalize bar time column
    if "time" not in df_bars.columns:
        if df_bars.index.name:
            df_bars = df_bars.reset_index()
        else:
            raise ValueError("df_bars must contain a 'time' column or a datetime index")

    df_bars = df_bars.copy()
    df_bars["time"] = _to_dt(df_bars["time"])

    # normalize trade entry time
    if spec.entry_time_col not in d.columns:
        raise ValueError(f"trades_df must contain '{spec.entry_time_col}'")
    d[spec.entry_time_col] = _to_dt(d[spec.entry_time_col])

    # build lookup (time -> row) (if duplicated times, keep last)
    bars_by_time = df_bars.set_index("time")
    # In 1m data it's usually unique; but to be safe:
    if not bars_by_time.index.is_unique:
        bars_by_time = bars_by_time[~bars_by_time.index.duplicated(keep="last")]

    rl_scores: List[float] = []
    states: List[str] = []
    actions: List[str] = []

    for t in d[spec.entry_time_col].tolist():
        if pd.isna(t) or t not in bars_by_time.index:
            rl_scores.append(float("nan"))
            states.append(str(d.loc[d[spec.entry_time_col] == t, spec.state_col].iloc[0]) if spec.state_col in d.columns else "")
            actions.append(str(d.loc[d[spec.entry_time_col] == t, spec.action_col].iloc[0]) if spec.action_col in d.columns else "")
            continue

        row = bars_by_time.loc[t]

        # row may be Series; ensure access with get
        decision = sim.decision_engine.decide_at_bar(
            p_buy_raw=float(row.get("pred_long_raw", np.nan)),
            p_sell_raw=float(row.get("pred_short_raw", np.nan)),
            p_buy_cal=float(row.get("pred_long_cal", np.nan)),
            p_sell_cal=float(row.get("pred_short_cal", np.nan)),
            market_condition=str(row.get("market_condition", row.get("state", ""))),
            o=float(row.get("open", np.nan)),
            h=float(row.get("high", np.nan)),
            l=float(row.get("low", np.nan)),
            c=float(row.get("close", np.nan)),
            atr=float(row.get("atr", np.nan)) if "atr" in row else float("nan"),
            adx14=float(row.get("adx", np.nan)) if "adx" in row else None,
            trend_dir=row.get("trend_dir", None),
        )

        # Extract RL score
        p_take = float("nan")
        if hasattr(decision, "rl") and isinstance(decision.rl, dict):
            p_take = float(decision.rl.get(score_field, float("nan")))
        rl_scores.append(p_take)

        # Extract canonical state / action
        st = getattr(decision, "state", None) or getattr(decision, "market_condition", None) or row.get("state", "")
        states.append(str(st))
        actions.append(str(getattr(decision, "action", "")))

    d[spec.rl_score_col] = rl_scores

    # prefer canonical state/action from decision if present
    d[spec.state_col] = states
    d[spec.action_col] = actions

    # output 'time' as entry time for sweeper
    d[spec.time_col] = d[spec.entry_time_col]

    return d


def build_trade_log_for_sweep(
    *,
    backtest_result: Mapping[str, Any],
    df_bars_predicted: pd.DataFrame,
    sim: Any,
    spec: TradeLogSpec = TradeLogSpec(),
) -> pd.DataFrame:
    """
    Builds a trade-level DataFrame with columns:
      time, state, action, rl_score, pnl

    It uses:
      backtest_result["trades"]  +  attach_rl_score_at_entry(...)
    """
    trades_df = _as_trades_df(backtest_result.get("trades"))
    if trades_df.empty:
        return pd.DataFrame(columns=[spec.time_col, spec.state_col, spec.action_col, spec.rl_score_col, spec.pnl_col])

    trades_df = attach_rl_score_at_entry(trades_df=trades_df, df_bars=df_bars_predicted, sim=sim, spec=spec)

    # ensure required cols exist
    for col in [spec.time_col, spec.state_col, spec.action_col, spec.rl_score_col, spec.pnl_col]:
        if col not in trades_df.columns:
            trades_df[col] = np.nan

    out = trades_df[[spec.time_col, spec.state_col, spec.action_col, spec.rl_score_col, spec.pnl_col]].copy()
    out.rename(columns={spec.action_col: "action"}, inplace=True)
    out.sort_values(spec.time_col, inplace=True)
    return out


def save_trade_log(df_trade_log: pd.DataFrame, path: str) -> str:
    path_l = path.lower()
    if path_l.endswith(".parquet"):
        df_trade_log.to_parquet(path, index=False)
    else:
        df_trade_log.to_csv(path, index=False)
    return path
