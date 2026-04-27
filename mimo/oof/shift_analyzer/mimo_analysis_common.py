from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


DEFAULT_SCORE_CANDIDATES = [
    "p_calibrated", "p_cal", "prob_calibrated", "prob_cal", "score_calibrated",
    "p_buy_calibrated", "p_long_calibrated", "probability", "score", "y_prob"
]
DEFAULT_TARGET_CANDIDATES = ["signal", "target", "y_true", "label", "is_positive"]
DEFAULT_STATE_CANDIDATES = ["state", "market_state", "regime"]
DEFAULT_TIME_CANDIDATES = ["time", "timestamp", "datetime", "date"]
DEFAULT_SIDE_CANDIDATES = ["side", "direction"]
DEFAULT_PNL_CANDIDATES = ["pnl", "net_pnl", "trade_pnl", "reward", "realized_pnl"]
DEFAULT_DECISION_CANDIDATES = [
    "decision", "accepted", "take_trade", "take", "is_accepted", "decision_accept"
]
DEFAULT_REASON_CANDIDATES = ["reason", "decision_reason", "block_reason", "debug_reason"]
DEFAULT_SESSION_FLAGS = ["is_asia", "is_london", "is_ny", "is_overlap"]


@dataclass
class ResolvedColumns:
    score: str
    target: str
    state: Optional[str] = None
    time: Optional[str] = None
    side: Optional[str] = None
    pnl: Optional[str] = None
    decision: Optional[str] = None
    reason: Optional[str] = None


def load_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input format: {suffix}. Use CSV, parquet or pickle.")


def save_json(data: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def first_existing(df: pd.DataFrame, candidates: Sequence[str], explicit: Optional[str] = None) -> Optional[str]:
    if explicit:
        if explicit not in df.columns:
            raise ValueError(f"Column '{explicit}' not found. Available columns: {sorted(df.columns)}")
        return explicit
    for c in candidates:
        if c in df.columns:
            return c
    return None


def normalize_binary_target(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        uniq = set(pd.Series(vals).dropna().unique().tolist())
        if uniq.issubset({0, 1}):
            return vals.astype("Int64").astype(float)
        raise ValueError(f"Target column '{s.name}' must be binary 0/1. Unique values sample: {sorted(list(uniq))[:10]}")
    lowered = s.astype(str).str.strip().str.lower()
    mapping = {
        "1": 1, "0": 0, "true": 1, "false": 0, "yes": 1, "no": 0,
        "take": 1, "skip": 0, "accepted": 1, "rejected": 0, "buy": 1, "sell": 0,
    }
    out = lowered.map(mapping)
    if out.isna().all():
        raise ValueError(f"Could not interpret target column '{s.name}' as binary.")
    return out.astype(float)


def normalize_decision(s: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(s):
        return s.astype(int)
    if pd.api.types.is_numeric_dtype(s):
        vals = pd.to_numeric(s, errors="coerce")
        return (vals > 0).astype(float)
    lowered = s.astype(str).str.strip().str.lower()
    mapping = {
        "take": 1, "accept": 1, "accepted": 1, "true": 1, "1": 1, "yes": 1,
        "skip": 0, "reject": 0, "rejected": 0, "false": 0, "0": 0, "no": 0,
        "buy": 1, "sell": 1, "none": 0,
    }
    return lowered.map(mapping).astype(float)


def resolve_columns(
    df: pd.DataFrame,
    score_col: Optional[str] = None,
    target_col: Optional[str] = None,
    state_col: Optional[str] = None,
    time_col: Optional[str] = None,
    side_col: Optional[str] = None,
    pnl_col: Optional[str] = None,
    decision_col: Optional[str] = None,
    reason_col: Optional[str] = None,
) -> ResolvedColumns:
    score = first_existing(df, DEFAULT_SCORE_CANDIDATES, score_col)
    target = first_existing(df, DEFAULT_TARGET_CANDIDATES, target_col)
    if score is None:
        raise ValueError(f"Could not resolve score column. Candidates: {DEFAULT_SCORE_CANDIDATES}")
    if target is None:
        raise ValueError(f"Could not resolve target column. Candidates: {DEFAULT_TARGET_CANDIDATES}")
    return ResolvedColumns(
        score=score,
        target=target,
        state=first_existing(df, DEFAULT_STATE_CANDIDATES, state_col),
        time=first_existing(df, DEFAULT_TIME_CANDIDATES, time_col),
        side=first_existing(df, DEFAULT_SIDE_CANDIDATES, side_col),
        pnl=first_existing(df, DEFAULT_PNL_CANDIDATES, pnl_col),
        decision=first_existing(df, DEFAULT_DECISION_CANDIDATES, decision_col),
        reason=first_existing(df, DEFAULT_REASON_CANDIDATES, reason_col),
    )


def add_period_column(
    df: pd.DataFrame,
    time_col: str,
    train_end: Optional[str],
    holdout_start: Optional[str],
    period_col_name: str = "period",
) -> pd.DataFrame:
    out = df.copy()
    out[time_col] = pd.to_datetime(out[time_col], errors="coerce")
    if out[time_col].isna().all():
        raise ValueError(f"Time column '{time_col}' could not be parsed as datetime.")
    out[period_col_name] = "unknown"
    if train_end is not None:
        out.loc[out[time_col] <= pd.Timestamp(train_end), period_col_name] = "train"
    if holdout_start is not None:
        out.loc[out[time_col] >= pd.Timestamp(holdout_start), period_col_name] = "holdout"
    if train_end is not None and holdout_start is not None:
        mask_middle = (out[time_col] > pd.Timestamp(train_end)) & (out[time_col] < pd.Timestamp(holdout_start))
        out.loc[mask_middle, period_col_name] = "validation"
    return out


def ensure_period_column(df: pd.DataFrame, explicit_period_col: Optional[str], time_col: Optional[str], train_end: Optional[str], holdout_start: Optional[str]) -> tuple[pd.DataFrame, str]:
    if explicit_period_col:
        if explicit_period_col not in df.columns:
            raise ValueError(f"Period column '{explicit_period_col}' not found.")
        out = df.copy()
        out[explicit_period_col] = out[explicit_period_col].astype(str)
        return out, explicit_period_col
    if time_col is None:
        raise ValueError("Need either --period-col or a parseable time column plus --train-end/--holdout-start.")
    out = add_period_column(df, time_col, train_end, holdout_start)
    return out, "period"


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    return float(np.mean((y_prob - y_true) ** 2))


def log_loss_binary(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(y_prob, eps, 1 - eps)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))


def ece_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True)
    ece = 0.0
    n = len(y_true)
    for b in range(1, n_bins + 1):
        mask = idx == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n) * abs(acc - conf)
    return float(ece)


def mce_score(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    idx = np.digitize(y_prob, bins, right=True)
    errors = []
    for b in range(1, n_bins + 1):
        mask = idx == b
        if not np.any(mask):
            continue
        acc = np.mean(y_true[mask])
        conf = np.mean(y_prob[mask])
        errors.append(abs(acc - conf))
    return float(max(errors) if errors else 0.0)


def rankdata_average(a: np.ndarray) -> np.ndarray:
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sorted_a = a[order]
    i = 0
    while i < len(a):
        j = i + 1
        while j < len(a) and sorted_a[j] == sorted_a[i]:
            j += 1
        avg_rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def auc_roc_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true.sum()
    neg = len(y_true) - pos
    if pos == 0 or neg == 0:
        return np.nan
    ranks = rankdata_average(y_score)
    sum_ranks_pos = ranks[y_true == 1].sum()
    auc = (sum_ranks_pos - pos * (pos + 1) / 2.0) / (pos * neg)
    return float(auc)


def auc_pr_fast(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true.sum()
    if pos == 0:
        return np.nan
    order = np.argsort(-y_score, kind="mergesort")
    y_sorted = y_true[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / pos
    precision = np.r_[1.0, precision]
    recall = np.r_[0.0, recall]
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_binary_metrics(df: pd.DataFrame, score_col: str, target_col: str) -> dict:
    tmp = df[[score_col, target_col]].copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce")
    tmp[target_col] = normalize_binary_target(tmp[target_col])
    tmp = tmp.dropna()
    if tmp.empty:
        return {"n": 0, "pos_rate": np.nan, "auc_roc": np.nan, "auc_pr": np.nan, "brier": np.nan, "log_loss": np.nan, "ece": np.nan, "mce": np.nan}
    y = tmp[target_col].to_numpy(dtype=int)
    p = np.clip(tmp[score_col].to_numpy(dtype=float), 0, 1)
    return {
        "n": int(len(tmp)),
        "pos_rate": float(np.mean(y)),
        "auc_roc": auc_roc_fast(y, p),
        "auc_pr": auc_pr_fast(y, p),
        "brier": brier_score(y, p),
        "log_loss": log_loss_binary(y, p),
        "ece": ece_score(y, p),
        "mce": mce_score(y, p),
        "score_mean": float(np.mean(p)),
        "score_std": float(np.std(p)),
        "score_p50": float(np.percentile(p, 50)),
        "score_p90": float(np.percentile(p, 90)),
        "score_p95": float(np.percentile(p, 95)),
    }


def assign_session_label(df: pd.DataFrame) -> pd.Series:
    out = pd.Series("other", index=df.index, dtype=object)
    for name in DEFAULT_SESSION_FLAGS:
        if name in df.columns:
            mask = pd.to_numeric(df[name], errors="coerce").fillna(0).astype(int) == 1
            out.loc[mask] = name.replace("is_", "")
    return out


def bucket_probs(s: pd.Series, n_bins: int = 10) -> pd.Categorical:
    s = pd.to_numeric(s, errors="coerce").clip(0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    labels = [f"[{bins[i]:.2f},{bins[i+1]:.2f})" if i < n_bins - 1 else f"[{bins[i]:.2f},1.00]" for i in range(n_bins)]
    return pd.cut(s, bins=bins, labels=labels, include_lowest=True, right=False)


def bucket_quantiles(s: pd.Series, q: int = 10, labels_prefix: str = "Q") -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    ranks = s.rank(method="average", pct=True)
    raw = np.ceil(ranks * q).clip(1, q)
    return raw.map(lambda x: f"{labels_prefix}{int(x)}" if pd.notna(x) else np.nan)


def profit_factor_from_pnl(pnl: Iterable[float]) -> float:
    vals = pd.to_numeric(pd.Series(list(pnl)), errors="coerce").dropna()
    gross_profit = vals[vals > 0].sum()
    gross_loss = -vals[vals < 0].sum()
    if gross_loss == 0:
        return float(np.inf if gross_profit > 0 else np.nan)
    return float(gross_profit / gross_loss)


def max_drawdown_from_pnl(pnl: Iterable[float]) -> float:
    vals = pd.to_numeric(pd.Series(list(pnl)), errors="coerce").fillna(0.0)
    eq = vals.cumsum()
    peak = eq.cummax()
    dd = eq - peak
    return float(dd.min())


def summarize_pnl(df: pd.DataFrame, pnl_col: str, target_col: Optional[str] = None) -> dict:
    pnl = pd.to_numeric(df[pnl_col], errors="coerce").dropna()
    out = {
        "n_pnl": int(len(pnl)),
        "avg_pnl": float(pnl.mean()) if len(pnl) else np.nan,
        "median_pnl": float(pnl.median()) if len(pnl) else np.nan,
        "total_pnl": float(pnl.sum()) if len(pnl) else np.nan,
        "profit_factor": profit_factor_from_pnl(pnl),
        "max_drawdown": max_drawdown_from_pnl(pnl),
    }
    if len(pnl):
        out["win_rate_pnl"] = float((pnl > 0).mean())
    if target_col and target_col in df.columns:
        tgt = normalize_binary_target(df[target_col]).dropna()
        if len(tgt):
            out["hit_rate_target"] = float(tgt.mean())
    return out
