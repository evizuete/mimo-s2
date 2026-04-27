#!/usr/bin/env python3
"""
Recalibración automática de thresholds / percentiles del DecisionEngine
comparando TRAIN vs HOLDOUT.

Objetivo:
- medir cómo cambia la calidad real de cada bucket de score
- proponer thresholds nuevos para cada estado
- exportar tablas fáciles de revisar

Input esperado:
  parquet/csv con al menos:
    time, state, score, target

Ejemplo:
python analyze_threshold_recalibration.py \
  --input calibration_dataset_200371_long.parquet \
  --score-col oof_proba_cal \
  --target-col signal \
  --state-col state \
  --time-col time \
  --train-end 2025-12-31 \
  --holdout-start 2026-01-01 \
  --quantiles 80 85 90 95 97 98 99
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


EPS = 1e-12


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"No existe: {path}")
    if p.suffix.lower() == ".parquet":
        return pd.read_parquet(p)
    if p.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(p)
    if p.suffix.lower() in {".pkl", ".pickle"}:
        return pd.read_pickle(p)
    raise ValueError(f"Formato no soportado: {p.suffix}")


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def profit_factor(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    pred = (y_score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return tp / max(fp, 1)


def precision_at_threshold(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    pred = (y_score >= thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    return tp / max(tp + fp, 1)


def signal_rate_at_threshold(y_score: np.ndarray, thr: float) -> float:
    return float((y_score >= thr).mean())


def percentile_thresholds(scores: pd.Series, quantiles: Iterable[int]) -> dict[str, float]:
    values = np.asarray(scores.dropna(), dtype=float)
    if len(values) == 0:
        return {f"p{q}": float("nan") for q in quantiles}
    return {f"p{q}": float(np.percentile(values, q)) for q in quantiles}


def bucket_edges_from_train(train_scores: pd.Series, quantiles: Iterable[int]) -> np.ndarray:
    q = [0, *sorted(set(int(x) for x in quantiles)), 100]
    edges = np.unique(np.percentile(np.asarray(train_scores.dropna(), dtype=float), q))
    if len(edges) < 2:
        raise ValueError("No se pudieron construir buckets: score demasiado degenerado")
    return edges


def performance_by_train_buckets(
    train_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    quantiles: Iterable[int],
) -> pd.DataFrame:
    train_scores = train_df[score_col].astype(float)
    edges = bucket_edges_from_train(train_scores, quantiles)

    rows: list[dict] = []
    for period_name, df in (("train", train_df), ("holdout", hold_df)):
        scores = df[score_col].astype(float)
        targets = df[target_col].astype(float)
        bins = pd.cut(scores, bins=edges, include_lowest=True, duplicates="drop")
        tmp = pd.DataFrame({"bucket": bins, "score": scores, "target": targets}).dropna()
        grouped = tmp.groupby("bucket", observed=False)
        for bucket, g in grouped:
            if len(g) == 0:
                continue
            rows.append(
                {
                    "period": period_name,
                    "bucket": str(bucket),
                    "n": int(len(g)),
                    "mean_score": safe_float(g["score"].mean()),
                    "empirical_rate": safe_float(g["target"].mean()),
                    "calibration_gap": safe_float(g["score"].mean() - g["target"].mean()),
                }
            )
    return pd.DataFrame(rows)


def recommend_threshold_for_state(
    train_df: pd.DataFrame,
    hold_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    quantiles: Iterable[int],
    min_precision: float,
    max_signal_rate: float,
) -> dict:
    train_scores = train_df[score_col].astype(float).dropna()
    hold_scores = hold_df[score_col].astype(float).dropna()
    y_train = train_df[target_col].astype(int).to_numpy()
    y_hold = hold_df[target_col].astype(int).to_numpy()
    s_train = train_df[score_col].astype(float).to_numpy()
    s_hold = hold_df[score_col].astype(float).to_numpy()

    train_thrs = percentile_thresholds(train_scores, quantiles)
    hold_thrs = percentile_thresholds(hold_scores, quantiles)

    candidates = []
    for q in quantiles:
        name = f"p{q}"
        thr_train = float(train_thrs[name])
        thr_hold = float(hold_thrs[name])

        candidates.append(
            {
                "threshold_source": "train_percentile_value",
                "quantile": int(q),
                "threshold": thr_train,
                "train_precision": precision_at_threshold(y_train, s_train, thr_train),
                "holdout_precision": precision_at_threshold(y_hold, s_hold, thr_train),
                "train_signal_rate": signal_rate_at_threshold(s_train, thr_train),
                "holdout_signal_rate": signal_rate_at_threshold(s_hold, thr_train),
                "holdout_profit_factor": profit_factor(y_hold, s_hold, thr_train),
            }
        )

        candidates.append(
            {
                "threshold_source": "holdout_percentile_value",
                "quantile": int(q),
                "threshold": thr_hold,
                "train_precision": precision_at_threshold(y_train, s_train, thr_hold),
                "holdout_precision": precision_at_threshold(y_hold, s_hold, thr_hold),
                "train_signal_rate": signal_rate_at_threshold(s_train, thr_hold),
                "holdout_signal_rate": signal_rate_at_threshold(s_hold, thr_hold),
                "holdout_profit_factor": profit_factor(y_hold, s_hold, thr_hold),
            }
        )

    cand_df = pd.DataFrame(candidates)

    feasible = cand_df[
        (cand_df["holdout_precision"] >= min_precision)
        & (cand_df["holdout_signal_rate"] <= max_signal_rate)
    ].copy()

    if feasible.empty:
        feasible = cand_df.sort_values(
            ["holdout_precision", "holdout_profit_factor", "holdout_signal_rate"],
            ascending=[False, False, True],
        ).head(1)
        reason = "fallback_best_available"
    else:
        feasible = feasible.sort_values(
            ["holdout_profit_factor", "holdout_precision", "holdout_signal_rate"],
            ascending=[False, False, True],
        ).head(1)
        reason = "meets_constraints"

    rec = feasible.iloc[0].to_dict()
    rec["selection_reason"] = reason

    baseline_q = max(quantiles)
    baseline = cand_df[
        (cand_df["quantile"] == baseline_q)
        & (cand_df["threshold_source"] == "train_percentile_value")
    ].head(1)
    baseline_rec = baseline.iloc[0].to_dict() if not baseline.empty else {}

    return {
        "train_thresholds": train_thrs,
        "holdout_thresholds": hold_thrs,
        "recommended": rec,
        "baseline": baseline_rec,
        "candidates": cand_df,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="DecisionEngine threshold recalibration")
    ap.add_argument("--input", required=True)
    ap.add_argument("--score-col", default="oof_proba_cal")
    ap.add_argument("--target-col", default="signal")
    ap.add_argument("--state-col", default="state")
    ap.add_argument("--time-col", default="time")
    ap.add_argument("--train-end", required=True)
    ap.add_argument("--holdout-start", required=True)
    ap.add_argument("--output-dir", default="./threshold_recalibration_report")
    ap.add_argument("--quantiles", nargs="+", type=int, default=[80, 85, 90, 95, 97, 98, 99])
    ap.add_argument("--min-state-n", type=int, default=500)
    ap.add_argument("--min-precision", type=float, default=0.45)
    ap.add_argument("--max-signal-rate", type=float, default=0.15)
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = read_table(args.input).copy()
    df[args.time_col] = pd.to_datetime(df[args.time_col], errors="coerce")

    cols_needed = [args.time_col, args.score_col, args.target_col, args.state_col]
    missing = [c for c in cols_needed if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas: {missing}")

    df = df.dropna(subset=[args.time_col, args.score_col, args.target_col, args.state_col]).copy()
    df[args.target_col] = pd.to_numeric(df[args.target_col], errors="coerce")
    df[args.score_col] = pd.to_numeric(df[args.score_col], errors="coerce")
    df = df.dropna(subset=[args.score_col, args.target_col]).copy()

    train_end = pd.Timestamp(args.train_end)
    holdout_start = pd.Timestamp(args.holdout_start)

    train_df = df[df[args.time_col] <= train_end].copy()
    hold_df = df[df[args.time_col] >= holdout_start].copy()

    if train_df.empty or hold_df.empty:
        raise ValueError("Train u holdout vacíos tras filtrar por fechas")

    summary_rows = []
    recommendations = {}

    states = sorted(set(train_df[args.state_col].unique()).intersection(set(hold_df[args.state_col].unique())))

    for state in states:
        train_s = train_df[train_df[args.state_col] == state].copy()
        hold_s = hold_df[hold_df[args.state_col] == state].copy()

        if len(train_s) < args.min_state_n or len(hold_s) < args.min_state_n:
            continue

        perf_buckets = performance_by_train_buckets(
            train_s,
            hold_s,
            args.score_col,
            args.target_col,
            args.quantiles,
        )
        perf_buckets.to_csv(out_dir / f"bucket_performance_{state}.csv", index=False)

        rec = recommend_threshold_for_state(
            train_s,
            hold_s,
            args.score_col,
            args.target_col,
            args.quantiles,
            min_precision=args.min_precision,
            max_signal_rate=args.max_signal_rate,
        )

        rec["candidates"].to_csv(out_dir / f"threshold_candidates_{state}.csv", index=False)

        recommended = rec["recommended"]
        baseline = rec["baseline"]

        summary_rows.append(
            {
                "state": state,
                "train_n": int(len(train_s)),
                "holdout_n": int(len(hold_s)),
                "train_pos_rate": safe_float(train_s[args.target_col].mean()),
                "holdout_pos_rate": safe_float(hold_s[args.target_col].mean()),
                "recommended_quantile": int(recommended["quantile"]),
                "recommended_threshold_source": str(recommended["threshold_source"]),
                "recommended_threshold": safe_float(recommended["threshold"]),
                "recommended_holdout_precision": safe_float(recommended["holdout_precision"]),
                "recommended_holdout_signal_rate": safe_float(recommended["holdout_signal_rate"]),
                "recommended_holdout_profit_factor": safe_float(recommended["holdout_profit_factor"]),
                "baseline_quantile": int(baseline.get("quantile", np.nan)) if baseline else np.nan,
                "baseline_threshold": safe_float(baseline.get("threshold", np.nan)) if baseline else np.nan,
                "baseline_holdout_precision": safe_float(baseline.get("holdout_precision", np.nan)) if baseline else np.nan,
                "baseline_holdout_signal_rate": safe_float(baseline.get("holdout_signal_rate", np.nan)) if baseline else np.nan,
                "selection_reason": str(recommended["selection_reason"]),
            }
        )

        recommendations[state] = {
            "train_thresholds": rec["train_thresholds"],
            "holdout_thresholds": rec["holdout_thresholds"],
            "recommended": {
                k: (safe_float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
                for k, v in recommended.items()
            },
            "baseline": {
                k: (safe_float(v) if isinstance(v, (float, int, np.floating, np.integer)) else v)
                for k, v in baseline.items()
            },
        }

    summary_df = pd.DataFrame(summary_rows).sort_values("state") if summary_rows else pd.DataFrame()
    summary_csv = out_dir / "threshold_recalibration_summary.csv"
    summary_df.to_csv(summary_csv, index=False)

    json_path = out_dir / "threshold_recommendations.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(recommendations, f, indent=2, ensure_ascii=False)

    print("\nTHRESHOLD RECALIBRATION REPORT")
    print("-" * 40)
    print(f"Input        : {args.input}")
    print(f"Train rows   : {len(train_df):,}")
    print(f"Holdout rows : {len(hold_df):,}")
    print(f"Estados      : {len(summary_df):,}")
    print(f"CSV resumen  : {summary_csv}")
    print(f"JSON recs    : {json_path}")

    if not summary_df.empty:
        cols = [
            "state",
            "recommended_quantile",
            "recommended_threshold",
            "recommended_holdout_precision",
            "recommended_holdout_signal_rate",
            "baseline_threshold",
            "baseline_holdout_precision",
        ]
        print("\nTop resumen:")
        print(summary_df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
