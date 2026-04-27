#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mimo_analysis_common import (
    bucket_quantiles,
    compute_binary_metrics,
    ensure_period_column,
    load_dataframe,
    max_drawdown_from_pnl,
    normalize_binary_target,
    profit_factor_from_pnl,
    resolve_columns,
    save_json,
)


def score_bucket_report(df: pd.DataFrame, score_col: str, target_col: str, pnl_col: str, group_cols: list[str], n_buckets: int) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").clip(0, 1)
    tmp[target_col] = normalize_binary_target(tmp[target_col])
    tmp[pnl_col] = pd.to_numeric(tmp[pnl_col], errors="coerce")
    tmp = tmp.dropna(subset=[score_col, target_col, pnl_col])
    for keys, part in tmp.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        part = part.copy()
        part["score_bucket"] = bucket_quantiles(part[score_col], q=n_buckets, labels_prefix="D")
        for bucket, bucket_df in part.groupby("score_bucket", observed=False):
            if bucket_df.empty:
                continue
            row = {c: k for c, k in zip(group_cols, keys)}
            pnl = bucket_df[pnl_col]
            row.update({
                "score_bucket": str(bucket),
                "n": int(len(bucket_df)),
                "mean_score": float(bucket_df[score_col].mean()),
                "hit_rate": float(bucket_df[target_col].mean()),
                "avg_pnl": float(pnl.mean()),
                "median_pnl": float(pnl.median()),
                "total_pnl": float(pnl.sum()),
                "profit_factor": profit_factor_from_pnl(pnl),
                "max_drawdown": max_drawdown_from_pnl(pnl),
                "win_rate_pnl": float((pnl > 0).mean()),
            })
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Relación score -> PnL para MIMO")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="./score_to_pnl_report")
    ap.add_argument("--score-col")
    ap.add_argument("--target-col")
    ap.add_argument("--pnl-col")
    ap.add_argument("--state-col")
    ap.add_argument("--time-col")
    ap.add_argument("--period-col")
    ap.add_argument("--train-end")
    ap.add_argument("--holdout-start")
    ap.add_argument("--buckets", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    cols = resolve_columns(
        df, score_col=args.score_col, target_col=args.target_col,
        state_col=args.state_col, time_col=args.time_col, pnl_col=args.pnl_col,
    )
    if cols.pnl is None:
        raise ValueError("No se pudo resolver la columna de PnL. Usa --pnl-col.")
    df, period_col = ensure_period_column(df, args.period_col, cols.time, args.train_end, args.holdout_start)
    df = df[df[period_col] != "unknown"].copy()

    by_period = score_bucket_report(df, cols.score, cols.target, cols.pnl, [period_col], args.buckets)
    by_period.to_csv(outdir / "score_to_pnl_by_period.csv", index=False)

    if cols.state:
        by_state = score_bucket_report(df, cols.score, cols.target, cols.pnl, [period_col, cols.state], args.buckets)
        by_state.to_csv(outdir / "score_to_pnl_by_period_state.csv", index=False)
    else:
        by_state = pd.DataFrame()

    # global metrics per period for context
    rows = []
    for period, part in df.groupby(period_col):
        m = compute_binary_metrics(part, cols.score, cols.target)
        m["period"] = period
        rows.append(m)
    pd.DataFrame(rows).to_csv(outdir / "classification_metrics_by_period.csv", index=False)

    summary = {
        "input": str(args.input),
        "resolved_columns": cols.__dict__,
        "period_col": period_col,
        "buckets": args.buckets,
        "rows_analyzed": int(len(df)),
    }
    save_json(summary, outdir / "summary.json")
    print(f"[OK] Reporte guardado en {outdir.resolve()}")


if __name__ == "__main__":
    main()
