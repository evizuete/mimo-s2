#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mimo_analysis_common import (
    ensure_period_column,
    load_dataframe,
    normalize_binary_target,
    profit_factor_from_pnl,
    resolve_columns,
    save_json,
)


def percentile_summary(df: pd.DataFrame, score_col: str, group_cols: list[str], percentiles: list[int]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        score = pd.to_numeric(part[score_col], errors="coerce").dropna()
        if score.empty:
            continue
        row = {c: k for c, k in zip(group_cols, keys)}
        row["n"] = int(len(score))
        row["mean_score"] = float(score.mean())
        for p in percentiles:
            row[f"p{p}"] = float(np.percentile(score, p))
        rows.append(row)
    return pd.DataFrame(rows)


def bucket_performance(df: pd.DataFrame, score_col: str, target_col: str, pnl_col: str | None, group_cols: list[str], cuts: list[float]) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        tmp = part.copy()
        tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").clip(0, 1)
        tmp[target_col] = normalize_binary_target(tmp[target_col])
        tmp = tmp.dropna(subset=[score_col, target_col])
        if tmp.empty:
            continue
        pct_rank = tmp[score_col].rank(method="average", pct=True) * 100.0
        labels = [f"{int(cuts[i])}-{int(cuts[i+1])}" for i in range(len(cuts)-1)]
        tmp["percentile_bucket"] = pd.cut(pct_rank, bins=cuts, labels=labels, include_lowest=True, right=True)
        for bucket, bucket_df in tmp.groupby("percentile_bucket", observed=False):
            if bucket_df.empty:
                continue
            row = {c: k for c, k in zip(group_cols, keys)}
            row["percentile_bucket"] = str(bucket)
            row["n"] = int(len(bucket_df))
            row["mean_score"] = float(bucket_df[score_col].mean())
            row["hit_rate"] = float(bucket_df[target_col].mean())
            if pnl_col and pnl_col in bucket_df.columns:
                pnl = pd.to_numeric(bucket_df[pnl_col], errors="coerce").dropna()
                row["avg_pnl"] = float(pnl.mean()) if len(pnl) else np.nan
                row["total_pnl"] = float(pnl.sum()) if len(pnl) else np.nan
                row["profit_factor"] = profit_factor_from_pnl(pnl) if len(pnl) else np.nan
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Estabilidad de percentiles por contexto para MIMO")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="./percentile_stability_report")
    ap.add_argument("--score-col")
    ap.add_argument("--target-col")
    ap.add_argument("--state-col")
    ap.add_argument("--time-col")
    ap.add_argument("--period-col")
    ap.add_argument("--train-end")
    ap.add_argument("--holdout-start")
    ap.add_argument("--pnl-col")
    ap.add_argument("--percentiles", default="50,75,80,85,90,95")
    ap.add_argument("--cuts", default="0,50,75,85,90,95,100")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    cols = resolve_columns(
        df, score_col=args.score_col, target_col=args.target_col,
        state_col=args.state_col, time_col=args.time_col, pnl_col=args.pnl_col,
    )
    df, period_col = ensure_period_column(df, args.period_col, cols.time, args.train_end, args.holdout_start)
    df = df[df[period_col] != "unknown"].copy()

    percentiles = [int(x) for x in args.percentiles.split(",") if x.strip()]
    cuts = [float(x) for x in args.cuts.split(",") if x.strip()]

    by_period = percentile_summary(df, cols.score, [period_col], percentiles)
    by_period.to_csv(outdir / "score_percentiles_by_period.csv", index=False)

    if cols.state:
        by_state = percentile_summary(df, cols.score, [period_col, cols.state], percentiles)
        by_state.to_csv(outdir / "score_percentiles_by_period_state.csv", index=False)
        bucket_state = bucket_performance(df, cols.score, cols.target, cols.pnl, [period_col, cols.state], cuts)
        bucket_state.to_csv(outdir / "bucket_performance_by_period_state.csv", index=False)
    else:
        by_state = pd.DataFrame()
        bucket_state = pd.DataFrame()

    bucket_period = bucket_performance(df, cols.score, cols.target, cols.pnl, [period_col], cuts)
    bucket_period.to_csv(outdir / "bucket_performance_by_period.csv", index=False)

    summary = {
        "input": str(args.input),
        "resolved_columns": cols.__dict__,
        "period_col": period_col,
        "percentiles": percentiles,
        "cuts": cuts,
        "rows_analyzed": int(len(df)),
    }
    save_json(summary, outdir / "summary.json")
    print(f"[OK] Reporte guardado en {outdir.resolve()}")


if __name__ == "__main__":
    main()
