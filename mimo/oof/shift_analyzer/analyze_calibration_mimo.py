#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .mimo_analysis_common import (
    assign_session_label,
    bucket_probs,
    compute_binary_metrics,
    ensure_period_column,
    load_dataframe,
    normalize_binary_target,
    resolve_columns,
    save_json,
)

def calibration_table(df: pd.DataFrame, score_col: str, target_col: str, n_bins: int) -> pd.DataFrame:
    tmp = df[[score_col, target_col]].copy()
    tmp[score_col] = pd.to_numeric(tmp[score_col], errors="coerce").clip(0, 1)
    tmp[target_col] = normalize_binary_target(tmp[target_col])
    tmp = tmp.dropna()
    if tmp.empty:
        return pd.DataFrame()
    tmp["prob_bucket"] = bucket_probs(tmp[score_col], n_bins=n_bins)
    res = tmp.groupby("prob_bucket", observed=False).agg(
        n=(target_col, "size"),
        mean_pred=(score_col, "mean"),
        empirical_rate=(target_col, "mean"),
        score_std=(score_col, "std"),
    ).reset_index()
    res["calibration_gap"] = res["empirical_rate"] - res["mean_pred"]
    return res


def grouped_metrics(df: pd.DataFrame, group_cols: list[str], score_col: str, target_col: str) -> pd.DataFrame:
    rows = []
    for keys, part in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        m = compute_binary_metrics(part, score_col, target_col)
        row = {c: k for c, k in zip(group_cols, keys)}
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Diagnóstico de calibración para MIMO")
    ap.add_argument("--input", required=True, help="CSV/parquet/pickle con score y target")
    ap.add_argument("--output-dir", default="./calibration_report")
    ap.add_argument("--score-col")
    ap.add_argument("--target-col")
    ap.add_argument("--state-col")
    ap.add_argument("--time-col")
    ap.add_argument("--period-col", help="Usa esta columna si ya existe train/holdout/validation")
    ap.add_argument("--train-end", help="Fecha fin de train, ej 2025-12-31")
    ap.add_argument("--holdout-start", help="Fecha inicio holdout, ej 2026-01-01")
    ap.add_argument("--bins", type=int, default=10)
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    cols = resolve_columns(df, score_col=args.score_col, target_col=args.target_col, state_col=args.state_col, time_col=args.time_col)
    df, period_col = ensure_period_column(df, args.period_col, cols.time, args.train_end, args.holdout_start)
    df["session"] = assign_session_label(df)

    overall = grouped_metrics(df[df[period_col] != "unknown"], [period_col], cols.score, cols.target)
    overall.to_csv(outdir / "metrics_by_period.csv", index=False)

    calib_rows = []
    for period, part in df.groupby(period_col):
        if period == "unknown":
            continue
        tab = calibration_table(part, cols.score, cols.target, args.bins)
        if tab.empty:
            continue
        tab.insert(0, period_col, period)
        calib_rows.append(tab)
    calib_df = pd.concat(calib_rows, ignore_index=True) if calib_rows else pd.DataFrame()
    calib_df.to_csv(outdir / "calibration_global.csv", index=False)

    by_state = pd.DataFrame()
    if cols.state:
        by_state = grouped_metrics(df[df[period_col] != "unknown"], [period_col, cols.state], cols.score, cols.target)
        by_state.to_csv(outdir / "metrics_by_period_state.csv", index=False)

        rows = []
        for keys, part in df.groupby([period_col, cols.state], dropna=False):
            tab = calibration_table(part, cols.score, cols.target, args.bins)
            if tab.empty:
                continue
            period, state = keys
            tab.insert(0, cols.state, state)
            tab.insert(0, period_col, period)
            rows.append(tab)
        pd.concat(rows, ignore_index=True).to_csv(outdir / "calibration_by_state.csv", index=False)

    by_session = grouped_metrics(df[df[period_col] != "unknown"], [period_col, "session"], cols.score, cols.target)
    by_session.to_csv(outdir / "metrics_by_period_session.csv", index=False)

    rows = []
    for keys, part in df.groupby([period_col, "session"], dropna=False):
        tab = calibration_table(part, cols.score, cols.target, args.bins)
        if tab.empty:
            continue
        period, session = keys
        tab.insert(0, "session", session)
        tab.insert(0, period_col, period)
        rows.append(tab)
    pd.concat(rows, ignore_index=True).to_csv(outdir / "calibration_by_session.csv", index=False)

    summary = {
        "input": str(args.input),
        "resolved_columns": cols.__dict__,
        "period_col": period_col,
        "rows_total": int(len(df)),
        "rows_known_period": int((df[period_col] != "unknown").sum()),
        "overall_periods": overall.to_dict(orient="records"),
    }
    save_json(summary, outdir / "summary.json")
    print(f"[OK] Reporte guardado en {outdir.resolve()}")
    print(f"[OK] metrics_by_period.csv")
    print(f"[OK] calibration_global.csv")
    if cols.state:
        print(f"[OK] metrics_by_period_state.csv")
        print(f"[OK] calibration_by_state.csv")
    print(f"[OK] metrics_by_period_session.csv")
    print(f"[OK] calibration_by_session.csv")


if __name__ == "__main__":
    main()
