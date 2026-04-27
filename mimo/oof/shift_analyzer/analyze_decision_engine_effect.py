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
    normalize_decision,
    resolve_columns,
    save_json,
    summarize_pnl,
)


def decision_summary(df: pd.DataFrame, decision_col: str, target_col: str, pnl_col: str | None, group_cols: list[str]) -> pd.DataFrame:
    rows = []
    tmp = df.copy()
    tmp[decision_col] = normalize_decision(tmp[decision_col])
    tmp[target_col] = normalize_binary_target(tmp[target_col])
    tmp = tmp.dropna(subset=[decision_col, target_col])
    for keys, part in tmp.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = {c: k for c, k in zip(group_cols, keys)}
        row["n_candidates"] = int(len(part))
        row["n_accepted"] = int((part[decision_col] == 1).sum())
        row["n_rejected"] = int((part[decision_col] == 0).sum())
        row["accept_rate"] = float((part[decision_col] == 1).mean())
        row["hit_rate_all"] = float(part[target_col].mean())
        acc = part[part[decision_col] == 1]
        rej = part[part[decision_col] == 0]
        row["hit_rate_accept"] = float(acc[target_col].mean()) if len(acc) else np.nan
        row["hit_rate_reject"] = float(rej[target_col].mean()) if len(rej) else np.nan
        if pnl_col and pnl_col in part.columns:
            row.update({f"all_{k}": v for k, v in summarize_pnl(part, pnl_col).items()})
            if len(acc):
                row.update({f"accept_{k}": v for k, v in summarize_pnl(acc, pnl_col).items()})
            if len(rej):
                row.update({f"reject_{k}": v for k, v in summarize_pnl(rej, pnl_col).items()})
        rows.append(row)
    return pd.DataFrame(rows)


def reason_summary(df: pd.DataFrame, decision_col: str, reason_col: str, group_cols: list[str]) -> pd.DataFrame:
    tmp = df.copy()
    tmp[decision_col] = normalize_decision(tmp[decision_col])
    tmp[reason_col] = tmp[reason_col].astype(str).fillna("<NA>")
    tmp = tmp[tmp[decision_col] == 0]
    if tmp.empty:
        return pd.DataFrame()
    res = tmp.groupby(group_cols + [reason_col], dropna=False).size().reset_index(name="n")
    total = res.groupby(group_cols, dropna=False)["n"].transform("sum")
    res["pct_within_group"] = res["n"] / total
    return res.sort_values(group_cols + ["n"], ascending=[True] * len(group_cols) + [False])


def main() -> None:
    ap = argparse.ArgumentParser(description="Auditoría del efecto del DecisionEngine")
    ap.add_argument("--input", required=True)
    ap.add_argument("--output-dir", default="./decision_engine_report")
    ap.add_argument("--score-col")
    ap.add_argument("--target-col")
    ap.add_argument("--decision-col")
    ap.add_argument("--reason-col")
    ap.add_argument("--state-col")
    ap.add_argument("--time-col")
    ap.add_argument("--period-col")
    ap.add_argument("--train-end")
    ap.add_argument("--holdout-start")
    ap.add_argument("--pnl-col")
    args = ap.parse_args()

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = load_dataframe(args.input)
    cols = resolve_columns(
        df, score_col=args.score_col, target_col=args.target_col,
        state_col=args.state_col, time_col=args.time_col,
        pnl_col=args.pnl_col, decision_col=args.decision_col, reason_col=args.reason_col,
    )
    if cols.decision is None:
        raise ValueError("No se pudo resolver la columna de decisión. Usa --decision-col.")
    df, period_col = ensure_period_column(df, args.period_col, cols.time, args.train_end, args.holdout_start)
    df = df[df[period_col] != "unknown"].copy()

    by_period = decision_summary(df, cols.decision, cols.target, cols.pnl, [period_col])
    by_period.to_csv(outdir / "decision_effect_by_period.csv", index=False)

    if cols.state:
        by_state = decision_summary(df, cols.decision, cols.target, cols.pnl, [period_col, cols.state])
        by_state.to_csv(outdir / "decision_effect_by_period_state.csv", index=False)
    else:
        by_state = pd.DataFrame()

    if cols.reason:
        reason_df = reason_summary(df, cols.decision, cols.reason, [period_col] + ([cols.state] if cols.state else []))
        reason_df.to_csv(outdir / "rejection_reasons.csv", index=False)
    else:
        reason_df = pd.DataFrame()

    summary = {
        "input": str(args.input),
        "resolved_columns": cols.__dict__,
        "period_col": period_col,
        "rows_analyzed": int(len(df)),
    }
    save_json(summary, outdir / "summary.json")
    print(f"[OK] Reporte guardado en {outdir.resolve()}")


if __name__ == "__main__":
    main()
