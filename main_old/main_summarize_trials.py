# summarize_trials.py
from __future__ import annotations

import argparse
import glob
import json
import os
from typing import Any, Dict, List

import pandas as pd


def _safe_get(d: Dict[str, Any], path: str, default=None):
    cur = d
    for p in path.split("."):
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--opt_dir", required=True, help="Directorio raíz con trials (p.ej. ./artifacts/rl_opt_200285)")
    ap.add_argument("--out_csv", default=None, help="CSV de salida (por defecto: <opt_dir>/trials_summary.csv)")
    ap.add_argument("--sort_by", default="metrics.score", help="Campo para ordenar (p.ej. metrics.score o metrics.net_pnl)")
    ap.add_argument("--top", type=int, default=20, help="Cuántos mostrar/guardar como top")
    args = ap.parse_args()

    opt_dir = args.opt_dir
    out_csv = args.out_csv or os.path.join(opt_dir, "trials_summary.csv")

    files = glob.glob(os.path.join(opt_dir, "**", "metrics.json"), recursive=True)
    if not files:
        raise SystemExit(f"No se encontraron metrics.json en {opt_dir}")

    rows: List[Dict[str, Any]] = []
    for fp in files:
        try:
            with open(fp, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        row = {
            "trial_dir": os.path.dirname(fp),
            "tag": data.get("tag"),
            "policy_path": data.get("policy_path"),
            "score": _safe_get(data, "metrics.score"),
            "net_pnl": _safe_get(data, "metrics.net_pnl"),
            "profit_factor": _safe_get(data, "metrics.profit_factor"),
            "win_rate": _safe_get(data, "metrics.win_rate"),
            "max_dd_pct": _safe_get(data, "metrics.max_dd_pct"),
            "n_trades": _safe_get(data, "metrics.n_trades"),
            "sharpe_daily_ann": _safe_get(data, "metrics.sharpe_daily_ann"),
            "sortino_daily_ann": _safe_get(data, "metrics.sortino_daily_ann"),
            "max_daily_loss_pct": _safe_get(data, "metrics.max_daily_loss_pct"),
            "rl_take_threshold": _safe_get(data, "metrics.rl_take_threshold"),
            "compound": _safe_get(data, "metrics.compound"),
            "sizing_equity_mode": _safe_get(data, "metrics.sizing_equity_mode"),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ordenación
    sort_key = args.sort_by.replace("metrics.", "")
    if sort_key not in df.columns:
        # fallback a score
        sort_key = "score"
    df = df.sort_values(sort_key, ascending=False, na_position="last")

    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    df.to_csv(out_csv, index=False)

    topn = df.head(args.top)

    print(f"\nSaved: {out_csv}")
    print(f"\nTop {args.top} by {sort_key}:\n")
    print(topn[[
        "tag", "score", "net_pnl", "profit_factor", "win_rate", "max_dd_pct", "n_trades",
        "max_daily_loss_pct", "rl_take_threshold", "compound", "sizing_equity_mode",
        "policy_path", "trial_dir"
    ]].to_string(index=False))


if __name__ == "__main__":
    main()
