# scripts/rl_eval_and_sweep.py
#
# Example runner:
#   1) runs RL eval backtest (loads policy)
#   2) exports candidate file for the eval window
#   3) merges candidates with realized trade pnl (you must provide a trade log)
#   4) sweeps thresholds by state and saves json/csv
#
# NOTE: Step (3) depends on how you log trades in TradingSimulator.
#       If you already have a trade log with time/state/action/rl_score/pnl, you can skip (2)+(3)
#       and pass it directly to sweep_thresholds_by_state.

import argparse
import json
import pandas as pd

from rl_thresholds import sweep_thresholds_by_state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trades_path", required=True, help="CSV/Parquet with columns: time,state,action,rl_score,pnl")
    ap.add_argument("--out_json", default="rl_threshold_by_state.json")
    ap.add_argument("--out_csv", default="rl_threshold_by_state_summary.csv")
    ap.add_argument("--grid_n", type=int, default=25)
    ap.add_argument("--grid_method", choices=["quantile", "linear"], default="quantile")
    ap.add_argument("--min_trades", type=int, default=30)
    ap.add_argument("--dd_weight", type=float, default=0.25)
    ap.add_argument("--actions", default="long,short", help="comma-separated actions or 'all'")
    args = ap.parse_args()

    if args.trades_path.lower().endswith(".parquet"):
        df = pd.read_parquet(args.trades_path)
    else:
        df = pd.read_csv(args.trades_path)

    required = {"time", "state", "action", "rl_score", "pnl"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}. Got: {list(df.columns)}")

    allowed_actions = None
    if args.actions.strip().lower() != "all":
        allowed_actions = [x.strip() for x in args.actions.split(",") if x.strip()]

    thr_by_state, summary, global_m = sweep_thresholds_by_state(
        df,
        grid_n=args.grid_n,
        grid_method=args.grid_method,
        min_trades=args.min_trades,
        dd_weight=args.dd_weight,
        allowed_actions=allowed_actions,
    )

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(thr_by_state, f, indent=2, ensure_ascii=False)
    summary.to_csv(args.out_csv, index=False)

    print("== Global metrics with per-state thresholds ==")
    print(global_m)
    print("\nSaved:", args.out_json, args.out_csv)


if __name__ == "__main__":
    main()
