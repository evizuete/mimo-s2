# main_eval_top_trials.py
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Any, Dict, List

import pandas as pd

from main_optimize_rl_params import build_experiment
from reinforcement_learning.rl_experiment import RLRunConfig
from reinforcement_learning.rl_policy_evaluator import RLPolicyEvaluator, EvalObjective


def _dt(s: str) -> datetime:
    # Accept YYYY-MM-DD or full ISO
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s + "T00:00:00")


def _safe_path(p: str) -> str:
    p = (p or "").strip()
    if not p:
        return ""
    # Expand user & env vars
    p = os.path.expandvars(os.path.expanduser(p))
    return p


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval-only de top trials usando la policy guardada (sin re-entrenar)."
    )
    ap.add_argument("--summary_csv", required=True, help="CSV generado por summarize_trials.py (incluye policy_path).")
    ap.add_argument("--release", required=True, help="Release id (p.ej. 200288) para construir el experimento.")
    ap.add_argument("--outdir", required=True, help="Directorio de salida (p.ej. ./artifacts/rl_opt_200288_eval)")
    ap.add_argument("--eval_from", required=True, help="Inicio eval YYYY-MM-DD")
    ap.add_argument("--eval_to", required=True, help="Fin eval YYYY-MM-DD")
    ap.add_argument("--top", type=int, default=5, help="Cuántos trials evaluar.")
    ap.add_argument("--sort_by", default="score", help="Columna para ordenar (default: score).")
    ap.add_argument("--deterministic", action="store_true", help="Eval determinista (sin muestreo RL).")
    ap.add_argument("--export_trade_log", action="store_true", help="Exporta trade_log.csv por trial.")
    ap.add_argument("--debug", action="store_true", help="Activa debug del experimento (incluye dist de p(TAKE)).")
    ap.add_argument("--initial_equity", type=float, default=10_000.0)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    eval_from = _dt(args.eval_from)
    eval_to = _dt(args.eval_to)

    # 1) Load summary
    df = pd.read_csv(args.summary_csv)
    if args.sort_by not in df.columns:
        raise SystemExit(f"--sort_by='{args.sort_by}' no existe en CSV. Columnas: {list(df.columns)}")

    df_sorted = df.sort_values(args.sort_by, ascending=False).head(int(args.top)).reset_index(drop=True)

    # 2) Build experiment (same codepath as optimize)
    exp = build_experiment(release=args.release)
    print(f'RLConfig: {exp.rl_config}')

    # 3) Evaluator used for summarize + scoring
    evaluator = RLPolicyEvaluator(
        experiment=exp,
        train_from=eval_from,  # not used in eval-only, but required by ctor
        train_to=eval_to,
        eval_from=eval_from,
        eval_to=eval_to,
        outdir=args.outdir,
        objective=EvalObjective(),
    )

    results: List[Dict[str, Any]] = []

    for _, row in df_sorted.iterrows():
        tag = str(row.get("tag") or "").strip() or "trial"
        trial_dir = str(row.get("trial_dir") or "").strip()

        policy_path = _safe_path(str(row.get("policy_path") or ""))
        if policy_path and not os.path.exists(policy_path) and trial_dir:
            # Common case: policy_path stored as relative
            maybe = os.path.join(trial_dir, os.path.basename(policy_path))
            if os.path.exists(maybe):
                policy_path = maybe

        if not policy_path or not os.path.exists(policy_path):
            print(f"[eval] {tag} SKIP: policy_path not found: {policy_path}")
            continue

        run_eval = RLRunConfig(
            policy_path=policy_path,
            use_rl=True,
            rl_train=False,
            rl_eval_deterministic=bool(args.deterministic),
            rl_take_threshold=float(row.get("rl_take_threshold", 0.50)),
            initial_equity=float(args.initial_equity),
            sizing_equity_mode=str(row.get("sizing_equity_mode", "balance")),
            compound=bool(row.get("compound", True)),
            max_daily_loss_pct=float(row.get("max_daily_loss_pct", 0.03)) if pd.notna(row.get("max_daily_loss_pct")) else 0.03,
            max_daily_profit_pct=None,
        )

        # Eval-only backtest
        eval_res = exp.run_backtest(
            from_date=eval_from,
            to_date=eval_to,
            mode="eval",
            run=run_eval,
            debug=bool(args.debug),
        )

        metrics = evaluator._summarize(eval_res, run_eval)  # includes net_pnl, dd, pf, etc.
        score = evaluator.score(metrics, run_eval)

        payload: Dict[str, Any] = {
            "tag": tag,
            "trial_dir": trial_dir,
            "policy_path": policy_path,
            "run": run_eval.__dict__,
            "metrics": {**metrics, "score": float(score)},
        }

        # Optional trade log export
        if args.export_trade_log:
            try:
                trade_log_path = os.path.join(args.outdir, f"{tag}_trade_log.csv")
                exp.export_trade_log(
                    from_date=eval_from,
                    to_date=eval_to,
                    run=run_eval,
                    out_path=trade_log_path,
                )
                payload["trade_log_path"] = trade_log_path
            except TypeError:
                # fallback positional signature
                try:
                    exp.export_trade_log(eval_from, eval_to, run_eval, trade_log_path)
                    payload["trade_log_path"] = trade_log_path
                except Exception as e:
                    payload["trade_log_error"] = str(e)
            except Exception as e:
                payload["trade_log_error"] = str(e)

        results.append(payload)

        print(
            f"[eval] {tag} score={payload['metrics']['score']:.2f} "
            f"net={payload['metrics']['net_pnl']:.2f} "
            f"pf={payload['metrics']['profit_factor']:.3f} "
            f"mdd%={payload['metrics']['max_dd_pct']:.4f} "
            f"trades={payload['metrics']['n_trades']}"
        )

    # Save results
    out_json = os.path.join(args.outdir, "eval_top_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # Also write a flat CSV summary
    rows = []
    for r in results:
        m = r.get("metrics", {})
        rows.append({
            "tag": r.get("tag"),
            "score": m.get("score"),
            "net_pnl": m.get("net_pnl"),
            "profit_factor": m.get("profit_factor"),
            "win_rate": m.get("win_rate"),
            "max_dd_pct": m.get("max_dd_pct"),
            "n_trades": m.get("n_trades"),
            "rl_take_threshold": r.get("run", {}).get("rl_take_threshold"),
            "max_daily_loss_pct": r.get("run", {}).get("max_daily_loss_pct"),
            "compound": r.get("run", {}).get("compound"),
            "sizing_equity_mode": r.get("run", {}).get("sizing_equity_mode"),
            "policy_path": r.get("policy_path"),
            "trial_dir": r.get("trial_dir"),
            "trade_log_path": r.get("trade_log_path", ""),
        })
    out_csv = os.path.join(args.outdir, "eval_top_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
