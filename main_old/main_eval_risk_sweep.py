# main_eval_risk_sweep.py
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
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        return datetime.fromisoformat(s + "T00:00:00")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Eval-only de una policy fija barriendo max_daily_loss_pct (stress test de riesgo diario)."
    )
    ap.add_argument("--release", required=True, help="Release id (p.ej. 200288)")
    ap.add_argument("--policy_path", required=True, help="Ruta a rl_policy.npz (policy ya entrenada)")
    ap.add_argument("--outdir", required=True, help="Directorio salida (p.ej. ./artifacts/rl_risk_sweep_200288)")
    ap.add_argument("--eval_from", required=True, help="YYYY-MM-DD (inicio)")
    ap.add_argument("--eval_to", required=True, help="YYYY-MM-DD (fin)")
    ap.add_argument("--tag", default="policy", help="Etiqueta para los outputs")
    ap.add_argument("--threshold", type=float, required=True, help="rl_take_threshold a usar en eval")
    ap.add_argument(
        "--daily_losses",
        default="0.03,0.04,0.05",
        help="Lista separada por comas, p.ej. 0.03,0.04,0.05,0.06",
    )
    ap.add_argument("--deterministic", action="store_true", help="Eval determinista")
    ap.add_argument("--export_trade_log", action="store_true", help="Exportar trade log por cada daily_loss")
    ap.add_argument("--debug", action="store_true", help="Debug (incluye dist de p(TAKE))")
    ap.add_argument("--initial_equity", type=float, default=10_000.0)
    ap.add_argument("--compound", action="store_true", help="Compound True (si no lo pones, False)")
    ap.add_argument("--sizing_equity_mode", default="balance", choices=["fixed", "balance", "mtm"])
    args = ap.parse_args()

    policy_path = os.path.expanduser(os.path.expandvars(args.policy_path))
    if not os.path.exists(policy_path):
        raise SystemExit(f"policy_path no existe: {policy_path}")

    os.makedirs(args.outdir, exist_ok=True)

    eval_from = _dt(args.eval_from)
    eval_to = _dt(args.eval_to)

    losses = []
    for x in str(args.daily_losses).split(","):
        x = x.strip()
        if not x:
            continue
        losses.append(float(x))
    if not losses:
        raise SystemExit("daily_losses vacío")

    exp = build_experiment(release=args.release)

    evaluator = RLPolicyEvaluator(
        experiment=exp,
        train_from=eval_from,  # not used, but required
        train_to=eval_to,
        eval_from=eval_from,
        eval_to=eval_to,
        outdir=args.outdir,
        objective=EvalObjective(),
    )

    results: List[Dict[str, Any]] = []

    for dl in losses:
        run_eval = RLRunConfig(
            policy_path=policy_path,
            use_rl=True,
            rl_train=False,
            rl_eval_deterministic=bool(args.deterministic),
            rl_take_threshold=float(args.threshold),
            initial_equity=float(args.initial_equity),
            sizing_equity_mode=str(args.sizing_equity_mode),
            compound=bool(args.compound),
            max_daily_loss_pct=float(dl),
            max_daily_profit_pct=None,
        )

        res_eval = exp.run_backtest(
            from_date=eval_from,
            to_date=eval_to,
            mode="eval",
            run=run_eval,
            debug=bool(args.debug),
        )

        metrics = evaluator._summarize(res_eval, run_eval)
        score = evaluator.score(metrics, run_eval)

        tag = f"{args.tag}_dl_{dl:.4f}".replace(".", "_")
        payload: Dict[str, Any] = {
            "tag": tag,
            "policy_path": policy_path,
            "run": run_eval.__dict__,
            "metrics": {**metrics, "score": float(score)},
        }

        if args.export_trade_log:
            try:
                trade_log_path = os.path.join(args.outdir, f"{tag}_trade_log.csv")
                exp.export_trade_log(eval_from, eval_to, run_eval, trade_log_path)
                payload["trade_log_path"] = trade_log_path
            except Exception as e:
                payload["trade_log_error"] = str(e)

        results.append(payload)

        print(
            f"[risk] dl={dl:.4f} "
            f"score={payload['metrics']['score']:.2f} "
            f"net={payload['metrics']['net_pnl']:.2f} "
            f"pf={payload['metrics']['profit_factor']:.3f} "
            f"mdd%={payload['metrics']['max_dd_pct']:.4f} "
            f"trades={payload['metrics']['n_trades']}"
        )

    # Save outputs
    out_json = os.path.join(args.outdir, "risk_sweep_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    rows = []
    for r in results:
        m = r.get("metrics", {})
        run = r.get("run", {})
        rows.append({
            "tag": r.get("tag"),
            "score": m.get("score"),
            "net_pnl": m.get("net_pnl"),
            "profit_factor": m.get("profit_factor"),
            "win_rate": m.get("win_rate"),
            "max_dd_pct": m.get("max_dd_pct"),
            "n_trades": m.get("n_trades"),
            "rl_take_threshold": run.get("rl_take_threshold"),
            "max_daily_loss_pct": run.get("max_daily_loss_pct"),
            "compound": run.get("compound"),
            "sizing_equity_mode": run.get("sizing_equity_mode"),
            "policy_path": r.get("policy_path"),
            "trade_log_path": r.get("trade_log_path", ""),
        })
    out_csv = os.path.join(args.outdir, "risk_sweep_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)

    print(f"\nSaved: {out_json}")
    print(f"Saved: {out_csv}")


if __name__ == "__main__":
    main()
