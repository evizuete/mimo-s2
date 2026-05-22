#!/usr/bin/env python3
"""
monitor_health.py — Orquesta ghost_predict + analyzers para producir un
health report unificado del deploy en producción.

Genera por cada ejecución (default: últimos 30 días):
  reports/monitor/health_<YYYY-MM-DD>/
    ├── ghost.parquet
    ├── calibration/      (analyze_calibration_mimo)
    ├── score_pnl/        (analyze_score_to_pnl)
    └── health_report.json (con alertas + recomendación)

Uso:
  python scripts/monitor_health.py \\
    --release 202500 \\
    --deploy-subdir deploy_PROD_combined_specialists_seed47 \\
    --policy-config decision_policies_config_202500_PROD

  # Ventana custom:
  --from 2026-05-01 --to 2026-05-31
"""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


ALERTS = {
    "ece_warn": 0.05,
    "ece_critical": 0.10,
    "auc_pr_min_factor": 1.0,    # auc_pr debe ser >= base_rate
    "state_toxic_R_per_trade": -0.10,
    "state_toxic_min_n": 30,
    "weekly_pnl_negative_streak": 2,
}


def run_cmd(cmd: List[str], name: str) -> int:
    print(f"\n🔧 {name}...")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"   ❌ rc={res.returncode}")
        if res.stderr:
            print(f"   stderr (tail): {res.stderr.strip()[-500:]}")
    else:
        print(f"   ✅ OK")
    return res.returncode


def evaluate_alerts(cal_summary: Optional[Dict], pnl_df: Optional[pd.DataFrame]) -> List[Dict]:
    alerts = []
    if cal_summary and "overall_periods" in cal_summary:
        hd = next((p for p in cal_summary["overall_periods"]
                   if p.get("period") == "holdout"), None)
        if hd:
            ece = float(hd.get("ece", 0) or 0)
            if ece > ALERTS["ece_critical"]:
                alerts.append({"level": "critical", "type": "calibration_drift_severe",
                               "value": ece, "threshold": ALERTS["ece_critical"]})
            elif ece > ALERTS["ece_warn"]:
                alerts.append({"level": "warning", "type": "calibration_drift_mild",
                               "value": ece, "threshold": ALERTS["ece_warn"]})

            auc_pr = float(hd.get("auc_pr", 0) or 0)
            pos_rate = float(hd.get("pos_rate", 0) or 0)
            if pos_rate > 0 and auc_pr < pos_rate * ALERTS["auc_pr_min_factor"]:
                alerts.append({"level": "critical", "type": "discrimination_loss",
                               "auc_pr": auc_pr, "base_rate": pos_rate})

    if pnl_df is not None and not pnl_df.empty:
        hd_df = pnl_df[pnl_df["period"] == "holdout"].copy()
        if not hd_df.empty:
            agg = hd_df.groupby("state").agg(
                n=("n", "sum"),
                total_pnl=("total_pnl", "sum"),
            )
            agg["pnl_per_trade"] = agg["total_pnl"] / agg["n"].clip(lower=1)
            for state, row in agg.iterrows():
                if row["n"] < ALERTS["state_toxic_min_n"]:
                    continue
                if row["pnl_per_trade"] < ALERTS["state_toxic_R_per_trade"]:
                    alerts.append({"level": "warning", "type": "state_toxic",
                                   "state": str(state),
                                   "pnl_per_trade": float(row["pnl_per_trade"]),
                                   "n_trades": int(row["n"])})
    return alerts


def decide(alerts: List[Dict]) -> Dict[str, str]:
    if not alerts:
        return {"level": "OK", "action": "Continuar monitorizando, sin acciones."}
    crit = [a for a in alerts if a["level"] == "critical"]
    warn = [a for a in alerts if a["level"] == "warning"]
    if any(a["type"] == "discrimination_loss" for a in crit):
        return {"level": "REDEPLOY",
                "action": "Modelo pierde poder predictivo. Redeploy completo "
                          "(rerun Fase B-E con cutoff actual)."}
    if any(a["type"] == "calibration_drift_severe" for a in crit):
        return {"level": "RECALIBRATE",
                "action": "Drift severo de calibración. Re-ejecutar "
                          "select_thresholds_from_tail con tail reciente; "
                          "si insuficiente, redeploy."}
    state_toxic = [a for a in warn if a["type"] == "state_toxic"]
    if state_toxic:
        states = ", ".join(a["state"] for a in state_toxic)
        return {"level": "POLICY_TUNE",
                "action": f"Estados tóxicos ({states}): subir quantile en "
                          f"gate_by_action_and_state o reducir risk_mult."}
    if len(warn) >= 2:
        return {"level": "VIGILANCE",
                "action": f"{len(warn)} warnings: revisar manualmente, "
                          f"vigilar próxima semana."}
    return {"level": "MONITOR", "action": "Alertas menores, monitorizar."}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="202500")
    ap.add_argument("--deploy-subdir", required=True)
    ap.add_argument("--policy-config", default="decision_policies_config_202500_PROD")
    ap.add_argument("--from", dest="from_date", default=None,
                    help="YYYY-MM-DD. Default: hoy-30d")
    ap.add_argument("--to", dest="to_date", default=None,
                    help="YYYY-MM-DD. Default: hoy")
    ap.add_argument("--out-root", default="reports/monitor")
    args = ap.parse_args()

    today = datetime.now().date()
    to_date = args.to_date or today.isoformat()
    from_date = args.from_date or (today - timedelta(days=30)).isoformat()

    out_dir = Path(args.out_root) / f"health_{today.isoformat()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    ghost_path = out_dir / "ghost.parquet"
    cal_out = out_dir / "calibration"
    pnl_out = out_dir / "score_pnl"

    print(f"\n{'═'*78}")
    print(f"  HEALTH MONITORING")
    print(f"  Deploy:  {args.deploy_subdir}")
    print(f"  Window:  {from_date} → {to_date}")
    print(f"  Outputs: {out_dir}")
    print(f"{'═'*78}")

    rc = run_cmd([sys.executable, "scripts/ghost_predict.py",
                  "--release", args.release,
                  "--deploy-subdir", args.deploy_subdir,
                  "--from", from_date, "--to", to_date,
                  "--include-tail", "--out", str(ghost_path)],
                 "ghost_predict")
    if rc != 0 or not ghost_path.exists():
        print("❌ ghost_predict falló o no produjo output. Abortando.")
        sys.exit(1)

    run_cmd([sys.executable, "-m", "mimo.oof.shift_analyzer.analyze_calibration_mimo",
             "--input", str(ghost_path),
             "--score-col", "oof_proba_cal", "--target-col", "signal",
             "--state-col", "state", "--time-col", "time",
             "--period-col", "period", "--output-dir", str(cal_out)],
            "analyze_calibration_mimo")

    run_cmd([sys.executable, "-m", "mimo.oof.shift_analyzer.analyze_score_to_pnl",
             "--input", str(ghost_path),
             "--score-col", "oof_proba_cal", "--target-col", "signal",
             "--pnl-col", "R_multiple", "--state-col", "state", "--time-col", "time",
             "--period-col", "period", "--output-dir", str(pnl_out)],
            "analyze_score_to_pnl")

    # Cargar outputs
    cal_summary = None
    if (cal_out / "summary.json").exists():
        cal_summary = json.load((cal_out / "summary.json").open())
    pnl_df = None
    pnl_csv = pnl_out / "score_to_pnl_by_period_state.csv"
    if pnl_csv.exists():
        pnl_df = pd.read_csv(pnl_csv)

    alerts = evaluate_alerts(cal_summary, pnl_df)
    recommendation = decide(alerts)

    report = {
        "timestamp": datetime.now().isoformat(),
        "deploy_subdir": args.deploy_subdir,
        "window": {"from": from_date, "to": to_date},
        "calibration": (cal_summary.get("overall_periods")
                        if cal_summary and "overall_periods" in cal_summary else None),
        "alerts": alerts,
        "recommendation": recommendation,
    }
    (out_dir / "health_report.json").write_text(json.dumps(report, indent=2, default=str))

    # Consola
    print(f"\n\n{'═'*78}\n  HEALTH REPORT\n{'═'*78}")
    if cal_summary and "overall_periods" in cal_summary:
        for p in cal_summary["overall_periods"]:
            if p.get("period") == "holdout":
                print(f"  HOLDOUT n={p.get('n', 0)}  "
                      f"ECE={float(p.get('ece', 0) or 0):.4f}  "
                      f"AUC-PR={float(p.get('auc_pr', 0) or 0):.4f}  "
                      f"pos_rate={float(p.get('pos_rate', 0) or 0):.4f}")

    print()
    if alerts:
        print(f"  🚨 {len(alerts)} alertas:")
        for a in alerts:
            icon = "🔴" if a["level"] == "critical" else "🟠"
            details = ", ".join(f"{k}={v}" for k, v in a.items()
                                if k not in ("level", "type"))
            print(f"     {icon} [{a['level'].upper()}] {a['type']}  ({details})")
    else:
        print(f"  ✅ Sin alertas")

    print(f"\n  📊 RECOMENDACIÓN: {recommendation['level']}")
    print(f"     {recommendation['action']}")
    print(f"\n  📁 Reporte JSON: {out_dir / 'health_report.json'}")


if __name__ == "__main__":
    main()