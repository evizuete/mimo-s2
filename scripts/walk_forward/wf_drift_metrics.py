#!/usr/bin/env python3
"""
wf_drift_metrics.py
─────────────────────────────────────────────────────────────────────────────
Lee los artifacts producidos por wf_train_at_date.py y decide auto-promote.

Métricas calculadas (a partir de holdout_predictions_<release>_<side>.parquet
y percentiles_<release>_<side>.json):

  - n_signals_<side>      = filas con proba_cal >= selected_threshold
  - win_rate_<side>       = mean(y_true) entre las señales
  - ev_R_<side>           = wr·tp_R − (1−wr)·sl_R − cost     (R-multiplos)
  - sharpe_<side>         = mean/std de R por señal (anualización omitida)

  - proba_cal_p50, p99    (informativos, drift soft)

Auto-promote criteria (configurable):
  · ev_R_long  >= MIN_EV_LONG   (default +0.05)
  · ev_R_short >= MIN_EV_SHORT  (default +0.03)
  · n_signals_long  >= MIN_N    (default 200)
  · n_signals_short >= MIN_N
  · win_rate_long > 0 AND win_rate_short > 0  (sanity)

Output:
  artifacts/<release>/wf_drift_metrics.json   (métricas + decision + reasons)
  artifacts/<release>/promote_decision.json   (alias compacto)

Exit code:
  0 si promoted=True
  10 si promoted=False (no bloquea, el orquestador decide qué hacer)
  >0 otros errores
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Defaults (alineados con read.me / 202500)
# ─────────────────────────────────────────────────────────────────────────────

TP_R = 2.0
SL_R = 0.8
COST_R = 0.05

MIN_EV_LONG = 0.05
MIN_EV_SHORT = 0.03
MIN_N_SIGNALS = 200


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _find_holdout_parquet(release: str, artifacts_root: Path,
                          side: str) -> Optional[Path]:
    """OOF v7 escribe en artifacts/<release>/oof/<exp_tag>/data/.
    Buscamos recursivamente."""
    base = artifacts_root / release / "oof"
    if not base.exists():
        return None
    matches = list(base.rglob(f"holdout_predictions_{release}_{side}.parquet"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _load_threshold(deploy_dir: Path, release: str,
                    side: str) -> Optional[float]:
    p = deploy_dir / f"percentiles_{release}_{side}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text())
    except json.JSONDecodeError:
        return None
    meta = data.get("_meta", {})
    thr = meta.get("selected_threshold")
    return float(thr) if thr is not None else None


def _proba_cal_col(df: pd.DataFrame, side: str) -> str:
    """Ubica la columna de proba calibrada según convención."""
    candidates = [
        "y_pred_cal",                # binary single-side
        "oof_proba_cal",
        f"oof_proba_{side}_cal",     # multitask combinado en un solo parquet
        "proba_cal",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"❌ No encuentro columna de proba calibrada en df. "
        f"Columnas: {list(df.columns)}"
    )


def _y_true_col(df: pd.DataFrame, side: str) -> str:
    candidates = [
        "y_true",
        f"y_true_{side}",
        f"signal_{side}",
        "signal",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(
        f"❌ No encuentro columna de y_true en df. "
        f"Columnas: {list(df.columns)}"
    )


def _side_metrics(parquet_path: Path, threshold: float, side: str,
                  tp_r: float, sl_r: float, cost_r: float) -> Dict[str, Any]:
    df = pd.read_parquet(parquet_path)
    proba_col = _proba_cal_col(df, side)
    y_col = _y_true_col(df, side)

    proba = df[proba_col].astype(float).to_numpy()
    y_true = df[y_col].astype(int).to_numpy()

    mask = proba >= threshold
    n_sig = int(mask.sum())
    n_total = int(len(df))

    if n_sig == 0:
        return {
            "side": side,
            "n_total": n_total,
            "n_signals": 0,
            "threshold": threshold,
            "win_rate": None,
            "ev_R": None,
            "sharpe": None,
            "proba_cal_p50": float(np.median(proba)),
            "proba_cal_p99": float(np.quantile(proba, 0.99)),
            "n_total_long": n_total,
        }

    y_sig = y_true[mask]
    wr = float(y_sig.mean())
    # R-multiplos por señal: y=1 → +tp_r ; y=0 → -sl_r ; coste constante
    r_per_signal = np.where(y_sig == 1, tp_r, -sl_r) - cost_r
    ev = float(r_per_signal.mean())
    sharpe = float(r_per_signal.mean() / r_per_signal.std()) if r_per_signal.std() > 0 else 0.0

    return {
        "side": side,
        "n_total": n_total,
        "n_signals": n_sig,
        "signal_rate": float(n_sig / n_total),
        "threshold": threshold,
        "win_rate": wr,
        "ev_R": ev,
        "sharpe": sharpe,
        "proba_cal_p50": float(np.median(proba)),
        "proba_cal_p99": float(np.quantile(proba, 0.99)),
        "tp_r": tp_r,
        "sl_r": sl_r,
        "cost_r": cost_r,
    }


def _decide_promote(m_long: Dict[str, Any], m_short: Dict[str, Any],
                    min_ev_long: float, min_ev_short: float,
                    min_n: int) -> Dict[str, Any]:
    reasons = []
    promoted = True

    for side, m, min_ev in [("long", m_long, min_ev_long),
                             ("short", m_short, min_ev_short)]:
        if m["n_signals"] < min_n:
            reasons.append(f"{side}: n_signals={m['n_signals']} < {min_n}")
            promoted = False
            continue
        if m["ev_R"] is None or m["ev_R"] < min_ev:
            reasons.append(
                f"{side}: ev_R={m['ev_R']} < {min_ev}"
                if m["ev_R"] is not None else
                f"{side}: ev_R no calculable"
            )
            promoted = False
        if m.get("win_rate") is not None and m["win_rate"] <= 0:
            reasons.append(f"{side}: win_rate={m['win_rate']} no positivo")
            promoted = False

    if promoted and not reasons:
        reasons.append("all checks pass")

    return {
        "promoted": promoted,
        "reasons": reasons,
        "criteria": {
            "min_ev_long": min_ev_long,
            "min_ev_short": min_ev_short,
            "min_n_signals": min_n,
        },
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--artifacts-root", type=Path,
                    default=REPO_ROOT / "artifacts")
    ap.add_argument("--deploy-subdir", default="deploy_full")
    ap.add_argument("--tp-r", type=float, default=TP_R)
    ap.add_argument("--sl-r", type=float, default=SL_R)
    ap.add_argument("--cost-r", type=float, default=COST_R)
    ap.add_argument("--min-ev-long", type=float, default=MIN_EV_LONG)
    ap.add_argument("--min-ev-short", type=float, default=MIN_EV_SHORT)
    ap.add_argument("--min-n-signals", type=int, default=MIN_N_SIGNALS)
    args = ap.parse_args()

    artifacts_root = args.artifacts_root.resolve()
    deploy_dir = artifacts_root / args.release / "oof" / args.deploy_subdir

    # Localizar parquets de holdout (OOF dir, no deploy)
    parquet_long = _find_holdout_parquet(args.release, artifacts_root, "long")
    parquet_short = _find_holdout_parquet(args.release, artifacts_root, "short")
    if parquet_long is None:
        sys.exit(f"❌ No encontré holdout_predictions_{args.release}_long.parquet")
    if parquet_short is None:
        sys.exit(f"❌ No encontré holdout_predictions_{args.release}_short.parquet")

    thr_long = _load_threshold(deploy_dir, args.release, "long")
    thr_short = _load_threshold(deploy_dir, args.release, "short")
    if thr_long is None:
        sys.exit(f"❌ selected_threshold no encontrado en percentiles_long json")
    if thr_short is None:
        sys.exit(f"❌ selected_threshold no encontrado en percentiles_short json")

    print(f"📂 long  preds: {parquet_long}  thr={thr_long:.4f}")
    print(f"📂 short preds: {parquet_short}  thr={thr_short:.4f}")

    m_long = _side_metrics(parquet_long, thr_long, "long",
                           args.tp_r, args.sl_r, args.cost_r)
    m_short = _side_metrics(parquet_short, thr_short, "short",
                            args.tp_r, args.sl_r, args.cost_r)

    decision = _decide_promote(m_long, m_short,
                               args.min_ev_long, args.min_ev_short,
                               args.min_n_signals)

    payload = {
        "release": args.release,
        "computed_at": datetime.now().isoformat(),
        "long": m_long,
        "short": m_short,
        "decision": decision,
    }

    out_dir = artifacts_root / args.release
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "wf_drift_metrics.json").write_text(json.dumps(payload, indent=2))
    (out_dir / "promote_decision.json").write_text(json.dumps({
        "release": args.release,
        "promoted": decision["promoted"],
        "reasons": decision["reasons"],
    }, indent=2))

    # ── Print summary ────────────────────────────────────────────────────────
    print("\n" + "═" * 72)
    print(f"  DRIFT METRICS — release={args.release}")
    print("═" * 72)
    for tag, m in [("LONG ", m_long), ("SHORT", m_short)]:
        if m["n_signals"] == 0:
            print(f"  {tag}  n_signals=0  (no señales sobre threshold)")
            continue
        print(f"  {tag}  n_sig={m['n_signals']:5d}/{m['n_total']:6d}  "
              f"wr={m['win_rate']:.3f}  EV={m['ev_R']:+.3f}R  "
              f"sharpe={m['sharpe']:+.2f}  thr={m['threshold']:.4f}")
    print("─" * 72)
    print(f"  decision: {'✅ PROMOTE' if decision['promoted'] else '❌ KEEP_PREVIOUS'}")
    for r in decision["reasons"]:
        print(f"     · {r}")
    print("═" * 72)

    sys.exit(0 if decision["promoted"] else 10)


if __name__ == "__main__":
    main()
