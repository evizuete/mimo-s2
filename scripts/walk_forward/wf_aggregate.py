#!/usr/bin/env python3
"""
wf_aggregate.py
─────────────────────────────────────────────────────────────────────────────
Recorre artifacts/walk_forward/<YYYY-MM-DD>/wf_step_summary.json + el
release activo y los artefactos de replay/drift, y construye:

    artifacts/walk_forward/aggregated/
      ├─ progression.parquet      # 1 fila por semana
      ├─ trades_all.parquet       # concat de todos los trades
      ├─ equity_curve_all.parquet # curva continua
      └─ progression_report.md    # tabla + warnings

Idempotente: regenera todo desde cero cada vez. Pensado para ejecutarse
durante o después del wf_orchestrator para visibilidad de progresión.

Ejemplo:
    python -m scripts.walk_forward.wf_aggregate \\
        --wf-root artifacts/walk_forward \\
        --artifacts-root artifacts
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_step_summaries(wf_root: Path) -> List[Dict[str, Any]]:
    rows = []
    for p in sorted(wf_root.glob("*/wf_step_summary.json")):
        try:
            rows.append(json.loads(p.read_text()))
        except json.JSONDecodeError:
            print(f"⚠️  no pude leer {p}")
    return rows


def load_drift_metrics(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


def load_replay_summary(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text())
    except json.JSONDecodeError:
        return {}


# ─────────────────────────────────────────────────────────────────────────────
# Aggregators
# ─────────────────────────────────────────────────────────────────────────────

def build_progression(step_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    pnl_cum = 0.0
    peak_eq = None
    initial_eq = None

    for step in step_summaries:
        drift = load_drift_metrics(step.get("paths", {}).get("drift_metrics"))
        replay = load_replay_summary(step.get("paths", {}).get("replay_summary"))

        long_m = drift.get("long", {}) or {}
        short_m = drift.get("short", {}) or {}

        replay_trades = int(replay.get("n_trades", 0)) if replay else 0
        # pnl_total fue añadido al summary v2; fallback a final-initial si falta.
        if replay:
            if "pnl_total" in replay:
                replay_pnl = float(replay["pnl_total"])
            else:
                replay_pnl = float(replay.get("final_equity", 0.0)) - float(replay.get("initial_equity", 0.0))
        else:
            replay_pnl = 0.0
        replay_dd_pct = float(replay.get("max_drawdown_pct", 0.0)) if replay else 0.0
        replay_final_eq = float(replay.get("final_equity", 0.0)) if replay else None
        replay_initial_eq = float(replay.get("initial_equity", 0.0)) if replay else None
        replay_wr_pct = float(replay.get("win_rate_pct", 0.0)) if replay else 0.0
        replay_wr = replay_wr_pct / 100.0  # mantenemos progresión en fracción

        if initial_eq is None and replay_initial_eq:
            initial_eq = replay_initial_eq
        pnl_cum += replay_pnl
        running_eq = (initial_eq or 0.0) + pnl_cum
        if peak_eq is None or running_eq > peak_eq:
            peak_eq = running_eq
        running_dd_pct = (
            (running_eq - peak_eq) / peak_eq * 100.0
            if peak_eq and peak_eq > 0 else 0.0
        )

        rows.append({
            "week_index": step.get("week_index"),
            "cutoff": step.get("cutoff"),
            "release_trained": step.get("release_trained"),
            "is_optuna_week": step.get("is_optuna_week"),
            "promoted": step.get("promoted"),
            "active_release": step.get("active_release_for_replay"),
            # holdout drift metrics
            "ev_R_long": long_m.get("ev_R"),
            "ev_R_short": short_m.get("ev_R"),
            "n_signals_long": long_m.get("n_signals"),
            "n_signals_short": short_m.get("n_signals"),
            "win_rate_long": long_m.get("win_rate"),
            "win_rate_short": short_m.get("win_rate"),
            "thr_long": long_m.get("threshold"),
            "thr_short": short_m.get("threshold"),
            "proba_p99_long": long_m.get("proba_cal_p99"),
            "proba_p99_short": short_m.get("proba_cal_p99"),
            # replay live week
            "replay_n_trades": replay_trades,
            "replay_pnl_week": replay_pnl,
            "replay_wr_week": replay_wr,
            "replay_final_eq": replay_final_eq,
            "replay_dd_pct_week": replay_dd_pct,
            "pnl_cum": pnl_cum,
            "running_dd_pct": running_dd_pct,
        })

    df = pd.DataFrame(rows)
    if "cutoff" in df.columns:
        df["cutoff_dt"] = pd.to_datetime(df["cutoff"])
    return df


def build_trades_all(step_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    frames = []
    for step in step_summaries:
        replay_summary = step.get("paths", {}).get("replay_summary")
        if not replay_summary:
            continue
        sp = Path(replay_summary)
        trades_path = sp.parent / "trades.parquet"
        if not trades_path.exists():
            continue
        try:
            df = pd.read_parquet(trades_path)
        except Exception:
            continue
        df["wf_cutoff"] = step.get("cutoff")
        df["wf_active_release"] = step.get("active_release_for_replay")
        df["wf_week_index"] = step.get("week_index")
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def build_equity_all(step_summaries: List[Dict[str, Any]]) -> pd.DataFrame:
    frames = []
    cum_offset = 0.0
    base_initial = None
    for step in step_summaries:
        replay_summary = step.get("paths", {}).get("replay_summary")
        if not replay_summary:
            continue
        sp = Path(replay_summary)
        eq_path = sp.parent / "equity_curve.parquet"
        if not eq_path.exists():
            continue
        try:
            df = pd.read_parquet(eq_path)
        except Exception:
            continue
        if df.empty:
            continue

        # Cualquier esquema razonable: time/timestamp + equity/equity_mtm
        time_col = "time" if "time" in df.columns else (
            "timestamp" if "timestamp" in df.columns else df.columns[0]
        )
        eq_col = (
            "equity_mtm" if "equity_mtm" in df.columns
            else ("equity" if "equity" in df.columns else df.columns[1])
        )
        if base_initial is None:
            base_initial = float(df[eq_col].iloc[0])

        # Re-base la curva sumando el offset acumulado de semanas anteriores
        df = df.rename(columns={time_col: "time", eq_col: "equity"}).copy()
        df["equity"] = df["equity"].astype(float) + cum_offset
        df["wf_cutoff"] = step.get("cutoff")
        frames.append(df[["time", "equity", "wf_cutoff"]])

        # Actualizar offset: PnL acumulado al final de esta semana respecto al
        # inicio de la curva (no respecto a base_initial).
        try:
            week_pnl = float(df["equity"].iloc[-1] - df["equity"].iloc[0])
        except Exception:
            week_pnl = 0.0
        cum_offset += week_pnl

    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    out["time"] = pd.to_datetime(out["time"])
    out = out.sort_values("time").reset_index(drop=True)
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Markdown report
# ─────────────────────────────────────────────────────────────────────────────

def render_report(df: pd.DataFrame) -> str:
    if df.empty:
        return "# Walk-Forward Progression\n\n_No data yet._\n"

    lines = ["# Walk-Forward Progression",
             f"_Generado: {datetime.now().isoformat()}_",
             ""]

    # Resumen ejecutivo
    n_weeks = len(df)
    n_promoted = int(df["promoted"].fillna(False).sum()) if "promoted" in df.columns else 0
    n_optuna = int(df["is_optuna_week"].fillna(False).sum()) if "is_optuna_week" in df.columns else 0
    pnl_total = float(df["replay_pnl_week"].fillna(0).sum()) if "replay_pnl_week" in df.columns else 0.0
    pnl_cum_last = float(df["pnl_cum"].fillna(0).iloc[-1]) if "pnl_cum" in df.columns else 0.0
    dd_min = float(df["running_dd_pct"].fillna(0).min()) if "running_dd_pct" in df.columns else 0.0
    n_trades_total = int(df["replay_n_trades"].fillna(0).sum()) if "replay_n_trades" in df.columns else 0

    lines += [
        "## Resumen ejecutivo",
        "",
        f"- semanas procesadas: **{n_weeks}**",
        f"- weeks Optuna: {n_optuna}  |  weeks weight-only: {n_weeks - n_optuna}",
        f"- promotions: {n_promoted}/{n_weeks}",
        f"- PnL total replay: **{pnl_total:+,.2f}**",
        f"- PnL acumulado al cierre: **{pnl_cum_last:+,.2f}**",
        f"- DD máximo running: **{dd_min:+.2f}%**",
        f"- trades totales: {n_trades_total}",
        "",
    ]

    # Tabla por semana
    lines += ["## Progresión semana a semana", ""]
    cols = ["cutoff", "release_trained", "is_optuna_week", "promoted",
            "ev_R_long", "ev_R_short", "n_signals_long", "n_signals_short",
            "replay_n_trades", "replay_pnl_week", "pnl_cum", "running_dd_pct"]
    cols = [c for c in cols if c in df.columns]
    table = df[cols].copy()

    def _fmt(x, kind):
        if x is None or (isinstance(x, float) and (x != x)):
            return "—"
        if kind == "int":
            return f"{int(x)}"
        if kind == "pct":
            return f"{x:+.2f}%"
        if kind == "money":
            return f"{x:+,.2f}"
        if kind == "R":
            return f"{x:+.3f}"
        if kind == "bool":
            return "✓" if x else "·"
        return str(x)

    headers = ["cutoff", "release", "Optuna", "promoted",
               "EV_R long", "EV_R short", "n_long", "n_short",
               "trades", "pnl_w", "pnl_cum", "dd_run"]
    lines += ["| " + " | ".join(headers) + " |",
              "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, r in table.iterrows():
        row = [
            str(r.get("cutoff", "—")),
            str(r.get("release_trained", "—")),
            _fmt(r.get("is_optuna_week"), "bool"),
            _fmt(r.get("promoted"), "bool"),
            _fmt(r.get("ev_R_long"), "R"),
            _fmt(r.get("ev_R_short"), "R"),
            _fmt(r.get("n_signals_long"), "int"),
            _fmt(r.get("n_signals_short"), "int"),
            _fmt(r.get("replay_n_trades"), "int"),
            _fmt(r.get("replay_pnl_week"), "money"),
            _fmt(r.get("pnl_cum"), "money"),
            _fmt(r.get("running_dd_pct"), "pct"),
        ]
        lines.append("| " + " | ".join(row) + " |")

    # Warnings de drift
    warnings = []
    if "ev_R_long" in df.columns:
        bad = df[df["ev_R_long"].fillna(99) < 0]
        if len(bad) > 0:
            warnings.append(f"{len(bad)} semanas con EV_R LONG < 0 en holdout")
    if "ev_R_short" in df.columns:
        bad = df[df["ev_R_short"].fillna(99) < 0]
        if len(bad) > 0:
            warnings.append(f"{len(bad)} semanas con EV_R SHORT < 0 en holdout")
    if "promoted" in df.columns:
        not_promoted = df[~df["promoted"].fillna(False).astype(bool)]
        if len(not_promoted) > 0:
            cutoffs = ", ".join(not_promoted["cutoff"].astype(str).tolist())
            warnings.append(f"{len(not_promoted)} semanas NO promovidas: {cutoffs}")
    if warnings:
        lines += ["", "## Warnings", ""]
        lines += [f"- {w}" for w in warnings]
    lines.append("")
    return "\n".join(lines)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wf-root", type=Path,
                    default=REPO_ROOT / "artifacts" / "walk_forward")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Default: <wf-root>/aggregated")
    args = ap.parse_args()

    wf_root = args.wf_root.resolve()
    out_dir = (args.out_dir or (wf_root / "aggregated")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = load_step_summaries(wf_root)
    if not steps:
        print(f"⚠️  no encontré wf_step_summary.json bajo {wf_root}")
        sys.exit(0)

    print(f"📂 step summaries: {len(steps)}")

    progression = build_progression(steps)
    progression_path = out_dir / "progression.parquet"
    progression.to_parquet(progression_path, index=False)
    print(f"✅ progression → {progression_path}  ({len(progression)} filas)")

    trades_all = build_trades_all(steps)
    trades_path = out_dir / "trades_all.parquet"
    if not trades_all.empty:
        trades_all.to_parquet(trades_path, index=False)
        print(f"✅ trades_all  → {trades_path}  ({len(trades_all)} filas)")
    else:
        print(f"⚠️  trades_all vacío (no hay replays todavía)")

    equity_all = build_equity_all(steps)
    eq_path = out_dir / "equity_curve_all.parquet"
    if not equity_all.empty:
        equity_all.to_parquet(eq_path, index=False)
        print(f"✅ equity_all  → {eq_path}  ({len(equity_all)} filas)")
    else:
        print(f"⚠️  equity_all vacío")

    md_path = out_dir / "progression_report.md"
    md_path.write_text(render_report(progression))
    print(f"✅ report      → {md_path}")


if __name__ == "__main__":
    import sys
    main()
