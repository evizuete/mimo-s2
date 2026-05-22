#!/usr/bin/env python3
"""
wf_compare_base_rates.py
─────────────────────────────────────────────────────────────────────────────
Diagnóstico rápido: compara base rate de la clase positiva (TP-first) en el
holdout de DOS o más releases. Sirve para confirmar si la causa de un WF
sin promotes es un regime shift de class balance entre el release del
bootstrap (ej: 202500) y los releases WF nuevos.

Salida: tabla por release/side con n_total, n_pos, base_rate, p50/p99 de proba
y delta vs el primer release (referencia).

Si la base rate cambia >30% relativo entre el bootstrap y un WF release,
los focal_alpha del bootstrap están descalibrados y los hyperparams
necesitan re-tunearse (Optuna mensual o specialists).

Uso:
    python -m scripts.walk_forward.wf_compare_base_rates \\
        --release 202500 \\
        --release wf_20260105 \\
        --release wf_20260112
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]


PROBA_CANDIDATES = [
    "y_pred_cal",
    "oof_proba_cal",
    "proba_cal",
]
PROBA_PER_SIDE = "oof_proba_{side}_cal"

Y_TRUE_CANDIDATES = [
    "y_true",
    "signal",
]
Y_TRUE_PER_SIDE = ["y_true_{side}", "signal_{side}"]


def _find_holdout(release: str, artifacts_root: Path, side: str) -> Path | None:
    base = artifacts_root / release / "oof"
    if not base.exists():
        return None
    matches = list(base.rglob(f"holdout_predictions_{release}_{side}.parquet"))
    if not matches:
        return None
    matches.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def _pick_proba_col(df: pd.DataFrame, side: str) -> str:
    candidates = [PROBA_PER_SIDE.format(side=side)] + PROBA_CANDIDATES
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"❌ no encuentro proba_cal en columnas: {list(df.columns)}")


def _pick_y_col(df: pd.DataFrame, side: str) -> str:
    candidates = [c.format(side=side) for c in Y_TRUE_PER_SIDE] + Y_TRUE_CANDIDATES
    for c in candidates:
        if c in df.columns:
            return c
    raise SystemExit(f"❌ no encuentro y_true en columnas: {list(df.columns)}")


def _stats(parquet: Path, side: str) -> dict:
    df = pd.read_parquet(parquet)
    proba_col = _pick_proba_col(df, side)
    y_col = _pick_y_col(df, side)
    proba = df[proba_col].astype(float).to_numpy()
    y = df[y_col].astype(int).to_numpy()
    return {
        "n_total": int(len(df)),
        "n_pos": int(y.sum()),
        "base_rate": float(y.mean()),
        "proba_p50": float(np.median(proba)),
        "proba_p99": float(np.quantile(proba, 0.99)),
        "proba_mean": float(proba.mean()),
        "rows_range": (
            f"{df.get('time', pd.Series(['?'])).iloc[0]} → "
            f"{df.get('time', pd.Series(['?'])).iloc[-1]}"
        ),
    }


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--release", action="append", required=True,
                    help="Release a comparar. Repetir para varios. El primero "
                         "se usa como referencia para los deltas.")
    ap.add_argument("--artifacts-root", type=Path,
                    default=REPO_ROOT / "artifacts")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    args = ap.parse_args()

    artifacts_root = args.artifacts_root.resolve()
    sides = ["long", "short"] if args.side == "both" else [args.side]

    rows = []
    for rel in args.release:
        for side in sides:
            p = _find_holdout(rel, artifacts_root, side)
            if p is None:
                print(f"⚠️  {rel}/{side}: no encontré holdout_predictions parquet")
                continue
            s = _stats(p, side)
            s["release"] = rel
            s["side"] = side
            s["parquet"] = p.relative_to(artifacts_root).as_posix()
            rows.append(s)

    if not rows:
        raise SystemExit("❌ ningún parquet encontrado")

    df = pd.DataFrame(rows)
    ref = args.release[0]
    df["base_rate_pct"] = df["base_rate"] * 100
    df = df.sort_values(["side", "release"]).reset_index(drop=True)

    print("\n" + "═" * 90)
    print(f"  BASE RATE & PROBA DISTRIBUTION (ref={ref})")
    print("═" * 90)
    for side in sides:
        sub = df[df["side"] == side].copy()
        if sub.empty:
            continue
        ref_row = sub[sub["release"] == ref]
        ref_br = float(ref_row["base_rate"].iloc[0]) if not ref_row.empty else None
        sub["Δ_vs_ref_pct"] = (
            (sub["base_rate"] - ref_br) / ref_br * 100 if ref_br else np.nan
        )
        print(f"\n  SIDE = {side.upper()}")
        cols = ["release", "n_total", "n_pos", "base_rate_pct",
                "Δ_vs_ref_pct", "proba_mean", "proba_p99"]
        print(sub[cols].round(3).to_string(index=False))

    print("\n" + "─" * 90)
    print("Interpretación rápida:")
    print("  · |Δ_vs_ref_pct| > 30  →  shift de class balance, focal_alpha mal calibrado")
    print("  · proba_mean ≪ proba_mean(ref) → modelo de-confidente, threshold no rescata")
    print("  · proba_p99 estable → modelo solo confía en pocos casos extremos")
    print("─" * 90)


if __name__ == "__main__":
    main()
