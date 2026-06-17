#!/usr/bin/env python3
"""
optimize_per_state_thresholds.py

Calcula los percentiles operativos optimos por estado para cada side, usando
el mismo formato que `gate_by_action_and_state` en decision_engine_percentiles.py.

Estrategia:
  - Lee calibration_dataset_<release>_<side>.parquet (OOF + holdout combinados).
  - Construye, sobre OOF, los percentiles dentro de estado (mismo grid que el
    engine: 50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99).
  - Para cada estado operativo, elige el percentile mas bajo (mas volumen) que
    cumple precision >= target en OOF, sujeto a recall_within_state >= min_recall.
  - Compara la policy propuesta contra la production actual (hardcodeada) en
    OOF y holdout: precision, recall, n, fp/tp.
  - Imprime el dict listo para pegar en decision_engine_percentiles.py y guarda
    un JSON.

Lo que NO hace:
  - No reentrena modelos. Solo desplaza umbrales.
  - No invade LOW_VOL (NO_TRADE).

Uso:
    python -m mimo.oof.optimize_per_state_thresholds
    python -m mimo.oof.optimize_per_state_thresholds --target-precision 0.27
    python -m mimo.oof.optimize_per_state_thresholds --min-recall 0.10
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(
    r"/mnt/c/Users/Usuario/Documents/Proyectos/TradingCo_s2/data_sample"
)
DEFAULT_RELEASE = "200393"

ENGINE_PERCENTILES: Tuple[int, ...] = (50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99)
NO_TRADE_STATES_LC = {"low_vol"}
MIN_OOF_SUPPORT = 800

# Mismo orden y valores que mimo/oof/decision_engine_percentiles.py @ 200393
CURRENT_PRODUCTION = {
    "long": {
        "trend_up": 97,
        "trend_down": 99,
        "breakout_wait_up": 99,
        "breakout_wait_down": 99,
        "range": 95,
        "transition_up": 99,
        "transition_down": 95,
        "_global": 97,
    },
    "short": {
        "trend_up": 99,
        "trend_down": 99,
        "breakout_wait_up": 99,
        "breakout_wait_down": 99,
        "range": 99,
        "transition_up": 99,
        "transition_down": 99,
        "_global": 99,
    },
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    ap.add_argument("--release", type=str, default=DEFAULT_RELEASE)
    ap.add_argument("--target-precision", type=float, default=0.25,
                    help="Precision objetivo en OOF por estado.")
    ap.add_argument("--min-recall", type=float, default=0.05,
                    help="Recall minimo dentro de estado en OOF (suelo de seguridad).")
    ap.add_argument("--out-json", type=str, default=None)
    return ap.parse_args()


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "source" not in df.columns:
        df["source"] = "oof_train"
    df = df.dropna(subset=["signal", "oof_proba_cal", "state", "source"]).copy()
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").astype(int)
    df["state_lc"] = df["state"].astype(str).str.lower()
    df["source"] = df["source"].astype(str)
    return df


def percentile_grid(df_state: pd.DataFrame) -> Dict[int, float]:
    vals = df_state["oof_proba_cal"].to_numpy()
    return {q: float(np.percentile(vals, q)) for q in ENGINE_PERCENTILES}


def metrics_at(df: pd.DataFrame, tau: float) -> dict:
    mask = df["oof_proba_cal"] >= tau
    n = int(mask.sum())
    tp = int(df.loc[mask, "signal"].sum())
    fp = n - tp
    pos_total = int(df["signal"].sum())
    return {
        "n_signals": n,
        "tp": tp,
        "fp": fp,
        "precision": (tp / n) if n > 0 else float("nan"),
        "recall": (tp / pos_total) if pos_total > 0 else float("nan"),
    }


def pick_percentile(
    df_state: pd.DataFrame,
    grid: Dict[int, float],
    target_precision: float,
    min_recall: float,
) -> Tuple[int, dict]:
    """
    Recorre el grid de menor a mayor q. Devuelve el q mas bajo (mas volumen) que:
        precision_oof >= target_precision   AND
        recall_oof    >= min_recall

    Si ninguno cumple precision: devuelve el q con mayor precision dentro de
    los que respetan min_recall.
    Si ninguno respeta min_recall: devuelve el q mas bajo (mas permisivo).
    """
    rows = []
    for q in sorted(grid.keys()):
        m = metrics_at(df_state, grid[q])
        m["q"] = q
        m["tau"] = grid[q]
        rows.append(m)

    with_recall = [m for m in rows if m["recall"] >= min_recall]
    if not with_recall:
        return rows[0]["q"], rows[0]
    meeting = [m for m in with_recall if m["precision"] >= target_precision]
    if meeting:
        chosen = min(meeting, key=lambda m: m["q"])
        return chosen["q"], chosen
    chosen = max(with_recall, key=lambda m: m["precision"])
    return chosen["q"], chosen


def apply_policy(df: pd.DataFrame, policy: Dict[str, int],
                 grids: Dict[str, Dict[int, float]]) -> pd.DataFrame:
    out = df.copy()
    out["tau"] = np.nan
    for state, q in policy.items():
        if state.startswith("_"):
            continue
        grid = grids.get(state)
        if grid is None or q not in grid:
            continue
        out.loc[out["state_lc"] == state, "tau"] = grid[q]

    g_q = policy.get("_global")
    if g_q is not None and "_global" in grids:
        tau_global = grids["_global"][g_q]
        unmatched = out["tau"].isna() & ~out["state_lc"].isin(NO_TRADE_STATES_LC)
        out.loc[unmatched, "tau"] = tau_global

    out["fired"] = (
        (out["oof_proba_cal"] >= out["tau"].fillna(np.inf))
        & (~out["state_lc"].isin(NO_TRADE_STATES_LC))
    )
    return out


def aggregate(fired_df: pd.DataFrame) -> dict:
    fired = fired_df[fired_df["fired"]]
    n = len(fired)
    if n == 0:
        return {"n_signals": 0, "tp": 0, "fp": 0, "precision": float("nan"),
                "recall": float("nan"), "fp_per_tp": float("inf")}
    tp = int(fired["signal"].sum())
    fp = n - tp
    pos_total = int(fired_df["signal"].sum())
    return {
        "n_signals": n,
        "tp": tp,
        "fp": fp,
        "precision": tp / n,
        "recall": (tp / pos_total) if pos_total > 0 else float("nan"),
        "fp_per_tp": (fp / tp) if tp > 0 else float("inf"),
    }


def per_state_breakdown(fired_df: pd.DataFrame) -> pd.DataFrame:
    fired = fired_df[fired_df["fired"]]
    if fired.empty:
        return pd.DataFrame()
    bd = fired.groupby("state_lc").agg(n=("signal", "size"), tp=("signal", "sum"))
    bd["fp"] = bd["n"] - bd["tp"]
    bd["precision"] = bd["tp"] / bd["n"]
    pos_total = fired_df.groupby("state_lc")["signal"].sum().rename("pos_total")
    bd = bd.join(pos_total, how="left")
    bd["recall_within_state"] = bd["tp"] / bd["pos_total"].clip(lower=1)
    return bd.sort_values("n", ascending=False)


def analyze_side(side: str, path: Path, target_precision: float, min_recall: float) -> Dict[str, int]:
    section(f"SIDE = {side.upper()}  |  {path.name}")
    if not path.exists():
        print(f"  [skip] no existe: {path}")
        return {}

    df = load(path)
    df_oof = df[df["source"] == "oof_train"].copy()
    df_hold = df[df["source"] == "holdout"].copy()
    print(f"  oof rows: {len(df_oof):,}   holdout rows: {len(df_hold):,}")

    grids: Dict[str, Dict[int, float]] = {}
    for state, g in df_oof.groupby("state_lc"):
        if len(g) >= MIN_OOF_SUPPORT:
            grids[state] = percentile_grid(g)
    grids["_global"] = percentile_grid(
        df_oof[~df_oof["state_lc"].isin(NO_TRADE_STATES_LC)]
    )

    section(f"[{side}] SELECCION DE PERCENTILE POR ESTADO (sobre OOF)")
    print(f"  target_precision={target_precision}   min_recall={min_recall}")
    print(f"\n  {'state':<22} {'q':>3}  {'tau':>7}  {'prec':>6}  {'recall':>6}  {'n':>8}")
    new_policy: Dict[str, int] = {}
    for state in sorted(grids.keys()):
        if state == "_global":
            continue
        g_state = df_oof[df_oof["state_lc"] == state]
        q, m = pick_percentile(g_state, grids[state], target_precision, min_recall)
        new_policy[state] = q
        print(f"  {state:<22} {q:>3}  {m['tau']:>7.4f}  {m['precision']:>6.4f}  {m['recall']:>6.4f}  {m['n_signals']:>8}")

    g_pool = df_oof[~df_oof["state_lc"].isin(NO_TRADE_STATES_LC)]
    q_g, m_g = pick_percentile(g_pool, grids["_global"], target_precision, min_recall)
    new_policy["_global"] = q_g
    print(f"  {'_global':<22} {q_g:>3}  {m_g['tau']:>7.4f}  {m_g['precision']:>6.4f}  {m_g['recall']:>6.4f}  {m_g['n_signals']:>8}")

    for k in CURRENT_PRODUCTION[side].keys():
        new_policy.setdefault(k, q_g)

    section(f"[{side}] COMPARATIVA  POLICY ACTUAL vs PROPUESTA")
    rows = []
    for label, policy in [("ACTUAL_PRODUCTION", CURRENT_PRODUCTION[side]),
                          ("PROPUESTA",         new_policy)]:
        for src_label, src_df in [("OOF", df_oof), ("HOLDOUT", df_hold)]:
            agg = aggregate(apply_policy(src_df, policy, grids))
            rows.append({
                "policy": label, "source": src_label,
                "n": agg["n_signals"], "tp": agg["tp"], "fp": agg["fp"],
                "precision": agg["precision"], "recall": agg["recall"],
                "fp_per_tp": agg["fp_per_tp"],
            })
    cmp_df = pd.DataFrame(rows)
    print(cmp_df.round(4).to_string(index=False))

    section(f"[{side}] DESGLOSE POR ESTADO EN HOLDOUT (POLICY PROPUESTA)")
    bd = per_state_breakdown(apply_policy(df_hold, new_policy, grids))
    if bd.empty:
        print("  (sin senales emitidas)")
    else:
        print(bd.round(4).to_string())

    section(f"[{side}] DICT PARA PEGAR EN decision_engine_percentiles.py")
    ordered: Dict[str, int] = {}
    for k in CURRENT_PRODUCTION[side].keys():
        if k in new_policy:
            ordered[k] = new_policy[k]
    for k, v in new_policy.items():
        if k not in ordered and not k.startswith("_"):
            ordered[k] = v
    if "_global" in new_policy:
        ordered["_global"] = new_policy["_global"]
    print(json.dumps({side: ordered}, indent=4))
    return ordered


def main() -> None:
    args = parse_args()
    base = Path(args.dir)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    out: Dict[str, Dict[str, int]] = {}
    for side in ("long", "short"):
        path = base / f"calibration_dataset_{args.release}_{side}.parquet"
        out[side] = analyze_side(side, path, args.target_precision, args.min_recall)

    out_path = Path(args.out_json) if args.out_json else (
        base / f"proposed_state_gates_{args.release}.json"
    )
    payload = {
        "training": out,
        "production": out,
        "_meta": {
            "release": args.release,
            "target_precision": args.target_precision,
            "min_recall": args.min_recall,
        },
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)
    print(f"\n[ok] propuesta guardada en: {out_path}")


if __name__ == "__main__":
    main()
