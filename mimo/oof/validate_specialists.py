#!/usr/bin/env python3
"""
validate_specialists.py

Sanity check de reproducibilidad tras `train_specialist`. Compara las
métricas OOF que reportó el trial original (en best_per_side.json) con
las que produce el reentreno del specialist al mismo threshold lockeado.

Si la diferencia es despreciable (ev_net dentro de ±1e-3, n_signals con
diferencia ≤ 2) el reentreno está reproduciendo bien. Si difiere fuerte,
hay un bug de seed/orden de folds que hay que investigar antes de seguir
a drift validation.

Inputs:
  --best-per-side-json        JSON de extract_best_per_side
  --specialist-long-dir       artifacts dir del long specialist
  --specialist-short-dir      artifacts dir del short specialist
  --side {long,short,both}    qué validar (default both)
  --release                   nombre del release (para localizar oof_<release>_<side>.parquet)
  --ohlcv | --from-db         para recomputar ATR
  --tp / --sl / --horizon     barreras (default 2.0 / 0.8 / 3)
  --cost                      coste por señal en R (default 0.05)
  --max-drawdown-R            límite MDD para penalty (default 30)

Output: tabla side-by-side de:
  thr | ev_net | ev_gross | n_signals | prec_TP | mdd_R
con dos columnas (RECORDED del trial / RECOMPUTED del specialist) y
delta absoluto. Veredicto PASS/FAIL al final.

Uso:
  python -m mimo.oof.validate_specialists \
    --best-per-side-json artifacts/202500/oof/<tag>/reports/best_per_side.json \
    --specialist-long-dir  artifacts/202500/oof/<tag>_long_specialist \
    --specialist-short-dir artifacts/202500/oof/<tag>_short_specialist \
    --release 202500 --from-db --base-tf 5min \
    --tp 2.0 --sl 0.8 --horizon 3 --cost 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

from mimo.oof.empirical_breakeven import wilder_atr
from mimo.oof.ev_objective import (
    _simulate_outcomes_for_indices,
    _max_drawdown_R,
)


# ─────────────────────────────────────────────────────────────────────────────
# Recompute helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_metrics_at_locked_thr(
    df_oof: pd.DataFrame,
    *,
    proba_col: str,
    side_is_long: bool,
    locked_thr: float,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    cost: float,
    max_drawdown_R: float,
) -> Dict[str, Any]:
    """Replay del barrier en df_oof al thr lockeado y recompute de métricas."""
    needed = {"time", "high", "low", "close", "atr", proba_col}
    miss = needed - set(df_oof.columns)
    if miss:
        raise ValueError(f"OOF parquet le faltan columnas: {miss}")

    mask = (
        df_oof[proba_col].notna()
        & df_oof["atr"].notna()
        & (df_oof["atr"] > 0)
    )
    df = df_oof.loc[mask].reset_index(drop=False).sort_values("time").reset_index(drop=True)

    high = df_oof["high"].to_numpy(dtype=np.float64)
    low = df_oof["low"].to_numpy(dtype=np.float64)
    close = df_oof["close"].to_numpy(dtype=np.float64)
    atr = df_oof["atr"].to_numpy(dtype=np.float64)
    orig_idx = df["index"].to_numpy(dtype=np.int64)
    p = df[proba_col].to_numpy(dtype=np.float64)

    outcome, r_real = _simulate_outcomes_for_indices(
        orig_idx, high, low, close, atr,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        side_is_long=side_is_long,
    )
    valid = outcome != 3

    pred = (p >= locked_thr) & valid
    sig = int(pred.sum())
    if sig == 0:
        return {
            "thr": float(locked_thr),
            "n_signals": 0,
            "n_TP": 0, "n_SL": 0, "n_EXPIRE": 0,
            "prec_TP": 0.0,
            "frac_SL": 0.0,
            "frac_EXP": 0.0,
            "ev_gross": 0.0,
            "ev_net": -cost,
            "mdd_R": 0.0,
        }

    sub_outcome = outcome[pred]
    sub_r = r_real[pred]
    n_tp = int((sub_outcome == 0).sum())
    n_sl = int((sub_outcome == 1).sum())
    n_exp = int((sub_outcome == 2).sum())
    ev_gross = float(sub_r.mean())
    ev_net = ev_gross - cost
    mdd = _max_drawdown_R(sub_r - cost)
    return {
        "thr": float(locked_thr),
        "n_signals": sig,
        "n_TP": n_tp, "n_SL": n_sl, "n_EXPIRE": n_exp,
        "prec_TP": n_tp / sig,
        "frac_SL": n_sl / sig,
        "frac_EXP": n_exp / sig,
        "ev_gross": float(ev_gross),
        "ev_net": float(ev_net),
        "mdd_R": float(mdd),
    }


def _ensure_atr_column(df_oof: pd.DataFrame, ohlcv: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Si el OOF parquet no trae 'atr', mergear con OHLCV y calcular Wilder."""
    if "atr" in df_oof.columns and df_oof["atr"].notna().sum() > 0:
        return df_oof
    print("   ('atr' no presente en OOF parquet — calculando desde OHLCV)")
    if not all(c in df_oof.columns for c in ("high", "low", "close")):
        # Necesitamos high/low/close del OHLCV
        df_oof = df_oof.merge(
            ohlcv[["time", "open", "high", "low", "close"]], on="time", how="left"
        )
    atr_arr = wilder_atr(
        df_oof["high"].to_numpy(),
        df_oof["low"].to_numpy(),
        df_oof["close"].to_numpy(),
        period=atr_window,
    )
    df_oof = df_oof.copy()
    df_oof["atr"] = atr_arr
    return df_oof


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────

def _abs_delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(a - b)


def _verdict(recorded: Dict[str, Any], recomputed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tolerancias:
      ev_net    : ≤ 1e-3 R
      ev_gross  : ≤ 1e-3 R
      n_signals : ≤ 2
      prec_TP   : ≤ 0.005
      mdd_R     : ≤ 1.0 R
    """
    d_ev_net = _abs_delta(recorded.get("ev_net", float("nan")),
                          recomputed.get("ev_net", float("nan")))
    d_ev_gross = _abs_delta(recorded.get("ev_gross", float("nan")),
                            recomputed.get("ev_gross", float("nan")))
    d_n_sig = _abs_delta(recorded.get("n_signals", 0),
                         recomputed.get("n_signals", 0))
    d_prec = _abs_delta(recorded.get("prec_TP", float("nan")),
                        recomputed.get("prec_TP", float("nan")))
    d_mdd = _abs_delta(recorded.get("mdd_R", float("nan")),
                       recomputed.get("mdd_R", float("nan")))

    checks = {
        "ev_net":    (d_ev_net,    1e-3),
        "ev_gross":  (d_ev_gross,  1e-3),
        "n_signals": (d_n_sig,     2),
        "prec_TP":   (d_prec,      5e-3),
        "mdd_R":     (d_mdd,       1.0),
    }
    fails = {k: v for k, (v, tol) in checks.items() if not (v <= tol)}
    return {"deltas": {k: v for k, (v, _) in checks.items()},
            "fails": fails,
            "passed": len(fails) == 0}


def _print_side_table(
    label: str, recorded: Dict[str, Any], recomputed: Dict[str, Any]
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"  VALIDACIÓN  {label}")
    print("=" * 78)
    print(f"  {'metric':<14}  {'RECORDED (trial)':>20}  {'RECOMPUTED (specialist)':>26}  {'Δ':>10}")
    print(f"  {'-'*14}  {'-'*20}  {'-'*26}  {'-'*10}")
    rows = [
        ("thr",        recorded.get("thr"),       recomputed.get("thr")),
        ("n_signals",  recorded.get("n_signals"), recomputed.get("n_signals")),
        ("prec_TP",    recorded.get("prec_TP"),   recomputed.get("prec_TP")),
        ("frac_SL",    recorded.get("frac_SL"),   recomputed.get("frac_SL")),
        ("frac_EXP",   recorded.get("frac_EXP"),  recomputed.get("frac_EXP")),
        ("ev_gross",   recorded.get("ev_gross"),  recomputed.get("ev_gross")),
        ("ev_net",     recorded.get("ev_net"),    recomputed.get("ev_net")),
        ("mdd_R",      recorded.get("mdd_R"),     recomputed.get("mdd_R")),
    ]
    for name, a, b in rows:
        a_str = f"{a:.4f}" if isinstance(a, float) else f"{a}"
        b_str = f"{b:.4f}" if isinstance(b, float) else f"{b}"
        d = _abs_delta(a if a is not None else float("nan"),
                       b if b is not None else float("nan"))
        d_str = f"{d:.4f}" if np.isfinite(d) else "n/a"
        print(f"  {name:<14}  {a_str:>20}  {b_str:>26}  {d_str:>10}")

    v = _verdict(recorded, recomputed)
    print()
    if v["passed"]:
        print(f"  ✅ PASS  — reentreno reproduce el trial original dentro de tolerancias.")
    else:
        print(f"  ❌ FAIL  — métricas fuera de tolerancia:")
        for k, delta in v["fails"].items():
            print(f"      {k:<10}  Δ={delta:.4f}")
        print("  Posibles causas: seed no fija, orden de folds distinto, "
              "diferencia en epochs efectivos por early stopping, calibración "
              "isotónica con datos ligeramente distintos.")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv(args, time_min: pd.Timestamp, time_max: pd.Timestamp) -> pd.DataFrame:
    if args.from_db:
        from mimo.data_managers.databases import Database
        from mimo.data_managers.data_manager import DataManager
        bar_min = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}.get(args.base_tf, 5)
        buf_min = args.db_buffer_bars * bar_min
        from_dt = (time_min - pd.Timedelta(minutes=buf_min)).normalize()
        to_dt = (time_max + pd.Timedelta(minutes=buf_min)).normalize() + pd.Timedelta(days=1)
        print(f"📂 OHLCV: BD ({args.base_tf}, {from_dt} → {to_dt})")
        db = Database()
        resample_arg = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
        dm = DataManager.from_database_historical_2(
            db, from_date=str(from_dt), to_date=str(to_dt), resample=resample_arg
        )
        return dm.df[["time", "open", "high", "low", "close"]].copy()
    if not args.ohlcv:
        raise SystemExit("Pasa --ohlcv <path> o --from-db")
    p = Path(args.ohlcv)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    return df[["time", "open", "high", "low", "close"]].copy()


def _locate_oof_parquet(specialist_dir: Path, release: str) -> Path:
    """Busca el oof_<release>_<side>.parquet del specialist."""
    candidates = list(specialist_dir.glob(f"oof_{release}_*.parquet"))
    if not candidates:
        candidates = list(specialist_dir.glob("oof_*.parquet"))
    if not candidates:
        raise SystemExit(f"❌ No encontré oof parquet en {specialist_dir}")
    if len(candidates) > 1:
        # Preferir el de multitask
        for c in candidates:
            if "multitask" in c.name:
                return c
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-per-side-json", required=True)
    ap.add_argument("--specialist-long-dir", default=None)
    ap.add_argument("--specialist-short-dir", default=None)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--release", required=True)
    ap.add_argument("--ohlcv", default=None)
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--db-buffer-bars", type=int, default=200)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--tp", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    args = ap.parse_args()

    json_path = Path(args.best_per_side_json)
    if not json_path.exists():
        raise SystemExit(f"❌ No existe: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    sides_to_check = []
    if args.side in ("long", "both"):
        if not args.specialist_long_dir:
            raise SystemExit("❌ --specialist-long-dir requerido para side=long/both")
        sides_to_check.append(("long", "ev_long", "oof_proba_long_cal", True,
                               Path(args.specialist_long_dir)))
    if args.side in ("short", "both"):
        if not args.specialist_short_dir:
            raise SystemExit("❌ --specialist-short-dir requerido para side=short/both")
        sides_to_check.append(("short", "ev_short", "oof_proba_short_cal", False,
                               Path(args.specialist_short_dir)))

    print("\n📂 Best-per-side JSON :", json_path)
    overall_pass = True
    results = {}

    for side_label, ev_key, proba_col, side_is_long, sp_dir in sides_to_check:
        print(f"\n{'=' * 78}\n   Procesando SPECIALIST {side_label.upper()}: {sp_dir}\n{'=' * 78}")
        if not sp_dir.exists():
            print(f"   ❌ Specialist dir no existe: {sp_dir}")
            overall_pass = False
            continue
        try:
            oof_parquet = _locate_oof_parquet(sp_dir, args.release)
        except SystemExit as e:
            print(f"   {e}")
            overall_pass = False
            continue
        print(f"   OOF parquet: {oof_parquet}")
        df_oof = pd.read_parquet(oof_parquet)
        df_oof["time"] = pd.to_datetime(df_oof["time"])
        if proba_col not in df_oof.columns:
            print(f"   ❌ Columna '{proba_col}' no en OOF parquet. "
                  f"Cols={list(df_oof.columns)}")
            overall_pass = False
            continue

        # Si falta atr, traer OHLCV
        if "atr" not in df_oof.columns or df_oof["atr"].isna().all():
            ohlcv = _load_ohlcv(args, df_oof["time"].min(), df_oof["time"].max())
            ohlcv["time"] = pd.to_datetime(ohlcv["time"])
            ohlcv = (ohlcv.dropna(subset=["close"])
                          .sort_values("time")
                          .drop_duplicates("time")
                          .reset_index(drop=True))
            df_oof = _ensure_atr_column(df_oof, ohlcv, atr_window=args.atr_window)

        # Recorded del trial
        sub = payload.get(f"top_{side_label}", [])
        if not sub:
            print(f"   ❌ top_{side_label} vacío en JSON")
            overall_pass = False
            continue
        recorded = sub[0].get(ev_key, {})
        trial_num = sub[0].get("trial", "?")
        print(f"   Trial origen: #{trial_num}")
        locked_thr = float(recorded.get("thr"))
        print(f"   thr lockeado: {locked_thr:.4f}")

        # Recompute en specialist
        recomputed = _recompute_metrics_at_locked_thr(
            df_oof,
            proba_col=proba_col,
            side_is_long=side_is_long,
            locked_thr=locked_thr,
            horizon=args.horizon,
            tp_mult=args.tp,
            sl_mult=args.sl,
            cost=args.cost,
            max_drawdown_R=args.max_drawdown_R,
        )

        v = _print_side_table(side_label.upper(), recorded, recomputed)
        results[side_label] = {
            "trial": trial_num,
            "thr_locked": locked_thr,
            "recorded": recorded,
            "recomputed": recomputed,
            "verdict": v,
        }
        if not v["passed"]:
            overall_pass = False

    # Veredicto global
    print("\n" + "=" * 78)
    print("  VEREDICTO GLOBAL")
    print("=" * 78)
    if overall_pass:
        print("  ✅ Todos los specialists reproducen el trial original.")
        print("     Puedes seguir adelante con drift validation.")
    else:
        print("  ❌ Al menos un specialist NO reproduce el trial original.")
        print("     Revisa la configuración antes de drift validation:")
        print("       - seed: confirma que self.seed=42 esté siendo respetada en"
              " todos los componentes (TF, numpy, sklearn).")
        print("       - early stopping: si difiere best_epoch por fold, las"
              " predicciones cambian. Comprueba que oof_epochs y oof_patience"
              " coinciden con las del trial original.")
        print("       - calibración isotónica: si la fit usa shuffle implícito,"
              " puede haber drift mínimo.")
        print("       - feature_masks o columnas: confirma mismas columnas y"
              " mismo orden.")


if __name__ == "__main__":
    main()
