#!/usr/bin/env python3
"""
select_thresholds_from_tail.py

Reemplaza el threshold por F1 elegido en `recalibrate_deploy_multitask` por
uno basado en EV-net (R-multiple neto de costes), replayando triple barrier
sobre la cola final usada para recalibrar las isotónicas.

Motivación
──────────
`recalibrate_deploy_multitask` (v6) selecciona el threshold maximizando F1
sobre la binarización de la tail. Con base rate ~7-8% F1 colapsa a thresholds
muy bajos (~0.10-0.13), que en producción (con costes y tail con sesgo
EXPIRE positivo) producen sobre-trading y EV mediocre. El threshold óptimo
EV-net del trial original suele caer en ~0.20-0.30.

Este script:
  1. Carga `data/deploy_calibration_tail_<release>_<side>.parquet` (escrito
     por v6 multitask).
  2. Carga OHLCV (BD vía DataManager o parquet/CSV).
  3. Calcula ATR Wilder.
  4. Replay triple barrier sobre TODAS las filas de la tail; cada una recibe
     outcome ∈ {TP, SL, EXPIRE, INVALID} y R-multiple realizado.
  5. Sweep de thresholds; para cada uno calcula EV_net = mean_R - cost.
  6. Elige el thr con EV_net máximo respetando --min-signals.
  7. Sobrescribe `_meta.selected_threshold` en
     `percentiles_<release>_<side>.json` y reporta la diff.

NOTA: la tail es ~21 días por defecto (DEPLOY_CALIB_DAYS) → ~1500 muestras.
Con --min-signals=30 hay margen de elección razonable, pero el threshold
final SIEMPRE deberá validarse en paper trading.

Uso
───
  python -m mimo.oof.select_thresholds_from_tail \
    --release 202500 \
    --deploy-dir artifacts/202500/oof/deploy_full_v6 \
    --side both --from-db --base-tf 5min \
    --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
    --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
    --cost 0.05 --min-signals 30
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

from mimo.oof.empirical_breakeven import simulate_outcomes, wilder_atr


def _load_tail(deploy_dir: Path, release: str, side: str) -> pd.DataFrame:
    p = deploy_dir / "data" / f"deploy_calibration_tail_{release}_{side}.parquet"
    if not p.exists():
        raise SystemExit(f"❌ Tail parquet no existe: {p}")
    df = pd.read_parquet(p)
    if "time" not in df.columns or "oof_proba_cal" not in df.columns:
        raise SystemExit(
            f"❌ {p} no tiene columnas requeridas (time, oof_proba_cal). "
            f"Cols={list(df.columns)}"
        )
    df["time"] = pd.to_datetime(df["time"])
    return df


def _load_ohlcv_from_db(time_min, time_max, base_tf: str, buffer_bars: int = 200) -> pd.DataFrame:
    from mimo.data_managers.databases import Database
    from mimo.data_managers.data_manager import DataManager

    bar_min = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}.get(base_tf, 5)
    buf_min = buffer_bars * bar_min
    from_dt = (time_min - pd.Timedelta(minutes=buf_min)).normalize()
    to_dt = (time_max + pd.Timedelta(minutes=buf_min)).normalize() + pd.Timedelta(days=1)
    print(f"📂 OHLCV: BD ({base_tf}, {from_dt} → {to_dt})")
    db = Database()
    resample_arg = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=str(from_dt), to_date=str(to_dt), resample=resample_arg
    )
    return dm.df[["time", "open", "high", "low", "close"]].copy()


def _load_ohlcv_from_file(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix == ".csv":
        ohlcv = pd.read_csv(p)
    else:
        ohlcv = pd.read_parquet(p)
    ohlcv.columns = [c.lower() for c in ohlcv.columns]
    return ohlcv


def _sweep_ev(
    j: pd.DataFrame,
    *,
    proba_col: str,
    thr_lo: float,
    thr_hi: float,
    n_points: int,
    min_signals: int,
    cost: float,
) -> pd.DataFrame:
    rows = []
    thrs = np.linspace(thr_lo, thr_hi, n_points)
    for thr in thrs:
        mask = j[proba_col].to_numpy() >= thr
        sig = int(mask.sum())
        if sig < min_signals:
            continue
        sub = j[mask]
        valid = sub[sub["outcome"] != "INVALID"]
        if len(valid) == 0:
            continue
        n_tp = int((valid["outcome"] == "TP").sum())
        n_sl = int((valid["outcome"] == "SL").sum())
        n_exp = int((valid["outcome"] == "EXPIRE").sum())
        mean_r = float(valid["R_multiple"].mean())
        ev_net = mean_r - cost
        rows.append({
            "thr": float(thr),
            "sig": sig,
            "n_TP": n_tp,
            "n_SL": n_sl,
            "n_EXP": n_exp,
            "prec_TP": n_tp / len(valid),
            "EV_gross": mean_r,
            "EV_net": ev_net,
        })
    return pd.DataFrame(rows)


def _process_side(
    side: str,
    *,
    deploy_dir: Path,
    release: str,
    ohlcv: pd.DataFrame,
    tp: float,
    sl: float,
    horizon: int,
    atr_window: int,
    cost: float,
    thr_lo: float,
    thr_hi: float,
    n_points: int,
    min_signals: int,
    proba_col: str,
    apply: bool,
) -> Optional[Dict]:
    print("\n" + "═" * 72)
    print(f"  SIDE = {side.upper()}  (tp={tp}, sl={sl}, h={horizon})")
    print("═" * 72)

    tail = _load_tail(deploy_dir, release, side)
    print(f"  tail rows: {len(tail):,}  rango: {tail['time'].min()} → {tail['time'].max()}")

    ohlcv2 = ohlcv.copy()
    ohlcv2["__row__"] = np.arange(len(ohlcv2))
    j = tail.merge(ohlcv2[["time", "__row__"]], on="time", how="inner")
    print(f"  alineadas (tail ∩ ohlcv): {len(j):,} / {len(tail):,}")
    if len(j) == 0:
        print(f"  ❌ sin coincidencias temporales para side={side}")
        return None

    atr = wilder_atr(
        ohlcv2["high"].to_numpy(),
        ohlcv2["low"].to_numpy(),
        ohlcv2["close"].to_numpy(),
        period=atr_window,
    )

    out = simulate_outcomes(
        j["__row__"].to_numpy(),
        ohlcv2["high"].to_numpy(),
        ohlcv2["low"].to_numpy(),
        ohlcv2["close"].to_numpy(),
        atr,
        horizon=horizon,
        tp_mult=tp,
        sl_mult=sl,
        side_is_long=(side == "long"),
    )
    j = pd.concat([j.reset_index(drop=True), out], axis=1)

    counts = j["outcome"].value_counts()
    total = len(j)
    print(f"  outcomes globales: " + ", ".join(
        f"{k}={int(counts.get(k, 0))} ({counts.get(k, 0) / total:.3f})"
        for k in ("TP", "SL", "EXPIRE", "INVALID")
    ))

    res = _sweep_ev(
        j,
        proba_col=proba_col,
        thr_lo=thr_lo, thr_hi=thr_hi, n_points=n_points,
        min_signals=min_signals, cost=cost,
    )
    if res.empty:
        print(f"  ❌ ningún threshold con >= {min_signals} señales en [{thr_lo}, {thr_hi}]")
        return None

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print("\n  SWEEP:")
    print(res.round(4).to_string(index=False))

    best = res.loc[res["EV_net"].idxmax()]
    print(f"\n  🏆 BEST EV_net  thr={best['thr']:.4f}  EV_net={best['EV_net']:+.4f}R  "
          f"sig={int(best['sig'])}  prec_TP={best['prec_TP']:.3f}")

    pct_path = deploy_dir / f"percentiles_{release}_{side}.json"
    if not pct_path.exists():
        print(f"  ⚠️  no existe {pct_path}, no se actualiza policy.")
        return {"side": side, "best_thr": float(best["thr"]), "applied": False}

    with pct_path.open("r", encoding="utf-8") as f:
        pct = json.load(f)
    meta = pct.setdefault("_meta", {})
    old_thr = float(meta.get("selected_threshold", float("nan")))
    new_thr = float(best["thr"])
    print(f"  thr antiguo (F1)        : {old_thr:.4f}")
    print(f"  thr nuevo  (EV-net)     : {new_thr:.4f}  Δ={new_thr - old_thr:+.4f}")

    if apply:
        meta["selected_threshold_old_f1"] = old_thr
        meta["selected_threshold"] = new_thr
        meta["threshold_source"] = "ev_net_tail_replay"
        meta["threshold_replay"] = {
            "tp_mult": float(tp), "sl_mult": float(sl), "horizon": int(horizon),
            "cost_per_signal": float(cost), "min_signals": int(min_signals),
            "n_aligned": int(len(j)), "ev_net_R": float(best["EV_net"]),
            "n_signals_at_thr": int(best["sig"]),
            "prec_tp_at_thr": float(best["prec_TP"]),
        }
        with pct_path.open("w", encoding="utf-8") as f:
            json.dump(pct, f, indent=2)
        print(f"  ✅ percentiles_{release}_{side}.json actualizado.")
    else:
        print(f"  (--dry-run: no se sobrescribe el JSON)")

    return {
        "side": side,
        "old_thr_f1": old_thr,
        "new_thr_evnet": new_thr,
        "ev_net_R": float(best["EV_net"]),
        "n_signals": int(best["sig"]),
        "applied": apply,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--deploy-dir", required=True,
                    help="Directorio de deploy donde están los percentiles "
                         "y data/deploy_calibration_tail_*.parquet.")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")

    # OHLCV source
    ap.add_argument("--from-db", action="store_true",
                    help="Carga OHLCV desde la BD vía DataManager.")
    ap.add_argument("--ohlcv", default=None,
                    help="parquet/csv con OHLCV 5min (alternativa a --from-db).")
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--db-buffer-bars", type=int, default=200)

    # Triple barrier (per side; defaults idénticos a 202500)
    ap.add_argument("--tp-long", type=float, default=2.0)
    ap.add_argument("--sl-long", type=float, default=0.8)
    ap.add_argument("--horizon-long", type=int, default=3)
    ap.add_argument("--tp-short", type=float, default=2.0)
    ap.add_argument("--sl-short", type=float, default=0.8)
    ap.add_argument("--horizon-short", type=int, default=3)
    ap.add_argument("--atr-window", type=int, default=14)

    # Sweep params
    ap.add_argument("--cost", type=float, default=0.05,
                    help="Coste por señal en R (round-trip). default 0.05R.")
    ap.add_argument("--thr-lo", type=float, default=0.10)
    ap.add_argument("--thr-hi", type=float, default=0.45)
    ap.add_argument("--n-points", type=int, default=70)
    ap.add_argument("--min-signals", type=int, default=30)
    ap.add_argument("--proba-col", default="oof_proba_cal",
                    choices=["oof_proba_cal", "oof_proba_raw"])

    ap.add_argument("--dry-run", action="store_true",
                    help="Solo reporta; no sobrescribe el JSON.")
    args = ap.parse_args()

    deploy_dir = Path(args.deploy_dir)
    if not deploy_dir.exists():
        raise SystemExit(f"❌ --deploy-dir no existe: {deploy_dir}")

    sides = ["long", "short"] if args.side == "both" else [args.side]

    # Determinamos rango de OHLCV uniendo las tails de los sides solicitados
    tails = []
    for s in sides:
        try:
            tails.append(_load_tail(deploy_dir, args.release, s))
        except SystemExit as e:
            print(str(e))
    if not tails:
        raise SystemExit("❌ no pude cargar ninguna tail.")
    t_all = pd.concat(tails, axis=0)["time"]
    time_min, time_max = t_all.min(), t_all.max()

    if args.from_db:
        ohlcv = _load_ohlcv_from_db(time_min, time_max, args.base_tf, args.db_buffer_bars)
    else:
        if not args.ohlcv:
            raise SystemExit("Pasa --from-db o --ohlcv <path>.")
        print(f"📂 OHLCV: {args.ohlcv}")
        ohlcv = _load_ohlcv_from_file(args.ohlcv)

    for c in ("time", "open", "high", "low", "close"):
        if c not in ohlcv.columns:
            raise SystemExit(f"❌ '{c}' no en OHLCV. Cols={list(ohlcv.columns)}")
    ohlcv["time"] = pd.to_datetime(ohlcv["time"])
    ohlcv = (
        ohlcv.dropna(subset=["close"])
        .sort_values("time")
        .drop_duplicates(subset="time", keep="first")
        .reset_index(drop=True)
    )
    print(f"  ohlcv rows={len(ohlcv):,}  rango={ohlcv['time'].min()} → {ohlcv['time'].max()}")

    summary = {}
    for s in sides:
        if s == "long":
            tp, sl, hz = args.tp_long, args.sl_long, args.horizon_long
        else:
            tp, sl, hz = args.tp_short, args.sl_short, args.horizon_short
        out = _process_side(
            s,
            deploy_dir=deploy_dir, release=args.release, ohlcv=ohlcv,
            tp=tp, sl=sl, horizon=hz, atr_window=args.atr_window,
            cost=args.cost, thr_lo=args.thr_lo, thr_hi=args.thr_hi,
            n_points=args.n_points, min_signals=args.min_signals,
            proba_col=args.proba_col, apply=not args.dry_run,
        )
        if out is not None:
            summary[s] = out

    print("\n" + "╔" + "═" * 70 + "╗")
    print("║  RESUMEN".ljust(71) + "║")
    print("╚" + "═" * 70 + "╝")
    for s, info in summary.items():
        if "old_thr_f1" in info:
            print(f"  {s.upper():5s}  F1 thr={info['old_thr_f1']:.4f} → "
                  f"EV-net thr={info['new_thr_evnet']:.4f}  "
                  f"(EV_net={info['ev_net_R']:+.4f}R, sig={info['n_signals']}, "
                  f"applied={info['applied']})")
    if args.dry_run:
        print("\n  ⚠️  --dry-run: nada se ha persistido. Re-ejecuta sin --dry-run para aplicar.")


if __name__ == "__main__":
    main()
