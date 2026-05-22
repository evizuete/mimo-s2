#!/usr/bin/env python3
"""
temporal_weekly_stability.py

Reporta EV_net, prec_TP, sig, MDD por SEMANA ISO sobre el holdout, replayando
el triple barrier al threshold óptimo del best_trial Optuna (o flag manual).

Pensado para detectar decay temporal: si EV_net cae monotónicamente semana a
semana, producción es arriesgada incluso si el agregado holdout luce positivo.

Inputs:
  --holdout-preds-long  parquet con time/state/y_true/y_pred_cal del lado LONG
  --holdout-preds-short ídem SHORT
  --ohlcv | --from-db   misma semántica que empirical_breakeven.py
  --thr-long, --thr-short  threshold operativo (del best_trial o manual)
  --tp, --sl, --horizon    barriers (default 2.0 / 0.8 / 3 para 202500)
  --cost                cost per signal en R (default 0.05)

Output:
  · Tabla por semana ISO con LONG y SHORT lado a lado.
  · Resumen global: % semanas EV>0, pendiente lineal del EV (decay test).
  · Worst week por lado.

Usage:
  python -m mimo.oof.temporal_weekly_stability \
    --holdout-preds-long  artifacts/202500/oof/<tag>/data/holdout_predictions_202500_long.parquet \
    --holdout-preds-short artifacts/202500/oof/<tag>/data/holdout_predictions_202500_short.parquet \
    --from-db --base-tf 5min \
    --thr-long 0.198 --thr-short 0.221 \
    --tp 2.0 --sl 0.8 --horizon 3 --cost 0.05
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from mimo.oof.empirical_breakeven import wilder_atr, simulate_outcomes


def _max_drawdown_R(r_signal_sorted: np.ndarray) -> float:
    if r_signal_sorted.size == 0:
        return 0.0
    eq = np.cumsum(r_signal_sorted)
    peak = np.maximum.accumulate(eq)
    dd = peak - eq
    return float(dd.max())


def _weekly_stats(
    df_join: pd.DataFrame,
    *,
    proba_col: str,
    thr: float,
    cost: float,
    side_label: str,
) -> pd.DataFrame:
    pred = (df_join[proba_col].to_numpy() >= thr) & (df_join["outcome"].to_numpy() != "INVALID")
    sub = df_join.loc[pred].copy()
    if sub.empty:
        return pd.DataFrame()

    sub["iso_year"] = sub["time"].dt.isocalendar().year
    sub["iso_week"] = sub["time"].dt.isocalendar().week
    sub["week_key"] = sub["iso_year"].astype(str) + "-W" + sub["iso_week"].astype(str).str.zfill(2)

    rows = []
    for week_key, grp in sub.sort_values("time").groupby("week_key", sort=True):
        week_start = grp["time"].min().normalize()
        week_end = grp["time"].max().normalize()
        n_sig = len(grp)
        n_tp = int((grp["outcome"] == "TP").sum())
        n_sl = int((grp["outcome"] == "SL").sum())
        n_exp = int((grp["outcome"] == "EXPIRE").sum())
        ev_gross = float(grp["R_multiple"].mean())
        ev_net = ev_gross - cost
        r_net_sorted = grp.sort_values("time")["R_multiple"].to_numpy() - cost
        mdd = _max_drawdown_R(r_net_sorted)
        rows.append({
            "week": week_key,
            "week_start": week_start.date(),
            "week_end": week_end.date(),
            "side": side_label,
            "sig": n_sig,
            "n_TP": n_tp,
            "n_SL": n_sl,
            "n_EXP": n_exp,
            "prec_TP": n_tp / n_sig,
            "frac_SL": n_sl / n_sig,
            "frac_EXP": n_exp / n_sig,
            "EV_gross": ev_gross,
            "EV_net": ev_net,
            "mdd_R": mdd,
            "deployable": ev_net > 0,
        })
    return pd.DataFrame(rows)


def _decay_test(weekly: pd.DataFrame, side: str) -> dict:
    """Regresión lineal de EV_net vs week_index. Devuelve pendiente y p-value."""
    df = weekly[weekly["side"] == side].sort_values("week").reset_index(drop=True)
    if len(df) < 4:
        return {"slope_per_week": float("nan"), "trend": "insufficient_weeks"}
    x = np.arange(len(df), dtype=float)
    y = df["EV_net"].to_numpy()
    n = len(x)
    x_mean, y_mean = x.mean(), y.mean()
    cov = ((x - x_mean) * (y - y_mean)).sum() / n
    var_x = ((x - x_mean) ** 2).sum() / n
    slope = cov / var_x if var_x > 0 else 0.0
    intercept = y_mean - slope * x_mean
    y_pred = slope * x + intercept
    ss_res = ((y - y_pred) ** 2).sum()
    ss_tot = ((y - y_mean) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    if slope < -0.005 and r2 > 0.20:
        trend = "DECAY (consistente)"
    elif slope < -0.005:
        trend = "decay leve (no consistente)"
    elif slope > 0.005:
        trend = "MEJORA"
    else:
        trend = "estable"
    return {
        "slope_per_week": float(slope),
        "intercept": float(intercept),
        "r2": float(r2),
        "n_weeks": int(n),
        "trend": trend,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-preds-long", required=True)
    ap.add_argument("--holdout-preds-short", required=True)
    ap.add_argument("--ohlcv", default=None,
                    help="parquet/csv OHLCV (alternativa a --from-db)")
    ap.add_argument("--from-db", action="store_true",
                    help="Cargar OHLCV desde la BD usando DataManager.")
    ap.add_argument("--db-buffer-bars", type=int, default=200)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--proba-col", default="y_pred_cal",
                    choices=["y_pred_cal", "y_pred_raw"])
    ap.add_argument("--thr-long", type=float, required=True,
                    help="Threshold operativo LONG (del best_trial)")
    ap.add_argument("--thr-short", type=float, required=True,
                    help="Threshold operativo SHORT")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--tp", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--cost", type=float, default=0.05,
                    help="Coste round-trip en R per signal")
    ap.add_argument("--save-csv", default=None,
                    help="Si se pasa, guarda la tabla semanal en CSV")
    args = ap.parse_args()

    # ── load preds ──────────────────────────────────────────────────────
    print(f"📂 LONG  preds: {args.holdout_preds_long}")
    pl = pd.read_parquet(args.holdout_preds_long)
    pl["time"] = pd.to_datetime(pl["time"])
    print(f"   rows={len(pl):,}")

    print(f"📂 SHORT preds: {args.holdout_preds_short}")
    ps = pd.read_parquet(args.holdout_preds_short)
    ps["time"] = pd.to_datetime(ps["time"])
    print(f"   rows={len(ps):,}")

    # ── load ohlcv ──────────────────────────────────────────────────────
    if args.from_db:
        from mimo.data_managers.databases import Database
        from mimo.data_managers.data_manager import DataManager
        bar_min = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}.get(args.base_tf, 5)
        buf_min = args.db_buffer_bars * bar_min
        from_dt = (
            min(pl["time"].min(), ps["time"].min())
            - pd.Timedelta(minutes=buf_min)
        ).normalize()
        to_dt = (
            max(pl["time"].max(), ps["time"].max())
            + pd.Timedelta(minutes=buf_min)
        ).normalize() + pd.Timedelta(days=1)
        print(f"📂 ohlcv: BD ({args.base_tf}, {from_dt} → {to_dt})")
        db = Database()
        resample_arg = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
        dm = DataManager.from_database_historical_2(
            db, from_date=str(from_dt), to_date=str(to_dt), resample=resample_arg
        )
        ohlcv = dm.df[["time", "open", "high", "low", "close"]].copy()
    else:
        if not args.ohlcv:
            raise SystemExit("Pasa --ohlcv <path> o --from-db")
        ohlcv_path = Path(args.ohlcv)
        if ohlcv_path.suffix == ".csv":
            ohlcv = pd.read_csv(ohlcv_path)
        else:
            ohlcv = pd.read_parquet(ohlcv_path)
        ohlcv.columns = [c.lower() for c in ohlcv.columns]
    for c in ("time", "open", "high", "low", "close"):
        if c not in ohlcv.columns:
            raise SystemExit(f"❌ '{c}' no en OHLCV")
    ohlcv["time"] = pd.to_datetime(ohlcv["time"])
    ohlcv = (ohlcv.dropna(subset=["close"])
                  .sort_values("time")
                  .drop_duplicates("time")
                  .reset_index(drop=True))
    print(f"   ohlcv rows={len(ohlcv):,}  rango={ohlcv['time'].min()} → {ohlcv['time'].max()}")

    # ── ATR + simulate barriers para todas las filas alineadas ──────────
    atr = wilder_atr(
        ohlcv["high"].to_numpy(),
        ohlcv["low"].to_numpy(),
        ohlcv["close"].to_numpy(),
        period=args.atr_window,
    )
    ohlcv["__row__"] = np.arange(len(ohlcv))
    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    close = ohlcv["close"].to_numpy()

    def _sim_join(preds: pd.DataFrame, side_is_long: bool) -> pd.DataFrame:
        j = preds.merge(ohlcv[["time", "__row__"]], on="time", how="inner")
        out = simulate_outcomes(
            j["__row__"].to_numpy(),
            high, low, close, atr,
            horizon=args.horizon,
            tp_mult=args.tp,
            sl_mult=args.sl,
            side_is_long=side_is_long,
        )
        return pd.concat([j.reset_index(drop=True), out], axis=1)

    print("\n🔬 Simulando barreras (LONG)...")
    j_long = _sim_join(pl, side_is_long=True)
    print(f"   alineadas: {len(j_long):,}")
    print("🔬 Simulando barreras (SHORT)...")
    j_short = _sim_join(ps, side_is_long=False)
    print(f"   alineadas: {len(j_short):,}")

    # ── weekly stats por lado ───────────────────────────────────────────
    weekly_long = _weekly_stats(
        j_long, proba_col=args.proba_col,
        thr=args.thr_long, cost=args.cost, side_label="LONG",
    )
    weekly_short = _weekly_stats(
        j_short, proba_col=args.proba_col,
        thr=args.thr_short, cost=args.cost, side_label="SHORT",
    )

    weekly = pd.concat([weekly_long, weekly_short], ignore_index=True)
    if weekly.empty:
        print("❌ No se generaron señales semanales. Revisa thresholds.")
        return

    # ── tabla pivot por semana (LONG y SHORT lado a lado) ───────────────
    pivot = weekly.pivot_table(
        index=["week", "week_start", "week_end"],
        columns="side",
        values=["sig", "prec_TP", "EV_net", "mdd_R"],
        fill_value=np.nan,
    ).reset_index()
    # aplanar nombres MultiIndex
    pivot.columns = [
        "_".join([str(x) for x in c if x != ""]).strip("_")
        for c in pivot.columns.to_flat_index()
    ]
    pivot = pivot.sort_values("week").reset_index(drop=True)

    print("\n" + "=" * 90)
    print("  TABLA POR SEMANA (LONG vs SHORT)")
    print("=" * 90)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.float_format", lambda x: f"{x:+.4f}" if pd.notna(x) else "  -  ")
    print(pivot.to_string(index=False))

    # ── decay test por lado ─────────────────────────────────────────────
    long_decay = _decay_test(weekly, "LONG")
    short_decay = _decay_test(weekly, "SHORT")

    print("\n" + "=" * 90)
    print("  TEST DE DECAY TEMPORAL (regresión lineal EV_net vs semana)")
    print("=" * 90)
    print(f"  LONG  : slope={long_decay.get('slope_per_week', float('nan')):+.5f} R/semana  "
          f"r²={long_decay.get('r2', 0):.3f}  n={long_decay.get('n_weeks', 0)}  "
          f"→ {long_decay['trend']}")
    print(f"  SHORT : slope={short_decay.get('slope_per_week', float('nan')):+.5f} R/semana  "
          f"r²={short_decay.get('r2', 0):.3f}  n={short_decay.get('n_weeks', 0)}  "
          f"→ {short_decay['trend']}")

    # ── resumen agregado ────────────────────────────────────────────────
    def _summary(side: str) -> dict:
        d = weekly[weekly["side"] == side]
        if d.empty:
            return {}
        n = len(d)
        n_pos = int((d["EV_net"] > 0).sum())
        worst = d.loc[d["EV_net"].idxmin()]
        best = d.loc[d["EV_net"].idxmax()]
        return {
            "n_weeks": n,
            "weeks_deployable": n_pos,
            "pct_deployable": n_pos / n,
            "median_EV_net": float(d["EV_net"].median()),
            "mean_EV_net": float(d["EV_net"].mean()),
            "total_R_net": float((d["EV_net"] * d["sig"]).sum()),
            "total_signals": int(d["sig"].sum()),
            "worst_week": worst["week"],
            "worst_EV_net": float(worst["EV_net"]),
            "best_week": best["week"],
            "best_EV_net": float(best["EV_net"]),
            "max_mdd_R": float(d["mdd_R"].max()),
        }

    summary_long = _summary("LONG")
    summary_short = _summary("SHORT")

    print("\n" + "=" * 90)
    print("  RESUMEN AGREGADO")
    print("=" * 90)
    for side, s in [("LONG", summary_long), ("SHORT", summary_short)]:
        if not s:
            print(f"  {side}: (sin datos)")
            continue
        print(f"\n  {side}:")
        print(f"    semanas analizadas        : {s['n_weeks']}")
        print(f"    semanas con EV_net > 0    : {s['weeks_deployable']} ({s['pct_deployable']*100:.0f}%)")
        print(f"    EV_net mediano por semana : {s['median_EV_net']:+.4f}R")
        print(f"    EV_net medio por semana   : {s['mean_EV_net']:+.4f}R")
        print(f"    Total R_net en holdout    : {s['total_R_net']:+.2f}R   ({s['total_signals']} señales)")
        print(f"    Mejor semana              : {s['best_week']}  EV_net={s['best_EV_net']:+.4f}R")
        print(f"    Peor semana               : {s['worst_week']}  EV_net={s['worst_EV_net']:+.4f}R")
        print(f"    Max MDD intra-semana      : {s['max_mdd_R']:.1f}R")

    # ── veredicto rápido ────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print("  VEREDICTO RÁPIDO")
    print("=" * 90)
    for side, s, decay in [("LONG", summary_long, long_decay),
                            ("SHORT", summary_short, short_decay)]:
        if not s:
            continue
        flags = []
        if s["pct_deployable"] < 0.55:
            flags.append(f"<55% semanas deployable ({s['pct_deployable']*100:.0f}%)")
        if "DECAY" in decay.get("trend", ""):
            flags.append(f"DECAY temporal ({decay['slope_per_week']:+.4f}R/sem, r²={decay['r2']:.2f})")
        if s["worst_EV_net"] < -0.10:
            flags.append(f"peor semana muy negativa ({s['worst_EV_net']:+.3f}R)")
        if s["max_mdd_R"] > 50:
            flags.append(f"MDD intra-semana alto ({s['max_mdd_R']:.0f}R)")
        if flags:
            print(f"  ⚠️  {side}: " + "  |  ".join(flags))
        else:
            print(f"  ✅ {side}: estable")

    if args.save_csv:
        out = Path(args.save_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        weekly.to_csv(out, index=False)
        print(f"\n📁 Tabla semanal guardada: {out}")


if __name__ == "__main__":
    main()
