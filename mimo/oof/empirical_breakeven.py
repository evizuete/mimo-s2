#!/usr/bin/env python3
"""
empirical_breakeven.py

Computes the EMPIRICAL break-even and EV per signal on the holdout, replaying
the triple barrier on each predicted-positive entry and categorising outcomes
into TP / SL / EXPIRE.

Why: theoretical BE = SL / (TP + SL) = 0.286 for tp=2.0, sl=0.8 assumes that
all false positives hit SL. With triple barrier at h=3 a large fraction of
predicted-positives expire without touching either barrier (~0R), which makes
the real BE substantially lower.

Inputs:
  --holdout-preds : parquet from save_holdout_predictions (time, state, y_true,
                    y_pred_cal, y_pred_raw, side)
  --ohlcv         : parquet/csv with the 5min OHLCV that covers the holdout
                    window. Must include columns: time, open, high, low, close.
  --horizon       : barrier horizon in bars (default 3)
  --tp / --sl     : barrier multipliers in ATR units (default 2.0 / 0.8 for
                    202300; we assume single tp/sl — regime variation is mild
                    in 202300 so this is a reasonable approximation).
  --atr-window    : ATR Wilder period (default 14)

What it does:
  1. Compute ATR-14 over the OHLCV.
  2. Inner-join with holdout_preds on `time`.
  3. For each row, simulate triple barrier h bars ahead:
       LONG  : tp_level = entry + tp*ATR ; sl_level = entry - sl*ATR
       SHORT : tp_level = entry - tp*ATR ; sl_level = entry + sl*ATR
       outcome = TP / SL / EXPIRE (and realised R-multiple).
  4. Sweep thresholds. At each thr report:
       precision, frac_TP, frac_SL, frac_EXPIRE,
       mean_R_expire, total mean_R per signal (= EV),
       BE empírico = SL × frac_SL_pred / TP / (frac_SL_pred + 1)
       (or the trivial: prec_needed_for_EV0 given the observed SL/EXPIRE mix
        among predicted positives).
  5. Pick the threshold that maximises EV with sig_rate >= min_sig.

Usage:
  python -m mimo.oof.empirical_breakeven \
    --holdout-preds artifacts/202300/oof/<exp_tag>/data/holdout_predictions_202300_long.parquet \
    --ohlcv data/xauusd_5min.parquet \
    --tp 2.0 --sl 0.8 --horizon 3
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# ----- ATR (Wilder) -----------------------------------------------------------

def wilder_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray,
               period: int = 14) -> np.ndarray:
    n = len(close)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        tr[i] = max(
            high[i] - low[i],
            abs(high[i] - close[i - 1]),
            abs(low[i] - close[i - 1]),
        )
    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    return atr


# ----- Triple barrier replay --------------------------------------------------

def simulate_outcomes(
    entry_idx: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    *,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    side_is_long: bool,
) -> pd.DataFrame:
    """
    For each entry index, simulates the triple barrier and returns a dataframe
    with: outcome ∈ {TP, SL, EXPIRE, INVALID}, R_multiple.
    """
    n = len(close)
    rows = []
    for idx in entry_idx:
        if idx + horizon >= n:
            rows.append(("INVALID", 0.0))
            continue
        a = atr[idx]
        if not np.isfinite(a) or a <= 0:
            rows.append(("INVALID", 0.0))
            continue
        entry = close[idx]
        if side_is_long:
            tp_level = entry + tp_mult * a
            sl_level = entry - sl_mult * a
        else:
            tp_level = entry - tp_mult * a
            sl_level = entry + sl_mult * a

        outcome = "EXPIRE"
        r_mult = 0.0
        for k in range(1, horizon + 1):
            j = idx + k
            if side_is_long:
                hit_tp = high[j] >= tp_level
                hit_sl = low[j] <= sl_level
            else:
                hit_tp = low[j] <= tp_level
                hit_sl = high[j] >= sl_level
            if hit_tp and hit_sl:
                # ambos en la misma barra: regla pesimista — asumimos SL primero
                outcome = "SL"
                r_mult = -sl_mult
                break
            if hit_tp:
                outcome = "TP"
                r_mult = tp_mult
                break
            if hit_sl:
                outcome = "SL"
                r_mult = -sl_mult
                break
        else:
            # expira sin tocar barreras → PnL = (close[idx+h] - entry) / atr (con signo)
            exit_close = close[idx + horizon]
            raw = (exit_close - entry) / a
            r_mult = raw if side_is_long else -raw
            outcome = "EXPIRE"

        rows.append((outcome, float(r_mult)))

    df = pd.DataFrame(rows, columns=["outcome", "R_multiple"])
    return df


# ----- Sweep -----------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout-preds", required=True,
                    help="parquet con time/state/y_true/y_pred_cal/...")
    ap.add_argument("--ohlcv", required=True,
                    help="parquet o CSV con OHLCV 5min (time, open, high, low, close)")
    ap.add_argument("--proba-col", default="y_pred_cal",
                    choices=["y_pred_cal", "y_pred_raw"])
    ap.add_argument("--side", default=None, choices=[None, "long", "short"],
                    help="Si no se pasa, se deduce del nombre del fichero.")
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--tp", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--n-points", type=int, default=60)
    ap.add_argument("--thr-lo", type=float, default=0.10)
    ap.add_argument("--thr-hi", type=float, default=0.40)
    ap.add_argument("--min-signals", type=int, default=30)
    args = ap.parse_args()

    # ── side ─────────────────────────────────────────────────────────────
    side = args.side
    if side is None:
        name = Path(args.holdout_preds).stem.lower()
        if "_long" in name:
            side = "long"
        elif "_short" in name:
            side = "short"
        else:
            raise SystemExit("No puedo deducir --side, pásalo explícitamente.")
    side_is_long = side == "long"

    # ── load ─────────────────────────────────────────────────────────────
    print(f"📂 holdout preds: {args.holdout_preds}")
    preds = pd.read_parquet(args.holdout_preds)
    preds["time"] = pd.to_datetime(preds["time"])
    print(f"   rows={len(preds):,}  cols={list(preds.columns)}")

    print(f"📂 ohlcv: {args.ohlcv}")
    ohlcv_path = Path(args.ohlcv)
    if ohlcv_path.suffix == ".csv":
        ohlcv = pd.read_csv(ohlcv_path)
    else:
        ohlcv = pd.read_parquet(ohlcv_path)
    ohlcv.columns = [c.lower() for c in ohlcv.columns]
    for c in ("time", "open", "high", "low", "close"):
        if c not in ohlcv.columns:
            raise SystemExit(f"❌ '{c}' no en OHLCV. Cols={list(ohlcv.columns)}")
    ohlcv["time"] = pd.to_datetime(ohlcv["time"])
    ohlcv = ohlcv.sort_values("time").reset_index(drop=True)
    print(f"   rows={len(ohlcv):,}  rango={ohlcv['time'].min()} → {ohlcv['time'].max()}")

    # ── ATR ──────────────────────────────────────────────────────────────
    atr = wilder_atr(
        ohlcv["high"].to_numpy(),
        ohlcv["low"].to_numpy(),
        ohlcv["close"].to_numpy(),
        period=args.atr_window,
    )

    # ── join por time ────────────────────────────────────────────────────
    ohlcv["__row__"] = np.arange(len(ohlcv))
    j = preds.merge(ohlcv[["time", "__row__"]], on="time", how="inner")
    print(f"   filas alineadas (preds ∩ ohlcv) = {len(j):,} / {len(preds):,}")
    if len(j) == 0:
        raise SystemExit("❌ no hay coincidencias temporales — ¿OHLCV equivocado?")
    if len(j) < len(preds) * 0.9:
        print("   ⚠️  Más de un 10% de preds sin OHLCV. ¿Mismo timeframe?")

    high = ohlcv["high"].to_numpy()
    low = ohlcv["low"].to_numpy()
    close = ohlcv["close"].to_numpy()

    # ── simulate barriers para TODAS las filas alineadas ─────────────────
    print(f"\n🔬 Simulando triple barrier (tp={args.tp}, sl={args.sl}, "
          f"h={args.horizon}, side={side})...")
    out = simulate_outcomes(
        j["__row__"].to_numpy(),
        high, low, close, atr,
        horizon=args.horizon,
        tp_mult=args.tp,
        sl_mult=args.sl,
        side_is_long=side_is_long,
    )
    j = pd.concat([j.reset_index(drop=True), out], axis=1)

    # ── distribución global ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  DISTRIBUCIÓN GLOBAL DE OUTCOMES (todas las filas del holdout)")
    print("=" * 70)
    counts = j["outcome"].value_counts()
    total = len(j)
    for k in ("TP", "SL", "EXPIRE", "INVALID"):
        c = int(counts.get(k, 0))
        print(f"  {k:8s}: {c:>7,d}  ({c/total:.4f})")
    valid = j[j["outcome"] != "INVALID"]
    print(f"  mean R  : {valid['R_multiple'].mean():+.4f}")
    print(f"  mean R | EXPIRE: "
          f"{valid.loc[valid['outcome']=='EXPIRE', 'R_multiple'].mean():+.4f}")

    # ── sweep ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SWEEP DE THRESHOLDS — EV REAL POR SEÑAL")
    print("=" * 70)
    rows = []
    thrs = np.linspace(args.thr_lo, args.thr_hi, args.n_points)
    for thr in thrs:
        mask = j[args.proba_col].to_numpy() >= thr
        sig = int(mask.sum())
        if sig < args.min_signals:
            continue
        sub = j[mask]
        n_tp = int((sub["outcome"] == "TP").sum())
        n_sl = int((sub["outcome"] == "SL").sum())
        n_exp = int((sub["outcome"] == "EXPIRE").sum())
        n_inv = int((sub["outcome"] == "INVALID").sum())
        n_valid = sig - n_inv
        if n_valid == 0:
            continue
        mean_r = float(sub.loc[sub["outcome"] != "INVALID", "R_multiple"].mean())
        mean_r_exp = float(
            sub.loc[sub["outcome"] == "EXPIRE", "R_multiple"].mean()
        ) if n_exp > 0 else 0.0
        prec_tp = n_tp / n_valid
        frac_sl = n_sl / n_valid
        frac_exp = n_exp / n_valid

        # BE empírico: precisión TP requerida para EV = 0,
        # asumiendo split SL/EXPIRE entre los negativos = igual al observado.
        # EV(p) = p*tp + (1-p)*[ (frac_sl/(frac_sl+frac_exp)) * (-sl)
        #                      + (frac_exp/(frac_sl+frac_exp)) * mean_r_exp ]
        denom = frac_sl + frac_exp
        if denom > 1e-9:
            neg_avg_R = (frac_sl / denom) * (-args.sl) + (frac_exp / denom) * mean_r_exp
            # EV = p*tp + (1-p)*neg_avg_R = 0 → p = -neg_avg_R / (tp - neg_avg_R)
            be_emp = -neg_avg_R / (args.tp - neg_avg_R) if (args.tp - neg_avg_R) > 0 else float("nan")
        else:
            be_emp = float("nan")

        rows.append({
            "thr": float(thr),
            "sig": sig,
            "n_TP": n_tp,
            "n_SL": n_sl,
            "n_EXP": n_exp,
            "prec_TP": prec_tp,
            "frac_SL": frac_sl,
            "frac_EXP": frac_exp,
            "mean_R_exp": mean_r_exp,
            "EV_per_sig": mean_r,
            "BE_emp": be_emp,
            "deployable_emp": prec_tp >= be_emp if np.isfinite(be_emp) else False,
        })

    res = pd.DataFrame(rows)
    if res.empty:
        print(f"❌ Ningún threshold produjo >= {args.min_signals} señales.")
        return

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(res.round(4).to_string(index=False))

    # ── highlights ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  HIGHLIGHTS")
    print("=" * 70)
    best_ev = res.loc[res["EV_per_sig"].idxmax()]
    print(f"  Mejor EV/señal: thr={best_ev['thr']:.3f}  "
          f"EV={best_ev['EV_per_sig']:+.4f}R  sig={int(best_ev['sig'])}  "
          f"prec_TP={best_ev['prec_TP']:.3f}  BE_emp={best_ev['BE_emp']:.3f}")
    pos_ev = res[res["EV_per_sig"] > 0]
    if not pos_ev.empty:
        max_sig = pos_ev.loc[pos_ev["sig"].idxmax()]
        print(f"  Más señales con EV>0: thr={max_sig['thr']:.3f}  "
              f"sig={int(max_sig['sig'])}  EV={max_sig['EV_per_sig']:+.4f}R  "
              f"prec_TP={max_sig['prec_TP']:.3f}")
    else:
        print("  ❌ Ningún threshold con EV>0.")

    # promedio de BE empírico (orientativo)
    be_emp_global = res["BE_emp"].dropna().median()
    print(f"\n  BE empírico mediano sobre el sweep: {be_emp_global:.3f} "
          f"(vs BE teórico {args.sl/(args.tp+args.sl):.3f})")


if __name__ == "__main__":
    main()
