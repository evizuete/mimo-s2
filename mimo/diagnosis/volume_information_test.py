#!/usr/bin/env python3
"""
volume_information_test.py

Test go/no-go: ¿aporta el volumen información predictiva sobre el retorno
forward a horizonte h, ANTES de tocar el modelo?

Fase 0 (sanity): nan-rate de volume, distribución, ceros, estacionalidad
                 por hour-of-week.
Fase 1 (info):   IC Spearman, MI discreta y separación por cuartil del feature
                 candidato vs:
                   - forward_return_atr (continuo)
                   - binary_long  (triple barrier tp/sl)
                   - binary_short (triple barrier tp/sl)

Si ningún feature muestra |IC| > 0.03 ni separación pos_rate Q4-Q1 > 2pp,
no merece la pena rehacer el pipeline para meter volumen.

Uso:
    python -m mimo.diagnosis.volume_information_test
    python -m mimo.diagnosis.volume_information_test --base-tf 5min --horizon 5
    python -m mimo.diagnosis.volume_information_test --tp 2.5 --sl 1.0
    python -m mimo.diagnosis.volume_information_test --rates-parquet rates.parquet
"""

from __future__ import annotations

import argparse
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from numba import jit

from mimo.diagnosis.barrier_sweep import (
    triple_barrier_outcomes,
    compute_atr,
)


# ---------------------------------------------------------------------------
# Carga de datos
# ---------------------------------------------------------------------------

def load_rates(args: argparse.Namespace) -> pd.DataFrame:
    if args.rates_parquet:
        print(f"[load] reading parquet {args.rates_parquet}")
        df = pd.read_parquet(args.rates_parquet)
    else:
        from mimo.data_managers.data_manager import DataManager
        from mimo.data_managers.databases import Database
        print(f"[load] reading DB from {args.from_date} to {args.to_date} | "
              f"base_tf={args.base_tf}")
        db = Database()
        from_dt = datetime.fromisoformat(args.from_date)
        to_dt = datetime.fromisoformat(args.to_date)
        dm = DataManager.from_database_historical_2(
            db, from_date=from_dt, to_date=to_dt, resample=args.base_tf
        )
        df = dm.df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    if "ticks_volume" not in df.columns:
        if "volume" in df.columns:
            df = df.rename(columns={"volume": "ticks_volume"})
        else:
            raise ValueError("No hay columna ticks_volume ni volume en el df.")

    if "atr" not in df.columns:
        df["atr"] = compute_atr(df, period=14)

    return df


# ---------------------------------------------------------------------------
# Fase 0 — Sanity check
# ---------------------------------------------------------------------------

def sanity_volume(df: pd.DataFrame) -> None:
    v = df["ticks_volume"]
    n = len(v)
    nan_n = int(v.isna().sum())
    zero_n = int((v == 0).sum())
    pos = v[v > 0]

    print("\n" + "=" * 80)
    print("FASE 0 — SANITY CHECK volume")
    print("=" * 80)
    print(f"rows         : {n:,}")
    print(f"nan          : {nan_n} ({nan_n/n:.4%})")
    print(f"zeros        : {zero_n} ({zero_n/n:.4%})")
    if len(pos):
        print(f"min(pos)     : {pos.min():.2f}")
        print(f"p01 / p50    : {pos.quantile(0.01):.2f} / {pos.median():.2f}")
        print(f"p99 / max    : {pos.quantile(0.99):.2f} / {pos.max():.2f}")
        print(f"mean / std   : {pos.mean():.2f} / {pos.std():.2f}")
        print(f"skew (log)   : {np.log1p(pos).skew():.3f}")

    # Estacionalidad por hour-of-week
    df["_how"] = df["time"].dt.dayofweek * 24 + df["time"].dt.hour
    season = df.groupby("_how")["ticks_volume"].agg(["mean", "median", "count"])
    p10 = season["median"].quantile(0.10)
    p90 = season["median"].quantile(0.90)
    print(f"hour-of-week median spread (p10/p90): {p10:.1f} / {p90:.1f} "
          f"(ratio={p90/max(p10,1e-9):.2f}x)")
    df.drop(columns=["_how"], inplace=True)


# ---------------------------------------------------------------------------
# Features de volumen
# ---------------------------------------------------------------------------

def add_volume_features(df: pd.DataFrame, base_tf: str) -> pd.DataFrame:
    """
    Crea candidatos de features de volumen usando solo info pasada.
    Todos escala-invariantes para que sean comparables histórico ↔ MT5 live.
    """
    out = df.copy()
    v = out["ticks_volume"].astype(float).clip(lower=0)
    log_v = np.log1p(v)

    # Ventanas según tf base
    if base_tf and base_tf.endswith("min"):
        n_per_h = max(1, 60 // int(base_tf.replace("min", "")))
    else:
        n_per_h = 60  # asumir 1m si no resampled
    w_1h = n_per_h
    w_4h = 4 * n_per_h
    w_24h = 24 * n_per_h

    # vol_spike: desviación del log-vol respecto a su EMA(12)
    out["vol_spike"] = log_v - log_v.ewm(span=12, adjust=False).mean()

    # vol_z_1h: z-score rolling 1h
    mu_1h = log_v.rolling(w_1h, min_periods=max(8, w_1h // 2)).mean()
    sd_1h = log_v.rolling(w_1h, min_periods=max(8, w_1h // 2)).std()
    out["vol_z_1h"] = (log_v - mu_1h) / sd_1h.replace(0, np.nan)

    # vol_z_4h
    mu_4h = log_v.rolling(w_4h, min_periods=max(16, w_4h // 4)).mean()
    sd_4h = log_v.rolling(w_4h, min_periods=max(16, w_4h // 4)).std()
    out["vol_z_4h"] = (log_v - mu_4h) / sd_4h.replace(0, np.nan)

    # vol_pct_1h: percentile rank en ventana 1h (más robusto que z)
    out["vol_pct_1h"] = (
        log_v.rolling(w_1h, min_periods=max(8, w_1h // 2))
             .rank(pct=True)
    )

    # vol_pct_4h
    out["vol_pct_4h"] = (
        log_v.rolling(w_4h, min_periods=max(16, w_4h // 4))
             .rank(pct=True)
    )

    # vol_dsd: deseasonalized — vol / median por hour-of-week (causal)
    # Para evitar leakage, usamos la mediana de las últimas 24h del mismo hour-of-week.
    # Aproximación rápida: dividimos por la mediana rolling de 24h del mismo bar idx.
    # Implementación simple: vol normalizado por mediana rolling 24h.
    med_24h = v.rolling(w_24h, min_periods=max(32, w_24h // 8)).median()
    out["vol_dsd"] = v / med_24h.replace(0, np.nan)

    # CMF (Chaikin Money Flow) rolling 20
    hl_range = (out["high"] - out["low"]).replace(0, np.nan)
    mfm = ((out["close"] - out["low"]) - (out["high"] - out["close"])) / hl_range
    mfv = mfm * v
    out["cmf_20"] = (
        mfv.rolling(20, min_periods=10).sum()
        / v.rolling(20, min_periods=10).sum().replace(0, np.nan)
    )

    # vol_trend: pendiente normalizada de log_v en ventana 1h
    # = (log_v - log_v.shift(w_1h)) / w_1h  (proxy de pendiente)
    out["vol_trend_1h"] = (log_v - log_v.shift(w_1h)) / max(1, w_1h)

    # Range/vol: eficiencia (price move por unidad de volumen)
    out["range_per_vol"] = (out["high"] - out["low"]) / v.replace(0, np.nan)
    # log para que tenga colas razonables
    out["log_range_per_vol"] = np.log1p(out["range_per_vol"].clip(lower=0))

    return out


VOLUME_FEATURES = [
    "vol_spike",
    "vol_z_1h",
    "vol_z_4h",
    "vol_pct_1h",
    "vol_pct_4h",
    "vol_dsd",
    "cmf_20",
    "vol_trend_1h",
    "log_range_per_vol",
]


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

def add_labels(df: pd.DataFrame, horizon: int, tp: float, sl: float) -> pd.DataFrame:
    out = df.copy()
    close = out["close"].to_numpy(np.float64)
    high = out["high"].to_numpy(np.float64)
    low = out["low"].to_numpy(np.float64)
    atr = out["atr"].to_numpy(np.float64)

    # forward return ATR-normalized signed (long perspective)
    fwd_close = pd.Series(close).shift(-horizon).to_numpy()
    out["fwd_ret_atr"] = (fwd_close - close) / np.where(atr > 0, atr, np.nan)

    # binary triple-barrier
    out_long = triple_barrier_outcomes(close, high, low, atr, horizon, tp, sl, True)
    out_short = triple_barrier_outcomes(close, high, low, atr, horizon, tp, sl, False)
    out["bin_long"] = np.where(out_long == 99, np.nan,
                               (out_long == 1).astype(float))
    out["bin_short"] = np.where(out_short == 99, np.nan,
                                (out_short == 1).astype(float))
    return out


# ---------------------------------------------------------------------------
# Information metrics
# ---------------------------------------------------------------------------

def spearman_ic(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 100:
        return np.nan
    rx = pd.Series(x[mask]).rank().to_numpy()
    ry = pd.Series(y[mask]).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def mi_discrete(x: np.ndarray, y: np.ndarray, bins_x: int = 10, bins_y: int = 10) -> float:
    """MI por discretización en quantiles. Escala libre, robusto a no-linealidad."""
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 200:
        return np.nan
    xx = x[mask]
    yy = y[mask]
    qx = pd.qcut(xx, bins_x, labels=False, duplicates="drop")
    qy = pd.qcut(yy, bins_y, labels=False, duplicates="drop")
    n = len(qx)
    joint = pd.crosstab(qx, qy).to_numpy().astype(float)
    p_xy = joint / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x @ p_y
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p_xy > 0) & (denom > 0), p_xy / denom, 1.0)
        mi = np.sum(p_xy * np.log(ratio))
    return float(mi)


def quartile_breakdown(feature: np.ndarray, target: np.ndarray) -> pd.DataFrame:
    mask = np.isfinite(feature) & np.isfinite(target)
    if mask.sum() < 200:
        return pd.DataFrame()
    f = feature[mask]
    t = target[mask]
    q = pd.qcut(f, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop")
    g = pd.DataFrame({"q": q, "t": t}).groupby("q", observed=True)["t"]
    return pd.DataFrame({
        "n": g.size(),
        "mean": g.mean(),
        "std": g.std(),
    })


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", default="2025-01-01")
    ap.add_argument("--to-date", default="2026-04-26")
    ap.add_argument("--base-tf", default="5min",
                    help="Resample tf (e.g. '5min'). Pon vacío para 1m.")
    ap.add_argument("--horizon", type=int, default=5,
                    help="Horizonte forward en velas (5 = 25min @ 5min).")
    ap.add_argument("--tp", type=float, default=2.5)
    ap.add_argument("--sl", type=float, default=1.0)
    ap.add_argument("--rates-parquet", default=None)
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    if not args.base_tf:
        args.base_tf = None
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)

    df = load_rates(args)
    print(f"[load] {len(df):,} filas | atr nan-rate: {df['atr'].isna().mean():.4f}")

    sanity_volume(df)

    print("\n" + "=" * 80)
    print(f"FASE 1 — INFORMATION TEST | h={args.horizon} | tp={args.tp} sl={args.sl}")
    print("=" * 80)

    df = add_volume_features(df, args.base_tf)
    df = add_labels(df, args.horizon, args.tp, args.sl)

    pos_rate_long = df["bin_long"].mean()
    pos_rate_short = df["bin_short"].mean()
    print(f"base pos_rate LONG  = {pos_rate_long:.4f}")
    print(f"base pos_rate SHORT = {pos_rate_short:.4f}")
    print(f"BE @ tp/sl          = {args.sl/(args.tp+args.sl):.4f}")

    # Tabla resumen IC + MI
    rows = []
    for f in VOLUME_FEATURES:
        x = df[f].to_numpy()
        rows.append({
            "feature": f,
            "ic_fwd": spearman_ic(x, df["fwd_ret_atr"].to_numpy()),
            "ic_long": spearman_ic(x, df["bin_long"].to_numpy()),
            "ic_short": spearman_ic(x, df["bin_short"].to_numpy()),
            "mi_fwd": mi_discrete(x, df["fwd_ret_atr"].to_numpy(), 10, 10),
            "mi_long": mi_discrete(x, df["bin_long"].to_numpy(), 10, 2),
            "mi_short": mi_discrete(x, df["bin_short"].to_numpy(), 10, 2),
        })
    summary = pd.DataFrame(rows)
    print("\n[IC Spearman + MI discreta]")
    print(summary.round(4).to_string(index=False))

    # Quartile breakdown — solo para los 3 mejores por |ic_fwd|
    summary["ic_fwd_abs"] = summary["ic_fwd"].abs()
    top = summary.sort_values("ic_fwd_abs", ascending=False).head(3)
    print("\n[Quartile breakdown — top 3 features por |ic_fwd|]")
    for _, r in top.iterrows():
        f = r["feature"]
        x = df[f].to_numpy()
        print(f"\n· {f} (ic_fwd={r['ic_fwd']:+.4f})")

        qb_fwd = quartile_breakdown(x, df["fwd_ret_atr"].to_numpy())
        if not qb_fwd.empty:
            print("    fwd_ret_atr (signed):")
            print(qb_fwd.round(4).to_string())

        qb_long = quartile_breakdown(x, df["bin_long"].to_numpy())
        if not qb_long.empty:
            qb_long["spread_pp"] = (qb_long["mean"] - qb_long["mean"].iloc[0]) * 100
            print("    pos_rate LONG por cuartil:")
            print(qb_long.round(4).to_string())

        qb_short = quartile_breakdown(x, df["bin_short"].to_numpy())
        if not qb_short.empty:
            qb_short["spread_pp"] = (qb_short["mean"] - qb_short["mean"].iloc[0]) * 100
            print("    pos_rate SHORT por cuartil:")
            print(qb_short.round(4).to_string())

    # Veredicto
    print("\n" + "=" * 80)
    print("VEREDICTO")
    print("=" * 80)
    max_ic = summary["ic_fwd_abs"].max()
    n_strong = int((summary["ic_fwd_abs"] > 0.03).sum())
    if max_ic > 0.03 and n_strong >= 1:
        print(f"  ✓ |IC|max = {max_ic:.4f} > 0.03 → {n_strong} feature(s) con señal")
        print("  → ADELANTE con Fase 2 (integrar al FeatureBuilder).")
    else:
        print(f"  ✗ |IC|max = {max_ic:.4f} ≤ 0.03 — volumen NO separa el target.")
        print("  → STOP. Considerar otras palancas (features de orderflow, "
              "régimen externo, multi-asset).")


if __name__ == "__main__":
    main()
