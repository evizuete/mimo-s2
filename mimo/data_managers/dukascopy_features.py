#!/usr/bin/env python3
"""
dukascopy_features.py

Toma un parquet de tick data (output de dukascopy_loader.py) y genera un
parquet a granularidad 1 min con las features sinteticas de microestructura.

Features generadas (8, plus las cruds para sanity):

    spread_mean_1m, spread_p95_1m, spread_z_score_60m
    quote_imbalance_mean_1m, quote_imbalance_change_5m
    tick_count_z_score_60m
    quote_volatility_1m
    quote_velocity_1m

mas las crudas:

    spread_mean, tick_count, mid_open, mid_close, mid_high, mid_low

Las z-scores son rolling 60 (causales). Las features se construyen UTC y se
ancla al "minuto cerrado" (label='right').

Uso:
    python -m mimo.data_managers.dukascopy_features \\
        --in  ../../data_sample/dukascopy_xauusd_2025-10.parquet \\
        --out ../../data_sample/dukascopy_xauusd_2025-10_features_1m.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROLLING_NORM = 60          # ventana de z-score (en minutos)
IMBALANCE_DELTA = 5        # ventana para el cambio de imbalance

OUTLIER_SPREAD_QUANTILE = 0.999  # filtro de ticks con spread atipico


def filter_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Quita ticks con bid>ask, spreads negativos o spreads en P99.9."""
    spread = df["ask"] - df["bid"]
    mask = (spread > 0) & (spread < spread.quantile(OUTLIER_SPREAD_QUANTILE))
    n_before = len(df)
    df = df.loc[mask].copy()
    n_after = len(df)
    if n_before != n_after:
        print(f"  [filter] eliminados {n_before - n_after:,} "
              f"ticks anomalos ({100*(n_before-n_after)/n_before:.2f}%)")
    return df


def aggregate_to_1m(df_ticks: pd.DataFrame) -> pd.DataFrame:
    """Construye features 1-minuto desde ticks. label='right'."""
    if df_ticks.empty:
        raise ValueError("DataFrame de ticks vacio")

    df = df_ticks.copy()
    if df["time"].dt.tz is None:
        df["time"] = df["time"].dt.tz_localize("UTC")

    df["spread"] = df["ask"] - df["bid"]
    df["mid"] = (df["ask"] + df["bid"]) / 2.0
    bv_plus_av = df["bid_volume"] + df["ask_volume"]
    df["imbalance"] = np.where(
        bv_plus_av > 0,
        df["bid_volume"] / np.maximum(bv_plus_av, 1e-9),
        0.5,
    )

    df = df.set_index("time")

    # ---- Spread (liquidez) ----
    spread_mean = df["spread"].resample("1min", label="right", closed="right").mean()
    spread_p95 = df["spread"].resample("1min", label="right", closed="right").quantile(0.95)

    # ---- Imbalance (presion direccional) ----
    imb_mean = df["imbalance"].resample("1min", label="right", closed="right").mean()

    # ---- Intensidad / actividad ----
    tick_count = df["spread"].resample("1min", label="right", closed="right").count()

    # ---- Volatilidad de cotizacion (intra-minuto) ----
    quote_vol = df["mid"].resample("1min", label="right", closed="right").std()

    # ---- Velocidad: ticks-por-segundo ~ tick_count / 60 (proxy) ----
    # Lo dejamos en escala "ticks por segundo" (mas interpretable) en lugar de p95
    # del inter-tick que requeriria mas trabajo. Es equivalente para z-scoring.
    quote_velocity = tick_count / 60.0

    # ---- Crudas: mid OHLC ----
    mid_resampled = df["mid"].resample("1min", label="right", closed="right")
    mid_open = mid_resampled.first()
    mid_close = mid_resampled.last()
    mid_high = mid_resampled.max()
    mid_low = mid_resampled.min()

    out = pd.DataFrame({
        "spread_mean": spread_mean,
        "spread_p95": spread_p95,
        "imbalance_mean": imb_mean,
        "tick_count": tick_count.astype(float),
        "quote_volatility": quote_vol,
        "quote_velocity": quote_velocity,
        "mid_open": mid_open,
        "mid_close": mid_close,
        "mid_high": mid_high,
        "mid_low": mid_low,
    })

    # Forward-fill suave para gaps de < 5 min, dejar NaN para gaps mas largos.
    # Util para evitar imputar valores en weekends.
    n_before = out.notna().sum()
    out = out.fillna(method="ffill", limit=4)
    n_after = out.notna().sum()
    print(f"  [ffill <=4 min] filas validas tras forward-fill suave:")
    for col in ["spread_mean", "tick_count", "imbalance_mean"]:
        print(f"    {col}: {n_before[col]:,} -> {n_after[col]:,}")

    # Z-scores rolling (causales) sobre lo que tengamos.
    def _zscore(s: pd.Series, w: int) -> pd.Series:
        m = s.rolling(w, min_periods=max(10, w // 4)).mean()
        sd = s.rolling(w, min_periods=max(10, w // 4)).std()
        return (s - m) / (sd + 1e-9)

    out["spread_z_score_60m"] = _zscore(out["spread_mean"], ROLLING_NORM).clip(-5, 5)
    out["tick_count_z_score_60m"] = _zscore(out["tick_count"], ROLLING_NORM).clip(-5, 5)

    # Cambio de imbalance (no z-score, es una diferencia rolling).
    out["imbalance_change_5m"] = (
        out["imbalance_mean"]
        - out["imbalance_mean"].rolling(IMBALANCE_DELTA, min_periods=2).mean()
    )

    # Replicas con nombres canonicos para join con el feature_builder.
    out = out.rename(columns={
        "spread_mean": "spread_mean_1m",
        "spread_p95": "spread_p95_1m",
        "imbalance_mean": "quote_imbalance_mean_1m",
        "imbalance_change_5m": "quote_imbalance_change_5m",
        "quote_volatility": "quote_volatility_1m",
        "quote_velocity": "quote_velocity_1m",
    })

    out = out.reset_index().rename(columns={"time": "time"})
    out["time"] = out["time"].dt.tz_convert("UTC")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    in_p = Path(args.in_path)
    if not in_p.exists():
        raise SystemExit(f"❌ No existe: {in_p}")

    print(f"\n📂 Cargando ticks: {in_p}")
    df = pd.read_parquet(in_p)
    print(f"   {len(df):,} ticks. Rango {df.time.min()} → {df.time.max()}")

    df = filter_outliers(df)
    print(f"   Tras filtro: {len(df):,} ticks")

    print("\n🔧 Agregando a 1-min...")
    feats = aggregate_to_1m(df)
    print(f"   {len(feats):,} barras 1m generadas")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    feats.to_parquet(out, index=False)
    print(f"\n✅ Features 1m guardados en: {out}")

    # Sanity quick stats.
    print("\n📊 Stats rapidas (top 10 columnas, no NaN):")
    print(feats.describe().T.round(4).to_string())


if __name__ == "__main__":
    main()
