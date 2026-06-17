#!/usr/bin/env python3
"""
microstructure_signal_correlation.py

Mide si las features de microestructura (output de dukascopy_features.py)
aportan informacion sobre la senal del modelo (output del triple-barrier).

Estrategia:
  - Cargar calibration_dataset_<release>_<side>.parquet (tiene time, state,
    signal — el label binario del triple-barrier).
  - Cargar el parquet de features de microestructura.
  - merge_asof por time con tolerancia configurable.
  - Calcular correlacion (Pearson y Spearman) de cada feature con signal:
        - global
        - por estado de mercado
  - Marcar las features con |corr| >= threshold como candidatas a integrar.

Uso:
    python -m mimo.data_managers.microstructure_signal_correlation \\
        --calib-parquet ../../artifacts/202105/oof/.../data/calibration_dataset_202105_long.parquet \\
        --micro-parquet ../../data_sample/dukascopy_features.parquet \\
        --side long \\
        --threshold 0.03

Veredicto:
    Si >=2 features pasan el threshold global, justifica integracion.
    Si todas <0.02, las features no aportan al problema actual.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

MICRO_FEATURES = [
    "spread_mean_1m",
    "spread_p95_1m",
    "spread_z_score_60m",
    "quote_imbalance_mean_1m",
    "quote_imbalance_change_5m",
    "tick_count",
    "tick_count_z_score_60m",
    "quote_volatility_1m",
    "quote_velocity_1m",
]


def load_calibration(path: Path, *, source_filter: str | None = None) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    if "signal" not in df.columns:
        raise ValueError(f"Falta columna 'signal' en {path.name}")
    if source_filter is not None and "source" in df.columns:
        df = df[df["source"] == source_filter].copy()
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["signal", "time"]).copy()
    df["signal"] = df["signal"].astype(int)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.lower()
    return df.sort_values("time").reset_index(drop=True)


def load_micro(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["time"] = pd.to_datetime(df["time"], utc=True)
    return df.sort_values("time").reset_index(drop=True)


def merge_micro_to_calib(
    calib: pd.DataFrame,
    micro: pd.DataFrame,
    tolerance: pd.Timedelta = pd.Timedelta("90s"),
) -> pd.DataFrame:
    cols_to_carry = [c for c in MICRO_FEATURES if c in micro.columns]
    if not cols_to_carry:
        raise ValueError(f"Ninguna feature canonica {MICRO_FEATURES} en el micro parquet. "
                         f"Cols disponibles: {list(micro.columns)}")
    merged = pd.merge_asof(
        calib,
        micro[["time"] + cols_to_carry],
        on="time",
        direction="backward",
        tolerance=tolerance,
    )
    return merged


def correlation_table(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    out = []
    for f in features:
        if f not in df.columns:
            continue
        sub = df.dropna(subset=[f, "signal"])
        n = len(sub)
        if n < 50:
            out.append({"feature": f, "n": n, "pearson": np.nan, "spearman": np.nan})
            continue
        pear = sub[f].corr(sub["signal"].astype(float), method="pearson")
        spea = sub[f].corr(sub["signal"].astype(float), method="spearman")
        out.append({"feature": f, "n": n, "pearson": pear, "spearman": spea})
    return pd.DataFrame(out)


def per_state_correlation(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    if "state" not in df.columns:
        return pd.DataFrame()
    rows = []
    for state, g in df.groupby("state"):
        for f in features:
            if f not in g.columns:
                continue
            sub = g.dropna(subset=[f, "signal"])
            n = len(sub)
            if n < 50:
                continue
            pear = sub[f].corr(sub["signal"].astype(float), method="pearson")
            rows.append({"state": state, "feature": f, "n": n, "pearson": pear})
    return pd.DataFrame(rows).sort_values(["feature", "state"]).reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--calib-parquet", required=True,
                    help="calibration_dataset_<release>_<side>.parquet")
    ap.add_argument("--micro-parquet", required=True,
                    help="output de dukascopy_features.py")
    ap.add_argument("--side", default="long",
                    help="solo informativo, no cambia la logica")
    ap.add_argument("--threshold", type=float, default=0.03,
                    help="|corr| minimo para considerar feature util")
    ap.add_argument("--source-filter", default=None,
                    choices=[None, "oof_train", "holdout"],
                    help="filtrar a una fuente concreta del calibration_dataset")
    ap.add_argument("--tolerance-sec", type=int, default=90,
                    help="tolerancia del merge_asof (segundos)")
    args = ap.parse_args()

    calib_p = Path(args.calib_parquet)
    micro_p = Path(args.micro_parquet)
    if not calib_p.exists():
        raise SystemExit(f"❌ No existe: {calib_p}")
    if not micro_p.exists():
        raise SystemExit(f"❌ No existe: {micro_p}")

    print(f"\n📂 calib : {calib_p}")
    print(f"📂 micro : {micro_p}\n")

    calib = load_calibration(calib_p, source_filter=args.source_filter)
    micro = load_micro(micro_p)

    print(f"  calib rows : {len(calib):,}  rango {calib.time.min()} → {calib.time.max()}")
    print(f"  micro rows : {len(micro):,}  rango {micro.time.min()} → {micro.time.max()}")

    overlap_lo = max(calib.time.min(), micro.time.min())
    overlap_hi = min(calib.time.max(), micro.time.max())
    if overlap_lo >= overlap_hi:
        raise SystemExit("❌ Sin solapamiento temporal entre calib y micro.")
    overlap_days = (overlap_hi - overlap_lo).total_seconds() / 86400
    print(f"  overlap    : {overlap_lo} → {overlap_hi}  ({overlap_days:.1f} dias)")

    calib_overlap = calib[(calib.time >= overlap_lo) & (calib.time <= overlap_hi)].copy()
    print(f"  calib en overlap: {len(calib_overlap):,}")

    merged = merge_micro_to_calib(
        calib_overlap, micro,
        tolerance=pd.Timedelta(seconds=args.tolerance_sec),
    )
    matched = merged[MICRO_FEATURES].notna().any(axis=1).sum()
    print(f"  rows con micro features tras merge_asof: {int(matched):,} "
          f"({100*matched/len(merged):.1f}%)\n")

    print("=" * 78)
    print(f"  CORRELACION GLOBAL  (side={args.side}, threshold ±{args.threshold:.3f})")
    print("=" * 78)
    table = correlation_table(merged, MICRO_FEATURES)
    table["|pearson|"] = table["pearson"].abs()
    table["pasa"] = table["|pearson|"] >= args.threshold
    print(table.round(4).to_string(index=False))
    print()

    pasan = table[table["pasa"]]["feature"].tolist()
    if not pasan:
        print(f"  ❌ Ninguna feature supera |corr|={args.threshold:.3f} a nivel global.")
    else:
        print(f"  ✅ {len(pasan)} feature(s) con |corr| >= {args.threshold:.3f}: {pasan}")
    print()

    print("=" * 78)
    print("  CORRELACION POR ESTADO  (Pearson, n>=50)")
    print("=" * 78)
    per_state = per_state_correlation(merged, MICRO_FEATURES)
    if per_state.empty:
        print("  (sin columna 'state' o sin datos suficientes)")
    else:
        pivot = per_state.pivot(index="feature", columns="state", values="pearson")
        print(pivot.round(4).to_string())
    print()

    # Resumen ejecutivo
    print("=" * 78)
    print("  VEREDICTO")
    print("=" * 78)
    n_global = int(table["pasa"].sum())
    if not per_state.empty:
        per_state["|pear|"] = per_state["pearson"].abs()
        n_state_strong = int((per_state["|pear|"] >= args.threshold).sum())
    else:
        n_state_strong = 0

    if n_global >= 2:
        print(f"  ✅ {n_global} features con |corr| global >= {args.threshold:.3f}.")
        print(f"     Justifica integracion en feature_builder.py.")
    elif n_state_strong >= 3:
        print(f"  🟡 {n_state_strong} (feature, estado) con |corr| >= {args.threshold:.3f}.")
        print(f"     Senal heterogenea: aporta solo en algunos regimenes. Considerar.")
    else:
        print(f"  ❌ Maximo {n_global} features globales y {n_state_strong} per-state cumplen.")
        print(f"     La microestructura sintetica NO aporta sobre el conjunto medido.")
        print(f"     Posibles causas:")
        print(f"       1) Ventana de overlap pequena ({overlap_days:.1f}d) — repetir con mas datos.")
        print(f"       2) Las features genuinamente no aportan en este horizon/setup.")
        print(f"       3) Falta una transformacion (interaccion con state, lag, etc).")


if __name__ == "__main__":
    main()
