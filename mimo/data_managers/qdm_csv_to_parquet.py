#!/usr/bin/env python3
"""
qdm_csv_to_parquet.py

Convierte el CSV exportado por QuantDataManager (Dukascopy tick data)
a un parquet en el schema esperado por dukascopy_features.py.

Formato CSV de entrada (header):
    DateTime,Bid,Ask,Volume
    20260407 00:00:00.186,4659.155,4659.945,90
    ...

DateTime es UTC en formato YYYYMMDD HH:MM:SS.fff.
Volume es total (no separado por lado).

Schema de salida (parquet):
    time         datetime64[ns, UTC]
    bid          float64
    ask          float64
    volume       float32

Uso:
    python -m mimo.data_managers.qdm_csv_to_parquet \\
        --in  data_sample/XAUUSD_2024-2026.csv \\
        --out data_sample/dukascopy_xauusd.parquet
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

QDM_DATETIME_FMT = "%Y%m%d %H:%M:%S.%f"


def convert(in_path: Path, out_path: Path, *, chunksize: int | None = None) -> None:
    """Lee el CSV de QDM (potencialmente grande) en chunks y vuelca parquet."""
    print(f"📂 Leyendo {in_path}")
    if not in_path.exists():
        raise SystemExit(f"❌ No existe: {in_path}")

    if chunksize is None:
        df = pd.read_csv(in_path)
        df = _normalize(df)
    else:
        chunks = []
        for i, chunk in enumerate(pd.read_csv(in_path, chunksize=chunksize)):
            chunks.append(_normalize(chunk))
            if (i + 1) % 10 == 0:
                print(f"  procesados {(i+1)*chunksize:,} ticks...")
        df = pd.concat(chunks, ignore_index=True)

    df = df.sort_values("time").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"✅ {len(df):,} ticks guardados en {out_path}")
    print(f"   Rango: {df.time.min()} → {df.time.max()}")
    print(f"   bid: [{df.bid.min():.3f}, {df.bid.max():.3f}]  "
          f"ask: [{df.ask.min():.3f}, {df.ask.max():.3f}]")
    print(f"   spread medio: {(df.ask - df.bid).mean():.4f}")
    print(f"   volume medio: {df.volume.mean():.2f}  total: {df.volume.sum():,.0f}")


def _normalize(df: pd.DataFrame) -> pd.DataFrame:
    # QDM usa "DateTime" pero algunos exports usan "Time (UTC)" — soportamos ambos.
    rename = {}
    for col in df.columns:
        c = col.strip()
        cl = c.lower().replace(" ", "")
        if cl in ("datetime", "time", "time(utc)", "timeutc"):
            rename[col] = "_dt"
        elif cl == "bid":
            rename[col] = "bid"
        elif cl == "ask":
            rename[col] = "ask"
        elif cl == "volume":
            rename[col] = "volume"
        elif cl in ("bidvolume", "bid_volume"):
            rename[col] = "bid_volume"
        elif cl in ("askvolume", "ask_volume"):
            rename[col] = "ask_volume"
    df = df.rename(columns=rename)

    if "_dt" not in df.columns:
        raise ValueError(f"No se detectó columna de timestamp. Columnas: {list(df.columns)}")

    # Parseo robusto: probamos formato QDM, luego fallback a auto.
    s = df["_dt"].astype(str)
    try:
        df["time"] = pd.to_datetime(s, format=QDM_DATETIME_FMT, utc=True, errors="raise")
    except (ValueError, TypeError):
        df["time"] = pd.to_datetime(s, utc=True, errors="coerce")
        if df["time"].isna().any():
            n_bad = int(df["time"].isna().sum())
            raise ValueError(
                f"No se pudieron parsear {n_bad} timestamps. Ejemplo malo: "
                f"{df.loc[df['time'].isna(), '_dt'].iloc[0]!r}"
            )

    out_cols = ["time", "bid", "ask"]
    for opt in ("volume", "bid_volume", "ask_volume"):
        if opt in df.columns:
            out_cols.append(opt)

    out = df[out_cols].copy()
    out["bid"] = out["bid"].astype("float64")
    out["ask"] = out["ask"].astype("float64")
    if "volume" in out.columns:
        out["volume"] = out["volume"].astype("float32")
    if "bid_volume" in out.columns:
        out["bid_volume"] = out["bid_volume"].astype("float32")
    if "ask_volume" in out.columns:
        out["ask_volume"] = out["ask_volume"].astype("float32")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in", dest="in_path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--chunksize", type=int, default=None,
                    help="Si el CSV es enorme, leer en chunks de N filas")
    args = ap.parse_args()
    convert(Path(args.in_path), Path(args.out), chunksize=args.chunksize)


if __name__ == "__main__":
    main()
