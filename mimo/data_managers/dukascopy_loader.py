#!/usr/bin/env python3
"""
dukascopy_loader.py

Descarga XAUUSD (u otro instrumento Dukascopy) en formato tick desde el
endpoint publico:

    https://datafeed.dukascopy.com/datafeed/{INSTR}/{YEAR}/{MONTH-1:02d}/{DAY:02d}/{HOUR:02d}h_ticks.bi5

Cada fichero .bi5 es LZMA y contiene una secuencia de ticks de 20 bytes:
    uint32  offset_ms_within_hour   (big-endian)
    uint32  ask_int                 (big-endian, pip-scaled)
    uint32  bid_int                 (big-endian, pip-scaled)
    float32 ask_volume              (big-endian)
    float32 bid_volume              (big-endian)

El precio real se obtiene como ask_int / point_factor. Para XAUUSD el
point_factor es 1000 (1 pip = 0.1 USD por onza).

Este loader no depende de paquetes exoticos: solo requests, lzma (stdlib),
struct (stdlib), pandas y numpy.

Uso:
    python -m mimo.data_managers.dukascopy_loader \\
        --instrument XAUUSD \\
        --from 2025-10-01 \\
        --to   2025-10-31 \\
        --out  ../../data_sample/dukascopy_xauusd_2025-10.parquet

El parquet generado tiene columnas: time (datetime UTC), bid, ask,
ask_volume, bid_volume. Una fila por tick.
"""
from __future__ import annotations

import argparse
import io
import lzma
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://datafeed.dukascopy.com/datafeed"
TICK_STRUCT = struct.Struct(">IIIff")  # 20 bytes per tick
TICK_SIZE = TICK_STRUCT.size

# pip factor por instrumento. XAUUSD usa 1000.
DEFAULT_POINT_FACTOR = {
    "XAUUSD": 1000.0,
    "EURUSD": 100000.0,
    "GBPUSD": 100000.0,
    "USDJPY": 1000.0,
    "BTCUSD": 1000.0,
}


def hour_url(instrument: str, t: datetime) -> str:
    """Dukascopy URL para una hora UTC concreta. month es 0-indexado."""
    return (
        f"{BASE_URL}/{instrument}/{t.year:04d}/{t.month - 1:02d}/"
        f"{t.day:02d}/{t.hour:02d}h_ticks.bi5"
    )


def parse_bi5(payload: bytes, hour_start: datetime, point_factor: float) -> np.ndarray:
    """Decodifica el payload LZMA de Dukascopy y devuelve un structured array."""
    if not payload:
        return np.empty(0, dtype=_STRUCT_DTYPE)
    try:
        raw = lzma.decompress(payload)
    except lzma.LZMAError:
        # algunos bi5 antiguos vienen sin comprimir
        raw = payload

    n = len(raw) // TICK_SIZE
    if n == 0:
        return np.empty(0, dtype=_STRUCT_DTYPE)

    out = np.empty(n, dtype=_STRUCT_DTYPE)
    base_ms = int(hour_start.timestamp() * 1000)
    for i in range(n):
        off, ask_i, bid_i, av, bv = TICK_STRUCT.unpack_from(raw, i * TICK_SIZE)
        out[i]["time_ms"] = base_ms + off
        out[i]["bid"] = bid_i / point_factor
        out[i]["ask"] = ask_i / point_factor
        out[i]["ask_volume"] = av
        out[i]["bid_volume"] = bv
    return out


_STRUCT_DTYPE = np.dtype([
    ("time_ms", np.int64),
    ("bid", np.float64),
    ("ask", np.float64),
    ("ask_volume", np.float32),
    ("bid_volume", np.float32),
])


def fetch_hour(
    instrument: str,
    hour_start: datetime,
    point_factor: float,
    *,
    session: requests.Session,
    timeout: float = 30.0,
    retries: int = 3,
) -> np.ndarray:
    url = hour_url(instrument, hour_start)
    last_err = None
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=timeout)
            if r.status_code == 404:
                # Hora sin datos (mercado cerrado, festivos). Es un caso normal.
                return np.empty(0, dtype=_STRUCT_DTYPE)
            r.raise_for_status()
            return parse_bi5(r.content, hour_start, point_factor)
        except (requests.RequestException, ValueError) as e:
            last_err = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Falló descarga {url} tras {retries} intentos: {last_err}")


def hours_in_range(start: datetime, end: datetime) -> List[datetime]:
    """Lista de horas UTC desde start (inclusive) hasta end (exclusive)."""
    cur = start.replace(minute=0, second=0, microsecond=0)
    out: List[datetime] = []
    while cur < end:
        out.append(cur)
        cur += timedelta(hours=1)
    return out


def download_range(
    instrument: str,
    start: datetime,
    end: datetime,
    *,
    point_factor: Optional[float] = None,
    workers: int = 8,
) -> pd.DataFrame:
    if point_factor is None:
        point_factor = DEFAULT_POINT_FACTOR.get(instrument.upper())
        if point_factor is None:
            raise ValueError(
                f"point_factor no especificado y no hay default para {instrument}. "
                f"Pasalo explicitamente con --point-factor."
            )

    if start.tzinfo is None:
        start = start.replace(tzinfo=timezone.utc)
    if end.tzinfo is None:
        end = end.replace(tzinfo=timezone.utc)

    hours = hours_in_range(start, end)
    print(f"[dukascopy] {instrument} {start.isoformat()} → {end.isoformat()} "
          f"= {len(hours):,} horas a descargar (workers={workers})")

    chunks: List[np.ndarray] = []
    session = requests.Session()
    completed = 0
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {
            ex.submit(fetch_hour, instrument, h, point_factor, session=session): h
            for h in hours
        }
        for fut in as_completed(futs):
            arr = fut.result()
            if arr.size:
                chunks.append(arr)
            completed += 1
            if completed % 100 == 0 or completed == len(hours):
                pct = 100.0 * completed / len(hours)
                rate = completed / max(time.perf_counter() - t0, 1e-6)
                eta = (len(hours) - completed) / max(rate, 1e-6)
                print(
                    f"  [{completed:>5}/{len(hours)}] {pct:5.1f}% | "
                    f"{rate:6.1f} h/s | ETA {eta/60:5.1f} min"
                )

    if not chunks:
        return pd.DataFrame(columns=["time", "bid", "ask", "ask_volume", "bid_volume"])

    arr = np.concatenate(chunks)
    arr.sort(order="time_ms")

    df = pd.DataFrame({
        "time": pd.to_datetime(arr["time_ms"], unit="ms", utc=True),
        "bid": arr["bid"],
        "ask": arr["ask"],
        "ask_volume": arr["ask_volume"],
        "bid_volume": arr["bid_volume"],
    })
    return df


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--instrument", default="XAUUSD")
    ap.add_argument("--from", dest="from_", required=True,
                    help="Fecha inicio inclusive, ISO YYYY-MM-DD (UTC)")
    ap.add_argument("--to", required=True,
                    help="Fecha fin exclusive, ISO YYYY-MM-DD (UTC)")
    ap.add_argument("--out", required=True, help="Path de parquet de salida")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--point-factor", type=float, default=None,
                    help="Override del factor pip. Default por instrumento.")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    start = datetime.fromisoformat(args.from_).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.to).replace(tzinfo=timezone.utc)
    if start >= end:
        raise SystemExit(f"--from ({start}) debe ser anterior a --to ({end})")

    df = download_range(
        args.instrument, start, end,
        point_factor=args.point_factor,
        workers=args.workers,
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"\n✅ {len(df):,} ticks guardados en {out}")
    if not df.empty:
        print(f"   Rango: {df.time.min()} → {df.time.max()}")
        print(f"   bid: [{df.bid.min():.3f}, {df.bid.max():.3f}]  "
              f"ask: [{df.ask.min():.3f}, {df.ask.max():.3f}]")
        print(f"   spread medio: {(df.ask - df.bid).mean():.4f}")
        print(f"   bid_volume medio: {df.bid_volume.mean():.4f}  "
              f"ask_volume medio: {df.ask_volume.mean():.4f}")


if __name__ == "__main__":
    main()
