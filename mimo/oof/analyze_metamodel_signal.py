#!/usr/bin/env python3
"""
analyze_metamodel_signal.py

Analisis exploratorio para valorar si un metamodelo de discriminacion de falsos
positivos aporta senal sobre la salida de main_oof_regime_weights_v7.py.

Acepta dos formatos de entrada (autodeteccion):

  1) calibration_dataset_<release>_<side>.parquet  (preferido)
        col: time, state, signal, oof_proba_raw, oof_proba_cal, source
        source ∈ {oof_train, holdout}  -> permite contraste OOF vs holdout

  2) oof_<release>_<side>.parquet  (fallback)
        col: time, state, signal, oof_proba_raw, oof_proba_cal
        sin holdout: se sintetiza source='oof_train' y se omiten secciones de
        holdout. Suficiente para el diagnostico primario de heterogeneidad.

Resolucion de paths (en este orden):
  - --long / --short si se pasan
  - <dir>/calibration_dataset_<release>_<side>.parquet
  - <dir>/oof_<release>_<side>.parquet
  - <dir>/../oof_<release>_<side>.parquet  (la raiz del experiment_tag)

Uso (Windows):
    python -m mimo.oof.analyze_metamodel_signal
    python -m mimo.oof.analyze_metamodel_signal --release 200393
    python -m mimo.oof.analyze_metamodel_signal --dir "C:\\ruta"
    python -m mimo.oof.analyze_metamodel_signal --long oof_200393_long.parquet ^
                                                --short oof_200393_short.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(
    r"/mnt/c/Users/Usuario/Documents/Proyectos/TradingCo_s2/data_sample"
)
DEFAULT_RELEASE = "200393"
THRESHOLD_PCTLS = [70, 80, 90, 95]
MIN_SUPPORT_FOR_HET = 30


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=str, default=str(DEFAULT_DIR))
    ap.add_argument("--release", type=str, default=DEFAULT_RELEASE)
    ap.add_argument("--long", type=str, default=None)
    ap.add_argument("--short", type=str, default=None)
    return ap.parse_args()


def section(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def resolve_input_path(explicit: str | None, base: Path, release: str, side: str) -> Path | None:
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None
    candidates = [
        base / f"calibration_dataset_{release}_{side}.parquet",
        base / f"oof_{release}_{side}.parquet",
        base.parent / f"oof_{release}_{side}.parquet",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"signal", "oof_proba_cal", "state"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {missing}")
    df = df.copy()
    if "source" not in df.columns:
        df["source"] = "oof_train"
        print(f"  [info] {path.name} sin columna 'source'. Asumido source='oof_train'.")
    df["signal"] = pd.to_numeric(df["signal"], errors="coerce")
    df = df.dropna(subset=["signal", "oof_proba_cal", "state", "source"])
    df["signal"] = df["signal"].astype(int)
    df["state"] = df["state"].astype(str)
    df["source"] = df["source"].astype(str)
    return df


def overall_summary(df: pd.DataFrame) -> None:
    section("RESUMEN GLOBAL")
    by_source = (
        df.groupby("source")
        .agg(
            n=("signal", "size"),
            pos_rate=("signal", "mean"),
            proba_mean=("oof_proba_cal", "mean"),
            proba_p50=("oof_proba_cal", "median"),
            proba_p90=("oof_proba_cal", lambda s: float(np.percentile(s, 90))),
        )
        .round(4)
    )
    print(by_source.to_string())

    print("\nPor (source, state):")
    by_ss = (
        df.groupby(["source", "state"])
        .agg(
            n=("signal", "size"),
            pos_rate=("signal", "mean"),
            proba_mean=("oof_proba_cal", "mean"),
        )
        .round(4)
    )
    print(by_ss.to_string())


def decile_edges_from_oof(df_oof: pd.DataFrame) -> np.ndarray:
    edges = np.unique(np.quantile(df_oof["oof_proba_cal"], np.linspace(0, 1, 11)))
    if len(edges) < 2:
        edges = np.array([df_oof["oof_proba_cal"].min(), df_oof["oof_proba_cal"].max() + 1e-9])
    edges = edges.astype(float)
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    return edges


def calibration_by_decile(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    out = []
    for src, g in df.groupby("source"):
        g = g.copy()
        g["decile"] = pd.cut(
            g["oof_proba_cal"], bins=edges, labels=False, include_lowest=True
        )
        agg = (
            g.groupby("decile", dropna=False)
            .agg(
                n=("signal", "size"),
                pos_rate=("signal", "mean"),
                proba_mean=("oof_proba_cal", "mean"),
                proba_min=("oof_proba_cal", "min"),
                proba_max=("oof_proba_cal", "max"),
            )
            .reset_index()
        )
        agg["source"] = src
        out.append(agg)
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True)[
        ["source", "decile", "n", "pos_rate", "proba_mean", "proba_min", "proba_max"]
    ]


def state_decile_table(df: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    rows = []
    for src, g in df.groupby("source"):
        g = g.copy()
        g["decile"] = pd.cut(
            g["oof_proba_cal"], bins=edges, labels=False, include_lowest=True
        )
        agg = (
            g.groupby(["state", "decile"], dropna=False)
            .agg(n=("signal", "size"), tp=("signal", "sum"))
            .reset_index()
        )
        agg["fp"] = agg["n"] - agg["tp"]
        agg["pos_rate"] = agg["tp"] / agg["n"].clip(lower=1)
        agg["source"] = src
        rows.append(agg)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def operating_point_table(df: pd.DataFrame) -> pd.DataFrame:
    df_oof = df[df["source"] == "oof_train"]
    rows = []
    for pct in THRESHOLD_PCTLS:
        if df_oof.empty:
            break
        tau = float(np.percentile(df_oof["oof_proba_cal"], pct))
        for src, g in df.groupby("source"):
            mask = g["oof_proba_cal"] >= tau
            n = int(mask.sum())
            if n == 0:
                continue
            tp = int(g.loc[mask, "signal"].sum())
            fp = n - tp
            total_pos = int(g["signal"].sum())
            rows.append(
                {
                    "source": src,
                    "pct": pct,
                    "tau": round(tau, 6),
                    "n_signals": n,
                    "tp": tp,
                    "fp": fp,
                    "precision": tp / n,
                    "recall": (tp / total_pos) if total_pos > 0 else float("nan"),
                    "fp_per_tp": (fp / tp) if tp > 0 else float("inf"),
                }
            )
    return pd.DataFrame(rows)


def state_breakdown_at_threshold(df: pd.DataFrame, tau: float, source: str) -> pd.DataFrame:
    base = df[df["source"] == source]
    if base.empty:
        return pd.DataFrame()
    pos_total = base.groupby("state")["signal"].sum().rename("pos_total")
    fired = base[base["oof_proba_cal"] >= tau]
    if fired.empty:
        return pd.DataFrame()
    agg = fired.groupby("state").agg(n=("signal", "size"), tp=("signal", "sum"))
    agg["fp"] = agg["n"] - agg["tp"]
    agg["precision"] = agg["tp"] / agg["n"]
    agg = agg.join(pos_total, how="left")
    agg["recall_within_state"] = agg["tp"] / agg["pos_total"].clip(lower=1)
    return agg.sort_values("n", ascending=False)


def heterogeneity(state_at_thr: pd.DataFrame) -> dict:
    if state_at_thr.empty:
        return {"n_states": 0}
    valid = state_at_thr[state_at_thr["n"] >= MIN_SUPPORT_FOR_HET]
    if valid.empty:
        return {"n_states": 0}
    return {
        "n_states": int(len(valid)),
        "precision_min": float(valid["precision"].min()),
        "precision_max": float(valid["precision"].max()),
        "spread": float(valid["precision"].max() - valid["precision"].min()),
        "precision_std": float(valid["precision"].std()),
        "worst_state": str(valid["precision"].idxmin()),
        "best_state": str(valid["precision"].idxmax()),
    }


def lectura_rapida(spreads: list[dict]) -> None:
    if not spreads:
        print("  Sin datos suficientes para juicio.")
        return
    worst = max(spreads, key=lambda r: r.get("spread", 0) or 0)
    s = worst.get("spread")
    if s is None:
        print("  Sin datos suficientes.")
        return
    print(
        f"  spread max de precision entre estados (holdout): {s:.3f} "
        f"(p{worst['pct']}, peor={worst['worst_state']}, mejor={worst['best_state']})"
    )
    if s >= 0.15:
        print("  --> senal heterogenea: un meta-clasificador deberia aportar.")
    elif s >= 0.07:
        print("  --> senal moderada: el meta puede aportar; el margen es estrecho.")
    else:
        print("  --> senal plana: el meta probablemente no compense la complejidad.")


def analyze_side(side: str, path: Path) -> None:
    section(f"SIDE = {side.upper()}  |  {path}")
    if not path.exists():
        print(f"  [skip] no existe: {path}")
        return

    df = load_dataset(path)
    print(f"  filas: {len(df):,}  |  pos_rate global: {df['signal'].mean():.4f}")
    print(f"  fuentes: {sorted(df['source'].unique())}")
    print(f"  estados: {sorted(df['state'].unique())}")

    overall_summary(df)

    df_oof = df[df["source"] == "oof_train"]
    if df_oof.empty:
        print("\n  [warn] no hay filas con source='oof_train'; se aborta.")
        return

    edges = decile_edges_from_oof(df_oof)
    print(f"\nEdges de decil (calculados sobre OOF): {np.round(edges, 4).tolist()}")

    section(f"[{side}] CALIBRACION POR DECIL DE oof_proba_cal")
    cal = calibration_by_decile(df, edges)
    print(cal.round(4).to_string(index=False))

    section(f"[{side}] HEATMAP pos_rate(state x decile)")
    sd = state_decile_table(df, edges)
    pivot = sd.pivot_table(
        index="state", columns=["source", "decile"], values="pos_rate", aggfunc="first"
    )
    print(pivot.round(3).to_string())

    out_csv = path.with_name(f"meta_signal_state_decile_{side}.csv")
    sd.to_csv(out_csv, index=False)
    print(f"\n  -> guardado: {out_csv}")

    section(f"[{side}] PUNTOS DE OPERACION (umbrales = percentiles del proba en OOF)")
    op = operating_point_table(df)
    print(op.round(4).to_string(index=False))

    section(f"[{side}] DESGLOSE POR ESTADO EN HOLDOUT POR UMBRAL")
    spreads = []
    for pct in THRESHOLD_PCTLS:
        tau = float(np.percentile(df_oof["oof_proba_cal"], pct))
        sb = state_breakdown_at_threshold(df, tau=tau, source="holdout")
        if sb.empty:
            continue
        print(f"\n--- tau=p{pct} ({tau:.4f}) ---")
        print(sb.round(4).to_string())
        het = heterogeneity(sb)
        het["pct"] = pct
        het["tau"] = tau
        spreads.append(het)

    if spreads:
        section(f"[{side}] DISPERSION DE PRECISION ENTRE ESTADOS (holdout)")
        print(pd.DataFrame(spreads).round(4).to_string(index=False))

    section(f"[{side}] LECTURA RAPIDA")
    lectura_rapida(spreads)


def main() -> None:
    args = parse_args()
    base = Path(args.dir)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    for side, explicit in (("long", args.long), ("short", args.short)):
        path = resolve_input_path(explicit, base, args.release, side)
        if path is None:
            section(f"SIDE = {side.upper()}  |  (sin parquet)")
            print(
                f"  no se encontro parquet en candidatos:\n"
                f"    {base / f'calibration_dataset_{args.release}_{side}.parquet'}\n"
                f"    {base / f'oof_{args.release}_{side}.parquet'}\n"
                f"    {base.parent / f'oof_{args.release}_{side}.parquet'}\n"
                f"  pasa la ruta con --{side} <path>."
            )
            continue
        analyze_side(side, path)


if __name__ == "__main__":
    main()
