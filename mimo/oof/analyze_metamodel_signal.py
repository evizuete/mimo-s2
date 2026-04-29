#!/usr/bin/env python3
"""
analyze_metamodel_signal.py

Analisis exploratorio sobre los calibration_dataset_*.parquet generados por
main_oof_regime_weights_v7.py para valorar si un metamodelo de discriminacion
de falsos positivos aporta senal.

Lee:
    <dir>/calibration_dataset_<release>_long.parquet
    <dir>/calibration_dataset_<release>_short.parquet

Produce por pantalla:
    - Resumen global y por (source, state)
    - Calibracion por decil de oof_proba_cal (en OOF y en holdout)
    - Heatmap pos_rate por (state x decil)
    - Punto de operacion en umbrales p70/p80/p90/p95 (precision, recall, fp/tp)
    - Desglose por estado en holdout para cada umbral
    - Dispersion de precision entre estados (la senal clave para el meta)
    - Lectura rapida con recomendacion

Y un CSV junto a cada parquet:
    meta_signal_state_decile_<side>.csv

Uso (Windows, valores por defecto apuntan a tu artifact):
    python analyze_metamodel_signal.py
    python analyze_metamodel_signal.py --release 200393
    python analyze_metamodel_signal.py --dir "C:\\ruta\\al\\data"
    python analyze_metamodel_signal.py --long ".\\cal_long.parquet" --short ".\\cal_short.parquet"
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_DIR = Path(
    r"C:\Users\Usuario\Documents\Proyectos\TradingCo_s2_v28\artifacts\200393\oof"
    r"\rw_both_Lbaseline_h10_Strend_robustness_v1_h10\data"
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


def load_dataset(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    required = {"signal", "oof_proba_cal", "state", "source"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: faltan columnas {missing}")
    df = df.copy()
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
    long_path = (
        Path(args.long)
        if args.long
        else base / f"calibration_dataset_{args.release}_long.parquet"
    )
    short_path = (
        Path(args.short)
        if args.short
        else base / f"calibration_dataset_{args.release}_short.parquet"
    )

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    analyze_side("long", long_path)
    analyze_side("short", short_path)


if __name__ == "__main__":
    main()
