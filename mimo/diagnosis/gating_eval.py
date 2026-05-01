#!/usr/bin/env python3
"""
gating_eval.py

Evalúa el gating del modelo direccional (binario long/short) con un modelo de
magnitud (binario direction-agnostic) sobre el mismo holdout.

Idea: el direccional propone dirección, el de magnitud filtra periodos sin
movimiento. La operación se toma solo si AMBAS señales coinciden:

    take = (P_dir >= thr_dir) AND (P_mag >= thr_mag)

Mide:
  - precisión y signal_rate del gating sobre el grid de (thr_dir, thr_mag).
  - comparativa contra el direccional sin gate al mismo signal_rate.
  - desglose por estado del gate ganador.

Uso:
    python -m mimo.diagnosis.gating_eval \\
        --dir-preds artifacts/200602/oof/rw_both_Lbaseline_h5_Sbaseline_h5/data/holdout_predictions_200602_short.parquet \\
        --mag-preds artifacts/200800/oof/rw_long_Lbaseline_h5_Sbaseline_h5/data/holdout_predictions_200800_long.parquet \\
        --min-signals 50 --top 15
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir-preds", required=True,
                    help="Parquet con predicciones holdout del modelo direccional "
                         "(LONG o SHORT) — release 200602 SHORT recomendado.")
    ap.add_argument("--mag-preds", required=True,
                    help="Parquet con predicciones holdout del modelo de magnitud "
                         "— release 200800 LONG.")
    ap.add_argument("--min-signals", type=int, default=50,
                    help="Mínimo de señales por config para que cuente. Default 50.")
    ap.add_argument("--top", type=int, default=15,
                    help="Top N configs a mostrar en el ranking.")
    ap.add_argument("--thr-dir-grid", type=str, default=None,
                    help="Lista CSV de umbrales direccionales. Si None, usa "
                         "percentiles 50,60,70,80,85,90,92,95.")
    ap.add_argument("--thr-mag-grid", type=str, default=None,
                    help="Lista CSV de umbrales magnitud. Si None, usa "
                         "percentiles 30,40,50,60,70,80,85,90.")
    ap.add_argument("--prob-col", default="y_pred_cal",
                    help="Columna de probabilidad a usar (default y_pred_cal).")
    ap.add_argument("--out", default="gating_grid.csv",
                    help="CSV de salida con el grid completo.")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_preds(path: str, prob_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"time", "y_true", prob_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}. Disponibles: {list(df.columns)}")
    df = df[["time", "y_true", prob_col] + (["state"] if "state" in df.columns else [])].copy()
    df["time"] = pd.to_datetime(df["time"])
    return df


def precision_at(df: pd.DataFrame, mask: np.ndarray) -> tuple[int, int, float]:
    n = int(mask.sum())
    if n == 0:
        return 0, 0, 0.0
    tp = int(((df["y_true"] == 1) & mask).sum())
    return tp, n, tp / n


def parse_grid(spec: Optional[str], default_percs: List[float],
               series: pd.Series) -> List[float]:
    if spec is not None:
        return [float(x) for x in spec.split(",") if x.strip()]
    arr = series.dropna().to_numpy()
    return sorted({float(np.percentile(arr, p)) for p in default_percs})


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    # 1. Cargar
    df_dir = load_preds(args.dir_preds, args.prob_col).rename(columns={args.prob_col: "p_dir"})
    df_mag = load_preds(args.mag_preds, args.prob_col).rename(columns={args.prob_col: "p_mag"})

    print(f"[load] direccional: {len(df_dir):,} filas | rango P=[{df_dir['p_dir'].min():.4f}, {df_dir['p_dir'].max():.4f}]")
    print(f"[load] magnitud:    {len(df_mag):,} filas | rango P=[{df_mag['p_mag'].min():.4f}, {df_mag['p_mag'].max():.4f}]")

    # 2. Merge por tiempo. y_true viene del direccional (es el target a optimizar).
    keep_dir = ["time", "p_dir", "y_true"] + (["state"] if "state" in df_dir.columns else [])
    keep_mag = ["time", "p_mag"]
    merged = pd.merge(df_dir[keep_dir], df_mag[keep_mag], on="time", how="inner")
    print(f"[merge] inner join → {len(merged):,} filas")

    if len(merged) == 0:
        raise SystemExit("Inner join vacío. ¿Las predicciones son del mismo holdout y mismo base_tf?")

    base_pos_rate = float(merged["y_true"].mean())
    pos_total = int(merged["y_true"].sum())
    print(f"[stats] base_pos_rate (direccional) = {base_pos_rate:.4f} ({pos_total:,}/{len(merged):,})")

    # 3. Construir grids
    thr_dir_grid = parse_grid(args.thr_dir_grid, [50, 60, 70, 80, 85, 90, 92, 95], merged["p_dir"])
    thr_mag_grid = parse_grid(args.thr_mag_grid, [30, 40, 50, 60, 70, 80, 85, 90], merged["p_mag"])
    print(f"[grid] thr_dir ({len(thr_dir_grid)}): {[round(t, 4) for t in thr_dir_grid]}")
    print(f"[grid] thr_mag ({len(thr_mag_grid)}): {[round(t, 4) for t in thr_mag_grid]}")

    # 4. Baseline: solo direccional
    print("\n" + "=" * 80)
    print("BASELINE — solo direccional, sin gate")
    print("=" * 80)
    base_rows = []
    for td in thr_dir_grid:
        mask = (merged["p_dir"] >= td).to_numpy()
        tp, n, p = precision_at(merged, mask)
        if n >= args.min_signals:
            base_rows.append({
                "thr_dir": td, "thr_mag": np.nan,
                "n_signals": n, "tp": tp,
                "precision": p,
                "recall": tp / pos_total if pos_total else 0.0,
                "signal_rate": n / len(merged),
            })
    base_df = pd.DataFrame(base_rows)
    if base_df.empty:
        print("⚠️ Ningún umbral direccional supera min_signals. Baja --min-signals.")
    else:
        print(base_df.round(4).to_string(index=False))

    # 5. Grid de gating completo
    print("\n" + "=" * 80)
    print(f"GATED — top {args.top} por precision (min_signals={args.min_signals})")
    print("=" * 80)
    rows = []
    for td in thr_dir_grid:
        mask_d = (merged["p_dir"] >= td).to_numpy()
        for tm in thr_mag_grid:
            mask = mask_d & (merged["p_mag"] >= tm).to_numpy()
            n = int(mask.sum())
            if n < args.min_signals:
                continue
            tp = int(((merged["y_true"] == 1) & mask).sum())
            rows.append({
                "thr_dir": td, "thr_mag": tm,
                "n_signals": n, "tp": tp,
                "precision": tp / n if n > 0 else 0.0,
                "recall": tp / pos_total if pos_total else 0.0,
                "signal_rate": n / len(merged),
            })
    grid_df = pd.DataFrame(rows)
    if grid_df.empty:
        print("⚠️ Ninguna combinación supera min_signals. Baja --min-signals "
              "o ajusta los grids.")
        return

    top_prec = grid_df.nlargest(args.top, "precision")
    print(top_prec.round(4).to_string(index=False))

    # 6. Comparativa: best gated vs baseline al mismo signal_rate
    print("\n" + "=" * 80)
    print("MEJOR GATING vs DIRECCIONAL (sin gate) AL MISMO signal_rate")
    print("=" * 80)
    best = top_prec.iloc[0]
    if base_df.empty:
        print("Sin baseline comparable.")
    else:
        base_df_cmp = base_df.copy()
        base_df_cmp["sig_diff"] = (base_df_cmp["signal_rate"] - best["signal_rate"]).abs()
        cmp = base_df_cmp.nsmallest(1, "sig_diff").iloc[0]
        delta_p = best["precision"] - cmp["precision"]
        print(f"  Gated  : thr_dir={best['thr_dir']:.4f} thr_mag={best['thr_mag']:.4f} | "
              f"n={int(best['n_signals'])} prec={best['precision']:.4f} sig={best['signal_rate']:.4f}")
        print(f"  Direct : thr_dir={cmp['thr_dir']:.4f} (no gate)               | "
              f"n={int(cmp['n_signals'])} prec={cmp['precision']:.4f} sig={cmp['signal_rate']:.4f}")
        print(f"  Δprec   : {delta_p:+.4f} ({delta_p * 100:+.2f}pp)")
        if delta_p > 0:
            print("  → ✓ El gate mejora la precisión a igual cobertura.")
        else:
            print("  → ✗ El gate NO mejora.")

    # 7. Veredicto vs BE (asumimos asimétrico tp=2.5/sl=1.0 → BE=0.286)
    BE = 0.286
    above_BE = grid_df[grid_df["precision"] >= BE]
    print("\n" + "=" * 80)
    print(f"CONFIGS QUE CRUZAN BE = {BE:.3f} (tp=2.5/sl=1.0)")
    print("=" * 80)
    if above_BE.empty:
        print(f"⚠️ Ninguna combinación con prec ≥ {BE:.3f} (con n ≥ {args.min_signals}).")
        max_p = float(grid_df["precision"].max())
        print(f"   Máxima precisión observada: {max_p:.4f} (gap a BE: {(BE - max_p) * 100:.2f}pp)")
    else:
        print(f"✓ {len(above_BE)} combinaciones cruzan BE. Top por n_signals:")
        print(above_BE.nlargest(args.top, "n_signals").round(4).to_string(index=False))

    # 8. Per-state breakdown del best gated
    if "state" in merged.columns:
        td, tm = float(best["thr_dir"]), float(best["thr_mag"])
        mask = (merged["p_dir"] >= td).to_numpy() & (merged["p_mag"] >= tm).to_numpy()
        sel = merged.loc[mask].copy()
        if not sel.empty:
            print("\n" + "=" * 80)
            print(f"PER-STATE — best gated (thr_dir={td:.4f}, thr_mag={tm:.4f})")
            print("=" * 80)
            grp = (
                sel.groupby("state", observed=True)["y_true"]
                .agg(["count", "sum", "mean"])
                .rename(columns={"count": "n", "sum": "tp", "mean": "precision"})
                .sort_values("precision", ascending=False)
            )
            grp["above_BE"] = grp["precision"] >= BE
            print(grp.round(4).to_string())

    # 9. Guardar grid completo
    out_path = Path(args.out)
    grid_df.sort_values("precision", ascending=False).to_csv(out_path, index=False)
    print(f"\n[ok] grid completo guardado: {out_path}")


if __name__ == "__main__":
    main()
