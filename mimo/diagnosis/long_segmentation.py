#!/usr/bin/env python3
"""
long_segmentation.py

Análisis post-hoc sobre un parquet de holdout para encontrar segmentos
(estado × sesión × hora × magnitud) donde la precisión cruza BE — sin
reentrenar nada.

Pensado tras observar que LONG 201200 globalmente no cruza BE=0.286 pero
en estado RANGE alcanza prec=0.37 (n=51). La hipótesis: hay segmentos
ocultos donde el modelo SÍ tiene edge operable.

Reportes:
  1. Por estado: prec, n, pos_rate, mejor threshold percentil, mejor prec.
  2. Por sesión (UTC hour buckets / EU / US / overlap): mismo análisis.
  3. Por (estado × sesión): combinación más granular.
  4. Por hora cruda (00..23 UTC).
  5. (Opcional) Segmento × magnitud quantile: si filtras por P_mag alto
     dentro del segmento, ¿mejora?

Uso:
    python -m mimo.diagnosis.long_segmentation \\
        --dir-preds artifacts/201200/.../holdout_predictions_201200_long.parquet \\
        --be 0.286 --min-n 50

    # Con join de magnitud:
    python -m mimo.diagnosis.long_segmentation \\
        --dir-preds artifacts/201200/.../holdout_predictions_201200_long.parquet \\
        --mag-preds artifacts/200800/.../holdout_predictions_200800_long.parquet \\
        --be 0.286 --min-n 50 --thr-mag-quantile 0.9
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
                    help="Parquet del modelo direccional (LONG o SHORT) con "
                         "columnas 'time', 'state', 'y_true', 'y_pred_cal'.")
    ap.add_argument("--mag-preds", default=None,
                    help="Opcional: parquet del modelo de magnitud para "
                         "filtrar por P_mag dentro de cada segmento.")
    ap.add_argument("--be", type=float, default=0.286,
                    help="Break-even precisión a cruzar. Default 0.286 "
                         "(tp=2.5/sl=1.0 o tp=2.0/sl=0.8).")
    ap.add_argument("--min-n", type=int, default=50,
                    help="Mínimo de señales por segmento para reportar.")
    ap.add_argument("--prob-col", default="y_pred_cal",
                    help="Columna de probabilidad. Default 'y_pred_cal'.")
    ap.add_argument("--thr-mag-quantile", type=float, default=None,
                    help="Si se pasa con --mag-preds, computa segmentos "
                         "con P_mag >= percentil(quantile) sobre el holdout.")
    ap.add_argument("--top", type=int, default=20,
                    help="Top N segmentos a mostrar en cada tabla.")
    ap.add_argument("--out", default="long_segmentation.csv",
                    help="CSV con todos los segmentos analizados.")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_preds(path: str, prob_col: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    needed = {"time", "y_true", prob_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"{path}: faltan columnas {missing}. "
                         f"Disponibles: {list(df.columns)}")
    df = df[list(needed) + (["state"] if "state" in df.columns else [])].copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.rename(columns={prob_col: "p"})
    return df


def add_session_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Añade columnas de sesión sobre el time UTC del df.

    Sesiones aproximadas (UTC):
      Asia        : 00:00–07:00
      EU_only     : 07:00–13:30  (Londres + Frankfurt sin US)
      EU_US       : 13:30–17:00  (overlap, alta liquidez)
      US_only     : 17:00–21:00
      Post_US     : 21:00–24:00

    También añade hour (0–23) y dow (0=Mon..6=Sun).
    """
    out = df.copy()
    t = pd.to_datetime(out["time"])
    out["hour"] = t.dt.hour
    out["dow"] = t.dt.dayofweek

    # session por hora UTC
    h = out["hour"]
    session = np.where(h < 7, "Asia",
              np.where(h < 13, "EU_only",
              np.where(h < 17, "EU_US",
              np.where(h < 21, "US_only", "Post_US"))))
    # 13:30 cae en EU_only por h<13; refinamos para ser fieles a la apertura
    # de NY pero la granularidad horaria es suficiente para diagnóstico.
    out["session"] = session
    return out


def best_threshold_in_group(p: np.ndarray, y: np.ndarray,
                            grid_q: List[float],
                            be: float, min_n: int) -> dict:
    """
    Sobre un grupo (subset del df), prueba percentiles de p y devuelve la
    mejor combinación (prec, n) con n >= min_n. Si ninguno cumple n_min,
    devuelve el de mayor n con prec ≥ pos_rate (gain trivial).
    """
    if len(p) == 0:
        return {"thr": np.nan, "n": 0, "tp": 0,
                "precision": np.nan, "above_be": False}

    pos_rate = float(np.mean(y))
    out_best = None

    for q in grid_q:
        thr = float(np.quantile(p, q))
        mask = p >= thr
        n = int(mask.sum())
        if n < min_n:
            continue
        tp = int(((y == 1) & mask).sum())
        prec = tp / n if n > 0 else 0.0
        gain = prec - pos_rate
        cand = {
            "thr": thr, "thr_q": q,
            "n": n, "tp": tp, "precision": prec,
            "gain_vs_base": gain,
            "above_be": prec >= be,
        }
        if (out_best is None) or (prec > out_best["precision"]):
            out_best = cand

    if out_best is None:
        # ningún threshold cumple min_n → reportamos el grupo entero
        out_best = {
            "thr": float(np.min(p)), "thr_q": 0.0,
            "n": int(len(p)),
            "tp": int(np.sum(y == 1)),
            "precision": pos_rate,
            "gain_vs_base": 0.0,
            "above_be": pos_rate >= be,
        }
    return out_best


def analyze_segments(df: pd.DataFrame, group_cols: List[str],
                     be: float, min_n: int,
                     grid_q: List[float]) -> pd.DataFrame:
    """Itera grupos por group_cols y reporta best threshold dentro de cada."""
    rows = []
    for keys, g in df.groupby(group_cols, observed=True, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        if len(g) < min_n:
            continue
        p = g["p"].to_numpy()
        y = g["y_true"].to_numpy().astype(int)
        best = best_threshold_in_group(p, y, grid_q, be, min_n)
        rec = {col: k for col, k in zip(group_cols, keys)}
        rec.update({
            "n_total": int(len(g)),
            "pos_rate_base": float(np.mean(y)),
            "best_thr": best["thr"],
            "best_thr_q": best["thr_q"],
            "best_n": best["n"],
            "best_tp": best["tp"],
            "best_prec": best["precision"],
            "gain_vs_base": best["gain_vs_base"],
            "above_be": best["above_be"],
        })
        rows.append(rec)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["above_be", "best_prec", "best_n"],
        ascending=[False, False, False]
    ).reset_index(drop=True)


def print_table(name: str, df: pd.DataFrame, top: int, be: float) -> None:
    print("\n" + "=" * 90)
    print(f"{name}")
    print("=" * 90)
    if df.empty:
        print("(sin segmentos con n >= min_n)")
        return
    cols_show = [c for c in df.columns
                 if c not in ("best_thr_q",)]
    print(df.head(top)[cols_show].to_string(index=False, float_format="%.4f"))
    n_above = int(df["above_be"].sum())
    print(f"\n→ {n_above} de {len(df)} segmentos cruzan BE={be:.3f}.")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    # 1. Cargar direccional
    df = load_preds(args.dir_preds, args.prob_col)
    print(f"[load] dir-preds: {len(df):,} filas | rango P=[{df['p'].min():.4f}, {df['p'].max():.4f}]")

    base_pos_rate = float(df["y_true"].mean())
    print(f"[stats] base_pos_rate = {base_pos_rate:.4f} | BE objetivo = {args.be:.4f} "
          f"(gap = {(args.be - base_pos_rate) * 100:+.2f}pp)")

    # 2. Sesiones
    df = add_session_cols(df)

    # 3. Join opcional con magnitud
    if args.mag_preds:
        mag = load_preds(args.mag_preds, args.prob_col).rename(
            columns={"p": "p_mag", "y_true": "y_true_mag"}
        )[["time", "p_mag"]]
        n_pre = len(df)
        df = df.merge(mag, on="time", how="inner")
        print(f"[merge] dir × mag → {len(df):,} filas (perdidas {n_pre - len(df)})")

        if args.thr_mag_quantile is not None:
            thr_mag = float(np.quantile(df["p_mag"], args.thr_mag_quantile))
            print(f"[mag-filter] aplicando P_mag >= percentil({args.thr_mag_quantile:.2f}) "
                  f"= {thr_mag:.4f}")
            df_filt = df[df["p_mag"] >= thr_mag].copy()
            print(f"[mag-filter] {len(df):,} → {len(df_filt):,} filas tras filtro mag")
            df = df_filt

    # 4. Grid de quantiles para threshold dentro del segmento
    grid_q = [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.95, 0.97]

    # 5. Análisis por dimensión
    if "state" in df.columns:
        seg_state = analyze_segments(df, ["state"], args.be, args.min_n, grid_q)
        print_table("POR ESTADO", seg_state, args.top, args.be)
    else:
        seg_state = pd.DataFrame()
        print("\n⚠️ Sin columna 'state' en el parquet; saltando análisis por estado.")

    seg_session = analyze_segments(df, ["session"], args.be, args.min_n, grid_q)
    print_table("POR SESIÓN UTC", seg_session, args.top, args.be)

    seg_hour = analyze_segments(df, ["hour"], args.be, args.min_n, grid_q)
    print_table("POR HORA UTC", seg_hour, args.top, args.be)

    seg_dow = analyze_segments(df, ["dow"], args.be, args.min_n, grid_q)
    print_table("POR DÍA DE LA SEMANA (0=Mon..6=Sun)", seg_dow, args.top, args.be)

    if "state" in df.columns:
        seg_state_session = analyze_segments(
            df, ["state", "session"], args.be, args.min_n, grid_q
        )
        print_table("POR (ESTADO × SESIÓN)", seg_state_session, args.top, args.be)

    # 6. Veredicto
    print("\n" + "=" * 90)
    print("VEREDICTO")
    print("=" * 90)

    candidatos = []
    for name, t in [
        ("estado", seg_state),
        ("sesión", seg_session),
        ("hora", seg_hour),
        ("dow", seg_dow),
        ("estado×sesión", seg_state_session if "state" in df.columns else pd.DataFrame()),
    ]:
        if t.empty:
            continue
        sub = t[t["above_be"] & (t["best_n"] >= args.min_n)]
        if not sub.empty:
            candidatos.append((name, sub))

    if not candidatos:
        print("✗ Ningún segmento cruza BE con n >= min_n.")
        print("  Ideas a probar:")
        print("  - Subir --min-n no ayuda; bájalo a 30 si quieres ver microsegmentos.")
        print("  - Lanzar con --mag-preds y --thr-mag-quantile 0.85 ó 0.90.")
        print("  - Si tampoco, el techo es del modelo: pivota a multi-task o features nuevas.")
    else:
        print(f"✓ {sum(len(s) for _, s in candidatos)} configs cruzan BE en al menos una "
              f"dimensión.")
        print("  Top combinables (filtrar operativa por intersección de estos segmentos):")
        for name, sub in candidatos:
            print(f"\n  · Por {name}:")
            print(sub.head(5).to_string(index=False, float_format="%.4f"))

    # 7. Guardar todo
    out_path = Path(args.out)
    all_segments = []
    for name, t in [
        ("state", seg_state),
        ("session", seg_session),
        ("hour", seg_hour),
        ("dow", seg_dow),
        ("state_session", seg_state_session if "state" in df.columns else pd.DataFrame()),
    ]:
        if t.empty:
            continue
        t2 = t.copy()
        t2["dimension"] = name
        all_segments.append(t2)
    if all_segments:
        big = pd.concat(all_segments, axis=0, ignore_index=True)
        big.to_csv(out_path, index=False)
        print(f"\n[ok] segmentos guardados: {out_path}")


if __name__ == "__main__":
    main()
