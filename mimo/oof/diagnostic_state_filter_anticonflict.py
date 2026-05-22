#!/usr/bin/env python3
"""
diagnostic_state_filter_anticonflict.py

Aplica dos filtros sobre las predicciones holdout ya guardadas, sin reentrenar:
  A) Filtro por régimen (state whitelist / blacklist)
  C) Anti-conflict gate (|P_long - P_short| > margen)

Lee ambos parquets (long/short) y los alinea por `time`. Reporta precisión,
recall, signal_rate y deployabilidad (precision >= BE) bajo cada escenario.

Uso:
  python -m mimo.oof.diagnostic_state_filter_anticonflict \
    --release 202300 \
    --exp-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \
    --be 0.286 \
    --min-sig 0.005

No reentrena. Cinco minutos.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd


BETA025_2 = 0.25 ** 2


def _metrics(y_true: np.ndarray, pred: np.ndarray) -> dict:
    pos_total = int(y_true.sum())
    n = len(y_true)
    tp = int((pred & (y_true == 1)).sum())
    fp = int((pred & (y_true == 0)).sum())
    signals = tp + fp
    prec = tp / signals if signals > 0 else 0.0
    rec = tp / pos_total if pos_total > 0 else 0.0
    sig = signals / n if n > 0 else 0.0
    f025 = (
        (1 + BETA025_2) * prec * rec / (BETA025_2 * prec + rec)
        if (BETA025_2 * prec + rec) > 0 else 0.0
    )
    return dict(precision=prec, recall=rec, signal_rate=sig,
                f0_25=f025, TP=tp, FP=fp, signals=signals,
                pos_total=pos_total, n=n)


def best_threshold(
    proba: np.ndarray,
    y_true: np.ndarray,
    *,
    objective: str = "f0_25",
    min_sig: float = 0.005,
    n_points: int = 400,
    lo: float = 0.05,
    hi: float = 0.99,
) -> dict:
    """Sweep clásico, devuelve fila con métrica óptima + thr."""
    thrs = np.linspace(lo, hi, n_points)
    best = None
    for thr in thrs:
        pred = proba >= thr
        m = _metrics(y_true, pred)
        if m["signal_rate"] < min_sig:
            continue
        score = m[objective]
        if best is None or score > best["_score"]:
            best = {**m, "thr": float(thr), "_score": score}
    if best is None:
        return dict(thr=float("nan"), precision=0.0, recall=0.0,
                    signal_rate=0.0, f0_25=0.0, TP=0, FP=0,
                    signals=0, pos_total=int(y_true.sum()), n=len(y_true))
    best.pop("_score", None)
    return best


def per_state_table(df: pd.DataFrame, *, thr: float, proba_col: str) -> pd.DataFrame:
    """Precisión/recall/signal_rate por estado al threshold dado."""
    pred = df[proba_col].to_numpy() >= thr
    y = df["y_true"].to_numpy().astype(int)
    rows = []
    for state, sub in df.assign(pred=pred).groupby("state", sort=False):
        sub_y = sub.y_true.to_numpy().astype(int)
        sub_pred = sub.pred.to_numpy()
        m = _metrics(sub_y, sub_pred)
        rows.append(dict(state=state, n=m["n"], pos_rate=sub_y.mean(),
                         signals=m["signals"], precision=m["precision"],
                         recall=m["recall"], signal_rate=m["signal_rate"],
                         TP=m["TP"], FP=m["FP"]))
    return (pd.DataFrame(rows)
            .sort_values("precision", ascending=False)
            .reset_index(drop=True))


def evaluate_with_whitelist(
    df: pd.DataFrame,
    whitelist: Optional[Iterable[str]],
    *,
    thr: float,
    proba_col: str,
) -> dict:
    """Evalúa precisión global tras filtrar por estados en whitelist (None = sin filtro)."""
    proba = df[proba_col].to_numpy()
    y = df["y_true"].to_numpy().astype(int)
    pred = proba >= thr
    if whitelist is not None:
        mask_state = df["state"].isin(list(whitelist)).to_numpy()
        pred = pred & mask_state
    return _metrics(y, pred)


def find_best_whitelist(
    df: pd.DataFrame,
    *,
    thr: float,
    proba_col: str,
    be: float,
    min_sig: float,
) -> dict:
    """
    Búsqueda greedy: arranca con todos los estados y va eliminando el peor por
    precisión hasta que (a) la precisión global supere BE manteniendo sig>=min_sig
    o (b) se agoten estados.
    """
    states = list(df["state"].unique())
    current = set(states)
    history = []
    while current:
        m = evaluate_with_whitelist(df, current, thr=thr, proba_col=proba_col)
        history.append({"states": tuple(sorted(current)), **m})
        if m["precision"] >= be and m["signal_rate"] >= min_sig:
            break
        # encontrar estado cuya eliminación más sube la precisión global
        best_drop, best_prec = None, m["precision"]
        for s in current:
            cand = current - {s}
            if not cand:
                continue
            mc = evaluate_with_whitelist(df, cand, thr=thr, proba_col=proba_col)
            if mc["signal_rate"] < min_sig:
                continue
            if mc["precision"] > best_prec:
                best_prec = mc["precision"]
                best_drop = s
        if best_drop is None:
            break
        current.remove(best_drop)
    return {"final": history[-1], "history": history}


def evaluate_anticonflict(
    df_long: pd.DataFrame,
    df_short: pd.DataFrame,
    *,
    thr_long: float,
    thr_short: float,
    margin: float,
    proba_col: str,
) -> dict:
    """
    Une long+short por time. Para cada lado, requiere prob >= thr Y
    (P_lado - P_otro_lado) >= margin.
    """
    join = df_long[["time", "state", "y_true", proba_col]].rename(
        columns={"y_true": "y_long", proba_col: "p_long"}
    ).merge(
        df_short[["time", "y_true", proba_col]].rename(
            columns={"y_true": "y_short", proba_col: "p_short"}
        ),
        on="time", how="inner",
    )
    margin_l = (join["p_long"] - join["p_short"]).to_numpy() >= margin
    margin_s = (join["p_short"] - join["p_long"]).to_numpy() >= margin

    pred_long = (join["p_long"].to_numpy() >= thr_long) & margin_l
    pred_short = (join["p_short"].to_numpy() >= thr_short) & margin_s

    m_long = _metrics(join["y_long"].to_numpy().astype(int), pred_long)
    m_short = _metrics(join["y_short"].to_numpy().astype(int), pred_short)

    return {"long": m_long, "short": m_short, "n_aligned": len(join)}


def evaluate_combined(
    df_long: pd.DataFrame,
    df_short: pd.DataFrame,
    *,
    thr_long: float,
    thr_short: float,
    whitelist_long: Optional[Iterable[str]],
    whitelist_short: Optional[Iterable[str]],
    margin: float,
    proba_col: str,
) -> dict:
    """A + C combinados."""
    join = df_long[["time", "state", "y_true", proba_col]].rename(
        columns={"y_true": "y_long", proba_col: "p_long"}
    ).merge(
        df_short[["time", "y_true", proba_col]].rename(
            columns={"y_true": "y_short", proba_col: "p_short"}
        ),
        on="time", how="inner",
    )

    margin_l = (join["p_long"] - join["p_short"]).to_numpy() >= margin
    margin_s = (join["p_short"] - join["p_long"]).to_numpy() >= margin

    pred_long = (join["p_long"].to_numpy() >= thr_long) & margin_l
    pred_short = (join["p_short"].to_numpy() >= thr_short) & margin_s

    if whitelist_long is not None:
        pred_long &= join["state"].isin(list(whitelist_long)).to_numpy()
    if whitelist_short is not None:
        pred_short &= join["state"].isin(list(whitelist_short)).to_numpy()

    m_long = _metrics(join["y_long"].to_numpy().astype(int), pred_long)
    m_short = _metrics(join["y_short"].to_numpy().astype(int), pred_short)
    return {"long": m_long, "short": m_short, "n_aligned": len(join)}


def _print_metrics(label: str, m: dict, *, be: float) -> None:
    flag = "✅" if m["precision"] >= be else "❌"
    print(f"  {flag} {label:42s} prec={m['precision']:.3f} "
          f"rec={m['recall']:.3f} sig={m['signal_rate']:.4f} "
          f"TP={m['TP']:>4d} FP={m['FP']:>4d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--exp-tag", required=True)
    ap.add_argument("--artifacts-root", default="../../artifacts")
    ap.add_argument("--be", type=float, default=0.286)
    ap.add_argument("--min-sig", type=float, default=0.005)
    ap.add_argument("--proba-col", default="y_pred_cal",
                    choices=["y_pred_cal", "y_pred_raw"])
    ap.add_argument("--margins", default="0.00,0.02,0.04,0.06,0.08,0.10",
                    help="Lista de márgenes para anti-conflict gate")
    args = ap.parse_args()

    base = Path(args.artifacts_root) / args.release / "oof" / args.exp_tag / "data"
    if not base.exists():
        raise SystemExit(f"❌ No existe: {base}")

    paths = {side: base / f"holdout_predictions_{args.release}_{side}.parquet"
             for side in ("long", "short")}
    for side, p in paths.items():
        if not p.exists():
            raise SystemExit(f"❌ No existe: {p}")

    print(f"\n📂 Cargando {paths['long'].name} y {paths['short'].name}\n")
    df_long = pd.read_parquet(paths["long"])
    df_short = pd.read_parquet(paths["short"])

    for name, df in [("long", df_long), ("short", df_short)]:
        for col in ("time", "state", "y_true", args.proba_col):
            if col not in df.columns:
                raise SystemExit(f"❌ '{col}' no en parquet {name}. "
                                 f"Cols={list(df.columns)}")

    # ---------- Baseline ----------
    print("=" * 78)
    print("  BASELINE (sin filtros, mejor F0.25 por lado)")
    print("=" * 78)
    base_long = best_threshold(
        df_long[args.proba_col].to_numpy(),
        df_long["y_true"].to_numpy().astype(int),
        min_sig=args.min_sig,
    )
    base_short = best_threshold(
        df_short[args.proba_col].to_numpy(),
        df_short["y_true"].to_numpy().astype(int),
        min_sig=args.min_sig,
    )
    print(f"  LONG  thr={base_long['thr']:.4f} prec={base_long['precision']:.3f} "
          f"rec={base_long['recall']:.3f} sig={base_long['signal_rate']:.4f} "
          f"TP={base_long['TP']} FP={base_long['FP']}")
    print(f"  SHORT thr={base_short['thr']:.4f} prec={base_short['precision']:.3f} "
          f"rec={base_short['recall']:.3f} sig={base_short['signal_rate']:.4f} "
          f"TP={base_short['TP']} FP={base_short['FP']}")
    print(f"  BE objetivo: {args.be:.3f}")

    thr_l, thr_s = base_long["thr"], base_short["thr"]

    # ---------- A: per-state breakdown ----------
    print("\n" + "=" * 78)
    print(f"  A) PRECISIÓN POR ESTADO @ thr_long={thr_l:.4f}")
    print("=" * 78)
    tab_l = per_state_table(df_long, thr=thr_l, proba_col=args.proba_col)
    print(tab_l.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    print("\n" + "=" * 78)
    print(f"  A) PRECISIÓN POR ESTADO @ thr_short={thr_s:.4f}")
    print("=" * 78)
    tab_s = per_state_table(df_short, thr=thr_s, proba_col=args.proba_col)
    print(tab_s.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    # ---------- A: greedy whitelist ----------
    print("\n" + "=" * 78)
    print("  A) GREEDY STATE WHITELIST (elimina iterativamente el peor estado)")
    print("=" * 78)
    res_l = find_best_whitelist(df_long, thr=thr_l, proba_col=args.proba_col,
                                be=args.be, min_sig=args.min_sig)
    res_s = find_best_whitelist(df_short, thr=thr_s, proba_col=args.proba_col,
                                be=args.be, min_sig=args.min_sig)
    print("\n  LONG:")
    for h in res_l["history"]:
        flag = "✅" if h["precision"] >= args.be else "❌"
        print(f"    {flag} states={list(h['states'])}")
        print(f"        prec={h['precision']:.3f} rec={h['recall']:.3f} "
              f"sig={h['signal_rate']:.4f} TP={h['TP']} FP={h['FP']}")
    print("\n  SHORT:")
    for h in res_s["history"]:
        flag = "✅" if h["precision"] >= args.be else "❌"
        print(f"    {flag} states={list(h['states'])}")
        print(f"        prec={h['precision']:.3f} rec={h['recall']:.3f} "
              f"sig={h['signal_rate']:.4f} TP={h['TP']} FP={h['FP']}")

    wl_long = list(res_l["final"]["states"])
    wl_short = list(res_s["final"]["states"])

    # ---------- C: anti-conflict gate ----------
    print("\n" + "=" * 78)
    print("  C) ANTI-CONFLICT GATE (|P_long - P_short| >= margen)")
    print("=" * 78)
    margins = [float(x) for x in args.margins.split(",")]
    print(f"  Probando márgenes: {margins}")
    print()
    for m in margins:
        out = evaluate_anticonflict(df_long, df_short,
                                    thr_long=thr_l, thr_short=thr_s,
                                    margin=m, proba_col=args.proba_col)
        _print_metrics(f"long  margin={m:.2f}", out["long"], be=args.be)
        _print_metrics(f"short margin={m:.2f}", out["short"], be=args.be)
        print()

    # ---------- A + C combinados ----------
    print("=" * 78)
    print("  A + C COMBINADOS  (whitelist + anti-conflict)")
    print("=" * 78)
    for m in margins:
        out = evaluate_combined(df_long, df_short,
                                thr_long=thr_l, thr_short=thr_s,
                                whitelist_long=wl_long, whitelist_short=wl_short,
                                margin=m, proba_col=args.proba_col)
        _print_metrics(f"long  margin={m:.2f} (wl={wl_long})",
                       out["long"], be=args.be)
        _print_metrics(f"short margin={m:.2f} (wl={wl_short})",
                       out["short"], be=args.be)
        print()


if __name__ == "__main__":
    main()
