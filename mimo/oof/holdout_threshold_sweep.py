#!/usr/bin/env python3
"""
holdout_threshold_sweep.py

Barre thresholds sobre las predicciones holdout ya guardadas y reporta cuáles
serían deployables (precision >= BE, signal_rate >= mínimo).

Lee:
  ../../artifacts/<release>/oof/<exp_tag>/data/holdout_predictions_<release>_<side>.parquet

Uso:
  python -m mimo.oof.holdout_threshold_sweep \
    --release 202105 \
    --exp-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \
    --be 0.286 \
    --min-sig 0.005

No reentrena nada. Cinco minutos máximo.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd


def sweep_thresholds(
    df: pd.DataFrame,
    *,
    proba_col: str = "y_pred_cal",
    label_col: str = "y_true",
    n_points: int = 400,
    lo: float = 0.05,
    hi: float = 0.99,
) -> pd.DataFrame:
    """Devuelve dataframe con (thr, precision, recall, signal_rate, f1, f025, f05, TP, FP, FN)."""
    y_true = df[label_col].astype(int).to_numpy()
    p = df[proba_col].astype(float).to_numpy()
    n = len(y_true)
    pos_total = int(y_true.sum())

    thrs = np.linspace(lo, hi, n_points)
    rows = []
    for thr in thrs:
        pred = p >= thr
        tp = int((pred & (y_true == 1)).sum())
        fp = int((pred & (y_true == 0)).sum())
        fn = int((~pred & (y_true == 1)).sum())
        signals = tp + fp
        prec = tp / signals if signals > 0 else 0.0
        rec = tp / pos_total if pos_total > 0 else 0.0
        sig = signals / n
        if (prec + rec) > 0:
            f1 = 2 * prec * rec / (prec + rec)
            beta025_2 = 0.25 ** 2
            beta05_2 = 0.5 ** 2
            f025 = (
                (1 + beta025_2) * prec * rec / (beta025_2 * prec + rec)
                if (beta025_2 * prec + rec) > 0
                else 0.0
            )
            f05 = (
                (1 + beta05_2) * prec * rec / (beta05_2 * prec + rec)
                if (beta05_2 * prec + rec) > 0
                else 0.0
            )
        else:
            f1 = f025 = f05 = 0.0

        rows.append(
            dict(
                thr=float(thr),
                precision=prec,
                recall=rec,
                signal_rate=sig,
                f1=f1,
                f0_25=f025,
                f0_5=f05,
                TP=tp,
                FP=fp,
                FN=fn,
            )
        )

    return pd.DataFrame(rows)


def find_deployable(
    sweep: pd.DataFrame, *, be: float, min_sig: float
) -> pd.DataFrame:
    return sweep[(sweep.precision >= be) & (sweep.signal_rate >= min_sig)].copy()


def report_side(
    side: str,
    df: pd.DataFrame,
    *,
    be: float,
    min_sig: float,
    proba_col: str,
) -> Dict[str, Optional[float]]:
    print("=" * 78)
    print(f"  {side.upper()}  (proba_col={proba_col}, n_rows={len(df):,})")
    print("=" * 78)

    pos_rate = float(df.y_true.mean())
    print(f"  base rate (pos_rate)        : {pos_rate:.4f}")
    print(f"  break-even target precision : {be:.3f}")
    print(f"  signal_rate mínimo aceptado : {min_sig:.4f} "
          f"(~{int(min_sig * len(df))} señales en holdout)")
    print()

    sweep = sweep_thresholds(df, proba_col=proba_col)

    # Mejor F0.25 del barrido (sanity check vs el que reportó el trainer)
    best_f025 = sweep.loc[sweep.f0_25.idxmax()]
    print(
        f"  [best F0.25]   thr={best_f025.thr:.4f}  "
        f"prec={best_f025.precision:.3f}  rec={best_f025.recall:.3f}  "
        f"sig={best_f025.signal_rate:.4f}  TP={int(best_f025.TP)}  FP={int(best_f025.FP)}"
    )

    # Mejor F0.5
    best_f05 = sweep.loc[sweep.f0_5.idxmax()]
    print(
        f"  [best F0.5]    thr={best_f05.thr:.4f}  "
        f"prec={best_f05.precision:.3f}  rec={best_f05.recall:.3f}  "
        f"sig={best_f05.signal_rate:.4f}  TP={int(best_f05.TP)}  FP={int(best_f05.FP)}"
    )

    # Mejor F1
    best_f1 = sweep.loc[sweep.f1.idxmax()]
    print(
        f"  [best F1]      thr={best_f1.thr:.4f}  "
        f"prec={best_f1.precision:.3f}  rec={best_f1.recall:.3f}  "
        f"sig={best_f1.signal_rate:.4f}  TP={int(best_f1.TP)}  FP={int(best_f1.FP)}"
    )

    print()
    deploy = find_deployable(sweep, be=be, min_sig=min_sig)
    if deploy.empty:
        print(f"  ❌ NO hay threshold con precision >= {be:.3f} y signal_rate >= {min_sig:.4f}.")
        # Reportar el mejor punto que cumple solo precision (sin restricción de sig_rate)
        only_prec = sweep[sweep.precision >= be]
        if not only_prec.empty:
            best_only = only_prec.loc[only_prec.signal_rate.idxmax()]
            print(
                f"     Mejor con precision >= BE pero sig < {min_sig:.4f}: "
                f"thr={best_only.thr:.4f} prec={best_only.precision:.3f} "
                f"sig={best_only.signal_rate:.4f} (TP={int(best_only.TP)}, FP={int(best_only.FP)})"
            )
        # Reportar el threshold que más se acerca a BE (top precision con sig >= min_sig)
        with_sig = sweep[sweep.signal_rate >= min_sig]
        if not with_sig.empty:
            top_prec = with_sig.loc[with_sig.precision.idxmax()]
            print(
                f"     Mejor precision con sig >= {min_sig:.4f}: "
                f"thr={top_prec.thr:.4f} prec={top_prec.precision:.3f} "
                f"sig={top_prec.signal_rate:.4f} (deficit -{(be - top_prec.precision)*100:.1f}pp)"
            )
        result_thr = None
        result_prec = None
    else:
        # Dentro de los deployables, escogemos el de mayor signal_rate (más operaciones)
        # mientras manteniendo precision >= BE.
        chosen = deploy.loc[deploy.signal_rate.idxmax()]
        # Y otro que maximice precision absoluta con sig >= min_sig
        max_prec = deploy.loc[deploy.precision.idxmax()]
        print(f"  ✅ DEPLOYABLE: {len(deploy)} thresholds cumplen.")
        print()
        print(f"     · max signal_rate dentro de deployables:")
        print(
            f"       thr={chosen.thr:.4f}  prec={chosen.precision:.3f}  "
            f"rec={chosen.recall:.3f}  sig={chosen.signal_rate:.4f}  "
            f"TP={int(chosen.TP)}  FP={int(chosen.FP)}"
        )
        print(f"     · max precision dentro de deployables:")
        print(
            f"       thr={max_prec.thr:.4f}  prec={max_prec.precision:.3f}  "
            f"rec={max_prec.recall:.3f}  sig={max_prec.signal_rate:.4f}  "
            f"TP={int(max_prec.TP)}  FP={int(max_prec.FP)}"
        )
        result_thr = float(chosen.thr)
        result_prec = float(chosen.precision)

    print()
    return {"deployable_thr": result_thr, "deployable_prec": result_prec}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--exp-tag", required=True,
                    help="Nombre del subdir bajo artifacts/<release>/oof/")
    ap.add_argument("--artifacts-root", default="../../artifacts",
                    help="Raíz desde la que se cuelga <release>/oof/<exp_tag>")
    ap.add_argument("--be", type=float, default=0.286,
                    help="Break-even precision objetivo")
    ap.add_argument("--min-sig", type=float, default=0.005,
                    help="Signal rate mínimo aceptado")
    ap.add_argument("--proba-col", default="y_pred_cal",
                    choices=["y_pred_cal", "y_pred_raw"],
                    help="Columna de probabilidad a usar")
    ap.add_argument("--save-csv", action="store_true",
                    help="Guarda los barridos completos en CSV junto al parquet")
    args = ap.parse_args()

    base = Path(args.artifacts_root) / args.release / "oof" / args.exp_tag / "data"
    if not base.exists():
        raise SystemExit(f"❌ No existe: {base}")

    print(f"\n📂 Cargando predicciones holdout desde: {base}\n")

    summary = {}
    for side in ("long", "short"):
        path = base / f"holdout_predictions_{args.release}_{side}.parquet"
        if not path.exists():
            print(f"⚠️  No encontrado: {path.name}, saltando.")
            continue
        df = pd.read_parquet(path)
        if args.proba_col not in df.columns:
            print(f"⚠️  Columna '{args.proba_col}' no en {path.name}. "
                  f"Cols={list(df.columns)}")
            continue

        info = report_side(
            side, df,
            be=args.be, min_sig=args.min_sig,
            proba_col=args.proba_col,
        )
        summary[side] = info

        if args.save_csv:
            sweep = sweep_thresholds(df, proba_col=args.proba_col)
            out_csv = base / f"threshold_sweep_{args.release}_{side}.csv"
            sweep.to_csv(out_csv, index=False)
            print(f"  📁 Sweep guardado: {out_csv}")
            print()

    print("=" * 78)
    print("  VEREDICTO")
    print("=" * 78)
    for side, info in summary.items():
        if info["deployable_thr"] is None:
            print(f"  {side.upper():5s}: ❌ NO deployable con BE={args.be} y sig>={args.min_sig}")
        else:
            print(
                f"  {side.upper():5s}: ✅ deployable @ thr={info['deployable_thr']:.4f} "
                f"prec={info['deployable_prec']:.3f}"
            )
    print()


if __name__ == "__main__":
    main()
