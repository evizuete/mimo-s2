#!/usr/bin/env python3
"""
feature_importance_permutation.py

Permutation importance sobre el modelo multitask ya entrenado. Para cada
feature en sequence_short / sequence_long / context, baraja la columna a lo
largo del eje batch y mide la caida de AUC-PR (LONG y SHORT). Las features
con caida < threshold se consideran ruido / redundantes y son candidatas a
eliminar para entrenar una release reducida.

Uso:
    python -m mimo.oof.feature_importance_permutation \\
        --release 202200 \\
        --exp-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
        --base-tf 5min \\
        --target-type multitask \\
        --label-horizon-long 3 --label-horizon-short 3 \\
        --train-from 2024-01-01 --train-to 2025-10-30 \\
        --holdout-from 2025-11-01 --holdout-to 2026-05-02 \\
        --n-permutations 1 \\
        --threshold-drop 0.0005

Salida:
    artifacts/<release>/oof/<exp_tag>/reports/feature_importance_permutation_<release>.csv
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import average_precision_score, roc_auc_score

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.databases import Database


def _parse_date(s: str) -> datetime:
    return datetime.fromisoformat(s)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True)
    ap.add_argument("--exp-tag", required=True,
                    help="Subdir bajo artifacts/<release>/oof/")
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--target-type", default="multitask",
                    choices=["multitask"],  # solo multitask por ahora
                    help="Implementado solo para multitask.")
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    ap.add_argument("--train-from", type=_parse_date, default=_parse_date("2024-01-01"))
    ap.add_argument("--train-to", type=_parse_date, default=_parse_date("2025-10-30"))
    ap.add_argument("--holdout-from", type=_parse_date, default=_parse_date("2025-11-01"))
    ap.add_argument("--holdout-to", type=_parse_date, default=_parse_date("2026-05-02"))
    ap.add_argument("--n-permutations", type=int, default=1,
                    help="Numero de shuffles por feature (>1 reduce varianza)")
    ap.add_argument("--threshold-drop", type=float, default=0.0005,
                    help="Features con max(drop_long, drop_short) < threshold = candidatas a eliminar")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=4096)
    return ap.parse_args()


def load_holdout_inputs(args: argparse.Namespace):
    """Reproduce la carga de holdout que hace evaluate_holdout para multitask.

    Devuelve:
        X: dict con seq_short, seq_long, context, time
        y_true: np.ndarray (N, 2) con cols [long, short]
        feature_columns: dict {block: [col_name, ...]} con el orden REAL de
                         las columnas en X[block] (post side-mask, post-union).
        artifacts: TrainerArtifacts del modelo multitask
        pipeline: DataPipeline con scalers cargados (para introspeccion)
    """
    from mimo.oof.main_oof_regime_weights_v7 import (
        build_trainer,
        load_existing_artifacts,
        holdout_eval_context,
    )

    print(f"\n📂 Cargando datos {args.train_from.date()} → {args.holdout_to.date()}")
    db = Database()
    resample = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    df_hold = df_rates[df_rates["time"] >= args.holdout_from].copy()
    print(f"   Holdout: {len(df_hold):,} barras")

    base_dir = Path("../../artifacts") / args.release / "oof"
    train_dir = base_dir / args.exp_tag

    print(f"\n🛠️  Construyendo trainer para release={args.release}...")
    trainer = build_trainer(
        args.release,
        args.label_horizon_long, args.label_horizon_short,
        train_dir,
        target_type=args.target_type,
        base_tf=args.base_tf,
    )
    # Cargar best params del estudio Optuna (para reconstruir model_config aunque
    # no lo usemos directamente; pipeline depende de seq_len configurados).
    trainer._load_best_from_storage(side="multitask")

    artifacts = load_existing_artifacts(train_dir, args.release, "multitask")
    pipeline = trainer._build_eval_pipeline(artifacts, "multitask",
                                             inference_policy="transform")

    print("\n🔧 Preparando features y secuencias...")
    with holdout_eval_context():
        df_prep = pipeline.prepare_data(
            df_hold,
            labels=True,
            side="both",
            set_market_condition=False,
            ensure_regime=True,
        )
        # Multitask: usar side='both' para obtener features UNION (no
        # filtradas por mascara side-specific) y labels duales (N, 2).
        # Esto coincide con como prepare_production_model entrena el modelo.
        sequences = pipeline.create_sequences_by_side(
            df_prep, sides=("both",),
            fit_scalers=False, train=True,
        )

    data = sequences["both"]
    X = {
        "seq_short": data["seq_short"],
        "seq_long":  data["seq_long"],
        "context":   data["context"],
        "time":      data["time"],
    }
    y_true = np.asarray(data["labels"]).astype(int)
    if y_true.ndim != 2 or y_true.shape[-1] != 2:
        raise RuntimeError(
            f"Multitask labels esperadas en shape (N, 2). Recibido {y_true.shape}"
        )
    print(f"   X.seq_short: {X['seq_short'].shape}")
    print(f"   X.seq_long : {X['seq_long'].shape}")
    print(f"   X.context  : {X['context'].shape}")
    print(f"   X.time     : {X['time'].shape}")
    print(f"   y_true     : {y_true.shape}  | pos_rate L={y_true[:,0].mean():.4f} "
          f"S={y_true[:,1].mean():.4f}")

    # En multitask las cols por bloque son la UNION (no filtradas por side).
    # data_pipeline_v2.create_sequences_by_side construye los arrays con
    # all_seq_short_cols/all_seq_long_cols/all_context_cols (union sorted).
    union_cols = pipeline._get_all_feature_columns()
    feature_columns = {
        "seq_short": list(union_cols.get("sequence_short", [])),
        "seq_long":  list(union_cols.get("sequence_long",  [])),
        "context":   list(union_cols.get("context",        [])),
    }
    # Sanity: el numero de cols debe coincidir con el ultimo eje de cada bloque.
    expected = {
        "seq_short": X["seq_short"].shape[-1],
        "seq_long":  X["seq_long"].shape[-1],
        "context":   X["context"].shape[-1],
    }
    for blk, cols in feature_columns.items():
        if len(cols) != expected[blk]:
            print(f"  ⚠️  [{blk}] union={len(cols)} cols != X tensor={expected[blk]}. "
                  f"Usando indices 0..{expected[blk]-1} sin nombre.")
            feature_columns[blk] = [f"{blk}_{i}" for i in range(expected[blk])]

    return X, y_true, feature_columns, artifacts, pipeline


def predict_multitask(model, X: dict, calibrator: dict, batch_size: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    inputs = [X["seq_short"], X["seq_long"], X["context"], X["time"]]
    raw = model.predict(inputs, verbose=0, batch_size=batch_size)
    if isinstance(raw, dict):
        p_l = np.asarray(raw["signal_long"]).reshape(-1)
        p_s = np.asarray(raw["signal_short"]).reshape(-1)
    elif isinstance(raw, (list, tuple)) and len(raw) == 2:
        p_l = np.asarray(raw[0]).reshape(-1)
        p_s = np.asarray(raw[1]).reshape(-1)
    else:
        raise RuntimeError(f"Predict shape inesperado: {type(raw)}")

    cal_l = calibrator["long"]
    cal_s = calibrator["short"]
    y_l = cal_l.predict(p_l) if hasattr(cal_l, "predict") else cal_l.transform(p_l)
    y_s = cal_s.predict(p_s) if hasattr(cal_s, "predict") else cal_s.transform(p_s)
    return y_l.astype(np.float32), y_s.astype(np.float32)


def permute_block_feature(X_orig: dict, block: str, idx: int, rng: np.random.Generator) -> dict:
    """Devuelve una copia de X con la columna `idx` del bloque `block` shuffled across batch."""
    X_perm = {k: v.copy() for k, v in X_orig.items()}
    arr = X_perm[block]
    perm = rng.permutation(arr.shape[0])
    if arr.ndim == 3:
        # (B, T, F): preservamos la estructura temporal dentro de cada sample,
        # solo barajamos qué sample tiene qué valor en la columna `idx`.
        arr[:, :, idx] = arr[perm, :, idx]
    elif arr.ndim == 2:
        arr[:, idx] = arr[perm, idx]
    else:
        raise ValueError(f"Bloque {block} con shape inesperado {arr.shape}")
    X_perm[block] = arr
    return X_perm


def main() -> None:
    args = parse_args()
    rng = np.random.default_rng(args.seed)

    X, y_true, feature_columns, artifacts, pipeline = load_holdout_inputs(args)

    print(f"\n🧠 Cargando modelo: {artifacts.model_path}")
    model = tf.keras.models.load_model(artifacts.model_path, safe_mode=False)

    print(f"🎛️  Cargando calibradores: {artifacts.calibrator_path}")
    calibrator = joblib.load(artifacts.calibrator_path)
    if not isinstance(calibrator, dict) or "long" not in calibrator or "short" not in calibrator:
        raise RuntimeError(f"Calibrator multitask inesperado: {type(calibrator)} keys={list(calibrator.keys()) if isinstance(calibrator, dict) else 'N/A'}")

    # ── Baseline ────────────────────────────────────────────────────────────
    print(f"\n📊 Baseline predict ({len(y_true):,} samples)...")
    y_l_base, y_s_base = predict_multitask(model, X, calibrator, batch_size=args.batch_size)

    auc_pr_l_base = float(average_precision_score(y_true[:, 0], y_l_base))
    auc_pr_s_base = float(average_precision_score(y_true[:, 1], y_s_base))
    auc_roc_l_base = float(roc_auc_score(y_true[:, 0], y_l_base))
    auc_roc_s_base = float(roc_auc_score(y_true[:, 1], y_s_base))

    print(f"   AUC-PR  long  : {auc_pr_l_base:.4f}")
    print(f"   AUC-PR  short : {auc_pr_s_base:.4f}")
    print(f"   AUC-ROC long  : {auc_roc_l_base:.4f}")
    print(f"   AUC-ROC short : {auc_roc_s_base:.4f}")

    # ── Permutaciones ───────────────────────────────────────────────────────
    blocks = ["seq_short", "seq_long", "context"]
    total_features = sum(len(feature_columns[b]) for b in blocks)
    total_iters = total_features * args.n_permutations
    print(f"\n🔬 Permutation importance: {total_features} features × {args.n_permutations} reps = {total_iters} forwards")
    print("   (skip 'time' block — son features ciclicas calendario, no features predictivas)\n")

    rows: List[dict] = []
    t0 = time.perf_counter()
    done = 0

    for block in blocks:
        feat_names = feature_columns[block]
        for idx, feat_name in enumerate(feat_names):
            d_long_list, d_short_list = [], []
            for rep in range(args.n_permutations):
                X_perm = permute_block_feature(X, block, idx, rng)
                y_l_p, y_s_p = predict_multitask(model, X_perm, calibrator, batch_size=args.batch_size)
                d_long_list.append(auc_pr_l_base - float(average_precision_score(y_true[:, 0], y_l_p)))
                d_short_list.append(auc_pr_s_base - float(average_precision_score(y_true[:, 1], y_s_p)))
                done += 1

            d_long = float(np.mean(d_long_list))
            d_short = float(np.mean(d_short_list))
            d_long_std = float(np.std(d_long_list)) if args.n_permutations > 1 else 0.0
            d_short_std = float(np.std(d_short_list)) if args.n_permutations > 1 else 0.0
            max_drop = max(d_long, d_short)

            rows.append({
                "block": block, "feature": feat_name,
                "drop_long": d_long, "drop_short": d_short,
                "drop_long_std": d_long_std, "drop_short_std": d_short_std,
                "max_drop": max_drop,
            })

            elapsed = time.perf_counter() - t0
            eta = (elapsed / done) * (total_iters - done) if done > 0 else 0
            flag = "🗑️ " if max_drop < args.threshold_drop else "  "
            print(f"  {flag}[{done:>3}/{total_iters}] {block:11s} {feat_name:30s}  "
                  f"Δ_L={d_long:+.5f}  Δ_S={d_short:+.5f}  max={max_drop:+.5f}  | ETA {eta/60:.1f}m")

    df_imp = pd.DataFrame(rows).sort_values("max_drop", ascending=False).reset_index(drop=True)

    # ── Save CSV ────────────────────────────────────────────────────────────
    out_dir = Path("../../artifacts") / args.release / "oof" / args.exp_tag / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"feature_importance_permutation_{args.release}.csv"
    df_imp.to_csv(csv_path, index=False)
    print(f"\n✅ Guardado: {csv_path}")

    # ── Summary ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 78)
    print("  TOP 25 FEATURES (mayor max_drop = más importantes)")
    print("=" * 78)
    print(df_imp.head(25)[["block", "feature", "drop_long", "drop_short", "max_drop"]].round(5).to_string(index=False))

    print("\n" + "=" * 78)
    print(f"  CANDIDATAS A ELIMINAR (max_drop < {args.threshold_drop:.4f}, posible ruido)")
    print("=" * 78)
    drop_candidates = df_imp[df_imp.max_drop < args.threshold_drop]
    n_drop = len(drop_candidates)
    n_total = len(df_imp)
    print(f"  {n_drop} de {n_total} features ({100*n_drop/n_total:.1f}%):\n")
    for blk in blocks:
        sub = drop_candidates[drop_candidates.block == blk]
        if len(sub):
            print(f"  [{blk}]  ({len(sub)} de {len(df_imp[df_imp.block == blk])})")
            for _, row in sub.iterrows():
                print(f"    - {row.feature:30s}  max_drop={row.max_drop:+.5f}")
            print()

    print("=" * 78)
    print("  PROXIMO PASO")
    print("=" * 78)
    print(f"  Si {n_drop} features se eliminan → entrenar release nueva (e.g. 202300)")
    print(f"  con use_reduced_features=True consumiendo solo el top {n_total - n_drop} features.")
    print(f"  Expected: mismo o mejor AUC-PR con menos overfitting.")


if __name__ == "__main__":
    main()
