"""
gbm_sanity_check.py
═══════════════════════════════════════════════════════════════════════════
Diagnóstico independiente del CNN-LSTM: ¿tienen las features (las mismas
que entrena la red) señal predictiva real para los labels triple-barrier?

Entrena un LightGBM rápido por TimeSeriesSplit sobre el feature set y las
labels de la release indicada (multitask: signal_long y signal_short).
Reporta AUC-ROC, AUC-PR y feature importance ranking.

VEREDICTO de la prueba:
  - GBM AUC-ROC > 0.55  →  las features SÍ tienen señal. Si la red da
                           AUC=0.50, el problema es la red (LR, capacity,
                           focal loss, dropout, etc.). Tunear hyperparams.
  - GBM AUC-ROC ~ 0.50  →  las features NO tienen señal predictiva para
                           ESTAS barriers/horizon. Cambiar barriers (tp/sl,
                           --label-horizon) o feature set. Tunear no resuelve.
  - GBM AUC-ROC 0.51-0.54 → señal débil. Hay algo aprovechable pero está
                           cerca del límite. Posible bug de leakage o regime
                           shift train→holdout.

Uso:
  python3 -m scripts.gbm_sanity_check \\
    --release 202600 --base-tf 5min \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --label-horizon-long 3 --label-horizon-short 3

Smoke test rápido (rango temporal corto):
  python3 -m scripts.gbm_sanity_check --release 202600 \\
    --train-from 2025-06-01 --train-to 2025-09-30 \\
    --holdout-from 2025-10-01 --holdout-to 2025-10-30
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.model_selection import TimeSeriesSplit

try:
    import lightgbm as lgb
except ImportError:
    print("❌ lightgbm no instalado. Instala con: pip install lightgbm", file=sys.stderr)
    sys.exit(1)

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import ModelConfig, Config
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.main_oof_regime_weights_v7 import (
    _get_barriers_for_release,
    _get_feature_masks_for_release,
    _tf_defaults,
    _VOL_INVARIANT_RELEASES,
    _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
)


# ─── helpers ────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _build_pipeline_like_v7(release: str, base_tf: str,
                            label_horizon_long: int, label_horizon_short: int,
                            ) -> Tuple[DataPipeline, Dict]:
    """Reconstruye el pipeline con la MISMA config que main_oof_regime_weights_v7
    para una release multitask. Devuelve (pipeline, tf_defaults)."""
    barriers = _get_barriers_for_release(release)
    tf_defaults = _tf_defaults(base_tf)
    use_vol_invariant = str(release) in _VOL_INVARIANT_RELEASES
    use_reduced = str(release) in _REDUCED_FEATURES_RELEASES
    use_ultra = str(release) in _ULTRA_REDUCED_FEATURES_RELEASES

    general = Config(release=release, use_oof=True, oof_splits=5, oof_epochs=1)

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier",
        label_horizon=max(label_horizon_long, label_horizon_short),
        tp_barrier=barriers["tp_base"],
        sl_barrier=barriers["sl_base"],
        label_method_long="triple_barrier",
        regime_barriers_long=barriers["regime_barriers_long"],
        label_method_short="triple_barrier",
        regime_barriers_short=barriers["regime_barriers_short"],
        tp_barrier_short=None,
        sl_barrier_short=None,
        feature_masks=_get_feature_masks_for_release(release),
        price_norm_window=tf_defaults["price_norm_window"],
        use_vol_invariant_features=use_vol_invariant,
        use_reduced_features=use_reduced,
        use_ultra_reduced_features=use_ultra,
    )

    regime_config = StateConfig(adx_trend_threshold=25.0)

    # ModelConfig SOLO para que el pipeline tenga seq_len_* configurados.
    # No entrenamos red, así que el resto da igual.
    model_config = ModelConfig(
        seq_len_short=tf_defaults["seq_len_short"],
        seq_len_long=tf_defaults["seq_len_long"],
        target_type="multitask",
    )

    pipeline = DataPipeline(
        general_config=general,
        feature_config=feature_config,
        model_config=model_config,
        regime_config=regime_config,
    )

    print(f"🪟 [TF DEFAULTS] base_tf={base_tf} | "
          f"seq_len_short={tf_defaults['seq_len_short']} | "
          f"seq_len_long={tf_defaults['seq_len_long']}")
    if use_vol_invariant:
        print(f"🛡️  [VOL-INVARIANT] release={release}")
    if use_reduced:
        print(f"✂️  [REDUCED FEATURES] release={release}")
    if use_ultra:
        print(f"✂️✂️ [ULTRA-REDUCED FEATURES] release={release}")

    return pipeline, tf_defaults


def _collect_tabular_features(pipeline: DataPipeline) -> List[str]:
    """Devuelve la lista plana de columnas que el modelo neuronal usa,
    para alimentarlas como tabulares al GBM (snapshot por timestamp)."""
    all_cols = pipeline._get_all_feature_columns()
    # all_cols es dict con keys: sequence_short, sequence_long, context, time
    flat = []
    for key in ("sequence_short", "sequence_long", "context", "time"):
        cols = all_cols.get(key, [])
        for c in cols:
            if c not in flat:
                flat.append(c)
    return flat


def _train_and_eval_gbm(
    df: pd.DataFrame,
    feature_cols: List[str],
    label_col: str,
    n_splits: int,
    seed: int,
    n_estimators: int,
    side_name: str,
    show_top_features: int,
) -> Dict[str, float]:
    """Entrena LightGBM con TimeSeriesSplit y reporta métricas agregadas."""
    print(f"\n{'═' * 70}")
    print(f"  GBM SANITY CHECK — side={side_name}  label={label_col}")
    print(f"{'═' * 70}")

    X = df[feature_cols].astype(np.float32).values
    y = df[label_col].astype(np.int8).values
    base = y.mean()
    print(f"  n_samples = {len(df):,}")
    print(f"  n_features = {X.shape[1]:,}")
    print(f"  base_rate = {base:.4f}")
    print(f"  n_splits  = {n_splits}")

    tscv = TimeSeriesSplit(n_splits=n_splits)
    rocs, prs, briers = [], [], []
    fi_acc = np.zeros(X.shape[1], dtype=np.float64)

    t0 = time.time()
    for fold, (tr_idx, va_idx) in enumerate(tscv.split(X), start=1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        # Skip si val no tiene positivos (no se puede AUC)
        if y_va.sum() == 0 or y_va.sum() == len(y_va):
            print(f"  Fold {fold}/{n_splits}: val sin ambas clases (y_va.sum={y_va.sum()}), skip")
            continue

        # Class weight: scale_pos_weight = N_neg / N_pos (balanceo razonable)
        n_pos = max(int(y_tr.sum()), 1)
        n_neg = max(len(y_tr) - n_pos, 1)
        spw = n_neg / n_pos

        params = dict(
            objective="binary",
            metric="average_precision",
            learning_rate=0.05,
            num_leaves=31,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=5,
            min_data_in_leaf=200,
            scale_pos_weight=spw,
            seed=seed,
            verbose=-1,
            n_jobs=-1,
        )
        train_ds = lgb.Dataset(X_tr, label=y_tr)
        valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)

        booster = lgb.train(
            params,
            train_ds,
            num_boost_round=n_estimators,
            valid_sets=[valid_ds],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )

        p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
        roc = roc_auc_score(y_va, p_va)
        pr = average_precision_score(y_va, p_va)
        br = brier_score_loss(y_va, p_va)
        rocs.append(roc)
        prs.append(pr)
        briers.append(br)
        fi_acc += np.asarray(booster.feature_importance(importance_type="gain"), dtype=np.float64)

        print(f"  Fold {fold}/{n_splits}: AUC-ROC={roc:.4f}  AUC-PR={pr:.4f}  "
              f"Brier={br:.4f}  iters={booster.best_iteration}  "
              f"n_tr={len(tr_idx):,} n_va={len(va_idx):,}")

    elapsed = time.time() - t0
    if not rocs:
        print(f"  ⚠️  Ningún fold válido. Abortando side={side_name}.")
        return {}

    roc_mean = float(np.mean(rocs))
    pr_mean = float(np.mean(prs))
    pr_lift = pr_mean / base if base > 0 else float("nan")
    print(f"\n  ── AGREGADO {side_name} ──")
    print(f"  AUC-ROC mean = {roc_mean:.4f}   (chance = 0.50)")
    print(f"  AUC-PR  mean = {pr_mean:.4f}   (chance = {base:.4f})   lift={pr_lift:.2f}x")
    print(f"  Brier   mean = {float(np.mean(briers)):.4f}")
    print(f"  tiempo total = {elapsed:.1f}s")

    # Veredicto rápido
    if roc_mean < 0.51:
        verdict = "❌ Features SIN señal predictiva — el problema NO es la red"
    elif roc_mean < 0.54:
        verdict = "⚠️  Señal MUY DÉBIL — posible regime shift o leakage"
    elif roc_mean < 0.58:
        verdict = "🟡 Señal moderada — la red debería poder explotarlo"
    else:
        verdict = "✅ Señal fuerte — si la red da 0.50, problema de optimización"
    print(f"  VEREDICTO: {verdict}")

    # Top features importance (gain agregado entre folds)
    if show_top_features > 0:
        feature_cols_arr = np.array(_feature_cols_global)
        top_idx = np.argsort(-fi_acc)[:show_top_features]
        print(f"\n  TOP {show_top_features} FEATURES (gain acumulado entre folds):")
        max_imp = fi_acc[top_idx[0]] if fi_acc[top_idx[0]] > 0 else 1.0
        for rank, idx in enumerate(top_idx, start=1):
            imp = fi_acc[idx]
            bar = "█" * int(40 * imp / max_imp)
            print(f"    {rank:>2d}. {feature_cols_arr[idx]:<50s} {imp:>10.1f} {bar}")

    return {
        "side": side_name,
        "auc_roc_mean": roc_mean,
        "auc_pr_mean": pr_mean,
        "auc_pr_lift": pr_lift,
        "n_folds": len(rocs),
        "base_rate": float(base),
    }


# Global to simplify passing to _train_and_eval_gbm
_feature_cols_global: List[str] = []


# ─── main ───────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", default="202600")
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--train-from", type=_parse_date,
                    default=_parse_date("2024-01-01"))
    ap.add_argument("--train-to", type=_parse_date,
                    default=_parse_date("2025-10-30"))
    ap.add_argument("--holdout-from", type=_parse_date,
                    default=_parse_date("2025-11-01"))
    ap.add_argument("--holdout-to", type=_parse_date,
                    default=_parse_date("2026-04-10"))
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    ap.add_argument("--n-splits", type=int, default=5)
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--top-features", type=int, default=20,
                    help="Cuántas top features mostrar por importancia (gain)")
    ap.add_argument("--side", choices=("long", "short", "both"), default="both",
                    help="Lado a evaluar. 'both' corre LONG y SHORT consecutivos.")
    args = ap.parse_args()

    print(f"\n{'═' * 70}")
    print(f"  GBM SANITY CHECK — release={args.release} base_tf={args.base_tf}")
    print(f"  Train: {args.train_from.date()} → {args.train_to.date()}")
    print(f"  Holdout: {args.holdout_from.date()} → {args.holdout_to.date()}")
    print(f"  Horizon long/short: {args.label_horizon_long}/{args.label_horizon_short} bars")
    print(f"{'═' * 70}")

    # 1) Load raw OHLCV
    print(f"\n📂 Cargando OHLCV desde DB...")
    db = Database()
    resample = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras cargadas")

    # 2) Build pipeline matching release config
    pipeline, tf_defaults = _build_pipeline_like_v7(
        args.release, args.base_tf,
        args.label_horizon_long, args.label_horizon_short,
    )

    # 3) Prepare data (features + labels multitask)
    print(f"\n🔧 Construyendo features y labels multitask...")
    df_prepared = pipeline.prepare_data(
        df_rates,
        labels=True,
        side="both",  # multitask
        set_market_condition=False,
        ensure_regime=True,
    )
    print(f"   df_prepared shape: {df_prepared.shape}")
    print(f"   columnas relevantes: {[c for c in df_prepared.columns if c.startswith('signal') or c == 'state']}")

    # 4) Determine label columns (multitask schema)
    label_long_col = None
    label_short_col = None
    for cand in ("signal_long", "y_long", "label_long"):
        if cand in df_prepared.columns:
            label_long_col = cand
            break
    for cand in ("signal_short", "y_short", "label_short"):
        if cand in df_prepared.columns:
            label_short_col = cand
            break

    if not (label_long_col and label_short_col):
        # Fallback binary mode: 'signal' column
        if "signal" in df_prepared.columns:
            print("⚠️  No encontré signal_long/signal_short, usando 'signal' como label genérico.")
            label_long_col = label_short_col = "signal"
        else:
            print(f"❌ No encuentro columnas de label. Columnas disponibles: {list(df_prepared.columns)[:30]}")
            sys.exit(1)

    # 5) Feature columns (snapshot por timestamp)
    global _feature_cols_global
    feature_cols = _collect_tabular_features(pipeline)
    feature_cols = [c for c in feature_cols if c in df_prepared.columns]
    _feature_cols_global = feature_cols
    print(f"   {len(feature_cols)} features tabulares para GBM")

    # 6) Drop rows with any NaN in features or label
    needed = feature_cols + [label_long_col, label_short_col]
    before = len(df_prepared)
    df_clean = df_prepared.dropna(subset=needed).reset_index(drop=True)
    print(f"   Tras dropna: {len(df_clean):,} filas ({before - len(df_clean):,} eliminadas)")

    # 7) Split train vs holdout por fecha
    df_train = df_clean[
        (df_clean["time"] >= args.train_from) & (df_clean["time"] < args.holdout_from)
    ].reset_index(drop=True)
    df_holdout = df_clean[
        (df_clean["time"] >= args.holdout_from) & (df_clean["time"] <= args.holdout_to)
    ].reset_index(drop=True)
    print(f"   Train: {len(df_train):,}  |  Holdout: {len(df_holdout):,}")

    if len(df_train) < 5000:
        print(f"⚠️  Train muy pequeño ({len(df_train):,}), TSCV puede dar resultados ruidosos.")

    # 8) Evaluar por TSCV sobre TRAIN (sin tocar holdout aún)
    results = []
    sides_to_run = (label_long_col, "long") if args.side == "long" else \
                   (label_short_col, "short") if args.side == "short" else \
                   None

    if args.side == "both":
        for (lc, sn) in [(label_long_col, "long"), (label_short_col, "short")]:
            r = _train_and_eval_gbm(
                df_train, feature_cols, lc,
                n_splits=args.n_splits, seed=args.seed,
                n_estimators=args.n_estimators, side_name=sn,
                show_top_features=args.top_features,
            )
            if r:
                results.append(r)
    else:
        lc, sn = sides_to_run
        r = _train_and_eval_gbm(
            df_train, feature_cols, lc,
            n_splits=args.n_splits, seed=args.seed,
            n_estimators=args.n_estimators, side_name=sn,
            show_top_features=args.top_features,
        )
        if r:
            results.append(r)

    # 9) Holdout check rápido: entrenar en TODO train y evaluar en holdout
    if len(df_holdout) > 1000:
        print(f"\n{'═' * 70}")
        print(f"  HOLDOUT EVAL (entrena sobre todo train, predice holdout)")
        print(f"{'═' * 70}")
        for r in results:
            sn = r["side"]
            lc = label_long_col if sn == "long" else label_short_col

            X_tr = df_train[feature_cols].astype(np.float32).values
            y_tr = df_train[lc].astype(np.int8).values
            X_ho = df_holdout[feature_cols].astype(np.float32).values
            y_ho = df_holdout[lc].astype(np.int8).values

            n_pos = max(int(y_tr.sum()), 1)
            n_neg = max(len(y_tr) - n_pos, 1)
            spw = n_neg / n_pos

            params = dict(
                objective="binary", metric="average_precision",
                learning_rate=0.05, num_leaves=31,
                feature_fraction=0.9, bagging_fraction=0.9, bagging_freq=5,
                min_data_in_leaf=200, scale_pos_weight=spw,
                seed=args.seed, verbose=-1, n_jobs=-1,
            )
            train_ds = lgb.Dataset(X_tr, label=y_tr)
            booster = lgb.train(params, train_ds, num_boost_round=args.n_estimators)
            p_ho = booster.predict(X_ho)
            base_ho = y_ho.mean()
            print(f"  {sn.upper():<6s} | base={base_ho:.4f}  "
                  f"AUC-ROC={roc_auc_score(y_ho, p_ho):.4f}  "
                  f"AUC-PR={average_precision_score(y_ho, p_ho):.4f}  "
                  f"lift={average_precision_score(y_ho, p_ho)/base_ho:.2f}x  "
                  f"Brier={brier_score_loss(y_ho, p_ho):.4f}")

    # 10) Resumen final
    print(f"\n{'═' * 70}")
    print(f"  RESUMEN — GBM Sanity Check")
    print(f"{'═' * 70}")
    for r in results:
        print(f"  {r['side'].upper():<6s} | TSCV AUC-ROC={r['auc_roc_mean']:.4f} "
              f"| AUC-PR={r['auc_pr_mean']:.4f} (base={r['base_rate']:.4f}, "
              f"lift={r['auc_pr_lift']:.2f}x) | n_folds={r['n_folds']}")

    # Veredicto global
    if results:
        best_roc = max(r["auc_roc_mean"] for r in results)
        print(f"\n  Best AUC-ROC = {best_roc:.4f}")
        if best_roc < 0.51:
            print(f"  📛 CONCLUSIÓN: las features NO contienen señal predictiva para")
            print(f"     barriers={args.label_horizon_long}/{args.label_horizon_short}, "
                  f"tp/sl del release. NO sirve tunear hyperparams del CNN.")
            print(f"     Acción recomendada: ampliar label-horizon (a 12 o 24) o reducir")
            print(f"     tp_base (de 2.0 a 1.5). El problema es la definición del problema.")
        elif best_roc < 0.54:
            print(f"  ⚠️  CONCLUSIÓN: señal MUY DÉBIL. Hay algo pero está cerca del ruido.")
            print(f"     Sospecha bug de leakage o regime shift train→holdout.")
            print(f"     Revisa también que el CNN no esté en colapso por LR/batches/focal.")
        else:
            print(f"  ✅ CONCLUSIÓN: las features SÍ tienen señal. Si el CNN da AUC=0.50")
            print(f"     mientras GBM da {best_roc:.3f}, el problema es la red:")
            print(f"     LR insuficiente, batch_size grande, focal_gamma alto, dropouts.")
            print(f"     El nuevo grid 202601 ya ataca estos puntos.")
    print()


if __name__ == "__main__":
    main()
