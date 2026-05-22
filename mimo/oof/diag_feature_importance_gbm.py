"""
diag_feature_importance_gbm.py
═══════════════════════════════════════════════════════════════════════════
Feature importance del best trial (refit completo en full train).

OUTPUTS:
  · GAIN (importance_type='gain')  — magnitud media del split por feature
  · SPLIT (importance_type='split') — número de splits que usaron la feature
  · SHAP top-20 (si shap está instalado) — impacto medio absoluto

USO:
  python -m mimo.oof.diag_feature_importance_gbm \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --shap-sample 5000 \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/feature_importance.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import optuna
import pandas as pd

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.main_oof_gbm_holdout import (
    _split_trial_params, _collect_tabular_columns,
    _refit_with_internal_val,
)
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE, LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release, _get_feature_masks_for_release, _tf_defaults,
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)

# Side-effect: inyectar _GBM_BARRIERS_BY_RELEASE.
from mimo.oof.main_oof_gbm import _GBM_EXCLUDE_PATTERNS  # noqa: F401

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _importance_table(booster, feat_names: List[str], top_n: int = 25) -> List[Dict[str, Any]]:
    gain  = booster.feature_importance(importance_type="gain")
    split = booster.feature_importance(importance_type="split")
    total_gain = max(gain.sum(), 1.0)
    rows = []
    for i, name in enumerate(feat_names):
        rows.append({
            "feature": name,
            "gain":         float(gain[i]),
            "gain_pct":     float(100.0 * gain[i] / total_gain),
            "split_count":  int(split[i]),
        })
    rows.sort(key=lambda r: r["gain"], reverse=True)
    return rows[:top_n]


def _shap_top(booster, X_sample: np.ndarray, feat_names: List[str], top_n: int = 20) -> List[Dict[str, Any]]:
    """SHAP mean |value| por feature."""
    if not HAS_SHAP: return []
    expl = shap.TreeExplainer(booster)
    sv = expl.shap_values(X_sample)
    # En LGBM binary, shap_values devuelve un array (n_samples, n_features)
    if isinstance(sv, list): sv = sv[1] if len(sv) > 1 else sv[0]
    mean_abs = np.abs(sv).mean(axis=0)
    rows = [{"feature": feat_names[i], "shap_mean_abs": float(mean_abs[i])}
            for i in range(len(feat_names))]
    rows.sort(key=lambda r: r["shap_mean_abs"], reverse=True)
    return rows[:top_n]


def _print_table(title: str, rows: List[Dict[str, Any]], col_format) -> None:
    print(f"\n══ {title} ══════════════════════════════════════════════════════")
    print(col_format("header"))
    for r in rows:
        print(col_format(r))


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--inherit-config-from", default=None)
    ap.add_argument("--best-json", required=True)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()), default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()), default="moderate")
    ap.add_argument("--regime-weights-long-json", default=None)
    ap.add_argument("--regime-weights-short-json", default=None)
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    ap.add_argument("--train-from", type=_parse_date, required=True)
    ap.add_argument("--train-to", type=_parse_date, required=True)
    ap.add_argument("--shap-sample", type=int, default=5000,
                    help="Tamaño del subsample para SHAP (caro O(N*T)).")
    ap.add_argument("--top-n", type=int, default=25)
    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-json", default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)

    if args.inherit_config_from:
        src, dst = str(args.inherit_config_from), release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:  _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    print(f"📚 SHAP disponible: {HAS_SHAP}")

    with open(args.best_json) as fh: best = json.load(fh)
    top_long, top_short = best["top_long"][0], best["top_short"][0]
    n_long, n_short = int(top_long["trial"]), int(top_short["trial"])

    study_name = f"{args.study_prefix}_{release}_multitask"
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    tr_long_obj  = next(t for t in study.trials if t.number == n_long)
    tr_short_obj = next(t for t in study.trials if t.number == n_short)
    lgbm_long,  meta_long  = _split_trial_params(tr_long_obj.params,  seed=int(args.seed))
    lgbm_short, meta_short = _split_trial_params(tr_short_obj.params, seed=int(args.seed))

    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    print(f"📊 Cargando OHLCV {args.train_from.date()} → {args.train_to.date()}")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.train_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])

    barriers = _get_barriers_for_release(release)
    tf_defaults = _tf_defaults(base_tf)
    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier_dual",
        label_horizon=max(args.label_horizon_long, args.label_horizon_short),
        tp_barrier=barriers["tp_base"], sl_barrier=barriers["sl_base"],
        label_method_long="triple_barrier_dual",
        regime_barriers_long=barriers["regime_barriers_long"],
        label_method_short="triple_barrier_dual",
        regime_barriers_short=barriers["regime_barriers_short"],
        tp_barrier_short=None, sl_barrier_short=None,
        feature_masks=_get_feature_masks_for_release(release),
        price_norm_window=tf_defaults["price_norm_window"],
        use_vol_invariant_features=(release in _VOL_INVARIANT_RELEASES),
        use_reduced_features=(release in _REDUCED_FEATURES_RELEASES),
        use_ultra_reduced_features=(release in _ULTRA_REDUCED_FEATURES_RELEASES),
    )
    general_config = Config(release=release, use_oof=True, oof_splits=5)
    regime_config = StateConfig(adx_trend_threshold=25.0)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=ModelConfig(seq_len_short=64, seq_len_long=256, target_type="multitask"),
        regime_config=regime_config,
    )
    df_prepared = pipeline.prepare_data(df_rates, labels=True, side="both",
                                        set_market_condition=False, ensure_regime=True)
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]
    _exclude = _GBM_EXCLUDE_PATTERNS.get(release)
    if _exclude:
        feat_cols = [c for c in feat_cols if not any(p in c for p in _exclude)]
        print(f"🚫 [EXCLUDE] release={release} → {len(feat_cols)} features")
    needed = feat_cols + ["signal_long", "signal_short"]
    df_clean = df_prepared.loc[df_prepared[needed].notna().all(axis=1)].reset_index(drop=True)

    X = df_clean[feat_cols].astype(np.float32).values
    y_long  = df_clean["signal_long"].astype(np.int8).values
    y_short = df_clean["signal_short"].astype(np.int8).values
    w = df_clean["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_clean.columns else np.ones(len(df_clean), dtype=np.float32)

    # Refit ambos
    print(f"\n🌳 Refit LONG  (trial #{n_long})...")
    booster_long, _ = _refit_with_internal_val(
        X=X, y=y_long, w=w, lgbm_params=lgbm_long,
        n_estimators=int(meta_long.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_long.get("early_stopping_rounds", 50)),
    )
    print(f"🌳 Refit SHORT (trial #{n_short})...")
    booster_short, _ = _refit_with_internal_val(
        X=X, y=y_short, w=w, lgbm_params=lgbm_short,
        n_estimators=int(meta_short.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_short.get("early_stopping_rounds", 50)),
    )

    # GAIN/SPLIT tables
    imp_long  = _importance_table(booster_long,  feat_cols, top_n=args.top_n)
    imp_short = _importance_table(booster_short, feat_cols, top_n=args.top_n)

    def _row_fmt(row):
        if row == "header":
            return f"  {'rank':>4} {'feature':<30} {'gain':>12} {'gain%':>7} {'splits':>7}"
        return (f"  {imp_long.index(row)+1 if row in imp_long else imp_short.index(row)+1:>4} "
                f"{row['feature']:<30} {row['gain']:>12.1f} {row['gain_pct']:>6.2f}% {row['split_count']:>7}")

    print(f"\n══ TOP-{args.top_n} LONG  por GAIN ══════════════════════════════════════")
    print(f"  {'rank':>4} {'feature':<30} {'gain':>12} {'gain%':>7} {'splits':>7}")
    for i, r in enumerate(imp_long, 1):
        print(f"  {i:>4} {r['feature']:<30} {r['gain']:>12.1f} {r['gain_pct']:>6.2f}% {r['split_count']:>7}")

    print(f"\n══ TOP-{args.top_n} SHORT por GAIN ══════════════════════════════════════")
    print(f"  {'rank':>4} {'feature':<30} {'gain':>12} {'gain%':>7} {'splits':>7}")
    for i, r in enumerate(imp_short, 1):
        print(f"  {i:>4} {r['feature']:<30} {r['gain']:>12.1f} {r['gain_pct']:>6.2f}% {r['split_count']:>7}")

    # SHAP (opcional)
    shap_long, shap_short = [], []
    if HAS_SHAP and args.shap_sample > 0:
        n_samp = min(int(args.shap_sample), len(X))
        rng = np.random.default_rng(int(args.seed))
        idx = rng.choice(len(X), size=n_samp, replace=False)
        X_samp = X[idx]
        print(f"\n🔍 SHAP sobre {n_samp:,} muestras (LONG)...")
        shap_long  = _shap_top(booster_long,  X_samp, feat_cols, top_n=20)
        print(f"🔍 SHAP sobre {n_samp:,} muestras (SHORT)...")
        shap_short = _shap_top(booster_short, X_samp, feat_cols, top_n=20)

        for side, sl in (("LONG", shap_long), ("SHORT", shap_short)):
            print(f"\n══ TOP-20 {side} por SHAP mean|value| ══════════════════════════")
            print(f"  {'rank':>4} {'feature':<30} {'shap':>14}")
            for i, r in enumerate(sl, 1):
                print(f"  {i:>4} {r['feature']:<30} {r['shap_mean_abs']:>14.6f}")
    elif not HAS_SHAP:
        print("\n⚠️  shap no instalado — saltando SHAP. Para instalarlo: pip install shap")

    report = {
        "release": release, "long_trial": n_long, "short_trial": n_short,
        "n_features": len(feat_cols),
        "feature_columns": feat_cols,
        "importance_long":  imp_long,
        "importance_short": imp_short,
        "shap_long":  shap_long,
        "shap_short": shap_short,
        "has_shap": HAS_SHAP,
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "feature_importance.json")
    with open(out_json, "w") as fh: json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
