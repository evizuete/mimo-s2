"""
main_oof_gbm_deploy.py
═══════════════════════════════════════════════════════════════════════════
FASE 5 GBM: deploy LONG-only para producción.

Toma el best trial del release configurado y produce:
  · production_booster_long.joblib       — booster refit sobre N meses recientes
  · production_threshold.json            — thr derivado de scan reciente
  · production_metadata.json             — manifest (release, trial, fechas, params)
  · production_features.json             — feat_cols esperados por el booster

ESTRATEGIA:
  1. Cargar best LONG trial del Optuna study (params, sig_rate target)
  2. Refit booster sobre los últimos --refit-months meses (default 12)
     usando los lgbm_params del trial. Sin val interna en refit final
     (usa best_iteration del trial original guardado en user_attrs).
  3. Predecir sobre los últimos --thr-scan-months meses con el refit
     (sub-rango del train usado).
  4. Escanear thresholds [0.05, 0.95] sobre esas preds:
     - Calcular EV-net para cada thr
     - Elegir thr que maximiza EV-net con n_signals >= floor
  5. Persistir todo en production/<release>/

EJECUCIÓN: típicamente se relanza una vez al mes para refrescar el modelo.

USO:
  python -m mimo.oof.main_oof_gbm_deploy \\
    --release 202603_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202603_GBM/oof/<tag>/reports/best_per_side.json \\
    --refit-months 12 --thr-scan-months 3 \\
    --as-of 2026-04-01 \\
    --out-dir production/202603_GBM/
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.ev_objective import compute_ev_at_best_threshold
from mimo.oof.main_oof_gbm_holdout import (
    _split_trial_params, _collect_tabular_columns,
)
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE, LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release, _get_feature_masks_for_release, _tf_defaults,
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)
from mimo.oof.main_oof_gbm import _GBM_EXCLUDE_PATTERNS


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _refit_long_booster(
    *, X: np.ndarray, y: np.ndarray, w: np.ndarray,
    lgbm_params: Dict[str, Any], n_estimators: int,
) -> lgb.Booster:
    """Refit final sin val set — usa num_boost_round=n_estimators directo.
    Si quisiéramos best_iter, deberíamos pasar val_set + early_stop, pero
    eso tira datos del refit final. Aquí confiamos en n_estimators del trial."""
    n_pos = int(y.sum()); n_neg = len(y) - n_pos
    params = dict(lgbm_params)
    params["scale_pos_weight"] = float(n_neg / max(n_pos, 1))
    train_ds = lgb.Dataset(X, label=y, weight=w)
    return lgb.train(params, train_ds, num_boost_round=int(n_estimators))


def main() -> None:
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

    ap.add_argument("--as-of", type=_parse_date, required=True,
                    help="Fecha de corte del refit (ej. último día disponible). "
                         "El refit usa datos hasta esta fecha.")
    ap.add_argument("--refit-months", type=int, default=12,
                    help="Cuántos meses retroactivos del refit final.")
    ap.add_argument("--thr-scan-months", type=int, default=3,
                    help="Sub-rango (últimos N meses del refit) para escanear "
                         "threshold. Más reciente = más alineado con producción.")
    ap.add_argument("--thr-lo", type=float, default=0.05)
    ap.add_argument("--thr-hi", type=float, default=0.95)
    ap.add_argument("--thr-n", type=int, default=180)
    ap.add_argument("--thr-min-signals", type=int, default=20,
                    help="Mínimo signals en el thr-scan para considerar el thr.")
    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)

    ap.add_argument("--use-calibrator", action="store_true", default=False,
                    help="Si se pasa, usa calibrator isotónico (no recomendado: "
                         "vimos saturación en 202602/3). Default: raw probs.")

    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")
    ap.add_argument("--seed", type=int, default=47)

    ap.add_argument("--out-dir", required=True,
                    help="Directorio donde persistir el booster + thresholds + manifest.")
    args = ap.parse_args()

    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── 1. Inherit config ─────────────────────────────────────────
    if args.inherit_config_from:
        src, dst = str(args.inherit_config_from), release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:    _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    # ─── 2. Cargar best LONG trial ────────────────────────────────
    with open(args.best_json) as fh:
        best = json.load(fh)
    top_long = best["top_long"][0]
    n_long = int(top_long["trial"])
    sig_rate_long_oof = float(top_long["ev_long"].get("sig_rate") or 0.0)
    print(f"📂 Best LONG: trial #{n_long} | sig_rate OOF target = {sig_rate_long_oof:.5f}")

    study_name = f"{args.study_prefix}_{release}_multitask"
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    tr_long = next(t for t in study.trials if t.number == n_long)
    lgbm_long, meta_long = _split_trial_params(tr_long.params, seed=int(args.seed))
    cal_long_path = tr_long.user_attrs.get("cal_long_path")

    if args.use_calibrator and cal_long_path and os.path.exists(cal_long_path):
        cal_long = joblib.load(cal_long_path)
        print(f"   cal_long: {cal_long_path}")
    else:
        cal_long = None
        if args.use_calibrator:
            print(f"⚠️  --use-calibrator pero no se encuentra el calibrator path. Usando raw probs.")
        else:
            print(f"   ℹ️  modo raw probs (sin calibrator)")

    # ─── 3. Cargar OHLCV ──────────────────────────────────────────
    refit_from = args.as_of - relativedelta(months=int(args.refit_months))
    # Padding extra para features ventana al inicio
    load_from = refit_from - relativedelta(months=2)
    print(f"\n📊 Cargando OHLCV {load_from.date()} → {args.as_of.date()}")
    print(f"   Refit window: {refit_from.date()} → {args.as_of.date()} ({args.refit_months}m)")

    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=load_from, to_date=args.as_of, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])

    # ─── 4. Pipeline + prepare_data ──────────────────────────────
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
    print("⚙️  prepare_data...")
    df_prepared = pipeline.prepare_data(df_rates, labels=True, side="both",
                                        set_market_condition=False, ensure_regime=True)
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]

    # Exclude por release
    exclude_patterns = _GBM_EXCLUDE_PATTERNS.get(release)
    if exclude_patterns:
        feat_cols = [c for c in feat_cols if not any(p in c for p in exclude_patterns)]
        print(f"🚫 [EXCLUDE] release={release} | excluyendo patterns → {len(feat_cols)} features")

    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long"]
    df_clean = df_prepared.loc[df_prepared[needed].notna().all(axis=1)].reset_index(drop=True)
    print(f"   {len(df_clean):,} filas × {len(feat_cols)} features")

    # ─── 5. Mask refit period ────────────────────────────────────
    refit_mask = (df_clean["time"] >= pd.Timestamp(refit_from)) & \
                 (df_clean["time"] < pd.Timestamp(args.as_of))
    df_refit = df_clean.loc[refit_mask].reset_index(drop=True)
    if len(df_refit) < 10000:
        raise SystemExit(f"❌ Refit window solo {len(df_refit)} filas — necesitas más historia")
    print(f"   refit: {len(df_refit):,} filas ({df_refit['time'].min()} → {df_refit['time'].max()})")

    X_ref = df_refit[feat_cols].astype(np.float32).values
    y_long_ref = df_refit["signal_long"].astype(np.int8).values
    w_ref = df_refit["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_refit.columns else np.ones(len(df_refit), dtype=np.float32)

    # ─── 6. Refit booster ────────────────────────────────────────
    n_estimators_final = int(meta_long.get("n_estimators", 200))
    print(f"\n🌳 Refit LONG booster sobre {len(df_refit):,} filas con "
          f"n_estimators={n_estimators_final}...")
    booster_long = _refit_long_booster(
        X=X_ref, y=y_long_ref, w=w_ref,
        lgbm_params=lgbm_long, n_estimators=n_estimators_final,
    )

    # ─── 7. Threshold scan sobre últimos thr-scan-months ─────────
    scan_from = args.as_of - relativedelta(months=int(args.thr_scan_months))
    scan_mask = (df_clean["time"] >= pd.Timestamp(scan_from)) & \
                (df_clean["time"] < pd.Timestamp(args.as_of))
    df_scan = df_clean.loc[scan_mask].reset_index(drop=True)
    print(f"\n🎯 Threshold scan sobre {len(df_scan):,} filas "
          f"({scan_from.date()} → {args.as_of.date()}, {args.thr_scan_months}m)")

    X_sc = df_scan[feat_cols].astype(np.float32).values
    p_raw = booster_long.predict(X_sc).astype(np.float32)
    if cal_long is not None:
        p_use = cal_long.predict(p_raw.astype(np.float64)).astype(np.float32)
    else:
        p_use = p_raw

    df_oof_style = pd.DataFrame({
        "time":  df_scan["time"].values,
        "high":  df_scan["high"].astype(np.float64).values,
        "low":   df_scan["low"].astype(np.float64).values,
        "close": df_scan["close"].astype(np.float64).values,
        "atr":   df_scan["atr"].astype(np.float64).values,
        "signal_long": df_scan["signal_long"].astype(np.int8).values,
        "oof_proba_long_cal": p_use,
    })

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    scan_res = compute_ev_at_best_threshold(
        df_oof_style, proba_col="oof_proba_long_cal", side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=args.cost_per_signal,
        n_thr=int(args.thr_n), thr_lo=float(args.thr_lo), thr_hi=float(args.thr_hi),
        min_signals=int(args.thr_min_signals), max_drawdown_R=args.max_drawdown_R,
    )
    thr_prod = float(scan_res.get("thr", float("nan")))
    if not np.isfinite(thr_prod):
        raise SystemExit("❌ Threshold scan no encontró un thr válido. "
                         "Reduce --thr-min-signals o amplia --thr-lo/hi.")
    print(f"   best thr={thr_prod:.4f}  "
          f"ev_net={scan_res.get('ev_net'):+.4f}R  "
          f"sig={int(scan_res.get('n_signals', 0))}  "
          f"prec={scan_res.get('prec_TP'):.3f}")

    # ─── 8. Persistir ────────────────────────────────────────────
    booster_path = out_dir / "production_booster_long.joblib"
    cal_path     = out_dir / "production_calibrator_long.joblib"
    features_path = out_dir / "production_features.json"
    threshold_path = out_dir / "production_threshold.json"
    manifest_path = out_dir / "production_metadata.json"

    joblib.dump(booster_long, booster_path)
    if cal_long is not None:
        joblib.dump(cal_long, cal_path)
    with open(features_path, "w") as fh:
        json.dump({"feat_cols": feat_cols, "n_features": len(feat_cols),
                   "exclude_patterns": exclude_patterns}, fh, indent=2)
    with open(threshold_path, "w") as fh:
        json.dump({
            "threshold": thr_prod,
            "scan_window": [str(scan_from.date()), str(args.as_of.date())],
            "scan_months": int(args.thr_scan_months),
            "scan_stats": {
                "ev_net":   scan_res.get("ev_net"),
                "n_signals": scan_res.get("n_signals"),
                "prec_TP":   scan_res.get("prec_TP"),
                "mdd_R":     scan_res.get("mdd_R"),
            },
            "use_calibrator": cal_long is not None,
        }, fh, indent=2, default=str)
    with open(manifest_path, "w") as fh:
        json.dump({
            "release": release,
            "trial": n_long,
            "trial_params": tr_long.params,
            "best_json": str(args.best_json),
            "as_of": str(args.as_of.date()),
            "refit_window": [str(refit_from.date()), str(args.as_of.date())],
            "refit_months": int(args.refit_months),
            "thr_scan_months": int(args.thr_scan_months),
            "n_train_rows": int(len(df_refit)),
            "n_scan_rows": int(len(df_scan)),
            "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
            "cost_per_signal": float(args.cost_per_signal),
            "use_calibrator": cal_long is not None,
            "exclude_patterns": exclude_patterns,
        }, fh, indent=2, default=str)

    print("\n" + "═" * 70)
    print(f"  📦 PRODUCCIÓN PERSISTIDA en {out_dir}/")
    print(f"     · {booster_path.name}")
    if cal_long is not None:
        print(f"     · {cal_path.name}")
    print(f"     · {features_path.name}")
    print(f"     · {threshold_path.name}    (thr={thr_prod:.4f})")
    print(f"     · {manifest_path.name}")
    print(f"  💡 Re-correr este script cada ~30 días para refrescar.")
    print("═" * 70)


if __name__ == "__main__":
    main()
