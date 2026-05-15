"""
main_oof_gbm_holdout.py
═══════════════════════════════════════════════════════════════════════════
FASE 2 GBM: validación honesta en holdout (2025-11 → 2026-04).

PROCESO:
  1. Carga best_per_side.json → identifica el trial ganador LONG y SHORT.
  2. Reconstruye configs idénticos al training (FeatureConfig, regime weights,
     barriers). Resultado: df_prepared en el MISMO espacio de features que vio
     el modelo durante el tuning.
  3. Para cada side (LONG, SHORT):
       a. Refit interno: 90% train → train, 10% final train → val para early stop.
          Captura best_iteration.
       b. Refit final: 100% train con num_boost_round=best_iteration.
       c. Predict sobre holdout.
       d. Calibrar con el IsotonicRegression OOF persistido (no re-calibrar:
          el calibrator OOF YA vio todo el train vía k-folds).
  4. Evaluar holdout con DOS lentes:
       - HONEST: aplicar thr fijo del OOF (lo que iría a producción).
       - OPTIMISTIC: escanear thr en holdout (información, NO para decisión).
  5. Persistir reporte JSON + métricas en stdout.

USO:
  python -m mimo.oof.main_oof_gbm_holdout \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --variant-long vol_boost_td_down --variant-short vol_boost \\
    --label-horizon-long 3 --label-horizon-short 3 \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/holdout_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.ev_objective import compute_ev_at_best_threshold
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE,
    LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES,
    _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release,
    _get_feature_masks_for_release,
    _tf_defaults,
    install_regime_weight_patch,
    resolve_regime_weights,
    set_global_seeds,
)


# Hiperparámetros LGBM que vienen del trial (suggest_*); el resto son fijos.
_TUNED_LGBM_KEYS = {
    "num_leaves", "max_depth", "min_data_in_leaf",
    "feature_fraction", "bagging_fraction", "bagging_freq",
    "lambda_l1", "lambda_l2", "learning_rate",
}
_TUNED_META_KEYS = {"n_estimators", "early_stopping_rounds"}


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _split_trial_params(params: Dict[str, Any], seed: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Separa params del trial en (lgbm_params_full, meta)."""
    lgbm = {
        "objective": "binary",
        "metric": "average_precision",
        "verbose": -1,
        "seed": seed,
        "n_jobs": -1,
        "boosting_type": "gbdt",
    }
    for k in _TUNED_LGBM_KEYS:
        if k in params:
            lgbm[k] = params[k]
    meta = {k: params[k] for k in _TUNED_META_KEYS if k in params}
    return lgbm, meta


def _collect_tabular_columns(pipeline: DataPipeline) -> list:
    all_cols = pipeline._get_all_feature_columns()
    flat = []
    for key in ("sequence_short", "sequence_long", "context", "time"):
        for c in all_cols.get(key, []):
            if c not in flat:
                flat.append(c)
    return flat


def _refit_with_internal_val(
    *,
    X: np.ndarray,
    y: np.ndarray,
    w: np.ndarray,
    lgbm_params: Dict[str, Any],
    n_estimators: int,
    early_stopping_rounds: int,
    val_frac: float = 0.10,
) -> Tuple[lgb.Booster, int]:
    """Refit en 2 pasos:
      a) 90% train → train, 10% más reciente → val. Early stopping da best_iter.
      b) 100% train, num_boost_round=best_iter. Booster final = el de (b).

    Devuelve (booster_final, best_iteration).
    """
    n = len(X)
    n_val = max(int(n * val_frac), 1000)
    n_tr = n - n_val

    X_tr, X_va = X[:n_tr], X[n_tr:]
    y_tr, y_va = y[:n_tr], y[n_tr:]
    w_tr, w_va = w[:n_tr], w[n_tr:]

    # scale_pos_weight sobre TR
    n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
    params = dict(lgbm_params)
    params["scale_pos_weight"] = float(n_neg / max(n_pos, 1))

    # Paso (a) — early stop
    train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
    val_ds = lgb.Dataset(X_va, label=y_va, weight=w_va, reference=train_ds)
    booster_es = lgb.train(
        params, train_ds,
        num_boost_round=int(n_estimators),
        valid_sets=[val_ds],
        callbacks=[lgb.early_stopping(stopping_rounds=int(early_stopping_rounds), verbose=False)],
    )
    best_iter = int(booster_es.best_iteration) if booster_es.best_iteration else int(n_estimators)
    print(f"     · early-stop best_iter = {best_iter} / {n_estimators}")

    # Paso (b) — refit 100% con best_iter
    # scale_pos_weight recalculado sobre el 100%
    n_pos_full = int(y.sum()); n_neg_full = len(y) - n_pos_full
    params_full = dict(lgbm_params)
    params_full["scale_pos_weight"] = float(n_neg_full / max(n_pos_full, 1))
    full_ds = lgb.Dataset(X, label=y, weight=w)
    booster_full = lgb.train(params_full, full_ds, num_boost_round=best_iter)
    return booster_full, best_iter


def _eval_at_thr(df_oof: pd.DataFrame, *, proba_col: str, side_is_long: bool,
                 horizon: int, tp_mult: float, sl_mult: float,
                 thr_target: float, cost_per_signal: float,
                 max_drawdown_R: float, min_signals: int) -> Dict[str, Any]:
    """EV-net evaluado a UN threshold fijo (honest: el del OOF aplicado a holdout)."""
    # compute_ev_at_best_threshold con n_thr=1 escanea ese único thr
    return compute_ev_at_best_threshold(
        df_oof, proba_col=proba_col, side_is_long=side_is_long,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=1, thr_lo=thr_target, thr_hi=thr_target,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )


def _eval_best_thr(df_oof: pd.DataFrame, *, proba_col: str, side_is_long: bool,
                   horizon: int, tp_mult: float, sl_mult: float,
                   cost_per_signal: float, max_drawdown_R: float, min_signals: int,
                   thr_lo: float = 0.05, thr_hi: float = 0.60,
                   n_thr: int = 80) -> Dict[str, Any]:
    """EV-net con sweep de thresholds en holdout (optimistic — solo informativo)."""
    return compute_ev_at_best_threshold(
        df_oof, proba_col=proba_col, side_is_long=side_is_long,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )


# ─── CLI ──────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--inherit-config-from", default=None)
    ap.add_argument("--best-json", required=True,
                    help="Path al best_per_side.json generado en fase 1.")
    ap.add_argument("--base-tf", default="5min")

    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()), default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()), default="moderate")
    ap.add_argument("--regime-weights-long-json", default=None)
    ap.add_argument("--regime-weights-short-json", default=None)
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)

    ap.add_argument("--train-from", type=_parse_date, required=True)
    ap.add_argument("--train-to",   type=_parse_date, required=True)
    ap.add_argument("--holdout-from", type=_parse_date, required=True)
    ap.add_argument("--holdout-to",   type=_parse_date, required=True)

    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--ev-min-signals", type=int, default=30,
                    help="Mín signals en holdout para que EV sea reportable "
                         "(más bajo que en train porque holdout es ~6 meses).")
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)

    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")
    ap.add_argument("--seed", type=int, default=47)

    ap.add_argument("--out-json", default=None,
                    help="Si se pasa, persiste el reporte completo a JSON.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)

    # ─── 1. Inherit config (no-destructivo) ─────────────────────────
    if args.inherit_config_from:
        src = str(args.inherit_config_from); dst = release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:  _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    # ─── 2. Cargar best_per_side.json ──────────────────────────────
    print(f"\n📂 Cargando best_per_side.json: {args.best_json}")
    with open(args.best_json) as fh:
        best = json.load(fh)
    if not best.get("top_long") or not best.get("top_short"):
        raise SystemExit("❌ best_per_side.json no tiene top_long/top_short")
    top_long_meta = best["top_long"][0]
    top_short_meta = best["top_short"][0]
    n_long = int(top_long_meta["trial"])
    n_short = int(top_short_meta["trial"])
    thr_long = float(top_long_meta["ev_long"]["thr"])
    thr_short = float(top_short_meta["ev_short"]["thr"])
    print(f"   LONG  best:  trial #{n_long} | thr={thr_long:.4f}")
    print(f"   SHORT best:  trial #{n_short} | thr={thr_short:.4f}")

    # ─── 3. Cargar Optuna trial para coger user_attrs (calibrator paths) ─
    study_name = f"{args.study_prefix}_{release}_multitask"
    print(f"📂 Optuna study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    trial_long_obj = next((t for t in study.trials if t.number == n_long), None)
    trial_short_obj = next((t for t in study.trials if t.number == n_short), None)
    if trial_long_obj is None or trial_short_obj is None:
        raise SystemExit("❌ No se encontraron los trials ganadores en el study")

    cal_long_path = trial_long_obj.user_attrs.get("cal_long_path")
    cal_short_path = trial_short_obj.user_attrs.get("cal_short_path")
    if not cal_long_path or not os.path.exists(cal_long_path):
        raise SystemExit(f"❌ calibrator LONG no encontrado: {cal_long_path}")
    if not cal_short_path or not os.path.exists(cal_short_path):
        raise SystemExit(f"❌ calibrator SHORT no encontrado: {cal_short_path}")
    cal_long = joblib.load(cal_long_path)
    cal_short = joblib.load(cal_short_path)
    print(f"   cal_long  : {cal_long_path}")
    print(f"   cal_short : {cal_short_path}")

    # Separar params del trial
    lgbm_long, meta_long = _split_trial_params(trial_long_obj.params, seed=int(args.seed))
    lgbm_short, meta_short = _split_trial_params(trial_short_obj.params, seed=int(args.seed))
    print(f"   LONG  hparams: {trial_long_obj.params}")
    print(f"   SHORT hparams: {trial_short_obj.params}")

    # ─── 4. Regime weights (mismo que training) ─────────────────────
    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    # ─── 5. Cargar OHLCV completo (train + holdout) ─────────────────
    print(f"\n📊 Cargando OHLCV {args.train_from.date()} → {args.holdout_to.date()}")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras")

    # ─── 6. Reconstruir FeatureConfig idéntico al training ──────────
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
    general_config = Config(release=release, use_oof=True, oof_splits=5,
                            oof_epochs=1, save_oof_artifacts=True)
    regime_config = StateConfig(adx_trend_threshold=25.0)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=ModelConfig(seq_len_short=64, seq_len_long=256, target_type="multitask"),
        regime_config=regime_config,
    )

    # ─── 7. prepare_data ─────────────────────────────────────────────
    print("⚙️  prepare_data...")
    df_prepared = pipeline.prepare_data(
        df_rates, labels=True, side="both",
        set_market_condition=False, ensure_regime=True,
    )

    # ─── 8. Feature matrix + split ──────────────────────────────────
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]
    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long", "signal_short"]
    keep = df_prepared[needed].notna().all(axis=1)
    df_clean = df_prepared.loc[keep].reset_index(drop=True)
    print(f"   {len(df_clean):,} filas × {len(feat_cols)} features tras limpieza")

    train_mask = df_clean["time"] < pd.Timestamp(args.holdout_from)
    hold_mask  = df_clean["time"] >= pd.Timestamp(args.holdout_from)
    df_train = df_clean.loc[train_mask].reset_index(drop=True)
    df_hold  = df_clean.loc[hold_mask].reset_index(drop=True)
    print(f"   train: {len(df_train):,} filas ({df_train['time'].min()} → {df_train['time'].max()})")
    print(f"   hold : {len(df_hold):,} filas ({df_hold['time'].min()} → {df_hold['time'].max()})")
    if len(df_hold) < 1000:
        raise SystemExit("❌ Holdout demasiado pequeño")

    X_tr = df_train[feat_cols].astype(np.float32).values
    y_long_tr  = df_train["signal_long"].astype(np.int8).values
    y_short_tr = df_train["signal_short"].astype(np.int8).values
    w_tr = df_train["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_train.columns else np.ones(len(df_train), dtype=np.float32)

    X_hd = df_hold[feat_cols].astype(np.float32).values

    print(f"   LONG  positive rate train: {y_long_tr.mean():.4f}  "
          f"hold: {df_hold['signal_long'].astype(np.int8).mean():.4f}")
    print(f"   SHORT positive rate train: {y_short_tr.mean():.4f}  "
          f"hold: {df_hold['signal_short'].astype(np.int8).mean():.4f}")

    # ─── 9. REFIT LONG ──────────────────────────────────────────────
    print(f"\n🌳 Refit LONG (params trial #{n_long})...")
    booster_long, bi_long = _refit_with_internal_val(
        X=X_tr, y=y_long_tr, w=w_tr,
        lgbm_params=lgbm_long,
        n_estimators=int(meta_long.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_long.get("early_stopping_rounds", 50)),
    )

    # ─── 10. REFIT SHORT ────────────────────────────────────────────
    print(f"🌳 Refit SHORT (params trial #{n_short})...")
    booster_short, bi_short = _refit_with_internal_val(
        X=X_tr, y=y_short_tr, w=w_tr,
        lgbm_params=lgbm_short,
        n_estimators=int(meta_short.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_short.get("early_stopping_rounds", 50)),
    )

    # ─── 11. Predict + calibrate sobre holdout ──────────────────────
    print("\n🔮 Predict + calibrate sobre holdout...")
    p_long_raw  = booster_long.predict(X_hd).astype(np.float32)
    p_short_raw = booster_short.predict(X_hd).astype(np.float32)
    p_long_cal  = cal_long.predict(p_long_raw.astype(np.float64)).astype(np.float32)
    p_short_cal = cal_short.predict(p_short_raw.astype(np.float64)).astype(np.float32)

    # ─── 12. Build df_oof-style para evaluación ─────────────────────
    df_oof_hold = pd.DataFrame({
        "time":  df_hold["time"].values,
        "high":  df_hold["high"].astype(np.float64).values,
        "low":   df_hold["low"].astype(np.float64).values,
        "close": df_hold["close"].astype(np.float64).values,
        "atr":   df_hold["atr"].astype(np.float64).values,
        "signal_long":  df_hold["signal_long"].astype(np.int8).values,
        "signal_short": df_hold["signal_short"].astype(np.int8).values,
        "oof_proba_long_cal":  p_long_cal,
        "oof_proba_short_cal": p_short_cal,
    })

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    # ─── 13. Honest (thr OOF aplicado) ──────────────────────────────
    print("\n🎯 Evaluación HONEST (thr fijo del OOF aplicado a holdout):")
    long_honest = _eval_at_thr(
        df_oof_hold, proba_col="oof_proba_long_cal", side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        thr_target=thr_long, cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R, min_signals=args.ev_min_signals,
    )
    short_honest = _eval_at_thr(
        df_oof_hold, proba_col="oof_proba_short_cal", side_is_long=False,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        thr_target=thr_short, cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R, min_signals=args.ev_min_signals,
    )

    def _fmt(side, oof_res, holdout_res, thr_oof):
        ev_oof = oof_res.get("ev_net", float("nan"))
        sig_oof = int(oof_res.get("n_signals", 0))
        prec_oof = oof_res.get("prec_TP", float("nan"))
        ev_h = holdout_res.get("ev_net", float("nan"))
        sig_h = int(holdout_res.get("n_signals", 0))
        prec_h = holdout_res.get("prec_TP", float("nan"))
        mdd_h = holdout_res.get("mdd_R", float("nan"))
        erosion = (ev_h - ev_oof) / abs(ev_oof) * 100 if ev_oof and np.isfinite(ev_oof) and ev_oof != 0 else float("nan")
        return (f"  {side} | thr={thr_oof:.4f}\n"
                f"     OOF       : ev_net={ev_oof:+.4f}R  sig={sig_oof}  prec_TP={prec_oof:.3f}\n"
                f"     HOLDOUT   : ev_net={ev_h:+.4f}R  sig={sig_h}  prec_TP={prec_h:.3f}  mdd={mdd_h:.1f}R\n"
                f"     Erosión EV: {erosion:+.1f}%")

    print(_fmt("LONG ", top_long_meta["ev_long"], long_honest, thr_long))
    print(_fmt("SHORT", top_short_meta["ev_short"], short_honest, thr_short))

    # ─── 14. Optimistic (thr re-escaneado en holdout) ──────────────
    print("\n🔍 Evaluación OPTIMISTIC (thr re-optimizado en holdout — solo info):")
    long_opt = _eval_best_thr(
        df_oof_hold, proba_col="oof_proba_long_cal", side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R, min_signals=args.ev_min_signals,
    )
    short_opt = _eval_best_thr(
        df_oof_hold, proba_col="oof_proba_short_cal", side_is_long=False,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R, min_signals=args.ev_min_signals,
    )
    print(f"  LONG  | best thr_hold={long_opt.get('thr', float('nan')):.4f}  "
          f"ev_net={long_opt.get('ev_net', float('nan')):+.4f}R  "
          f"sig={int(long_opt.get('n_signals', 0))}  "
          f"prec_TP={long_opt.get('prec_TP', float('nan')):.3f}")
    print(f"  SHORT | best thr_hold={short_opt.get('thr', float('nan')):.4f}  "
          f"ev_net={short_opt.get('ev_net', float('nan')):+.4f}R  "
          f"sig={int(short_opt.get('n_signals', 0))}  "
          f"prec_TP={short_opt.get('prec_TP', float('nan')):.3f}")

    # ─── 15. Veredicto + R total ───────────────────────────────────
    total_R_honest = (
        float(long_honest.get("ev_net", 0.0)) * int(long_honest.get("n_signals", 0))
        + float(short_honest.get("ev_net", 0.0)) * int(short_honest.get("n_signals", 0))
    )
    months = (args.holdout_to - args.holdout_from).days / 30.44
    print("\n" + "═" * 70)
    print(f"  POTENCIAL HOLDOUT (thr OOF aplicado): {total_R_honest:+.2f}R en {months:.1f} meses")
    print(f"  → ~{total_R_honest/months:+.2f}R/mes  (un R = 1× volatilidad ATR)")
    print("═" * 70)

    # ─── 16. Persistir reporte JSON ────────────────────────────────
    report = {
        "release": release,
        "best_json": str(args.best_json),
        "train_period":   [str(args.train_from.date()), str(args.train_to.date())],
        "holdout_period": [str(args.holdout_from.date()), str(args.holdout_to.date())],
        "long_trial":  n_long,
        "short_trial": n_short,
        "long_best_iter_internal":  bi_long,
        "short_best_iter_internal": bi_short,
        "oof": {
            "long":  _json_safe(top_long_meta["ev_long"]),
            "short": _json_safe(top_short_meta["ev_short"]),
        },
        "holdout_honest": {
            "thr_long":  thr_long,
            "thr_short": thr_short,
            "long":  _json_safe(long_honest),
            "short": _json_safe(short_honest),
            "total_R": total_R_honest,
            "months":  months,
        },
        "holdout_optimistic": {
            "long":  _json_safe(long_opt),
            "short": _json_safe(short_opt),
        },
        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "holdout_report.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


def _json_safe(d):
    if not isinstance(d, dict): return d
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            out[k] = ("nan" if np.isnan(fv) else
                      ("inf" if np.isinf(fv) and fv > 0 else
                       ("-inf" if np.isinf(fv) else fv)))
        elif isinstance(v, (np.integer, int)): out[k] = int(v)
        elif isinstance(v, np.ndarray): out[k] = v.tolist()
        else: out[k] = v
    return out


if __name__ == "__main__":
    main()
