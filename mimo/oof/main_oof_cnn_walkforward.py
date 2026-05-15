"""
main_oof_cnn_walkforward.py
═══════════════════════════════════════════════════════════════════════════
WALKFORWARD CNN-LSTM con refit por ventana (equivalente al
main_oof_gbm_walkforward.py).

Para cada ventana mensual (default 15 ventanas, 2025-01 a 2026-04):
  · Carga OHLCV completo y aplica prepare_data
  · Split temporal train_window vs test_window
  · Entrena UN modelo CNN multitask con los best hyperparams del Optuna study
  · Predice sobre test_window y calcula EV-net por side (LONG/SHORT)
  · Threshold scan independiente por side y ventana

Genera walkforward_report_cnn.json compatible con diag_long_only_simulator.py
para comparativa formal vs el walkforward GBM ya generado.

SIMPLIFICACIONES vs producción CNN-202500:
  · Sin specialists (solo CNN base multitask)
  · Sin RL gate
  · Sin calibrator isotónico (raw probs + threshold scan per ventana)
  · Mismos hyperparams del best Optuna trial del CNN

USO:
  python -m mimo.oof.main_oof_cnn_walkforward \\
    --release 202500 \\
    --cnn-study-name oof_study_202500_multitask \\
    --walk-from 2025-01-01 --walk-to 2026-04-10 \\
    --train-months 12 --test-months 1 --step-months 1 \\
    --epochs 40 --patience 8 \\
    --out-json artifacts/202500/oof/<tag>/reports/walkforward_report_cnn.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig, TradingModel
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.ev_objective import compute_ev_at_best_threshold
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE, LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release, _get_feature_masks_for_release, _tf_defaults,
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)


# ─── helpers ────────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _generate_windows(
    *, walk_from: datetime, walk_to: datetime,
    train_months: int, test_months: int, step_months: int,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """Devuelve (train_start, train_end, test_start, test_end) por ventana."""
    out = []
    test_start = walk_from
    while True:
        test_end = test_start + relativedelta(months=test_months)
        if test_end > walk_to:
            break
        train_start = test_start - relativedelta(months=train_months)
        train_end = test_start
        out.append((train_start, train_end, test_start, test_end))
        test_start = test_start + relativedelta(months=step_months)
    return out


def _load_best_params_from_study(study_name: str, storage: str) -> Tuple[Dict[str, Any], int]:
    """Extrae los best params del CNN Optuna study. Devuelve (params, trial_number)."""
    print(f"📂 Optuna study CNN: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=storage)
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    if not completed:
        raise SystemExit(f"❌ Study {study_name} sin trials COMPLETE")
    # Best por value (Optuna direction=maximize)
    best = max(completed, key=lambda t: (t.value if t.value is not None else float("-inf")))
    print(f"   Best trial #{best.number}  value={best.value:+.4f}")
    print(f"   Params: {best.params}")
    return dict(best.params), int(best.number)


def _model_config_from_params(params: Dict[str, Any], *, target_type: str = "multitask",
                              epochs: int = 40, patience: int = 8) -> ModelConfig:
    """Construye un ModelConfig desde los best params del Optuna trial."""
    mc = ModelConfig(
        seq_len_short=int(params.get("seq_len_short", 24)),
        seq_len_long=int(params.get("seq_len_long", 96)),
        epochs=int(epochs),
        patience=int(patience),
        target_type=target_type,
        # Capacidad
        conv1d_filters=int(params.get("conv1d_filters", 64)),
        lstm_units=int(params.get("lstm_units", 64)),
        context_units=int(params.get("context_units", 32)),
        head_units=int(params.get("head_units", 64)),
        time_units=int(params.get("time_units", 16)),
        # Regularización
        dropout_seq=float(params.get("dropout_seq", 0.05)),
        dropout_lstm=float(params.get("dropout_lstm", 0.1)),
        dropout_dense=float(params.get("dropout_dense", 0.1)),
        l2_reg=float(params.get("l2_reg", 1e-5)),
        # Optimizer
        learning_rate=float(params.get("learning_rate", 5e-4)),
        batch_size=int(params.get("batch_size", 2048)),
        # Loss
        focal_gamma=float(params.get("focal_gamma", 1.0)),
        focal_alpha_long=float(params.get("focal_alpha_long", 0.30)),
        focal_alpha_short=float(params.get("focal_alpha_short", 0.30)),
        loss_weight_long=float(params.get("loss_weight_long", 1.0)),
        loss_weight_short=float(params.get("loss_weight_short", 1.0)),
        ranking_loss_weight=float(params.get("ranking_loss_weight", 0.0)),
        # Arch
        use_hierarchical_fusion=bool(params.get("use_hierarchical_fusion", True)),
    )
    return mc


def _train_and_predict_window(
    *,
    df_prepared: pd.DataFrame,
    train_start: datetime, train_end: datetime,
    test_start: datetime, test_end: datetime,
    general_config: Config, model_config: ModelConfig,
    feature_config: FeatureConfig, regime_config: StateConfig,
    seed: int = 47,
) -> Optional[Dict[str, Any]]:
    """Entrena CNN multitask sobre train_window, predice sobre test_window.
    Devuelve dict con df_eval (con probs y outcomes simulados) o None si falla."""
    import tensorflow as tf
    tf.keras.utils.set_random_seed(int(seed))

    # Split por fecha
    df_train = df_prepared.loc[
        (df_prepared["time"] >= pd.Timestamp(train_start)) &
        (df_prepared["time"] <  pd.Timestamp(train_end))
    ].reset_index(drop=True)
    df_test = df_prepared.loc[
        (df_prepared["time"] >= pd.Timestamp(test_start)) &
        (df_prepared["time"] <  pd.Timestamp(test_end))
    ].reset_index(drop=True)

    L = int(model_config.seq_len_long)
    if len(df_train) < max(L * 5, 10000) or len(df_test) < L + 100:
        return {"skipped": True, "reason": "too_few_rows",
                "n_train": len(df_train), "n_test": len(df_test)}

    # Pipeline NUEVA por ventana (scalers vírgenes)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=model_config, regime_config=regime_config,
    )

    # Sequences train con fit_scalers=True
    try:
        seq_train = pipeline.create_sequences_by_side(
            df_train, sides=("long", "short"),
            fit_scalers=True, train=True,
        )
    except Exception as e:
        return {"skipped": True, "reason": f"create_sequences_train_failed: {str(e)[:100]}"}

    # En multitask, los inputs/labels son compartidos. Tomamos 'long' como representante.
    pack = seq_train.get("long") or {}
    X_seq_short = pack.get("seq_short"); X_seq_long = pack.get("seq_long")
    X_context   = pack.get("context");   X_time     = pack.get("time")
    y_train     = pack.get("labels")     # multitask → shape (N, 2)
    if y_train is None or y_train.ndim != 2:
        return {"skipped": True, "reason": f"y_train shape inesperado: "
                                            f"{None if y_train is None else y_train.shape}"}

    # Validation interna (90/10 de train)
    n = len(X_seq_long)
    n_val = max(int(n * 0.10), 500)
    n_tr  = n - n_val
    X_tr = [X_seq_short[:n_tr], X_seq_long[:n_tr], X_context[:n_tr], X_time[:n_tr]]
    X_va = [X_seq_short[n_tr:], X_seq_long[n_tr:], X_context[n_tr:], X_time[n_tr:]]
    y_tr = y_train[:n_tr]
    y_va = y_train[n_tr:]

    # Build & train modelo
    try:
        model = TradingModel(
            general_config=general_config, model_config=model_config, side=None,  # multitask
        )
        if getattr(model_config, "use_hierarchical_fusion", True):
            model.build_model_v3()
        else:
            model.build_model_v2()
        model.compile_model()
    except Exception as e:
        return {"skipped": True, "reason": f"build_model_failed: {str(e)[:150]}"}

    try:
        history = model.train(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va,   y_val=y_va,
            sample_weight=pack.get("sample_weight"),
            verbose=0,
        )
        best_iter = int(history.get("best_epoch", len(history.get("loss", [])) if isinstance(history, dict) else 0))
    except Exception as e:
        return {"skipped": True, "reason": f"train_failed: {str(e)[:200]}"}

    # Sequences test con fit_scalers=False
    try:
        seq_test = pipeline.create_sequences_by_side(
            df_test, sides=("long", "short"),
            fit_scalers=False, train=False,
        )
    except Exception as e:
        return {"skipped": True, "reason": f"create_sequences_test_failed: {str(e)[:100]}"}

    pack_te = seq_test.get("long") or {}
    Xt_seq_short = pack_te.get("seq_short"); Xt_seq_long = pack_te.get("seq_long")
    Xt_context   = pack_te.get("context");   Xt_time     = pack_te.get("time")
    if Xt_seq_long is None or len(Xt_seq_long) == 0:
        return {"skipped": True, "reason": "test_sequences_empty"}

    # Predict
    try:
        preds = model.predict([Xt_seq_short, Xt_seq_long, Xt_context, Xt_time])
        # Para multitask: preds shape (N, 2) con [p_long, p_short]
        if preds.ndim != 2 or preds.shape[1] != 2:
            return {"skipped": True, "reason": f"preds shape inesperado: {preds.shape}"}
        p_long = preds[:, 0].astype(np.float32)
        p_short = preds[:, 1].astype(np.float32)
    except Exception as e:
        return {"skipped": True, "reason": f"predict_failed: {str(e)[:200]}"}

    # Alinear con df_test: context_offset = seq_len_long - 1
    context_offset = L - 1
    df_te_aligned = df_test.iloc[context_offset:context_offset + len(p_long)].reset_index(drop=True)
    if len(df_te_aligned) < len(p_long):
        # truncar predictions si hay desalineamiento
        p_long  = p_long[:len(df_te_aligned)]
        p_short = p_short[:len(df_te_aligned)]

    # Build df_oof-style para evaluar
    df_eval = pd.DataFrame({
        "time": df_te_aligned["time"].values,
        "high": df_te_aligned["high"].astype(np.float64).values,
        "low":  df_te_aligned["low"].astype(np.float64).values,
        "close":df_te_aligned["close"].astype(np.float64).values,
        "atr":  df_te_aligned["atr"].astype(np.float64).values,
        "signal_long":  df_te_aligned["signal_long"].astype(np.int8).values
            if "signal_long" in df_te_aligned.columns else 0,
        "signal_short": df_te_aligned["signal_short"].astype(np.int8).values
            if "signal_short" in df_te_aligned.columns else 0,
        "p_long_raw":  p_long,
        "p_short_raw": p_short,
    })
    return {
        "skipped": False,
        "df_eval": df_eval,
        "n_train": len(df_train), "n_test": len(df_test),
        "best_epoch": best_iter,
        "n_features_seq_long": X_seq_long.shape[-1],
        "n_features_context": X_context.shape[-1] if X_context is not None else 0,
    }


def _eval_predictions(
    df_eval: pd.DataFrame, *, horizon: int, tp_mult: float, sl_mult: float,
    cost_per_signal: float, max_drawdown_R: float,
    min_signals: int, use_raw_probs: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Threshold scan por side sobre df_eval y devuelve (long_res, short_res)."""
    if use_raw_probs:
        thr_lo, thr_hi, n_thr = 0.05, 0.95, 180
    else:
        thr_lo, thr_hi, n_thr = 0.05, 0.60, 80

    long_res = compute_ev_at_best_threshold(
        df_eval, proba_col="p_long_raw", side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )
    short_res = compute_ev_at_best_threshold(
        df_eval, proba_col="p_short_raw", side_is_long=False,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals, max_drawdown_R=max_drawdown_R,
    )
    return long_res, short_res


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(d, dict): return d
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            out[k] = None if (np.isnan(fv) or np.isinf(fv)) else fv
        elif isinstance(v, (np.integer, int)): out[k] = int(v)
        elif isinstance(v, np.ndarray): out[k] = v.tolist()
        else: out[k] = v
    return out


def _summarize(results: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    valid = [r for r in results if not r.get("skipped") and
             r.get(side, {}).get("ev_net") is not None]
    if not valid:
        return {"n_windows": 0}
    evs = np.array([r[side]["ev_net"] for r in valid], dtype=np.float64)
    sigs = np.array([r[side].get("n_signals", 0) or 0 for r in valid], dtype=np.int64)
    precs = np.array([r[side].get("prec_TP", float("nan")) for r in valid], dtype=np.float64)
    contrib = np.where(np.isfinite(evs), evs * sigs, 0.0)
    return {
        "n_windows": len(valid),
        "n_pos_windows": int((evs > 0).sum()),
        "pwr": float((evs > 0).mean()),
        "ev_median": float(np.nanmedian(evs)),
        "ev_p10":    float(np.nanpercentile(evs, 10)),
        "ev_p90":    float(np.nanpercentile(evs, 90)),
        "n_signals_total":  int(sigs.sum()),
        "n_signals_median": float(np.median(sigs)),
        "prec_median":      float(np.nanmedian(precs)),
        "R_total":          float(contrib.sum()),
    }


# ─── CLI ──────────────────────────────────────────────────────────────

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--cnn-study-name", required=True,
                    help="Nombre del Optuna study del CNN (ej. oof_study_202500_multitask).")
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()), default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()), default="moderate")
    ap.add_argument("--regime-weights-long-json", default=None)
    ap.add_argument("--regime-weights-short-json", default=None)
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)

    ap.add_argument("--walk-from", type=_parse_date, required=True)
    ap.add_argument("--walk-to", type=_parse_date, required=True)
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--step-months", type=int, default=1)

    ap.add_argument("--epochs", type=int, default=40,
                    help="Epochs por ventana (más bajo que producción para acelerar).")
    ap.add_argument("--patience", type=int, default=8,
                    help="Early stopping patience.")

    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--min-signals-window", type=int, default=15)

    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)

    # 1) Best CNN params
    best_params, best_trial_n = _load_best_params_from_study(
        args.cnn_study_name, args.optuna_storage)
    model_config = _model_config_from_params(
        best_params, target_type="multitask",
        epochs=int(args.epochs), patience=int(args.patience),
    )

    # 2) Regime weights + barriers + features (mismo patrón que GBM)
    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    # 3) Generar ventanas
    windows = _generate_windows(
        walk_from=args.walk_from, walk_to=args.walk_to,
        train_months=args.train_months, test_months=args.test_months,
        step_months=args.step_months,
    )
    if not windows: raise SystemExit("❌ Sin ventanas generadas")
    earliest = windows[0][0] - relativedelta(months=2)  # padding features
    latest = windows[-1][3]

    print(f"\n🪟 {len(windows)} ventanas: train={args.train_months}m "
          f"test={args.test_months}m step={args.step_months}m  "
          f"({args.walk_from.date()} → {args.walk_to.date()})")

    # 4) OHLCV completo
    print(f"\n📊 Cargando OHLCV {earliest.date()} → {latest.date()}")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=earliest, to_date=latest, resample=resample,
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras")

    # 5) Build configs (general, feature, regime) — fixed para todas las ventanas
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
                            oof_epochs=int(args.epochs), save_oof_artifacts=False)
    regime_config = StateConfig(adx_trend_threshold=25.0)

    # 6) prepare_data UNA vez sobre todo el rango (más eficiente)
    print("⚙️  prepare_data global (esto puede tardar)...")
    pipeline_master = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=model_config, regime_config=regime_config,
    )
    df_prepared = pipeline_master.prepare_data(
        df_rates, labels=True, side="both",
        set_market_condition=False, ensure_regime=True,
    )
    print(f"   {len(df_prepared):,} filas tras prepare_data")

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    # 7) Walk-forward loop
    print(f"\n🔄 Walkforward CNN refit por ventana | epochs={args.epochs} "
          f"patience={args.patience}")

    results: List[Dict[str, Any]] = []
    for i, (ts, te, vs, ve) in enumerate(windows):
        t0 = time.time()
        print(f"\n── Window {i+1}/{len(windows)} | train={ts.date()}→{te.date()} "
              f"test={vs.date()}→{ve.date()}")
        try:
            tr_out = _train_and_predict_window(
                df_prepared=df_prepared,
                train_start=ts, train_end=te, test_start=vs, test_end=ve,
                general_config=general_config, model_config=model_config,
                feature_config=feature_config, regime_config=regime_config,
                seed=int(args.seed),
            )
        except Exception as e:
            tr_out = {"skipped": True, "reason": f"unexpected_error: {str(e)[:200]}"}

        if tr_out.get("skipped"):
            print(f"   ⏭️  saltada: {tr_out.get('reason', 'unknown')}")
            results.append({
                "window": [str(ts.date()), str(te.date()),
                           str(vs.date()), str(ve.date())],
                "n_train": tr_out.get("n_train", 0),
                "n_test":  tr_out.get("n_test", 0),
                "skipped": True, "_reason": tr_out.get("reason"),
            })
            continue

        df_eval = tr_out["df_eval"]
        long_res, short_res = _eval_predictions(
            df_eval, horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=args.cost_per_signal,
            max_drawdown_R=args.max_drawdown_R,
            min_signals=args.min_signals_window,
            use_raw_probs=True,   # raw probs + threshold scan
        )

        dt_sec = time.time() - t0
        ev_l = long_res.get("ev_net"); sig_l = int(long_res.get("n_signals", 0) or 0)
        ev_s = short_res.get("ev_net"); sig_s = int(short_res.get("n_signals", 0) or 0)
        print(f"   ⏱  {dt_sec:.0f}s | best_epoch={tr_out.get('best_epoch')}")
        if ev_l is not None and np.isfinite(ev_l):
            print(f"   LONG  ev_net={ev_l:+.4f}R  sig={sig_l}  prec={long_res.get('prec_TP', 0):.3f}")
        else:
            print(f"   LONG  sin señales")
        if ev_s is not None and np.isfinite(ev_s):
            print(f"   SHORT ev_net={ev_s:+.4f}R  sig={sig_s}  prec={short_res.get('prec_TP', 0):.3f}")
        else:
            print(f"   SHORT sin señales")

        results.append({
            "window": [str(ts.date()), str(te.date()),
                       str(vs.date()), str(ve.date())],
            "n_train": tr_out["n_train"], "n_test": tr_out["n_test"],
            "best_epoch": tr_out.get("best_epoch"),
            "skipped": False,
            "long":  _json_safe(long_res),
            "short": _json_safe(short_res),
        })

    # 8) Summary
    summary_long  = _summarize(results, "long")
    summary_short = _summarize(results, "short")

    print("\n" + "═" * 70)
    print("  WALK-FORWARD CNN SUMMARY")
    print("═" * 70)
    for side, sm in (("LONG", summary_long), ("SHORT", summary_short)):
        if sm.get("n_windows", 0) == 0:
            print(f"  {side}: 0 ventanas válidas"); continue
        print(f"  {side}:")
        print(f"    Ventanas válidas        : {sm['n_windows']}")
        print(f"    PWR (positivas)         : {sm['n_pos_windows']}/{sm['n_windows']} "
              f"({100*sm['pwr']:.0f}%)")
        print(f"    EV mediana              : {sm['ev_median']:+.4f}R")
        print(f"    EV p10 / p90            : {sm['ev_p10']:+.4f}R / {sm['ev_p90']:+.4f}R")
        print(f"    Signals totales         : {sm['n_signals_total']}")
        print(f"    Prec mediana            : {sm['prec_median']:.3f}")
        print(f"    💰 R total              : {sm['R_total']:+.2f}R")

    # 9) Persist
    report = {
        "release": release,
        "cnn_study_name": args.cnn_study_name,
        "cnn_best_trial": best_trial_n,
        "cnn_best_params": best_params,
        "mode": "raw_probs",
        "walk_config": {
            "walk_from": str(args.walk_from.date()),
            "walk_to":   str(args.walk_to.date()),
            "train_months": args.train_months,
            "test_months":  args.test_months,
            "step_months":  args.step_months,
        },
        "epochs": int(args.epochs),
        "patience": int(args.patience),
        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
        "summary_long":  summary_long,
        "summary_short": summary_short,
        "windows": results,
    }
    out_json = args.out_json or f"artifacts/{release}/oof/walkforward_report_cnn.json"
    os.makedirs(os.path.dirname(os.path.abspath(out_json)), exist_ok=True)
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")

    if args.out_csv:
        rows = []
        for r in results:
            if r.get("skipped"): continue
            row = {"train_start": r["window"][0], "train_end": r["window"][1],
                   "test_start":  r["window"][2], "test_end":  r["window"][3]}
            for side in ("long", "short"):
                d = r.get(side) or {}
                row[f"{side}_ev_net"]  = d.get("ev_net")
                row[f"{side}_sig"]     = d.get("n_signals")
                row[f"{side}_prec"]    = d.get("prec_TP")
                row[f"{side}_thr"]     = d.get("thr")
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"📊 CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
