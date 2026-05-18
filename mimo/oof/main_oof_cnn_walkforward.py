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

# DETERMINISTIC=1 activa modo bit-exact en TensorFlow. Debe configurarse antes
# de cualquier import de TF (que ocurre transitivamente via mimo.models o
# directamente en _train_and_predict_window). Coste: 2-3x más lento en GPU.
if os.environ.get("DETERMINISTIC", "0") == "1":
    os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
    os.environ.setdefault("TF_CUDNN_DETERMINISTIC", "1")
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

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
    """Construye un ModelConfig desde los best params del Optuna trial.

    Robusto a evolución del schema: si el trial guardó keys que ya no son
    campos del dataclass (p.ej. focal_alpha_long/short cuando ModelConfig
    solo tiene focal_alpha único), las pasamos vía setattr y emitimos warning
    en lugar de crashear el constructor."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}

    # Defaults explícitos sobre params (params del trial sobreescriben defaults)
    candidate = {
        "seq_len_short": int(params.get("seq_len_short", 24)),
        "seq_len_long":  int(params.get("seq_len_long", 96)),
        "epochs":        int(epochs),
        "patience":      int(patience),
        "target_type":   target_type,
        "conv1d_filters": int(params.get("conv1d_filters", 64)),
        "lstm_units":     int(params.get("lstm_units", 64)),
        "context_units":  int(params.get("context_units", 32)),
        "head_units":     int(params.get("head_units", 64)),
        "time_units":     int(params.get("time_units", 16)),
        "dropout_seq":    float(params.get("dropout_seq", 0.05)),
        "dropout_lstm":   float(params.get("dropout_lstm", 0.1)),
        "dropout_dense":  float(params.get("dropout_dense", 0.1)),
        "l2_reg":         float(params.get("l2_reg", 1e-5)),
        "learning_rate":  float(params.get("learning_rate", 5e-4)),
        "batch_size":     int(params.get("batch_size", 2048)),
        "focal_gamma":    float(params.get("focal_gamma", 1.0)),
        # focal_alpha único: si el trial guardó _long/_short separados,
        # promediamos como mejor approximation para el campo único.
        "focal_alpha":    float(
            params.get("focal_alpha",
                       (float(params.get("focal_alpha_long", 0.30))
                        + float(params.get("focal_alpha_short", 0.30))) / 2.0)),
        "loss_weight_long":  float(params.get("loss_weight_long", 1.0)),
        "loss_weight_short": float(params.get("loss_weight_short", 1.0)),
        "ranking_loss_weight": float(params.get("ranking_loss_weight", 0.0)),
        "use_hierarchical_fusion": bool(params.get("use_hierarchical_fusion", True)),
        # mlp_flatten lo lee; otras archs ignoran (gelu hardcoded)
        "activation":     str(params.get("activation", "gelu")),
    }

    # Filtrar a campos válidos del dataclass (silenciosamente ignora los no reconocidos)
    init_kwargs = {k: v for k, v in candidate.items() if k in valid_fields}
    dropped = [k for k in candidate if k not in valid_fields]
    if dropped:
        print(f"   ⚠️ Params no soportados por ModelConfig (ignorados): {dropped}")

    mc = ModelConfig(**init_kwargs)

    # Atributos no-init (consumidos por compile_model vía getattr) — set
    # también los de _long/_short por si compile_model los lee directos.
    for k in ("focal_alpha_long", "focal_alpha_short"):
        if k in params:
            setattr(mc, k, float(params[k]))

    return mc


def _average_ensemble_long(out_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Promedia la columna `p_long_raw` a través de N modelos LONG entrenados
    con seeds distintos sobre la MISMA ventana (test, val_internal).

    Solo se promedia LONG; `p_short_raw` se hereda del primer miembro porque
    el ensemble se aplica únicamente al side problemático (LONG TCN v4 mostró
    varianza por seed alta; SHORT fue estable en 3 runs).

    Asume que todos los miembros producen df_eval/df_val_eval con MISMA
    estructura (mismo test slice, misma alineación seq_len_long). Si las
    longitudes difieren entre miembros (raro pero posible si algún predict
    trunca filas), se trunca al mínimo común antes de promediar.
    """
    assert len(out_list) >= 1, "ensemble requires at least 1 member"

    # ─── df_eval (test) ─────────────────────────────────────────────────
    min_len = min(len(o["df_eval"]) for o in out_list)
    members = [o for o in out_list]  # alias
    base = members[0]
    df_eval = base["df_eval"].iloc[:min_len].reset_index(drop=True).copy()
    p_stack = np.column_stack([
        o["df_eval"]["p_long_raw"].values[:min_len].astype(np.float64)
        for o in members
    ])
    df_eval["p_long_raw"] = p_stack.mean(axis=1).astype(np.float32)

    # ─── df_val_eval (val_internal, opcional cuando NO_LOOKAHEAD_SCANNER=1) ─
    df_val_eval = None
    val_evals = [o.get("df_val_eval") for o in members
                 if o.get("df_val_eval") is not None]
    if val_evals:
        min_len_va = min(len(d) for d in val_evals)
        df_val_eval = val_evals[0].iloc[:min_len_va].reset_index(drop=True).copy()
        pv_stack = np.column_stack([
            d["p_long_raw"].values[:min_len_va].astype(np.float64)
            for d in val_evals
        ])
        df_val_eval["p_long_raw"] = pv_stack.mean(axis=1).astype(np.float32)

    return {
        "skipped": False,
        "df_eval": df_eval,
        "df_val_eval": df_val_eval,
        "n_train": base.get("n_train"),
        "n_test":  base.get("n_test"),
        "n_val_thr": base.get("n_val_thr", 0),
        "best_epoch": base.get("best_epoch"),
        "n_features_seq_long": base.get("n_features_seq_long"),
        "n_features_context": base.get("n_features_context"),
        # Metadata del ensemble (informativa)
        "ensemble_n_members": len(members),
    }


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
    # set_random_seed setea numpy, tf y random simultáneamente
    tf.keras.utils.set_random_seed(int(seed))
    if os.environ.get("DETERMINISTIC", "0") == "1":
        # Reforzar por si algún módulo deshizo el seed entre ventanas
        import random as _random
        _random.seed(int(seed))
        np.random.seed(int(seed))

    # Modo sin look-ahead: reservar último mes del train como val_threshold.
    # El threshold scanner se aplicará sobre val_threshold (no test), evitando
    # la optimización post-hoc del threshold con datos futuros.
    no_lookahead = os.environ.get("NO_LOOKAHEAD_SCANNER", "0") == "1"

    # Split por fecha
    if no_lookahead:
        # train: [train_start, train_end - VAL_THR_MONTHS]
        # val_threshold: [train_end - VAL_THR_MONTHS, train_end]
        val_thr_months = int(os.environ.get("VAL_THR_MONTHS", "1"))
        val_thr_start = pd.Timestamp(train_end) - relativedelta(months=val_thr_months)
        df_train = df_prepared.loc[
            (df_prepared["time"] >= pd.Timestamp(train_start)) &
            (df_prepared["time"] <  val_thr_start)
        ].reset_index(drop=True)
        df_val_thr = df_prepared.loc[
            (df_prepared["time"] >= val_thr_start) &
            (df_prepared["time"] <  pd.Timestamp(train_end))
        ].reset_index(drop=True)
    else:
        df_train = df_prepared.loc[
            (df_prepared["time"] >= pd.Timestamp(train_start)) &
            (df_prepared["time"] <  pd.Timestamp(train_end))
        ].reset_index(drop=True)
        df_val_thr = None

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

    # Validation interna (90/10 de train).
    # TradingModel.train espera X como dict keyed por 'seq_short'/'seq_long'/
    # 'context'/'time' (indexa con strings). Construir como list rompía:
    #   TypeError: list indices must be integers or slices, not str
    n = len(X_seq_long)
    n_val = max(int(n * 0.10), 500)
    n_tr  = n - n_val
    X_tr = {
        "seq_short": X_seq_short[:n_tr], "seq_long": X_seq_long[:n_tr],
        "context":   X_context[:n_tr],   "time":     X_time[:n_tr],
    }
    X_va = {
        "seq_short": X_seq_short[n_tr:], "seq_long": X_seq_long[n_tr:],
        "context":   X_context[n_tr:],   "time":     X_time[n_tr:],
    }
    y_tr = y_train[:n_tr]
    y_va = y_train[n_tr:]

    # sample_weight: debe ir alineado con X_tr (n_tr filas), no con N completo.
    _sw_full = pack.get("sample_weight")
    sw_tr = _sw_full[:n_tr] if _sw_full is not None else None

    # Build & train modelo
    #
    # Shapes extraídos una vez fuera del if/else porque build_model_v2/v3 los
    # requieren igual que la rama de arch alternativa. Antes (pre-fix Phase 0)
    # la rama original_v3 llamaba model.build_model_v3() sin args y todos los
    # windows del walkforward CNN-LSTM fallaban con
    # "missing 4 required positional arguments".
    try:
        model = TradingModel(
            general_config=general_config, model_config=model_config, side=None,  # multitask
        )
        shape_short = X_seq_short.shape[1:]   # (seq_short, n_feat_short)
        shape_long  = X_seq_long.shape[1:]
        n_ctx       = int(X_context.shape[1])
        n_time_feat = int(X_time.shape[1])
        init_b      = float(getattr(model_config, "init_bias", 0.0))

        arch = str(getattr(model_config, "_walkforward_arch", "original_v3"))
        if arch == "original_v3":
            if getattr(model_config, "use_hierarchical_fusion", True):
                model.build_model_v3(
                    shape_short=shape_short, shape_long=shape_long,
                    n_context=n_ctx, n_time=n_time_feat, init_bias=init_b,
                )
            else:
                model.build_model_v2(
                    shape_short=shape_short, shape_long=shape_long,
                    n_context=n_ctx, n_time=n_time_feat, init_bias=init_b,
                )
        else:
            # Arquitectura alternativa drop-in (mlp / hybrid / transformer / tcn).
            # Sobrescribimos model.model con el modelo built por la factory.
            from mimo.models.model_alternatives import build_model_by_arch
            model.model = build_model_by_arch(
                arch_name=arch,
                shape_short=shape_short, shape_long=shape_long,
                n_context=n_ctx, n_time=n_time_feat,
                model_config=model_config, init_bias=init_b,
            )
        model.compile_model()
    except Exception as e:
        return {"skipped": True, "reason": f"build_model_failed[{arch}]: {str(e)[:200]}"}

    try:
        history = model.train(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va,   y_val=y_va,
            sample_weight=sw_tr,  # split a n_tr para alinear con X_tr
            verbose=0,
        )
        best_iter = int(history.get("best_epoch", len(history.get("loss", [])) if isinstance(history, dict) else 0))
    except Exception as e:
        import traceback as _tb
        _trace = _tb.format_exc()
        print(f"    ❌ train_failed traceback:\n{_trace}")
        return {"skipped": True, "reason": f"train_failed: {str(e)[:200]}",
                "trace_tail": _trace[-1500:]}

    # ─── Calibración isotónica (opt-in: CALIBRATION=isotonic) ───
    # Entrena IsotonicRegression sobre las probas crudas de X_va (10% del train
    # reservado para early-stop) usando los labels reales. Las probas dejan de
    # ser "scores" y pasan a ser frecuencias relativas calibradas, lo que
    # estabiliza el threshold scanner entre val_internal y test.
    calibrator = os.environ.get("CALIBRATION", "").lower()
    iso_long = iso_short = None
    if calibrator == "isotonic":
        try:
            from sklearn.isotonic import IsotonicRegression
            preds_cal = model.model.predict(
                X_va,
                batch_size=int(model_config.batch_size), verbose=0,
            )
            if isinstance(preds_cal, dict):
                pcal_long  = np.asarray(preds_cal["signal_long"]).reshape(-1).astype(np.float64)
                pcal_short = np.asarray(preds_cal["signal_short"]).reshape(-1).astype(np.float64)
                y_long_cal  = y_va[:, 0].astype(np.float64)
                y_short_cal = y_va[:, 1].astype(np.float64) if y_va.shape[1] >= 2 else None
                if len(pcal_long) >= 200 and y_long_cal.sum() >= 5:
                    iso_long = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pcal_long, y_long_cal)
                if y_short_cal is not None and y_short_cal.sum() >= 5:
                    iso_short = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip").fit(pcal_short, y_short_cal)
                print(f"    🎯 calibration: iso_long={'✓' if iso_long else '✗'} iso_short={'✓' if iso_short else '✗'}")
            else:
                print(f"    ⚠️  calibration: preds_cal tipo inesperado, skip")
        except Exception as e:
            print(f"    ⚠️  calibration_failed: {str(e)[:200]} → fallback sin calibración")

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

    # Predict — directo sobre model.model porque TradingModel.predict hace
    # .ravel() asumiendo single-output, lo cual rompe en multitask
    # (outputs = dict {'signal_long', 'signal_short'}).
    try:
        preds_dict = model.model.predict(
            {"seq_short": Xt_seq_short, "seq_long": Xt_seq_long,
             "context":   Xt_context,   "time":     Xt_time},
            batch_size=int(model_config.batch_size), verbose=0,
        )
        if not isinstance(preds_dict, dict):
            return {"skipped": True,
                    "reason": f"preds tipo inesperado: {type(preds_dict).__name__}"}
        p_long  = np.asarray(preds_dict["signal_long"]).reshape(-1).astype(np.float32)
        p_short = np.asarray(preds_dict["signal_short"]).reshape(-1).astype(np.float32)
        if len(p_long) != len(p_short):
            return {"skipped": True,
                    "reason": f"preds longitud inconsistente: long={len(p_long)} short={len(p_short)}"}
        # Aplicar calibración isotónica si se entrenó
        if iso_long is not None:
            p_long = iso_long.transform(p_long.astype(np.float64)).astype(np.float32)
        if iso_short is not None:
            p_short = iso_short.transform(p_short.astype(np.float64)).astype(np.float32)
    except Exception as e:
        import traceback as _tb
        print(f"    ❌ predict_failed traceback:\n{_tb.format_exc()}")
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
    # ─── Modo sin look-ahead: predecir también sobre val_thr ───
    df_val_eval = None
    if no_lookahead and df_val_thr is not None and len(df_val_thr) >= L + 50:
        try:
            seq_val = pipeline.create_sequences_by_side(
                df_val_thr, sides=("long", "short"),
                fit_scalers=False, train=False,
            )
            pack_va = seq_val.get("long") or {}
            Xv_seq_short = pack_va.get("seq_short"); Xv_seq_long = pack_va.get("seq_long")
            Xv_context   = pack_va.get("context");   Xv_time     = pack_va.get("time")
            if Xv_seq_long is not None and len(Xv_seq_long) > 0:
                preds_val = model.model.predict(
                    {"seq_short": Xv_seq_short, "seq_long": Xv_seq_long,
                     "context":   Xv_context,   "time":     Xv_time},
                    batch_size=int(model_config.batch_size), verbose=0,
                )
                if isinstance(preds_val, dict):
                    pv_long  = np.asarray(preds_val["signal_long"]).reshape(-1).astype(np.float32)
                    pv_short = np.asarray(preds_val["signal_short"]).reshape(-1).astype(np.float32)
                    # Aplicar calibración isotónica si está disponible
                    if iso_long is not None:
                        pv_long = iso_long.transform(pv_long.astype(np.float64)).astype(np.float32)
                    if iso_short is not None:
                        pv_short = iso_short.transform(pv_short.astype(np.float64)).astype(np.float32)
                    df_va_aligned = df_val_thr.iloc[context_offset:context_offset + len(pv_long)].reset_index(drop=True)
                    if len(df_va_aligned) < len(pv_long):
                        pv_long  = pv_long[:len(df_va_aligned)]
                        pv_short = pv_short[:len(df_va_aligned)]
                    df_val_eval = pd.DataFrame({
                        "time": df_va_aligned["time"].values,
                        "high": df_va_aligned["high"].astype(np.float64).values,
                        "low":  df_va_aligned["low"].astype(np.float64).values,
                        "close":df_va_aligned["close"].astype(np.float64).values,
                        "atr":  df_va_aligned["atr"].astype(np.float64).values,
                        "signal_long":  df_va_aligned["signal_long"].astype(np.int8).values
                            if "signal_long" in df_va_aligned.columns else 0,
                        "signal_short": df_va_aligned["signal_short"].astype(np.int8).values
                            if "signal_short" in df_va_aligned.columns else 0,
                        "p_long_raw":  pv_long,
                        "p_short_raw": pv_short,
                    })
        except Exception as e:
            print(f"    ⚠️  predict_val_thr_failed: {str(e)[:200]} → fallback al modo con look-ahead")
            df_val_eval = None

    return {
        "skipped": False,
        "df_eval": df_eval,
        "df_val_eval": df_val_eval,   # None si NO_LOOKAHEAD_SCANNER!=1 o si falló
        "n_train": len(df_train), "n_test": len(df_test),
        "n_val_thr": (len(df_val_thr) if df_val_thr is not None else 0),
        "best_epoch": best_iter,
        "n_features_seq_long": X_seq_long.shape[-1],
        "n_features_context": X_context.shape[-1] if X_context is not None else 0,
    }


def _eval_predictions(
    df_eval: pd.DataFrame, *, horizon: int, tp_mult: float, sl_mult: float,
    cost_per_signal: float, max_drawdown_R: float,
    min_signals: int, use_raw_probs: bool = True,
    df_val_eval: Optional[pd.DataFrame] = None,
    df_eval_long: Optional[pd.DataFrame] = None,
    df_eval_short: Optional[pd.DataFrame] = None,
    df_val_eval_long: Optional[pd.DataFrame] = None,
    df_val_eval_short: Optional[pd.DataFrame] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Threshold scan por side y devuelve (long_res, short_res).

    MODO ESTÁNDAR (df_val_eval=None): scanner sobre df_eval (test). Las
    métricas reportadas son las del best threshold encontrado en test.
    OJO: el threshold se eligió viendo outcomes futuros → look-ahead.

    MODO SIN LOOK-AHEAD (df_val_eval != None): scanner sobre df_val_eval
    (val_internal, último mes del train) para ELEGIR threshold. Luego se
    APLICA ese threshold al df_eval (test) y se calculan métricas reales.
    Adicionalmente, el dict resultante incluye una sub-key `scan_*` con
    las métricas del scanner sobre val (útil para predecir performance).
    """
    if use_raw_probs:
        thr_lo, thr_hi, n_thr = 0.05, 0.95, 180
    else:
        thr_lo, thr_hi, n_thr = 0.05, 0.60, 80

    # Robustez del scanner (opt-in, controlados por env vars):
    #   THR_MIN_PREC: precision mínima exigida en val_internal para considerar
    #                 un threshold candidato. Filtra "mínimos por suerte
    #                 estadística" con pocas señales y prec inflado.
    #   THR_SMOOTH_K: tamaño de kernel (vecinos por lado) para suavizar el
    #                 score sobre el eje de thresholds antes de elegir el mejor.
    # Asimétricos por side (overrides los genéricos): THR_MIN_PREC_LONG,
    # THR_MIN_PREC_SHORT, THR_SMOOTH_K_LONG, THR_SMOOTH_K_SHORT. Útil cuando
    # long y short tienen distinta distribución de probs (e.g. long es escaso
    # y selectivo → sin filtros; short es ruidoso → con filtros).
    def _f(name: str, default: str = "0") -> float:
        try:
            return float(os.environ.get(name, default) or 0)
        except ValueError:
            return 0.0
    def _i(name: str, default: str = "0") -> int:
        try:
            return int(os.environ.get(name, default) or 0)
        except ValueError:
            return 0
    base_min_prec = _f("THR_MIN_PREC")
    base_smooth_k = _i("THR_SMOOTH_K")
    long_min_prec  = _f("THR_MIN_PREC_LONG",  str(base_min_prec))
    short_min_prec = _f("THR_MIN_PREC_SHORT", str(base_min_prec))
    long_smooth_k  = _i("THR_SMOOTH_K_LONG",  str(base_smooth_k))
    short_smooth_k = _i("THR_SMOOTH_K_SHORT", str(base_smooth_k))

    # Scanner asimétrico por side: recortar df_val_eval a los últimos N meses
    # contados desde el final de val (proxy de train_end). Útil cuando el modelo
    # se entrena con VAL_THR_MONTHS grande pero queremos que el threshold scanner
    # use ventanas distintas por side (e.g. long usa 1mo, short usa 3mo).
    # Si no se setean, usa todo el df_val_eval disponible (comportamiento previo).
    val_scan_m_long  = _i("VAL_SCAN_MONTHS_LONG",  "0")
    val_scan_m_short = _i("VAL_SCAN_MONTHS_SHORT", "0")

    def _val_eval_for_side(side_is_long: bool,
                            source: Optional[pd.DataFrame] = None) -> Optional[pd.DataFrame]:
        src = source if source is not None else df_val_eval
        if src is None:
            return None
        m = val_scan_m_long if side_is_long else val_scan_m_short
        if m <= 0:
            return src
        t_end = pd.Timestamp(src["time"].max()) + pd.Timedelta(microseconds=1)
        t_lo = t_end - relativedelta(months=int(m))
        sub = src.loc[src["time"] >= t_lo]
        return sub.reset_index(drop=True) if len(sub) >= 50 else src

    def _one_side(proba_col: str, side_is_long: bool) -> Dict[str, Any]:
        side_min_prec = long_min_prec if side_is_long else short_min_prec
        side_smooth_k = long_smooth_k if side_is_long else short_smooth_k
        # En modo SPLIT_MODELS, df_eval_long y df_eval_short vienen de modelos
        # distintos (uno entrenado para optimizar long, otro para short).
        # Si están seteados, usar el correspondiente al side; si no, usar df_eval.
        cand_eval = df_eval_long if side_is_long else df_eval_short
        side_df_eval = cand_eval if cand_eval is not None else df_eval
        cand_val = df_val_eval_long if side_is_long else df_val_eval_short
        side_df_val_eval_raw = cand_val if cand_val is not None else df_val_eval
        side_val_eval = _val_eval_for_side(side_is_long, side_df_val_eval_raw)
        if side_val_eval is None:
            # Modo estándar: scanner sobre test
            return compute_ev_at_best_threshold(
                side_df_eval, proba_col=proba_col, side_is_long=side_is_long,
                horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
                cost_per_signal=cost_per_signal,
                n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
                min_signals=min_signals, max_drawdown_R=max_drawdown_R,
                min_prec=side_min_prec, smooth_k=side_smooth_k,
            )
        # Modo sin look-ahead: scanner sobre val_internal (side-specific)
        val_min_signals = max(int(min_signals / 6), 10)
        scan_res = compute_ev_at_best_threshold(
            side_val_eval, proba_col=proba_col, side_is_long=side_is_long,
            horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=cost_per_signal,
            n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
            min_signals=val_min_signals, max_drawdown_R=max_drawdown_R,
            min_prec=side_min_prec, smooth_k=side_smooth_k,
        )
        chosen_thr = scan_res.get("thr")
        if chosen_thr is None or (isinstance(chosen_thr, float) and (np.isnan(chosen_thr) or not np.isfinite(chosen_thr))):
            # Scanner no encontró threshold válido en val → no operar en test
            from mimo.oof.ev_objective import _empty_result
            empty = _empty_result(reason="no_valid_thr_in_val")
            empty["scan"] = scan_res
            return empty
        # Aplicar threshold elegido al test (df_eval del side específico)
        from mimo.oof.ev_objective import apply_fixed_threshold
        test_res = apply_fixed_threshold(
            side_df_eval, proba_col=proba_col, side_is_long=side_is_long,
            thr=float(chosen_thr),
            horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=cost_per_signal,
            min_signals=1,  # ya tenemos thr, no exigimos volumen mínimo en test
        )
        # Adjuntar las métricas del scanner para trazabilidad y filtros
        test_res["scan"] = {
            "thr":         scan_res.get("thr"),
            "score":       scan_res.get("score"),
            "ev_net":      scan_res.get("ev_net"),
            "ev_gross":    scan_res.get("ev_gross"),
            "prec_TP":     scan_res.get("prec_TP"),
            "n_signals":   scan_res.get("n_signals"),
            "sig_rate":    scan_res.get("sig_rate"),
        }
        return test_res

    long_res  = _one_side("p_long_raw",  side_is_long=True)
    short_res = _one_side("p_short_raw", side_is_long=False)
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
        elif isinstance(v, dict): out[k] = _json_safe(v)  # recursivo para sub-dicts (ej. 'scan')
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
                    help="Nombre del Optuna study del CNN (ej. oof_study_202500_multitask). "
                         "Usado por defecto para ambos modelos cuando SPLIT_MODELS=1, "
                         "salvo que se especifique --cnn-study-name-long/short por separado.")
    ap.add_argument("--cnn-study-name-long", default=None,
                    help="(Opcional) Study específico para el modelo LONG cuando "
                         "SPLIT_MODELS=1. Si se omite, se usa --cnn-study-name.")
    ap.add_argument("--cnn-study-name-short", default=None,
                    help="(Opcional) Study específico para el modelo SHORT cuando "
                         "SPLIT_MODELS=1. Si se omite, se usa --cnn-study-name.")
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
    ap.add_argument("--arch", default="original_v3",
                    choices=("original_v3", "mlp", "mlp_flatten", "hybrid", "transformer", "tcn"),
                    help="Arquitectura del modelo. original_v3 = CNN-LSTM jerárquico "
                         "del proyecto. mlp/mlp_flatten/hybrid/transformer/tcn = "
                         "alternativas de model_alternatives.py.")

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

    # 1) Best CNN params (study principal — usado en modo no-split o como fallback)
    best_params, best_trial_n = _load_best_params_from_study(
        args.cnn_study_name, args.optuna_storage)
    model_config = _model_config_from_params(
        best_params, target_type="multitask",
        epochs=int(args.epochs), patience=int(args.patience),
    )
    # Pasamos el arch via attribute en model_config (consumido en
    # _train_and_predict_window). No es un campo "oficial" de ModelConfig
    # pero Python no se queja por atributos extra.
    setattr(model_config, "_walkforward_arch", str(args.arch))
    print(f"🏗️  Arquitectura: {args.arch}")

    # 1b) Studies por side (opcionales): si --cnn-study-name-long/short están
    # seteados, cargar best_params separados y construir model_configs distintos.
    # Cada uno se usa en _train_one() según el side. Permite Camino B
    # (Optuna single-side LONG + Optuna single-side SHORT).
    model_config_long  = model_config
    model_config_short = model_config
    best_params_long  = best_params
    best_params_short = best_params
    best_trial_n_long  = best_trial_n
    best_trial_n_short = best_trial_n
    if args.cnn_study_name_long:
        bp_l, bt_l = _load_best_params_from_study(
            args.cnn_study_name_long, args.optuna_storage)
        mc_l = _model_config_from_params(
            bp_l, target_type="multitask",
            epochs=int(args.epochs), patience=int(args.patience),
        )
        setattr(mc_l, "_walkforward_arch", str(args.arch))
        model_config_long = mc_l
        best_params_long  = bp_l
        best_trial_n_long = bt_l
        print(f"🎯 LONG study override: {args.cnn_study_name_long} (trial #{bt_l})")
    if args.cnn_study_name_short:
        bp_s, bt_s = _load_best_params_from_study(
            args.cnn_study_name_short, args.optuna_storage)
        mc_s = _model_config_from_params(
            bp_s, target_type="multitask",
            epochs=int(args.epochs), patience=int(args.patience),
        )
        setattr(mc_s, "_walkforward_arch", str(args.arch))
        model_config_short = mc_s
        best_params_short  = bp_s
        best_trial_n_short = bt_s
        print(f"🎯 SHORT study override: {args.cnn_study_name_short} (trial #{bt_s})")

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

    no_lookahead = os.environ.get("NO_LOOKAHEAD_SCANNER", "0") == "1"
    val_thr_m   = int(os.environ.get("VAL_THR_MONTHS", "1"))
    calibration = os.environ.get("CALIBRATION", "").lower() or "none"
    if no_lookahead:
        print(f"   🔬 MODO SIN LOOK-AHEAD: threshold scanner sobre val_internal "
              f"({val_thr_m}m antes del train_end), no test.")
    else:
        print(f"   ⚠️  MODO ESTÁNDAR: threshold scanner sobre test (look-ahead). "
              f"Para producción usar NO_LOOKAHEAD_SCANNER=1.")
    print(f"   🎯 Calibración: {calibration}")
    thr_min_prec_env = os.environ.get("THR_MIN_PREC", "0")
    thr_smooth_k_env = os.environ.get("THR_SMOOTH_K", "0")
    long_mp = os.environ.get("THR_MIN_PREC_LONG", thr_min_prec_env)
    short_mp = os.environ.get("THR_MIN_PREC_SHORT", thr_min_prec_env)
    long_sk = os.environ.get("THR_SMOOTH_K_LONG", thr_smooth_k_env)
    short_sk = os.environ.get("THR_SMOOTH_K_SHORT", thr_smooth_k_env)
    if (float(long_mp or 0) > 0 or float(short_mp or 0) > 0 or
        int(long_sk or 0) > 0 or int(short_sk or 0) > 0):
        print(f"   🛡️  Scanner robustness: "
              f"long(min_prec={long_mp}, smooth_k={long_sk})  "
              f"short(min_prec={short_mp}, smooth_k={short_sk})")
    if os.environ.get("DETERMINISTIC", "0") == "1":
        try:
            import tensorflow as tf
            tf.config.experimental.enable_op_determinism()
            print(f"   🎯 DETERMINISTIC=1: TF_DETERMINISTIC_OPS, TF_CUDNN_DETERMINISTIC, "
                  f"enable_op_determinism() activos")
        except Exception as e:
            print(f"   ⚠️  enable_op_determinism failed: {str(e)[:120]}")
    vsm_long  = int(os.environ.get("VAL_SCAN_MONTHS_LONG",  "0") or 0)
    vsm_short = int(os.environ.get("VAL_SCAN_MONTHS_SHORT", "0") or 0)
    if vsm_long > 0 or vsm_short > 0:
        print(f"   🪟 Scanner val ventana: "
              f"long={vsm_long or val_thr_m}m  short={vsm_short or val_thr_m}m "
              f"(modelo entrenado con VAL_THR_MONTHS={val_thr_m})")

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

    # 6) prepare_data
    # ─────────────────────────────────────────────────────────────────────
    # STRICT_NO_LEAK=1 (default 0):
    #   · 0 = MODO LEGACY (look-ahead conocido): prepare_data UNA vez sobre el
    #         df completo (train + test + lockbox). Las quantiles que usa
    #         StateDetector (vol_low/high, bb_p20/35/70, rexp_p80) se calculan
    #         con TODAS las filas, incluyendo test futuros → state assignment
    #         y por tanto regime_barriers / labels triple-barrier / sample_weight
    #         por estado tienen leak ~mild (típico 1-5% de filas cambian de
    #         estado). Es el modo más rápido (~30s prepare único).
    #   · 1 = SIN LEAK (recomendado para evaluación honesta): prepare_data PER
    #         WINDOW. Fitea StateDetector solo sobre train slice (padding+train),
    #         extrae thresholds, los inyecta, y luego prepara test slice con
    #         thresholds train-fitted. Coste ~+10-15min (15 ventanas × 2 prepares).
    # ─────────────────────────────────────────────────────────────────────
    strict_no_leak = os.environ.get("STRICT_NO_LEAK", "0") == "1"
    pipeline_master = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=model_config, regime_config=regime_config,
    )
    if not strict_no_leak:
        print("⚙️  prepare_data global (esto puede tardar)...")
        df_prepared = pipeline_master.prepare_data(
            df_rates, labels=True, side="both",
            set_market_condition=False, ensure_regime=True,
        )
        print(f"   {len(df_prepared):,} filas tras prepare_data")
    else:
        print("🔬 STRICT_NO_LEAK=1: prepare_data se hará per-window "
              "(sin leak de quantiles StateDetector; ~+10-15min)")
        df_prepared = None

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    def _reset_state_detector() -> None:
        """Devuelve StateDetector a un estado limpio antes del prepare de cada
        ventana — limpia fixed_* (inyectados de la ventana anterior) y la
        caché dinámica. Necesario para que el path dinámico de _get_thresholds
        refitee con los datos de la ventana actual."""
        sd = pipeline_master.state_detector
        cfg = sd.config
        cfg.fixed_vol_low             = None
        cfg.fixed_vol_high            = None
        cfg.fixed_bb_width_p20        = None
        cfg.fixed_bb_width_p35        = None
        cfg.fixed_bb_width_p70        = None
        cfg.fixed_range_expansion_p80 = None
        sd._cache_key         = None
        sd._cached_thresholds = None
        # Re-armar el warning una vez por ventana (informativo, no error):
        # se silencia tras la primera ventana porque ya se entiende el patrón.
        sd._dynamic_warned = True

    def _prepare_window_strict(
        ts: datetime, te: datetime, vs: datetime, ve: datetime,
        padding_months: int = 2,
    ) -> pd.DataFrame:
        """Prepara datos para una ventana sin leak temporal.

        Flujo:
          1) Slice [ts - padding, te) → prepare_data → fit thresholds en train.
          2) Extraer thresholds del state_detector y inject_thresholds (fija).
          3) Slice [ts - padding, ve) → prepare_data con thresholds inyectados.
             Esto asegura que las features rodantes del test tengan lookback
             válido en train PERO las quantiles de estado son las del train.
          4) Concatenar train_prep [ts,te) + test_prep [vs,ve).
        """
        ts_padded = pd.Timestamp(ts) - relativedelta(months=padding_months)
        # Paso 1: padding + train, StateDetector libre para fitear
        _reset_state_detector()
        mask_train = (
            (df_rates["time"] >= ts_padded) &
            (df_rates["time"] <  pd.Timestamp(te))
        )
        train_rates = df_rates.loc[mask_train].reset_index(drop=True)
        df_train_full = pipeline_master.prepare_data(
            train_rates, labels=True, side="both",
            set_market_condition=False, ensure_regime=True,
        )
        df_train_prep = df_train_full.loc[
            df_train_full["time"] >= pd.Timestamp(ts)
        ].reset_index(drop=True).copy()
        # Paso 2: extraer + inyectar thresholds
        thresholds = pipeline_master.state_detector.compute_thresholds(df_train_full)
        pipeline_master.state_detector.inject_thresholds(thresholds)
        # Paso 3: padding + train + test, StateDetector con thresholds fijos
        mask_full = (
            (df_rates["time"] >= ts_padded) &
            (df_rates["time"] <  pd.Timestamp(ve))
        )
        full_rates = df_rates.loc[mask_full].reset_index(drop=True)
        df_full_prep = pipeline_master.prepare_data(
            full_rates, labels=True, side="both",
            set_market_condition=False, ensure_regime=True,
        )
        df_test_prep = df_full_prep.loc[
            df_full_prep["time"] >= pd.Timestamp(vs)
        ].reset_index(drop=True).copy()
        # Paso 4: concat para downstream (las slice de _train_and_predict_window
        # son por tiempo, así que da igual el orden de concat).
        return pd.concat([df_train_prep, df_test_prep], ignore_index=True)

    # 7) Walk-forward loop
    print(f"\n🔄 Walkforward CNN refit por ventana | epochs={args.epochs} "
          f"patience={args.patience}")

    # SPLIT_MODELS=1: entrenar dos modelos por ventana (uno para long, otro
    # para short), cada uno con su VAL_THR_MONTHS_LONG/_SHORT. Permite tener
    # long con val=1mo (modelo entrenado con 12mo de train) y short con val=3mo
    # (modelo entrenado con 11mo + filtros), simultáneamente.
    split_models = os.environ.get("SPLIT_MODELS", "0") == "1"
    val_thr_m_long  = int(os.environ.get("VAL_THR_MONTHS_LONG",  str(val_thr_m)) or val_thr_m)
    val_thr_m_short = int(os.environ.get("VAL_THR_MONTHS_SHORT", str(val_thr_m)) or val_thr_m)
    # SPLIT_ZERO_OTHER_LOSS=1 hace que cada modelo de SPLIT_MODELS entrene
    # SOLO su side: el modelo LONG pone loss_weight_short=0 (la head SHORT
    # existe pero no contribuye al loss → gradient solo del LONG, backbone
    # se especializa). Equivalente operacional a single-output sin tocar la
    # arquitectura. Conserva los HPs del trial multitask actual.
    split_zero_other = os.environ.get("SPLIT_ZERO_OTHER_LOSS", "0") == "1"
    if split_models:
        print(f"   🪞 SPLIT_MODELS=1: 2 modelos por ventana "
              f"(LONG con VAL_THR={val_thr_m_long}m, SHORT con VAL_THR={val_thr_m_short}m)")
        if split_zero_other:
            print(f"   🎯 SPLIT_ZERO_OTHER_LOSS=1: cada modelo entrena single-side "
                  f"(loss_weight=0 en la head opuesta)")
    _ens_seeds_env = os.environ.get("LONG_ENSEMBLE_SEEDS", "").strip()
    if _ens_seeds_env:
        print(f"   🎲 LONG_ENSEMBLE_SEEDS='{_ens_seeds_env}': "
              f"se entrenarán N modelos LONG por ventana y se promediará "
              f"p_long_raw (test + val_internal) antes del threshold scanner. "
              f"SHORT se queda como modelo único (seed={args.seed}).")

    def _train_one(side_label: str, vthm: int, df_w: pd.DataFrame) -> Dict[str, Any]:
        prev = os.environ.get("VAL_THR_MONTHS")
        os.environ["VAL_THR_MONTHS"] = str(vthm)
        # Selecciona model_config específico del side (Camino B: studies
        # separados). Si no hay overrides, ambos apuntan al mismo objeto.
        mc_active = model_config_long if side_label == "long" else model_config_short
        prev_lw_long  = float(getattr(mc_active, "loss_weight_long",  1.0))
        prev_lw_short = float(getattr(mc_active, "loss_weight_short", 1.0))
        if split_zero_other:
            if side_label == "long":
                mc_active.loss_weight_long  = prev_lw_long if prev_lw_long > 0 else 1.0
                mc_active.loss_weight_short = 0.0
            elif side_label == "short":
                mc_active.loss_weight_long  = 0.0
                mc_active.loss_weight_short = prev_lw_short if prev_lw_short > 0 else 1.0

        # Ensemble path para LONG: si LONG_ENSEMBLE_SEEDS está definido (lista
        # CSV de seeds), entrenar N modelos LONG con seeds distintos y promediar
        # p_long_raw en test + val_internal antes del threshold scanner.
        # Motivo: TCN v4 LONG mostró varianza alta entre seeds (seed=47: +64R,
        # seed=42: -17R con misma config) — promedio reduce la inestabilidad
        # estructural por inicialización.
        # SHORT no entra en ensemble (ya estable en 3 runs: +16, +22, +23R).
        ens_seeds: List[int] = []
        if side_label == "long":
            raw_env = os.environ.get("LONG_ENSEMBLE_SEEDS", "").strip()
            if raw_env:
                try:
                    ens_seeds = [int(s.strip()) for s in raw_env.split(",")
                                 if s.strip()]
                except ValueError:
                    print(f"   ⚠️  LONG_ENSEMBLE_SEEDS inválido: '{raw_env}', "
                          f"se ignora; fallback a single-seed {args.seed}")
                    ens_seeds = []

        try:
            if ens_seeds:
                members: List[Dict[str, Any]] = []
                for sd in ens_seeds:
                    m_out = _train_and_predict_window(
                        df_prepared=df_w,
                        train_start=ts, train_end=te, test_start=vs, test_end=ve,
                        general_config=general_config, model_config=mc_active,
                        feature_config=feature_config, regime_config=regime_config,
                        seed=int(sd),
                    )
                    if m_out.get("skipped"):
                        print(f"      ⚠️  ensemble member seed={sd} skipped: "
                              f"{m_out.get('reason')}")
                        continue
                    members.append(m_out)
                if not members:
                    out = {"skipped": True,
                           "reason": "all_ensemble_members_skipped"}
                else:
                    out = _average_ensemble_long(members)
                    print(f"   🎯 LONG ensemble: {len(members)}/{len(ens_seeds)} "
                          f"miembros promediados (seeds={ens_seeds})")
            else:
                out = _train_and_predict_window(
                    df_prepared=df_w,
                    train_start=ts, train_end=te, test_start=vs, test_end=ve,
                    general_config=general_config, model_config=mc_active,
                    feature_config=feature_config, regime_config=regime_config,
                    seed=int(args.seed),
                )
        finally:
            if prev is None:
                os.environ.pop("VAL_THR_MONTHS", None)
            else:
                os.environ["VAL_THR_MONTHS"] = prev
            if split_zero_other:
                mc_active.loss_weight_long  = prev_lw_long
                mc_active.loss_weight_short = prev_lw_short
        return out

    results: List[Dict[str, Any]] = []
    for i, (ts, te, vs, ve) in enumerate(windows):
        t0 = time.time()
        print(f"\n── Window {i+1}/{len(windows)} | train={ts.date()}→{te.date()} "
              f"test={vs.date()}→{ve.date()}")
        tr_out_long = tr_out_short = None
        try:
            # Selecciona df de la ventana: global slice (modo legacy) o
            # prepare per-window con threshold injection (STRICT_NO_LEAK=1).
            if strict_no_leak:
                t_prep = time.time()
                df_w = _prepare_window_strict(ts, te, vs, ve)
                print(f"   🔬 prepare per-window: {len(df_w):,} filas "
                      f"({time.time()-t_prep:.0f}s)")
            else:
                df_w = df_prepared
            if split_models:
                print(f"   ⚙️  modelo LONG (val={val_thr_m_long}m)...")
                tr_out_long = _train_one("long", val_thr_m_long, df_w)
                if not tr_out_long.get("skipped"):
                    print(f"   ⚙️  modelo SHORT (val={val_thr_m_short}m)...")
                    tr_out_short = _train_one("short", val_thr_m_short, df_w)
                # Si LONG falló, usar SHORT como representante y viceversa
                tr_out = tr_out_long if not tr_out_long.get("skipped") \
                    else (tr_out_short or tr_out_long)
            else:
                tr_out = _train_and_predict_window(
                    df_prepared=df_w,
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

        df_eval     = tr_out["df_eval"]
        df_val_eval = tr_out.get("df_val_eval")
        if split_models and tr_out_long is not None and tr_out_short is not None \
           and not tr_out_long.get("skipped") and not tr_out_short.get("skipped"):
            long_res, short_res = _eval_predictions(
                df_eval, horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
                cost_per_signal=args.cost_per_signal,
                max_drawdown_R=args.max_drawdown_R,
                min_signals=args.min_signals_window,
                use_raw_probs=True,
                df_eval_long=tr_out_long["df_eval"],
                df_eval_short=tr_out_short["df_eval"],
                df_val_eval_long=tr_out_long.get("df_val_eval"),
                df_val_eval_short=tr_out_short.get("df_val_eval"),
            )
        else:
            long_res, short_res = _eval_predictions(
                df_eval, horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
                cost_per_signal=args.cost_per_signal,
                max_drawdown_R=args.max_drawdown_R,
                min_signals=args.min_signals_window,
                use_raw_probs=True,   # raw probs + threshold scan
                df_val_eval=df_val_eval,  # None salvo NO_LOOKAHEAD_SCANNER=1
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

        win_entry: Dict[str, Any] = {
            "window": [str(ts.date()), str(te.date()),
                       str(vs.date()), str(ve.date())],
            "n_train": tr_out["n_train"], "n_test": tr_out["n_test"],
            "n_val_thr": tr_out.get("n_val_thr", 0),
            "best_epoch": tr_out.get("best_epoch"),
            "skipped": False,
            "long":  _json_safe(long_res),
            "short": _json_safe(short_res),
            "no_lookahead": (df_val_eval is not None) or split_models,
        }
        if split_models and tr_out_long is not None and tr_out_short is not None \
           and not tr_out_long.get("skipped") and not tr_out_short.get("skipped"):
            win_entry["split_models"] = {
                "long":  {"n_train": tr_out_long["n_train"],
                          "n_val_thr": tr_out_long.get("n_val_thr", 0),
                          "best_epoch": tr_out_long.get("best_epoch")},
                "short": {"n_train": tr_out_short["n_train"],
                          "n_val_thr": tr_out_short.get("n_val_thr", 0),
                          "best_epoch": tr_out_short.get("best_epoch")},
            }
        results.append(win_entry)

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
        "cnn_study_name_long":  args.cnn_study_name_long,
        "cnn_study_name_short": args.cnn_study_name_short,
        "cnn_best_trial_long":  best_trial_n_long if args.cnn_study_name_long  else None,
        "cnn_best_trial_short": best_trial_n_short if args.cnn_study_name_short else None,
        "cnn_best_params_long":  best_params_long  if args.cnn_study_name_long  else None,
        "cnn_best_params_short": best_params_short if args.cnn_study_name_short else None,
        "mode": "raw_probs",
        "walk_config": {
            "walk_from": str(args.walk_from.date()),
            "walk_to":   str(args.walk_to.date()),
            "train_months": args.train_months,
            "test_months":  args.test_months,
            "step_months":  args.step_months,
        },
        "arch": str(args.arch),
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
