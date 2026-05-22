"""
main_oof_arch_tuning.py
═══════════════════════════════════════════════════════════════════════════
Optuna tuning específico para arquitecturas alternativas (mlp, hybrid,
transformer, tcn). Es la pieza COMPLEMENTARIA a main_oof_cnn_walkforward.py:

  · CNN-LSTM original tiene su propio study (oof_study_<RELEASE>_multitask).
  · Para evitar comparar archs alternativas con hps optimizados para CNN-LSTM
    (lo que sesgaría a favor del original), este script tunea cada arch
    sobre SU PROPIO espacio de hps.

ESTRATEGIA:
  · Un único split train/val (NO walkforward) → coste ~3-6h por arch.
  · ~30-50 trials.
  · Objective: avg(val_signal_long_auc_pr, val_signal_short_auc_pr).
    Ranking-based, estable, sin necesidad de simular signals.

USO:
  python3 -m mimo.oof.main_oof_arch_tuning --arch mlp --n-trials 30
  # genera study oof_study_202500_mlp_multitask
  # luego:
  CNN_STUDY=oof_study_202500_mlp_multitask ARCH=mlp bash 010_walkforward_cnn.sh

SCHEMA STUDY NAME: oof_study_<RELEASE>_<arch>_multitask
  · Coherente con la convención existente del proyecto.
  · El walkforward los carga con --cnn-study-name automáticamente.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Dict, Any, Tuple, Union

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler, NSGAIISampler
from dateutil.relativedelta import relativedelta

# Reutilizamos toda la infra de carga de datos del walkforward existente
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig, TradingModel
from mimo.models.model_alternatives import build_model_by_arch
from mimo.states_manager.state_detector import StateConfig
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.oof.main_oof_cnn_walkforward import (
    _get_barriers_for_release, _get_feature_masks_for_release,
    _tf_defaults, _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
)
from mimo.oof.main_oof_regime_weights_v7 import (
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)


# ─── Espacio de búsqueda por arch ───────────────────────────────────────

def _suggest_hp(trial: optuna.Trial, arch: str, side: str = "both") -> Dict[str, Any]:
    """Define HP space específico por arch. Comunes para todos (loss,
    optimizer, regularización) + específicos según arch (filters, conv, etc.)."""

    # mlp_flatten ve un input ~20x mayor que mlp_tabular (~3600 vs 180 dim)
    # por aplanar las secuencias en lugar de colapsarlas a stats.
    #
    # Espacio v2 (post run 30-trial inicial que dio best_value=0.144 con
    # l2_reg, batch_size, head_units todos en el suelo del rango y
    # focal_alpha_short cerca del techo): ampliamos los suelos y añadimos
    # HPs nuevos para dar más libertad al TPE.
    #   · l2_reg: suelo 1e-7 (vs 1e-5) — el TPE pidió aún menos regularización.
    #   · batch_size: añadir {64, 128}; quitar {2048, 4096} (nunca elegidos).
    #   · head_units: quitar 64 (nunca elegido); mantener {128, 256, 512}.
    #   · focal_alpha_short: ampliar techo a 0.70.
    #   · activation: NUEVO — categorical {gelu, relu, swish, elu}.
    #   · loss_weight_short: NUEVO — float [0.5, 2.0]. loss_weight_long ancla a 1.0.
    if arch == "mlp_flatten":
        hp = {
            "l2_reg":             trial.suggest_float("l2_reg", 1e-7, 1e-3, log=True),
            "dropout_dense":      trial.suggest_float("dropout_dense", 0.20, 0.60),
            "learning_rate":      trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
            "batch_size":         trial.suggest_categorical("batch_size",
                                                            [64, 128, 256, 512, 1024]),
            "head_units":         trial.suggest_categorical("head_units", [128, 256, 512]),
            "focal_alpha_long":   trial.suggest_float("focal_alpha_long", 0.20, 0.60),
            "focal_alpha_short":  trial.suggest_float("focal_alpha_short", 0.20, 0.70),
            "focal_gamma":        trial.suggest_float("focal_gamma", 0.5, 3.0),
            "activation":         trial.suggest_categorical("activation",
                                                            ["gelu", "relu", "swish", "elu"]),
            "loss_weight_short":  trial.suggest_float("loss_weight_short", 0.5, 2.0),
        }
        return hp

    # TCN v4: refinamiento sobre v3 (study oof_study_202500_tcn_v3_multitask, 35
    # trials, best_value=0.144 multitask + best_value=0.16 short_only). Hallazgos:
    #   l2_reg            → 1.7e-7 (suelo v3 1e-7)        → v4 [1e-8, 5e-4]
    #   kernel_size       → 3 (suelo)                     → v4 [2, 3, 5]
    #   n_tcn_blocks_long → 3 (suelo v3 3)                → v4 (2, 5)
    #   conv1d_filters    → 48 (suelo bajo de v3 32..256) → v4 [16..96]
    #   head_units        → 256 (centro)                  → v4 [64, 128, 192, 256]
    #   learning_rate     → 2.3e-3 (centro)               → v4 (5e-4, 5e-3)
    #   tcn_short_only eligió ksize=7, n_blocks=4, filt=128 (todos techos):
    #     LONG y SHORT prefieren receptive fields distintos → v4 prioriza
    #     SIDE=long y SIDE=short por separado (studies independientes).
    #
    # HPs nuevos en v4 (antes hard-coded o derivados):
    #   · n_tcn_blocks_short        (antes derivado de n_long)
    #   · tcn_filters_short_ratio   (antes 0.5 hard-coded en builder)
    #   · ctx_dense_units           (antes 64 hard-coded en builder)
    #   · tcn_pooling añade "attention" (attention pooling aprendible)
    #   · use_se + se_ratio         (squeeze-and-excite por bloque TCN)
    #
    # IMPORTANTE: estudio nuevo (espacio no es superconjunto del v3). Usar:
    #   SIDE=long  STUDY_NAME=oof_study_202500_tcn_v4_long_only \
    #     bash 012_optuna_arch.sh tcn 40
    #   SIDE=short STUDY_NAME=oof_study_202500_tcn_v4_short_only \
    #     bash 012_optuna_arch.sh tcn 40
    if arch == "tcn":
        hp = {
            "l2_reg":             trial.suggest_float("l2_reg", 1e-8, 5e-4, log=True),
            "dropout_dense":      trial.suggest_float("dropout_dense", 0.05, 0.40),
            "dropout_seq":        trial.suggest_float("dropout_seq", 0.03, 0.30),
            "learning_rate":      trial.suggest_float("learning_rate", 5e-4, 5e-3, log=True),
            "batch_size":         trial.suggest_categorical("batch_size",
                                                            [128, 256, 512, 1024]),
            "head_units":         trial.suggest_categorical("head_units",
                                                            [64, 128, 192, 256]),
            "focal_gamma":        trial.suggest_float("focal_gamma", 0.10, 2.5),
            "activation":         trial.suggest_categorical("activation",
                                                            ["gelu", "relu", "swish", "elu"]),
            "conv1d_filters":     trial.suggest_categorical("conv1d_filters",
                                                            [16, 24, 32, 48, 64, 96]),
            "kernel_size":        trial.suggest_categorical("kernel_size", [2, 3, 5]),
            "n_tcn_blocks_long":  trial.suggest_int("n_tcn_blocks_long", 2, 5),
            "n_tcn_blocks_short": trial.suggest_int("n_tcn_blocks_short", 2, 4),
            "tcn_filters_short_ratio": trial.suggest_categorical(
                "tcn_filters_short_ratio", [0.5, 0.75, 1.0]),
            "ctx_dense_units":    trial.suggest_categorical("ctx_dense_units",
                                                            [32, 64, 128]),
            "tcn_pooling":        trial.suggest_categorical(
                "tcn_pooling", ["gap", "gmp", "gap_gmp", "attention"]),
            "use_se":             trial.suggest_categorical("use_se", [0, 1]),
        }
        # se_ratio condicional — solo si use_se=1, evita explorar zona muerta.
        if hp["use_se"]:
            hp["se_ratio"] = trial.suggest_categorical("se_ratio", [4, 8, 16])
        # Side-specific HPs:
        #  · both  → multitask: ambos focal_alpha + loss_weight_short libre.
        #  · long  → solo focal_alpha_long; loss_weight_short=0 (head SHORT no entrena).
        #  · short → solo focal_alpha_short; loss_weight_long=0 (head LONG no entrena).
        if side in ("both", "long"):
            hp["focal_alpha_long"] = trial.suggest_float("focal_alpha_long", 0.30, 0.80)
        if side in ("both", "short"):
            hp["focal_alpha_short"] = trial.suggest_float("focal_alpha_short", 0.30, 0.85)
        if side == "both":
            hp["loss_weight_short"] = trial.suggest_float("loss_weight_short", 0.5, 2.5)
        elif side == "long":
            hp["loss_weight_long"]  = 1.0
            hp["loss_weight_short"] = 0.0
        elif side == "short":
            hp["loss_weight_long"]  = 0.0
            hp["loss_weight_short"] = 1.0
        return hp

    hp = {
        # Optimizer + regularización (todos los archs)
        "l2_reg":             trial.suggest_float("l2_reg", 1e-6, 1e-2, log=True),
        "dropout_dense":      trial.suggest_float("dropout_dense", 0.10, 0.50),
        "learning_rate":      trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "batch_size":         trial.suggest_categorical("batch_size",
                                                        [256, 512, 1024, 2048, 4096]),
        # Capacidad de la cabeza/MLP
        "head_units":         trial.suggest_categorical("head_units", [32, 64, 128, 256]),
        # Loss (multitask focal)
        "focal_alpha_long":   trial.suggest_float("focal_alpha_long", 0.20, 0.60),
        "focal_alpha_short":  trial.suggest_float("focal_alpha_short", 0.20, 0.60),
        "focal_gamma":        trial.suggest_float("focal_gamma", 0.5, 3.0),
    }

    # Específicos por arch
    if arch in ("hybrid", "tcn", "transformer"):
        hp["dropout_seq"] = trial.suggest_float("dropout_seq", 0.05, 0.30)
    if arch == "hybrid":
        hp["conv1d_filters"] = trial.suggest_categorical(
            "conv1d_filters", [16, 32, 48, 64])
        hp["context_units"]  = trial.suggest_categorical(
            "context_units", [32, 64, 128])
    if arch == "tcn":
        hp["conv1d_filters"] = trial.suggest_categorical(
            "conv1d_filters", [32, 48, 64, 96])
    # Transformer: d_model/heads se infieren del shape; no son hp explícitos
    # para mantener el espacio compacto.

    return hp


def _build_model_config(hp: Dict[str, Any], epochs: int, patience: int,
                        seq_len_short: int = 24, seq_len_long: int = 96) -> ModelConfig:
    """ModelConfig a partir del trial dict. Filtra a campos válidos del
    dataclass (igual patrón que main_oof_cnn_walkforward._model_config_from_params)."""
    import dataclasses
    valid_fields = {f.name for f in dataclasses.fields(ModelConfig)}

    # focal_alpha único = promedio de long+short (ModelConfig solo tiene 1 campo)
    fa_l = float(hp.get("focal_alpha_long", 0.30))
    fa_s = float(hp.get("focal_alpha_short", 0.30))

    candidate = {
        "seq_len_short": seq_len_short,
        "seq_len_long":  seq_len_long,
        "target_type":   "multitask",
        "epochs":        int(epochs),
        "patience":      int(patience),
        "focal_alpha":   (fa_l + fa_s) / 2.0,
        "use_hierarchical_fusion": False,  # no aplica a archs alternativas
        # Pasamos el resto de hp; los no soportados se filtran abajo
        **{k: v for k, v in hp.items()
           if k not in ("focal_alpha_long", "focal_alpha_short")},
    }
    init_kwargs = {k: v for k, v in candidate.items() if k in valid_fields}
    mc = ModelConfig(**init_kwargs)
    # Adjunto _long/_short como atributos por si compile_model los lee
    setattr(mc, "focal_alpha_long",  fa_l)
    setattr(mc, "focal_alpha_short", fa_s)
    # HPs extra que NO están en el dataclass pero los lee algún builder via
    # getattr (p.ej. TCN v3: kernel_size, n_tcn_blocks_long, tcn_pooling;
    # TCN v4 añade: n_tcn_blocks_short, tcn_filters_short_ratio,
    # ctx_dense_units, use_se, se_ratio).
    EXTRA_HP_KEYS = (
        "kernel_size", "n_tcn_blocks_long", "tcn_pooling",
        "n_tcn_blocks_short", "tcn_filters_short_ratio",
        "ctx_dense_units", "use_se", "se_ratio",
    )
    for k in EXTRA_HP_KEYS:
        if k in candidate:
            setattr(mc, k, candidate[k])
    return mc


def _eval_ev_score(
    model, X_va, y_va, side: str,
    tp_mult: float, sl_mult: float, cost_per_signal: float,
    min_signals: int,
) -> Tuple[float, float]:
    """Threshold sweep en val para estimar max EV_net por side, alineado con la
    métrica del walkforward final (en R, con triple-barrier tp_mult/sl_mult).

    Para cada side activo:
      ev_net(t) = precision(t) * tp_mult - (1 - precision(t)) * sl_mult - cost
      sujeto a n_signals(t) >= min_signals. Si no hay threshold con suficientes
      señales → -1.0 R (penaliza configs ranking-buenas pero sin volumen útil).

    Devuelve (ev_long, ev_short); side no activo → 0.0 (no entra en el score)."""
    try:
        preds = model.model.predict(X_va, verbose=0)
    except Exception as e:
        print(f"      ❌ predict failed en EV sweep: {e}")
        return -1.0, -1.0

    if isinstance(preds, dict):
        p_long  = np.asarray(preds.get("signal_long")).ravel()
        p_short = np.asarray(preds.get("signal_short")).ravel()
    elif isinstance(preds, (list, tuple)) and len(preds) >= 2:
        p_long  = np.asarray(preds[0]).ravel()
        p_short = np.asarray(preds[1]).ravel()
    else:
        arr = np.asarray(preds)
        p_long  = arr[:, 0] if arr.ndim == 2 else arr.ravel()
        p_short = arr[:, 1] if arr.ndim == 2 and arr.shape[1] >= 2 else p_long

    y_long  = y_va[:, 0].astype(float)
    y_short = y_va[:, 1].astype(float)

    def _best_ev(p: np.ndarray, y: np.ndarray) -> float:
        if p.size == 0 or p.size != y.size:
            return -1.0
        lo, hi = np.quantile(p, [0.10, 0.99])
        if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
            return -1.0
        thrs = np.linspace(lo, hi, 41)
        best = -1.0
        for t in thrs:
            mask = p > t
            n = int(mask.sum())
            if n < min_signals:
                continue
            prec = float(y[mask].mean())
            ev = prec * tp_mult - (1.0 - prec) * sl_mult - cost_per_signal
            if ev > best:
                best = ev
        return best

    ev_l = _best_ev(p_long, y_long)  if side in ("both", "long")  else 0.0
    ev_s = _best_ev(p_short, y_short) if side in ("both", "short") else 0.0
    return ev_l, ev_s


def _train_and_eval(
    arch: str, hp: Dict[str, Any],
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    general_config: Config, feature_config: FeatureConfig,
    regime_config: StateConfig,
    epochs: int, patience: int, seed: int,
    side: str = "both",
    objective: str = "auc_pr",
    multi_obj: bool = False,
    ev_tp_mult: float = 2.0, ev_sl_mult: float = 0.8,
    ev_cost: float = 0.05, ev_min_signals: int = 30,
) -> Union[float, Tuple[float, float]]:
    """Entrena modelo con hp del trial sobre df_train, evalúa sobre df_val.

    objective="auc_pr" → score = max val_signal_<side>_auc_pr (ranking puro).
    objective="ev_net" → score = max EV_net via threshold sweep en val,
                          alineado con la métrica final del walkforward.

    side="both"  multi_obj=False → media de ambos sides.
    side="both"  multi_obj=True  → tupla (long, short) para Pareto front.
    side="long"  → solo LONG.
    side="short" → solo SHORT.

    Si algo falla, devuelve -1.0 (o (-1.0, -1.0) en multi_obj) y emite traceback."""
    import tensorflow as tf
    import traceback as _tb
    tf.keras.utils.set_random_seed(int(seed))

    mc = _build_model_config(hp, epochs=epochs, patience=patience)

    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=mc, regime_config=regime_config,
    )

    # Sequences train
    try:
        seq_train = pipeline.create_sequences_by_side(
            df_train, sides=("long", "short"), fit_scalers=True, train=True,
        )
    except Exception as e:
        print(f"      ❌ create_sequences_train failed: {e}")
        print(_tb.format_exc())
        return -1.0
    pack_tr = seq_train.get("long") or {}
    X_tr = {k: pack_tr.get(k) for k in ("seq_short", "seq_long", "context", "time")}
    y_tr = pack_tr.get("labels")
    if y_tr is None or y_tr.ndim != 2:
        print(f"      ❌ y_tr inválido: "
              f"{'None' if y_tr is None else f'shape={y_tr.shape} ndim={y_tr.ndim}'}")
        print(f"         pack_tr keys: {list(pack_tr.keys())}")
        for k, v in X_tr.items():
            print(f"         X_tr[{k!r}] shape: "
                  f"{None if v is None else v.shape}")
        return -1.0

    # Sequences val
    # IMPORTANTE: train=True para que extraiga labels (train=False → labels=None
    # porque el pipeline asume modo inferencia). fit_scalers=False mantiene
    # los scalers ya fitted en el paso anterior.
    try:
        seq_val = pipeline.create_sequences_by_side(
            df_val, sides=("long", "short"), fit_scalers=False, train=True,
        )
    except Exception as e:
        print(f"      ❌ create_sequences_val failed: {e}")
        print(_tb.format_exc())
        return -1.0
    pack_va = seq_val.get("long") or {}
    X_va = {k: pack_va.get(k) for k in ("seq_short", "seq_long", "context", "time")}
    y_va = pack_va.get("labels")
    if y_va is None or y_va.ndim != 2:
        print(f"      ❌ y_va inválido: "
              f"{'None' if y_va is None else f'shape={y_va.shape}'}")
        return -1.0

    # init_bias = logit del prior (clase positiva por side)
    pr_long  = float(np.clip(y_tr[:, 0].mean(), 1e-4, 1 - 1e-4))
    pr_short = float(np.clip(y_tr[:, 1].mean(), 1e-4, 1 - 1e-4))
    init_bias = {
        "long":  float(np.log(pr_long  / (1 - pr_long))),
        "short": float(np.log(pr_short / (1 - pr_short))),
    }
    print(f"      n_train={len(y_tr)} n_val={len(y_va)} "
          f"pr_long={pr_long:.4f} pr_short={pr_short:.4f}")

    # Build modelo alternativo
    shape_short = X_tr["seq_short"].shape[1:]
    shape_long  = X_tr["seq_long"].shape[1:]
    n_ctx       = int(X_tr["context"].shape[1])
    n_time      = int(X_tr["time"].shape[1])

    try:
        model = TradingModel(general_config=general_config, model_config=mc, side=None)
        model.model = build_model_by_arch(
            arch, shape_short, shape_long, n_ctx, n_time,
            mc, init_bias=init_bias,
        )
        model.compile_model()
    except Exception as e:
        print(f"      ❌ build_or_compile failed: {e}")
        print(_tb.format_exc())
        return -1.0

    try:
        history = model.train(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            sample_weight=pack_tr.get("sample_weight"),
            verbose=0,
        )
    except Exception as e:
        print(f"      ❌ train failed: {e}")
        print(_tb.format_exc())
        return -1.0

    # Score por side según objective:
    #   auc_pr → max(history[val_signal_<side>_auc_pr]) (ranking puro)
    #   ev_net → max EV_net via threshold sweep en val (alineado a walkforward)
    fail = (-1.0, -1.0) if (multi_obj and side == "both") else -1.0

    if objective == "ev_net":
        ev_l, ev_s = _eval_ev_score(
            model, X_va, y_va, side,
            tp_mult=float(ev_tp_mult), sl_mult=float(ev_sl_mult),
            cost_per_signal=float(ev_cost), min_signals=int(ev_min_signals),
        )
        score_long, score_short = float(ev_l), float(ev_s)
        print(f"      ev_long={score_long:+.4f}  ev_short={score_short:+.4f}")
    else:
        hk_long  = history.get("val_signal_long_auc_pr",  [])
        hk_short = history.get("val_signal_short_auc_pr", [])
        if side in ("both", "long") and not hk_long:
            print(f"      ❌ val_signal_long_auc_pr vacío. "
                  f"history keys: {list(history.keys())}")
            return fail
        if side in ("both", "short") and not hk_short:
            print(f"      ❌ val_signal_short_auc_pr vacío. "
                  f"history keys: {list(history.keys())}")
            return fail
        score_long  = float(max(hk_long))  if hk_long  else 0.0
        score_short = float(max(hk_short)) if hk_short else 0.0

    if side == "long":
        return score_long
    if side == "short":
        return score_short
    # side == "both"
    if multi_obj:
        return (score_long, score_short)
    return float((score_long + score_short) / 2.0)


# ─── Main entrypoint ────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Optuna tuning para arquitecturas alternativas drop-in.")
    ap.add_argument("--arch", required=True,
                    choices=("mlp", "mlp_flatten", "hybrid", "transformer", "tcn"),
                    help="Arquitectura a tunear.")
    ap.add_argument("--release", default="202500")
    ap.add_argument("--side", default="both",
                    choices=("both", "long", "short"),
                    help="Side a optimizar. 'both' (default) = multitask "
                         "(ambos focal_alpha + loss_weight_short, objective "
                         "promedia ambos val_auc_pr). 'long' = single-side "
                         "LONG (focal_alpha_short y loss_weight_short fijos "
                         "a 0 → head SHORT no entrena, objective solo LONG). "
                         "'short' = simétrico al anterior pero para SHORT. "
                         "Camino B: lanzar dos studies separados con --side=long "
                         "y --side=short.")
    ap.add_argument("--study-name", default=None,
                    help="Default: oof_study_<RELEASE>_<arch>_multitask para "
                         "side=both, oof_study_<RELEASE>_<arch>_<side>_only "
                         "para side=long|short.")
    ap.add_argument("--n-trials", type=int, default=30,
                    help="Número de trials Optuna. 30 razonable para HP space ~10-dim.")
    ap.add_argument("--epochs", type=int, default=15,
                    help="Epochs por trial (acortado vs producción para velocidad).")
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--variant-long",  default="vol_boost_td_down")
    ap.add_argument("--variant-short", default="vol_boost")
    ap.add_argument("--label-horizon-long",  type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    # Split train/val para tuning (apart de los walks)
    ap.add_argument("--train-from", default="2024-05-01")
    ap.add_argument("--train-to",   default="2025-05-01")
    ap.add_argument("--val-from",   default="2025-05-01")
    ap.add_argument("--val-to",     default="2025-07-01")
    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--reset-study", action="store_true",
                    help="Borra el study existente antes de crear uno nuevo. "
                         "Útil cuando se cambia el espacio de búsqueda (HPs nuevos, "
                         "rangos distintos) o se quiere empezar limpio sin trials "
                         "previos contaminando el sampler. Usa optuna.delete_study() "
                         "internamente; idempotente si el study no existe.")
    ap.add_argument("--objective", default="auc_pr",
                    choices=("auc_pr", "ev_net"),
                    help="Métrica a maximizar. 'auc_pr' (default) = max val_auc_pr "
                         "por side (ranking puro, estable). 'ev_net' = max EV_net "
                         "via threshold sweep en val, alineado con la métrica del "
                         "walkforward (en R con triple-barrier). EV_net es más "
                         "ruidoso (depende de min_signals) pero refleja mejor el "
                         "deploy. Recomendado: auc_pr para tuning amplio, ev_net "
                         "como refinamiento final.")
    ap.add_argument("--multi-objective", action="store_true",
                    help="Solo para SIDE=both. Optimiza simultáneamente "
                         "(score_long, score_short) y devuelve el frente de Pareto. "
                         "Usa NSGAIISampler en vez de TPE. Recomendado cuando "
                         "LONG y SHORT entran en trade-off (caso típico TCN 202500). "
                         "El JSON de salida incluye 'pareto_front' en vez de "
                         "'best_params'; elige luego el punto que respete tu floor "
                         "de SHORT (ver --min-short-score).")
    ap.add_argument("--ev-tp-mult", type=float, default=2.0,
                    help="TP multiplier para EV sweep. Default 2.0 (debe coincidir "
                         "con el walkforward; ver tp_mult en walk_config).")
    ap.add_argument("--ev-sl-mult", type=float, default=0.8,
                    help="SL multiplier para EV sweep. Default 0.8.")
    ap.add_argument("--ev-cost", type=float, default=0.05,
                    help="Coste por señal en R (slippage + comisiones). Default 0.05.")
    ap.add_argument("--ev-min-signals", type=int, default=30,
                    help="Mínimo de señales en val para considerar un threshold. "
                         "Default 30. Si val es ~2 meses (~17k filas), 30 es "
                         "~0.18% — razonable. Subir si val es más grande.")
    ap.add_argument("--out-dir", default=None,
                    help="Default: artifacts/<RELEASE>/oof/tuning")
    # resolve_regime_weights() los lee del Namespace; pasamos None por defecto
    # (mismo comportamiento que el walkforward).
    ap.add_argument("--regime-weights-long-json", default=None)
    ap.add_argument("--regime-weights-short-json", default=None)
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    set_global_seeds(int(args.seed))

    release = str(args.release)
    arch = str(args.arch)
    side = str(args.side)
    if args.study_name:
        study_name = args.study_name
    elif side == "both":
        study_name = f"oof_study_{release}_{arch}_multitask"
    else:
        study_name = f"oof_study_{release}_{arch}_{side}_only"

    print(f"\n{'═'*65}")
    print(f"  OPTUNA TUNING — arch={arch} side={side}")
    print(f"{'═'*65}")
    print(f"  Study:       {study_name}")
    print(f"  N trials:    {args.n_trials}")
    print(f"  Train:       {args.train_from} → {args.train_to}")
    print(f"  Val:         {args.val_from} → {args.val_to}")
    print(f"  Epochs/pat:  {args.epochs} / {args.patience}")
    print(f"  Seed:        {args.seed}")

    # 1) Configs (igual que walkforward, fixed)
    ts_train_from = pd.Timestamp(args.train_from)
    ts_val_to     = pd.Timestamp(args.val_to)
    earliest = ts_train_from - relativedelta(months=2)  # padding features
    latest = ts_val_to

    barriers   = _get_barriers_for_release(release)
    tf_def     = _tf_defaults(args.base_tf)
    fc = FeatureConfig(
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
        price_norm_window=tf_def["price_norm_window"],
        use_vol_invariant_features=(release in _VOL_INVARIANT_RELEASES),
        use_reduced_features=(release in _REDUCED_FEATURES_RELEASES),
        use_ultra_reduced_features=(release in _ULTRA_REDUCED_FEATURES_RELEASES),
    )
    gc = Config(release=release, use_oof=True, oof_splits=5,
                oof_epochs=int(args.epochs), save_oof_artifacts=False)
    rc = StateConfig(adx_trend_threshold=25.0)

    regime_weights = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights, verbose=False)

    # 2) OHLCV + prepare_data UNA vez sobre todo el rango (caro pero one-shot)
    print(f"\n📊 Cargando OHLCV {earliest.date()} → {latest.date()}")
    db = Database()
    resample = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=earliest, to_date=latest, resample=resample,
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras")

    print("⚙️  prepare_data global (puede tardar ~5-15min)")
    mc_dummy = ModelConfig(target_type="multitask",
                           epochs=args.epochs, patience=args.patience)
    pipeline_master = DataPipeline(
        general_config=gc, feature_config=fc,
        model_config=mc_dummy, regime_config=rc,
    )
    df_prep = pipeline_master.prepare_data(
        df_rates, labels=True, side="both",
        set_market_condition=False, ensure_regime=True,
    )
    print(f"   {len(df_prep):,} filas tras prepare_data")

    # 3) Split fijo train/val
    df_train = df_prep.loc[
        (df_prep["time"] >= ts_train_from) &
        (df_prep["time"] <  pd.Timestamp(args.train_to))
    ].reset_index(drop=True)
    df_val = df_prep.loc[
        (df_prep["time"] >= pd.Timestamp(args.val_from)) &
        (df_prep["time"] <  ts_val_to)
    ].reset_index(drop=True)
    print(f"   Train: {len(df_train):,} filas | Val: {len(df_val):,} filas")
    if len(df_train) < 10000 or len(df_val) < 500:
        raise SystemExit("❌ Splits insuficientes para tuning")

    # 4) Crear/cargar study
    if args.reset_study:
        try:
            optuna.delete_study(study_name=study_name, storage=args.optuna_storage)
            print(f"🗑️  Study previo '{study_name}' BORRADO (--reset-study)")
        except KeyError:
            print(f"🗑️  --reset-study: no había study previo con nombre '{study_name}' (OK)")
        except Exception as e:
            print(f"⚠️  --reset-study: error al borrar study previo: {e}")

    multi_obj = bool(args.multi_objective)
    if multi_obj and side != "both":
        raise SystemExit(
            "❌ --multi-objective requiere SIDE=both. "
            f"side actual='{side}'. Usa SIDE=both o quita --multi-objective."
        )

    if multi_obj:
        # NSGA-II para multi-obj; TPE no soporta directions natively de forma estable.
        study = optuna.create_study(
            study_name=study_name, storage=args.optuna_storage,
            directions=["maximize", "maximize"], load_if_exists=True,
            sampler=NSGAIISampler(seed=int(args.seed)),
        )
        print(f"🎯 Multi-objective study (long, short) — sampler=NSGAIISampler")
    else:
        study = optuna.create_study(
            study_name=study_name, storage=args.optuna_storage,
            direction="maximize", load_if_exists=True,
            sampler=TPESampler(seed=int(args.seed), n_startup_trials=5),
        )

    completed_before = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(f"\n🚀 Study ya tiene {completed_before} trials COMPLETE. "
          f"Lanzando {args.n_trials} adicionales.")
    print(f"   objective={args.objective}  multi_obj={multi_obj}")
    if args.objective == "ev_net":
        print(f"   EV sweep: tp={args.ev_tp_mult}R sl={args.ev_sl_mult}R "
              f"cost={args.ev_cost}R min_signals={args.ev_min_signals}")

    # 5) Optimización
    fail_score = (-1.0, -1.0) if multi_obj else -1.0

    def _objective(trial: optuna.Trial):
        t0 = time.time()
        hp = _suggest_hp(trial, arch, side=side)
        print(f"\n  Trial #{trial.number} hp={hp}")
        try:
            score = _train_and_eval(
                arch=arch, hp=hp,
                df_train=df_train, df_val=df_val,
                general_config=gc, feature_config=fc, regime_config=rc,
                epochs=int(args.epochs), patience=int(args.patience),
                seed=int(args.seed),
                side=side,
                objective=str(args.objective),
                multi_obj=multi_obj,
                ev_tp_mult=float(args.ev_tp_mult),
                ev_sl_mult=float(args.ev_sl_mult),
                ev_cost=float(args.ev_cost),
                ev_min_signals=int(args.ev_min_signals),
            )
        except Exception as e:
            print(f"     ❌ Trial {trial.number} crashed: {str(e)[:200]}")
            return fail_score
        dt = time.time() - t0
        if isinstance(score, tuple):
            print(f"     → score=(L={score[0]:+.4f}, S={score[1]:+.4f})  "
                  f"({dt/60:.1f}min)")
        else:
            print(f"     → score={score:+.4f}  ({dt/60:.1f}min)")
        return score

    study.optimize(_objective, n_trials=int(args.n_trials), show_progress_bar=False)

    # 6) Best params + persistencia
    print(f"\n{'═'*65}")
    print(f"  TUNING DONE — arch={arch}")
    print(f"{'═'*65}")

    out_dir = args.out_dir or f"artifacts/{release}/oof/tuning"
    os.makedirs(out_dir, exist_ok=True)
    side_suffix = {"both": "multitask", "long": "long_only", "short": "short_only"}[args.side]
    out_path = os.path.join(out_dir, f"best_params_{arch}_{side_suffix}.json")

    common = {
        "arch": arch, "release": release,
        "study_name": study_name,
        "objective": str(args.objective),
        "multi_objective": multi_obj,
        "n_trials_total": len(study.trials),
        "n_trials_completed": sum(
            1 for t in study.trials if t.state.name == "COMPLETE"),
        "epochs_per_trial": int(args.epochs),
        "split": {
            "train_from": args.train_from, "train_to": args.train_to,
            "val_from":   args.val_from,   "val_to":   args.val_to,
        },
    }
    if args.objective == "ev_net":
        common["ev_config"] = {
            "tp_mult": float(args.ev_tp_mult), "sl_mult": float(args.ev_sl_mult),
            "cost_per_signal": float(args.ev_cost),
            "min_signals": int(args.ev_min_signals),
        }

    if multi_obj:
        pareto = sorted(
            [
                {
                    "trial_number": int(t.number),
                    "score_long":  float(t.values[0]),
                    "score_short": float(t.values[1]),
                    "params":      dict(t.params),
                }
                for t in study.best_trials
                if t.values is not None and len(t.values) == 2
            ],
            key=lambda d: d["score_long"] + d["score_short"], reverse=True,
        )
        print(f"  Pareto front: {len(pareto)} trials")
        for i, p in enumerate(pareto[:5]):
            print(f"    #{p['trial_number']:3d}  "
                  f"L={p['score_long']:+.4f}  S={p['score_short']:+.4f}")
        out = {**common, "pareto_front": pareto, "n_pareto_trials": len(pareto)}
    else:
        print(f"  Best trial:  #{study.best_trial.number}")
        print(f"  Best value:  {study.best_value:+.4f}")
        print(f"  Best params: {study.best_params}")
        out = {**common,
               "best_trial":  int(study.best_trial.number),
               "best_value":  float(study.best_value),
               "best_params": dict(study.best_params)}

    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n💾 Resultados guardados en {out_path}")
    print(f"\n📋 Lanzar walkforward con esta arch tuneada:")
    print(f"   CNN_STUDY={study_name} ARCH={arch} bash 010_walkforward_cnn.sh")
    if multi_obj:
        print(f"   ⚠️  Multi-obj: elige un trial del Pareto en {out_path} y "
              f"pásalo a mano vía best_params_override.")


if __name__ == "__main__":
    main()
