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
from typing import Dict, Any

import numpy as np
import pandas as pd
import optuna
from optuna.samplers import TPESampler
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

def _suggest_hp(trial: optuna.Trial, arch: str) -> Dict[str, Any]:
    """Define HP space específico por arch. Comunes para todos (loss,
    optimizer, regularización) + específicos según arch (filters, conv, etc.)."""
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
    return mc


def _train_and_eval(
    arch: str, hp: Dict[str, Any],
    df_train: pd.DataFrame, df_val: pd.DataFrame,
    general_config: Config, feature_config: FeatureConfig,
    regime_config: StateConfig,
    epochs: int, patience: int, seed: int,
) -> float:
    """Entrena modelo con hp del trial sobre df_train, evalúa sobre df_val.
    Devuelve avg val_auc_pr (long+short)/2. Si algo falla, devuelve -1.0."""
    import tensorflow as tf
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
        print(f"      create_sequences_train failed: {e}")
        return -1.0
    pack_tr = seq_train.get("long") or {}
    X_tr = {k: pack_tr.get(k) for k in ("seq_short", "seq_long", "context", "time")}
    y_tr = pack_tr.get("labels")
    if y_tr is None or y_tr.ndim != 2:
        return -1.0

    # Sequences val
    try:
        seq_val = pipeline.create_sequences_by_side(
            df_val, sides=("long", "short"), fit_scalers=False, train=False,
        )
    except Exception as e:
        print(f"      create_sequences_val failed: {e}")
        return -1.0
    pack_va = seq_val.get("long") or {}
    X_va = {k: pack_va.get(k) for k in ("seq_short", "seq_long", "context", "time")}
    y_va = pack_va.get("labels")
    if y_va is None or y_va.ndim != 2:
        return -1.0

    # init_bias = logit del prior (clase positiva por side)
    pr_long  = float(np.clip(y_tr[:, 0].mean(), 1e-4, 1 - 1e-4))
    pr_short = float(np.clip(y_tr[:, 1].mean(), 1e-4, 1 - 1e-4))
    init_bias = {
        "long":  float(np.log(pr_long  / (1 - pr_long))),
        "short": float(np.log(pr_short / (1 - pr_short))),
    }

    # Build modelo alternativo
    shape_short = X_tr["seq_short"].shape[1:]
    shape_long  = X_tr["seq_long"].shape[1:]
    n_ctx       = int(X_tr["context"].shape[1])
    n_time      = int(X_tr["time"].shape[1])

    model = TradingModel(general_config=general_config, model_config=mc, side=None)
    model.model = build_model_by_arch(
        arch, shape_short, shape_long, n_ctx, n_time,
        mc, init_bias=init_bias,
    )
    model.compile_model()

    try:
        history = model.train(
            X_train=X_tr, y_train=y_tr,
            X_val=X_va, y_val=y_va,
            sample_weight=pack_tr.get("sample_weight"),
            verbose=0,
        )
    except Exception as e:
        print(f"      train failed: {e}")
        return -1.0

    # Score: avg max val AUC PR across epochs
    hk_long  = history.get("val_signal_long_auc_pr",  [])
    hk_short = history.get("val_signal_short_auc_pr", [])
    if not hk_long or not hk_short:
        return -1.0
    return float((max(hk_long) + max(hk_short)) / 2.0)


# ─── Main entrypoint ────────────────────────────────────────────────────

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Optuna tuning para arquitecturas alternativas drop-in.")
    ap.add_argument("--arch", required=True,
                    choices=("mlp", "hybrid", "transformer", "tcn"),
                    help="Arquitectura a tunear.")
    ap.add_argument("--release", default="202500")
    ap.add_argument("--study-name", default=None,
                    help="Default: oof_study_<RELEASE>_<arch>_multitask")
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
    ap.add_argument("--out-dir", default=None,
                    help="Default: artifacts/<RELEASE>/oof/tuning")
    return ap


def main() -> None:
    args = _build_argparser().parse_args()
    set_global_seeds(int(args.seed))

    release = str(args.release)
    arch = str(args.arch)
    study_name = args.study_name or f"oof_study_{release}_{arch}_multitask"

    print(f"\n{'═'*65}")
    print(f"  OPTUNA TUNING — arch={arch}")
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
    study = optuna.create_study(
        study_name=study_name, storage=args.optuna_storage,
        direction="maximize", load_if_exists=True,
        sampler=TPESampler(seed=int(args.seed), n_startup_trials=5),
    )

    completed_before = sum(1 for t in study.trials if t.state.name == "COMPLETE")
    print(f"\n🚀 Study ya tiene {completed_before} trials COMPLETE. "
          f"Lanzando {args.n_trials} adicionales.")

    # 5) Optimización
    def _objective(trial: optuna.Trial) -> float:
        t0 = time.time()
        hp = _suggest_hp(trial, arch)
        print(f"\n  Trial #{trial.number} hp={hp}")
        try:
            score = _train_and_eval(
                arch=arch, hp=hp,
                df_train=df_train, df_val=df_val,
                general_config=gc, feature_config=fc, regime_config=rc,
                epochs=int(args.epochs), patience=int(args.patience),
                seed=int(args.seed),
            )
        except Exception as e:
            print(f"     ❌ Trial {trial.number} crashed: {str(e)[:200]}")
            return -1.0
        dt = time.time() - t0
        print(f"     → score={score:+.4f}  ({dt/60:.1f}min)")
        return score

    study.optimize(_objective, n_trials=int(args.n_trials), show_progress_bar=False)

    # 6) Best params + persistencia
    print(f"\n{'═'*65}")
    print(f"  TUNING DONE — arch={arch}")
    print(f"{'═'*65}")
    print(f"  Best trial:  #{study.best_trial.number}")
    print(f"  Best value:  {study.best_value:+.4f}")
    print(f"  Best params: {study.best_params}")

    out_dir = args.out_dir or f"artifacts/{release}/oof/tuning"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"best_params_{arch}.json")
    with open(out_path, "w") as f:
        json.dump({
            "arch": arch, "release": release,
            "study_name": study_name,
            "best_trial":  int(study.best_trial.number),
            "best_value":  float(study.best_value),
            "best_params": study.best_params,
            "n_trials_total": len(study.trials),
            "n_trials_completed": sum(
                1 for t in study.trials if t.state.name == "COMPLETE"),
            "epochs_per_trial": int(args.epochs),
            "split": {
                "train_from": args.train_from, "train_to": args.train_to,
                "val_from":   args.val_from,   "val_to":   args.val_to,
            },
        }, f, indent=2)
    print(f"\n💾 Best params guardados en {out_path}")
    print(f"\n📋 Lanzar walkforward con esta arch tuneada:")
    print(f"   CNN_STUDY={study_name} ARCH={arch} bash 010_walkforward_cnn.sh")


if __name__ == "__main__":
    main()
