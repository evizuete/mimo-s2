#!/usr/bin/env python3
"""
resume_deploy_full_v6_multitask.py
─────────────────────────────────────────────────────────────────────────────
Sucesor moderno de resume_deploy_full_v5 pensado para releases multitask
(202200, 202300, 202400, 202500, 202501) y para futuros releases binarios
single-side. NO toca v5: v5 sigue siendo el punto de entrada para releases
viejas (200383, 200393, 200398...).

Diferencias clave respecto a v5:
  · Configuración del trainer parametrizable por CLI (release, fechas,
    base_tf, variantes, horizons, label_method...). Reusa `build_trainer`
    de main_oof_regime_weights_v7 para garantizar la misma config que el
    run de entrenamiento original.
  · Detecta automáticamente target_type (multitask vs binary) leyendo
    qué modelos hay en el source_artifacts_dir.
  · En multitask:
      - reentrena UN solo modelo (con ambas heads) sobre full-minus-tail.
      - recalibra DOS isotónicas (una por side) con la tail, persistidas
        como dict en oof_calibrator_<release>_multitask.joblib.
      - recompute percentiles por estado para CADA side desde la tail
        recalibrada.
      - genera policy final por side.
  · En binary single-side: delega en las funciones de v5 (back-compat).

Uso ejemplo (multitask 202500):
  python -m mimo.oof.resume_deploy_full_v6_multitask \
    --release 202500 --base-tf 5min --target-type multitask \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from 2024-01-01 --holdout-from 2025-11-01 --holdout-to 2026-05-02 \
    --train-artifacts-subdir rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \
    --deploy-subdir deploy_full_v6 \
    --deploy-calib-days 21 \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30 \
    --oof-epochs 120 --oof-patience 15
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import precision_recall_curve

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.databases import Database
from mimo.helpers.helper import Helper
from mimo.helpers.json_serialization import NumpyEncoder
from mimo.oof.optuna_oof_trainer_v2 import TrainerArtifacts

# Reuso de helpers y constantes de v5 (no las duplico)
from mimo.oof.resume_deploy_full_v5 import (
    DEPLOY_CALIB_DAYS,
    free_memory,
    _load_pipeline_scalers_for_side,
    recompute_percentiles_from_tail,
    generate_policy_from_deploy_percentiles,
    save_deploy_sequence_summary,
    split_deploy_train_and_calib,
    train_single_side as v5_train_single_side,
    recalibrate_deploy_side as v5_recalibrate_deploy_side,
    validate_after_training_with_metrics,
)

# Reuso del builder de v7 para mantener idéntica config con el training
from mimo.oof.main_oof_regime_weights_v7 import build_trainer as v7_build_trainer


# ─────────────────────────────────────────────────────────────────────────────
# Detección y resolución de paths
# ─────────────────────────────────────────────────────────────────────────────

def detect_target_type(source_dir: Path, release: str) -> Optional[str]:
    """Detecta target_type por los modelos presentes en source_dir."""
    multi_path = source_dir / f"model_{release}_multitask.keras"
    long_path = source_dir / f"model_{release}_long.keras"
    short_path = source_dir / f"model_{release}_short.keras"
    if multi_path.exists():
        return "multitask"
    if long_path.exists() and short_path.exists():
        return "binary"
    return None


def resolve_source_dir(
    base_dir: Path,
    release: str,
    explicit_dir: Optional[str],
    explicit_subdir: Optional[str],
) -> Path:
    """Resuelve la carpeta fuente de artifacts del training original."""
    if explicit_dir:
        p = Path(explicit_dir)
        if not p.exists():
            raise FileNotFoundError(f"--train-artifacts-dir no existe: {p}")
        return p
    if explicit_subdir and explicit_subdir != "auto":
        p = base_dir / explicit_subdir
        if not p.exists():
            raise FileNotFoundError(f"--train-artifacts-subdir no existe: {p}")
        return p
    # auto: buscar el subdir más reciente con model.*.keras
    candidates = [d for d in base_dir.iterdir() if d.is_dir()]
    candidates = sorted(candidates, key=lambda d: d.stat().st_mtime, reverse=True)
    for c in candidates:
        if detect_target_type(c, release) is not None:
            return c
    raise FileNotFoundError(
        f"No encontré subdir con artifacts del release {release} bajo {base_dir}. "
        "Pasa --train-artifacts-subdir explícito."
    )


# ─────────────────────────────────────────────────────────────────────────────
# Trainer reload con multitask
# ─────────────────────────────────────────────────────────────────────────────

def load_best_params_from_db(trainer, target_type: str) -> None:
    """Carga best_params_by_side desde la BD Optuna del trainer.

    Esto es necesario porque build_trainer crea el trainer con reload=False
    (no toca la BD). Aquí leemos el best trial del study correspondiente.
    """
    if target_type == "multitask":
        sides = ["multitask"]
    elif target_type == "binary":
        sides = ["long", "short"]
    else:
        raise ValueError(f"target_type desconocido: {target_type}")
    for side in sides:
        # _load_best_from_storage es el helper interno de OptunaOOFTrainer
        trainer._load_best_from_storage(side=side)
        print(f"  ✅ best_params cargados para side={side} "
              f"(trial #{getattr(trainer, 'best_params_by_side', {}).get(side, '?')})"
              if False else f"  ✅ best_params cargados para side={side}")


def load_locked_params(
    trainer,
    target_type: str,
    locked_path: Path,
    locked_side_key: Optional[str],
) -> Dict[str, Any]:
    """Sobrescribe best_params_by_side / best_model_config_by_side directamente
    desde un JSON con hyperparams locked (formato de extract_best_per_side o
    dict plano de params), sin tocar la BD Optuna.

    Returns: dict con metainfo (sides, trial origen, etc.) para logging.
    """
    if not locked_path.exists():
        raise SystemExit(f"❌ --locked-params-json no existe: {locked_path}")

    with locked_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    is_extract_report = "top_long" in payload or "top_short" in payload

    info: Dict[str, Any] = {"path": str(locked_path), "sides": {}}

    if target_type == "multitask":
        # En multitask se entrena UN modelo. El usuario debe indicar de qué
        # side toma los hyperparams (long o short del best_per_side, o pasa
        # un dict plano).
        if is_extract_report:
            if not locked_side_key:
                raise SystemExit(
                    "❌ --locked-params-json es un best_per_side; pasa "
                    "--locked-side-key {long,short} para indicar qué set "
                    "de hyperparams aplicar al modelo multitask."
                )
            sub = payload.get(f"top_{locked_side_key}", [])
            if not sub:
                raise SystemExit(f"❌ top_{locked_side_key} vacío en {locked_path}.")
            params = dict(sub[0].get("params", {}))
            info["sides"]["multitask"] = {
                "source": f"top_{locked_side_key}[0]",
                "trial": sub[0].get("trial", "?"),
                "params": params,
            }
        else:
            params = dict(payload)
            info["sides"]["multitask"] = {"source": "flat_dict", "params": params}

        mc = trainer._model_config_from_params(params)
        trainer.best_params_by_side["multitask"] = params
        trainer.best_model_config_by_side["multitask"] = mc

    elif target_type == "binary":
        # En binary single-side cada side puede usar sus propios locked params.
        # Solo aceptamos el report de extract_best_per_side; con dict plano no
        # podemos diferenciar long de short.
        if not is_extract_report:
            raise SystemExit(
                "❌ binary requiere best_per_side.json con top_long/top_short "
                "para diferenciar hyperparams por side. Dict plano no es suficiente."
            )
        for side in ("long", "short"):
            sub = payload.get(f"top_{side}", [])
            if not sub:
                print(f"  ⚠️  top_{side} vacío en {locked_path.name}; "
                      f"saltando side={side}.")
                continue
            params = dict(sub[0].get("params", {}))
            # Multitask → binary: colapsar focal_alpha_<side> a focal_alpha
            # escalar. _model_config_from_params construiría un dict
            # {long, short} si ve ambos, lo que rompe ClippedBinaryFocalCrossentropy
            # (binary head espera float, no dict).
            other = "short" if side == "long" else "long"
            if f"focal_alpha_{side}" in params:
                params["focal_alpha"] = float(params.pop(f"focal_alpha_{side}"))
                params.pop(f"focal_alpha_{other}", None)
            # loss_weight_long/short solo se usan en compile multitask
            # (loss_weights={'signal_long', 'signal_short'}). En binary el
            # head es uno solo y el escalar no se aplica → drop ambos.
            params.pop(f"loss_weight_long", None)
            params.pop(f"loss_weight_short", None)
            mc = trainer._model_config_from_params(params)
            trainer.best_params_by_side[side] = params
            trainer.best_model_config_by_side[side] = mc
            info["sides"][side] = {
                "source": f"top_{side}[0]",
                "trial": sub[0].get("trial", "?"),
                "params": params,
            }
    else:
        raise ValueError(f"target_type desconocido para locked params: {target_type}")

    print(f"\n🔒 [LOCKED PARAMS] {locked_path}")
    for side, meta in info["sides"].items():
        trial = meta.get("trial", "?")
        print(f"   · side={side:9s}  source={meta['source']}  trial=#{trial}")
        for k, v in sorted(meta["params"].items()):
            print(f"        {k:>22s} : {v}")
    return info


# ─────────────────────────────────────────────────────────────────────────────
# Train + recalibrate (multitask)
# ─────────────────────────────────────────────────────────────────────────────

def train_production_multitask(trainer, df_rates: pd.DataFrame,
                               *, reuse_best_trial_oof: bool = True) -> TrainerArtifacts:
    """Reentrena el modelo MULTITASK sobre full-minus-tail."""
    print("\n[DEPLOY] Reentrenando MULTITASK production model sobre FULL-minus-tail")
    if not reuse_best_trial_oof:
        print("   ℹ️  reuse_best_trial_oof=False (locked params: regenerar OOF desde cero).")
    artifacts = trainer.prepare_production_model(
        df_rates=df_rates,
        side="multitask",
        reuse_best_trial_oof=reuse_best_trial_oof,
    )
    free_memory()
    return artifacts


def train_production_single_side(trainer, df_rates: pd.DataFrame, side: str,
                                 *, reuse_best_trial_oof: bool = True) -> TrainerArtifacts:
    """Reentrena un modelo single-side sobre full-minus-tail.

    Equivalente a v5_train_single_side pero permite controlar reuse_best_trial_oof
    (necesario con --locked-params-json: el study locked está vacío y el branch
    de reuse intentaría leer study.best_trial → ValueError).
    """
    print(f"\n[DEPLOY] Preparing production {side.upper()} model based on FULL-minus-tail period")
    if not reuse_best_trial_oof:
        print("   ℹ️  reuse_best_trial_oof=False (locked params: regenerar OOF desde cero).")
    artifacts = trainer.prepare_production_model(
        df_rates=df_rates,
        side=side,
        reuse_best_trial_oof=reuse_best_trial_oof,
    )
    free_memory()
    return artifacts


def _select_threshold_f1(y_true: np.ndarray, y_pred_cal: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.5
    if len(np.unique(y_true)) < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_cal)
    if len(thresholds) == 0:
        return 0.5
    f1 = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    return float(thresholds[int(np.nanargmax(f1))])


def recalibrate_deploy_multitask(
    artifacts: TrainerArtifacts,
    deploy_dir: Path,
    release: str,
    df_calib: pd.DataFrame,
    general_config,
    feature_config,
    model_config,
    regime_config,
) -> Dict[str, Any]:
    """
    Para el modelo multitask reentrenado, recalibra DOS isotónicas (long, short)
    con la tail final. Persiste:
      - oof_calibrator_<release>_multitask.joblib  (dict {long, short})
      - percentiles_<release>_long.json
      - percentiles_<release>_short.json
      - data/deploy_calibration_tail_<release>_long.parquet
      - data/deploy_calibration_tail_<release>_short.parquet
    """
    print("\n[RECAL] Recalibrando MULTITASK (long+short) sobre cola final reciente...")
    model = tf.keras.models.load_model(artifacts.model_path)

    pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)
    scalers_used = _load_pipeline_scalers_for_side(pipeline, artifacts.model_path, "multitask")
    print(f"  · scalers: {scalers_used}")

    df_calib_prep = pipeline.prepare_data(df_calib.copy(), labels=True, side="both")
    sequences = pipeline.create_sequences_by_side(
        df_calib_prep, sides=("both",), fit_scalers=False, train=True
    )

    pack = sequences["both"]
    X_calib = [pack["seq_short"], pack["seq_long"], pack["context"], pack["time"]]
    labels = np.asarray(pack["labels"])  # shape (N, 2): [signal_long, signal_short]
    if labels.ndim != 2 or labels.shape[1] != 2:
        raise ValueError(
            f"Etiquetas multitask con shape inesperado: {labels.shape}. "
            "Se esperaba (N, 2) [signal_long, signal_short]."
        )
    y_long = labels[:, 0].astype(int)
    y_short = labels[:, 1].astype(int)

    if len(y_long) == 0:
        raise ValueError("Cola final vacía; no se puede recalibrar.")
    for side_label, y in (("long", y_long), ("short", y_short)):
        if len(np.unique(y)) < 2:
            raise ValueError(
                f"Cola final sin ambas clases para side={side_label}; "
                "amplía --deploy-calib-days o cambia la ventana."
            )

    raw_out = model.predict(X_calib, verbose=0)
    if isinstance(raw_out, dict) and "signal_long" in raw_out and "signal_short" in raw_out:
        p_raw_long = np.asarray(raw_out["signal_long"]).reshape(-1)
        p_raw_short = np.asarray(raw_out["signal_short"]).reshape(-1)
    elif isinstance(raw_out, (list, tuple)) and len(raw_out) == 2:
        p_raw_long = np.asarray(raw_out[0]).reshape(-1)
        p_raw_short = np.asarray(raw_out[1]).reshape(-1)
    else:
        raise ValueError(f"Salida del modelo multitask no reconocida: {type(raw_out)}")

    # Fit isotónicas
    cal_long = IsotonicRegression(out_of_bounds="clip")
    cal_long.fit(p_raw_long, y_long)
    p_cal_long = cal_long.predict(p_raw_long)

    cal_short = IsotonicRegression(out_of_bounds="clip")
    cal_short.fit(p_raw_short, y_short)
    p_cal_short = cal_short.predict(p_raw_short)

    thr_long = _select_threshold_f1(y_long, p_cal_long)
    thr_short = _select_threshold_f1(y_short, p_cal_short)

    # Persistir calibrador como dict (formato multitask consistente con training)
    cal_dict = {"long": cal_long, "short": cal_short}
    cal_out = deploy_dir / f"oof_calibrator_{release}_multitask.joblib"
    joblib.dump(cal_dict, str(cal_out))
    artifacts.calibrator_path = str(cal_out)
    print(f"  ✅ Calibrador multitask (dict) guardado: {cal_out}")
    print(f"  ✅ Threshold final LONG  (F1): {thr_long:.4f}")
    print(f"  ✅ Threshold final SHORT (F1): {thr_short:.4f}")

    # Datasets de calibración por side y percentiles por estado
    seq_len = len(y_long)
    df_tail_base = df_calib_prep.iloc[-seq_len:].copy()
    keep_cols = [c for c in ["time", "state"] if c in df_tail_base.columns]

    pct_results: Dict[str, Any] = {}
    for side_label, y_true_s, p_raw_s, p_cal_s, thr_s in [
        ("long", y_long, p_raw_long, p_cal_long, thr_long),
        ("short", y_short, p_raw_short, p_cal_short, thr_short),
    ]:
        df_tail = df_tail_base[keep_cols].copy()
        df_tail["signal"] = y_true_s
        df_tail["oof_proba_raw"] = p_raw_s
        df_tail["oof_proba_cal"] = p_cal_s
        df_tail["side"] = side_label
        df_tail["source"] = "deploy_calib_tail"
        out_parquet = deploy_dir / "data" / f"deploy_calibration_tail_{release}_{side_label}.parquet"
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        df_tail.to_parquet(out_parquet, index=False)

        # Percentiles por estado: el src_pct viene del TRAINING original
        # (percentiles_<release>_<side>.json) — usamos como base.
        src_pct = Path(artifacts.percentiles_path).parent / f"percentiles_{release}_{side_label}.json"
        if not src_pct.exists():
            src_pct = Path(artifacts.percentiles_path)  # fallback
        pct = recompute_percentiles_from_tail(
            df_tail,
            selected_threshold=thr_s,
            src_pct=src_pct,
            proba_col="oof_proba_cal",
            regime_col="state",
        )
        pct.setdefault("_meta", {})
        pct["_meta"]["threshold_source"] = "deploy_calibration_tail"
        pct["_meta"]["deploy_calib_rows"] = int(len(y_true_s))
        if "time" in df_tail.columns:
            pct["_meta"]["deploy_calib_from"] = str(pd.to_datetime(df_tail["time"]).min())
            pct["_meta"]["deploy_calib_to"] = str(pd.to_datetime(df_tail["time"]).max())

        pct_out = deploy_dir / f"percentiles_{release}_{side_label}.json"
        with open(pct_out, "w") as f:
            json.dump(pct, f, indent=2, cls=NumpyEncoder)
        pct_results[side_label] = {"thr": thr_s, "pct_path": str(pct_out)}
        print(f"  ✅ Percentiles {side_label} guardados: {pct_out}")
        print(f"  ✅ Calibration tail {side_label}: {out_parquet}")

    # Sobrescribir artifacts.percentiles_path al multitask (solo informativo)
    artifacts.percentiles_path = str(deploy_dir / f"percentiles_{release}_long.json")

    free_memory()
    return {"artifacts": artifacts, "thresholds": {"long": thr_long, "short": thr_short}}


# ─────────────────────────────────────────────────────────────────────────────
# Validación post-deploy (multitask: usa el mismo modelo dos veces)
# ─────────────────────────────────────────────────────────────────────────────

def validate_deploy_multitask(
    artifacts_multi: TrainerArtifacts,
    df_full: pd.DataFrame,
    deploy_dir: Path,
    general_config,
    feature_config,
    model_config,
    regime_config,
) -> Dict[str, Any]:
    """
    Para multitask, el modelo es uno solo: reusamos validate_after_training_with_metrics
    pasando dos `artifacts` que apuntan al mismo modelo y al mismo calibrador
    (que es un dict, pero la función solo lo carga; el branching por side ocurre
    posteriormente cuando se predice).
    """
    print("\n🏁 Iniciando validación de MULTITASK deploy...")
    # Construye dos TrainerArtifacts apuntando al mismo modelo. El calibrador
    # es un dict {long, short}; la validación carga ambos por separado.
    return validate_after_training_with_metrics(
        artifacts_long=artifacts_multi,
        artifacts_short=artifacts_multi,
        df_full=df_full,
        out_dir=str(deploy_dir),
        general_config=general_config,
        feature_config=feature_config,
        model_config=model_config,
        regime_config=regime_config,
    )


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description=(
            "Reentrena el modelo de producción sobre full-minus-tail y recalibra "
            "isotónicas + percentiles con la tail. Soporta multitask (202200+) y "
            "binary single-side. Configuración parametrizable; reusa build_trainer "
            "de main_oof_regime_weights_v7."
        )
    )
    ap.add_argument("--release", required=True)
    ap.add_argument(
        "--target-type",
        choices=["auto", "multitask", "binary", "quantile", "magnitude", "triple_class"],
        default="auto",
        help="auto = detectar leyendo los modelos en el source_artifacts_dir.",
    )
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--variant-long", default="vol_boost_td_down")
    ap.add_argument("--variant-short", default="vol_boost")
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)

    ap.add_argument("--train-from", required=True,
                    help="Inicio de la ventana COMPLETA (train + holdout).")
    ap.add_argument("--holdout-from", required=True,
                    help="Inicio del holdout (no se usa para split, solo informativo).")
    ap.add_argument("--holdout-to", required=True,
                    help="Fin de los datos. Equivale al final del holdout.")

    ap.add_argument("--deploy-calib-days", type=int, default=DEPLOY_CALIB_DAYS,
                    help=f"Días reservados al final para tail recalibration "
                         f"(default {DEPLOY_CALIB_DAYS}).")
    ap.add_argument("--train-artifacts-subdir", default="auto",
                    help="Subcarpeta bajo artifacts/<release>/oof/ con artifacts del training.")
    ap.add_argument("--train-artifacts-dir", default=None,
                    help="Path completo a la carpeta de training (override del subdir).")
    ap.add_argument("--deploy-subdir", default="deploy_full_v6",
                    help="Subcarpeta bajo artifacts/<release>/oof/ donde escribir el deploy.")

    ap.add_argument("--side", choices=["long", "short", "both"], default="both",
                    help="Solo aplica en target_type=binary single-side.")
    ap.add_argument("--skip-validation", action="store_true",
                    help="No correr la validación post-deploy (en multitask se "
                         "salta SIEMPRE salvo --force-validation porque el "
                         "validador heredado de v5 no soporta multitask).")
    ap.add_argument("--force-validation", action="store_true",
                    help="Fuerza la validación incluso en multitask. Las tests "
                         "TEST 7/TEST 8 fallarán por incompatibilidad del "
                         "validador v5; sólo útil para debug.")
    ap.add_argument("--validate-only", action="store_true")

    # Forwarded to build_trainer
    ap.add_argument("--objective", choices=["aucpr", "ev_net"], default="aucpr")
    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--ev-min-signals", type=int, default=100)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--ev-thr-lo", type=float, default=0.10)
    ap.add_argument("--ev-thr-hi", type=float, default=0.40)
    ap.add_argument("--oof-epochs", type=int, default=120)
    ap.add_argument("--oof-patience", type=int, default=15)

    # Locked params: si se pasa, NO se cargan best_params del study Optuna;
    # se usa el JSON directamente. Imprescindible para deploys de specialists.
    ap.add_argument("--locked-params-json", type=str, default=None,
                    help="JSON con hyperparams locked (best_per_side.json o "
                         "dict plano). Si se pasa, omite la carga de best "
                         "params desde la BD Optuna.")
    ap.add_argument("--locked-side-key", choices=["long", "short"], default=None,
                    help="En multitask con best_per_side.json: indica si tomar "
                         "top_long[0] o top_short[0] como hyperparams del "
                         "modelo multitask reentrenado. En binary se ignora "
                         "(cada side toma su propio top_<side>[0]).")

    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    start = time.perf_counter()

    base_dir = Path("../../artifacts") / args.release / "oof"
    deploy_dir = base_dir / args.deploy_subdir
    deploy_dir.mkdir(parents=True, exist_ok=True)
    (deploy_dir / "data").mkdir(parents=True, exist_ok=True)
    (deploy_dir / "reports").mkdir(parents=True, exist_ok=True)

    print(f"\n🚀 resume_deploy_full_v6_multitask | release={args.release} | "
          f"target_type={args.target_type}")

    # 1. Resolver source dir y detectar target_type si auto
    source_dir = resolve_source_dir(
        base_dir, args.release,
        explicit_dir=args.train_artifacts_dir,
        explicit_subdir=args.train_artifacts_subdir,
    )
    print(f"📂 Source artifacts dir: {source_dir}")

    detected_tt = detect_target_type(source_dir, args.release)
    if args.target_type == "auto":
        if detected_tt is None:
            raise SystemExit(
                f"❌ No pude detectar target_type en {source_dir}. "
                f"Pasa --target-type explícito."
            )
        target_type = detected_tt
    else:
        target_type = args.target_type
        if detected_tt is not None and detected_tt != target_type:
            print(f"⚠️  target_type={target_type} pasado pero detectado {detected_tt}.")

    print(f"🧠 target_type efectivo: {target_type}")

    # 2. Cargar OHLCV (full window)
    print("\n📂 Cargando OHLCV...")
    db = Database()
    resample_arg = (
        None
        if str(args.base_tf).lower() in ("1min", "1m", "none", "")
        else args.base_tf
    )
    dm = DataManager.from_database_historical_2(
        db,
        from_date=args.train_from,
        to_date=args.holdout_to,
        resample=resample_arg,
    )
    df_rates = dm.df
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    df_rates = df_rates.sort_values("time").drop_duplicates("time").reset_index(drop=True)
    print(f"   {len(df_rates):,} filas | rango {df_rates['time'].iloc[0]} → "
          f"{df_rates['time'].iloc[-1]}")

    # 3. Construir trainer reusando build_trainer de v7 (config alineada)
    print("\n🔧 Reconstruyendo trainer (build_trainer de main_oof_regime_weights_v7)...")
    trainer = v7_build_trainer(
        release=args.release,
        label_horizon_long=args.label_horizon_long,
        label_horizon_short=args.label_horizon_short,
        train_dir=source_dir,
        target_type=target_type,
        base_tf=args.base_tf,
        objective_kind=args.objective,
        cost_per_signal=args.cost_per_signal,
        ev_min_signals=args.ev_min_signals,
        max_drawdown_R=args.max_drawdown_R,
        ev_thr_lo=args.ev_thr_lo,
        ev_thr_hi=args.ev_thr_hi,
        oof_epochs=args.oof_epochs,
        oof_patience=args.oof_patience,
    )
    print("   ✅ trainer construido")

    # 4. Cargar best_params: desde JSON locked si --locked-params-json,
    #    si no desde la BD Optuna del trainer.
    locked_info: Optional[Dict[str, Any]] = None
    if args.locked_params_json:
        print("\n🔒 Cargando best params desde JSON locked (omitiendo Optuna DB)...")
        locked_info = load_locked_params(
            trainer,
            target_type=target_type,
            locked_path=Path(args.locked_params_json),
            locked_side_key=args.locked_side_key,
        )
    else:
        print("\n📥 Cargando best params desde Optuna DB...")
        load_best_params_from_db(trainer, target_type)

    if args.validate_only:
        print("\n--validate-only: cargo artifacts existentes y valido.")
        if target_type == "multitask":
            model_path = deploy_dir / f"model_{args.release}_multitask.keras"
            cal_path = deploy_dir / f"oof_calibrator_{args.release}_multitask.joblib"
            pct_path = deploy_dir / f"percentiles_{args.release}_long.json"
            if not all(p.exists() for p in [model_path, cal_path, pct_path]):
                raise SystemExit(f"❌ Faltan artifacts deploy en {deploy_dir}")
            artifacts_multi = TrainerArtifacts(
                study_name=f"oof_study_{args.release}_multitask",
                best_params={},
                best_value=0.0,
                params_signature="",
                model_config=trainer.base_model_config,
                feature_config=trainer.feature_config,
                regime_config=trainer.regime_config,
                side="multitask",
                model_path=str(model_path),
                calibrator_path=str(cal_path),
                percentiles_path=str(pct_path),
                oof_df_path=None,
            )
            if not args.skip_validation:
                validate_deploy_multitask(
                    artifacts_multi, df_rates, deploy_dir,
                    trainer.general_config, trainer.feature_config,
                    trainer.base_model_config, trainer.regime_config,
                )
            print("✅ Validación validate-only completa.")
            return
        else:
            raise SystemExit("validate-only para binary aún no está implementado en v6.")

    # 5. Split deploy_train (full minus tail) y deploy_calib (tail)
    df_deploy_train, df_deploy_calib, calib_from = split_deploy_train_and_calib(
        df_rates, calib_days=args.deploy_calib_days
    )

    # Apuntar trainer al deploy_dir para que escriba ahí (no en source_dir)
    trainer.out_dir = str(deploy_dir)

    Helper.save_meta(
        {
            "release": args.release,
            "target_type": target_type,
            "trained_on": "full_minus_tail",
            "train_from": str(args.train_from),
            "holdout_to": str(args.holdout_to),
            "deploy_calib_from": str(calib_from),
            "deploy_calib_days": int(args.deploy_calib_days),
            "source_artifacts_dir": str(source_dir),
            "base_tf": args.base_tf,
            "label_horizon_long": args.label_horizon_long,
            "label_horizon_short": args.label_horizon_short,
            "variant_long": args.variant_long,
            "variant_short": args.variant_short,
            "locked_params_json": (
                str(args.locked_params_json) if args.locked_params_json else None
            ),
            "locked_side_key": args.locked_side_key,
            "locked_info": locked_info,
            "note": (
                "v6: trained on full-minus-tail; tail reserved for final "
                "isotonic recalibration. Old Optuna OOF calibrator NOT "
                "reused as final calibrator. reuse_best_trial_oof=True only "
                "to avoid OOM during model build."
            ),
        },
        str(deploy_dir / "data" / "train_meta.json"),
    )

    # 6. Reentrenar production model (multitask: 1 modelo; binary: 2 modelos)
    deploy_artifacts: Dict[str, Any] = {}

    # Con locked params, el study Optuna está vacío (sin trials previos),
    # así que reuse_best_trial_oof=True falla en study.best_trial. Forzar False
    # cuando hay locked params: regenerar OOF desde cero con la config locked.
    reuse_oof = not bool(args.locked_params_json)

    if target_type == "multitask":
        artifacts_multi = train_production_multitask(
            trainer, df_deploy_train, reuse_best_trial_oof=reuse_oof,
        )
        result = recalibrate_deploy_multitask(
            artifacts=artifacts_multi,
            deploy_dir=deploy_dir,
            release=args.release,
            df_calib=df_deploy_calib,
            general_config=trainer.general_config,
            feature_config=trainer.feature_config,
            model_config=trainer.base_model_config,
            regime_config=trainer.regime_config,
        )
        deploy_artifacts["multitask"] = result["artifacts"]

        thr_long_f1 = result["thresholds"]["long"]
        thr_short_f1 = result["thresholds"]["short"]
        print("\n" + "⚠️ " * 18)
        print("  AVISO: thresholds elegidos por F1 sobre la tail")
        print("⚠️ " * 18)
        print(f"  · LONG  F1-thr  : {thr_long_f1:.4f}")
        print(f"  · SHORT F1-thr  : {thr_short_f1:.4f}")
        print()
        print("  El criterio F1 maximiza precision×recall sobre etiquetas binarias")
        print("  pero NO refleja el EV económico real (tp/sl, horizon, EXPIRE bias).")
        print("  Con tasa base ~7-8%, F1 tiende a thresholds bajos (~0.10-0.13)")
        print("  mientras que el thr óptimo EV-net del trial original suele caer")
        print("  alrededor de ~0.20-0.30.")
        print()
        print("  📌 Recomendado antes de pasar este deploy a paper trading:")
        print("     python -m mimo.oof.select_thresholds_from_tail \\")
        print(f"       --release {args.release} --deploy-dir {deploy_dir} \\")
        print(f"       --side both --from-db --base-tf {args.base_tf} \\")
        print(f"       --cost {args.cost_per_signal} --min-signals 30")
        print()

    elif target_type == "binary":
        sides = ["long", "short"] if args.side == "both" else [args.side]
        for side in sides:
            art = train_production_single_side(
                trainer, df_deploy_train, side, reuse_best_trial_oof=reuse_oof,
            )
            art = v5_recalibrate_deploy_side(
                artifacts=art,
                deploy_dir=deploy_dir,
                release=args.release,
                side=side,
                df_calib=df_deploy_calib,
                general_config=trainer.general_config,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )
            deploy_artifacts[side] = art

    else:
        raise SystemExit(
            f"target_type={target_type} no implementado en v6 todavía. "
            "Soportados: multitask, binary."
        )

    # 7. Generar policy desde percentiles deploy
    generated_policy_paths = generate_policy_from_deploy_percentiles(deploy_dir, args.release) or {}

    # 8. Validación post-deploy
    final_validation = None
    if args.skip_validation:
        print("\nℹ️  --skip-validation: validación post-deploy omitida.")
    elif target_type == "multitask" and not args.force_validation:
        print("\nℹ️  Validación post-deploy omitida en multitask por defecto.")
        print("    El validador heredado de v5 (validate_after_training_with_metrics)")
        print("    no soporta multitask: TEST 7 falla por shape mismatch entre el")
        print("    feature_mask single-side (24/23 cols) y el modelo multitask que")
        print("    espera la unión (25 cols), y TEST 8 falla porque el calibrador")
        print("    ahora se persiste como dict {long, short}. Pasa --force-validation")
        print("    si quieres ver los fallos para debug.")
    elif target_type == "multitask" and args.force_validation:
        final_validation = validate_deploy_multitask(
            deploy_artifacts["multitask"], df_rates, deploy_dir,
            trainer.general_config, trainer.feature_config,
            trainer.base_model_config, trainer.regime_config,
        )
    elif target_type == "binary" and args.side == "both":
        final_validation = validate_after_training_with_metrics(
            artifacts_long=deploy_artifacts["long"],
            artifacts_short=deploy_artifacts["short"],
            df_full=df_rates,
            out_dir=str(deploy_dir),
            general_config=trainer.general_config,
            feature_config=trainer.feature_config,
            model_config=trainer.base_model_config,
            regime_config=trainer.regime_config,
        )
    else:
        print("\nℹ️  Validación final omitida (binary single-side).")

    # 9. Resumen
    save_deploy_sequence_summary(
        deploy_dir,
        args.release,
        holdout_policy_info={},
        policy_paths=generated_policy_paths,
        validation_results=final_validation,
    )

    elapsed = time.perf_counter() - start
    print("\n" + "╔" + "═" * 68 + "╗")
    print(f"║  ✅ deploy_full_v6 listo en {deploy_dir}".ljust(69) + "║")
    print(f"║  ⏱️  {elapsed:.1f}s".ljust(69) + "║")
    print("╚" + "═" * 68 + "╝\n")


if __name__ == "__main__":
    main()
