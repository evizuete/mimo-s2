"""
resume_deploy_full_v6.py
─────────────────────────────────────────────────────────────────────────────
Versión v4 de resume_deploy_full.py alineada con main_oof.py, con
recalibración final sobre una cola temporal reciente, recomputación completa
de percentiles por estado sobre esa misma cola recalibrada y generación
automática del fichero final de policy.

Objetivo:
  1. Mantener el flujo anti-OOM de deploy_full con reuse_best_trial_oof=True.
  2. Alinear la configuración con main_oof.py.
  3. Reemplazar calibrador, threshold global y percentiles por estado del deploy
     por unos recalculados sobre una cola reciente del dataset completo, evitando
     reutilizar sin más los percentiles heredados de train_only.
  4. Generar automáticamente un fichero final de policy a partir de esos percentiles.

Notas:
  - El modelo final se entrena sobre full-minus-tail.
  - La cola final (tail) se reserva solo para recalibración final.
  - Se sobrescriben en deploy_full:
      * oof_calibrator_<release>_<side>.joblib
      * percentiles_<release>_<side>.json   (recomputado completo por estado + threshold final)
  - Se guarda además un parquet con las predicciones raw/cal de la cola final
    para auditar drift/calibración posterior.
"""

import argparse
import ctypes
import gc
import json
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
)

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.databases import Database
from mimo.features.feature_builder import FeatureConfig
from mimo.helpers.helper import Helper
from mimo.helpers.json_serialization import NumpyEncoder
from mimo.models.model_builder import Config, ModelConfig
from mimo.oof.main_oof import build_calibration_dataset
from mimo.oof.optuna_oof_trainer_v2 import OptunaOOFTrainer, TrainerArtifacts
from mimo.oof.validate_full_training import TrainingValidator
from mimo.states_manager.state_detector import StateConfig

# ─────────────────────────────────────────────────────────────────────────────
# Runtime / memoria
# ─────────────────────────────────────────────────────────────────────────────

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private"
os.environ["TF_GPU_THREAD_COUNT"] = "1"

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configurada: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  Error configurando GPU: {e}")

gc.set_threshold(700, 10, 10)

DEPLOY_CALIB_DAYS = 21


# ─────────────────────────────────────────────────────────────────────────────
# Utilidades de memoria
# ─────────────────────────────────────────────────────────────────────────────

def free_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()


def release_artifact_refs(*names):
    """Pone a None referencias pesadas del scope del caller si se le pasan dicts()."""
    for scope, keys in names:
        for key in keys:
            if key in scope:
                scope[key] = None
    free_memory()


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de scalers
# ─────────────────────────────────────────────────────────────────────────────

def _load_pipeline_scalers_for_side(pipeline, model_path, side):
    """
    Carga scalers side-specific si existen; si no, cae al formato genérico.
    Devuelve la ruta lógica utilizada.
    """
    import shutil
    import tempfile

    model_dir = Path(model_path).parent
    release = getattr(pipeline.general_config, "release", None)
    generic_name = f"scalers_{release}" if release is not None else "scalers"
    side_name = f"{generic_name}_{side}"

    side_dir = model_dir / side_name
    generic_dir = model_dir / generic_name

    if side_dir.exists() and side_dir.is_dir():
        tmp_root = Path(tempfile.mkdtemp(prefix=f"scalers_{side}_"))
        tmp_target = tmp_root / generic_name
        shutil.copytree(side_dir, tmp_target)
        pipeline.load_scalers(str(tmp_root))
        return str(side_dir)

    if generic_dir.exists() and generic_dir.is_dir():
        pipeline.load_scalers(str(model_dir))
        return str(generic_dir)

    pipeline.load_scalers(str(model_dir))
    return str(model_dir)


# ─────────────────────────────────────────────────────────────────────────────
# Validación post-entrenamiento
# ─────────────────────────────────────────────────────────────────────────────

def validate_after_training_with_metrics(
    artifacts_long,
    artifacts_short,
    df_full,
    out_dir,
    general_config,
    feature_config,
    model_config,
    regime_config,
):
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 14 + "VALIDACIÓN POST-ENTRENAMIENTO" + " " * 24 + "║")
    print("║" + " " * 20 + "CON MÉTRICAS COMPLETAS" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    validation_results = {}

    try:
        split_point = int(len(df_full) * 0.8)
        df_train_v = df_full.iloc[:split_point]
        df_val_v = df_full.iloc[split_point:]
        print(f"\n🔍 Validando con {len(df_train_v):,} train + {len(df_val_v):,} val...")

        print("\n📦 Cargando artifacts...")
        model_long = tf.keras.models.load_model(artifacts_long.model_path)
        model_short = tf.keras.models.load_model(artifacts_short.model_path)
        print("  ✅ Modelos cargados")

        with open(artifacts_long.calibrator_path, "rb") as f:
            cal_long = joblib.load(f)
        with open(artifacts_short.calibrator_path, "rb") as f:
            cal_short = joblib.load(f)
        print("  ✅ Calibradores cargados")

        pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)
        artifacts_dir = Path(artifacts_long.model_path).parent
        try:
            pipeline.load_scalers(base_path=str(artifacts_dir))
            print(f"  ✅ Pipeline y scalers cargados desde: {artifacts_dir}")
        except Exception as e:
            print(f"  ⚠️  Scalers no cargados desde {artifacts_dir}: {type(e).__name__}: {e}")

        print("\n" + "─" * 70)
        print("PARTE 1: VALIDACIÓN DEL PIPELINE")
        print("─" * 70)

        validator = TrainingValidator()
        pipeline_results = validator.run_all_checks(
            pipeline=pipeline,
            df_train=df_train_v,
            df_val=df_val_v,
            models={"long": model_long, "short": model_short},
            calibrators={"long": cal_long, "short": cal_short},
            scalers=pipeline.scalers if hasattr(pipeline, "scalers") else None,
        )
        validation_results["pipeline_validation"] = pipeline_results

        print("\n" + "─" * 70)
        print("PARTE 2: MÉTRICAS DETALLADAS")
        print("─" * 70)

        models_metrics = {}
        for side in ["long", "short"]:
            print(f"\n📊 Evaluando {side.upper()}...")
            model = model_long if side == "long" else model_short
            calibrator = cal_long if side == "long" else cal_short
            try:
                pipeline_side = DataPipeline(general_config, feature_config, model_config, regime_config)
                scalers_used = _load_pipeline_scalers_for_side(
                    pipeline_side,
                    artifacts_long.model_path if side == "long" else artifacts_short.model_path,
                    side,
                )
                print(f"     · scalers: {scalers_used}")
                df_val_prep = pipeline_side.prepare_data(df_val_v.copy(), labels=True, side=side)
                sequences = pipeline_side.create_sequences_by_side(
                    df_val_prep, sides=(side,), fit_scalers=False, train=True
                )

                X_val = [
                    sequences[side]["seq_short"],
                    sequences[side]["seq_long"],
                    sequences[side]["context"],
                    sequences[side]["time"],
                ]
                y_true = sequences[side]["labels"]

                y_pred_raw = model.predict(X_val, verbose=0).reshape(-1)
                y_pred_cal = calibrator.predict(y_pred_raw) if calibrator else y_pred_raw

                auc_roc = float(roc_auc_score(y_true, y_pred_cal))
                auc_pr = float(average_precision_score(y_true, y_pred_cal))

                artifacts_side = artifacts_long if side == "long" else artifacts_short
                oof_threshold = None
                try:
                    with open(artifacts_side.percentiles_path) as f:
                        pct = json.load(f)
                    oof_threshold = (
                        pct.get("_meta", {}).get("selected_threshold")
                        or pct.get("selected_threshold")
                    )
                except Exception:
                    pass

                precision_curve, recall_curve, thresholds_curve = precision_recall_curve(y_true, y_pred_cal)
                if len(thresholds_curve) > 0:
                    f1_scores = 2 * (precision_curve[:-1] * recall_curve[:-1]) / (
                        precision_curve[:-1] + recall_curve[:-1] + 1e-10
                    )
                    val_opt_thr = float(thresholds_curve[np.argmax(f1_scores)])
                else:
                    val_opt_thr = 0.5

                if oof_threshold is not None:
                    best_thr = float(oof_threshold)
                    thr_source = f"deploy threshold ({best_thr:.4f})"
                else:
                    best_thr = val_opt_thr
                    thr_source = f"óptimo val (fallback) ({best_thr:.4f})"
                    print(
                        f"  ⚠️  [{side.upper()}] threshold deploy no encontrado en percentiles_path, "
                        f"usando óptimo sobre val (puede ser optimista)"
                    )

                y_pred_bin = (y_pred_cal >= best_thr).astype(int)
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_bin).ravel()

                models_metrics[side] = {
                    "auc_roc": float(auc_roc),
                    "auc_pr": float(auc_pr),
                    "best_threshold": best_thr,
                    "threshold_source": thr_source,
                    "val_optimal_thr": val_opt_thr,
                    "precision": float(tp / (tp + fp + 1e-10)),
                    "recall": float(tp / (tp + fn + 1e-10)),
                    "f1": float(2 * tp / (2 * tp + fp + fn + 1e-10)),
                    "accuracy": float((tp + tn) / (tp + tn + fp + fn)),
                    "signal_rate": float(y_pred_bin.mean()),
                    "base_rate": float(y_true.mean()),
                    "confusion_matrix": {"tp": int(tp), "fp": int(fp), "tn": int(tn), "fn": int(fn)},
                    "n_samples": int(len(y_true)),
                }
                m = models_metrics[side]
                print(f"  ✅ AUC-ROC: {m['auc_roc']:.4f} | AUC-PR: {m['auc_pr']:.4f}")
                print(f"     Threshold: {thr_source}")
                print(
                    f"     P: {m['precision']:.4f} | R: {m['recall']:.4f} | "
                    f"F1: {m['f1']:.4f} | signal_rate: {m['signal_rate']:.4f}"
                )
                print(f"     (ref) threshold óptimo val: {val_opt_thr:.4f}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                models_metrics[side] = {"error": str(e)}

        validation_results["models_metrics"] = models_metrics
        validation_results["oof_metrics"] = {
            "long": artifacts_long.oof_metrics if artifacts_long.oof_metrics else {},
            "short": artifacts_short.oof_metrics if artifacts_short.oof_metrics else {},
        }

        validation_path = Path(out_dir) / "reports" / "validation_post_training_complete.json"
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        with open(validation_path, "w") as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)
        print(f"\n✅ Resultados guardados: {validation_path}")

        pipeline_ok = pipeline_results["summary"]["status"] == "PASS"
        models_ok = all("error" not in models_metrics.get(s, {"error": ""}) for s in ["long", "short"])
        print("\n" + "=" * 70)
        print("✅ VALIDACIÓN COMPLETA EXITOSA" if (pipeline_ok and models_ok) else "⚠️  VALIDACIÓN PARCIAL")
        print("=" * 70 + "\n")

        return validation_results

    except Exception as e:
        import traceback

        print(f"\n❌ Error crítico en validación: {e}")
        traceback.print_exc()
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Reconstrucción de TrainerArtifacts desde disco
# ─────────────────────────────────────────────────────────────────────────────

def load_artifacts_from_disk(artifacts_dir: Path, release: str, side: str) -> TrainerArtifacts:
    """
    Reconstruye un TrainerArtifacts leyendo los ficheros ya guardados.
    No requiere re-entrenar nada.
    """
    model_path = str(artifacts_dir / f"model_{release}_{side}.keras")
    cal_path = str(artifacts_dir / f"oof_calibrator_{release}_{side}.joblib")
    pct_path = str(artifacts_dir / f"percentiles_{release}_{side}.json")
    oof_df_path = str(artifacts_dir / f"oof_{release}_{side}.parquet")
    oof_meta_path = str(artifacts_dir / f"oof_meta_{release}_{side}.json")

    missing = [p for p in [model_path, cal_path, pct_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos para side={side} en {artifacts_dir}: {missing}")

    oof_metrics = {}
    if os.path.exists(oof_df_path):
        try:
            df_oof = pd.read_parquet(oof_df_path)
            mask = np.isfinite(df_oof["oof_proba_cal"]) & np.isfinite(df_oof["signal"])
            if mask.sum() > 0:
                y = df_oof.loc[mask, "signal"].to_numpy().astype(int)
                p = df_oof.loc[mask, "oof_proba_cal"].to_numpy()
                oof_metrics["auc_pr"] = float(average_precision_score(y, p))
            del df_oof
        except Exception as ex:
            print(f"  ⚠️  No se pudo cargar oof_metrics para {side}: {ex}")

    best_params = {}
    if os.path.exists(oof_meta_path):
        try:
            with open(oof_meta_path) as f:
                meta = json.load(f)
            best_params = meta.get("params", {})
        except Exception:
            pass

    print(f"  ✅ Artefactos {side.upper()} cargados desde disco: {artifacts_dir}")
    return TrainerArtifacts(
        best_params=best_params,
        best_value=float("nan"),
        study_name=f"recovered_{release}_{side}",
        side=side,
        oof_metrics=oof_metrics,
        percentiles={},
        calibrator_path=cal_path,
        percentiles_path=pct_path,
        model_path=model_path,
        oof_meta_path=oof_meta_path,
        oof_df_path=oof_df_path if os.path.exists(oof_df_path) else None,
    )


def _side_list_from_arg(side_arg: str) -> list[str]:
    return ["long", "short"] if side_arg == "both" else [side_arg]


def _artifact_paths_for_side(base_dir: Path, release: str, side: str) -> dict:
    return {
        "model": base_dir / f"model_{release}_{side}.keras",
        "calibrator": base_dir / f"oof_calibrator_{release}_{side}.joblib",
        "percentiles": base_dir / f"percentiles_{release}_{side}.json",
        "oof_df": base_dir / f"oof_{release}_{side}.parquet",
        "oof_meta": base_dir / f"oof_meta_{release}_{side}.json",
    }


def _has_required_artifacts(base_dir: Path, release: str, side: str) -> bool:
    paths = _artifact_paths_for_side(base_dir, release, side)
    return all(paths[k].exists() for k in ["model", "calibrator", "percentiles"])


def _describe_missing_artifacts(base_dir: Path, release: str, side: str) -> list[str]:
    paths = _artifact_paths_for_side(base_dir, release, side)
    return [str(v) for k, v in paths.items() if k in {"model", "calibrator", "percentiles"} and not v.exists()]


def resolve_train_artifacts_dir(
    base_dir: Path,
    release: str,
    requested_sides: list[str],
    *,
    explicit_dir: str | None = None,
    explicit_subdir: str | None = None,
) -> Path:
    """
    Resuelve la carpeta fuente de artefactos de entrenamiento.

    Orden:
      1) --train-artifacts-dir
      2) --train-artifacts-subdir
      3) autodetección: train_only, luego carpetas rw_* por mtime desc
    """
    candidates: list[Path] = []

    if explicit_dir:
        candidates.append(Path(explicit_dir).expanduser())
    elif explicit_subdir and explicit_subdir != "auto":
        candidates.append(base_dir / explicit_subdir)
    else:
        train_only = base_dir / "train_only"
        if train_only.exists():
            candidates.append(train_only)
        rw_dirs = sorted(
            [p for p in base_dir.glob("rw_*") if p.is_dir()],
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        candidates.extend(rw_dirs)

    checked = []
    for candidate in candidates:
        checked.append(str(candidate))
        if not candidate.exists() or not candidate.is_dir():
            continue
        if all(_has_required_artifacts(candidate, release, side) for side in requested_sides):
            print(f"✅ Carpeta fuente de artefactos resuelta: {candidate}")
            return candidate

    detail = []
    for candidate in candidates:
        side_info = {}
        for side in requested_sides:
            side_info[side] = _describe_missing_artifacts(candidate, release, side)
        detail.append({"candidate": str(candidate), "missing": side_info})

    raise FileNotFoundError(
        "No se encontró ninguna carpeta de artefactos válida para los lados solicitados. "
        f"release={release}, sides={requested_sides}, candidates={checked}, detail={detail}"
    )


# ─────────────────────────────────────────────────────────────────────────────
# Caché OOF / anti-OOM
# ─────────────────────────────────────────────────────────────────────────────

def verify_optuna_cache(artifacts_dir: Path, study_prefix: str, release: str, side: str) -> bool:
    """
    Verifica que el caché OOF del mejor trial existe en disco.
    Necesario para reuse_best_trial_oof=True funcione sin regenerar OOF completo.
    """
    cache_base = artifacts_dir / "_optuna_cache"
    study_name = f"{study_prefix}_{release}_{side}"
    side_dir = cache_base / study_name / side

    if not side_dir.exists():
        print(f"  ⚠️  Caché OOF no encontrado: {side_dir}")
        return False

    required = {"oof_df.parquet", "calibrator.joblib", "percentiles.json"}
    trial_dirs = sorted(side_dir.glob("trial_*"))

    for trial_dir in trial_dirs:
        found = {f.name for f in trial_dir.iterdir() if f.is_file()}
        if required.issubset(found):
            print(f"  ✅ Caché OOF encontrado: {trial_dir.name} ({side.upper()})")
            return True

    print(f"  ⚠️  Ningún trial en caché tiene los 3 ficheros requeridos ({side.upper()})")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Split final train/calibration
# ─────────────────────────────────────────────────────────────────────────────

def split_deploy_train_and_calib(df_rates: pd.DataFrame, calib_days: int = DEPLOY_CALIB_DAYS):
    df = df_rates.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    calib_from = df["time"].max() - pd.Timedelta(days=calib_days)
    df_train_final = df[df["time"] < calib_from].copy()
    df_calib_final = df[df["time"] >= calib_from].copy()

    if len(df_train_final) == 0 or len(df_calib_final) == 0:
        raise ValueError("Split deploy/calibración inválido")

    print(
        f"\n🧪 Cola final de calibración desde {calib_from} | "
        f"train_final={len(df_train_final):,} | calib_final={len(df_calib_final):,}"
    )
    return df_train_final, df_calib_final, calib_from


# ─────────────────────────────────────────────────────────────────────────────
# Entrenamiento deploy
# ─────────────────────────────────────────────────────────────────────────────

def train_single_side(trainer, df_rates, side):
    print(f"\n[DEPLOY] Preparing production {side.upper()} model based on FULL-minus-tail period")
    artifacts = trainer.prepare_production_model(
        df_rates=df_rates,
        side=side,
        reuse_best_trial_oof=True,
    )
    free_memory()
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Percentiles por estado sobre cola recalibrada
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PERCENTILE_QUANTILES = [50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
DEFAULT_NO_TRADE_STATES = ["LOW_VOL"]


def _safe_float(x):
    try:
        return float(x)
    except Exception:
        return None


def _load_existing_percentile_meta(src_pct: Path):
    meta = {}
    no_trade_states = list(DEFAULT_NO_TRADE_STATES)
    quantiles = list(DEFAULT_PERCENTILE_QUANTILES)
    existing = {}

    if src_pct.exists():
        try:
            with open(src_pct) as f:
                existing = json.load(f)
            meta = dict(existing.get("_meta", {}))
            nts = meta.get("no_trade_states")
            if isinstance(nts, list) and nts:
                no_trade_states = [str(s) for s in nts]
            qs = meta.get("quantiles")
            if isinstance(qs, list) and qs:
                quantiles = [int(q) for q in qs]
        except Exception as ex:
            print(f"  ⚠️  No se pudo leer metadata previa de percentiles: {ex}")

    return existing, meta, no_trade_states, quantiles


def recompute_percentiles_from_tail(
    df_tail: pd.DataFrame,
    selected_threshold: float,
    src_pct: Path,
    *,
    proba_col: str = "oof_proba_cal",
    regime_col: str = "state",
):
    existing, meta_prev, no_trade_states, quantiles = _load_existing_percentile_meta(src_pct)

    if regime_col not in df_tail.columns:
        raise ValueError(f"Columna de régimen '{regime_col}' no presente en df_tail")
    if proba_col not in df_tail.columns:
        raise ValueError(f"Columna de probabilidades '{proba_col}' no presente en df_tail")

    work = df_tail[[regime_col, proba_col]].copy()
    work = work.dropna(subset=[regime_col, proba_col])
    if work.empty:
        raise ValueError("Sin datos válidos para recomputar percentiles por estado")

    result = {}
    states_found = sorted([str(s) for s in work[regime_col].astype(str).unique().tolist()])

    for state in states_found:
        vals = work.loc[work[regime_col].astype(str) == state, proba_col].to_numpy(dtype=float)
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            continue
        pct_dict = {f"p{q}": float(np.percentile(vals, q)) for q in quantiles}
        state_payload = {
            "percentiles": pct_dict,
            "n": int(len(vals)),
        }
        if state in no_trade_states:
            state_payload["no_trade"] = True
        result[state] = state_payload

    global_vals = work.loc[~work[regime_col].astype(str).isin(no_trade_states), proba_col].to_numpy(dtype=float)
    global_vals = global_vals[np.isfinite(global_vals)]
    if len(global_vals) > 0:
        result["_global"] = {
            "percentiles": {f"p{q}": float(np.percentile(global_vals, q)) for q in quantiles},
            "n": int(len(global_vals)),
            "note": "Excludes NO_TRADE_STATES",
        }

    meta = dict(meta_prev)
    meta["proba_col"] = proba_col
    meta["regime_col"] = regime_col
    meta["quantiles"] = quantiles
    meta["no_trade_states"] = no_trade_states
    meta["states_found"] = states_found
    meta["selected_threshold"] = float(selected_threshold)
    result["_meta"] = meta

    return result


# ─────────────────────────────────────────────────────────────────────────────
# Recalibración final del deploy
# ─────────────────────────────────────────────────────────────────────────────

def recalibrate_deploy_side(
    artifacts,
    deploy_dir: Path,
    release: str,
    side: str,
    df_calib: pd.DataFrame,
    general_config,
    feature_config,
    model_config,
    regime_config,
):
    print(f"\n[RECAL] Recalibrando {side.upper()} sobre cola final reciente...")

    model = tf.keras.models.load_model(artifacts.model_path)

    pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)
    scalers_used = _load_pipeline_scalers_for_side(pipeline, artifacts.model_path, side)
    print(f"  · scalers: {scalers_used}")

    df_calib_prep = pipeline.prepare_data(df_calib.copy(), labels=True, side=side)
    sequences = pipeline.create_sequences_by_side(
        df_calib_prep, sides=(side,), fit_scalers=False, train=True
    )

    X_calib = [
        sequences[side]["seq_short"],
        sequences[side]["seq_long"],
        sequences[side]["context"],
        sequences[side]["time"],
    ]
    y_true = np.asarray(sequences[side]["labels"]).astype(int)

    if len(y_true) == 0:
        raise ValueError(f"[{side}] Sin muestras válidas para recalibrar deploy")
    if len(np.unique(y_true)) < 2:
        raise ValueError(f"[{side}] Cola final sin ambas clases; no se puede recalibrar de forma fiable")

    y_pred_raw = model.predict(X_calib, verbose=0).reshape(-1)

    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_raw, y_true)
    y_pred_cal = calibrator.predict(y_pred_raw)

    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_cal)
    if len(thresholds) > 0:
        f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
        selected_threshold = float(thresholds[np.argmax(f1_scores)])
    else:
        selected_threshold = 0.5

    # Guardar calibrador NUEVO en deploy_dir
    cal_out = deploy_dir / f"oof_calibrator_{release}_{side}.joblib"
    joblib.dump(calibrator, cal_out)
    artifacts.calibrator_path = str(cal_out)

    # Dataset para análisis posterior y percentiles por estado
    data_out = deploy_dir / "data" / f"deploy_calibration_tail_{release}_{side}.parquet"
    data_out.parent.mkdir(parents=True, exist_ok=True)

    seq_len = len(y_true)
    df_tail = df_calib_prep.iloc[-seq_len:].copy()
    keep_cols = [c for c in ["time", "state", "signal"] if c in df_tail.columns]
    df_tail = df_tail[keep_cols]
    df_tail["oof_proba_raw"] = y_pred_raw
    df_tail["oof_proba_cal"] = y_pred_cal
    df_tail["side"] = side
    df_tail["source"] = "deploy_calib_tail"
    df_tail.to_parquet(data_out, index=False)

    # Recomputar percentiles COMPLETOS por estado sobre esta misma cola recalibrada
    pct_out = deploy_dir / f"percentiles_{release}_{side}.json"
    src_pct = Path(artifacts.percentiles_path)
    pct = recompute_percentiles_from_tail(
        df_tail,
        selected_threshold=selected_threshold,
        src_pct=src_pct,
        proba_col="oof_proba_cal",
        regime_col="state",
    )
    pct.setdefault("_meta", {})
    pct["_meta"]["threshold_source"] = "deploy_calibration_tail"
    pct["_meta"]["deploy_calib_rows"] = int(len(y_true))
    pct["_meta"]["deploy_calib_from"] = str(pd.to_datetime(df_tail["time"]).min()) if "time" in df_tail.columns else None
    pct["_meta"]["deploy_calib_to"] = str(pd.to_datetime(df_tail["time"]).max()) if "time" in df_tail.columns else None

    with open(pct_out, "w") as f:
        json.dump(pct, f, indent=2, cls=NumpyEncoder)
    artifacts.percentiles_path = str(pct_out)

    print(f"  ✅ Nuevo calibrador guardado: {cal_out}")
    print(f"  ✅ Threshold final: {selected_threshold:.4f}")
    print(f"  ✅ Percentiles por estado recomputados: {pct_out}")
    print(f"  ✅ Dataset calibración final: {data_out}")

    free_memory()
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Generación automática de policy a partir de percentiles deploy
# ─────────────────────────────────────────────────────────────────────────────

POLICY_PERCENTILE_RULES = {
    "long": {
        "LOW_VOL": {"mode": "no_trade"},
        "VOLATILE": {"mode": "no_trade"},
        "TREND_DOWN": {"mode": "no_trade"},
        "TREND_UP": {"quantile": "p85"},
        "RANGE": {"quantile": "p80"},
        "TRANSITION_DOWN": {"quantile": "p90"},
        "TRANSITION_UP": {"quantile": "p85"},
        "BREAKOUT_WAIT_UP": {"quantile": "p80"},
        "BREAKOUT_WAIT_DOWN": {"quantile": "p90"},
        "_global": {"quantile": "p85"},
    },
    "short": {
        "LOW_VOL": {"mode": "no_trade"},
        "VOLATILE": {"mode": "no_trade"},
        "TREND_DOWN": {"quantile": "p80"},
        "TREND_UP": {"quantile": "p90"},
        "RANGE": {"quantile": "p80"},
        "TRANSITION_DOWN": {"quantile": "p85"},
        "TRANSITION_UP": {"quantile": "p90"},
        "BREAKOUT_WAIT_DOWN": {"quantile": "p85"},
        "BREAKOUT_WAIT_UP": {"mode": "no_trade"},
        "_global": {"quantile": "p85"},
    },
}


def _load_percentiles_json(path: Path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def _resolve_policy_threshold_for_state(side: str, state: str, payload: dict, selected_threshold: float):
    rules = POLICY_PERCENTILE_RULES.get(side, {})
    rule = rules.get(state, rules.get("_global", {"quantile": "p85"}))
    pcts = payload.get("percentiles", {}) or {}
    n = int(payload.get("n", 0) or 0)

    if payload.get("no_trade") or rule.get("mode") == "no_trade":
        return {
            "state": state,
            "mode": "no_trade",
            "threshold": None,
            "rule": rule,
            "n": n,
            "fallback_used": False,
        }

    q = str(rule.get("quantile", "p85"))
    thr = pcts.get(q)
    fallback_used = False
    if thr is None:
        fallback_used = True
        thr = selected_threshold

    if n < 50:
        fallback_used = True
        thr = max(float(thr), float(selected_threshold))

    return {
        "state": state,
        "mode": "threshold",
        "threshold": float(max(float(thr), float(selected_threshold))),
        "rule": rule,
        "n": n,
        "fallback_used": fallback_used,
    }


def build_policy_for_side(percentiles: dict, side: str):
    if not percentiles:
        return None

    meta = dict(percentiles.get("_meta", {}))
    selected_threshold = float(meta.get("selected_threshold", 0.5) or 0.5)
    no_trade_states = set(str(s) for s in meta.get("no_trade_states", []) if s)
    states = sorted([s for s in percentiles.keys() if not str(s).startswith("_")])

    gate = {}
    state_meta = {}
    for state in states:
        payload = percentiles.get(state, {}) or {}
        resolved = _resolve_policy_threshold_for_state(side, state, payload, selected_threshold)
        state_meta[state] = resolved
        if resolved["mode"] == "no_trade":
            gate[state] = None
            no_trade_states.add(state)
        else:
            gate[state] = resolved["threshold"]

    global_payload = percentiles.get("_global", {}) or {}
    resolved_global = _resolve_policy_threshold_for_state(side, "_global", global_payload, selected_threshold)
    global_threshold = selected_threshold if resolved_global["mode"] == "no_trade" else float(resolved_global["threshold"])

    return {
        "selected_threshold": selected_threshold,
        "default_threshold": global_threshold,
        "gate_by_action_and_state": gate,
        "no_trade_states": sorted(no_trade_states),
        "states_meta": state_meta,
        "percentiles_meta": meta,
    }


def save_generated_policy_files(deploy_dir: Path, release: str, policy_bundle: dict):
    deploy_dir.mkdir(parents=True, exist_ok=True)
    json_path = deploy_dir / f"decision_policy_{release}_auto.json"
    py_path = deploy_dir / f"decision_policies_config_{release}_auto.py"

    with open(json_path, "w") as f:
        json.dump(policy_bundle, f, indent=2, cls=NumpyEncoder)

    py_lines = [
        "# -*- coding: utf-8 -*-",
        '"""',
        f"Auto-generated deploy policy for release {release}.",
        "Generated by resume_deploy_full_v6.py from deploy percentiles/calibration tail.",
        "Do not edit manually unless you want to break reproducibility.",
        '"""',
        "",
        f"RELEASE = {release!r}",
        f"GENERATED_AT = {policy_bundle.get('generated_at')!r}",
        "",
        f"DECISION_POLICY_AUTO = {json.dumps(policy_bundle, indent=2, ensure_ascii=False)}",
        "",
        "DEFAULT_THRESHOLDS = {",
        "    side: DECISION_POLICY_AUTO['sides'][side]['default_threshold']",
        "    for side in DECISION_POLICY_AUTO['sides']",
        "}",
        "",
        "GATE_BY_ACTION_AND_STATE = {",
        "    side: DECISION_POLICY_AUTO['sides'][side]['gate_by_action_and_state']",
        "    for side in DECISION_POLICY_AUTO['sides']",
        "}",
        "",
        "NO_TRADE_STATES = {",
        "    side: DECISION_POLICY_AUTO['sides'][side]['no_trade_states']",
        "    for side in DECISION_POLICY_AUTO['sides']",
        "}",
        "",
    ]
    py_path.write_text("\n".join(py_lines), encoding="utf-8")
    return {"json": str(json_path), "python": str(py_path)}


def generate_policy_from_deploy_percentiles(deploy_dir: Path, release: str):
    long_path = deploy_dir / f"percentiles_{release}_long.json"
    short_path = deploy_dir / f"percentiles_{release}_short.json"

    long_pct = _load_percentiles_json(long_path)
    short_pct = _load_percentiles_json(short_path)

    sides = {}
    if long_pct is not None:
        sides["long"] = build_policy_for_side(long_pct, "long")
    if short_pct is not None:
        sides["short"] = build_policy_for_side(short_pct, "short")

    if not sides:
        print("ℹ️  No hay percentiles deploy para generar policy automática.")
        return None

    bundle = {
        "release": release,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "resume_deploy_full_v6",
        "notes": [
            "Thresholds generated deterministically from deploy percentiles recalculated on calibration tail.",
            "State rules are intentionally more conservative than selected_threshold.",
            "States marked as no_trade are emitted as null thresholds in gate_by_action_and_state.",
        ],
        "rules": POLICY_PERCENTILE_RULES,
        "sides": sides,
    }

    paths = save_generated_policy_files(deploy_dir, release, bundle)
    print(f"✅ Policy auto guardada (json): {paths['json']}")
    print(f"✅ Policy auto guardada (python): {paths['python']}")
    return paths

# ─────────────────────────────────────────────────────────────────────────────
# Persistencia de decisiones holdout / sequence summary
# ─────────────────────────────────────────────────────────────────────────────

def _selected_mode_label(selected_policy: str) -> str:
    return "walkforward" if selected_policy == "transform_then_update" else "static"


def save_holdout_policy_decisions(deploy_dir: Path, release: str, decisions: dict, *, holdout_report: dict | None = None):
    """
    Persiste la decisión canónica del holdout por side.

    Archivos generados:
      - data/inference_policy_<release>_<side>.json
      - reports/holdout_policy_decisions_<release>.json
    """
    if not decisions:
        return {}

    data_dir = deploy_dir / "data"
    reports_dir = deploy_dir / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    bundle = {
        "release": release,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "resume_deploy_full_v5",
        "sides": {},
    }

    side_paths = {}
    for side, decision in decisions.items():
        if not decision:
            continue
        payload = dict(decision)
        payload["selected_mode_label"] = _selected_mode_label(payload["selected_policy"])
        payload["selected_report_key"] = (
            "walkforward" if payload["selected_policy"] == "transform_then_update" else "static"
        )
        if holdout_report and side in holdout_report:
            payload["holdout_metrics"] = {
                "static": holdout_report[side].get("static", {}).get("metrics", {}),
                "walkforward": holdout_report[side].get("walkforward", {}).get("metrics", {}),
            }

        side_path = data_dir / f"inference_policy_{release}_{side}.json"
        with open(side_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

        bundle["sides"][side] = payload
        side_paths[side] = str(side_path)

    bundle_path = reports_dir / f"holdout_policy_decisions_{release}.json"
    with open(bundle_path, "w", encoding="utf-8") as f:
        json.dump(bundle, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"✅ Decisiones de holdout guardadas: {bundle_path}")
    for side, path in side_paths.items():
        print(f"   · {side.upper()}: {path}")

    return {"bundle_path": str(bundle_path), "side_paths": side_paths, "bundle": bundle}


def load_holdout_policy_decisions(deploy_dir: Path, release: str) -> dict:
    path = deploy_dir / "reports" / f"holdout_policy_decisions_{release}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception as ex:
        print(f"⚠️  No se pudo cargar holdout_policy_decisions: {ex}")
    return {}


def save_deploy_sequence_summary(
    deploy_dir: Path,
    release: str,
    *,
    holdout_policy_info: dict | None = None,
    policy_paths: dict | None = None,
    validation_results: dict | None = None,
):
    summary = {
        "release": release,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "source": "resume_deploy_full_v5",
        "holdout_policy": holdout_policy_info or {},
        "generated_policy_paths": policy_paths or {},
        "validation_status": None,
    }

    if validation_results:
        summary["validation_status"] = (
            validation_results.get("pipeline_validation", {})
            .get("summary", {})
            .get("status")
        )

    out_path = deploy_dir / "reports" / f"deploy_sequence_summary_{release}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2, cls=NumpyEncoder)

    print(f"✅ Resumen de secuencia deploy guardado: {out_path}")
    return str(out_path)


# ─────────────────────────────────────────────────────────────────────────────
# Carga deploy artifacts existentes
# ─────────────────────────────────────────────────────────────────────────────

def compute_holdout_for_side(trainer, artifacts_train, df_hold, deploy_dir: Path, release: str, side: str):
    side_report = {
        "static": trainer.evaluate_holdout(artifacts_train, df_hold, side=side),
        "walkforward": trainer.evaluate_holdout_walkforward_fast(
            artifacts_train,
            df_hold,
            side=side,
            inference_batch_size=64,
            return_predictions=True,
        ),
    }

    holdout_preds_path = deploy_dir / "data" / f"holdout_predictions_{release}_{side}.parquet"
    Helper.save_holdout_predictions(side_report, holdout_preds_path, side=side)

    if artifacts_train.oof_df_path and Path(artifacts_train.oof_df_path).exists():
        calibration_path = deploy_dir / "data" / f"calibration_dataset_{release}_{side}.parquet"
        build_calibration_dataset(
            oof_path=Path(artifacts_train.oof_df_path),
            holdout_preds_path=holdout_preds_path,
            out_path=calibration_path,
            side=side,
        )
    else:
        print(f"  ⚠️  [{side.upper()}] oof_df_path no disponible; se omite calibration_dataset")

    decision = trainer.choose_inference_policy(side_report["static"], side_report["walkforward"])
    print(f"{side.upper()} policy:", decision["selected_policy"])
    print(f"{side.upper()} mode  :", _selected_mode_label(decision["selected_policy"]))
    print(f"{side.upper()} reason:", decision["reason"])

    free_memory()
    return side_report, decision


def load_deploy_artifacts_or_fail(deploy_dir: Path, release: str, requested_sides: list[str]):
    print('\n📦 Cargando artifacts deploy_full existentes...')
    return {side: load_artifacts_from_disk(deploy_dir, release, side) for side in requested_sides}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recupera deploy_full evitando OOM, alineado con main_oof/main_oof_regime_weights, "
            "y añade recalibración final + percentiles por estado + generación automática de policy."
        )
    )
    parser.add_argument("--release", default="200383", help="Release objetivo.")
    parser.add_argument(
        "--side",
        choices=["long", "short", "both"],
        default="both",
        help="Lado a procesar. Usa long o short para lanzarlos en procesos separados.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Omite la validación final. Recomendado cuando se ejecuta un único lado.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="No entrena. Carga los artifacts deploy_full ya existentes y valida si están ambos lados disponibles.",
    )
    parser.add_argument(
        "--deploy-calib-days",
        type=int,
        default=DEPLOY_CALIB_DAYS,
        help=f"Número de días reservados al final para recalibración deploy (default: {DEPLOY_CALIB_DAYS}).",
    )
    parser.add_argument(
        "--train-artifacts-subdir",
        default="auto",
        help=(
            "Subcarpeta bajo ../../artifacts/<release>/oof desde la que cargar los artifacts de entrenamiento. "
            "Ejemplo: train_only o rw_both_Lauto_soft_h10_Sauto_soft_h10. Por defecto: auto."
        ),
    )
    parser.add_argument(
        "--train-artifacts-dir",
        default=None,
        help="Ruta completa a la carpeta fuente de artifacts de entrenamiento. Tiene prioridad sobre --train-artifacts-subdir.",
    )
    parser.add_argument(
        "--deploy-subdir",
        default="deploy_full",
        help="Subcarpeta bajo ../../artifacts/<release>/oof donde escribir el deploy final.",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    start_time = time.perf_counter()
    requested_sides = _side_list_from_arg(args.side)

    print(
        f"\n🚀 Modo de ejecución: release={args.release} | side={args.side} | "
        f"skip_validation={args.skip_validation} | validate_only={args.validate_only} | "
        f"deploy_calib_days={args.deploy_calib_days} | train_artifacts_subdir={args.train_artifacts_subdir}"
    )

    # ── Configuración alineada con main_oof_regime_weights_v3 ───────────────
    general = Config(
        release=args.release,
        use_oof=True,
        oof_splits=5,
        oof_epochs=90,
        save_oof_artifacts=True,
    )

    optuna_from = datetime(2025, 1, 1)
    optuna_to = datetime(2026, 1, 31)
    holdout_from = datetime(2026, 2, 1)
    holdout_to = datetime(2026, 4, 10)

    base_dir = Path("../../artifacts") / general.release / "oof"
    deploy_dir = base_dir / args.deploy_subdir

    source_artifacts_dir = None
    if not args.validate_only:
        source_artifacts_dir = resolve_train_artifacts_dir(
            base_dir,
            general.release,
            requested_sides,
            explicit_dir=args.train_artifacts_dir,
            explicit_subdir=args.train_artifacts_subdir,
        )

    print('\n📂 Cargando datos...')
    db = Database()
    dm = DataManager.from_database_historical_2(db, from_date=optuna_from, to_date=holdout_to)
    df_rates = dm.df

    if not pd.to_datetime(df_rates["time"]).is_monotonic_increasing:
        print("⚠️  Ordenando datos...")
        df_rates = df_rates.sort_values("time").reset_index(drop=True)
        n_before = len(df_rates)
        df_rates = df_rates.drop_duplicates(subset="time", keep="first")
        n_after = len(df_rates)
        if n_before != n_after:
            print(f"   Eliminados {n_before - n_after} timestamps duplicados")

    print(f"✅ Datos cargados: {len(df_rates):,} filas")
    print(f"   Rango: {df_rates['time'].iloc[0]} → {df_rates['time'].iloc[-1]}")

    df_train = df_rates[df_rates.time < holdout_from].copy()
    df_hold = df_rates[df_rates.time >= holdout_from].copy()
    print(f"   Train: {len(df_train):,} filas | Holdout: {len(df_hold):,} filas")

    trainer_out_dir = str(source_artifacts_dir if source_artifacts_dir is not None else deploy_dir)

    print('\n🔧 Reconstruyendo trainer desde Optuna DB...')
    trainer = OptunaOOFTrainer(
        general_config=general,
        feature_config=FeatureConfig(
            ema_periods=[9, 21, 50],
            label_method="triple_barrier",
            label_horizon=10,
            tp_barrier=2.5,
            sl_barrier=1.5,
            label_method_long="triple_barrier",
            regime_barriers_long={
                "trending": {"tp": 3.5, "sl": 1.25},
                "ranging": {"tp": 2.25, "sl": 1.25},
                "low_vol": {"tp": 2.75, "sl": 1.00},
                "high_vol": {"tp": 3.50, "sl": 2.00},
            },
            label_method_short="triple_barrier",
            regime_barriers_short={
                "trending": {"tp": 3.0, "sl": 1.25},
                "ranging": {"tp": 2.25, "sl": 1.25},
                "low_vol": {"tp": 2.50, "sl": 1.00},
                "high_vol": {"tp": 3.25, "sl": 2.00},
            },
            tp_barrier_short=None,
            sl_barrier_short=None,
            feature_masks={
                "long": {"ema_bull": True, "rsi_oversold": True, "macd_positive": True},
                "short": {"ema_bear": True, "rsi_overbought": True, "macd_negative": True},
            },
        ),
        regime_config=StateConfig(adx_trend_threshold=25.0),
        base_model_config=ModelConfig(
            seq_len_short=64,
            seq_len_long=256,
            epochs=90,
            patience=12,
            use_hierarchical_fusion=True,
            ranking_loss_weight=0.2,
        ),
        out_dir=trainer_out_dir,
        optuna_db="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
        study_prefix="oof_study",
        seed=42,
        reload=True,
        temperature_long=1.0,
        temperature_short=1.0,
    )
    print("✅ Trainer reconstruido")

    release = general.release
    final_validation_results = None
    holdout_policy_info = {}
    generated_policy_paths = {}

    if args.validate_only:
        del df_train, df_hold
        free_memory()

        deploy_artifacts = load_deploy_artifacts_or_fail(deploy_dir, release, requested_sides)
        holdout_policy_info = load_holdout_policy_decisions(deploy_dir, release)
        if holdout_policy_info.get("sides"):
            print('\n📘 Decisiones de holdout recuperadas:')
            for side_name, payload in holdout_policy_info["sides"].items():
                print(f"   · {side_name.upper()}: {payload.get('selected_policy')} ({payload.get('selected_mode_label')})")

        generated_policy_paths = generate_policy_from_deploy_percentiles(deploy_dir, release) or {}

        if args.side == "both":
            print('\n🏁 Iniciando validación SOLO de modelos deploy existentes...')
            final_validation_results = validate_after_training_with_metrics(
                artifacts_long=deploy_artifacts["long"],
                artifacts_short=deploy_artifacts["short"],
                df_full=df_rates,
                out_dir=str(deploy_dir),
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )
        else:
            print('\nℹ️  validate-only con un solo lado: se cargan artifacts y se omite la validación cruzada final.')

        save_deploy_sequence_summary(
            deploy_dir,
            release,
            holdout_policy_info=holdout_policy_info,
            policy_paths=generated_policy_paths,
            validation_results=final_validation_results,
        )

    else:
        print(f"\n📦 Reconstruyendo artifacts desde disco ({source_artifacts_dir.name})...")
        train_artifacts = {
            side: load_artifacts_from_disk(source_artifacts_dir, release, side)
            for side in requested_sides
        }

        print('\n🔍 Verificando caché OOF del mejor trial...')
        cache_results = {
            side: verify_optuna_cache(source_artifacts_dir, "oof_study", release, side)
            for side in requested_sides
        }
        if not all(cache_results.values()):
            print('\n⚠️  ADVERTENCIA: El caché OOF de uno o más lados no está disponible.')
            print("   El trainer intentará copiar el OOF desde Optuna DB (user_attrs del best_trial).")
            print("   Si eso tampoco funciona, regenerará el OOF desde cero sobre df_rates completo.")
            print("   Esto puede causar OOM en VRAM. Considera reducir batch_size o usar más VRAM.\n")

        holdout_report_path = deploy_dir / "reports" / "holdout_report.json"
        holdout_policy_info = load_holdout_policy_decisions(deploy_dir, release)
        holdout_report = {}
        merged_decisions = {}

        if holdout_report_path.exists():
            print('\n✅ Holdout report ya existe en disco, se reutiliza cuando sea posible')
            with open(holdout_report_path, "r", encoding="utf-8") as f:
                holdout_report = json.load(f)

            if isinstance(holdout_report.get("_policy_decisions"), dict):
                merged_decisions.update(holdout_report.get("_policy_decisions", {}))

            if not merged_decisions and isinstance(holdout_policy_info, dict):
                merged_decisions.update(holdout_policy_info.get("sides", {}))

            for side in requested_sides:
                if side not in merged_decisions and isinstance(holdout_report.get(side), dict):
                    side_payload = holdout_report[side]
                    if "static" in side_payload and "walkforward" in side_payload:
                        merged_decisions[side] = trainer.choose_inference_policy(
                            side_payload["static"], side_payload["walkforward"]
                        )

        missing_holdout_sides = [side for side in requested_sides if side not in merged_decisions]

        if missing_holdout_sides:
            print(f"\n📊 Recalculando holdout para: {missing_holdout_sides}")
            for side in missing_holdout_sides:
                side_report, decision = compute_holdout_for_side(
                    trainer,
                    train_artifacts[side],
                    df_hold,
                    deploy_dir,
                    release,
                    side,
                )
                holdout_report[side] = side_report
                merged_decisions[side] = decision

            holdout_report["_policy_decisions"] = merged_decisions
            Helper.save_meta(holdout_report, str(holdout_report_path))
            print(f"✅ Holdout report guardado: {holdout_report_path}")

            holdout_policy_info = save_holdout_policy_decisions(
                deploy_dir,
                release,
                merged_decisions,
                holdout_report=holdout_report,
            )
        else:
            print('\n✅ Decisiones de holdout ya disponibles para los lados solicitados')
            holdout_policy_info = {
                "release": release,
                "source": "existing_holdout_policy",
                "sides": merged_decisions,
            }
            for side_name in requested_sides:
                payload = merged_decisions.get(side_name, {})
                print(f"   · {side_name.upper()}: {payload.get('selected_policy')} ({_selected_mode_label(payload.get('selected_policy')) if payload.get('selected_policy') else 'n/a'})")

        print('\n🧹 Limpiando memoria antes de deploy_full...')
        trainer.best_oof_df_by_side.clear()
        trainer.best_calibrator_by_side.clear()
        trainer.best_percentiles_by_side.clear()
        del df_train, df_hold
        free_memory()
        print("✅ Memoria liberada")

        df_deploy_train, df_deploy_calib, deploy_calib_from = split_deploy_train_and_calib(
            df_rates,
            calib_days=args.deploy_calib_days,
        )

        holdout_selected = {
            side_name: payload.get("selected_policy")
            for side_name, payload in (holdout_policy_info.get("sides", {}) if isinstance(holdout_policy_info, dict) else {}).items()
        }

        Helper.save_meta(
            {
                "release": general.release,
                "trained_on": "full_minus_tail",
                "from_date": optuna_from.isoformat(),
                "to_date": holdout_to.isoformat(),
                "deploy_calib_from": str(deploy_calib_from),
                "deploy_calib_days": int(args.deploy_calib_days),
                "train_artifacts_dir": str(source_artifacts_dir),
                "holdout_selected_policies": holdout_selected,
                "note": (
                    "trained on full-minus-tail; last tail reserved for final deploy calibration. "
                    "Old Optuna OOF calibrator NOT reused as final calibrator. "
                    "reuse_best_trial_oof=True only used to avoid OOM during model build."
                ),
            },
            str(deploy_dir / "data" / "train_meta.json"),
        )

        trainer.out_dir = str(deploy_dir)
        deploy_artifacts = {}

        if "long" in requested_sides:
            deploy_artifacts["long"] = train_single_side(trainer, df_deploy_train, "long")
            deploy_artifacts["long"] = recalibrate_deploy_side(
                artifacts=deploy_artifacts["long"],
                deploy_dir=deploy_dir,
                release=release,
                side="long",
                df_calib=df_deploy_calib,
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )

        if "short" in requested_sides:
            deploy_artifacts["short"] = train_single_side(trainer, df_deploy_train, "short")
            deploy_artifacts["short"] = recalibrate_deploy_side(
                artifacts=deploy_artifacts["short"],
                deploy_dir=deploy_dir,
                release=release,
                side="short",
                df_calib=df_deploy_calib,
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )

        generated_policy_paths = generate_policy_from_deploy_percentiles(deploy_dir, release) or {}

        should_validate = (
            not args.skip_validation
            and args.side == "both"
            and all(side in deploy_artifacts for side in ["long", "short"])
        )

        if should_validate:
            print("\n" + "🏁 " * 35)
            print("Iniciando validación de modelos finales (deploy)...")

            final_validation_results = validate_after_training_with_metrics(
                artifacts_long=deploy_artifacts["long"],
                artifacts_short=deploy_artifacts["short"],
                df_full=df_rates,
                out_dir=str(deploy_dir),
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )

            print("🏁 " * 35 + "\n")
        else:
            print('\nℹ️  Validación final omitida.')
            if args.side != "both":
                print("   Ejecuta la validación completa solo cuando ya existan ambos artifacts deploy (long y short).")

        save_deploy_sequence_summary(
            deploy_dir,
            release,
            holdout_policy_info=holdout_policy_info,
            policy_paths=generated_policy_paths,
            validation_results=final_validation_results,
        )

    end_time = time.perf_counter()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 23 + "RESUMEN FINAL" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n⏱️  Tiempo total: {end_time - start_time:.2f}s ({(end_time - start_time) / 60:.1f} min)")

    if final_validation_results:
        status = (
            "✅ PASS"
            if final_validation_results.get("pipeline_validation", {})
            .get("summary", {})
            .get("status")
            == "PASS"
            else "⚠️  WARNINGS"
        )
        print(f"🏁 Validación final (deploy): {status}")

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70 + "\n")
    print("OK")
