"""
resume_deploy_full_patched.py
─────────────────────────────────────────────────────────────────────────────
Versión completa de resume_deploy_full.py alineada con main_oof.py y con
recalibración final barata sobre una cola temporal reciente para evitar el
problema de desalineación del calibrador heredado del mejor trial OOF.

Objetivo:
  1. Mantener el flujo anti-OOM de deploy_full con reuse_best_trial_oof=True.
  2. Alinear la configuración con main_oof.py.
  3. Reemplazar el calibrador/threshold final del deploy por uno recalculado
     sobre una cola reciente del dataset completo, evitando reutilizar sin más
     el calibrador viejo de train_only.

Notas:
  - El modelo final se entrena sobre full-minus-tail.
  - La cola final (tail) se reserva solo para recalibración final.
  - Se sobrescriben en deploy_full:
      * oof_calibrator_<release>_<side>.joblib
      * percentiles_<release>_<side>.json   (al menos el threshold final)
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
from mimo.oof.optuna_oof_trainer import OptunaOOFTrainer, TrainerArtifacts
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

def load_artifacts_from_disk(train_dir: Path, release: str, side: str) -> TrainerArtifacts:
    """
    Reconstruye un TrainerArtifacts leyendo los ficheros ya guardados.
    No requiere re-entrenar nada.
    """
    model_path = str(train_dir / f"model_{release}_{side}.keras")
    cal_path = str(train_dir / f"oof_calibrator_{release}_{side}.joblib")
    pct_path = str(train_dir / f"percentiles_{release}_{side}.json")
    oof_df_path = str(train_dir / f"oof_{release}_{side}.parquet")
    oof_meta_path = str(train_dir / f"oof_meta_{release}_{side}.json")

    missing = [p for p in [model_path, cal_path, pct_path] if not os.path.exists(p)]
    if missing:
        raise FileNotFoundError(f"Faltan artefactos para side={side}: {missing}")

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

    print(f"  ✅ Artefactos {side.upper()} cargados desde disco")
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


# ─────────────────────────────────────────────────────────────────────────────
# Caché OOF / anti-OOM
# ─────────────────────────────────────────────────────────────────────────────

def verify_optuna_cache(train_dir: Path, study_prefix: str, release: str, side: str) -> bool:
    """
    Verifica que el caché OOF del mejor trial existe en disco.
    Necesario para reuse_best_trial_oof=True funcione sin regenerar OOF completo.
    """
    cache_base = train_dir / "_optuna_cache"
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

    # Copiar/actualizar percentiles deploy existentes
    pct_out = deploy_dir / f"percentiles_{release}_{side}.json"
    pct = {}
    src_pct = Path(artifacts.percentiles_path)
    if src_pct.exists():
        with open(src_pct) as f:
            pct = json.load(f)

    pct.setdefault("_meta", {})
    pct["_meta"]["selected_threshold"] = selected_threshold
    pct["_meta"]["threshold_source"] = "deploy_calibration_tail"
    pct["_meta"]["deploy_calib_rows"] = int(len(y_true))
    pct["_meta"]["deploy_calib_from"] = str(pd.to_datetime(df_calib["time"]).min())
    pct["_meta"]["deploy_calib_to"] = str(pd.to_datetime(df_calib["time"]).max())

    with open(pct_out, "w") as f:
        json.dump(pct, f, indent=2, cls=NumpyEncoder)
    artifacts.percentiles_path = str(pct_out)

    # Dataset para análisis posterior
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

    print(f"  ✅ Nuevo calibrador guardado: {cal_out}")
    print(f"  ✅ Threshold final: {selected_threshold:.4f}")
    print(f"  ✅ Dataset calibración final: {data_out}")

    free_memory()
    return artifacts


# ─────────────────────────────────────────────────────────────────────────────
# Carga deploy artifacts existentes
# ─────────────────────────────────────────────────────────────────────────────

def load_deploy_artifacts_or_fail(deploy_dir: Path, release: str):
    print("\n📦 Cargando artifacts deploy_full existentes para validación...")
    artifacts_long = load_artifacts_from_disk(deploy_dir, release, "long")
    artifacts_short = load_artifacts_from_disk(deploy_dir, release, "short")
    return artifacts_long, artifacts_short


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Recupera deploy_full evitando OOM, alineado con main_oof.py, "
            "y añade recalibración final sobre una cola reciente."
        )
    )
    parser.add_argument(
        "--side",
        choices=["long", "short", "both"],
        default="both",
        help="Lado a entrenar. Usa long o short para lanzarlos en procesos separados.",
    )
    parser.add_argument(
        "--skip-validation",
        action="store_true",
        help="Omite la validación final. Recomendado cuando se ejecuta un único lado.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="No entrena. Carga los artifacts deploy_full ya existentes (long y short) y ejecuta solo la validación final.",
    )
    parser.add_argument(
        "--deploy-calib-days",
        type=int,
        default=DEPLOY_CALIB_DAYS,
        help=f"Número de días reservados al final para recalibración deploy (default: {DEPLOY_CALIB_DAYS}).",
    )
    return parser.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    args = parse_args()
    start_time = time.perf_counter()
    print(
        f"\n🚀 Modo de ejecución: side={args.side} | "
        f"skip_validation={args.skip_validation} | validate_only={args.validate_only} | "
        f"deploy_calib_days={args.deploy_calib_days}"
    )

    # ── Configuración alineada con main_oof.py ───────────────────────────────
    general = Config(
        release="200377",
        use_oof=True,
        oof_splits=5,
        oof_epochs=60,
        save_oof_artifacts=True,
    )

    optuna_from = datetime(2025, 1, 1)
    optuna_to = datetime(2026, 1, 31)
    holdout_from = datetime(2026, 2, 1)
    holdout_to = datetime(2026, 4, 10)

    base_dir = Path("../../artifacts") / general.release / "oof"
    train_dir = base_dir / "train_only"
    deploy_dir = base_dir / "deploy_full"

    print("\n📂 Cargando datos...")
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

    print("\n🔧 Reconstruyendo trainer desde Optuna DB...")
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
                "trending": {"tp": 3.5, "sl": 1.5},
                "ranging": {"tp": 2.0, "sl": 1.5},
                "low_vol": {"tp": 2.5, "sl": 1.0},
                "high_vol": {"tp": 3.0, "sl": 2.0},
            },
            label_method_short="adaptive",
            regime_barriers_short={
                "trending": {"adaptive_q": 0.72},
                "ranging": {"adaptive_q": 0.65},
                "low_vol": {"adaptive_q": 0.68},
                "high_vol": {"adaptive_q": 0.75},
            },
            feature_masks={
                "long": {"ema_bull": True, "rsi_oversold": True, "macd_positive": True},
                "short": {"ema_bear": True, "rsi_overbought": True, "macd_negative": True},
            },
        ),
        regime_config=StateConfig(adx_trend_threshold=25.0),
        base_model_config=ModelConfig(
            seq_len_short=64,
            seq_len_long=256,
            epochs=60,
            patience=10,
            use_hierarchical_fusion=True,
            ranking_loss_weight=0.2,
        ),
        out_dir=str(train_dir),
        optuna_db="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
        study_prefix="oof_study",
        seed=42,
        reload=True,
        temperature_long=1.0,
        temperature_short=1.2,
    )
    print("✅ Trainer reconstruido")

    release = general.release
    final_validation_results = None

    if args.validate_only:
        del df_train, df_hold
        free_memory()
        artifacts_long_deploy, artifacts_short_deploy = load_deploy_artifacts_or_fail(deploy_dir, release)
        print("\n🏁 Iniciando validación SOLO de modelos deploy existentes...")
        final_validation_results = validate_after_training_with_metrics(
            artifacts_long=artifacts_long_deploy,
            artifacts_short=artifacts_short_deploy,
            df_full=df_rates,
            out_dir=str(deploy_dir),
            general_config=general,
            feature_config=trainer.feature_config,
            model_config=trainer.base_model_config,
            regime_config=trainer.regime_config,
        )
    else:
        print("\n📦 Reconstruyendo artifacts desde disco (train_only)...")
        artifacts_long_train = load_artifacts_from_disk(train_dir, release, "long")
        artifacts_short_train = load_artifacts_from_disk(train_dir, release, "short")

        print("\n🔍 Verificando caché OOF del mejor trial...")
        cache_ok_long = verify_optuna_cache(train_dir, "oof_study", release, "long")
        cache_ok_short = verify_optuna_cache(train_dir, "oof_study", release, "short")

        if not cache_ok_long or not cache_ok_short:
            print("\n⚠️  ADVERTENCIA: El caché OOF de uno o ambos lados no está disponible.")
            print("   El trainer intentará copiar el OOF desde Optuna DB (user_attrs del best_trial).")
            print("   Si eso tampoco funciona, regenerará el OOF desde cero sobre df_rates completo.")
            print("   Esto puede causar OOM en VRAM. Considera reducir batch_size o usar más VRAM.\n")

        holdout_report_path = deploy_dir / "reports" / "holdout_report.json"
        if holdout_report_path.exists():
            print("\n✅ Holdout report ya existe en disco, se omite recalcular")
            with open(holdout_report_path) as f:
                holdout_report = json.load(f)
        else:
            print("\n📊 Holdout report no encontrado, recalculando...")

            holdout_report_long = {
                "static": trainer.evaluate_holdout(artifacts_long_train, df_hold, side="long"),
                "walkforward": trainer.evaluate_holdout_walkforward_fast(
                    artifacts_long_train,
                    df_hold,
                    side="long",
                    inference_batch_size=64,
                    return_predictions=True,
                ),
            }
            holdout_preds_long_path = deploy_dir / "data" / f"holdout_predictions_{general.release}_long.parquet"
            Helper.save_holdout_predictions(holdout_report_long, holdout_preds_long_path, side="long")

            calibration_long_path = deploy_dir / "data" / f"calibration_dataset_{general.release}_long.parquet"
            oof_long_path = Path(artifacts_long_train.oof_df_path)
            build_calibration_dataset(
                oof_path=oof_long_path,
                holdout_preds_path=holdout_preds_long_path,
                out_path=calibration_long_path,
                side="long",
            )

            decision_long = trainer.choose_inference_policy(
                holdout_report_long["static"], holdout_report_long["walkforward"]
            )
            print("LONG policy:", decision_long["selected_policy"])
            print("LONG reason:", decision_long["reason"])

            free_memory()

            holdout_report_short = {
                "static": trainer.evaluate_holdout(artifacts_short_train, df_hold, side="short"),
                "walkforward": trainer.evaluate_holdout_walkforward_fast(
                    artifacts_short_train,
                    df_hold,
                    side="short",
                    inference_batch_size=64,
                    return_predictions=True,
                ),
            }
            holdout_preds_short_path = deploy_dir / "data" / f"holdout_predictions_{general.release}_short.parquet"
            Helper.save_holdout_predictions(holdout_report_short, holdout_preds_short_path, side="short")

            calibration_short_path = deploy_dir / "data" / f"calibration_dataset_{general.release}_short.parquet"
            oof_short_path = Path(artifacts_short_train.oof_df_path)
            build_calibration_dataset(
                oof_path=oof_short_path,
                holdout_preds_path=holdout_preds_short_path,
                out_path=calibration_short_path,
                side="short",
            )

            decision_short = trainer.choose_inference_policy(
                holdout_report_short["static"], holdout_report_short["walkforward"]
            )
            print("SHORT policy:", decision_short["selected_policy"])
            print("SHORT reason:", decision_short["reason"])

            free_memory()

            holdout_report = {"long": holdout_report_long, "short": holdout_report_short}
            Helper.save_meta(holdout_report, str(holdout_report_path))
            print(f"✅ Holdout report guardado: {holdout_report_path}")

        print("\n🧹 Limpiando memoria antes de deploy_full...")
        trainer.best_oof_df_by_side.clear()
        trainer.best_calibrator_by_side.clear()
        trainer.best_percentiles_by_side.clear()
        del df_train, df_hold
        free_memory()
        print("✅ Memoria liberada")

        # Split final: entrenar full-minus-tail y recalibrar con la cola reciente
        df_deploy_train, df_deploy_calib, deploy_calib_from = split_deploy_train_and_calib(
            df_rates,
            calib_days=args.deploy_calib_days,
        )

        Helper.save_meta(
            {
                "release": general.release,
                "trained_on": "full_minus_tail",
                "from_date": optuna_from.isoformat(),
                "to_date": holdout_to.isoformat(),
                "deploy_calib_from": str(deploy_calib_from),
                "deploy_calib_days": int(args.deploy_calib_days),
                "note": (
                    "trained on full-minus-tail; last tail reserved for final deploy calibration. "
                    "Old Optuna OOF calibrator NOT reused as final calibrator. "
                    "reuse_best_trial_oof=True only used to avoid OOM during model build."
                ),
            },
            str(deploy_dir / "data" / "train_meta.json"),
        )

        trainer.out_dir = str(deploy_dir)

        artifacts_long_deploy = None
        artifacts_short_deploy = None

        if args.side in ("long", "both"):
            artifacts_long_deploy = train_single_side(trainer, df_deploy_train, "long")
            artifacts_long_deploy = recalibrate_deploy_side(
                artifacts=artifacts_long_deploy,
                deploy_dir=deploy_dir,
                release=release,
                side="long",
                df_calib=df_deploy_calib,
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )
            release_artifact_refs((locals(), ["artifacts_long_train"]))

        if args.side in ("short", "both"):
            artifacts_short_deploy = train_single_side(trainer, df_deploy_train, "short")
            artifacts_short_deploy = recalibrate_deploy_side(
                artifacts=artifacts_short_deploy,
                deploy_dir=deploy_dir,
                release=release,
                side="short",
                df_calib=df_deploy_calib,
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )
            release_artifact_refs((locals(), ["artifacts_short_train"]))

        should_validate = (
            not args.skip_validation
            and args.side == "both"
            and artifacts_long_deploy is not None
            and artifacts_short_deploy is not None
        )

        if should_validate:
            print("\n" + "🏁 " * 35)
            print("Iniciando validación de modelos finales (deploy)...")

            final_validation_results = validate_after_training_with_metrics(
                artifacts_long=artifacts_long_deploy,
                artifacts_short=artifacts_short_deploy,
                df_full=df_rates,
                out_dir=str(deploy_dir),
                general_config=general,
                feature_config=trainer.feature_config,
                model_config=trainer.base_model_config,
                regime_config=trainer.regime_config,
            )

            print("🏁 " * 35 + "\n")
        else:
            print("\nℹ️  Validación final omitida.")
            if args.side != "both":
                print("   Ejecuta la validación solo cuando ya existan ambos artifacts deploy (long y short).")

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
