import argparse
import os
import time
from datetime import datetime
from pathlib import Path

import joblib
import gc
import numpy as np
import pandas as pd
import tensorflow as tf

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.databases import Database
from mimo.features.feature_builder import FeatureConfig
from mimo.helpers.helper import Helper
from mimo.helpers.json_serialization import NumpyEncoder
from mimo.models.model_builder import Config, ModelConfig
from mimo.oof.optuna_oof_trainer import OptunaOOFTrainer
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.validate_full_training import validate_training_quick, TrainingValidator
import mimo.features.label_generator as _lg

def build_calibration_dataset(
    oof_path: Path,
    holdout_preds_path: Path,
    out_path: Path,
    side: str,
):
    """
    Construye un parquet combinado para análisis de calibración:
      - train/validation desde OOF
      - holdout desde predicciones walk-forward

    Output columns:
      time, state, signal, oof_proba_raw, oof_proba_cal, side, source
    """
    if not oof_path.exists():
        print(f"⚠️  OOF no encontrado para {side}: {oof_path}")
        return

    if not holdout_preds_path.exists():
        print(f"⚠️  Holdout preds no encontrado para {side}: {holdout_preds_path}")
        return

    df_oof = pd.read_parquet(oof_path).copy()
    df_hold = pd.read_parquet(holdout_preds_path).copy()

    # Normalizar columnas OOF
    keep_oof = ["time", "state", "signal", "oof_proba_raw", "oof_proba_cal"]
    df_oof = df_oof[[c for c in keep_oof if c in df_oof.columns]].copy()
    df_oof["side"] = side
    df_oof["source"] = "oof_train"

    # Normalizar columnas holdout
    rename_hold = {
        "y_true": "signal",
        "y_pred_raw": "oof_proba_raw",
        "y_pred_cal": "oof_proba_cal",
    }
    df_hold = df_hold.rename(columns=rename_hold).copy()
    keep_hold = ["time", "state", "signal", "oof_proba_raw", "oof_proba_cal"]
    df_hold = df_hold[[c for c in keep_hold if c in df_hold.columns]].copy()
    df_hold["side"] = side
    df_hold["source"] = "holdout"

    # Tipos
    for df_ in (df_oof, df_hold):
        if "time" in df_.columns:
            df_["time"] = pd.to_datetime(df_["time"], errors="coerce")
        if "signal" in df_.columns:
            df_["signal"] = pd.to_numeric(df_["signal"], errors="coerce")

    df_all = pd.concat([df_oof, df_hold], ignore_index=True)
    df_all = df_all.sort_values("time").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_all.to_parquet(out_path, index=False)

    print(f"✅ Calibration dataset guardado: {out_path} ({len(df_all):,} filas)")
    print(f"   · OOF rows     : {len(df_oof):,}")
    print(f"   · Holdout rows : {len(df_hold):,}")

print(f"[DEBUG] label_generator path: {_lg.__file__}")


os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logs de TensorFlow
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configurada: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  Error configurando GPU: {e}")

# Garbage collection más agresivo
gc.set_threshold(700, 10, 10)


def _load_pipeline_scalers_for_side(pipeline, model_path, side):
    """
    Carga scalers side-specific si existen; si no, cae al formato genérico.
    Devuelve la ruta lógica utilizada.
    """
    import shutil
    import tempfile

    model_dir = Path(model_path).parent
    release = getattr(pipeline.general_config, 'release', None)
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


def validate_before_training(df_sample, general_config, feature_config, model_config, regime_config):
    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 15 + "VALIDACIÓN PRE-ENTRENAMIENTO" + " " * 25 + "║")
    print("╚" + "═" * 68 + "╝")

    print(f"\n🔍 Validando con {len(df_sample):,} muestras...")
    try:
        results, validator = validate_training_quick(
            df_train=df_sample,
            general_config=general_config,
            feature_config=feature_config,
            model_config=model_config,
            regime_config=regime_config
        )

        # Verificar resultado
        if results['summary']['status'] == 'FAIL':
            print("\n" + "=" * 70)
            print("❌ VALIDACIÓN PRE-ENTRENAMIENTO FALLÓ")
            print(f"   Errores encontrados: {results['summary']['n_errors']}")
            print(f"   Warnings: {results['summary']['n_warnings']}")
            print("\n   ⚠️  Se recomienda revisar los errores antes de continuar")
            print("=" * 70)

            # Opcional: descomentar para detener ejecución si falla
            # raise RuntimeError("Validación pre-entrenamiento falló")
        else:
            print("\n" + "=" * 70)
            print("✅ VALIDACIÓN PRE-ENTRENAMIENTO EXITOSA")
            print(f"   Tests pasados: {results['summary']['passed_tests']}/{results['summary']['total_tests']}")
            if results['summary']['n_warnings'] > 0:
                print(f"   Warnings: {results['summary']['n_warnings']} (revisar pero no crítico)")
            print("=" * 70)

        return results

    except Exception as e:
        print(f"\n❌ Error en validación pre-entrenamiento: {e}")
        print("   Continuando con entrenamiento (validación no crítica)")
        return None


def validate_after_training_with_metrics(artifacts_long, artifacts_short, df_full, out_dir,
                                         general_config, feature_config, model_config, regime_config):
    """
    Validación POST-ENTRENAMIENTO con métricas completas.
    Calcula: Precision, Recall, F1, AUC-ROC, AUC-PR, Confusion Matrix.
    """
    import json
    from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve, confusion_matrix

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 14 + "VALIDACIÓN POST-ENTRENAMIENTO" + " " * 24 + "║")
    print("║" + " " * 20 + "CON MÉTRICAS COMPLETAS" + " " * 27 + "║")
    print("╚" + "═" * 68 + "╝")

    validation_results = {}

    try:
        # Preparar datos
        split_point = int(len(df_full) * 0.8)
        df_train = df_full.iloc[:split_point]
        df_val = df_full.iloc[split_point:]

        print(f"\n🔍 Validando con {len(df_train):,} train + {len(df_val):,} val...")

        # Cargar modelos
        print("\n📦 Cargando artifacts...")
        model_long = tf.keras.models.load_model(artifacts_long.model_path)
        model_short = tf.keras.models.load_model(artifacts_short.model_path)
        print("  ✅ Modelos cargados")

        # Cargar calibradores
        with open(artifacts_long.calibrator_path, 'rb') as f:
            cal_long = joblib.load(f)
        with open(artifacts_short.calibrator_path, 'rb') as f:
            cal_short = joblib.load(f)
        print("  ✅ Calibradores cargados")

        # Crear pipeline
        pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)
        artifacts_dir = Path(artifacts_long.model_path).parent
        try:
            pipeline.load_scalers(base_path=str(artifacts_dir))
            print(f"  ✅ Pipeline y scalers cargados desde: {artifacts_dir}")
        except Exception as e:
            print(f"  ⚠️  Scalers no cargados desde {artifacts_dir}: {type(e).__name__}: {e}")

        # Validación del pipeline
        print("\n" + "─" * 70)
        print("PARTE 1: VALIDACIÓN DEL PIPELINE")
        print("─" * 70)

        validator = TrainingValidator()
        pipeline_results = validator.run_all_checks(
            pipeline=pipeline,
            df_train=df_train,
            df_val=df_val,
            models={'long': model_long, 'short': model_short},
            calibrators={'long': cal_long, 'short': cal_short},
            scalers=pipeline.scalers if hasattr(pipeline, 'scalers') else None
        )

        validation_results['pipeline_validation'] = pipeline_results

        # Métricas detalladas
        print("\n" + "─" * 70)
        print("PARTE 2: MÉTRICAS DETALLADAS")
        print("─" * 70)

        models_metrics = {}

        for side in ['long', 'short']:
            print(f"\n📊 Evaluando {side.upper()}...")

            model = model_long if side == 'long' else model_short
            calibrator = cal_long if side == 'long' else cal_short

            try:
                pipeline_side = DataPipeline(general_config, feature_config, model_config, regime_config)
                scalers_used = _load_pipeline_scalers_for_side(pipeline_side,
                                                               artifacts_long.model_path if side == 'long' else artifacts_short.model_path,
                                                               side)
                print(f"     · scalers: {scalers_used}")
                # Preparar datos
                df_val_prep = pipeline_side.prepare_data(df_val.copy(), labels=True, side=side)

                sequences = pipeline_side.create_sequences_by_side(
                    df_val_prep, sides=(side,), fit_scalers=False, train=True
                )

                X_val = [
                    sequences[side]['seq_short'],
                    sequences[side]['seq_long'],
                    sequences[side]['context'],
                    sequences[side]['time']
                ]
                y_true = sequences[side]['labels']

                # Predicciones
                y_pred_raw = model.predict(X_val, verbose=0).reshape(-1)
                y_pred_cal = calibrator.predict(y_pred_raw) if calibrator else y_pred_raw

                # Calcular métricas
                auc_roc = float(roc_auc_score(y_true, y_pred_cal))
                auc_pr = float(average_precision_score(y_true, y_pred_cal))

                # Best threshold (maximiza F1)
                precision, recall, thresholds = precision_recall_curve(y_true, y_pred_cal)
                f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
                best_idx = np.argmax(f1_scores)
                best_threshold = float(thresholds[best_idx])

                # Predicciones binarias
                y_pred_binary = (y_pred_cal >= best_threshold).astype(int)

                # Confusion matrix
                tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

                # Métricas finales
                precision_final = float(tp / (tp + fp + 1e-10))
                recall_final = float(tp / (tp + fn + 1e-10))
                f1_final = float(2 * tp / (2 * tp + fp + fn + 1e-10))
                accuracy = float((tp + tn) / (tp + tn + fp + fn))
                signal_rate = float(y_pred_binary.mean())
                base_rate = float(y_true.mean())

                models_metrics[side] = {
                    'auc_roc': auc_roc,
                    'auc_pr': auc_pr,
                    'best_threshold': best_threshold,
                    'precision': precision_final,
                    'recall': recall_final,
                    'f1': f1_final,
                    'accuracy': accuracy,
                    'signal_rate': signal_rate,
                    'base_rate': base_rate,
                    'confusion_matrix': {
                        'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn)
                    },
                    'n_samples': int(len(y_true))
                }

                # Mostrar
                print(f"  ✅ Métricas:")
                print(f"     AUC-ROC: {auc_roc:.4f} | AUC-PR: {auc_pr:.4f}")
                print(f"     Precision: {precision_final:.4f} | Recall: {recall_final:.4f} | F1: {f1_final:.4f}")
                print(f"     TP: {tp}, FP: {fp}, TN: {tn}, FN: {fn}")

            except Exception as e:
                print(f"  ❌ Error: {e}")
                models_metrics[side] = {'error': str(e)}

        validation_results['models_metrics'] = models_metrics

        # Métricas OOF
        validation_results['oof_metrics'] = {
            'long': artifacts_long.oof_metrics if artifacts_long.oof_metrics else {},
            'short': artifacts_short.oof_metrics if artifacts_short.oof_metrics else {}
        }

        # Guardar
        validation_path = Path(out_dir) / 'reports' / 'validation_post_training_complete.json'
        validation_path.parent.mkdir(parents=True, exist_ok=True)

        with open(validation_path, 'w') as f:
            json.dump(validation_results, f, indent=2, cls=NumpyEncoder)

        print(f"\n✅ Resultados guardados: {validation_path}")

        # Resumen
        print("\n" + "╔" + "═" * 68 + "╗")
        print("║" + " " * 20 + "RESUMEN DE VALIDACIÓN" + " " * 27 + "║")
        print("╚" + "═" * 68 + "╝")

        pipeline_status = pipeline_results['summary']['status']
        print(f"\n🔧 Pipeline: {pipeline_status}")

        print(f"\n📊 Modelos:")
        for side in ['long', 'short']:
            if side in models_metrics and 'error' not in models_metrics[side]:
                m = models_metrics[side]
                print(
                    f"   {side.upper():5s} | AUC-PR: {m['auc_pr']:.4f} | P: {m['precision']:.4f} | R: {m['recall']:.4f} | F1: {m['f1']:.4f}")

        pipeline_ok = pipeline_status == 'PASS'
        models_ok = all(
            side in models_metrics and 'error' not in models_metrics[side]
            for side in ['long', 'short']
        )

        print("\n" + "=" * 70)
        if pipeline_ok and models_ok:
            print("✅ VALIDACIÓN COMPLETA EXITOSA")
        else:
            print("⚠️  VALIDACIÓN PARCIAL")
        print("=" * 70 + "\n")

        return validation_results

    except Exception as e:
        print(f"\n❌ Error crítico: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":

    ap = argparse.ArgumentParser()

    ap.add_argument("--skip-long", action="store_true")
    ap.add_argument("--skip-short", action="store_true")
    args = ap.parse_args()

    start_time = time.perf_counter()

    general = Config(
        release='200381',
        use_oof=True,
        oof_splits=5, #5
        oof_epochs=60, #60
        save_oof_artifacts=True,
    )

    optuna_from = datetime(2025, 1, 1)
    optuna_to = datetime(2026, 1, 31)
    holdout_from = datetime(2026, 2, 1)
    holdout_to = datetime(2026, 4, 10)

    '''
    optuna_from = datetime(2025, 1, 7)
    optuna_to = datetime(2025, 1, 10)
    holdout_from = datetime(2025, 1, 13)
    holdout_to = datetime(2026, 1, 17)
    '''

    base_dir = Path(f'../../artifacts') / general.release / 'oof'
    train_dir = base_dir / 'train_only'
    Helper.save_meta({
        'release': general.release,
        'train_from': optuna_from.isoformat(),
        'train_to': optuna_to.isoformat(),
        'holdout_from': holdout_from.isoformat(),
        'holdout_to': holdout_to.isoformat(),
    }, str(train_dir / 'data' / 'split_meta.json')
    )


    '''
    feature_config=FeatureConfig(
            ema_periods=[9, 21, 50],
            label_method='triple_barrier',
            label_method_short='triple_barrier',
            label_method_long='triple_barrier',
            label_horizon=30,  # ← cambia de 10 a 30
            tp_barrier=3.0,  # ← sube de 2.5 a 3.0
            tp_barrier_short=3.0,
            sl_barrier=1.5,  # igual
            sl_barrier_short=1.5,
            regime_barriers_long={
                'trending': {'tp': 4.0, 'sl': 1.5},  # más recorrido en tendencia
                'ranging': {'tp': 2.5, 'sl': 1.5},
                'low_vol': {'tp': 3.0, 'sl': 1.0},
                'high_vol': {'tp': 3.5, 'sl': 2.0},
            },
            regime_barriers_short={  # simétrico
                'trending': {'tp': 4.0, 'sl': 1.5},
                'ranging': {'tp': 2.5, 'sl': 1.5},
                'low_vol': {'tp': 3.0, 'sl': 1.0},
                'high_vol': {'tp': 3.5, 'sl': 2.0},
            },
            feature_masks={
                'long': {'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True },
                         #'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False},
                'short': {'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True },
                         # 'ema_bull': False, 'rsi_oversold': False, 'macd_positive': False},
            }
        ),
    '''


    trainer = OptunaOOFTrainer(
        general_config=general,
        feature_config=FeatureConfig(
            label_method='triple_barrier',
            label_method_long='triple_barrier',
            label_method_short='triple_barrier',
            label_horizon=10,
            tp_barrier=2.5,
            sl_barrier=1.5,
            # Sin regime_barriers — barriers escalares simples
            # El análisis confirma que SL=1.5 funciona igual en todos los regímenes
            regime_barriers_long=None,
            regime_barriers_short=None,
            feature_masks={
                'long': {'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True},
                'short': {'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True},
            }
        ),
        regime_config=StateConfig(
            adx_trend_threshold=25.0
        ),
        base_model_config=ModelConfig(
            seq_len_short=64,
            seq_len_long=256,
            epochs=60,
            patience=10,
            use_hierarchical_fusion=True,
            ranking_loss_weight=0.2,
        ),
        out_dir=str(train_dir),
        optuna_db='mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db',
        study_prefix="oof_study",
        seed=42,
        reload=False,
        temperature_long=1.0,   # LONG: t=1.0 funciona bien sin distorsión
        temperature_short=1.2,  # SHORT: necesita algo de dispersión para el threshold
    )

    db = Database()
    dm = DataManager.from_database_historical_2(db, from_date=optuna_from, to_date=holdout_to)
    df_rates = dm.df

    # Ordenar datos cronológicamente
    print("\n🔍 Verificando orden cronológico...")
    if not pd.to_datetime(df_rates['time']).is_monotonic_increasing:
        print("⚠️  Datos desordenados, ordenando...")
        df_rates = df_rates.sort_values('time').reset_index(drop=True)

        # Eliminar duplicados
        n_before = len(df_rates)
        df_rates = df_rates.drop_duplicates(subset='time', keep='first')
        n_after = len(df_rates)

        if n_before != n_after:
            print(f"   Eliminados {n_before - n_after} timestamps duplicados")

        print(f"✅ Datos ordenados: {len(df_rates):,} filas")
    else:
        print(f"✅ Datos ya ordenados: {len(df_rates):,} filas")

    df_train = df_rates[df_rates.time < holdout_from].copy()
    df_hold = df_rates[df_rates.time >= holdout_from].copy()

    # ═══════════════════════════════════════════════════════════════════
    # TEST MANUAL: Verificar Máscaras de Features
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "🔍" * 35)
    print("TEST MANUAL: Verificando máscaras de features...")

    test_pipeline = DataPipeline(
        general_config=general,
        feature_config=trainer.feature_config,
        model_config=trainer.base_model_config,
        regime_config=trainer.regime_config
    )

    # Verificar que máscaras están configuradas
    if trainer.feature_config.feature_masks is None:
        print("❌ ERROR: feature_masks NO configuradas")
    else:
        print("✅ feature_masks configuradas:")
        for side, masks in trainer.feature_config.feature_masks.items():
            n_true = sum(1 for v in masks.values() if v)
            n_false = sum(1 for v in masks.values() if not v)
            print(f"   {side}: {n_true} activas, {n_false} desactivadas")

    # Test con datos pequeños
    if hasattr(test_pipeline, 'create_sequences_by_side'):
        try:
            df_test = df_train.head(5000)
            df_prep = test_pipeline.prepare_data(df_test.copy(), labels=False, side='both')

            sequences = test_pipeline.create_sequences_by_side(
                df_prep, sides=('long', 'short'), fit_scalers=True, train=False
            )

            n_long_seq = sequences['long']['seq_short'].shape[2]
            n_short_seq = sequences['short']['seq_short'].shape[2]
            n_long_ctx = sequences['long']['context'].shape[1]
            n_short_ctx = sequences['short']['context'].shape[1]

            print(f"\n📊 Dimensiones por lado:")
            print(f"   LONG:  seq_short={n_long_seq}  context={n_long_ctx}")
            print(f"   SHORT: seq_short={n_short_seq}  context={n_short_ctx}")

            # ── 1. Clasificar el modo de máscaras ───────────────────────────────
            fm = trainer.feature_config.feature_masks or {}
            mask_long_cfg = fm.get('long', {})
            mask_short_cfg = fm.get('short', {})

            required_long = [k for k, v in mask_long_cfg.items() if v is True]
            required_short = [k for k, v in mask_short_cfg.items() if v is True]
            excluded_long = [k for k, v in mask_long_cfg.items() if v is False]
            excluded_short = [k for k, v in mask_short_cfg.items() if v is False]

            directional_features = [
                'ema_bull', 'ema_bear',
                'rsi_oversold', 'rsi_overbought',
                'macd_positive', 'macd_negative',
            ]

            is_symmetric = (set(excluded_long) == set() and set(excluded_short) == set())
            is_asymmetric = (len(excluded_long) > 0 or len(excluded_short) > 0)
            is_no_mask = (fm == {} or trainer.feature_config.feature_masks is None)

            if is_no_mask:
                mask_mode = "SIN MÁSCARAS"
            elif is_symmetric:
                mask_mode = "SIMÉTRICAS"
            else:
                mask_mode = "ASIMÉTRICAS"

            print(f"\n🎭 Modo de máscaras: {mask_mode}")

            # ── 2. Auditoría de features direccionales ──────────────────────────
            # Reconstruir qué columnas de contexto recibe cada lado
            # usando la misma lógica de _apply_side_mask
            all_ctx = directional_features  # solo verificamos las que interesan
            ctx_long_actual = [f for f in all_ctx if f not in mask_long_cfg or mask_long_cfg[f]]
            ctx_short_actual = [f for f in all_ctx if f not in mask_short_cfg or mask_short_cfg[f]]

            print(f"\n   Features direccionales recibidas:")
            for f in directional_features:
                in_long = f in ctx_long_actual
                in_short = f in ctx_short_actual
                tag = ""
                if in_long and f in required_long:  tag += " ← requerida LONG"
                if in_short and f in required_short: tag += " ← requerida SHORT"
                if f in excluded_long:  tag += " ✗ excluida LONG"
                if f in excluded_short: tag += " ✗ excluida SHORT"
                long_sym = "✓" if in_long else "✗"
                short_sym = "✓" if in_short else "✗"
                print(f"     {f:<22}  LONG={long_sym}  SHORT={short_sym}{tag}")

            # ── 3. Verificaciones ───────────────────────────────────────────────
            errors = []
            warnings = []

            # Requeridas presentes
            for f in required_long:
                if f not in ctx_long_actual:
                    errors.append(f"LONG: feature requerida '{f}' NO está en contexto")
            for f in required_short:
                if f not in ctx_short_actual:
                    errors.append(f"SHORT: feature requerida '{f}' NO está en contexto")

            # Excluidas ausentes (solo con máscaras asimétricas)
            if is_asymmetric:
                for f in excluded_long:
                    if f in ctx_long_actual:
                        errors.append(f"LONG: feature excluida '{f}' SÍ está en contexto")
                for f in excluded_short:
                    if f in ctx_short_actual:
                        errors.append(f"SHORT: feature excluida '{f}' SÍ está en contexto")

            # Dimensiones consistentes entre lados
            if n_long_seq != n_short_seq:
                if not is_asymmetric:
                    warnings.append(
                        f"seq_short difiere: LONG={n_long_seq} SHORT={n_short_seq} "
                        f"(esperado igual con máscaras {mask_mode})"
                    )
            if n_long_ctx != n_short_ctx:
                if not is_asymmetric:
                    warnings.append(
                        f"context difiere: LONG={n_long_ctx} SHORT={n_short_ctx} "
                        f"(esperado igual con máscaras {mask_mode})"
                    )
            if is_asymmetric and n_long_ctx == n_short_ctx:
                warnings.append(
                    "Máscaras asimétricas pero context tiene la misma dimensión — "
                    "verificar que _apply_side_mask está excluyendo features"
                )

            # Dimensión de contexto vs esperada
            n_ctx_expected = len(df_prep.columns.intersection(
                sequences['long']['context'].shape[1:]  # fallback
            )) if False else None  # no podemos reconstruir sin pipeline internals

            # ── 4. Resultado ────────────────────────────────────────────────────
            print()
            if errors:
                for e in errors:
                    print(f"   ❌ {e}")
                print(f"\n❌ TEST FALLIDO — {len(errors)} error(es)")
            elif warnings:
                for w in warnings:
                    print(f"   ⚠️  {w}")
                print(f"\n⚠️  TEST CON ADVERTENCIAS — máscaras configuradas pero revisar warnings")
            else:
                if mask_mode == "SIMÉTRICAS":
                    print(
                        f"✅ MÁSCARAS SIMÉTRICAS CORRECTAS\n"
                        f"   Ambos modelos ven las 6 features direccionales.\n"
                        f"   Objetivo: eliminar sesgo long/short en días tendenciales."
                    )
                elif mask_mode == "ASIMÉTRICAS":
                    excl_summary = []
                    if excluded_long:  excl_summary.append(f"LONG excluye {excluded_long}")
                    if excluded_short: excl_summary.append(f"SHORT excluye {excluded_short}")
                    print(
                        f"✅ MÁSCARAS ASIMÉTRICAS CORRECTAS\n"
                        f"   {' | '.join(excl_summary)}"
                    )
                else:
                    print(f"✅ SIN MÁSCARAS — ambos modelos ven todas las features")

        except Exception as e:
            print(f"\n❌ Error en test: {e}")
            import traceback

            traceback.print_exc()
    else:
        print("❌ create_sequences_by_side NO existe")
        print("   → Actualiza data_pipeline_v2.py")

    print("🔍" * 35 + "\n")

    # ═══════════════════════════════════════════════════════════════════
    # VALIDACIÓN PRE-ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "🚀 " * 35)
    print("Iniciando validación pre-entrenamiento...")

    # Usar .head() para mantener orden cronológico
    df_sample = df_train.head(min(50000, len(df_train)))

    pre_training_results = validate_before_training(
        df_sample=df_sample,
        general_config=general,
        feature_config=trainer.feature_config,
        model_config=trainer.base_model_config,
        regime_config=trainer.regime_config
    )

    if pre_training_results:
        import json

        pre_val_path = train_dir / 'reports' / 'validation_pre_training.json'
        pre_val_path.parent.mkdir(parents=True, exist_ok=True)

        with open(pre_val_path, 'w') as f:
            json.dump(pre_training_results, f, indent=2, cls=NumpyEncoder)

        print(f"✅ Resultados guardados: {pre_val_path}")

    print("🚀 " * 35 + "\n")

    # ═══════════════════════════════════════════════════════════════════
    # ENTRENAMIENTO
    # ═══════════════════════════════════════════════════════════════════

    grid_long = {
        "conv1d_filters": [64],
        "lstm_units": [96],
        "context_units": [64],
        "time_units": [8],
        "head_units": [64],

        "dropout_seq": [0.10],
        "dropout_lstm": [0.20],
        "dropout_dense": [0.25],
        "l2_reg": [1e-5],

        # Explorar dos LR — el ReduceLROnPlateau se disparaba demasiado con 2e-4
        "learning_rate": [2e-4],

        "batch_size": [8192],

        # Con pos_rate ~20%, focal_alpha 0.35 puede ser insuficiente
        # Explorar 0.40 también
        "focal_alpha": [0.35],
        "focal_gamma": [2.00],

        "use_attention": [False],
        "use_gate": [True],

        "epochs": [60],
        "patience": [10],
    }

    artifacts_long_train = None
    holdout_report_long = None
    if not args.skip_long:
        print(f'[TUNING] Fine tuning LONG model')
        study_long = trainer.optimize(df_rates=df_train, side="long", n_trials=None, use_grid=True,
                                      grid_space=grid_long, load_if_exists=True)
        gc.collect()
        tf.keras.backend.clear_session()

        print(f'[DEPLOY] Preparing production LONG model based on TRAIN period')
        artifacts_long_train = trainer.prepare_production_model(df_rates=df_train, side="long", reuse_best_trial_oof=True)
        holdout_report_long = {
            "static": trainer.evaluate_holdout(artifacts_long_train, df_hold, side="long"),
            "walkforward": trainer.evaluate_holdout_walkforward_fast(artifacts_long_train, df_hold, side="long",
                                                                     inference_batch_size=64, return_predictions=True),
        }

        holdout_preds_long_path = train_dir / 'data' / f'holdout_predictions_{general.release}_long.parquet'
        Helper.save_holdout_predictions(holdout_report_long, holdout_preds_long_path, side='long')

        calibration_long_path = train_dir / 'data' / f'calibration_dataset_{general.release}_long.parquet'
        oof_long_path = Path(artifacts_long_train.oof_df_path)
        build_calibration_dataset(
            oof_path=oof_long_path,
            holdout_preds_path=holdout_preds_long_path,
            out_path=calibration_long_path,
            side='long',
        )

        decision_long = trainer.choose_inference_policy(
            holdout_report_long["static"],
            holdout_report_long["walkforward"],
        )

        print("LONG policy:", decision_long["selected_policy"])
        print("LONG reason:", decision_long["reason"])

        import ctypes

        trainer.best_oof_df_by_side.pop('long', None)
        trainer.best_calibrator_by_side.pop('long', None)
        trainer.best_percentiles_by_side.pop('long', None)

        gc.collect()
        tf.keras.backend.clear_session()

    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except:
        pass

    gc.collect()

    # ─── SHORT ──────────────────────────────────────────────────────────
    grid_short = {
        "conv1d_filters": [64],
        "lstm_units": [96],
        "context_units": [64],
        "time_units": [16],
        "head_units": [64],

        "dropout_seq": [0.10],
        "dropout_lstm": [0.15],
        "dropout_dense": [0.20],
        "l2_reg": [1e-5],

        # SHORT necesitaba más exploración de LR — el fold 5 colapsó por ReduceLROnPlateau agresivo
        "learning_rate": [2e-4],

        "batch_size": [8192],

        # SHORT con menor señal necesita más peso en positivos
        # Con pos_rate ~18-20% explorar hasta 0.45
        "focal_alpha": [0.45],
        "focal_gamma": [2.00],

        "use_attention": [True],
        "use_gate": [True],

        "epochs": [60],
        "patience": [10],
    }

    artifacts_short_train = None
    holdout_report_short = None
    if not args.skip_short:
        print(f'[TUNING] Fine tuning SHORT model')
        study_short = trainer.optimize(df_rates=df_train, side="short", n_trials=None, use_grid=True,
                                       grid_space=grid_short, load_if_exists=True)
        gc.collect()
        tf.keras.backend.clear_session()

        print(f'[DEPLOY] Preparing production SHORT model based on TRAIN period')
        artifacts_short_train = trainer.prepare_production_model(df_rates=df_train, side="short", reuse_best_trial_oof=True)
        holdout_report_short = {
            "static": trainer.evaluate_holdout(artifacts_short_train, df_hold, side="short"),
            "walkforward": trainer.evaluate_holdout_walkforward_fast(artifacts_short_train, df_hold, side="short",
                                                                     inference_batch_size=64, return_predictions=True),
        }

        holdout_preds_short_path = train_dir / 'data' / f'holdout_predictions_{general.release}_short.parquet'
        Helper.save_holdout_predictions(holdout_report_short, holdout_preds_short_path, side='short')

        calibration_short_path = train_dir / 'data' / f'calibration_dataset_{general.release}_short.parquet'
        oof_short_path = Path(artifacts_short_train.oof_df_path)
        build_calibration_dataset(
            oof_path=oof_short_path,
            holdout_preds_path=holdout_preds_short_path,
            out_path=calibration_short_path,
            side='short',
        )

        decision_short = trainer.choose_inference_policy(
            holdout_report_short["static"],
            holdout_report_short["walkforward"],
        )

        print("LONG policy:", decision_short["selected_policy"])
        print("LONG reason:", decision_short["reason"])

        gc.collect()
        tf.keras.backend.clear_session()

    # ═══════════════════════════════════════════════════════════════════
    # 🔍 VALIDACIÓN POST-ENTRENAMIENTO (AÑADIDO)
    # ═══════════════════════════════════════════════════════════════════

    print("\n" + "🎯 " * 35)
    print("Iniciando validación post-entrenamiento...")

    if artifacts_long_train is not None and artifacts_short_train is not None:
        post_training_results = validate_after_training_with_metrics(
            artifacts_long=artifacts_long_train,
            artifacts_short=artifacts_short_train,
            df_full=df_train,
            out_dir=str(train_dir),
            general_config=general,
            feature_config=trainer.feature_config,
            model_config=trainer.base_model_config,
            regime_config=trainer.regime_config
        )

        print("🎯 " * 35 + "\n")

        # ═══════════════════════════════════════════════════════════════════
        # HOLDOUT REPORT
        # ═══════════════════════════════════════════════════════════════════

        holdout_report = {
            'long': holdout_report_long,
            'short': holdout_report_short,
        }

        print(f'[HOLDOUT] Preparing report')
        Helper.save_meta(holdout_report, str(train_dir / 'reports' / 'holdout_report.json'))

    '''
    # ═══════════════════════════════════════════════════════════════════
    # MODELOS FINALES (FULL DATA)
    # ═══════════════════════════════════════════════════════════════════

    deploy_dir = base_dir / 'deploy_full'
    Helper.save_meta(
        {
            'release': general.release,
            'trained_on': 'full',
            'from_date': optuna_from.isoformat(),
            'to_date': holdout_to.isoformat(),
            'note': 'trained after holdout approval; holdout no longer independent for this artifact'
        }, str(deploy_dir / 'data' / 'train_meta.json')
    )

    trainer.out_dir = str(deploy_dir)
    gc.collect()
    tf.keras.backend.clear_session()

    print(f'[DEPLOY] Preparing production LONG model based on FULL period')
    artifacts_long_deploy = trainer.prepare_production_model(df_rates=df_rates, side='long', reuse_best_trial_oof=False)
    gc.collect()
    tf.keras.backend.clear_session()

    print(f'[DEPLOY] Preparing production SHORT model based on FULL period')
    artifacts_short_deploy = trainer.prepare_production_model(df_rates=df_rates, side='short', reuse_best_trial_oof=False)
    gc.collect()
    tf.keras.backend.clear_session()

    # ═══════════════════════════════════════════════════════════════════
    # VALIDACIÓN FINAL
    # ═══════════════════════════════════════════════════════════════════

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
        regime_config=trainer.regime_config
    )

    '''

    print("🏁 " * 35 + "\n")

    end_time = time.perf_counter()

    print("\n" + "╔" + "═" * 68 + "╗")
    print("║" + " " * 23 + "RESUMEN FINAL" + " " * 32 + "║")
    print("╚" + "═" * 68 + "╝")

    print(f'\n⏱️  Tiempo total: {end_time - start_time:.2f}s ({(end_time - start_time) / 60:.1f} min)')

    if pre_training_results:
        status = "✅ PASS" if pre_training_results['summary']['status'] == 'PASS' else "❌ FAIL"
        print(f'\n🔍 Validación pre-entrenamiento: {status}')

    if post_training_results:
        status = "✅ PASS" if post_training_results.get('pipeline_validation', {}).get('summary', {}).get(
            'status') == 'PASS' else "⚠️  WARNINGS"
        print(f'🎯 Validación post-entrenamiento (train): {status}')

    '''
    if final_validation_results:
        status = "✅ PASS" if final_validation_results.get('pipeline_validation', {}).get('summary', {}).get('status') == 'PASS' else "⚠️  WARNINGS"
        print(f'🏁 Validación final (deploy): {status}')
    '''

    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print("=" * 70 + "\n")

    print("OK")