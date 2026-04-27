#!/usr/bin/env python3
"""
main_oof_regime_weights.py

Experimento OOF con limpieza fina por regímenes vía sample_weight para LONG y SHORT,
sin eliminar filas ni romper la continuidad temporal del dataset.

Idea:
  - Se mantiene intacta la generación de features/labels/secuencias.
  - Se multiplica el weight del TARGET de cada lado por un factor dependiente del estado.
  - Las barras siguen presentes dentro de las ventanas históricas, pero las muestras cuyo
    target cae en estados no deseados pesan 0 o poco.

Implementación:
  - Monkey patch de DataPipeline.create_sequences_by_side(...).
  - Para cada lado y train=True, toma los últimos N estados del dataframe preparado
    y multiplica seq['weights'] por el factor por estado.
  - El resto del pipeline permanece igual.

Casos de uso típicos:
  python main_oof_regime_weights.py --side both --variant-long baseline --variant-short baseline
  python main_oof_regime_weights.py --side both --variant-long moderate --variant-short moderate
  python main_oof_regime_weights.py --side long --variant-long moderate_h15
  python main_oof_regime_weights.py --side both --skip-optuna

Notas:
  - Mantiene la continuidad temporal completa de las secuencias.
  - Los pesos por régimen se aplican solo al target final de cada ventana.
  - Pensado como script experimental; no reemplaza tu main_oof.py principal.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Tuple

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
from mimo.oof.main_oof import build_calibration_dataset
from mimo.oof.optuna_oof_trainer import OptunaOOFTrainer, TrainerArtifacts
from mimo.states_manager.state_detector import StateConfig

# ─────────────────────────────────────────────────────────────────────────────
# Runtime
# ─────────────────────────────────────────────────────────────────────────────

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_GPU_THREAD_MODE", "gpu_private")
os.environ.setdefault("TF_GPU_THREAD_COUNT", "1")

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print(f"✅ GPU configurada: {gpus[0].name}")
    except RuntimeError as e:
        print(f"⚠️  Error configurando GPU: {e}")

gc.set_threshold(700, 10, 10)


def free_memory():
    gc.collect()
    tf.keras.backend.clear_session()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# Variantes de pesos por régimen
# ─────────────────────────────────────────────────────────────────────────────

LONG_VARIANTS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "TREND_UP": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_UP": 1.0,
        "TRANSITION_UP": 1.0,
        "TRANSITION_DOWN": 1.0,
        "BREAKOUT_WAIT_DOWN": 1.0,
        "TREND_DOWN": 1.0,
        "VOLATILE": 1.0,
        "LOW_VOL": 1.0,
    },
    "moderate": {
        "TREND_UP": 0.90,
        "RANGE": 1.05,
        "TRANSITION_DOWN": 1.00,
        "TRANSITION_UP": 0.98,
        "BREAKOUT_WAIT_UP": 1.10,
        "TREND_DOWN": 0.75,
        "LOW_VOL": 1.08,
        "VOLATILE": 0.88,
        "BREAKOUT_WAIT_DOWN": 1.05,
    },
    "strong": {
        "TREND_UP": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_UP": 1.0,
        "TRANSITION_UP": 0.25,
        "TRANSITION_DOWN": 0.0,
        "BREAKOUT_WAIT_DOWN": 0.0,
        "TREND_DOWN": 0.0,
        "VOLATILE": 0.0,
        "LOW_VOL": 0.0,
    },
    "moderate_h15": {
        "TREND_UP": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_UP": 1.0,
        "TRANSITION_UP": 0.5,
        "TRANSITION_DOWN": 0.25,
        "BREAKOUT_WAIT_DOWN": 0.0,
        "TREND_DOWN": 0.0,
        "VOLATILE": 0.0,
        "LOW_VOL": 0.0,
    },
    "auto_soft": {
        "__mode__": "auto_soft"
    },
}

SHORT_VARIANTS: Dict[str, Dict[str, float]] = {
    "baseline": {
        "TREND_DOWN": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_DOWN": 1.0,
        "TRANSITION_DOWN": 1.0,
        "TRANSITION_UP": 1.0,
        "BREAKOUT_WAIT_UP": 1.0,
        "TREND_UP": 1.0,
        "VOLATILE": 1.0,
        "LOW_VOL": 1.0,
    },
    "moderate": {
        "TREND_DOWN": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_DOWN": 1.0,
        "TRANSITION_DOWN": 0.75,
        "TRANSITION_UP": 0.25,
        "BREAKOUT_WAIT_UP": 0.0,
        "TREND_UP": 0.0,
        "VOLATILE": 0.0,
        "LOW_VOL": 0.0,
    },
    "strong": {
        "TREND_DOWN": 1.0,
        "RANGE": 1.0,
        "BREAKOUT_WAIT_DOWN": 1.0,
        "TRANSITION_DOWN": 0.5,
        "TRANSITION_UP": 0.0,
        "BREAKOUT_WAIT_UP": 0.0,
        "TREND_UP": 0.0,
        "VOLATILE": 0.0,
        "LOW_VOL": 0.0,
    },
    "auto_soft": {
        "__mode__": "auto_soft"
    },
}

GRID_COMMON = {
    "conv1d_filters": [64],
    "lstm_units": [96],
    "context_units": [64],
    "time_units": [8],
    "head_units": [64],
    "dropout_seq": [0.10],
    "dropout_lstm": [0.20],
    "dropout_dense": [0.25],
    "l2_reg": [1e-5],
    "learning_rate": [2e-4],
    "batch_size": [8192],
    "focal_alpha": [0.35],
    "focal_gamma": [2.00],
    "use_attention": [False],
    "use_gate": [True],
    "epochs": [90],
    "patience": [12],
}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Experimento OOF con regime weights finos (sample_weight por estado) para LONG y SHORT."
    )
    ap.add_argument("--release", default="200382")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()), default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()), default="moderate")
    ap.add_argument("--label-horizon-long", type=int, default=None)
    ap.add_argument("--label-horizon-short", type=int, default=None)
    ap.add_argument("--regime-weights-long-json", type=str, default=None)
    ap.add_argument("--regime-weights-short-json", type=str, default=None)
    ap.add_argument("--skip-optuna", action="store_true", help="Saltar optimize() y reusar best_params ya existentes.")
    ap.add_argument("--holdout-only", action="store_true", help="No entrena; evalúa artifacts existentes.")
    ap.add_argument("--notes", type=str, default="")
    return ap.parse_args()


def resolve_label_horizons(args: argparse.Namespace) -> tuple[int, int]:
    h_long = args.label_horizon_long
    if h_long is None:
        h_long = 15 if args.variant_long == "moderate_h15" else 10

    h_short = args.label_horizon_short if args.label_horizon_short is not None else h_long
    return int(h_long), int(h_short)


def resolve_regime_weights(args: argparse.Namespace) -> Dict[str, Dict[str, float]]:
    long_w = deepcopy(LONG_VARIANTS[args.variant_long])
    short_w = deepcopy(SHORT_VARIANTS[args.variant_short])
    if args.regime_weights_long_json:
        user_map = json.loads(args.regime_weights_long_json)
        long_w.update({str(k): float(v) for k, v in user_map.items()})
    if args.regime_weights_short_json:
        user_map = json.loads(args.regime_weights_short_json)
        short_w.update({str(k): float(v) for k, v in user_map.items()})
    return {"long": long_w, "short": short_w}


# ─────────────────────────────────────────────────────────────────────────────
# Monkey patch de DataPipeline.create_sequences_by_side
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_CREATE_SEQUENCES_BY_SIDE = DataPipeline.create_sequences_by_side
_PATCH_INSTALLED = False
_LAST_AUDIT: Dict[str, Dict[str, Any]] = {}


def _is_auto_soft_mode(weights_map: Dict[str, Any] | None) -> bool:
    return isinstance(weights_map, dict) and weights_map.get("__mode__") == "auto_soft"


def compute_soft_regime_weights(
    states: pd.Series | np.ndarray,
    labels: np.ndarray,
    *,
    min_weight: float = 0.75,
    max_weight: float = 1.25,
    shrink_k: float = 2000.0,
    power: float = 0.70,
    min_count: int = 256,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, Dict[str, float], Dict[str, Any]]:
    """
    Calcula pesos suaves por régimen de forma automática y sin ceros.

    Idea:
      1. Calcula pos_rate por estado.
      2. Hace shrinkage hacia la tasa global.
      3. Convierte la ventaja relativa en peso.
      4. Comprime la amplitud con una potencia.
      5. Reduce el efecto en estados con pocas muestras.
      6. Recorta al rango [min_weight, max_weight].
      7. Renormaliza para que el peso medio ponderado sea ~1.

    Devuelve:
      - regime_weights_sample: peso por muestra alineado con states
      - learned_map: mapa {state: weight}
      - audit: dict con métricas intermedias
    """
    states_s = pd.Series(states, copy=False).astype(str)
    y = np.asarray(labels, dtype=np.float32).reshape(-1)

    if len(states_s) != len(y):
        raise ValueError(
            f"compute_soft_regime_weights: len(states)={len(states_s)} != len(labels)={len(y)}"
        )

    tmp = pd.DataFrame({
        "state": states_s.values,
        "label": y,
    })

    global_pos_rate = float(tmp["label"].mean())

    # Caso degenerado: si por lo que sea hay una sola clase, devolver neutro
    if not np.isfinite(global_pos_rate) or global_pos_rate <= 0.0 or global_pos_rate >= 1.0:
        learned_map = {str(s): 1.0 for s in sorted(tmp["state"].unique())}
        regime_weights_sample = states_s.map(learned_map).to_numpy(dtype=np.float32)
        audit = {
            "mode": "auto_soft",
            "global_pos_rate": global_pos_rate,
            "min_weight": min_weight,
            "max_weight": max_weight,
            "power": power,
            "shrink_k": shrink_k,
            "min_count": min_count,
            "state_summary": [],
            "note": "degenerate_global_rate_neutral_weights",
        }
        return regime_weights_sample, learned_map, audit

    stats = (
        tmp.groupby("state", observed=False)
        .agg(
            n=("label", "size"),
            pos_rate=("label", "mean"),
        )
        .reset_index()
    )

    # shrinkage hacia la tasa global
    stats["alpha"] = stats["n"] / (stats["n"] + float(shrink_k))
    stats["shrunk_pos_rate"] = (
        stats["alpha"] * stats["pos_rate"]
        + (1.0 - stats["alpha"]) * global_pos_rate
    )

    # ventaja relativa vs tasa global
    stats["relative_edge"] = (stats["shrunk_pos_rate"] + eps) / (global_pos_rate + eps)

    # compresión suave para evitar pesos extremos
    stats["raw_weight"] = np.power(stats["relative_edge"], power)

    # si el estado tiene pocas muestras, lo acercamos a 1.0
    stats["confidence"] = np.clip(stats["n"] / float(min_count), 0.0, 1.0)
    stats["soft_weight"] = 1.0 + (stats["raw_weight"] - 1.0) * stats["confidence"]

    # normalizar para que la media ponderada por n quede ~1
    mean_w = float(np.average(stats["soft_weight"], weights=stats["n"]))
    if mean_w > 0:
        stats["soft_weight"] = stats["soft_weight"] / mean_w

    # clip
    stats["soft_weight"] = stats["soft_weight"].clip(lower=min_weight, upper=max_weight)

    # renormalización final ligera
    mean_w2 = float(np.average(stats["soft_weight"], weights=stats["n"]))
    if mean_w2 > 0:
        stats["soft_weight"] = stats["soft_weight"] / mean_w2

    stats["soft_weight"] = stats["soft_weight"].clip(lower=min_weight, upper=max_weight)

    learned_map = {
        str(row.state): float(row.soft_weight)
        for row in stats.itertuples(index=False)
    }

    regime_weights_sample = (
        states_s.map(learned_map).fillna(1.0).to_numpy(dtype=np.float32)
    )

    stats = stats.sort_values(["soft_weight", "n"], ascending=[False, False]).reset_index(drop=True)

    audit = {
        "mode": "auto_soft",
        "global_pos_rate": global_pos_rate,
        "min_weight": float(min_weight),
        "max_weight": float(max_weight),
        "power": float(power),
        "shrink_k": float(shrink_k),
        "min_count": int(min_count),
        "learned_map": learned_map,
        "state_summary": stats.to_dict(orient="records"),
    }

    return regime_weights_sample, learned_map, audit


def install_regime_weight_patch(regime_weights_by_side: Dict[str, Dict[str, float]], verbose: bool = True) -> None:
    global _PATCH_INSTALLED
    if _PATCH_INSTALLED:
        return

    def wrapped_create_sequences_by_side(self, df, *args, **kwargs):
        results = ORIGINAL_CREATE_SEQUENCES_BY_SIDE(self, df, *args, **kwargs)

        sides = kwargs.get("sides")
        if sides is None and len(args) >= 1:
            sides = args[0]
        if sides is None:
            sides = ("long", "short")

        train_flag = kwargs.get("train")
        if train_flag is None and len(args) >= 3:
            train_flag = args[2]
        train_flag = True if train_flag is None else bool(train_flag)

        if not train_flag:
            return results
        if not isinstance(df, pd.DataFrame) or "state" not in df.columns:
            return results

        global _LAST_AUDIT

        for side in sides:
            seq = results.get(side)
            weights_map = regime_weights_by_side.get(side)
            if not seq or not weights_map:
                continue
            if "weights" not in seq or seq.get("weights") is None:
                continue
            if "labels" not in seq or seq.get("labels") is None:
                continue

            n = int(len(seq["labels"]))
            if n <= 0:
                continue
            if len(df) < n:
                raise ValueError(
                    f"Alineación inválida para {side}: len(df)={len(df)} < len(labels)={n}."
                )

            target_df = df.iloc[-n:].copy()
            states = target_df["state"].astype(str)

            labels_arr = np.asarray(seq["labels"], dtype=np.int32)
            base_weights = np.asarray(seq["weights"], dtype=np.float32)

            if len(base_weights) != n:
                raise ValueError(
                    f"Alineación inválida para {side}: len(weights)={len(base_weights)} != len(labels)={n}"
                )

            if _is_auto_soft_mode(weights_map):
                regime_weights, learned_map, auto_audit = compute_soft_regime_weights(
                    states=states,
                    labels=labels_arr,
                    min_weight=0.75,
                    max_weight=1.25,
                    shrink_k=2000.0,
                    power=0.70,
                    min_count=256,
                )
            else:
                learned_map = {str(k): float(v) for k, v in weights_map.items()}
                regime_weights = states.map(learned_map).fillna(1.0).to_numpy(dtype=np.float32)
                auto_audit = {
                    "mode": "manual",
                    "learned_map": learned_map,
                }

            final_weights = base_weights * regime_weights

            seq[f"weights_base_{side}"] = base_weights
            seq[f"regime_weights_{side}"] = regime_weights
            seq["weights"] = final_weights

            by_state = (
                pd.DataFrame(
                    {
                        "state": states.values,
                        "regime_weight": regime_weights,
                        "base_weight": base_weights,
                        "final_weight": final_weights,
                        "label": labels_arr,
                    }
                )
                .groupby("state", observed=False)
                .agg(
                    n=("label", "size"),
                    pos_rate=("label", "mean"),
                    regime_weight_mean=("regime_weight", "mean"),
                    base_weight_sum=("base_weight", "sum"),
                    final_weight_sum=("final_weight", "sum"),
                )
                .reset_index()
                .sort_values(["final_weight_sum", "n"], ascending=[False, False])
            )

            _LAST_AUDIT[side] = {
                "n_samples": int(n),
                "sum_base_weight": float(base_weights.sum()),
                "sum_final_weight": float(final_weights.sum()),
                "effective_weight_ratio": float(final_weights.sum() / (base_weights.sum() + 1e-12)),
                "regime_weights_requested": weights_map,
                "regime_weights_applied": learned_map,
                "state_summary": by_state.to_dict(orient="records"),
                **auto_audit,
            }

            if verbose:
                print(f"\n[REGIME WEIGHTS][{side.upper()}] sample_weight por régimen aplicado")
                print(
                    f"  · n={n:,} | sum_base={base_weights.sum():,.2f} | "
                    f"sum_final={final_weights.sum():,.2f} | "
                    f"ratio={final_weights.sum() / (base_weights.sum() + 1e-12):.4f}"
                )
                print(by_state.to_string(index=False, justify="left"))

        return results

    DataPipeline.create_sequences_by_side = wrapped_create_sequences_by_side
    _PATCH_INSTALLED = True


# ─────────────────────────────────────────────────────────────────────────────
# Config / trainer
# ─────────────────────────────────────────────────────────────────────────────


def build_trainer(release: str, label_horizon_long: int, label_horizon_short: int, train_dir: Path) -> OptunaOOFTrainer:
    general = Config(
        release=release,
        use_oof=True,
        oof_splits=5,
        oof_epochs=90,
        save_oof_artifacts=True,
    )

    trainer = OptunaOOFTrainer(
        general_config=general,
        feature_config=FeatureConfig(
            ema_periods=[9, 21, 50],
            label_method="triple_barrier",
            label_horizon=max(label_horizon_long, label_horizon_short),
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
        out_dir=str(train_dir),
        optuna_db="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
        study_prefix="oof_study",
        seed=42,
        reload=False,
        temperature_long=1.0,
        temperature_short=1.2,
    )
    return trainer


# ─────────────────────────────────────────────────────────────────────────────
# Helpers artifacts
# ─────────────────────────────────────────────────────────────────────────────


def load_existing_artifacts(train_dir: Path, release: str, side: str) -> TrainerArtifacts:
    model_path = train_dir / f"model_{release}_{side}.keras"
    calibrator_path = train_dir / f"oof_calibrator_{release}_{side}.joblib"
    percentiles_path = train_dir / f"percentiles_{release}_{side}.json"
    oof_df_path = train_dir / f"oof_{release}_{side}.parquet"
    oof_meta_path = train_dir / f"oof_meta_{release}_{side}.json"
    return TrainerArtifacts(
        best_params={},
        best_value=float("nan"),
        study_name=f"recovered_{release}_{side}",
        side=side,
        oof_metrics={},
        percentiles={},
        calibrator_path=str(calibrator_path),
        percentiles_path=str(percentiles_path),
        model_path=str(model_path),
        oof_meta_path=str(oof_meta_path),
        oof_df_path=str(oof_df_path),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Ejecución por lado
# ─────────────────────────────────────────────────────────────────────────────


def run_side(
    trainer: OptunaOOFTrainer,
    side: str,
    df_train: pd.DataFrame,
    df_hold: pd.DataFrame,
    train_dir: Path,
    release: str,
    holdout_only: bool,
    skip_optuna: bool,
) -> Dict[str, Any]:
    if holdout_only:
        print(f"\n[HOLDOUT-ONLY] Cargando artifacts {side.upper()} existentes desde train_dir...")
        artifacts = load_existing_artifacts(train_dir, release, side)
    else:
        if skip_optuna:
            print(f"\n⏭️  [{side.upper()}] Saltando optimize(); se reusarán best_params ya existentes si están en DB/cache")
            trainer.reload = True
        else:
            print(f"\n[TUNING] Fine tuning {side.upper()} model con regime weights...")
            trainer.optimize(
                df_rates=df_train,
                side=side,
                n_trials=None,
                use_grid=True,
                grid_space=GRID_COMMON,
                load_if_exists=True,
            )
            free_memory()

        print(f"\n[DEPLOY] Preparing production {side.upper()} model based on TRAIN period")
        artifacts = trainer.prepare_production_model(
            df_rates=df_train,
            side=side,
            reuse_best_trial_oof=True,
        )
        free_memory()

    print(f"\n[HOLDOUT] Evaluando {side.upper()}...")
    holdout_report = {
        "static": trainer.evaluate_holdout(artifacts, df_hold, side=side),
        "walkforward": trainer.evaluate_holdout_walkforward_fast(
            artifacts,
            df_hold,
            side=side,
            inference_batch_size=64,
            return_predictions=True,
        ),
    }

    holdout_preds_path = train_dir / "data" / f"holdout_predictions_{release}_{side}.parquet"
    Helper.save_holdout_predictions(holdout_report, holdout_preds_path, side=side)

    calibration_path = train_dir / "data" / f"calibration_dataset_{release}_{side}.parquet"
    oof_path = Path(artifacts.oof_df_path)
    build_calibration_dataset(
        oof_path=oof_path,
        holdout_preds_path=holdout_preds_path,
        out_path=calibration_path,
        side=side,
    )

    decision = trainer.choose_inference_policy(
        holdout_report["static"],
        holdout_report["walkforward"],
    )
    print(f"{side.upper()} policy:", decision["selected_policy"])
    print(f"{side.upper()} reason:", decision["reason"])

    report = {
        "release": release,
        "side": side,
        "audit": _LAST_AUDIT.get(side, {}),
        "holdout_policy": decision,
        "holdout_report": holdout_report,
    }
    report_path = train_dir / "reports" / f"{side}_regime_weight_report_{release}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, cls=NumpyEncoder, ensure_ascii=False)

    print(f"✅ Reporte {side.upper()} guardado: {report_path}")

    if _LAST_AUDIT.get(side, {}).get("state_summary"):
        df_audit = pd.DataFrame(_LAST_AUDIT[side]["state_summary"])
        audit_csv = train_dir / "reports" / f"{side}_regime_weight_state_summary_{release}.csv"
        df_audit.to_csv(audit_csv, index=False)
        print(f"✅ Auditoría por estado {side.upper()} guardada: {audit_csv}")

    return {
        "artifacts": artifacts,
        "holdout_report": holdout_report,
        "decision": decision,
        "report_path": str(report_path),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────


def main() -> None:
    args = parse_args()
    start_time = time.perf_counter()

    label_h_long, label_h_short = resolve_label_horizons(args)
    regime_weights_by_side = resolve_regime_weights(args)

    print(
        f"\n🚀 Regime-weight experiment | release={args.release} | side={args.side} | "
        f"long_variant={args.variant_long} (h={label_h_long}) | "
        f"short_variant={args.variant_short} (h={label_h_short}) | holdout_only={args.holdout_only}"
    )

    install_regime_weight_patch(regime_weights_by_side, verbose=True)

    optuna_from = datetime(2025, 1, 1)
    holdout_from = datetime(2026, 2, 1)
    holdout_to = datetime(2026, 4, 10)

    experiment_tag = f"rw_both_L{args.variant_long}_h{label_h_long}_S{args.variant_short}_h{label_h_short}"
    if args.regime_weights_long_json:
        experiment_tag += "_customL"
    if args.regime_weights_short_json:
        experiment_tag += "_customS"

    base_dir = Path("../../artifacts") / args.release / "oof"
    train_dir = base_dir / experiment_tag
    train_dir.mkdir(parents=True, exist_ok=True)
    (train_dir / "data").mkdir(parents=True, exist_ok=True)
    (train_dir / "reports").mkdir(parents=True, exist_ok=True)

    Helper.save_meta(
        {
            "release": args.release,
            "train_from": optuna_from.isoformat(),
            "holdout_from": holdout_from.isoformat(),
            "holdout_to": holdout_to.isoformat(),
            "side": args.side,
            "variant_long": args.variant_long,
            "variant_short": args.variant_short,
            "label_horizon_long": label_h_long,
            "label_horizon_short": label_h_short,
            "regime_weights_long": regime_weights_by_side["long"],
            "regime_weights_short": regime_weights_by_side["short"],
            "notes": args.notes,
        },
        str(train_dir / "data" / "split_and_weights_meta.json"),
    )

    print("\n📂 Cargando datos...")
    db = Database()
    dm = DataManager.from_database_historical_2(db, from_date=optuna_from, to_date=holdout_to)
    df_rates = dm.df

    if not pd.to_datetime(df_rates["time"]).is_monotonic_increasing:
        print("⚠️  Datos desordenados, ordenando...")
        df_rates = df_rates.sort_values("time").reset_index(drop=True)
        n_before = len(df_rates)
        df_rates = df_rates.drop_duplicates(subset="time", keep="first")
        n_after = len(df_rates)
        if n_before != n_after:
            print(f"   Eliminados {n_before - n_after} timestamps duplicados")
    print(f"✅ Datos listos: {len(df_rates):,} filas")

    df_train = df_rates[df_rates.time < holdout_from].copy()
    df_hold = df_rates[df_rates.time >= holdout_from].copy()
    print(f"   Train: {len(df_train):,} | Holdout: {len(df_hold):,}")

    trainer = build_trainer(args.release, label_h_long, label_h_short, train_dir)

    combined_report = {
        "release": args.release,
        "side": args.side,
        "variant_long": args.variant_long,
        "variant_short": args.variant_short,
        "label_horizon_long": label_h_long,
        "label_horizon_short": label_h_short,
        "regime_weights_long": regime_weights_by_side["long"],
        "regime_weights_short": regime_weights_by_side["short"],
        "results": {},
        "notes": args.notes,
    }

    requested_sides = [args.side] if args.side in ("long", "short") else ["long", "short"]

    for side in requested_sides:
        result = run_side(
            trainer=trainer,
            side=side,
            df_train=df_train,
            df_hold=df_hold,
            train_dir=train_dir,
            release=args.release,
            holdout_only=args.holdout_only,
            skip_optuna=args.skip_optuna,
        )
        combined_report["results"][side] = {
            "audit": _LAST_AUDIT.get(side, {}),
            "decision": result["decision"],
            "report_path": result["report_path"],
        }
        free_memory()

    combined_report_path = train_dir / "reports" / f"regime_weight_combined_report_{args.release}.json"
    with open(combined_report_path, "w") as f:
        json.dump(combined_report, f, indent=2, cls=NumpyEncoder)
    print(f"✅ Reporte combinado guardado: {combined_report_path}")

    elapsed = time.perf_counter() - start_time
    print("\n" + "=" * 70)
    print("✅ PROCESO COMPLETADO")
    print(f"⏱️  Tiempo total: {elapsed:.2f}s ({elapsed/60:.1f} min)")
    print(f"📁 Output: {train_dir}")
    print("=" * 70)
    print("OK")


if __name__ == "__main__":
    main()
