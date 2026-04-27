#!/usr/bin/env python3
"""
main_oof_regime_weights.py

Experimento OOF con limpieza fina por regímenes vía sample_weight para LONG y SHORT,
sin eliminar filas ni romper la continuidad temporal del dataset.

Esta versión:
  - añade soporte estable para variant=auto_soft
  - usa Helper.build_nonzero_final_weights(...) como primera opción
  - incluye fallback local si el helper no es invocable tal cual
  - evita errores de json.dump con pandas.Timestamp y otros tipos no serializables
  - mantiene continuidad temporal completa del dataset
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
from mimo.models.model_builder import Config, ModelConfig
from mimo.oof.optuna_oof_trainer_v2 import OptunaOOFTrainer, TrainerArtifacts
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


def free_memory() -> None:
    gc.collect()
    tf.keras.backend.clear_session()
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass
    gc.collect()


# ─────────────────────────────────────────────────────────────────────────────
# JSON safe serialization
# ─────────────────────────────────────────────────────────────────────────────


def _json_safe(obj: Any) -> Any:
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, Path):
        return str(obj)

    if isinstance(obj, (datetime, pd.Timestamp)):
        return obj.isoformat()

    if isinstance(obj, pd.Timedelta):
        return obj.isoformat()

    if isinstance(obj, np.generic):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return [_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, pd.Series):
        return [_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, pd.Index):
        return [_json_safe(x) for x in obj.tolist()]

    if isinstance(obj, pd.DataFrame):
        return [_json_safe(x) for x in obj.to_dict(orient="records")]

    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}

    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(x) for x in obj]

    return str(obj)


def dump_json_safe(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_json_safe(payload), f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Variantes de pesos por régimen
# ─────────────────────────────────────────────────────────────────────────────

LONG_VARIANTS: Dict[str, Dict[str, Any]] = {
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
        "TRANSITION_UP": 0.50,
        "TRANSITION_DOWN": 0.25,
        "BREAKOUT_WAIT_DOWN": 0.0,
        "TREND_DOWN": 0.0,
        "VOLATILE": 0.0,
        "LOW_VOL": 0.0,
    },
    "auto_soft": {
        "__mode__": "auto_soft",
    },
}

SHORT_VARIANTS: Dict[str, Dict[str, Any]] = {
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
        "__mode__": "auto_soft",
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


# ─────────────────────────────────────────────────────────────────────────────
# Args
# ─────────────────────────────────────────────────────────────────────────────


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


def resolve_regime_weights(args: argparse.Namespace) -> Dict[str, Dict[str, Any]]:
    long_w = deepcopy(LONG_VARIANTS[args.variant_long])
    short_w = deepcopy(SHORT_VARIANTS[args.variant_short])

    if args.regime_weights_long_json:
        user_map = json.loads(args.regime_weights_long_json)
        for k, v in user_map.items():
            if k == "__mode__":
                long_w[k] = v
            else:
                long_w[str(k)] = float(v)

    if args.regime_weights_short_json:
        user_map = json.loads(args.regime_weights_short_json)
        for k, v in user_map.items():
            if k == "__mode__":
                short_w[k] = v
            else:
                short_w[str(k)] = float(v)

    return {"long": long_w, "short": short_w}


# ─────────────────────────────────────────────────────────────────────────────
# Regime weight helpers
# ─────────────────────────────────────────────────────────────────────────────

ORIGINAL_CREATE_SEQUENCES_BY_SIDE = DataPipeline.create_sequences_by_side
_PATCH_INSTALLED = False
_LAST_AUDIT: Dict[str, Dict[str, Any]] = {}
_HELPER_NONZERO_WARNED = False
# FIX BUG-C: flag que desactiva el patch durante evaluaciones de holdout.
# El monkey-patch no debe modificar los pesos cuando se evalúa el holdout
# porque contaminaría las métricas con los mismos criterios del entrenamiento.
# Se activa/desactiva mediante el context manager holdout_eval_context().
_HOLDOUT_EVAL_ACTIVE = False


from contextlib import contextmanager

@contextmanager
def holdout_eval_context():
    """Context manager que desactiva el regime-weight patch durante evaluaciones
    de holdout. Uso:

        with holdout_eval_context():
            trainer.evaluate_holdout(artifacts, df_hold, side=side)

    Garantiza que _HOLDOUT_EVAL_ACTIVE se restaura a False aunque haya excepción.
    """
    global _HOLDOUT_EVAL_ACTIVE
    _HOLDOUT_EVAL_ACTIVE = True
    try:
        yield
    finally:
        _HOLDOUT_EVAL_ACTIVE = False


def _is_auto_soft_mode(weights_map: Dict[str, Any] | None) -> bool:
    return isinstance(weights_map, dict) and weights_map.get("__mode__") == "auto_soft"


def _to_numpy_local(x: Any, dtype=np.float32) -> np.ndarray:
    if isinstance(x, (pd.Series, pd.Index)):
        return x.to_numpy(dtype=dtype)
    return np.asarray(x, dtype=dtype)


def _flatten_summary_df(summary: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(summary.columns, pd.MultiIndex):
        return summary.reset_index()

    out = summary.copy()
    flat_cols = []
    for col in out.columns:
        if isinstance(col, tuple):
            parts = [str(x) for x in col if x not in (None, "")]
            flat_cols.append("_".join(parts).strip("_"))
        else:
            flat_cols.append(str(col))
    out.columns = flat_cols
    out = out.reset_index()

    rename_map = {
        "state_": "state",
        "base_weight_sum": "base_weight_sum",
        "safe_base_sum": "safe_base_sum",
        "regime_weight_mean": "regime_weight_mean",
        "final_weight_sum": "final_weight_sum",
        "final_weight_min": "final_weight_min",
        "final_weight_max": "final_weight_max",
        "y_mean": "pos_rate",
        "y_sum": "pos_count",
        "y_count": "n",
    }
    out = out.rename(columns=rename_map)
    return out


def _local_build_nonzero_final_weights(
    base_weight,
    states,
    y=None,
    *,
    min_base_weight=0.10,
    min_final_weight=0.05,
    regime_weight_map=None,
    auto_from_pos_rate=False,
    auto_strength=0.35,
    auto_clip=(0.85, 1.20),
    renorm_to_base_sum=True,
    return_debug=False,
):
    base_weight = _to_numpy_local(base_weight, dtype=np.float32)
    states = np.asarray(states)
    n = len(base_weight)

    if len(states) != n:
        raise ValueError("states y base_weight deben tener la misma longitud")

    if auto_from_pos_rate:
        if y is None:
            raise ValueError("Si auto_from_pos_rate=True, debes pasar y")
        y = _to_numpy_local(y, dtype=np.float32)
        if len(y) != n:
            raise ValueError("y y base_weight deben tener la misma longitud")

        tmp = pd.DataFrame({"state": states, "y": y})
        stats = (
            tmp.groupby("state", dropna=False)["y"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "pos_rate", "count": "n"})
        )

        global_pos_rate = float(np.mean(y))
        eps = 1e-8

        auto_map = {}
        for state, row in stats.iterrows():
            pos_rate = float(row["pos_rate"])
            ratio = (pos_rate + eps) / (global_pos_rate + eps)
            w = 1.0 + auto_strength * (ratio - 1.0)
            w = float(np.clip(w, auto_clip[0], auto_clip[1]))
            auto_map[state] = w

        regime_weight_map = auto_map if regime_weight_map is None else {**auto_map, **regime_weight_map}

    if regime_weight_map is None:
        regime_weight_map = {}

    safe_base = np.maximum(base_weight, float(min_base_weight)).astype(np.float32)
    regime_weight = np.array(
        [max(float(regime_weight_map.get(s, 1.0)), 1e-6) for s in states],
        dtype=np.float32,
    )

    final_weight = safe_base * regime_weight
    final_weight = np.maximum(final_weight, float(min_final_weight)).astype(np.float32)

    if renorm_to_base_sum:
        original_sum = float(np.sum(base_weight))
        final_sum = float(np.sum(final_weight))
        if final_sum > 0 and original_sum > 0:
            scale = original_sum / final_sum
            final_weight = final_weight * scale
            final_weight = np.maximum(final_weight, float(min_final_weight)).astype(np.float32)

    if not return_debug:
        return final_weight

    dbg_df = pd.DataFrame({
        "state": states,
        "base_weight": base_weight,
        "safe_base": safe_base,
        "regime_weight": regime_weight,
        "final_weight": final_weight,
    })
    if y is not None:
        dbg_df["y"] = _to_numpy_local(y, dtype=np.float32)

    agg = {
        "base_weight": "sum",
        "safe_base": "sum",
        "regime_weight": "mean",
        "final_weight": ["sum", "min", "max"],
    }
    if y is not None:
        agg["y"] = ["mean", "sum", "count"]

    summary = dbg_df.groupby("state", dropna=False).agg(agg)

    debug = {
        "original_sum": float(np.sum(base_weight)),
        "safe_base_sum": float(np.sum(safe_base)),
        "final_sum": float(np.sum(final_weight)),
        "final_min": float(np.min(final_weight)),
        "final_max": float(np.max(final_weight)),
        "zero_count_final": int(np.sum(final_weight <= 0.0)),
        "summary_by_state": summary,
    }
    return final_weight, debug


def build_nonzero_final_weights_bridge(**kwargs):
    global _HELPER_NONZERO_WARNED
    helper_fn = getattr(Helper, "build_nonzero_final_weights", None)
    if callable(helper_fn):
        try:
            return helper_fn(**kwargs)
        except Exception as e:
            if not _HELPER_NONZERO_WARNED:
                print(f"⚠️  Helper.build_nonzero_final_weights falló ({type(e).__name__}: {e}). Usando fallback local.")
                _HELPER_NONZERO_WARNED = True

    return _local_build_nonzero_final_weights(**kwargs)


def _build_final_weights_for_side(
    side: str,
    base_weights: np.ndarray,
    states: pd.Series,
    labels_arr: np.ndarray,
    weights_map: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float], Dict[str, Any]]:
    if _is_auto_soft_mode(weights_map):
        final_weights, dbg = build_nonzero_final_weights_bridge(
            base_weight=base_weights,
            states=states.astype(str).to_numpy(),
            y=labels_arr,
            min_base_weight=0.10,
            min_final_weight=0.05,
            regime_weight_map=None,
            auto_from_pos_rate=True,
            auto_strength=0.35,
            auto_clip=(0.85, 1.20),
            renorm_to_base_sum=True,
            return_debug=True,
        )
        mode = "auto_soft"
    else:
        manual_map = {str(k): float(v) for k, v in weights_map.items() if k != "__mode__"}
        final_weights, dbg = build_nonzero_final_weights_bridge(
            base_weight=base_weights,
            states=states.astype(str).to_numpy(),
            y=labels_arr,
            min_base_weight=0.10,
            min_final_weight=0.05,
            regime_weight_map=manual_map,
            auto_from_pos_rate=False,
            renorm_to_base_sum=True,
            return_debug=True,
        )
        mode = "manual"

    summary_df = dbg.get("summary_by_state")
    if isinstance(summary_df, pd.DataFrame):
        summary_df = _flatten_summary_df(summary_df)
        if "state" not in summary_df.columns and "index" in summary_df.columns:
            summary_df = summary_df.rename(columns={"index": "state"})
    else:
        summary_df = pd.DataFrame()

    if not summary_df.empty:
        if "state" in summary_df.columns and "regime_weight_mean" in summary_df.columns:
            learned_map = {
                str(row["state"]): float(row["regime_weight_mean"])
                for _, row in summary_df.iterrows()
            }
        else:
            learned_map = {}
    else:
        learned_map = {}

    if mode == "manual" and not learned_map:
        learned_map = {str(k): float(v) for k, v in weights_map.items() if k != "__mode__"}

    safe_base = np.maximum(base_weights.astype(np.float32), 0.10).astype(np.float32)
    regime_weights_effective = (final_weights / np.maximum(safe_base, 1e-8)).astype(np.float32)

    audit = {
        "mode": mode,
        "requested_map": {str(k): v for k, v in weights_map.items()},
        "learned_map": learned_map,
        "helper_debug": {
            k: v for k, v in dbg.items() if k != "summary_by_state"
        },
        "state_summary": summary_df.to_dict(orient="records") if not summary_df.empty else [],
    }

    return final_weights.astype(np.float32), regime_weights_effective, learned_map, audit


# ─────────────────────────────────────────────────────────────────────────────
# Monkey patch de DataPipeline.create_sequences_by_side
# ─────────────────────────────────────────────────────────────────────────────


def install_regime_weight_patch(regime_weights_by_side: Dict[str, Dict[str, Any]], verbose: bool = True) -> None:
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

        # FIX BUG-C: no modificar pesos durante evaluación de holdout.
        # Aunque train=True (necesario para generar labels), el patch no debe
        # aplicar regime_weights en contexto de evaluación neutral.
        if _HOLDOUT_EVAL_ACTIVE:
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

            final_weights, regime_weights_effective, learned_map, auto_audit = _build_final_weights_for_side(
                side=side,
                base_weights=base_weights,
                states=states,
                labels_arr=labels_arr,
                weights_map=weights_map,
            )

            seq[f"weights_base_{side}"] = base_weights
            seq[f"regime_weights_{side}"] = regime_weights_effective
            seq["weights"] = final_weights

            by_state = pd.DataFrame(auto_audit.get("state_summary", []))
            if by_state.empty:
                by_state = (
                    pd.DataFrame(
                        {
                            "state": states.values,
                            "regime_weight_mean": regime_weights_effective,
                            "base_weight": base_weights,
                            "final_weight": final_weights,
                            "label": labels_arr,
                        }
                    )
                    .groupby("state", observed=False)
                    .agg(
                        n=("label", "size"),
                        pos_rate=("label", "mean"),
                        regime_weight_mean=("regime_weight_mean", "mean"),
                        base_weight_sum=("base_weight", "sum"),
                        final_weight_sum=("final_weight", "sum"),
                    )
                    .reset_index()
                    .sort_values(["final_weight_sum", "n"], ascending=[False, False])
                )

            sum_base = float(base_weights.sum())
            sum_final = float(final_weights.sum())

            _LAST_AUDIT[side] = {
                "n_samples": int(n),
                "sum_base_weight": sum_base,
                "sum_final_weight": sum_final,
                "effective_weight_ratio": float(sum_final / (sum_base + 1e-12)),
                "regime_weights_requested": _json_safe(weights_map),
                "regime_weights_applied": _json_safe(learned_map),
                **_json_safe(auto_audit),
            }

            if verbose:
                print(f"\n[REGIME WEIGHTS][{side.upper()}] sample_weight por régimen aplicado")
                print(
                    f"  · n={n:,} | sum_base={sum_base:,.2f} | "
                    f"sum_final={sum_final:,.2f} | "
                    f"ratio={sum_final / (sum_base + 1e-12):.4f}"
                )
                try:
                    print(by_state.to_string(index=False, justify="left"))
                except Exception:
                    print(by_state)

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
    with holdout_eval_context():
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
        min_auc_pr_gain=0.005
    )

    print(f"{side.upper()} policy:", decision["selected_policy"])
    print(f"{side.upper()} mode :", decision["selected_mode_label"])
    print(f"{side.upper()} reason:", decision["reason"])

    # Persistir la decisión final elegida tras holdout
    persist_info = trainer.persist_selected_inference_policy(
        side=side,
        selected_policy=decision["selected_policy"],
        decision=decision,
    )

    # Copia adicional en data/ para que quede visible junto al calibration dataset
    policy_data_path = train_dir / "data" / f"inference_policy_{release}_{side}.json"
    dump_json_safe(
        {
            "release": release,
            "side": side,
            "selected_policy": decision["selected_policy"],
            "selected_mode_label": decision["selected_mode_label"],
            "is_walkforward_selected": decision["is_walkforward_selected"],
            "reason": decision["reason"],
            "persist_info": persist_info,
        },
        policy_data_path,
    )
    print(f"✅ Policy final {side.upper()} guardada en: {policy_data_path}")

    report = {
        "release": release,
        "side": side,
        "audit": _LAST_AUDIT.get(side, {}),
        "holdout_policy": decision,
        "holdout_report": holdout_report,
        "persisted_policy": persist_info,
    }
    report_path = train_dir / "reports" / f"{side}_regime_weight_report_{release}.json"
    dump_json_safe(report, report_path)
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
        "persisted_policy": persist_info,
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
    holdout_to = datetime(2026, 4, 26)

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
            "persisted_policy": result.get("persisted_policy", {}),
            "report_path": result["report_path"],
        }
        free_memory()

    combined_report_path = train_dir / "reports" / f"regime_weight_combined_report_{args.release}.json"
    dump_json_safe(combined_report, combined_report_path)
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