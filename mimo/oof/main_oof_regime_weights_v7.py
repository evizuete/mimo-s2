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
    "trend_robustness_v1": {
        # v6: bajar masa relativa de TREND para forzar mayor selectividad,
        # redistribuir a estados con pos_rate alta (TRANSITION, BREAKOUT, RANGE).
        # VOLATILE excluido (pos_rate 0.095 < global 0.150).
        # Diseñado sobre la distribución observada en fold 1 LONG (200385).
        "TREND_UP":           0.85,
        "TREND_DOWN":         0.85,
        "TRANSITION_UP":      1.15,
        "TRANSITION_DOWN":    1.10,
        "BREAKOUT_WAIT_UP":   1.20,
        "BREAKOUT_WAIT_DOWN": 1.25,
        "RANGE":              0.80,
        "VOLATILE":           0.0,
        "LOW_VOL":            0.0,
    },
    "no_trade_zero": {
        "TREND_UP": 1.0,
        "TREND_DOWN": 1.0,
        "TRANSITION_UP": 1.0,
        "TRANSITION_DOWN": 1.0,
        "BREAKOUT_WAIT_UP": 1.0,
        "BREAKOUT_WAIT_DOWN": 1.0,
        "RANGE": 1.0,
        "VOLATILE": 1.0,
        "LOW_VOL": 0.0,             # alinea con NO_TRADE_STATES en probs_calibration

    }
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
    "trend_robustness_v1": {
        # v6: simétrico al LONG. Sin holdout previo no hay base para
        # asimetría direccional; si se observa que SHORT falla peor en
        # algún estado concreto, ajustar entonces.
        "TREND_UP":           0.85,
        "TREND_DOWN":         0.85,
        "TRANSITION_UP":      1.10,
        "TRANSITION_DOWN":    1.15,
        "BREAKOUT_WAIT_UP":   1.25,
        "BREAKOUT_WAIT_DOWN": 1.20,
        "RANGE":              0.80,
        "VOLATILE":           0.0,
        "LOW_VOL":            0.0,
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# GRID por release (v7)
# ─────────────────────────────────────────────────────────────────────────────
# Cada entrada del dict define el grid_space para una release concreta.
# El producto cartesiano se evalúa con GridSampler en optuna_oof_trainer_v2.py.
# Para añadir nuevos releases, copiar una entrada existente y modificar lo necesario.
# Si no hay entrada para el release solicitado, se usa _DEFAULT_GRID (config baseline).

_DEFAULT_GRID = {
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

GRID_BY_RELEASE = {
    # ── 200391: control replicable de baseline (= 200390 sin overrides)
    # Sirve como referencia de estabilidad. Resultados deberían estar a ±0.005 de 200388/200390.
    "200391": {
        **_DEFAULT_GRID,
    },

    # ── 200392: +capacidad LSTM + más dropout para no overfittear
    # Hipótesis: el modelo está underfitting (train_loss y val_loss planas en últimas 10 epochs).
    # Subir lstm_units 96→128 y dropout_lstm 0.20→0.30 para que la capacidad mayor no sobreajuste.
    "200392": {
        **_DEFAULT_GRID,
        "lstm_units":   [128],
        "dropout_lstm": [0.30],
    },

    # ── 200393: +learning_rate + patience más alta
    # Hipótesis: ReduceLROnPlateau dispara muy temprano (epoch 21-47) sin rescatar AUC-PR.
    # lr=5e-4 puede explorar más antes de plana. patience=18 da margen para que el LR alto no muera.
    # Si en epoch 1-2 val_loss explota, reducir a 3.5e-4 con patience=15.
    "200393": {
        **_DEFAULT_GRID,
        "learning_rate": [5e-4],
        "patience":      [18],
    },
    "200394": {
        **_DEFAULT_GRID,
        "focal_alpha": [0.50],
        "focal_gamma": [2.50],
    },
    "200395": {
        **_DEFAULT_GRID,
        #"ranking_loss_weight": [0.35],
    },
    "200396": {
        **_DEFAULT_GRID,
    },
    # ── 200397: h=15 + barriers simétricas pequeñas (~1 ATR)
    # Hipótesis del barrier_sweep: con tp/sl ≈ 1 ATR en horizonte largo (12-15
    # velas de 1m) la pos_rate sube a ~0.40-0.50 y el lift_to_breakeven cae a
    # ~1.0-1.05x. El modelo solo necesita lift > 1.0 para ser rentable.
    # Cambio radical respecto a la familia 200391-200395 (movimientos
    # explosivos raros) → ahora aprende direccionalidad a corto plazo.
    "200397": {
        **_DEFAULT_GRID,
    },
    # ── 200398: h=5 (mantiene horizonte de 200395) + barriers reducidos
    # Tras descubrir en 200397 que pos_rate ~0.4 mata la learnability del modelo
    # (val_auc_roc cae de 0.66 a 0.52), volvemos a h=5 — donde el modelo SI
    # extraía señal — pero con tp/sl mas pequeños que las originales (3.5/1.25
    # → 2.0/1.10) para subir pos_rate de 0.065 a un rango operable (~0.15) y
    # bajar break-even de 0.26 a ~0.36, manteniendo SL >= 1 ATR.
    "200398": {
        **_DEFAULT_GRID,
    },
    # ── 200399: misma config que 200398 + auto_soft sample weighting
    # Ultimo experimento de la rama "barriers/horizon/weights" para agotar la
    # palanca de regime weights con las barriers de 200398. auto_soft ajusta
    # weight = 1 + 0.35*(pos_rate_state/pos_rate_global - 1) clipeado a
    # [0.85, 1.20], adaptandose automaticamente a los nuevos pos_rates por
    # estado (e.g. LOW_VOL ahora alto, VOLATILE bajo). Si esto da +>=4pp
    # AUC-ROC vs 200398, weights aporta. Si da <2pp, confirmado que el techo
    # esta en features y toca pivotar a microstructure/orderflow.
    "200399": {
        **_DEFAULT_GRID,
    },
    # ── 200400: mismas barriers que 200398 + features multi-timeframe (5m/15m/1h)
    # y calendario extendido. Primera prueba de la palanca "features" tras
    # confirmar que (h, tp, sl) y weights estan agotados.
    # Cambios respecto a 200398: solo features. Mismos barriers, mismo grid.
    # Esperado: +2-4 pp AUC-ROC si las features multi-TF aportan contexto util.
    "200400": {
        **_DEFAULT_GRID,
    }
}


def _get_grid_for_release(release: str) -> dict:
    """Devuelve el grid para el release solicitado, con fallback a _DEFAULT_GRID."""
    release_str = str(release)
    grid = GRID_BY_RELEASE.get(release_str, _DEFAULT_GRID)
    if release_str not in GRID_BY_RELEASE:
        print(f"⚠️  [GRID] No hay entrada en GRID_BY_RELEASE para release={release_str}. Usando _DEFAULT_GRID.")
    else:
        delta_keys = [k for k in grid if grid[k] != _DEFAULT_GRID.get(k)]
        if delta_keys:
            deltas = ", ".join(f"{k}={grid[k]}" for k in delta_keys)
            print(f"🎯 [GRID] release={release_str} | overrides vs default: {deltas}")
        else:
            print(f"🎯 [GRID] release={release_str} | usando configuración default (sin overrides)")
    return grid


# Alias retrocompatible (no se usa en el flujo, mantiene compatibilidad si algún módulo lo importa)
GRID_COMMON = _DEFAULT_GRID


# ─────────────────────────────────────────────────────────────────────────────
# BARRIERS por release
# ─────────────────────────────────────────────────────────────────────────────
# Permite cambiar tp/sl por release sin tocar build_trainer cada vez.
# Para releases sin entrada se usan los barriers default (la familia 200391-200395).

_DEFAULT_BARRIERS = {
    "tp_base": 2.5,
    "sl_base": 1.5,
    "regime_barriers_long": {
        "trending": {"tp": 3.5, "sl": 1.25},
        "ranging":  {"tp": 2.25, "sl": 1.25},
        "low_vol":  {"tp": 2.75, "sl": 1.00},
        "high_vol": {"tp": 3.50, "sl": 2.00},
    },
    "regime_barriers_short": {
        "trending": {"tp": 3.0, "sl": 1.25},
        "ranging":  {"tp": 2.25, "sl": 1.25},
        "low_vol":  {"tp": 2.50, "sl": 1.00},
        "high_vol": {"tp": 3.25, "sl": 2.00},
    },
}

BARRIERS_BY_RELEASE = {
    # 200397: barriers simétricas pequeñas para h=15. Decisión basada en el
    # barrier_sweep empírico (mimo/diagnosis/barrier_sweep.py), que mostró que
    # configs con tp/sl ≈ 1.0-1.5 ATR en h=12-15 tienen lift_to_breakeven ~1.0x
    # — alcanzable con el lift 1.7x del modelo en h=5.
    # SL ≥ 1.0 ATR para no caer en ruido (ATR es por definición el rango medio).
    "200397": {
        "tp_base": 1.5,
        "sl_base": 1.0,
        "regime_barriers_long": {
            "trending": {"tp": 1.5,  "sl": 1.0},
            "ranging":  {"tp": 1.25, "sl": 1.0},
            "low_vol":  {"tp": 1.25, "sl": 1.0},
            "high_vol": {"tp": 1.5,  "sl": 1.25},
        },
        "regime_barriers_short": {
            "trending": {"tp": 1.5,  "sl": 1.0},
            "ranging":  {"tp": 1.25, "sl": 1.0},
            "low_vol":  {"tp": 1.25, "sl": 1.0},
            "high_vol": {"tp": 1.5,  "sl": 1.25},
        },
    },
    # 200398: h=5 + barriers reducidos pero NO en zona de ruido. SL >= 1 ATR
    # siempre. tp/sl medio entre los originales (3.5/1.25) y los de 200397
    # (1.5/1.0). Preserva la naturaleza del problema (movimientos
    # significativos a 5 min) que era aprendible — solo baja la magnitud
    # exigida para subir pos_rate y bajar break-even.
    # BE objetivos: trending 0.355, ranging 0.386, low_vol 0.364, high_vol 0.400.
    "200398": {
        "tp_base": 2.0,
        "sl_base": 1.10,
        "regime_barriers_long": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
    },
    # 200399: hereda los barriers de 200398. Solo cambia el sample weighting
    # via --variant-long auto_soft --variant-short auto_soft en la linea de
    # comandos.
    "200399": {
        "tp_base": 2.0,
        "sl_base": 1.10,
        "regime_barriers_long": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
    },
    # 200400: hereda los barriers de 200398. El cambio efectivo viene de
    # feature_builder.py (multi-TF + calendario extendido).
    "200400": {
        "tp_base": 2.0,
        "sl_base": 1.10,
        "regime_barriers_long": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
    },
    # 200600: A/B contra 200400 cambiando solo el timeframe base (5min).
    # Usaba barriers default agresivos (tp=2.5/sl=1.5) → BE=0.375 inalcanzable.
    # AUC-ROC mejoró +0.06 vs 200400 pero precision absoluta no cruzaba breakeven.
    # Para limpiar el A/B mantenemos los barriers idénticos a 200400 (200398).
    "200600": {
        "tp_base": 2.0,
        "sl_base": 1.10,
        "regime_barriers_long": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
    },
    # 200601: idéntico a 200600. Reservado por si quieres correr una variante
    # adicional (p.ej. distinto label-horizon) manteniendo el mismo set de
    # barriers para A/B paritario contra 200400.
    "200601": {
        "tp_base": 2.0,
        "sl_base": 1.10,
        "regime_barriers_long": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.00, "sl": 1.10},
            "ranging":  {"tp": 1.75, "sl": 1.10},
            "low_vol":  {"tp": 1.75, "sl": 1.00},
            "high_vol": {"tp": 2.25, "sl": 1.50},
        },
    },
    # 200602: 5m base + barriers asimétricos tp=2.5 / sl=1.0 (BE=0.286).
    # Hipótesis tras la saga 200400→200601: el modelo logra precision ~0.27-0.30
    # a percentiles altos sobre BE=0.355 (no cruza). Bajando BE a 0.286
    # mediante TP/SL más asimétricos, esa misma precision SÍ cruzaría margen
    # positivo. SL=1.0 ATR como suelo en todos los regímenes para evitar
    # ejecuciones por slippage.
    #
    # Breakeven por régimen (sl/(tp+sl)):
    #   trending: 1.0/(2.5+1.0) = 0.286
    #   ranging:  1.0/(2.25+1.0) = 0.308
    #   low_vol:  1.0/(2.25+1.0) = 0.308
    #   high_vol: 1.25/(2.75+1.25) = 0.313
    "200602": {
        "tp_base": 2.5,
        "sl_base": 1.00,
        "regime_barriers_long": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
    },
    # 200603: idéntico a 200602 en barriers. La diferencia es que se ejecutará
    # con --target-type=quantile en vez del modo binario. Permite evaluar
    # quantile regression bajo el target asimétrico operativo (no aplicable a
    # entrenamiento — barriers se ignoran en quantile_return — pero el campo
    # existe para mantener consistencia con el flujo si más adelante quieres
    # comparar precision binaria sobre el mismo dataset).
    "200603": {
        "tp_base": 2.5,
        "sl_base": 1.00,
        "regime_barriers_long": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
    },
    # 200700: barriers idénticas a 200602 (binario, asimétrico tp=2.5/sl=1.0,
    # BE=0.286). La novedad está en las features: el FeatureBuilder ahora
    # incluye 4 features de volumen (vol_z_1h, vol_pct_1h, vol_spike,
    # vol_trend_1h) como context inputs. Test A/B contra 200602 para medir
    # el lift del volumen sobre AUC-ROC y precisión.
    "200700": {
        "tp_base": 2.5,
        "sl_base": 1.00,
        "regime_barriers_long": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
        "regime_barriers_short": {
            "trending": {"tp": 2.50, "sl": 1.00},
            "ranging":  {"tp": 2.25, "sl": 1.00},
            "low_vol":  {"tp": 2.25, "sl": 1.00},
            "high_vol": {"tp": 2.75, "sl": 1.25},
        },
    },
}


def _get_barriers_for_release(release: str) -> dict:
    """Devuelve los barriers para el release, con fallback a _DEFAULT_BARRIERS."""
    release_str = str(release)
    barriers = BARRIERS_BY_RELEASE.get(release_str, _DEFAULT_BARRIERS)
    if release_str not in BARRIERS_BY_RELEASE:
        print(f"🎯 [BARRIERS] release={release_str} | usando barriers default")
    else:
        print(f"🎯 [BARRIERS] release={release_str} | barriers especificos:")
        print(f"             tp_base={barriers['tp_base']} sl_base={barriers['sl_base']}")
        print(f"             long  trending={barriers['regime_barriers_long']['trending']}")
        print(f"             short trending={barriers['regime_barriers_short']['trending']}")
    return barriers


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
    ap.add_argument(
        "--target-type",
        choices=["binary", "quantile"],
        default="binary",
        help=(
            "Tipo de target: 'binary' (triple-barrier, default) o 'quantile' "
            "(regresión cuantílica sobre forward return / ATR con pinball loss)."
        ),
    )
    ap.add_argument(
        "--quantile-h", type=int, default=5,
        help="Horizonte (en barras) del forward return en modo target-type=quantile.",
    )
    ap.add_argument(
        "--quantiles", type=str, default="0.25,0.50,0.75",
        help="Lista CSV de cuantiles a predecir en modo target-type=quantile.",
    )
    ap.add_argument(
        "--base-tf", default="1min",
        help=(
            "Resolución base de las velas. Default '1min' (sin resample). "
            "Valores típicos: '5min', '15min', '1h'. Cambia el SNR del input "
            "y reduce el número de samples; ajusta consecuentemente "
            "label-horizon-* y, si quieres, los seq_len_short/seq_len_long "
            "(no expuestos por CLI; defaults 64/256 son adecuados a 5min)."
        ),
    )
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
        # FIX v4: suelos en 0.0 para PRESERVAR el filtro original de state_weight.
        # state_weight viene de StateDetector con LOW_VOL=0.0 y VOLATILE=0.1.
        # Con suelos 0.10/0.05 (v3), LOW_VOL subía a 0.05 efectivo, contaminando
        # el entrenamiento con muestras que el detector quería excluir.
        # Con min_final_weight=1e-6 evitamos literal 0 (por seguridad numérica)
        # sin reintroducir masa significativa en estados filtrados.
        final_weights, dbg = build_nonzero_final_weights_bridge(
            base_weight=base_weights,
            states=states.astype(str).to_numpy(),
            y=labels_arr,
            min_base_weight=0.0,
            min_final_weight=1e-6,
            regime_weight_map=None,
            auto_from_pos_rate=True,
            auto_strength=0.35,
            auto_clip=(0.85, 1.20),
            renorm_to_base_sum=True,
            return_debug=True,
        )
        mode = "auto_soft"
    else:
        # FIX v4: mismos suelos en 0.0 para variantes manuales.
        # Mantiene la coherencia: si el usuario pide TREND_DOWN=0.0 en LONG strong,
        # el peso final efectivo es 0 (excluido), no 0.05.
        manual_map = {str(k): float(v) for k, v in weights_map.items() if k != "__mode__"}
        final_weights, dbg = build_nonzero_final_weights_bridge(
            base_weight=base_weights,
            states=states.astype(str).to_numpy(),
            y=labels_arr,
            min_base_weight=0.0,
            min_final_weight=1e-6,
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

    # FIX v4: alineado con min_base_weight=0.0 del helper. Con suelo en 0.10
    # los regime_weights_effective de LOW_VOL/VOLATILE quedaban deformados en
    # auditoría (≈0.5x del real) porque el divisor estaba inflado.
    safe_base = base_weights.astype(np.float32)
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


def build_trainer(
    release: str,
    label_horizon_long: int,
    label_horizon_short: int,
    train_dir: Path,
    target_type: str = "binary",
    quantile_h: int = 5,
    quantile_levels: tuple = (0.25, 0.50, 0.75),
) -> OptunaOOFTrainer:
    general = Config(
        release=release,
        use_oof=True,
        oof_splits=5,
        oof_epochs=90,
        save_oof_artifacts=True,
    )

    barriers = _get_barriers_for_release(release)

    # En modo quantile usamos label_method='quantile_return' y se ignoran los
    # barriers (los barriers solo aplican al triple-barrier binario). El
    # ranking_loss tampoco aplica conceptualmente — la cabeza es regresión.
    is_quantile = target_type == "quantile"
    label_method_active = "quantile_return" if is_quantile else "triple_barrier"

    trainer = OptunaOOFTrainer(
        general_config=general,
        feature_config=FeatureConfig(
            ema_periods=[9, 21, 50],
            label_method=label_method_active,
            label_horizon=max(label_horizon_long, label_horizon_short),
            tp_barrier=barriers["tp_base"],
            sl_barrier=barriers["sl_base"],
            label_method_long=label_method_active,
            regime_barriers_long=None if is_quantile else barriers["regime_barriers_long"],
            label_method_short=label_method_active,
            regime_barriers_short=None if is_quantile else barriers["regime_barriers_short"],
            tp_barrier_short=None,
            sl_barrier_short=None,
            quantile_horizon=int(quantile_h),
            quantile_levels=tuple(quantile_levels),
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
            ranking_loss_weight=0.0 if is_quantile else 0.2,
            target_type=target_type,
            quantile_levels=tuple(quantile_levels),
        ),
        out_dir=str(train_dir),
        optuna_db="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
        study_prefix="oof_study",
        seed=42,
        reload=False,
        temperature_long=1.0,
        temperature_short=1.0,
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

    # FIX v5: validar que los artifacts críticos existen antes de devolver el
    # TrainerArtifacts. Sin esta comprobación, --holdout-only fallaba más tarde
    # con un error oscuro de tf.keras.models.load_model o joblib.load.
    required = {
        "model": model_path,
        "calibrator": calibrator_path,
        "oof_df": oof_df_path,
    }
    missing = {k: str(p) for k, p in required.items() if not p.exists()}
    if missing:
        raise FileNotFoundError(
            f"--holdout-only requiere artifacts entrenados para release={release} "
            f"side={side}. Faltan: {missing}. Ejecuta sin --holdout-only primero."
        )

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
            print(f"\n⏭️  [{side.upper()}] Saltando optimize(); cargando best_params desde Optuna DB...")
            # FIX v5: trainer.reload=True solo se usaba en __init__. Aquí hay que
            # llamar explícitamente al cargador de best params para que
            # prepare_production_model() encuentre best_model_config_by_side[side].
            try:
                trainer._load_best_from_storage(side=side)
            except Exception as e:
                raise RuntimeError(
                    f"--skip-optuna requiere un Optuna study previo para side={side}. "
                    f"Error cargando: {type(e).__name__}: {e}"
                )
        else:
            print(f"\n[TUNING] Fine tuning {side.upper()} model con regime weights...")
            trainer.optimize(
                df_rates=df_train,
                side=side,
                n_trials=None,
                use_grid=True,
                grid_space=_get_grid_for_release(release),
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
    is_quantile_mode = (
        getattr(trainer.base_model_config, "target_type", "binary") == "quantile"
    )
    with holdout_eval_context():
        if is_quantile_mode:
            # En modo quantile el walkforward eval no aplica (las métricas
            # binarias auc_pr/precision sobre las que decide la policy no
            # tienen sentido sobre cuantiles). Forzamos policy='static'
            # reusando la evaluación static como walkforward para que el
            # resto del flujo (choose_inference_policy, persist_*) funcione
            # sin cambios.
            static_eval = trainer.evaluate_holdout(artifacts, df_hold, side=side)
            holdout_report = {
                "static": static_eval,
                "walkforward": dict(static_eval),  # alias — fuerza static via decisión
            }
        else:
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
    # base_tf=='1min' → no resample (comportamiento original).
    # base_tf=='5min'/'15min'/'1h' → resample en el loader; el resto del
    # pipeline trabaja sobre la nueva resolución sin cambios adicionales.
    resample_arg = None if str(args.base_tf).lower() in ("1min", "1m", "none", "") else args.base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=optuna_from, to_date=holdout_to, resample=resample_arg
    )
    df_rates = dm.df
    if resample_arg is not None:
        print(f"   Base TF: {args.base_tf} ({len(df_rates):,} barras tras resample)")

    # FIX v5: garantizar que df_rates['time'] es datetime ANTES de cualquier
    # comparación o split. Si DataManager devuelve 'time' como string en algún
    # caso, los filtros df.time < holdout_from comparan str vs Timestamp y
    # devuelven resultados silenciosamente erróneos.
    df_rates["time"] = pd.to_datetime(df_rates["time"])

    if not df_rates["time"].is_monotonic_increasing:
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

    quantile_levels_parsed = tuple(
        float(x) for x in str(args.quantiles).split(",") if x.strip()
    )
    if args.target_type == "quantile" and len(quantile_levels_parsed) < 2:
        raise ValueError(
            f"--target-type=quantile requiere al menos 2 cuantiles. "
            f"Recibido: {quantile_levels_parsed}"
        )

    trainer = build_trainer(
        args.release,
        label_h_long,
        label_h_short,
        train_dir,
        target_type=args.target_type,
        quantile_h=args.quantile_h,
        quantile_levels=quantile_levels_parsed,
    )

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