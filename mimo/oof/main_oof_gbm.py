"""
main_oof_gbm.py
═══════════════════════════════════════════════════════════════════════════
Entry point CLI para el tuning GBM (LightGBM), paralelo a
main_oof_regime_weights_v7 pero usando árboles en lugar de la CNN-LSTM.

DECISIONES DE DISEÑO:
  - Mismo objetivo: EV-net balanceado long+short con triple barrier y
    sweep de thresholds (compute_balanced_objective).
  - Misma pipeline de features y labels que el CNN (FeatureConfig idéntico
    salvo label_method='triple_barrier_dual' para multitask).
  - Mismas regime weights y feature masks por release.
  - Mismas barriers por release (BARRIERS_BY_RELEASE).
  - Sólo cambian: modelo (LGBM), grid de hiperparámetros (GRID_GBM_BY_RELEASE)
    y nombre del study (oof_study_gbm_<release>_multitask).

USO:
  python3 -m mimo.oof.main_oof_gbm \\
    --release 202600_GBM --inherit-config-from 202500 \\
    --base-tf 5min --target-type multitask --side both \\
    --variant-long vol_boost_td_down --variant-short vol_boost \\
    --label-horizon-long 3 --label-horizon-short 3 \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --use-tpe --optuna-trials 40 \\
    --objective ev_net --cost-per-signal 0.05 \\
    --ev-min-signals 100 --max-drawdown-R 30 \\
    --ev-thr-lo 0.10 --ev-thr-hi 0.40
"""

from __future__ import annotations

import argparse
import os
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

# Reuse TODO model-agnostic from the CNN main: configs, regime weights,
# barriers, feature masks, vol-invariant / reduced features sets.
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.gbm_oof_trainer import GBMOOFTrainer
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE,
    GRID_BY_RELEASE,
    LONG_VARIANTS,
    SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES,
    _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release,
    _get_feature_masks_for_release,
    _tf_defaults,
    install_regime_weight_patch,
    resolve_regime_weights,
    set_global_seeds,
)


# ─── Grid GBM por release ────────────────────────────────────────────────
# Estructura idéntica a GRID_BY_RELEASE pero con HIPERPARÁMETROS de LightGBM
# en lugar de la red neuronal. El trainer detecta dict (suggest_float/int)
# vs list (suggest_categorical) automáticamente, igual que el CNN.
#
# Ranges informados por el GBM sanity check, que dio AUC-ROC=0.66 (TSCV)
# y 0.63 (holdout) sobre las mismas features que la CNN no podía aprovechar.
# El espacio aquí amplía alrededor de esos defaults.

_DEFAULT_GBM_GRID: Dict[str, Any] = {
    # Capacidad del árbol
    "num_leaves":         [15, 31, 63, 127],
    "max_depth":          [-1, 6, 8, 12],
    # Regularización
    "min_data_in_leaf":   [50, 100, 200, 400],
    "feature_fraction":   {"low": 0.6, "high": 1.0, "step": 0.1},
    "bagging_fraction":   {"low": 0.6, "high": 1.0, "step": 0.1},
    "bagging_freq":       [0, 5, 10],
    "lambda_l1":          {"low": 1e-8, "high": 1.0, "log": True},
    "lambda_l2":          {"low": 1e-8, "high": 1.0, "log": True},
    # Boosting
    "learning_rate":      {"low": 0.01, "high": 0.20, "log": True},
    "n_estimators":       [200, 500, 1000],
    "early_stopping_rounds": [30, 50],
}

GRID_GBM_BY_RELEASE: Dict[str, Dict[str, Any]] = {
    # 202600_GBM: primer espacio GBM para BTC/USD intraday 5min triple-barrier.
    # Hereda barriers de 202500 (tp=2.5/sl=1.5, ratio 1.67 — conservador).
    # Espacio amplio inicial; tras la primera corrida, refinamos como hicimos
    # con CNN (lo que TPE explote → lo estrechamos).
    "202600_GBM": {
        **_DEFAULT_GBM_GRID,
    },

    # 202602_GBM: track GBM sobre las MISMAS barriers que 202601 (CNN), para
    # comparación apples-to-apples. Hereda tp=2.0/sl=0.8 (ratio 2.50 — agresivo
    # pero alcanzable; mismo régimen-by-régimen) + VOL_INVARIANT + REDUCED.
    # Lanzar con --inherit-config-from 202601.
    #
    # GRID LIGERAMENTE REFINADO vs 202600_GBM:
    #   1) learning_rate techo bajado de 0.20 → 0.10. Con 130k filas y barriers
    #      agresivas, lr>0.1 suele overfittear en boosting clásico.
    #   2) max_depth: quitamos -1 (ilimitado). Con num_leaves grande + sin
    #      límite de depth, los árboles se vuelven proxies de sobreajuste por
    #      ruta. Mantenemos {6, 8, 12} como techo.
    #   3) min_data_in_leaf: añadimos opción 800 para forzar generalización en
    #      hojas. Con 130k filas y label_horizon=3, hay ~5-10k señales TP por
    #      lado; pedir >=800 ejemplos por hoja garantiza ramas no triviales.
    #   4) num_leaves: quitamos 127 (demasiada capacidad para 130k filas con
    #      ~7% positive rate). Mantenemos {15, 31, 63} = rango más sano.
    #   5) n_estimators: ampliamos a [200, 500, 1000, 1500]. Con lr más bajo,
    #      necesitamos más rounds para converger; early stopping decide el
    #      óptimo real en cada fold.
    #   6) feature_fraction: rango idéntico (subsampling de columnas siempre
    #      ayuda con ~80 features).
    #
    # NOTAS POST-CORRIDA (a completar tras primera tanda de trials):
    #   · Si la mayoría de top trials concentran en num_leaves=15-31 → bajamos
    #     a {7, 15, 31, 47} en 202603_GBM.
    #   · Si min_data_in_leaf grandes (400-800) dominan → bajamos l1/l2.
    #   · Si learning_rate explora bajo (0.01-0.03) → ampliamos n_estimators a
    #     [1000, 2000, 3000] para dejar converger.
    "202602_GBM": {
        # Capacidad del árbol — más conservadora que 202600_GBM
        "num_leaves":         [15, 31, 63],
        "max_depth":          [6, 8, 12],
        # Regularización — añadimos opción más fuerte
        "min_data_in_leaf":   [50, 100, 200, 400, 800],
        "feature_fraction":   {"low": 0.6, "high": 1.0, "step": 0.1},
        "bagging_fraction":   {"low": 0.6, "high": 1.0, "step": 0.1},
        "bagging_freq":       [0, 5, 10],
        "lambda_l1":          {"low": 1e-8, "high": 1.0, "log": True},
        "lambda_l2":          {"low": 1e-8, "high": 1.0, "log": True},
        # Boosting — techo de LR más bajo, más rounds disponibles
        "learning_rate":      {"low": 0.01, "high": 0.10, "log": True},
        "n_estimators":       [200, 500, 1000, 1500],
        "early_stopping_rounds": [30, 50, 80],
    },

    # 202603_GBM: track regularización fuerte tras diagnóstico del holdout 202602.
    # Hereda barriers de 202601 (igual que 202602_GBM, tp=2.0/sl=0.8).
    # Lanzar con --inherit-config-from 202601.
    #
    # ROOT-CAUSE ANALYSIS del 202602_GBM (10 trials → holdout colapso):
    #
    #   OOF best LONG  (#6): num_leaves=15 ← MIN del grid
    #                        max_depth=6   ← MIN del grid
    #                        n_estimators=200 ← MIN del grid
    #                        Pegado a 3 mínimos = grid demasiado amplio hacia
    #                        capacidad alta. El óptimo real probablemente
    #                        está MÁS ALLÁ del extremo regularizado.
    #
    #   OOF best SHORT (#2): num_leaves=31, max_depth=12 ← MAX
    #                        min_data_in_leaf=800 ← MAX
    #                        lambda_l2=0.26 (alto)
    #                        Asimetría LONG vs SHORT confirmada: el grid
    #                        permite la geometría que cada lado prefiere.
    #
    #   Holdout: LONG  +0.28R OOF → 0 signals (thr_OOF muy alto)
    #            SHORT +0.32R OOF → -0.13R holdout (prec -34%)
    #            → overfit y/o regime shift. Más regularización.
    #
    # CAMBIOS vs 202602_GBM (todos van en la dirección "más regularización"):
    #
    #   1) num_leaves: añadimos 7 (más pequeño). Si LONG quiere 15 ya como
    #      mínimo, probablemente quiere 7. Quitamos 63 (sobra capacidad).
    #
    #   2) max_depth: añadimos 4. Mantenemos 12 porque SHORT lo aprovecha.
    #
    #   3) min_data_in_leaf: techo 800 → 2500. Más ejemplos por hoja = más
    #      generalización. Quitamos 50 y 100 (poca regularización).
    #
    #   4) feature_fraction: rango 0.4-0.9 (antes 0.6-1.0). Más subsampling
    #      de columnas = menos overfit a features espurias.
    #
    #   5) bagging_fraction: rango 0.5-0.9 (antes 0.6-1.0). Más subsampling
    #      de filas. bagging_freq se mantiene en {0, 5, 10}.
    #
    #   6) lambda_l1, lambda_l2: techos 1.0 → 5.0. La asimetría 202602
    #      mostró que SHORT usa lambda_l2 alto; le damos margen.
    #
    #   7) learning_rate: rango 0.005-0.05 (antes 0.01-0.10). LR más bajo
    #      requiere más rounds pero produce ensembles más suaves.
    #
    #   8) n_estimators: rango 300-2000 (antes 200-1500). Compensa LR bajo.
    #
    #   9) early_stopping_rounds: rango {50, 100, 150} (antes {30, 50, 80}).
    #      Más paciencia para que LR bajo converja.
    #
    # COSTE ESTIMADO: 80 trials × ~10min = ~13h CPU. Mismo orden que 202602.
    "202603_GBM": {
        # Capacidad — empuja hacia árboles MÁS PEQUEÑOS
        "num_leaves":         [7, 15, 31],
        "max_depth":          [4, 6, 8, 12],
        # Regularización por hoja — techo MUCHO más alto
        "min_data_in_leaf":   [200, 400, 800, 1500, 2500],
        # Subsampling — rangos más bajos para más diversidad
        "feature_fraction":   {"low": 0.4, "high": 0.9, "step": 0.1},
        "bagging_fraction":   {"low": 0.5, "high": 0.9, "step": 0.1},
        "bagging_freq":       [0, 5, 10],
        # L1/L2 — techos más altos (SHORT 202602 usó l2=0.26)
        "lambda_l1":          {"low": 1e-4, "high": 5.0, "log": True},
        "lambda_l2":          {"low": 1e-4, "high": 5.0, "log": True},
        # Boosting — LR más bajo, más rounds, más paciencia
        "learning_rate":      {"low": 0.005, "high": 0.05, "log": True},
        "n_estimators":       [300, 600, 1200, 2000],
        "early_stopping_rounds": [50, 100, 150],
    },

    # 202604_GBM: barriers CONSERVADORAS (TP=2.5/SL=1.5, ratio 1.67) tras
    # 2 holdouts colapsados con barriers agresivas (TP=2.0/SL=0.8, ratio 2.5).
    #
    # ROOT-CAUSE del colapso holdout 202602 y 202603:
    #   · SHORT SL rate dobló en holdout (33% → 64-67%) → mecha intrabar
    #     dispara SL antes de que el movimiento real ocurra
    #   · LONG SL rate también subió (27% → 51%)
    #   · El TP/SL=2.5 requiere precision >=28.6% para BE. Holdout dió 18-22%.
    #
    # FIX: SL más ancho (0.8 → 1.5 ATR) absorbe mecha intrabar pero requiere
    # más precision (37.5% BE en ratio 1.67). La intuición: si el modelo
    # tiene señal genuina, una mecha de ruido NO debería desactivarla.
    # Y si NO tiene señal, ningún ratio salva. Validamos esa hipótesis aquí.
    #
    # GRID afinado por TPE en los 58 trials del "202603 con grid default":
    #   · num_leaves: TPE picked 15 en TODOS los top → narrow [11, 15, 21]
    #   · n_estimators: TPE picked 200 en TODOS → narrow [150, 200, 300]
    #   · max_depth: split entre 6 (SHORT) y 12 (LONG) → mantener ambos
    #   · learning_rate: sweet spot 0.05-0.08 → rango [0.02, 0.10]
    #   · min_data_in_leaf: extremos 50 y 400 → ampliar abajo [25-400]
    #   · bagging_fraction=0.8 en todos → narrow [0.7, 0.9]
    #   · feature_fraction: 0.7-0.8 → [0.6, 0.9]
    #   · lambda_l2: 1e-7 a 1e-6 (muy BAJO) → confirmar rango bajo
    #   · lambda_l1: 0.001-0.4 → rango amplio (TPE no concentró)
    #   · early_stopping=30 en todos → [20, 30, 50]
    #
    # COSTE ESTIMADO: 80 trials × ~10min = ~13h CPU.
    "202604_GBM": {
        # Capacidad — narrow alrededor de num_leaves=15 que TPE eligió
        "num_leaves":         [11, 15, 21],
        "max_depth":          [4, 6, 8, 12],
        # Regularización — incluye opción agresiva 25 (menos que 50)
        "min_data_in_leaf":   [25, 50, 100, 200, 400],
        # Subsampling — narrow alrededor del sweet spot 0.8
        "feature_fraction":   {"low": 0.6, "high": 0.9, "step": 0.1},
        "bagging_fraction":   {"low": 0.7, "high": 0.9, "step": 0.1},
        "bagging_freq":       [5, 10],
        # L1: rango amplio (TPE no concentró)
        "lambda_l1":          {"low": 1e-4, "high": 1.0, "log": True},
        # L2: rango BAJO (TPE confirmó que prefiere lambda_l2 < 1e-5)
        "lambda_l2":          {"low": 1e-8, "high": 1e-4, "log": True},
        # Boosting — sweet spot que TPE encontró
        "learning_rate":      {"low": 0.02, "high": 0.10, "log": True},
        "n_estimators":       [150, 200, 300],
        "early_stopping_rounds": [20, 30, 50],
    },

    # 202605_GBM_NO_TIME: igual grid que 202604, pero EXCLUYE features
    # time-of-day (minute_*, hour_*, dow_*, is_asia/london/ny/overlap,
    # is_friday/month_end/us_*). El feature_importance mostró que ~38%
    # del SHAP eran features temporales, y esas son la fuente del regime
    # drift (en holdout 2025-11+ los patrones intraday cambian).
    # Mismas barriers que 202604 (TP=2.5/SL=1.5).
    "202605_GBM_NO_TIME": {
        "num_leaves":         [11, 15, 21, 31],
        "max_depth":          [4, 6, 8, 12],
        "min_data_in_leaf":   [25, 50, 100, 200, 400],
        "feature_fraction":   {"low": 0.6, "high": 0.9, "step": 0.1},
        "bagging_fraction":   {"low": 0.7, "high": 0.9, "step": 0.1},
        "bagging_freq":       [5, 10],
        "lambda_l1":          {"low": 1e-4, "high": 1.0, "log": True},
        "lambda_l2":          {"low": 1e-8, "high": 1e-4, "log": True},
        "learning_rate":      {"low": 0.02, "high": 0.10, "log": True},
        "n_estimators":       [150, 200, 300],
        "early_stopping_rounds": [20, 30, 50],
    },

    # 202606_GBM_H6: redesign radical
    #   · Sin features time-of-day (mismo exclude que 202605)
    #   · label_horizon = 6 (30 min vs 15 min) — patrones más estructurales
    #   · cost_per_signal = 0.10 (más estricto que default 0.05)
    #   · Barriers conservadoras (heredadas de 202604, TP=2.5/SL=1.5)
    # Hipótesis: h=3 capta noise intrabar, h=6 capta swings reales que el
    # modelo puede aprender. Sin time-features + h más largo, modelo
    # debería ser robusto a regime shift intraday.
    # Grid ligeramente ampliado por incertidumbre (espacio nuevo).
    "202606_GBM_H6": {
        "num_leaves":         [7, 15, 31, 63],
        "max_depth":          [4, 6, 8, 12],
        "min_data_in_leaf":   [25, 50, 100, 200, 400, 800],
        "feature_fraction":   {"low": 0.5, "high": 0.9, "step": 0.1},
        "bagging_fraction":   {"low": 0.6, "high": 0.9, "step": 0.1},
        "bagging_freq":       [0, 5, 10],
        "lambda_l1":          {"low": 1e-5, "high": 1.0, "log": True},
        "lambda_l2":          {"low": 1e-8, "high": 1.0, "log": True},
        "learning_rate":      {"low": 0.01, "high": 0.10, "log": True},
        "n_estimators":       [150, 300, 500, 1000],
        "early_stopping_rounds": [30, 50, 100],
    },
}


# ─── Feature exclusion por release ────────────────────────────────────
# Patrones para excluir features (substring match en el nombre de columna).
# Aplicado por GBMOOFTrainer si exclude_feature_patterns se pasa.
_GBM_EXCLUDE_PATTERNS: Dict[str, List[str]] = {
    "202605_GBM_NO_TIME": [
        "minute_of_day", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "is_asia", "is_london", "is_ny", "is_overlap",
        "is_friday", "is_month_end",
        "is_us_first_hour", "is_us_last_hour",
    ],
    "202606_GBM_H6": [
        "minute_of_day", "hour_sin", "hour_cos",
        "dow_sin", "dow_cos",
        "is_asia", "is_london", "is_ny", "is_overlap",
        "is_friday", "is_month_end",
        "is_us_first_hour", "is_us_last_hour",
    ],
}


# ─── Defaults experimentales por release ──────────────────────────────
# label_horizon y cost_per_signal pueden variar por release.
# Si la CLI no pasa valores, se usan estos.
_GBM_RELEASE_DEFAULTS: Dict[str, Dict[str, Any]] = {
    "202606_GBM_H6": {"label_horizon_long": 6, "label_horizon_short": 6,
                       "cost_per_signal": 0.10},
}


# ─── Barriers GBM-specific (no en main_oof_regime_weights_v7) ─────────
# Se inyectan en BARRIERS_BY_RELEASE al cargar el módulo. Permite tener
# releases GBM con barriers propias sin tocar el dict del CNN.

_GBM_BARRIERS_BY_RELEASE: Dict[str, Dict[str, Any]] = {}

_GBM_CONSERVATIVE_BARRIERS = {
    "tp_base": 2.5,
    "sl_base": 1.5,
    "regime_barriers_long": {
        "trending": {"tp": 2.75, "sl": 1.50},
        "ranging":  {"tp": 2.25, "sl": 1.50},
        "low_vol":  {"tp": 2.00, "sl": 1.25},
        "high_vol": {"tp": 3.00, "sl": 1.75},
    },
    "regime_barriers_short": {
        "trending": {"tp": 2.75, "sl": 1.50},
        "ranging":  {"tp": 2.25, "sl": 1.50},
        "low_vol":  {"tp": 2.00, "sl": 1.25},
        "high_vol": {"tp": 3.00, "sl": 1.75},
    },
}

# 202604_GBM: ratio 1.67 (vs 2.5 de 202601). SL más ancho 1.5 ATR
# (vs 0.8) absorbe mecha intrabar. Régimen barriers ligeramente más
# restringidas que _DEFAULT_BARRIERS (que tenía TP=3.5 en trending);
# 2.5-3.0 es realista para 5min h=3.
_GBM_BARRIERS_BY_RELEASE["202604_GBM"] = _GBM_CONSERVATIVE_BARRIERS

# 202605_GBM_NO_TIME y 202606_GBM_H6: mismas barriers conservadoras
_GBM_BARRIERS_BY_RELEASE["202605_GBM_NO_TIME"] = _GBM_CONSERVATIVE_BARRIERS
_GBM_BARRIERS_BY_RELEASE["202606_GBM_H6"]      = _GBM_CONSERVATIVE_BARRIERS

# Inyección no-destructiva en el dict global. Si ya existe la entrada
# (p.ej. añadida directamente al v7 en el futuro), se respeta esa.
for _k, _v in _GBM_BARRIERS_BY_RELEASE.items():
    if _k not in BARRIERS_BY_RELEASE:
        BARRIERS_BY_RELEASE[_k] = _v


def _get_gbm_grid_for_release(release: str) -> Dict[str, Any]:
    release_str = str(release)
    grid = GRID_GBM_BY_RELEASE.get(release_str)
    if grid is None:
        # Si la release no tiene grid GBM específico, intentamos derivar uno
        # razonable: usar _DEFAULT_GBM_GRID tal cual (suficiente para arrancar).
        print(f"⚠️  [GBM GRID] No hay entrada para release={release_str}. "
              f"Usando _DEFAULT_GBM_GRID.")
        grid = _DEFAULT_GBM_GRID
    else:
        print(f"🎯 [GBM GRID] release={release_str} | keys: {sorted(grid.keys())}")
    return grid


# ─── CLI parser ──────────────────────────────────────────────────────────

def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True,
                    help="Tag de release (ej. '202600_GBM').")
    ap.add_argument("--inherit-config-from", default=None,
                    help="Hereda BARRIERS, VOL_INVARIANT y REDUCED de otra release "
                         "(NO sobreescribe si la actual ya tiene entradas explícitas).")
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--target-type", default="multitask",
                    choices=("multitask",),
                    help="GBM solo soporta multitask por ahora.")
    ap.add_argument("--side", default="both", choices=("both",),
                    help="GBM trainer entrena ambas cabezas en cada trial.")

    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()),
                    default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()),
                    default="moderate")
    # Overrides JSON opcionales sobre las variantes (mismos flags que el CNN
    # main; pueden ser None y resolve_regime_weights los ignora).
    ap.add_argument("--regime-weights-long-json", default=None,
                    help="JSON con pesos por régimen para LONG. Sobreescribe "
                         "selectivamente las keys de variant-long.")
    ap.add_argument("--regime-weights-short-json", default=None,
                    help="JSON con pesos por régimen para SHORT.")
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)

    ap.add_argument("--train-from", type=_parse_date, required=True)
    ap.add_argument("--train-to",   type=_parse_date, required=True)
    ap.add_argument("--holdout-from", type=_parse_date, required=True)
    ap.add_argument("--holdout-to",   type=_parse_date, required=True)

    # Optuna
    ap.add_argument("--use-tpe", action="store_true", default=True,
                    help="GBM trainer SIEMPRE usa TPE (no soporta GridSampler).")
    ap.add_argument("--optuna-trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=47)

    # EV-net objective
    ap.add_argument("--objective", default="ev_net", choices=("ev_net",),
                    help="GBM trainer solo soporta objective=ev_net por ahora.")
    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--ev-min-signals", type=int, default=100)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--ev-thr-lo", type=float, default=0.10)
    ap.add_argument("--ev-thr-hi", type=float, default=0.40)
    ap.add_argument("--ev-n-thr", type=int, default=60)

    # Storage Optuna
    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")

    ap.add_argument("--exp-tag-suffix", default=None)
    ap.add_argument("--notes", default=None)
    return ap


# ─── main ────────────────────────────────────────────────────────────────

def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))

    release = str(args.release)
    base_tf = str(args.base_tf)

    # 1) Inherit config no-destructivo (igual lógica que el CNN main)
    if args.inherit_config_from:
        src = str(args.inherit_config_from); dst = release
        if dst != src:
            inherited, skipped = [], []
            if src in BARRIERS_BY_RELEASE:
                if dst in BARRIERS_BY_RELEASE: skipped.append("BARRIERS")
                else: BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]; inherited.append("BARRIERS")
            if src in GRID_BY_RELEASE and dst in GRID_BY_RELEASE:
                # GRID del CNN no aplica a GBM, no se hereda; informativo
                pass
            if src in _VOL_INVARIANT_RELEASES:
                _VOL_INVARIANT_RELEASES.add(dst); inherited.append("VOL_INVARIANT")
            if src in _REDUCED_FEATURES_RELEASES:
                _REDUCED_FEATURES_RELEASES.add(dst); inherited.append("REDUCED")
            if src in _ULTRA_REDUCED_FEATURES_RELEASES:
                _ULTRA_REDUCED_FEATURES_RELEASES.add(dst); inherited.append("ULTRA_REDUCED")
            msg = f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}': "
            msg += ', '.join(inherited) if inherited else '(nada)'
            if skipped: msg += f" | NO sobreescritos: {', '.join(skipped)}"
            print(msg)

    # 2) Regime weights (mismo patrón que CNN main)
    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=True)

    # 3) Cargar OHLCV
    print(f"\n📂 Cargando OHLCV desde {args.train_from.date()} a {args.holdout_to.date()}...")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras cargadas")

    # 4) Configurar FeatureConfig idéntico al CNN multitask (mismas features,
    #    mismas barriers) — solo cambia que entrenamos un GBM
    barriers = _get_barriers_for_release(release)
    tf_defaults = _tf_defaults(base_tf)
    use_vol_invariant = release in _VOL_INVARIANT_RELEASES
    use_reduced = release in _REDUCED_FEATURES_RELEASES
    use_ultra = release in _ULTRA_REDUCED_FEATURES_RELEASES
    print(
        f"🪟 [TF DEFAULTS] base_tf={base_tf} | "
        f"seq_len_short={tf_defaults['seq_len_short']} | "
        f"seq_len_long={tf_defaults['seq_len_long']} | "
        f"price_norm_window={tf_defaults['price_norm_window']}"
    )
    if use_vol_invariant: print(f"🛡️  [VOL-INVARIANT] release={release}")
    if use_reduced:       print(f"✂️  [REDUCED FEATURES] release={release}")
    if use_ultra:         print(f"✂️✂️ [ULTRA-REDUCED FEATURES] release={release}")

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier_dual",     # multitask: dual labels
        label_horizon=max(args.label_horizon_long, args.label_horizon_short),
        tp_barrier=barriers["tp_base"],
        sl_barrier=barriers["sl_base"],
        label_method_long="triple_barrier_dual",
        regime_barriers_long=barriers["regime_barriers_long"],
        label_method_short="triple_barrier_dual",
        regime_barriers_short=barriers["regime_barriers_short"],
        tp_barrier_short=None,
        sl_barrier_short=None,
        feature_masks=_get_feature_masks_for_release(release),
        price_norm_window=tf_defaults["price_norm_window"],
        use_vol_invariant_features=use_vol_invariant,
        use_reduced_features=use_reduced,
        use_ultra_reduced_features=use_ultra,
    )

    general_config = Config(
        release=release,
        use_oof=True,
        oof_splits=5,
        oof_epochs=1,            # no aplica al GBM
        save_oof_artifacts=True,
    )
    regime_config = StateConfig(adx_trend_threshold=25.0)

    # 5) Crear artifacts dir consistente con CNN: artifacts/<release>/oof/<tag>/
    experiment_tag = (
        f"rw_both_L{args.variant_long}_h{args.label_horizon_long}"
        f"_S{args.variant_short}_h{args.label_horizon_short}"
    )
    if args.exp_tag_suffix:
        experiment_tag += args.exp_tag_suffix
    out_dir = Path("artifacts") / release / "oof" / experiment_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)
    print(f"📁 [OUTPUT] {out_dir}")

    # 6) Lanzar trainer GBM
    grid = _get_gbm_grid_for_release(release)
    exclude_patterns = _GBM_EXCLUDE_PATTERNS.get(release)
    if exclude_patterns:
        print(f"🚫 [GBM EXCLUDE] release={release} | excluyendo patterns: {exclude_patterns}")
    trainer = GBMOOFTrainer(
        general_config=general_config,
        feature_config=feature_config,
        regime_config=regime_config,
        out_dir=str(out_dir),
        optuna_db=args.optuna_storage,
        study_prefix=args.study_prefix,
        seed=int(args.seed),
        grid_space=grid,
        cost_per_signal=float(args.cost_per_signal),
        ev_min_signals=int(args.ev_min_signals),
        max_drawdown_R=float(args.max_drawdown_R),
        ev_thr_lo=float(args.ev_thr_lo),
        ev_thr_hi=float(args.ev_thr_hi),
        ev_n_thr=int(args.ev_n_thr),
        exclude_feature_patterns=exclude_patterns,
    )

    study = trainer.optimize(
        df_rates=df_rates,
        n_trials=int(args.optuna_trials),
        n_splits=int(general_config.oof_splits),
        load_if_exists=True,
    )

    # 7) Resumen
    completed = [t for t in study.trials if t.state.name == "COMPLETE"]
    print(f"\n✅ FINAL — {len(completed)} trials completados de {args.optuna_trials} lanzados")
    if completed:
        best = study.best_trial
        print(f"   Best trial #{best.number}: value={best.value:+.4f}")
        ev_long = best.user_attrs.get("ev_long", {})
        ev_short = best.user_attrs.get("ev_short", {})
        print(f"   LONG:  ev_net={ev_long.get('ev_net', float('nan')):+.4f}R "
              f"thr={ev_long.get('thr', float('nan')):.3f} "
              f"sig={ev_long.get('n_signals', 0)}")
        print(f"   SHORT: ev_net={ev_short.get('ev_net', float('nan')):+.4f}R "
              f"thr={ev_short.get('thr', float('nan')):.3f} "
              f"sig={ev_short.get('n_signals', 0)}")


if __name__ == "__main__":
    main()
