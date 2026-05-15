#!/usr/bin/env bash
#
# 005_diagnostics_gbm.sh — Diagnósticos GBM (calibración + feature importance + per-state).
#
# Los tres tools requieren cargar OHLCV, prepare_data y refit (~5min cada uno).
# Se pueden correr individualmente o todos en cadena.
#
# CONTROL:
#   DIAG_CAL=1            Calibración (histograma, ECE, reliability)
#   DIAG_FI=1             Feature importance + SHAP (si está instalado)
#   DIAG_PERSTATE=1       Breakdown por market_state
#
# USO:
#   bash 005_diagnostics_gbm.sh                       # los 3
#   DIAG_FI=0 bash 005_diagnostics_gbm.sh             # sin feature importance
#   DIAG_CAL=1 DIAG_FI=0 DIAG_PERSTATE=0 bash 005_diagnostics_gbm.sh  # solo calibración

set -euo pipefail

export RELEASE=${RELEASE:-202602_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}
export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

# Toggles (default: todos activos)
export DIAG_CAL=${DIAG_CAL:-1}
export DIAG_FI=${DIAG_FI:-1}
export DIAG_PERSTATE=${DIAG_PERSTATE:-1}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "DIAGNÓSTICOS GBM — ${RELEASE} / ${TAG}"
echo "  Calibración:        $([ "${DIAG_CAL}"     = "1" ] && echo "✓" || echo "skip")"
echo "  Feature importance: $([ "${DIAG_FI}"      = "1" ] && echo "✓" || echo "skip")"
echo "  Per-state:          $([ "${DIAG_PERSTATE}"= "1" ] && echo "✓" || echo "skip")"

[ -f "${BEST_JSON}" ] || abort "${BEST_JSON} no existe"

COMMON_ARGS="--release ${RELEASE} --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --best-json ${BEST_JSON} --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --optuna-storage ${OPTUNA_STORAGE} --seed ${SEED}"

if [ "${DIAG_CAL}" = "1" ]; then
  log_section "1. Diagnóstico de calibración"
  python3 -m mimo.oof.diag_calibration_gbm \
    ${COMMON_ARGS} \
    --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
    --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
    --out-json ${REPORTS_DIR}/calibration_diag.json
fi

if [ "${DIAG_FI}" = "1" ]; then
  log_section "2. Feature importance + SHAP"
  python3 -m mimo.oof.diag_feature_importance_gbm \
    ${COMMON_ARGS} \
    --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
    --shap-sample 5000 --top-n 25 \
    --out-json ${REPORTS_DIR}/feature_importance.json
fi

if [ "${DIAG_PERSTATE}" = "1" ]; then
  log_section "3. Breakdown por régimen"
  python3 -m mimo.oof.diag_per_state_gbm \
    ${COMMON_ARGS} \
    --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
    --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
    --min-signals-per-state 10 \
    --out-json ${REPORTS_DIR}/per_state_diag.json
fi

log_section "DIAGNÓSTICOS COMPLETADOS"
echo "  Reports en: ${REPORTS_DIR}/"
[ "${DIAG_CAL}"      = "1" ] && echo "    · calibration_diag.json"
[ "${DIAG_FI}"       = "1" ] && echo "    · feature_importance.json"
[ "${DIAG_PERSTATE}" = "1" ] && echo "    · per_state_diag.json"
