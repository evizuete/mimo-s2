#!/usr/bin/env bash
#
# 008_deploy_long_only_gbm.sh — Fase 5 GBM: deploy LONG-only para producción.
#
# Produce los artifacts necesarios para inferencia diaria:
#   · production_booster_long.joblib       (booster refit últimos N meses)
#   · production_threshold.json            (thr seleccionado por scan reciente)
#   · production_features.json             (lista de feat_cols esperados)
#   · production_metadata.json             (manifest: release, trial, fechas)
#
# CADENCIA DE USO:
#   Lanzar una vez al mes (o cuando detectes drift). El script:
#     1. Carga el best LONG trial del Optuna study
#     2. Refit booster sobre los últimos --refit-months meses (default 12)
#     3. Escanea threshold sobre los últimos --thr-scan-months meses
#     4. Persiste todo en production/<release>/
#
# IMPORTANTE: por defecto NO usa calibrador isotónico (--use-calibrator desactivado).
# Razón: vimos en 202602/3 que el calibrator satura → thr no se traduce bien.
# Raw probs + threshold scan son más robustos.
#
# USO:
#   bash 008_deploy_long_only_gbm.sh                          # default 202603_GBM
#   RELEASE=202604_GBM bash 008_deploy_long_only_gbm.sh       # override
#   AS_OF=2026-04-01 bash 008_deploy_long_only_gbm.sh         # fecha de corte

set -euo pipefail

export RELEASE=${RELEASE:-202603_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}

# Defaults release-specific
case "${RELEASE}" in
  *_H6)
    _DEFAULT_LH=6
    _DEFAULT_COST=0.10
    ;;
  *)
    _DEFAULT_LH=3
    _DEFAULT_COST=0.05
    ;;
esac
export LH_LONG=${LH_LONG:-${_DEFAULT_LH}}
export LH_SHORT=${LH_SHORT:-${_DEFAULT_LH}}
export COST_PER_SIGNAL=${COST_PER_SIGNAL:-${_DEFAULT_COST}}

# Fechas
export AS_OF=${AS_OF:-2026-04-10}
export REFIT_MONTHS=${REFIT_MONTHS:-12}
export THR_SCAN_MONTHS=${THR_SCAN_MONTHS:-3}

# Threshold scan
export THR_LO=${THR_LO:-0.05}
export THR_HI=${THR_HI:-0.95}
export THR_N=${THR_N:-180}
export THR_MIN_SIGNALS=${THR_MIN_SIGNALS:-20}

# Calibrator: por defecto OFF (vimos saturación). Para activar: USE_CALIBRATOR=1
export USE_CALIBRATOR=${USE_CALIBRATOR:-0}

export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json

PROD_DIR=${PROD_DIR:-production/${RELEASE}}

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "FASE 5 GBM — Deploy LONG-only"
echo "  Release:           ${RELEASE}"
echo "  As-of:             ${AS_OF}"
echo "  Refit window:      ${REFIT_MONTHS} meses (hasta ${AS_OF})"
echo "  Thr scan window:   últimos ${THR_SCAN_MONTHS} meses"
echo "  Thr scan range:    [${THR_LO}, ${THR_HI}] × ${THR_N} pts"
echo "  Calibrator:        $([ "${USE_CALIBRATOR}" = "1" ] && echo "ON" || echo "OFF (raw probs)")"
echo "  Output dir:        ${PROD_DIR}"

[ -f "${BEST_JSON}" ] || abort "${BEST_JSON} no existe. Corre fase 1 primero."

CAL_FLAG=""
[ "${USE_CALIBRATOR}" = "1" ] && CAL_FLAG="--use-calibrator"

mkdir -p "${PROD_DIR}"

log_section "Refit + threshold scan + persistencia"

python3 -m mimo.oof.main_oof_gbm_deploy \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --best-json "${BEST_JSON}" \
  --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long ${LH_LONG} --label-horizon-short ${LH_SHORT} \
  --as-of ${AS_OF} \
  --refit-months ${REFIT_MONTHS} \
  --thr-scan-months ${THR_SCAN_MONTHS} \
  --thr-lo ${THR_LO} --thr-hi ${THR_HI} --thr-n ${THR_N} \
  --thr-min-signals ${THR_MIN_SIGNALS} \
  --cost-per-signal ${COST_PER_SIGNAL} \
  ${CAL_FLAG} \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --out-dir "${PROD_DIR}"

log_section "FASE 5 COMPLETADA"
echo "  Artifacts en: ${PROD_DIR}/"
ls -la "${PROD_DIR}/" 2>/dev/null | grep -v "^total" | tail -n +2 | awk '{printf "    %-40s  %s\n", $NF, $5}'

echo ""
echo "📋 PARA USO EN PRODUCCIÓN:"
echo "   1. Cargar booster: joblib.load('${PROD_DIR}/production_booster_long.joblib')"
echo "   2. Cargar features: json.load(open('${PROD_DIR}/production_features.json'))['feat_cols']"
echo "   3. Cargar threshold: json.load(open('${PROD_DIR}/production_threshold.json'))['threshold']"
echo "   4. Para cada bar: X = features[feat_cols].values"
echo "                     p = booster.predict(X)"
echo "                     signal = p >= threshold"
echo ""
echo "📅 RELANZAR este script ~mensualmente para refrescar el booster + threshold"
