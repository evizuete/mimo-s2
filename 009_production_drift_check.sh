#!/usr/bin/env bash
#
# 009_production_drift_check.sh
# ═══════════════════════════════════════════════════════════════════════
# Detector de drift para el modelo GBM en producción.
#
# Lee production/<release>/logs/gbm_inference_*.jsonl de los últimos N días
# y compara contra el baseline del deploy. Output: veredicto automático
#   ✅ OK              — métricas dentro de tolerancias
#   ⚠️  WARNING        — drift moderado, monitorear
#   🔴 RETUNE_NOW     — drift fuerte, relanzar 008_deploy_long_only_gbm.sh
#
# USO:
#   bash 009_production_drift_check.sh                                 # default 202603_GBM, 30 días
#   RELEASE=202603_GBM LOOKBACK_DAYS=7 bash 009_production_drift_check.sh
#
# CADENCIA RECOMENDADA: cron diario, mensaje a slack/telegram si !=OK

set -euo pipefail

export RELEASE=${RELEASE:-202603_GBM}
export LOOKBACK_DAYS=${LOOKBACK_DAYS:-30}
export PRODUCTION_BASE=${PRODUCTION_BASE:-production}

PROD_DIR=${PRODUCTION_BASE}/${RELEASE}
OUT_JSON=${OUT_JSON:-${PROD_DIR}/drift_report.json}

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "PRODUCTION DRIFT CHECK"
echo "  Release:       ${RELEASE}"
echo "  Production:    ${PROD_DIR}"
echo "  Lookback:      ${LOOKBACK_DAYS} días"

[ -d "${PROD_DIR}" ] || abort "${PROD_DIR} no existe. ¿Has deployado?"
[ -f "${PROD_DIR}/production_threshold.json" ] || \
  abort "${PROD_DIR}/production_threshold.json no existe — corre 008_deploy_long_only_gbm.sh primero"

if [ ! -d "${PROD_DIR}/logs" ] || [ -z "$(ls -A ${PROD_DIR}/logs/gbm_inference_*.jsonl 2>/dev/null)" ]; then
  echo "⚠️  Sin logs en ${PROD_DIR}/logs/. El servicio s2_main_gbm.py probablemente no ha corrido aún."
  exit 0
fi

python3 -m mimo.oof.diag_production_drift \
  --release "${RELEASE}" \
  --production-base "${PRODUCTION_BASE}" \
  --lookback-days "${LOOKBACK_DAYS}" \
  --out-json "${OUT_JSON}"

# Exit code según veredicto (útil para integrar con monitoring)
VERDICT=$(python3 -c "
import json
r = json.load(open('${OUT_JSON}'))
print(r['verdict'])
")

echo ""
echo "Final verdict: ${VERDICT}"

# Exit codes:
#   0 = OK
#   1 = WARNING (drift moderado)
#   2 = RETUNE_NOW (drift fuerte)
case "${VERDICT}" in
  *OK*)
    exit 0
    ;;
  *WARNING*)
    exit 1
    ;;
  *RETUNE_NOW*)
    exit 2
    ;;
  *)
    exit 0
    ;;
esac
