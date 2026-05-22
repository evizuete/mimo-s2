#!/usr/bin/env bash
#
# 008_recalibrate_thresholds.sh
# ════════════════════════════════════════════════════════════════════
# MANTENIMIENTO — Recalibración de thresholds del StateDetector.
# (Ver docs/RUNBOOK.md → Apéndice C para flujo completo.)
#
# Recalcula los 6 thresholds del StateDetector (vol_low, vol_high, bb_p20,
# bb_p35, bb_p70, rexp_p80) sobre datos recientes y los persiste en los
# meta.json del deploy activo. NO toca el modelo TCN/CNN-LSTM ni los
# calibradores.
#
# CUÁNDO EJECUTARLO
# ─────────────────
#   · Mensual de rutina (cron o ad-hoc) tras unos 30-60 días en producción.
#   · Tras eventos macro que cambien el régimen de volatilidad (FED, ruptura
#     de tendencia macro, crisis geopolítica, etc.).
#   · Cuando el monitoreo diario reporta >50% de tiempo en VOLATILE (o
#     simétricamente, >50% en LOW_VOL).
#
# CUÁNDO *NO* BASTA Y HAY QUE IR A FASE 1 (re-Optuna)
# ───────────────────────────────────────────────────
# Si tras recalibrar el PnL del deploy degrada >10% sostenido vs lockbox,
# el problema NO es solo los thresholds sino la distribución de features
# (atr_norm, bb_width, etc.) ha drifted al punto donde el modelo entrenado
# ya no reconoce los patrones. En ese caso → ciclo completo 001→006.
#
# FLUJO
# ─────
#   1. Por defecto auto-detecta el deploy activo leyendo main/s2_main.py.
#   2. Lanza scripts/recalibrate_regime_thresholds.py en modo dry-run.
#   3. Muestra comparativa old vs new + diagnóstico de drift.
#   4. Si APPLY=1, repite con --apply y persiste meta.json con backup.
#   5. Te recuerda reiniciar s2.
#
# USO
# ───
#   # Dry-run (default — solo muestra comparativa, no escribe):
#   bash 008_recalibrate_thresholds.sh
#
#   # Aplicar:
#   APPLY=1 bash 008_recalibrate_thresholds.sh
#
#   # Override deploy:
#   DEPLOY_SUBDIR=deploy_PROD_combined_seed47 APPLY=1 bash 008_recalibrate_thresholds.sh
#
#   # Override lookback (180 días para entornos muy estables):
#   LOOKBACK_DAYS=180 bash 008_recalibrate_thresholds.sh
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202500}
export LOOKBACK_DAYS=${LOOKBACK_DAYS:-90}
# Si DEPLOY_SUBDIR está vacío → auto-detect desde main/s2_main.py
export DEPLOY_SUBDIR=${DEPLOY_SUBDIR:-}
export APPLY=${APPLY:-0}
# Path al python del venv del proyecto
export PYBIN=${PYBIN:-/home/evizuete/boti/bin/python3}

# ─── Helpers ────────────────────────────────────────────────────────
log_section() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
}

# ─── Banner ─────────────────────────────────────────────────────────
log_section "MANTENIMIENTO — Recalibrate regime thresholds"
echo "  Release:        ${RELEASE}"
echo "  Lookback days:  ${LOOKBACK_DAYS}"
if [ -z "${DEPLOY_SUBDIR}" ]; then
  echo "  Deploy:         (auto-detect desde main/s2_main.py)"
else
  echo "  Deploy:         ${DEPLOY_SUBDIR}"
fi
echo "  Mode:           $( [ "${APPLY}" = "1" ] && echo 'APPLY (escribir meta.json)' || echo 'DRY-RUN (solo comparativa)' )"
echo "  Python:         ${PYBIN}"

# ─── Construir args del python ────────────────────────────────────
ARGS=(--release "${RELEASE}" --days "${LOOKBACK_DAYS}")
if [ -z "${DEPLOY_SUBDIR}" ]; then
  ARGS+=(--auto-detect-deploy)
else
  ARGS+=(--deploy-subdir "${DEPLOY_SUBDIR}")
fi
if [ "${APPLY}" = "1" ]; then
  ARGS+=(--apply)
fi

# ─── Lanzar ────────────────────────────────────────────────────────
log_section "Ejecutando recalibrate_regime_thresholds.py"
"${PYBIN}" scripts/recalibrate_regime_thresholds.py "${ARGS[@]}"

if [ "${APPLY}" != "1" ]; then
  log_section "DRY-RUN COMPLETADO"
  echo "📋 Si los nuevos thresholds te convencen, aplica con:"
  echo "   APPLY=1 bash 008_recalibrate_thresholds.sh"
else
  log_section "RECALIBRACIÓN APLICADA"
  echo "📋 Próximo paso — reiniciar s2 (TARDA ~5s, downtime mínimo):"
  echo ""
  echo "   pkill -f s2_main"
  echo "   sleep 3"
  echo "   cd main"
  echo "   nohup ${PYBIN} s2_main.py > ../logs/s2_recal_\$(date +%Y%m%d_%H%M).log 2>&1 &"
  echo ""
  echo "📋 Verificación post-restart — confirmar que los nuevos thresholds se inyectaron:"
  echo ""
  echo "   sleep 30 && grep -i 'Regime thresholds loaded' \$(ls -t logs/s2_recal_*.log | head -1) | tail -1"
fi
