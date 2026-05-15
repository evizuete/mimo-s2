#!/usr/bin/env bash
#
# 007_long_only_simulator.sh — Tarjeta de presentación del modelo.
#
# Lee el walkforward_report.json (preferir el raw_probs) y produce métricas
# comparativas de 3 estrategias: LONG-only, SHORT-only, COMBINED.
#
# Para cada estrategia: R total, mean/median/std/Sharpe/Sortino/Calmar/
# Max DD/Win rate/Profit factor/Max consec losses/Skew/Kurt + equity curve.
#
# USO:
#   bash 007_long_only_simulator.sh
#   WF_JSON=walkforward_report.json bash 007_long_only_simulator.sh
#
# La salida da un veredicto cualitativo: ✅ STRATEGY OK / ⚠️ MARGINAL / ❌

set -euo pipefail

export RELEASE=${RELEASE:-202603_GBM}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}

REPORTS_DIR=artifacts/${RELEASE}/oof/${TAG}/reports

# Por defecto usar walkforward_report_raw.json (raw probs). Fallback al
# calibrated si no existe.
WF_JSON=${WF_JSON:-${REPORTS_DIR}/walkforward_report_raw.json}
if [ ! -f "${WF_JSON}" ]; then
  echo "⚠️  walkforward_report_raw.json no existe, intento walkforward_report.json"
  WF_JSON=${REPORTS_DIR}/walkforward_report.json
fi

OUT_JSON=${OUT_JSON:-${REPORTS_DIR}/long_only_sim.json}

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "LONG-ONLY SIMULATOR — ${RELEASE} / ${TAG}"
echo "  Walkforward JSON: ${WF_JSON}"

[ -f "${WF_JSON}" ] || abort "${WF_JSON} no existe. Corre fase 3 primero."

python3 -m mimo.oof.diag_long_only_simulator \
  --walkforward-json "${WF_JSON}" \
  --periods-per-year 12 \
  --out-json "${OUT_JSON}"

echo ""
echo "📁 Reporte JSON: ${OUT_JSON}"
