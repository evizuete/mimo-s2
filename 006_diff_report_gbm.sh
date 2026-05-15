#!/usr/bin/env bash
#
# 006_diff_report_gbm.sh — Cross-release diff report.
#
# Lee los JSONs de TODOS los releases en RELEASES y los pone en una sola
# tabla markdown comparativa. NO requiere BD, NO requiere refit.
# Sólo necesita los JSONs persistidos por fases 1-4.
#
# USO:
#   bash 006_diff_report_gbm.sh
#   RELEASES="202602_GBM 202603_GBM" bash 006_diff_report_gbm.sh

set -euo pipefail

export RELEASES=${RELEASES:-"202602_GBM 202603_GBM"}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export OUT_MD=${OUT_MD:-artifacts/_cross_release_report.md}

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }

log_section "CROSS-RELEASE DIFF REPORT"
echo "  Releases: ${RELEASES}"
echo "  Tag:      ${TAG}"
echo "  Output:   ${OUT_MD}"

python3 -m mimo.oof.diag_diff_report_gbm \
  --releases ${RELEASES} \
  --tag ${TAG} \
  --out-md ${OUT_MD}

log_section "DONE"
echo "  Markdown: ${OUT_MD}"
echo ""
echo "📋 Para visualizar la tabla:"
echo "   cat ${OUT_MD}"
echo "   o cargarlo en cualquier visor de markdown (VS Code, GitHub preview, etc.)"
