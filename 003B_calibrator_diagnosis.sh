#!/usr/bin/env bash
# 003B_calibrator_diagnosis.sh — Diagnóstico informativo. NO modifica deploy.
#
# Veredicto automático:
#   ✅ OK         — ECE < 0.05, sin acción
#   🟡 WATCH      — ECE 0.05-0.08, vigilar
#   ⚠️  EXPERIMENT — ECE 0.08-0.15 + alt mejora ≥30% → ejecutar 003C
#   🔴 ALERT      — ECE > 0.15, redeploy
set -euo pipefail

cd /mnt/c/Users/Usuario/Documents/Proyectos/TradingCo_s2
export PYTHONPATH="$PWD"

export RELEASE=${RELEASE:-202600}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}
export ECE_OK=${ECE_OK:-0.05}
export ECE_WATCH=${ECE_WATCH:-0.08}
export ECE_ALERT=${ECE_ALERT:-0.15}
export MIN_IMPROVEMENT=${MIN_IMPROVEMENT:-0.30}

DEPLOY=artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED}
TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=reports/calibrator_diagnosis
mkdir -p ${REPORT_DIR}

log() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log "DIAGNÓSTICO DE CALIBRADORES (informativo)"
echo "  Release: ${RELEASE}  Deploy: ${DEPLOY}"
echo "  Holdout: ${HOLDOUT_FROM} → ${HOLDOUT_TO}"

log "1. Pre-checks"
[ -d "${DEPLOY}" ] || abort "${DEPLOY} no existe — ejecuta 003 antes"
echo "✅ Pre-checks OK"

log "2. ghost_predict + simulate_calibrators"
GHOST=/tmp/ghost_holdout_${RELEASE}_${TS}.parquet
python3 scripts/ghost_predict.py --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} --include-tail --out ${GHOST}

CAL_REPORT=${REPORT_DIR}/cal_sim_${RELEASE}_${TS}.csv
python3 scripts/simulate_calibrators.py --release ${RELEASE} \
  --specialist-tag ${TAG} --seed ${SEED} \
  --test-parquet ${GHOST} --out-report ${CAL_REPORT}

log "3. Veredicto automático"
python3 <<EOF
import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("${CAL_REPORT}")
ece_ok, ece_watch, ece_alert = ${ECE_OK}, ${ECE_WATCH}, ${ECE_ALERT}
min_imp = ${MIN_IMPROVEMENT}
baseline = "iso_21d"

verdicts = {}
for side in df["side"].unique():
    ds = df[df["side"] == side]
    bl = ds[ds["method"] == baseline]
    if bl.empty:
        verdicts[side] = {"status": "ERROR"}; continue
    bl_ece = float(bl["ece"].iloc[0])
    alts = ds[ds["method"] != baseline].sort_values("ece")
    best = alts.iloc[0].to_dict() if not alts.empty else None

    if bl_ece < ece_ok: status, action = "✅ OK", "Sin acción"
    elif bl_ece < ece_watch: status, action = "🟡 WATCH", "Vigilar"
    elif bl_ece > ece_alert: status, action = "🔴 ALERT", "Redeploy"
    elif best:
        imp = (bl_ece - float(best["ece"])) / bl_ece
        if imp >= min_imp:
            status = "⚠️  EXPERIMENT"
            action = f"003C: {best['method']} reduce ECE {bl_ece:.4f}→{best['ece']:.4f}"
        else:
            status, action = "🟡 WATCH", "ECE moderado sin alternativa clara"
    else:
        status, action = "🟡 WATCH", "Sin alternativas"

    verdicts[side] = {"status": status, "ece": bl_ece, "action": action}
    print(f"\n  {side.upper()}: {status} — ECE={bl_ece:.4f}")
    print(f"    Acción: {action}")

prio = {"🔴 ALERT": 4, "⚠️  EXPERIMENT": 3, "🟡 WATCH": 2, "✅ OK": 1, "ERROR": 0}
gs = max(verdicts.values(), key=lambda v: prio.get(v["status"], 0))["status"]
print(f"\n  📊 VEREDICTO GLOBAL: {gs}")

Path("${REPORT_DIR}/verdict_${RELEASE}_${TS}.json").write_text(
    json.dumps({"timestamp": "${TS}", "release": "${RELEASE}",
                "global_status": gs, "by_side": verdicts}, indent=2, default=str))
EOF

log "DIAGNÓSTICO COMPLETADO"
echo "  Reporte:  ${CAL_REPORT}"
echo "  Veredicto: ${REPORT_DIR}/verdict_${RELEASE}_${TS}.json"
echo "  ℹ️  NO modifica el deploy. Si veredicto EXPERIMENT → ejecuta 003C."
