#!/usr/bin/env bash
# 003C_calibrator_swap_experiment.sh — EXPERIMENTO con rollback automático.
#
# Aplica swap del champion_config, valida sobre LOCKBOX, decide ADOPT/ROLLBACK.
# Si rechaza → rollback automático al estado pre-experimento.
set -euo pipefail

export RELEASE=${RELEASE:-202600}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}
export LOCKBOX_FROM=${LOCKBOX_FROM:-2026-04-11}
export LOCKBOX_TO=${LOCKBOX_TO:-2026-05-10}
export WARMUP_FROM=${WARMUP_FROM:-2026-03-25}
export ADOPTION_FACTOR=${ADOPTION_FACTOR:-1.05}
export MDD_DEGRADATION_LIMIT=${MDD_DEGRADATION_LIMIT:-1.5}
export CHAMPION_CONFIG=${CHAMPION_CONFIG:-reports/champion_config.json}
export DRY_RUN=${DRY_RUN:-0}

ARTIFACTS=artifacts/${RELEASE}/oof
DEPLOY=${ARTIFACTS}/deploy_validation_combined_seed${SEED}
POLICY_VAL=config/decision_policies_config_${RELEASE}_validation.py
TS=$(date +%Y%m%d_%H%M%S)

log() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }
extract() { python3 -c "import json; s=json.load(open('$1')); print(f\"{float(s.get('$2',0) or 0):.4f}\")"; }

log "EXPERIMENTO — Calibrator swap controlado"
echo "  Deploy: ${DEPLOY}"
echo "  LOCKBOX: ${LOCKBOX_FROM} → ${LOCKBOX_TO}"
echo "  Adoption factor: ${ADOPTION_FACTOR}× baseline PnL"

[ -d "${DEPLOY}" ] || abort "${DEPLOY} no existe"
[ -f "${POLICY_VAL}" ] || abort "${POLICY_VAL} no existe"

if [ "${DRY_RUN}" = "1" ]; then
  echo "🔍 DRY-RUN: nada se ejecutará"; exit 0
fi

log "1. Backup pre-experimento"
BACKUP_DEPLOY=${DEPLOY}.backup_${TS}
BACKUP_POL=${POLICY_VAL}.backup_${TS}
cp -r "${DEPLOY}" "${BACKUP_DEPLOY}"
cp "${POLICY_VAL}" "${BACKUP_POL}"
echo "📦 ${BACKUP_DEPLOY}"

rollback_all() {
  echo "🔄 ROLLBACK..."
  rm -rf "${DEPLOY}"; mv "${BACKUP_DEPLOY}" "${DEPLOY}"
  cp "${BACKUP_POL}" "${POLICY_VAL}"
  rm -f "config/decision_policies_config_${RELEASE}_experiment.py"
  echo "✅ Restaurado"
}
trap 'rollback_all; exit 1' ERR

log "2. Replay BASELINE"
BL=/tmp/replay_baseline_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --policy-config decision_policies_config_${RELEASE}_validation \
  --warmup-from ${WARMUP_FROM} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} --out ${BL}
BL_PNL=$(extract ${BL}/summary.json pnl_pct)
BL_MDD=$(extract ${BL}/summary.json max_drawdown_pct)
echo "  Baseline: PnL=${BL_PNL}% MDD=${BL_MDD}%"

log "3. simulate_calibrators"
GHOST=/tmp/ghost_holdout_${RELEASE}_${TS}.parquet
if [ ! -f "${CHAMPION_CONFIG}" ] || [ "${REFRESH:-0}" = "1" ]; then
  python3 scripts/ghost_predict.py --release ${RELEASE} \
    --deploy-subdir $(basename ${DEPLOY}) \
    --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} --include-tail --out ${GHOST}
  python3 scripts/simulate_calibrators.py --release ${RELEASE} \
    --specialist-tag ${TAG} --seed ${SEED} --test-parquet ${GHOST} \
    --out-report reports/cal_sim_${RELEASE}_${TS}.csv
fi
[ -f "${CHAMPION_CONFIG}" ] || abort "champion_config no se generó"

log "4. Aplicar swap"
python3 scripts/swap_calibrators_per_champion.py --release ${RELEASE} \
  --specialist-tag ${TAG} --seed ${SEED} \
  --champion-config ${CHAMPION_CONFIG} --deploy-dir ${DEPLOY}

log "5. Re-select thresholds + percentiles"
python3 -m mimo.oof.select_thresholds_from_tail --release ${RELEASE} \
  --deploy-dir ${DEPLOY} --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 --from-db

POL_EXP=config/decision_policies_config_${RELEASE}_experiment.py
python3 -m mimo.oof.compute_state_percentiles --release ${RELEASE} \
  --deploy-dir ${DEPLOY} --emit-config-stub --out-stub ${POL_EXP}

log "6. Replay con swap"
SW=/tmp/replay_swap_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --policy-config decision_policies_config_${RELEASE}_experiment \
  --warmup-from ${WARMUP_FROM} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} --out ${SW}
SW_PNL=$(extract ${SW}/summary.json pnl_pct)
SW_MDD=$(extract ${SW}/summary.json max_drawdown_pct)
echo "  Swap: PnL=${SW_PNL}% MDD=${SW_MDD}%"

trap - ERR

log "7. DECISIÓN"
echo "             Baseline    Swap"
printf "  PnL%%:      %+8.2f  %+8.2f\n" "${BL_PNL}" "${SW_PNL}"
printf "  MDD%%:      %+8.2f  %+8.2f\n" "${BL_MDD}" "${SW_MDD}"

DEC=$(python3 <<EOF
bl_pnl = ${BL_PNL}; sw_pnl = ${SW_PNL}
bl_mdd = abs(${BL_MDD}); sw_mdd = abs(${SW_MDD})
factor = ${ADOPTION_FACTOR}; mdd_lim = ${MDD_DEGRADATION_LIMIT}

if bl_mdd > 0.1 and sw_mdd > bl_mdd * mdd_lim:
    print("ROLLBACK")
    print(f"  Razón: MDD se degrada ({sw_mdd:.2f}% > {bl_mdd:.2f}% × {mdd_lim})")
elif bl_pnl <= 0:
    if sw_pnl > 0 and sw_pnl > 1.0:
        print("ADOPT")
        print(f"  Razón: baseline negativo, swap positivo ({sw_pnl:+.2f}%)")
    else:
        print("ROLLBACK")
        print(f"  Razón: ni baseline ni swap claramente positivos")
else:
    thr = bl_pnl * factor
    if sw_pnl >= thr:
        print("ADOPT")
        print(f"  Razón: swap ({sw_pnl:+.2f}%) >= baseline×{factor} ({thr:+.2f}%)")
    else:
        print("ROLLBACK")
        print(f"  Razón: swap ({sw_pnl:+.2f}%) < baseline×{factor} ({thr:+.2f}%)")
EOF
)

VERDICT=$(echo "$DEC" | head -1)
REASON=$(echo "$DEC" | tail -1)
echo "  📊 ${VERDICT}: ${REASON}"

if [ "${VERDICT}" = "ADOPT" ]; then
  log "✅ ADOPTAR"
  echo "Swap aplicado. Policy: ${POL_EXP}"
  echo "Para promover a PROD: 006_promote_to_production.sh con NEW_POLICY=${POL_EXP%.py}"
  echo "Backup mantenido: ${BACKUP_DEPLOY}"
else
  log "🔴 ROLLBACK"
  rollback_all
fi

mkdir -p reports/experiments
cat > reports/experiments/calibrator_swap_${RELEASE}_${TS}.json <<EOF
{"timestamp":"${TS}","release":"${RELEASE}",
 "baseline":{"pnl_pct":${BL_PNL},"mdd_pct":${BL_MDD}},
 "swap":{"pnl_pct":${SW_PNL},"mdd_pct":${SW_MDD}},
 "decision":"${VERDICT}","reason":"${REASON}"}
EOF
echo "📁 Log: reports/experiments/calibrator_swap_${RELEASE}_${TS}.json"
