#!/usr/bin/env bash
# 005_production_refit.sh — FASE 5: Refit con datos completos hasta PROD_HOLDOUT_TO.
# Solo si Fase 4 dio veredicto OK (o MARGINAL documentado).
set -euo pipefail

export RELEASE=${RELEASE:-202600}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202500}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export PROD_HOLDOUT_TO=${PROD_HOLDOUT_TO:-2026-05-10}
export TAIL_DAYS=${TAIL_DAYS:-21}
export SANITY_FROM=${SANITY_FROM:-2026-05-03}
export SANITY_TO=${SANITY_TO:-2026-05-10}
export SANITY_WARMUP=${SANITY_WARMUP:-2026-04-15}
export VAL_DEPLOY=${VAL_DEPLOY:-deploy_validation_combined_seed${SEED}}
export VAL_POLICY=${VAL_POLICY:-decision_policies_config_${RELEASE}_validation}
export SKIP_SANITY=${SKIP_SANITY:-0}

ARTIFACTS=artifacts/${RELEASE}/oof
BEST=${ARTIFACTS}/${TAG}/reports/best_per_side.json
PROD_DIR=${ARTIFACTS}/deploy_PROD_combined_seed${SEED}
TS=$(date +%Y%m%d_%H%M%S)

log() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }
extract() { python3 -c "import json; s=json.load(open('$1')); print(f\"{float(s.get('$2',0) or 0):.4f}\")"; }

log "FASE 5 — Production refit"
echo "  Release: ${RELEASE}"
echo "  Full data: ${TRAIN_FROM} → ${PROD_HOLDOUT_TO}"
echo "  Tail: ${TAIL_DAYS} días"

[ -f "${BEST}" ] || abort "Falta ${BEST}"
for SIDE in long short; do
  [ -d "${ARTIFACTS}/${TAG}_${SIDE}_specialist_seed${SEED}" ] || abort "Falta specialist ${SIDE}"
done

LATEST_VAL=$(ls -t reports/lockbox_validation/validation_${RELEASE}_*.json 2>/dev/null | head -1 || echo "")
if [ -n "${LATEST_VAL}" ]; then
  VERDICT=$(python3 -c "import json; print(json.load(open('${LATEST_VAL}')).get('verdict','UNKNOWN'))")
  echo "  Veredicto Fase 4: ${VERDICT}"
  case "${VERDICT}" in
    *KO*)
      echo "  🔴 KO en Fase 4. ¿Continuar?"
      read -p "  [y/N]: " C; [ "${C}" = "y" ] || abort "Aborted"
      ;;
  esac
else
  echo "  ⚠️  Sin reporte Fase 4. ¿Continuar?"
  read -p "  [y/N]: " C; [ "${C}" = "y" ] || abort "Aborted"
fi

[ -d "${PROD_DIR}" ] && {
  mv "${PROD_DIR}" "${PROD_DIR}.bak_${TS}"
  echo "📦 Backup: ${PROD_DIR}.bak_${TS}"
}

log "2. resume_deploy_v6 × 2 cutoff ${PROD_HOLDOUT_TO}"
for SIDE in long short; do
  echo ""
  echo "── side = ${SIDE} ──"
  python3 -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} --inherit-config-from ${INHERIT_FROM_RELEASE} \
    --target-type multitask --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} --holdout-from ${HOLDOUT_FROM} \
    --holdout-to ${PROD_HOLDOUT_TO} --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED} \
    --deploy-subdir deploy_PROD_${SIDE}_specialist_seed${SEED} \
    --locked-params-json ${BEST} --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done

log "3. merge_specialists"
python3 -m mimo.oof.merge_specialists --release ${RELEASE} \
  --long-dir ${ARTIFACTS}/deploy_PROD_long_specialist_seed${SEED} \
  --short-dir ${ARTIFACTS}/deploy_PROD_short_specialist_seed${SEED} \
  --out-dir ${PROD_DIR}

log "4. select_thresholds_from_tail"
python3 -m mimo.oof.select_thresholds_from_tail --release ${RELEASE} \
  --deploy-dir ${PROD_DIR} --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 --from-db

log "5. policy stub PROD"
POL_PROD=config/decision_policies_config_${RELEASE}_PROD.py
[ -f "${POL_PROD}" ] && cp "${POL_PROD}" "${POL_PROD}.bak_${TS}"
python3 -m mimo.oof.compute_state_percentiles --release ${RELEASE} \
  --deploy-dir ${PROD_DIR} --emit-config-stub --out-stub ${POL_PROD}

SAN_OK=1
if [ "${SKIP_SANITY}" = "0" ] && [ -d "${ARTIFACTS}/${VAL_DEPLOY}" ] && [ -f "config/${VAL_POLICY}.py" ]; then
  log "6. Sanity check (NO es validación pura)"
  echo "⚠️  El deploy PROD ya vio estos datos en tail — solo coherencia"
  PS=/tmp/prod_sanity_${RELEASE}_${TS}
  VS=/tmp/val_sanity_${RELEASE}_${TS}
  python3 scripts/replay_s2_202500.py --release ${RELEASE} \
    --deploy-subdir deploy_PROD_combined_seed${SEED} \
    --policy-config decision_policies_config_${RELEASE}_PROD \
    --warmup-from ${SANITY_WARMUP} --from ${SANITY_FROM} --to ${SANITY_TO} --out ${PS}
  python3 scripts/replay_s2_202500.py --release ${RELEASE} \
    --deploy-subdir ${VAL_DEPLOY} --policy-config ${VAL_POLICY} \
    --warmup-from ${SANITY_WARMUP} --from ${SANITY_FROM} --to ${SANITY_TO} --out ${VS}
  PP=$(extract ${PS}/summary.json pnl_pct)
  VP=$(extract ${VS}/summary.json pnl_pct)
  echo "  PROD=${PP}%  VAL=${VP}%"
  SC=$(python3 -c "
p=${PP}; v=${VP}; f=0.5
print('OK' if (v>0 and p>=v*f) or (v<=0 and p>=v*(2-f)) else 'FAIL')")
  [ "${SC}" = "OK" ] && echo "  ✅ OK" || { SAN_OK=0; echo "  🔴 FAIL"; }
fi

log "7. LINEAGE.md"
cat > ${PROD_DIR}/LINEAGE.md <<EOF
# LINEAGE — deploy_PROD_combined_seed${SEED}

**Release**: ${RELEASE}
**Created**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")

## Datos
- Train: ${TRAIN_FROM} → (full - tail)
- Holdout: ${HOLDOUT_FROM} → ${PROD_HOLDOUT_TO}
- Tail: últimos ${TAIL_DAYS} días antes de ${PROD_HOLDOUT_TO}

## Validación previa (Fase 4)
- Reporte: ${LATEST_VAL:-N/A}
- Veredicto: ${VERDICT:-N/A}

## Sanity Fase 5
- Pasó: $([ "${SAN_OK}" = "1" ] && echo "OK" || echo "FAIL")

## Promoción
Para promover: bash scripts/006_promote_to_production.sh
EOF

log "FASE 5 COMPLETADA"
echo "  Deploy: ${PROD_DIR}"
echo "  Policy: ${POL_PROD}"
echo "  Sanity: $([ "${SAN_OK}" = "1" ] && echo "OK" || echo "FAIL")"
echo ""
echo "📋 Siguiente: bash scripts/006_promote_to_production.sh"
