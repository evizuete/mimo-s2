#!/usr/bin/env bash
#
# 003C_calibrator_swap_experiment.sh
# ════════════════════════════════════════════════════════════════════
# EXPERIMENTO controlado de cambio de calibrador con rollback automático.
#
# Aplica un swap de calibradores siguiendo champion_config.json, revalida
# sobre LOCKBOX, compara PnL contra baseline, y AUTOMÁTICAMENTE:
#   - ADOPTA el swap si mejora ≥ ADOPTION_FACTOR×
#   - ROLLBACK al estado pre-experimento si no mejora
#
# DISEÑO DE SEGURIDAD:
#   1. Backup completo antes de tocar nada (deploy dir + policy stub)
#   2. Replay baseline con calibrador actual (snapshot del comportamiento previo)
#   3. Aplica swap + recalibra thresholds + recompute state_percentiles
#   4. Replay con swap sobre LOCKBOX
#   5. Compara PnL, MDD, weeks_positive entre baseline y swap
#   6. Decisión automática: adoptar o rollback
#
# CUÁNDO USAR:
#   - Tras 003B_calibrator_diagnosis.sh que muestre ECE > 0.08 sostenido
#   - Tras alerta de monitor_health (ECE alto en datos recientes)
#   - Decisión humana puntual de intentar mejorar la calibración
#
# CUÁNDO NO USAR:
#   - Como paso rutinario del pipeline (no lo es — usa 003 normal)
#   - Si baseline funciona bien (PnL > 0, ECE < 0.05)
#   - Inmediatamente antes de operar con dinero real (deja vivir 1+ semana)
#
# USO:
#   bash scripts/003C_calibrator_swap_experiment.sh
#   # o con overrides:
#   RELEASE=202700 ADOPTION_FACTOR=1.10 bash scripts/003C_calibrator_swap_experiment.sh
#   # dry-run (no toca nada, solo muestra plan):
#   DRY_RUN=1 bash scripts/003C_calibrator_swap_experiment.sh
# ════════════════════════════════════════════════════════════════════

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202600}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202500}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}
export LOCKBOX_FROM=${LOCKBOX_FROM:-2026-04-11}
export LOCKBOX_TO=${LOCKBOX_TO:-2026-05-10}
export WARMUP_FROM=${WARMUP_FROM:-2026-03-25}

# Criterio de adopción: PnL swap debe ser >= baseline × ADOPTION_FACTOR
# Default 1.05 = al menos 5% mejor para justificar el cambio
export ADOPTION_FACTOR=${ADOPTION_FACTOR:-1.05}

# MDD máximo aceptable del swap (no aceptar si MDD se duplica vs baseline)
export MDD_DEGRADATION_LIMIT=${MDD_DEGRADATION_LIMIT:-1.5}

# Champion config: por defecto auto-generado, pero puedes pasar uno custom
export CHAMPION_CONFIG=${CHAMPION_CONFIG:-reports/champion_config.json}

# Dry-run: si =1, solo muestra el plan, no ejecuta nada
export DRY_RUN=${DRY_RUN:-0}

ARTIFACTS=artifacts/${RELEASE}/oof
DEPLOY=${ARTIFACTS}/deploy_validation_combined_seed${SEED}
POLICY_VAL=config/decision_policies_config_${RELEASE}_validation.py
TS=$(date +%Y%m%d_%H%M%S)

# ─── Helpers ─────────────────────────────────────────────────────────
log_section() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
}

abort() {
  echo "❌ $1"
  exit 1
}

extract_summary() {
  local summary_file="$1"
  local key="$2"
  python3 -c "
import json, sys
try:
    s = json.load(open('${summary_file}'))
    v = s.get('${key}', 0) or 0
    print(f'{float(v):.4f}')
except Exception as e:
    print('0.0', file=sys.stderr)
    print(f'⚠️  Error leyendo ${key} de ${summary_file}: {e}', file=sys.stderr)
    print('0.0')
"
}

# ─── Banner ─────────────────────────────────────────────────────────
log_section "EXPERIMENTO — Calibrator swap controlado"
echo "  Release:           ${RELEASE}"
echo "  Inherit from:      ${INHERIT_FROM_RELEASE}"
echo "  Deploy:            ${DEPLOY}"
echo "  Policy validation: ${POLICY_VAL}"
echo "  LOCKBOX:           ${LOCKBOX_FROM} → ${LOCKBOX_TO}"
echo "  Adoption factor:   ${ADOPTION_FACTOR}× baseline PnL"
echo "  MDD limit:         ${MDD_DEGRADATION_LIMIT}× baseline MDD"
echo "  Dry-run:           ${DRY_RUN}"
echo "  Timestamp:         ${TS}"

# ─── Pre-checks ─────────────────────────────────────────────────────
log_section "Pre-checks"
[ -d "${DEPLOY}" ] || abort "${DEPLOY} no existe — ejecuta 003 antes"
[ -f "${POLICY_VAL}" ] || abort "${POLICY_VAL} no existe — ejecuta 003 antes"
[ -f "${ARTIFACTS}/${TAG}_long_specialist_seed${SEED}/data/holdout_predictions_${RELEASE}_long.parquet" ] \
  || abort "Falta holdout_predictions LONG — ejecuta 002 antes"
[ -f "${ARTIFACTS}/${TAG}_short_specialist_seed${SEED}/data/holdout_predictions_${RELEASE}_short.parquet" ] \
  || abort "Falta holdout_predictions SHORT — ejecuta 002 antes"
echo "✅ Pre-requisitos OK"

if [ "${DRY_RUN}" = "1" ]; then
  echo ""
  echo "🔍 DRY-RUN: nada se ejecutará. Plan:"
  echo "   1. Backup ${DEPLOY} → ${DEPLOY}.backup_${TS}"
  echo "   2. Backup ${POLICY_VAL} → ${POLICY_VAL}.backup_${TS}"
  echo "   3. Replay baseline sobre LOCKBOX"
  echo "   4. ghost_predict + simulate_calibrators"
  echo "   5. swap_calibrators_per_champion"
  echo "   6. select_thresholds + compute_state_percentiles"
  echo "   7. Replay con swap sobre LOCKBOX"
  echo "   8. Comparativa y decisión automática"
  exit 0
fi

# ─── 1. Backup completo ─────────────────────────────────────────────
log_section "1. Backup pre-experimento"
BACKUP_DEPLOY=${DEPLOY}.backup_${TS}
BACKUP_POLICY=${POLICY_VAL}.backup_${TS}

echo "📦 Backup deploy: ${BACKUP_DEPLOY}"
cp -r "${DEPLOY}" "${BACKUP_DEPLOY}"

echo "📦 Backup policy: ${BACKUP_POLICY}"
cp "${POLICY_VAL}" "${BACKUP_POLICY}"

# Función de rollback (se ejecuta automáticamente al final si no se adopta)
rollback_all() {
  echo ""
  echo "🔄 EJECUTANDO ROLLBACK..."
  rm -rf "${DEPLOY}"
  mv "${BACKUP_DEPLOY}" "${DEPLOY}"
  cp "${BACKUP_POLICY}" "${POLICY_VAL}"
  # Borrar policy stub experimental si se creó
  rm -f "config/decision_policies_config_${RELEASE}_experiment.py"
  echo "✅ Rollback completado — estado restaurado al pre-experimento"
}

# Trap: si el script falla antes de la decisión final, rollback automático
trap 'echo "⚠️  Script abortó — ejecutando rollback de seguridad"; rollback_all; exit 1' ERR

# ─── 2. Replay baseline ─────────────────────────────────────────────
log_section "2. Replay BASELINE sobre LOCKBOX"
BASELINE_OUT=/tmp/replay_baseline_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --policy-config decision_policies_config_${RELEASE}_validation \
  --warmup-from ${WARMUP_FROM} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --out ${BASELINE_OUT}

BASELINE_PNL=$(extract_summary ${BASELINE_OUT}/summary.json pnl_pct)
BASELINE_MDD=$(extract_summary ${BASELINE_OUT}/summary.json max_drawdown_pct)
BASELINE_TRADES=$(extract_summary ${BASELINE_OUT}/summary.json n_trades)
echo ""
echo "  📊 Baseline LOCKBOX:"
echo "     PnL%:    ${BASELINE_PNL}"
echo "     MDD%:    ${BASELINE_MDD}"
echo "     Trades:  ${BASELINE_TRADES}"

# ─── 3. Generar champion_config si no existe ────────────────────────
log_section "3. simulate_calibrators (genera champion_config)"
GHOST=/tmp/ghost_holdout_${RELEASE}_${TS}.parquet

if [ ! -f "${CHAMPION_CONFIG}" ] || [ "${REFRESH_CHAMPION:-0}" = "1" ]; then
  python3 scripts/ghost_predict.py \
    --release ${RELEASE} \
    --deploy-subdir $(basename ${DEPLOY}) \
    --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} \
    --include-tail \
    --out ${GHOST}

  python3 scripts/simulate_calibrators.py \
    --release ${RELEASE} \
    --specialist-tag ${TAG} \
    --seed ${SEED} \
    --test-parquet ${GHOST} \
    --out-report reports/cal_sim_${RELEASE}_${TS}.csv
else
  echo "  Reusing existing champion_config: ${CHAMPION_CONFIG}"
fi

[ -f "${CHAMPION_CONFIG}" ] || abort "champion_config no se generó"

# ─── 4. Aplicar swap ────────────────────────────────────────────────
log_section "4. Aplicar swap de calibradores"
python3 scripts/swap_calibrators_per_champion.py \
  --release ${RELEASE} \
  --specialist-tag ${TAG} \
  --seed ${SEED} \
  --champion-config ${CHAMPION_CONFIG} \
  --deploy-dir ${DEPLOY}

# ─── 5. Re-calibrar thresholds + percentiles ────────────────────────
log_section "5. Re-select thresholds + compute_state_percentiles"

python3 -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir ${DEPLOY} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

POLICY_EXP=config/decision_policies_config_${RELEASE}_experiment.py
python3 -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir ${DEPLOY} \
  --emit-config-stub \
  --out-stub ${POLICY_EXP}

# ─── 6. Replay con swap ─────────────────────────────────────────────
log_section "6. Replay SWAP sobre LOCKBOX"
SWAP_OUT=/tmp/replay_swap_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --policy-config decision_policies_config_${RELEASE}_experiment \
  --warmup-from ${WARMUP_FROM} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --out ${SWAP_OUT}

SWAP_PNL=$(extract_summary ${SWAP_OUT}/summary.json pnl_pct)
SWAP_MDD=$(extract_summary ${SWAP_OUT}/summary.json max_drawdown_pct)
SWAP_TRADES=$(extract_summary ${SWAP_OUT}/summary.json n_trades)
echo ""
echo "  📊 Swap LOCKBOX:"
echo "     PnL%:    ${SWAP_PNL}"
echo "     MDD%:    ${SWAP_MDD}"
echo "     Trades:  ${SWAP_TRADES}"

# Desactivar trap de ERR — el resto del script maneja decisiones manualmente
trap - ERR

# ─── 7. Comparativa y decisión ──────────────────────────────────────
log_section "7. COMPARATIVA Y DECISIÓN"

echo ""
echo "                          Baseline     Swap     Δ"
echo "  ───────────────────────────────────────────────────"
printf "  PnL%%:                  %+8.2f   %+8.2f   %+8.2f\n" \
  "${BASELINE_PNL}" "${SWAP_PNL}" \
  "$(python3 -c "print(${SWAP_PNL} - ${BASELINE_PNL})")"
printf "  MDD%%:                  %+8.2f   %+8.2f\n" \
  "${BASELINE_MDD}" "${SWAP_MDD}"
printf "  Trades:                %8.0f   %8.0f\n" \
  "${BASELINE_TRADES}" "${SWAP_TRADES}"
echo ""

# Lógica de decisión:
#   - Si baseline PnL <= 0: solo adopta si swap es claramente positivo
#   - Si baseline > 0: swap debe ser >= baseline × ADOPTION_FACTOR
#   - En ambos casos: MDD swap no debe superar baseline × MDD_DEGRADATION_LIMIT
DECISION=$(python3 <<EOF
baseline_pnl = ${BASELINE_PNL}
swap_pnl = ${SWAP_PNL}
baseline_mdd = abs(${BASELINE_MDD})
swap_mdd = abs(${SWAP_MDD})
factor = ${ADOPTION_FACTOR}
mdd_limit = ${MDD_DEGRADATION_LIMIT}

reasons = []

# Check 1: MDD no debe degradarse
if baseline_mdd > 0.1 and swap_mdd > baseline_mdd * mdd_limit:
    reasons.append(f"MDD se degrada: {swap_mdd:.2f}% > {baseline_mdd:.2f}% × {mdd_limit}")
    print("ROLLBACK")
    print("  Razón:", reasons[0])
    raise SystemExit

# Check 2: PnL debe mejorar suficientemente
if baseline_pnl <= 0:
    # Caso especial: baseline en negativo, swap solo si claramente positivo
    if swap_pnl > 0 and swap_pnl > 1.0:
        print("ADOPT")
        print(f"  Razón: baseline pérdida ({baseline_pnl:+.2f}%), swap gana ({swap_pnl:+.2f}%)")
    else:
        print("ROLLBACK")
        print(f"  Razón: ni baseline ni swap son claramente positivos (baseline={baseline_pnl:+.2f}%, swap={swap_pnl:+.2f}%)")
else:
    threshold = baseline_pnl * factor
    if swap_pnl >= threshold:
        print("ADOPT")
        print(f"  Razón: swap ({swap_pnl:+.2f}%) >= baseline × {factor} ({threshold:+.2f}%)")
    else:
        print("ROLLBACK")
        print(f"  Razón: swap ({swap_pnl:+.2f}%) < baseline × {factor} ({threshold:+.2f}%)")
EOF
)

DECISION_VERDICT=$(echo "$DECISION" | head -1)
DECISION_REASON=$(echo "$DECISION" | tail -n +2)

echo "  📊 DECISIÓN: ${DECISION_VERDICT}"
echo "${DECISION_REASON}"
echo ""

# ─── 8. Ejecutar acción ─────────────────────────────────────────────
if [ "${DECISION_VERDICT}" = "ADOPT" ]; then
  log_section "8. ADOPTAR — Swap aceptado"
  echo "✅ El swap mejora suficiente. Se mantiene en el deploy."
  echo ""
  echo "  Deploy actualizado:     ${DEPLOY}"
  echo "  Policy stub nuevo:      ${POLICY_EXP}"
  echo "  Policy stub anterior:   ${BACKUP_POLICY} (backup, no se restaura)"
  echo "  Backup completo:        ${BACKUP_DEPLOY} (puedes borrar manualmente cuando quieras)"
  echo ""
  echo "📋 PRÓXIMOS PASOS:"
  echo "  1. Actualizar s2_main.py para apuntar al nuevo policy:"
  echo "       from config.decision_policies_config_${RELEASE}_experiment import ..."
  echo "  2. Reinicia el servicio s2."
  echo "  3. Monitor durante 1+ semana antes de confirmar promoción definitiva."
  echo ""
  echo "📋 SI EL SWAP DEGRADA EN VIVO:"
  echo "  bash scripts/rollback_deploy.py \\"
  echo "    --target-deploy <previous> --target-policy decision_policies_config_${RELEASE}_validation \\"
  echo "    --apply"
else
  log_section "8. ROLLBACK — Swap NO adoptado"
  rollback_all
  rm -f "${POLICY_EXP}"
  echo ""
  echo "  Deploy restaurado:      ${DEPLOY}"
  echo "  Policy original:        ${POLICY_VAL}"
  echo "  Backup mantenido:       ${BACKUP_DEPLOY} (auto-cleanup en próxima ejecución)"
  echo ""
  echo "📋 OUTPUTS PARA ANÁLISIS:"
  echo "  Replay baseline:        ${BASELINE_OUT}/"
  echo "  Replay swap (rejected): ${SWAP_OUT}/"
  echo "  Comparativa cal:        reports/cal_sim_${RELEASE}_${TS}.csv"
fi

# ─── 9. Resumen y cleanup ───────────────────────────────────────────
echo ""
log_section "EXPERIMENTO COMPLETADO"
echo "  Timestamp:          ${TS}"
echo "  Decisión:           ${DECISION_VERDICT}"
echo "  Baseline PnL%:      ${BASELINE_PNL}"
echo "  Swap PnL%:          ${SWAP_PNL}"
echo "  Baseline MDD%:      ${BASELINE_MDD}"
echo "  Swap MDD%:          ${SWAP_MDD}"
echo ""

# Persistir log del experimento para auditoría
mkdir -p reports/experiments
cat > reports/experiments/calibrator_swap_${RELEASE}_${TS}.json <<EOF
{
  "timestamp": "${TS}",
  "release": "${RELEASE}",
  "lockbox_window": ["${LOCKBOX_FROM}", "${LOCKBOX_TO}"],
  "champion_config": "${CHAMPION_CONFIG}",
  "adoption_factor": ${ADOPTION_FACTOR},
  "baseline": {
    "pnl_pct": ${BASELINE_PNL},
    "mdd_pct": ${BASELINE_MDD},
    "n_trades": ${BASELINE_TRADES}
  },
  "swap": {
    "pnl_pct": ${SWAP_PNL},
    "mdd_pct": ${SWAP_MDD},
    "n_trades": ${SWAP_TRADES}
  },
  "decision": "${DECISION_VERDICT}",
  "decision_reason": "$(echo "${DECISION_REASON}" | tr -d '\n' | sed 's/"/\\"/g')",
  "outputs": {
    "baseline_replay": "${BASELINE_OUT}",
    "swap_replay": "${SWAP_OUT}",
    "policy_experiment": "${POLICY_EXP}",
    "backup_deploy": "${BACKUP_DEPLOY}",
    "backup_policy": "${BACKUP_POLICY}"
  }
}
EOF

echo "📁 Log del experimento: reports/experiments/calibrator_swap_${RELEASE}_${TS}.json"
echo ""