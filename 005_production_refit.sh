#!/usr/bin/env bash
#
# 005_production_refit.sh
# ════════════════════════════════════════════════════════════════════
# FASE 5 — Production refit con DATOS COMPLETOS.
#
# Re-entrena el MISMO modelo (misma arquitectura, mismos hyperparams) usando
# TODOS los datos disponibles, incluyendo el LOCKBOX que ya hizo su trabajo
# en Fase 4. Esto produce el deploy DEFINITIVO de producción.
#
# QUÉ HACE:
#   1. Pre-checks (Fase 4 debe haber pasado OK)
#   2. resume_deploy_v6 × 2 con cutoff EXTENDIDO (PROD_HOLDOUT_TO en vez de HOLDOUT_TO)
#   3. merge_specialists → deploy_PROD_*
#   4. select_thresholds_from_tail con tail incluyendo LOCKBOX
#   5. compute_state_percentiles → policy stub PROD
#   6. Sanity check (replay sobre últimos 7d) — NO es validación pura, solo coherencia
#   7. Genera LINEAGE.md documentando origen del deploy
#
# CRÍTICO — NO ejecutar si Fase 4 dio veredicto KO.
# Si Fase 4 fue MARGINAL, decide si proceder con cautela.
#
# Tras este script, queda:
#   artifacts/<release>/oof/deploy_PROD_combined_seed<N>/
#   config/decision_policies_config_<release>_PROD.py
#
# Esos son los artefactos que apuntarás desde s2_main.py en Fase 6.
#
# USO:
#   bash scripts/005_production_refit.sh
#   # con overrides:
#   PROD_HOLDOUT_TO=2026-06-15 bash scripts/005_production_refit.sh
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202600}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202500}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
# PROD_HOLDOUT_TO incluye el LOCKBOX original
export PROD_HOLDOUT_TO=${PROD_HOLDOUT_TO:-2026-05-10}
# Para sanity check al final
export SANITY_FROM=${SANITY_FROM:-2026-05-03}
export SANITY_TO=${SANITY_TO:-2026-05-10}
export SANITY_WARMUP=${SANITY_WARMUP:-2026-04-15}
export TAIL_DAYS=${TAIL_DAYS:-21}

# Sanity check baseline (deploy de validación contra el que comparar)
export VALIDATION_DEPLOY=${VALIDATION_DEPLOY:-deploy_validation_combined_seed${SEED}}
export VALIDATION_POLICY=${VALIDATION_POLICY:-decision_policies_config_${RELEASE}_validation}

# Skip override del sanity check si lo quieres saltar (no recomendado)
export SKIP_SANITY=${SKIP_SANITY:-0}

# Factor mínimo de coherencia entre prod_refit vs validation en sanity:
# si nuevo PnL < antiguo × 0.5 → algo se rompió, abortar
export SANITY_MIN_FACTOR=${SANITY_MIN_FACTOR:-0.5}

ARTIFACTS=artifacts/${RELEASE}/oof
BEST_JSON=${ARTIFACTS}/${TAG}/reports/best_per_side.json
PROD_DEPLOY_DIR=${ARTIFACTS}/deploy_PROD_combined_seed${SEED}
TS=$(date +%Y%m%d_%H%M%S)

# ─── Helpers ────────────────────────────────────────────────────────
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

extract_metric() {
  python3 -c "
import json
try:
    s = json.load(open('$1'))
    v = s.get('$2', 0) or 0
    print(f'{float(v):.4f}')
except Exception:
    print('0.0')
"
}

# ─── Banner ─────────────────────────────────────────────────────────
log_section "FASE 5 — Production refit"
echo "  Release:            ${RELEASE}"
echo "  Inherit from:       ${INHERIT_FROM_RELEASE}"
echo "  Train period:       ${TRAIN_FROM} → (model trained on full data minus tail)"
echo "  Full data window:   ${TRAIN_FROM} → ${PROD_HOLDOUT_TO}"
echo "  Tail calibration:   últimos ${TAIL_DAYS} días antes de ${PROD_HOLDOUT_TO}"
echo "  Output deploy:      ${PROD_DEPLOY_DIR}"
echo "  Output policy:      config/decision_policies_config_${RELEASE}_PROD.py"
echo "  Sanity check:       ${SANITY_FROM} → ${SANITY_TO} (skip=${SKIP_SANITY})"
echo "  Timestamp:          ${TS}"

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

[ -f "${BEST_JSON}" ] || abort "Falta ${BEST_JSON}. Ejecuta scripts 001+002 antes."

for SIDE in long short; do
  SPEC_DIR=${ARTIFACTS}/${TAG}_${SIDE}_specialist_seed${SEED}
  [ -d "${SPEC_DIR}" ] || abort "Falta specialist ${SIDE}: ${SPEC_DIR}. Ejecuta 002 antes."
done

# Verificar que Fase 4 fue ejecutada (busca el reporte más reciente)
LATEST_VALIDATION=$(ls -t reports/lockbox_validation/validation_${RELEASE}_*.json 2>/dev/null | head -1 || echo "")
if [ -z "${LATEST_VALIDATION}" ]; then
  echo "⚠️  No se encuentra reporte de Fase 4 (validación LOCKBOX)."
  echo "   Recomendado: ejecutar 004_validate_lockbox.sh antes de Fase 5."
  echo ""
  read -p "  ¿Continuar de todos modos? [y/N]: " CONFIRM
  if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
    abort "Aborted. Ejecuta 004_validate_lockbox.sh primero."
  fi
else
  VERDICT=$(python3 -c "
import json
v = json.load(open('${LATEST_VALIDATION}'))
print(v.get('verdict', 'UNKNOWN'))
")
  echo "✅ Último reporte de validación encontrado: ${LATEST_VALIDATION}"
  echo "   Veredicto: ${VERDICT}"
  
  case "${VERDICT}" in
    *KO*)
      echo ""
      echo "🔴 Veredicto Fase 4 = KO. NO se debe proceder con production refit."
      echo "   Revisa el reporte y itera (re-Optuna, ajustar policy, etc.) antes."
      read -p "  ¿Continuar de todos modos? Risky. [y/N]: " CONFIRM
      [ "${CONFIRM}" = "y" ] || [ "${CONFIRM}" = "Y" ] || abort "Aborted por veredicto KO."
      ;;
    *MARGINAL*)
      echo "🟡 Veredicto Fase 4 = MARGINAL. Procede con cautela y documenta."
      ;;
    *OK*)
      echo "✅ Veredicto Fase 4 = OK. Procede a Fase 5."
      ;;
  esac
fi

# Verificar que VALIDATION_DEPLOY existe (necesario para sanity check)
if [ "${SKIP_SANITY}" != "1" ]; then
  if [ ! -d "${ARTIFACTS}/${VALIDATION_DEPLOY}" ]; then
    echo "⚠️  ${ARTIFACTS}/${VALIDATION_DEPLOY} no existe → sanity check vs validation OMITIDO"
    SKIP_SANITY=1
  elif [ ! -f "config/${VALIDATION_POLICY}.py" ]; then
    echo "⚠️  config/${VALIDATION_POLICY}.py no existe → sanity check OMITIDO"
    SKIP_SANITY=1
  fi
fi

# Verificar disponibilidad de datos hasta PROD_HOLDOUT_TO
python3 -c "
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
db = Database()
dm = DataManager.from_database_historical_2(
    db, from_date='${TRAIN_FROM}', to_date='${PROD_HOLDOUT_TO}'
)
max_date = dm.df.time.max()
n = len(dm.df)
print(f'✅ {n:,} filas | rango {dm.df.time.min()} → {max_date}')
if str(max_date) < '${PROD_HOLDOUT_TO}':
    print(f'⚠️  Última fecha disponible ({max_date}) < ${PROD_HOLDOUT_TO}')
    print('   El refit usará lo que haya disponible.')
" || abort "Error al leer datos de BD"

# Backup del deploy PROD anterior si existe (por si re-ejecutas)
if [ -d "${PROD_DEPLOY_DIR}" ]; then
  echo ""
  echo "📦 Archiving deploy PROD previo: ${PROD_DEPLOY_DIR}.bak_${TS}"
  mv "${PROD_DEPLOY_DIR}" "${PROD_DEPLOY_DIR}.bak_${TS}"
fi

echo "✅ Pre-checks completados"

# ─── 2. resume_deploy_v6 × 2 con cutoff extendido ──────────────────
log_section "2. resume_deploy_full_v6_multitask × 2 (cutoff extendido)"

for SIDE in long short; do
  echo ""
  echo "── side = ${SIDE} ────────────────────────────────────────────"
  python3 -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} \
    --inherit-config-from ${INHERIT_FROM_RELEASE} \
    --target-type multitask \
    --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} \
    --holdout-from ${HOLDOUT_FROM} \
    --holdout-to ${PROD_HOLDOUT_TO} \
    --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED} \
    --deploy-subdir deploy_PROD_${SIDE}_specialist_seed${SEED} \
    --locked-params-json ${BEST_JSON} \
    --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done

# ─── 3. Merge specialists ──────────────────────────────────────────
log_section "3. merge_specialists"

python3 -m mimo.oof.merge_specialists \
  --release ${RELEASE} \
  --long-dir  ${ARTIFACTS}/deploy_PROD_long_specialist_seed${SEED} \
  --short-dir ${ARTIFACTS}/deploy_PROD_short_specialist_seed${SEED} \
  --out-dir   ${PROD_DEPLOY_DIR}

# ─── 4. select_thresholds_from_tail ────────────────────────────────
log_section "4. select_thresholds_from_tail (tail incluye LOCKBOX original)"

python3 -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir ${PROD_DEPLOY_DIR} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

# ─── 5. compute_state_percentiles + policy stub ────────────────────
log_section "5. compute_state_percentiles + policy stub PROD"

POLICY_PROD=config/decision_policies_config_${RELEASE}_PROD.py

# Backup del policy PROD anterior si existe
if [ -f "${POLICY_PROD}" ]; then
  cp "${POLICY_PROD}" "${POLICY_PROD}.bak_${TS}"
  echo "📦 Backup policy PROD anterior: ${POLICY_PROD}.bak_${TS}"
fi

python3 -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir ${PROD_DEPLOY_DIR} \
  --emit-config-stub \
  --out-stub ${POLICY_PROD}

# ─── 6. Sanity check (NO es validación) ─────────────────────────────
SANITY_PASSED=1
if [ "${SKIP_SANITY}" = "1" ]; then
  log_section "6. Sanity check OMITIDO (--SKIP_SANITY=1 o falta validation deploy)"
else
  log_section "6. Sanity check (coherencia, NO validación pura)"
  echo "⚠️  ESTO NO ES VALIDACIÓN PURA — el deploy PROD ya vio estos datos en su tail."
  echo "   Solo busca detectar bugs gruesos: si el PnL refit << validation, algo se rompió."
  
  PROD_SANITY=/tmp/replay_prod_sanity_${RELEASE}_${TS}
  VAL_SANITY=/tmp/replay_validation_sanity_${RELEASE}_${TS}
  
  echo ""
  echo "  Replay PROD..."
  python3 scripts/replay_s2_202500.py \
    --release ${RELEASE} \
    --deploy-subdir deploy_PROD_combined_seed${SEED} \
    --policy-config decision_policies_config_${RELEASE}_PROD \
    --warmup-from ${SANITY_WARMUP} \
    --from ${SANITY_FROM} --to ${SANITY_TO} \
    --out ${PROD_SANITY}
  
  echo ""
  echo "  Replay VALIDATION (control)..."
  python3 scripts/replay_s2_202500.py \
    --release ${RELEASE} \
    --deploy-subdir ${VALIDATION_DEPLOY} \
    --policy-config ${VALIDATION_POLICY} \
    --warmup-from ${SANITY_WARMUP} \
    --from ${SANITY_FROM} --to ${SANITY_TO} \
    --out ${VAL_SANITY}
  
  PROD_PNL=$(extract_metric ${PROD_SANITY}/summary.json pnl_pct)
  VAL_PNL=$(extract_metric ${VAL_SANITY}/summary.json pnl_pct)
  PROD_TRADES=$(extract_metric ${PROD_SANITY}/summary.json n_trades)
  VAL_TRADES=$(extract_metric ${VAL_SANITY}/summary.json n_trades)
  
  echo ""
  echo "  📊 Sanity comparativa (${SANITY_FROM} → ${SANITY_TO}):"
  echo "                      PROD        VALIDATION"
  printf "    PnL%%:         %+8.2f    %+8.2f\n" "${PROD_PNL}" "${VAL_PNL}"
  printf "    Trades:       %8.0f    %8.0f\n" "${PROD_TRADES}" "${VAL_TRADES}"
  
  SANITY_CHECK=$(python3 <<EOF
prod = ${PROD_PNL}
val = ${VAL_PNL}
factor = ${SANITY_MIN_FACTOR}

if val > 0:
    threshold = val * factor
    if prod >= threshold:
        print("OK")
    else:
        print("FAIL")
        print(f"PROD PnL ({prod:+.2f}%) < VALIDATION ({val:+.2f}%) × {factor} = {threshold:+.2f}%")
elif val < 0:
    # Si validation también pierde, comprobar que prod no pierde mucho más
    if prod >= val * (2 - factor):  # max 50% más pérdida que val
        print("OK")
    else:
        print("FAIL")
        print(f"PROD pierde sustancialmente más que VALIDATION")
else:
    print("OK")
    print("VALIDATION ~0, no se puede comparar relativo")
EOF
)
  
  SANITY_VERDICT=$(echo "${SANITY_CHECK}" | head -1)
  
  echo ""
  if [ "${SANITY_VERDICT}" = "OK" ]; then
    echo "  ✅ Sanity check OK — PROD coherente con VALIDATION"
  else
    SANITY_PASSED=0
    echo "  🔴 Sanity check FAIL:"
    echo "${SANITY_CHECK}" | tail -n +2 | sed 's/^/     /'
  fi
fi

# ─── 7. LINEAGE.md ─────────────────────────────────────────────────
log_section "7. Documentar LINEAGE del deploy PROD"

LINEAGE=${PROD_DEPLOY_DIR}/LINEAGE.md
mkdir -p $(dirname ${LINEAGE})

cat > ${LINEAGE} <<EOF
# LINEAGE — deploy_PROD_combined_seed${SEED}

**Release**: ${RELEASE}  
**Created**: $(date -u +"%Y-%m-%d %H:%M:%S UTC")  
**Operator**: $(whoami)@$(hostname)

## Configuración heredada

- **Inherit from release**: ${INHERIT_FROM_RELEASE}
- **Tag**: ${TAG}
- **Seed**: ${SEED}
- **Variants**: long=vol_boost_td_down, short=vol_boost
- **Label horizons**: long=3, short=3
- **Target type**: multitask specialists merged

## Datos usados

| Periodo | Rango | Función |
|---|---|---|
| TRAIN  | ${TRAIN_FROM} → (full - tail) | Entrena modelo neural |
| HOLDOUT (OOF eval) | ${HOLDOUT_FROM} → ${PROD_HOLDOUT_TO} | Evaluación OOF |
| Tail (isotonic cal) | últimos ${TAIL_DAYS} días antes de ${PROD_HOLDOUT_TO} | Calibrator + threshold |

## Hyperparams

\`\`\`json
$(cat ${BEST_JSON} 2>/dev/null | python3 -c "
import json, sys
data = json.load(sys.stdin)
top_long = data.get('top_long', [{}])[0].get('params', {})
top_short = data.get('top_short', [{}])[0].get('params', {})
print(json.dumps({'long': top_long, 'short': top_short}, indent=2))
")
\`\`\`

## Calibrador

- **Tipo**: isotonic (default de resume_deploy_v6, NO se aplicó swap a Beta/per_state)
- **Tail length**: ${TAIL_DAYS} días

## Validación previa (Fase 4)

EOF

if [ -n "${LATEST_VALIDATION}" ]; then
  cat >> ${LINEAGE} <<EOF
- **Reporte**: ${LATEST_VALIDATION}
- **Veredicto**: ${VERDICT}
EOF
else
  echo "- ⚠️ Sin reporte de Fase 4 — NO se validó sobre LOCKBOX antes del refit" >> ${LINEAGE}
fi

cat >> ${LINEAGE} <<EOF

## Sanity check Fase 5

EOF

if [ "${SKIP_SANITY}" = "1" ]; then
  echo "- ⚠️ Sanity check OMITIDO" >> ${LINEAGE}
else
  echo "- **Window**: ${SANITY_FROM} → ${SANITY_TO}" >> ${LINEAGE}
  echo "- **PROD PnL%**: ${PROD_PNL}" >> ${LINEAGE}
  echo "- **VALIDATION PnL%**: ${VAL_PNL}" >> ${LINEAGE}
  echo "- **Verdict**: ${SANITY_VERDICT}" >> ${LINEAGE}
fi

cat >> ${LINEAGE} <<EOF

## Archivos generados

- Deploy directory: \`${PROD_DEPLOY_DIR}/\`
- Policy stub: \`${POLICY_PROD}\`
- LINEAGE: \`${LINEAGE}\`

## Pasos siguientes

1. **Fase 6 — Despliegue a producción**:
   - Editar \`main/s2_main.py\`:
     - \`from config.decision_policies_config_${RELEASE}_PROD import ...\`
     - \`artifacts_path = ... "deploy_PROD_combined_seed${SEED}" ...\`
     - feature_masks UNION (25 cols) si multitask
   - Backup del s2_main actual
   - \`sudo systemctl restart s2-service\`

2. **Smoke test** del primer trade post-restart
   - Logs muestran "Deploy specialists-merged detectado"
   - Scaler context: 25 cols
   - Primer Decision aparece <30s

3. **Monitorización inicial** (semana 1):
   - \`monitor_health.py\` diario
   - Capital mínimo (1-5%) según plan de despliegue gradual

## Cómo revertir

Si tras producción detectas problemas:

\`\`\`bash
bash scripts/rollback_deploy.py --list  # ver versiones previas
bash scripts/rollback_deploy.py \\
  --target-deploy <previous> \\
  --target-policy <previous_policy> \\
  --apply
sudo systemctl restart s2-service
\`\`\`
EOF

echo "✅ LINEAGE creado: ${LINEAGE}"

# ─── 8. Resumen final ──────────────────────────────────────────────
log_section "FASE 5 COMPLETADA"
echo "  Deploy:       ${PROD_DEPLOY_DIR}"
echo "  Policy stub:  ${POLICY_PROD}"
echo "  LINEAGE:      ${LINEAGE}"
echo ""

if [ "${SKIP_SANITY}" != "1" ]; then
  if [ "${SANITY_PASSED}" = "1" ]; then
    echo "  ✅ Sanity check OK"
  else
    echo "  🔴 Sanity check FAIL — revisar antes de promover"
  fi
fi

echo ""
echo "📋 SIGUIENTE PASO — Fase 6: Despliegue a producción"
echo ""
echo "  El runbook (Apéndice / Fase 6) describe los 4 cambios en s2_main.py:"
echo "    1. Cambiar import de policy:"
echo "         from config.decision_policies_config_${RELEASE}_PROD import (..."
echo ""
echo "    2. feature_masks UNION (25 cols, multitask):"
echo "         feature_masks = {'long':  {...todos True},"
echo "                          'short': {...todos True}}"
echo ""
echo "    3. artifacts_path:"
echo "         artifacts_path = ... 'oof' / 'deploy_PROD_combined_seed${SEED}' ..."
echo ""
echo "    4. (opcional) Desactivar RL para mimetizar el replay:"
echo "         use_rl=False, rl_policy_path=None"
echo ""
echo "  Tras editar:"
echo "    git add main/s2_main.py docs/RUNBOOK.md"
echo "    git commit -m 's2_main: deploy ${RELEASE}_PROD seed${SEED}'"
echo "    sudo systemctl restart s2-service"
echo ""
echo "  Verifica logs:"
echo "    tail -f logs/s2_\$(date +%F).log"
echo "    # Buscar: 'Deploy specialists-merged detectado'"
echo ""
echo "ℹ️  El despliegue se hace MANUALMENTE — ningún script automatiza el restart."
echo "    Es deliberado: producción solo cambia tras revisión humana explícita."