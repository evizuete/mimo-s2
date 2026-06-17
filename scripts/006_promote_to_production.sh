#!/usr/bin/env bash
# 006_promote_to_production.sh — Edita s2_main.py para promover un deploy.
# Auto-detecta tipo (multitask/binary, con/sin swap), backup, sintaxis check.
# NO reinicia servicio (manual), NO hace push (te muestra los comandos).
set -euo pipefail

export RELEASE=${RELEASE:-202600}
export SEED=${SEED:-47}
export NEW_DEPLOY=${NEW_DEPLOY:-deploy_PROD_combined_seed${SEED}}
export NEW_POLICY=${NEW_POLICY:-decision_policies_config_${RELEASE}_PROD}
export DISABLE_RL=${DISABLE_RL:-1}
export AUTO_COMMIT=${AUTO_COMMIT:-1}
export DRY_RUN=${DRY_RUN:-0}

REPO=$(pwd)
S2=${REPO}/main/s2_main.py
ARTIFACTS=artifacts/${RELEASE}/oof
DEPLOY_DIR=${ARTIFACTS}/${NEW_DEPLOY}
POLICY_PATH=config/${NEW_POLICY}.py
TS=$(date +%Y%m%d_%H%M%S)
SNAPSHOT=docs/promotion_snapshots/promote_${RELEASE}_${TS}

log() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log "FASE 6 — Promote to production"
echo "  Release: ${RELEASE}"
echo "  Deploy:  ${NEW_DEPLOY}"
echo "  Policy:  ${NEW_POLICY}"
echo "  Disable RL: ${DISABLE_RL}"
echo "  Dry-run: ${DRY_RUN}"

log "1. Pre-checks"
[ -f "${S2}" ] || abort "No existe ${S2}"
[ -d "${DEPLOY_DIR}" ] || abort "No existe ${DEPLOY_DIR}"
[ -f "${POLICY_PATH}" ] || abort "No existe ${POLICY_PATH}"
echo "✅ Pre-checks OK"

log "2. Detectar tipo de deploy"
TYPE="binary"
[ -f "${DEPLOY_DIR}/model_${RELEASE}_multitask.keras" ] && TYPE="multitask"
[ -f "${DEPLOY_DIR}/merge_specialists_meta.json" ] && TYPE="specialists_merged"
echo "  Tipo: ${TYPE}"

log "3. Detectar calibrator swap"
HAS_SWAP=0
[ -f "${DEPLOY_DIR}/calibrator_swap_meta.json" ] && HAS_SWAP=1
[ "${HAS_SWAP}" = "1" ] && echo "  🔄 Deploy con swap" || echo "  ℹ️  Deploy sin swap"

# Inconsistencia detection
VAL_DIR=${ARTIFACTS}/deploy_validation_combined_seed${SEED}
if [ -d "${VAL_DIR}" ] && [ "${NEW_DEPLOY}" != "deploy_validation_combined_seed${SEED}" ]; then
  if [ -f "${VAL_DIR}/calibrator_swap_meta.json" ] && [ "${HAS_SWAP}" = "0" ]; then
    echo ""
    echo "⚠️  INCONSISTENCIA: validation tiene swap pero PROD no."
    echo "    Para propagar swap a PROD:"
    echo "      DEPLOY=${DEPLOY_DIR} bash scripts/003C_calibrator_swap_experiment.sh"
    read -p "  ¿Continuar sin propagar? [y/N]: " C
    [ "${C}" = "y" ] || abort "Aborted"
  fi
fi

if [ "${DRY_RUN}" != "1" ]; then
  mkdir -p ${SNAPSHOT}
  cp ${S2} ${SNAPSHOT}/s2_main.py.before
  CURRENT_POLICY=$(grep -oE "from config\.decision_policies_config[a-zA-Z0-9_]*" ${S2} | head -1 | sed 's|from config\.||')
  CURRENT_DEPLOY=$(grep -oE '"oof" / "[^"]+"' ${S2} | head -1 | sed 's|"oof" / "||;s|"||')
  echo "  📦 Snapshot: ${SNAPSHOT}/"
  echo "  Estado actual:"
  echo "    Policy:  ${CURRENT_POLICY}"
  echo "    Deploy:  ${CURRENT_DEPLOY}"
fi

log "4. Generar patch"

if [ "${TYPE}" = "specialists_merged" ] || [ "${TYPE}" = "multitask" ]; then
  FM_TYPE="UNION (multitask, 25 cols)"
  FM_PY='feature_masks={
            "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                      "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
            "short": {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                      "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
        },'
else
  FM_TYPE="ASIMÉTRICAS (binary, 24 cols)"
  FM_PY='feature_masks={
            "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                      "ema_bear": False, "rsi_overbought": False, "macd_negative": False},
            "short": {"ema_bull": False, "rsi_oversold": False, "macd_positive": False,
                      "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
        },'
fi

echo "  Cambios planificados:"
echo "    1. Import policy → from config.${NEW_POLICY} import ..."
echo "    2. artifacts_path → .../${NEW_DEPLOY}/"
echo "    3. feature_masks → ${FM_TYPE}"
[ "${DISABLE_RL}" = "1" ] && echo "    4. RL → desactivado"

PATCHED=${SNAPSHOT}/s2_main.py.after
[ "${DRY_RUN}" = "1" ] && PATCHED=/tmp/s2_main_proposed_${TS}.py
cp ${S2} ${PATCHED}

python3 <<EOF
import re
c = open("${PATCHED}").read()
c, n1 = re.subn(r'from\s+config\.decision_policies_config[a-zA-Z0-9_]*\s+import\s+([^\n]+)',
                r'from config.${NEW_POLICY} import \1', c, count=1)
c, n2 = re.subn(r'("artifacts"\s*/\s*release\s*/\s*"oof"\s*/\s*)"[^"]+"(\s*\)\.resolve\(\)\))',
                r'\g<1>"${NEW_DEPLOY}"\g<2>', c, count=1)
new_fm = '''${FM_PY}'''
c, n3 = re.subn(
    r'feature_masks\s*=\s*\{[^}]*"long"[^}]*\{[^}]*\}[^}]*,[^}]*"short"[^}]*\{[^}]*\}[^}]*,?\s*\}\s*,',
    new_fm, c, count=1, flags=re.DOTALL)
if "${DISABLE_RL}" == "1":
    c = re.sub(r'use_rl\s*=\s*True', 'use_rl=False', c)
    c = re.sub(r'rl_policy_path\s*=\s*policy_path', 'rl_policy_path=None', c)
open("${PATCHED}", "w").write(c)
print(f"  Reemplazos: import={n1}  artifacts_path={n2}  feature_masks={n3}")
EOF

log "5. Diff"
diff -u ${S2} ${PATCHED} | head -100 || true

if [ "${DRY_RUN}" = "1" ]; then
  log "DRY-RUN — no aplicado"
  echo "  Patch en: ${PATCHED}"
  exit 0
fi

log "6. Confirmación"
echo "  ⚠️  ESTÁS A PUNTO DE MODIFICAR ${S2}"
read -p "  ¿APLICAR? [y/N]: " CONFIRM
if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
  echo "❌ Aborted. Snapshot en ${SNAPSHOT}/"
  exit 0
fi

log "7. Aplicar"
cp ${PATCHED} ${S2}
python3 -m py_compile ${S2} || {
  echo "❌ Sintaxis inválida. Restaurando..."
  cp ${SNAPSHOT}/s2_main.py.before ${S2}
  abort "Sintaxis inválida"
}
echo "✅ s2_main.py modificado y sintaxis OK"

if [ "${AUTO_COMMIT}" = "1" ]; then
  log "8. git commit"
  git add main/s2_main.py
  COMMIT_MSG="s2_main: promote ${RELEASE} ${NEW_DEPLOY} (seed${SEED})"
  echo "  Commit: ${COMMIT_MSG}"
  read -p "  ¿Commit? [y/N]: " CC
  if [ "${CC}" = "y" ] || [ "${CC}" = "Y" ]; then
    git commit -m "${COMMIT_MSG}"
    echo "✅ Commit creado (push manual)"
  fi
fi

log "9. Restart guidance"
if systemctl is-active s2-service > /dev/null 2>&1; then
  echo "  sudo systemctl restart s2-service"
else
  PID=$(pgrep -f s2_main.py 2>/dev/null || echo "")
  if [ -n "${PID}" ]; then
    echo "  kill -TERM ${PID} && sleep 5"
    echo "  nohup python3 main/s2_main.py > logs/s2_\$(date +%F).log 2>&1 &"
  else
    echo "  nohup python3 main/s2_main.py > logs/s2_\$(date +%F).log 2>&1 &"
  fi
fi
echo "  ⚠️  No reinicio automáticamente"

log "10. Smoke test"
echo "  tail -f logs/s2_\$(date +%F).log"
echo "  Buscar: 'Scaler context: $([ "${TYPE}" = "binary" ] && echo "24" || echo "25") cols'"
echo "  Buscar: 'Deploy ${TYPE}'"

log "Rollback (si algo va mal)"
echo "  cp ${SNAPSHOT}/s2_main.py.before ${S2}"

log "FASE 6 COMPLETADA"
echo "  s2_main.py: ✅ modificado"
echo "  Snapshot:   ${SNAPSHOT}/"
echo "  Servicio:   ❌ MANUAL"
echo ""
echo "📋 PLAN DE DESPLIEGUE GRADUAL:"
echo "    Semana 1-2: capital 1%"
echo "    Semana 3-4: capital 5%"
echo "    Semana 5-8: capital 25%"
echo "    Semana 9-12: capital 50%"
echo "    Semana 13+: capital 100%"
