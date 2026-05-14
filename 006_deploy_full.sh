#!/usr/bin/env bash
#
# 006_promote_to_production.sh
# ════════════════════════════════════════════════════════════════════
# FASE 6 — Promueve un deploy a producción editando main/s2_main.py.
#
# Acepta cualquier deploy + policy como input. Por defecto promueve el
# deploy_PROD generado por Fase 5, pero puedes promover deploy_validation
# (si decidiste saltar Fase 5) o cualquier otro deploy histórico.
#
# QUÉ HACE:
#   1. Pre-checks (deploy + policy + s2_main.py existen)
#   2. Detecta tipo de deploy (multitask vs binary, con/sin swap)
#   3. Snapshot del estado actual de s2_main.py + policy actual
#   4. Patch automático de s2_main.py:
#      - import policy nueva
#      - artifacts_path → nuevo deploy
#      - feature_masks (UNION si multitask, asimétricas si binary)
#      - (opcional) desactivar RL
#   5. Muestra diff visual
#   6. Confirma con humano antes de aplicar
#   7. (Opcional) Commit en git
#   8. Instrucciones de restart + smoke test
#
# CASOS SOPORTADOS:
#   - Promover deploy_PROD tras Fase 5 (CAMINO A — recomendado)
#   - Promover deploy_PROD tras Fase 5 + 003C en PROD (CAMINO B)
#   - Promover deploy_validation saltando Fase 5 (CAMINO C — pierde LOCKBOX en train)
#   - Promover deploy histórico para rollback rápido
#
# DETECTA Y AVISA:
#   - Si el deploy tiene calibrator_swap_meta.json (003C aplicado)
#   - Si el deploy es multitask (merge_specialists_meta.json presente)
#   - Mismatch entre validation_swap y prod_no_swap (te recuerda Camino B)
#
# LO QUE NO HACE:
#   - NO reinicia el servicio automáticamente (deja al humano decidir)
#   - NO hace git push (te muestra el commit pero el push es manual)
#   - NO hace smoke test post-restart automático (te da el comando)
#
# USO:
#   # Promover deploy_PROD (default):
#   bash scripts/006_promote_to_production.sh
#
#   # Promover otro deploy:
#   NEW_DEPLOY=deploy_validation_combined_seed47 \
#   NEW_POLICY=decision_policies_config_202600_validation \
#     bash scripts/006_promote_to_production.sh
#
#   # Dry-run (muestra patch sin aplicar):
#   DRY_RUN=1 bash scripts/006_promote_to_production.sh
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202600}
export SEED=${SEED:-47}
export NEW_DEPLOY=${NEW_DEPLOY:-deploy_PROD_combined_seed${SEED}}
export NEW_POLICY=${NEW_POLICY:-decision_policies_config_${RELEASE}_PROD}

# Disable RL en s2_main (recomendado para mimetizar el replay)
export DISABLE_RL=${DISABLE_RL:-1}

# Git commit auto
export AUTO_COMMIT=${AUTO_COMMIT:-1}

# Dry-run: solo muestra cambios
export DRY_RUN=${DRY_RUN:-0}

REPO_ROOT=$(pwd)
S2_MAIN=${REPO_ROOT}/main/s2_main.py
ARTIFACTS=artifacts/${RELEASE}/oof
DEPLOY_DIR=${ARTIFACTS}/${NEW_DEPLOY}
POLICY_PATH=config/${NEW_POLICY}.py
TS=$(date +%Y%m%d_%H%M%S)
SNAPSHOT_DIR=docs/promotion_snapshots/promote_${RELEASE}_${TS}

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

# ─── Banner ─────────────────────────────────────────────────────────
log_section "FASE 6 — Promote to production"
echo "  Release:       ${RELEASE}"
echo "  Deploy:        ${NEW_DEPLOY}"
echo "  Policy:        ${NEW_POLICY}"
echo "  Disable RL:    ${DISABLE_RL}"
echo "  Auto-commit:   ${AUTO_COMMIT}"
echo "  Dry-run:       ${DRY_RUN}"
echo "  Timestamp:     ${TS}"

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

[ -f "${S2_MAIN}" ] || abort "No existe ${S2_MAIN}"
[ -d "${DEPLOY_DIR}" ] || abort "No existe deploy: ${DEPLOY_DIR}"
[ -f "${POLICY_PATH}" ] || abort "No existe policy: ${POLICY_PATH}"

# Verificar que el deploy tiene artefactos mínimos
[ -f "${DEPLOY_DIR}/scalers_${RELEASE}/context_scaler.pkl" ] \
  || [ -f "${DEPLOY_DIR}/scalers_${RELEASE}" ] \
  || abort "No encuentro scalers en ${DEPLOY_DIR}"

# Modelo: multitask (un solo .keras) o binary (long.keras + short.keras)
HAS_MULTITASK_MODEL=0
HAS_BINARY_MODEL=0
[ -f "${DEPLOY_DIR}/model_${RELEASE}_multitask.keras" ] && HAS_MULTITASK_MODEL=1
[ -f "${DEPLOY_DIR}/model_${RELEASE}_long.keras" ] && [ -f "${DEPLOY_DIR}/model_${RELEASE}_short.keras" ] && HAS_BINARY_MODEL=1

if [ "${HAS_MULTITASK_MODEL}" = "0" ] && [ "${HAS_BINARY_MODEL}" = "0" ]; then
  abort "No encuentro modelo en ${DEPLOY_DIR}"
fi

echo "✅ Deploy y policy verificados"

# ─── 2. Detectar tipo de deploy ─────────────────────────────────────
log_section "2. Detectar tipo de deploy"

IS_SPECIALISTS_MERGED=0
IS_PURE_MULTITASK=0
IS_BINARY=0

if [ -f "${DEPLOY_DIR}/merge_specialists_meta.json" ]; then
  IS_SPECIALISTS_MERGED=1
  echo "✅ Tipo: SPECIALISTS MERGED (modelos *_long.keras y *_short.keras son multitask renombrados)"
  echo "   → feature_masks UNION (25 cols), ambos sides ven todas las features"
elif [ "${HAS_MULTITASK_MODEL}" = "1" ]; then
  IS_PURE_MULTITASK=1
  echo "✅ Tipo: MULTITASK puro (model_${RELEASE}_multitask.keras presente)"
  echo "   → feature_masks UNION (25 cols)"
else
  IS_BINARY=1
  echo "✅ Tipo: BINARY single-side (modelos independientes por side)"
  echo "   → feature_masks asimétricas (24 cols)"
fi

# ─── 3. Detectar si el deploy tiene calibrator swap ────────────────
log_section "3. Detectar calibrator swap"

HAS_SWAP=0
if [ -f "${DEPLOY_DIR}/calibrator_swap_meta.json" ]; then
  HAS_SWAP=1
  SWAP_METHOD_LONG=$(python3 -c "
import json
m = json.load(open('${DEPLOY_DIR}/calibrator_swap_meta.json'))
sides = m.get('sides', {})
if 'long' in sides:
    s = sides['long']
    if s.get('level') == 1:
        print(s.get('method', '?'))
    else:
        print('Nivel 2 (per-state)')
else:
    print('?')
")
  SWAP_METHOD_SHORT=$(python3 -c "
import json
m = json.load(open('${DEPLOY_DIR}/calibrator_swap_meta.json'))
sides = m.get('sides', {})
if 'short' in sides:
    s = sides['short']
    if s.get('level') == 1:
        print(s.get('method', '?'))
    else:
        print('Nivel 2 (per-state)')
else:
    print('?')
")
  echo "🔄 Deploy tiene calibrator swap aplicado:"
  echo "   LONG:  ${SWAP_METHOD_LONG}"
  echo "   SHORT: ${SWAP_METHOD_SHORT}"
  echo "   ⚠️  Asegúrate de que validaste este swap en Fase 4 (LOCKBOX)."
else
  echo "ℹ️  Deploy SIN swap (calibrador estándar isotonic 21d/45d)"
fi

# ─── 3.5. Detectar inconsistencia validation vs PROD ────────────────
VALIDATION_DEPLOY_DIR=${ARTIFACTS}/deploy_validation_combined_seed${SEED}
if [ -d "${VALIDATION_DEPLOY_DIR}" ] && [ "${NEW_DEPLOY}" != "deploy_validation_combined_seed${SEED}" ]; then
  if [ -f "${VALIDATION_DEPLOY_DIR}/calibrator_swap_meta.json" ] && [ "${HAS_SWAP}" = "0" ]; then
    echo ""
    echo "⚠️  POSIBLE INCONSISTENCIA DETECTADA:"
    echo "   - deploy_validation TIENE swap aplicado"
    echo "   - deploy ${NEW_DEPLOY} NO tiene swap"
    echo "   - Si tu intención era promover el swap a PROD, tienes que ejecutar 003C"
    echo "     sobre deploy_PROD ANTES de promover:"
    echo "         DEPLOY=${DEPLOY_DIR} bash scripts/003C_calibrator_swap_experiment.sh"
    echo ""
    read -p "  ¿Continuar promoción sin propagar swap? [y/N]: " CONFIRM
    [ "${CONFIRM}" = "y" ] || [ "${CONFIRM}" = "Y" ] || abort "Aborted por inconsistencia."
  fi
fi

# ─── 4. Snapshot estado actual ──────────────────────────────────────
log_section "4. Snapshot del estado actual"

if [ "${DRY_RUN}" = "1" ]; then
  echo "🔍 DRY-RUN: no se crea snapshot ni se modifica nada"
else
  mkdir -p ${SNAPSHOT_DIR}
  cp ${S2_MAIN} ${SNAPSHOT_DIR}/s2_main.py.before
  
  # Detectar policy actual
  CURRENT_POLICY=$(grep -oE "from config\.decision_policies_config[a-zA-Z0-9_]*" ${S2_MAIN} | head -1 | sed 's|from config\.||')
  CURRENT_DEPLOY=$(grep -oE '"oof" / "[^"]+"' ${S2_MAIN} | head -1 | sed 's|"oof" / "||;s|"||')
  
  cat > ${SNAPSHOT_DIR}/snapshot_meta.json <<EOF
{
  "timestamp": "${TS}",
  "release": "${RELEASE}",
  "previous_policy": "${CURRENT_POLICY}",
  "previous_deploy": "${CURRENT_DEPLOY}",
  "new_policy": "${NEW_POLICY}",
  "new_deploy": "${NEW_DEPLOY}",
  "deploy_type": "$([ "${IS_SPECIALISTS_MERGED}" = "1" ] && echo "specialists_merged" || ([ "${IS_PURE_MULTITASK}" = "1" ] && echo "multitask" || echo "binary"))",
  "has_swap": ${HAS_SWAP},
  "disable_rl": ${DISABLE_RL}
}
EOF
  
  echo "📦 Snapshot guardado: ${SNAPSHOT_DIR}/"
  echo "   Estado actual:"
  echo "     Policy actual:  ${CURRENT_POLICY}"
  echo "     Deploy actual:  ${CURRENT_DEPLOY}"
fi

# ─── 5. Generar el patch ───────────────────────────────────────────
log_section "5. Generar patch para s2_main.py"

# Determinar feature_masks según tipo de deploy
if [ "${IS_SPECIALISTS_MERGED}" = "1" ] || [ "${IS_PURE_MULTITASK}" = "1" ]; then
  FEATURE_MASKS_TYPE="UNION (multitask)"
  FEATURE_MASKS_PY='feature_masks={
            "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                      "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
            "short": {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                      "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
        },'
else
  FEATURE_MASKS_TYPE="ASIMÉTRICAS (binary)"
  FEATURE_MASKS_PY='feature_masks={
            "long":  {"ema_bull": True,  "rsi_oversold": True,  "macd_positive": True,
                      "ema_bear": False, "rsi_overbought": False, "macd_negative": False},
            "short": {"ema_bull": False, "rsi_oversold": False,  "macd_positive": False,
                      "ema_bear": True,  "rsi_overbought": True, "macd_negative": True},
        },'
fi

echo "  Cambios planificados:"
echo "    1. Import policy → from config.${NEW_POLICY} import ..."
echo "    2. artifacts_path → .../${NEW_DEPLOY}/"
echo "    3. feature_masks → ${FEATURE_MASKS_TYPE}"
if [ "${DISABLE_RL}" = "1" ]; then
  echo "    4. RL → desactivado (use_rl=False)"
fi

# Generar archivo patched
PATCHED=${SNAPSHOT_DIR}/s2_main.py.after
if [ "${DRY_RUN}" != "1" ]; then
  cp ${S2_MAIN} ${PATCHED}
else
  # En dry-run, generar en /tmp
  PATCHED=/tmp/s2_main_proposed_${TS}.py
  cp ${S2_MAIN} ${PATCHED}
fi

# 1) Reemplazar import de policy
python3 <<EOF
import re
content = open("${PATCHED}").read()
pattern = r'from\s+config\.decision_policies_config[a-zA-Z0-9_]*\s+import\s+([^\n]+)'
replacement = r'from config.${NEW_POLICY} import \1'
new_content, n = re.subn(pattern, replacement, content, count=1)
if n == 0:
    raise SystemExit("❌ No encuentro patrón de import de policy")
open("${PATCHED}", "w").write(new_content)
print(f"   ✓ import policy reemplazado")
EOF

# 2) Reemplazar artifacts_path
python3 <<EOF
import re
content = open("${PATCHED}").read()
pattern = r'(artifacts_path\s*=\s*str\(\(base_dir\s*/\s*"\.\."\s*/\s*"artifacts"\s*/\s*release\s*/\s*"oof"\s*/\s*)"[^"]+"(\s*\)\.resolve\(\)\))'
replacement = r'\g<1>"${NEW_DEPLOY}"\g<2>'
new_content, n = re.subn(pattern, replacement, content, count=1)
if n == 0:
    raise SystemExit("❌ No encuentro patrón de artifacts_path")
open("${PATCHED}", "w").write(new_content)
print(f"   ✓ artifacts_path reemplazado")
EOF

# 3) Reemplazar feature_masks (más complejo — bloque multilínea)
python3 <<EOF
import re
content = open("${PATCHED}").read()
# Match el bloque feature_masks completo (asimétrico o union, multilínea)
pattern = r'feature_masks\s*=\s*\{[^}]*"long"[^}]*\{[^}]*\}[^}]*,[^}]*"short"[^}]*\{[^}]*\}[^}]*,?\s*\}\s*,'
new_block = '''${FEATURE_MASKS_PY}'''
new_content, n = re.subn(pattern, new_block, content, count=1, flags=re.DOTALL)
if n == 0:
    print("   ⚠️  No pude reemplazar feature_masks automáticamente.")
    print("   ⚠️  Edita manualmente s2_main.py para usar máscaras ${FEATURE_MASKS_TYPE}.")
else:
    open("${PATCHED}", "w").write(new_content)
    print(f"   ✓ feature_masks reemplazado")
EOF

# 4) (Opcional) Desactivar RL
if [ "${DISABLE_RL}" = "1" ]; then
python3 <<EOF
import re
content = open("${PATCHED}").read()
content = re.sub(r'use_rl\s*=\s*True', 'use_rl=False', content)
content = re.sub(r'rl_policy_path\s*=\s*policy_path', 'rl_policy_path=None', content)
open("${PATCHED}", "w").write(content)
print(f"   ✓ RL desactivado")
EOF
fi

# ─── 6. Mostrar diff ────────────────────────────────────────────────
log_section "6. Diff vs s2_main.py actual"

diff -u ${S2_MAIN} ${PATCHED} || true

# ─── 7. Confirmación humana ─────────────────────────────────────────
if [ "${DRY_RUN}" = "1" ]; then
  log_section "DRY-RUN — Patch propuesto (no aplicado)"
  echo "  Patch generado en: ${PATCHED}"
  echo "  Para aplicar: re-ejecuta sin DRY_RUN=1"
  exit 0
fi

log_section "7. Confirmación"
echo ""
echo "  ⚠️  ESTÁS A PUNTO DE MODIFICAR ${S2_MAIN}"
echo ""
echo "  Cambios:"
echo "    - Policy:        ${CURRENT_POLICY} → ${NEW_POLICY}"
echo "    - Deploy:        ${CURRENT_DEPLOY} → ${NEW_DEPLOY}"
echo "    - feature_masks: ${FEATURE_MASKS_TYPE}"
[ "${DISABLE_RL}" = "1" ] && echo "    - RL:            desactivado"
echo ""
echo "  Si algo va mal, rollback automático:"
echo "    cp ${SNAPSHOT_DIR}/s2_main.py.before ${S2_MAIN}"
echo ""

read -p "  ¿APLICAR los cambios a producción? [y/N]: " CONFIRM
if [ "${CONFIRM}" != "y" ] && [ "${CONFIRM}" != "Y" ]; then
  echo "❌ Aborted by user. Snapshot guardado en ${SNAPSHOT_DIR}/ por si lo quieres consultar."
  exit 0
fi

# ─── 8. Aplicar patch ───────────────────────────────────────────────
log_section "8. Aplicar patch"

cp ${PATCHED} ${S2_MAIN}
echo "✅ ${S2_MAIN} modificado"

# Validar sintaxis Python del archivo modificado
python3 -m py_compile ${S2_MAIN} || {
  echo "❌ Error de sintaxis en s2_main.py modificado. Restaurando..."
  cp ${SNAPSHOT_DIR}/s2_main.py.before ${S2_MAIN}
  abort "Sintaxis inválida. Estado restaurado."
}
echo "✅ Sintaxis Python OK"

# ─── 9. (Opcional) git commit ───────────────────────────────────────
if [ "${AUTO_COMMIT}" = "1" ]; then
  log_section "9. git commit"
  cd ${REPO_ROOT}
  
  git add main/s2_main.py
  
  COMMIT_MSG="s2_main: promote ${RELEASE} ${NEW_DEPLOY} (seed${SEED})

Deploy:        ${NEW_DEPLOY}
Policy:        ${NEW_POLICY}
Feature masks: ${FEATURE_MASKS_TYPE}
RL:            $([ "${DISABLE_RL}" = "1" ] && echo "disabled" || echo "enabled")
Has swap:      ${HAS_SWAP}
Snapshot:      ${SNAPSHOT_DIR}/"
  
  echo "  Commit message:"
  echo "${COMMIT_MSG}" | sed 's/^/    /'
  echo ""
  read -p "  ¿Hacer commit? [y/N]: " COMMIT_CONFIRM
  if [ "${COMMIT_CONFIRM}" = "y" ] || [ "${COMMIT_CONFIRM}" = "Y" ]; then
    git commit -m "${COMMIT_MSG}"
    echo "✅ Commit creado (no se ha hecho push — manual)"
  else
    echo "ℹ️  Commit saltado. Para hacerlo después:"
    echo "       git add main/s2_main.py && git commit -m \"${COMMIT_MSG}\""
  fi
fi

# ─── 10. Restart guidance ───────────────────────────────────────────
log_section "10. Restart del servicio S2"

# Detectar systemd
HAS_SYSTEMD=0
if systemctl is-active s2-service > /dev/null 2>&1 || systemctl list-units --all | grep -q s2-service; then
  HAS_SYSTEMD=1
fi

# Detectar proceso manual
PID=$(pgrep -f s2_main.py 2>/dev/null || echo "")

echo ""
if [ "${HAS_SYSTEMD}" = "1" ]; then
  echo "  Sistema usa systemd:"
  echo "    sudo systemctl restart s2-service"
  echo "    sudo systemctl status s2-service"
elif [ -n "${PID}" ]; then
  echo "  Sistema usa proceso manual (PID ${PID}):"
  echo "    kill -TERM ${PID}"
  echo "    sleep 5  # esperar cierre limpio"
  echo "    nohup python3 main/s2_main.py > logs/s2_\$(date +%F).log 2>&1 &"
else
  echo "  No detecto servicio activo. Asumiendo arranque desde cero:"
  echo "    nohup python3 main/s2_main.py > logs/s2_\$(date +%F).log 2>&1 &"
fi
echo ""
echo "  ⚠️  No reinicio automáticamente — eso es decisión humana explícita."

# ─── 11. Smoke test guide ──────────────────────────────────────────
log_section "11. Smoke test post-restart"

echo ""
echo "  Tras reiniciar, verifica los logs:"
echo "    tail -f logs/s2_\$(date +%F).log"
echo ""
echo "  BUSCA estas líneas (en orden):"
echo "    1. '🔧 Construyendo TradingSimulator...'"
if [ "${IS_SPECIALISTS_MERGED}" = "1" ]; then
  echo "    2. '🧠 Deploy specialists-merged detectado'"
elif [ "${IS_PURE_MULTITASK}" = "1" ]; then
  echo "    2. '🧠 Deploy multitask detectado'"
else
  echo "    2. '🧠 Deploy binary'"
fi
echo "    3. 'Scaler context: $([ "${IS_BINARY}" = "1" ] && echo "24" || echo "25") cols total'"
echo "    4. 'Scalers loaded from: .../${NEW_DEPLOY}/scalers_${RELEASE}'"
echo "    5. Primera línea '[DECIDE_LIVE]' en los primeros 30s"
echo ""
echo "  NO debe aparecer:"
echo "    - 'ValueError: Input X is incompatible' (mismatch de feature_masks)"
echo "    - Tracebacks"
echo "    - 'ModuleNotFoundError'"

# ─── 12. Rollback instructions ──────────────────────────────────────
log_section "12. Rollback (si algo va mal)"

echo ""
echo "  Si tras restart hay problemas, rollback inmediato:"
echo ""
echo "    cp ${SNAPSHOT_DIR}/s2_main.py.before ${S2_MAIN}"
echo "    # Si hiciste commit antes:"
echo "    git revert HEAD --no-edit"
if [ "${HAS_SYSTEMD}" = "1" ]; then
  echo "    sudo systemctl restart s2-service"
else
  echo "    pkill -f s2_main.py"
  echo "    nohup python3 main/s2_main.py > logs/s2_\$(date +%F)_rollback.log 2>&1 &"
fi
echo ""
echo "  O usando el script de rollback:"
echo "    bash scripts/rollback_deploy.py --list  # ver deploys anteriores"
echo "    bash scripts/rollback_deploy.py \\"
echo "      --target-deploy ${CURRENT_DEPLOY} \\"
echo "      --target-policy ${CURRENT_POLICY} \\"
echo "      --apply"

# ─── 13. Resumen final ─────────────────────────────────────────────
log_section "FASE 6 COMPLETADA"
echo "  s2_main.py modificado:    ✅"
echo "  Snapshot:                 ${SNAPSHOT_DIR}/"
echo "  Commit hecho:             $([ "${AUTO_COMMIT}" = "1" ] && echo "según confirmación" || echo "no")"
echo "  Servicio reiniciado:      ❌ MANUAL (ver arriba)"
echo ""
echo "📋 PLAN DE DESPLIEGUE GRADUAL (recomendado):"
echo ""
echo "  Semana 1-2: capital 1%   → monitor_health diario"
echo "  Semana 3-4: capital 5%   → si métricas OK"
echo "  Semana 5-8: capital 25%  → si 2 meses sanos"
echo "  Semana 9-12: capital 50% → consistencia confirmada"
echo "  Semana 13+: capital 100% → producción plena"
echo ""
echo "📋 MONITORIZACIÓN INMEDIATA:"
echo "    bash scripts/monitor_health.py \\"
echo "      --release ${RELEASE} \\"
echo "      --deploy-subdir ${NEW_DEPLOY} \\"
echo "      --policy-config ${NEW_POLICY}"
echo ""
echo "📋 DASHBOARD SEMANAL:"
echo "    bash scripts/dashboard_health.py --out reports/dashboard.html"
echo "    explorer.exe reports/dashboard.html  # WSL"
echo ""
echo "🚨 ALERTAS QUE DEBEN DISPARAR ACCIÓN:"
echo "    - Daily DD > 5%       → halt manual + análisis"
echo "    - 3 días seguidos PnL < -2% → halt + investigación"
echo "    - ECE > 0.10 sostenido 2 semanas → 003B diagnosis"