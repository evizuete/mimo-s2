#!/usr/bin/env bash
#
# 003B_calibrator_diagnosis.sh
# ════════════════════════════════════════════════════════════════════
# DIAGNÓSTICO informativo de calibradores. NO modifica el deploy.
#
# Genera reportes que te dicen:
#   - Si el calibrador actual está bien calibrado (ECE)
#   - Qué calibrador alternativo daría mejor ECE/AUC-PR
#   - Si vale la pena intentar un experimento de swap (003C)
#
# CUÁNDO USAR:
#   - Auditoría mensual rutinaria post-deploy
#   - Tras alerta de monitor_health (ECE > 0.08)
#   - Curiosidad: "¿hay margen de mejora en mi calibrador actual?"
#
# QUÉ HACE (en orden):
#   1. Verifica pre-requisitos (deploy + specialists deben existir)
#   2. Genera ghost_predict sobre HOLDOUT (predicciones offline del modelo deployado)
#   3. Lanza simulate_calibrators (compara iso_21d/iso_45d/iso_full/beta_*/per_state)
#   4. Lee el output y emite VEREDICTO automático con recomendación:
#        ✅ "Sin acción" si ECE actual < 0.05
#        🟡 "Vigilar" si ECE entre 0.05 y 0.08
#        ⚠️  "Considerar 003C" si ECE > 0.08 Y hay alternativa que mejora ≥ 30%
#        🔴 "Investigar drift" si ECE > 0.15 sin alternativa clara
#
# LO QUE NUNCA HACE:
#   - No toca el deploy
#   - No modifica calibradores
#   - No cambia policy stubs
#   - No reinicia servicio
#
# USO:
#   bash scripts/003B_calibrator_diagnosis.sh
#   # con overrides:
#   RELEASE=202700 HOLDOUT_TO=2026-06-30 bash scripts/003B_calibrator_diagnosis.sh
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración (sobreescribible vía env) ─────────────────────────
export RELEASE=${RELEASE:-202600}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}

# Umbrales del veredicto (ajustables)
export ECE_OK_THRESHOLD=${ECE_OK_THRESHOLD:-0.05}        # ECE < esto → sin acción
export ECE_WATCH_THRESHOLD=${ECE_WATCH_THRESHOLD:-0.08}  # ECE < esto → vigilar
export ECE_ALERT_THRESHOLD=${ECE_ALERT_THRESHOLD:-0.15}  # ECE > esto → drift severo
export MIN_RELATIVE_IMPROVEMENT=${MIN_RELATIVE_IMPROVEMENT:-0.30}  # alt mejora ≥ 30%

ARTIFACTS=artifacts/${RELEASE}/oof
DEPLOY=${ARTIFACTS}/deploy_validation_combined_seed${SEED}
TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=reports/calibrator_diagnosis
mkdir -p ${REPORT_DIR}

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

# ─── Banner ──────────────────────────────────────────────────────────
log_section "DIAGNÓSTICO DE CALIBRADORES (informativo, no modifica nada)"
echo "  Release:       ${RELEASE}"
echo "  Deploy:        ${DEPLOY}"
echo "  Holdout window: ${HOLDOUT_FROM} → ${HOLDOUT_TO}"
echo "  Timestamp:     ${TS}"
echo ""
echo "  Umbrales del veredicto:"
echo "    ECE OK:     < ${ECE_OK_THRESHOLD}"
echo "    ECE watch:  < ${ECE_WATCH_THRESHOLD}"
echo "    ECE alert:  > ${ECE_ALERT_THRESHOLD}"
echo "    Alt mejora min: ≥ ${MIN_RELATIVE_IMPROVEMENT}× (sobre baseline)"

# ─── Pre-checks ─────────────────────────────────────────────────────
log_section "1. Pre-checks"

[ -d "${DEPLOY}" ] || abort "${DEPLOY} no existe — ejecuta script 003 antes"

for SIDE in long short; do
  HP=${ARTIFACTS}/${TAG}_${SIDE}_specialist_seed${SEED}/data/holdout_predictions_${RELEASE}_${SIDE}.parquet
  [ -f "${HP}" ] || abort "Falta ${HP} — ejecuta 002 antes"
done

# Verifica que tenemos las herramientas necesarias
python3 -c "import sklearn, pandas, numpy, optuna" 2>/dev/null \
  || abort "Faltan dependencias Python (sklearn/pandas/numpy/optuna)"

python3 -c "import betacal" 2>/dev/null \
  || echo "  ⚠️  betacal no instalado → métodos beta_* se saltarán. \`pip install betacal\` para usarlos."

echo "✅ Pre-requisitos OK"

# ─── 2. Ghost predict sobre HOLDOUT ─────────────────────────────────
log_section "2. ghost_predict sobre HOLDOUT"

GHOST=/tmp/ghost_holdout_${RELEASE}_${TS}.parquet

python3 scripts/ghost_predict.py \
  --release ${RELEASE} \
  --deploy-subdir $(basename ${DEPLOY}) \
  --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} \
  --include-tail \
  --out ${GHOST}

[ -f "${GHOST}" ] || abort "ghost_predict no produjo output"
echo "✅ Parquet generado: ${GHOST}"

# ─── 3. simulate_calibrators ────────────────────────────────────────
log_section "3. simulate_calibrators (comparativa de métodos)"

CAL_REPORT=${REPORT_DIR}/cal_sim_${RELEASE}_${TS}.csv
CHAMPION_CONFIG=${REPORT_DIR}/champion_config_${RELEASE}_${TS}.json

python3 scripts/simulate_calibrators.py \
  --release ${RELEASE} \
  --specialist-tag ${TAG} \
  --seed ${SEED} \
  --test-parquet ${GHOST} \
  --out-report ${CAL_REPORT}

# Copiar el champion_config emitido por simulate_calibrators al dir versionado
# (simulate_calibrators lo escribe en reports/champion_config.json por defecto)
if [ -f "reports/champion_config.json" ]; then
  cp reports/champion_config.json ${CHAMPION_CONFIG}
fi

[ -f "${CAL_REPORT}" ] || abort "simulate_calibrators no produjo CSV"
echo "✅ Reporte: ${CAL_REPORT}"

# ─── 4. Análisis del reporte y veredicto ────────────────────────────
log_section "4. Análisis automático del reporte"

VERDICT_OUT=$(python3 <<EOF
import pandas as pd
import json
from pathlib import Path

df = pd.read_csv("${CAL_REPORT}")
ece_ok = ${ECE_OK_THRESHOLD}
ece_watch = ${ECE_WATCH_THRESHOLD}
ece_alert = ${ECE_ALERT_THRESHOLD}
min_improvement = ${MIN_RELATIVE_IMPROVEMENT}

baseline_method = "iso_21d"  # lo que produce resume_deploy_v6 por defecto

# Por side, comparar baseline vs alternativas
verdicts = {}

for side in df["side"].unique():
    df_side = df[df["side"] == side]

    baseline_row = df_side[df_side["method"] == baseline_method]
    if baseline_row.empty:
        verdicts[side] = {
            "status": "ERROR",
            "reason": f"No hay baseline ({baseline_method}) en el reporte",
        }
        continue

    baseline_ece = float(baseline_row["ece"].iloc[0])
    baseline_aucpr = float(baseline_row["auc_pr"].iloc[0]) if "auc_pr" in baseline_row.columns else 0.0
    baseline_satur = float(baseline_row["pct_at_max"].iloc[0]) if "pct_at_max" in baseline_row.columns else 0.0

    # Mejor alternativa (no baseline)
    df_alts = df_side[df_side["method"] != baseline_method].sort_values("ece")
    if df_alts.empty:
        best_alt = None
    else:
        best_alt = df_alts.iloc[0].to_dict()

    # Veredicto por side
    if baseline_ece < ece_ok:
        status = "✅ OK"
        action = "Sin acción. Baseline bien calibrado."
    elif baseline_ece < ece_watch:
        status = "🟡 WATCH"
        action = "Calibración aceptable pero monitorizar próximas semanas."
    elif baseline_ece > ece_alert:
        status = "🔴 ALERT"
        action = "Drift severo. Probable necesidad de retraining (Fase 1-2) o redeploy."
    else:
        # Entre watch y alert: ¿hay alternativa significativamente mejor?
        if best_alt is not None:
            alt_ece = float(best_alt["ece"])
            improvement = (baseline_ece - alt_ece) / baseline_ece if baseline_ece > 0 else 0
            if improvement >= min_improvement:
                status = "⚠️  EXPERIMENT"
                action = f"Considera 003C: {best_alt['method']} reduce ECE de {baseline_ece:.4f} a {alt_ece:.4f} ({improvement*100:.0f}% mejor)"
            else:
                status = "🟡 WATCH"
                action = f"ECE moderado pero ninguna alternativa mejora ≥{min_improvement*100:.0f}%. Vigilar."
        else:
            status = "🟡 WATCH"
            action = "ECE moderado, sin alternativas evaluables."

    verdicts[side] = {
        "status": status,
        "baseline_method": baseline_method,
        "baseline_ece": baseline_ece,
        "baseline_aucpr": baseline_aucpr,
        "baseline_saturation_pct": baseline_satur,
        "best_alternative": best_alt["method"] if best_alt else None,
        "best_alt_ece": float(best_alt["ece"]) if best_alt else None,
        "best_alt_aucpr": float(best_alt.get("auc_pr", 0)) if best_alt else None,
        "action": action,
    }

# Determinar veredicto global (el más severo de los dos sides)
status_priority = {"🔴 ALERT": 4, "⚠️  EXPERIMENT": 3, "🟡 WATCH": 2, "✅ OK": 1, "ERROR": 0}
global_status = max(verdicts.values(),
                    key=lambda v: status_priority.get(v["status"], 0))["status"]

# Imprimir por side
for side, v in verdicts.items():
    print(f"\n──────── SIDE = {side.upper()} ────────")
    print(f"  Status:           {v['status']}")
    print(f"  Baseline:         {v.get('baseline_method', '?')}")
    print(f"  Baseline ECE:     {v.get('baseline_ece', 0):.4f}")
    print(f"  Baseline AUC-PR:  {v.get('baseline_aucpr', 0):.4f}")
    print(f"  Saturación:       {v.get('baseline_saturation_pct', 0):.2f}% en cap")
    if v.get("best_alternative"):
        print(f"  Mejor alternativa: {v['best_alternative']}")
        print(f"     ECE:           {v.get('best_alt_ece', 0):.4f}")
        print(f"     AUC-PR:        {v.get('best_alt_aucpr', 0):.4f}")
    print(f"  📋 Acción:        {v['action']}")

print(f"\n══════════════════════════════════════════════════════════")
print(f"  VEREDICTO GLOBAL: {global_status}")
print(f"══════════════════════════════════════════════════════════")

# Guardar veredicto estructurado para auditoría
out_json = Path("${REPORT_DIR}/verdict_${RELEASE}_${TS}.json")
out_json.parent.mkdir(parents=True, exist_ok=True)
out_json.write_text(json.dumps({
    "timestamp": "${TS}",
    "release": "${RELEASE}",
    "holdout_window": ["${HOLDOUT_FROM}", "${HOLDOUT_TO}"],
    "deploy": "${DEPLOY}",
    "global_status": global_status,
    "by_side": verdicts,
    "thresholds": {
        "ece_ok": ece_ok,
        "ece_watch": ece_watch,
        "ece_alert": ece_alert,
        "min_relative_improvement": min_improvement,
    },
    "reports": {
        "cal_sim": "${CAL_REPORT}",
        "champion_config": "${CHAMPION_CONFIG}",
    },
}, indent=2, default=str))
print(f"\n📁 Veredicto JSON: {out_json}")
EOF
)

echo "${VERDICT_OUT}"

# ─── 5. Recomendaciones según veredicto ─────────────────────────────
log_section "5. Recomendaciones"

# Extraer veredicto global de la salida (busca la línea "VEREDICTO GLOBAL:")
GLOBAL_STATUS=$(echo "${VERDICT_OUT}" | grep "VEREDICTO GLOBAL:" | sed 's/.*VEREDICTO GLOBAL: //')

case "${GLOBAL_STATUS}" in
  *OK*)
    echo "✅ Tu calibrador actual funciona bien. No requiere acción."
    echo ""
    echo "   Próxima diagnosis recomendada: en 4 semanas o tras alerta de monitor_health."
    ;;
  *WATCH*)
    echo "🟡 Calibración en zona aceptable pero no ideal."
    echo ""
    echo "   Acciones sugeridas:"
    echo "   - Repetir este diagnóstico en 2 semanas"
    echo "   - Ejecutar monitor_health.py semanalmente"
    echo "   - NO ejecutar 003C todavía (no hay margen claro de mejora)"
    ;;
  *EXPERIMENT*)
    echo "⚠️  Existe una alternativa de calibrador con potencial mejora."
    echo ""
    echo "   Acciones sugeridas:"
    echo "   1. Revisa ${CAL_REPORT} para entender qué calibrador propone el champion"
    echo "   2. Si te parece razonable, ejecuta el experimento controlado:"
    echo "        bash scripts/003C_calibrator_swap_experiment.sh"
    echo ""
    echo "      003C aplicará el swap, validará sobre LOCKBOX, y AUTOMÁTICAMENTE"
    echo "      decidirá si adoptar o rollback según PnL real (no solo ECE)."
    echo ""
    echo "   3. Si NO te parece justificado, ignora la sugerencia."
    echo ""
    echo "   ⚠️  Recordatorio: ECE mejor ≠ PnL mejor."
    echo "       El swap solo se adopta si mejora PnL sobre LOCKBOX."
    ;;
  *ALERT*)
    echo "🔴 Drift severo detectado en la calibración."
    echo ""
    echo "   Acciones URGENTES:"
    echo "   1. Revisa que el modelo está operando con datos válidos (no gaps, no drift de mercado)"
    echo "   2. Lanza monitor_health.py sobre las últimas 2 semanas:"
    echo "        python3 scripts/monitor_health.py --release ${RELEASE} \\"
    echo "          --deploy-subdir \$(basename ${DEPLOY}) \\"
    echo "          --from \$(date -d '14 days ago' +%F) --to \$(date +%F)"
    echo ""
    echo "   3. Si el PnL en vivo también muestra degradación → REDEPLOY COMPLETO:"
    echo "      bash 001_optuna.sh  # nueva búsqueda de hyperparams"
    echo "      bash 002_train_specialists.sh"
    echo "      bash 003_calibrate_policy.sh"
    echo "      bash 004_validate_lockbox.sh"
    echo ""
    echo "   4. Si NO hay degradación de PnL, puede ser falsa alarma de calibración."
    echo "      Aún así, programa redeploy preventivo en próximos 7 días."
    ;;
  *ERROR*)
    echo "❌ Error en el análisis. Revisa ${CAL_REPORT} y ${VERDICT_OUT} manualmente."
    ;;
esac

# ─── 6. Resumen final ───────────────────────────────────────────────
log_section "DIAGNÓSTICO COMPLETADO"
echo "  Timestamp:           ${TS}"
echo "  Veredicto global:    ${GLOBAL_STATUS}"
echo ""
echo "  📁 Outputs (versionados):"
echo "     - Reporte calibradores: ${CAL_REPORT}"
echo "     - Champion config:      ${CHAMPION_CONFIG}"
echo "     - Veredicto JSON:       ${REPORT_DIR}/verdict_${RELEASE}_${TS}.json"
echo ""
echo "  📁 Outputs efímeros (/tmp, se borrarán):"
echo "     - Ghost parquet:        ${GHOST}"
echo ""
echo "ℹ️  Este script NO modificó nada del deploy."
echo "   Para aplicar swap → 003C_calibrator_swap_experiment.sh"
echo "   Para monitorización continua → monitor_health.py + dashboard_health.py"