#!/usr/bin/env bash
#
# 004_validate_lockbox.sh
# ════════════════════════════════════════════════════════════════════
# FASE 4 — Validación final sobre LOCKBOX antes de promover a producción.
#
# Toma el deploy_validation generado por 003 y lo valida sobre datos JAMÁS
# vistos durante el entrenamiento ni la calibración (LOCKBOX). Emite veredicto
# automático: OK, MARGINAL, o KO.
#
# QUÉ HACE:
#   1. Pre-checks (deploy + policy + parquets requeridos)
#   2. Replay del NUEVO deploy sobre LOCKBOX (apr-11 → may-10 por defecto)
#   3. Ghost predict + analyze_calibration + analyze_score_to_pnl
#   4. Replay del deploy PREVIO (si existe) sobre el mismo LOCKBOX → control
#   5. Comparativa baseline vs nuevo + métricas de drift
#   6. Decisión automática:
#        ✅ OK       → puedes proceder a Fase 5 (production refit con datos full)
#        🟡 MARGINAL → continúa con cautela, documenta áreas de vigilancia
#        🔴 KO       → no promover; iterar en Fase 1-3 o re-Optuna
#
# LO QUE NUNCA HACE:
#   - No promueve a producción (eso es Fase 6 manual)
#   - No modifica calibradores ni policy stubs
#   - No reentrena nada (eso es Fase 5)
#
# CUÁNDO USAR:
#   - Inmediatamente después del script 003
#   - Antes de Fase 5 (production refit)
#   - NUNCA usar el LOCKBOX más de una vez con el mismo deploy
#     (cada uso "quema" la unbiasedness — si fallas, regenera nuevo LOCKBOX)
#
# USO:
#   bash scripts/004_validate_lockbox.sh
#   # con overrides:
#   RELEASE=202700 LOCKBOX_TO=2026-06-30 bash scripts/004_validate_lockbox.sh
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202600}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export LOCKBOX_FROM=${LOCKBOX_FROM:-2026-04-11}
export LOCKBOX_TO=${LOCKBOX_TO:-2026-05-10}
export WARMUP_FROM=${WARMUP_FROM:-2026-03-25}

# Deploys a comparar
export NEW_DEPLOY=${NEW_DEPLOY:-deploy_validation_combined_seed${SEED}}
export NEW_POLICY=${NEW_POLICY:-decision_policies_config_${RELEASE}_validation}

# Deploy ANTERIOR (control / baseline). Opcional pero recomendado.
# Si no existe, el script omite la comparativa pero sigue funcionando.
export BASELINE_DEPLOY=${BASELINE_DEPLOY:-deploy_2026_04_combined_specialists_seed47}
export BASELINE_POLICY=${BASELINE_POLICY:-decision_policies_config_202500_mar31}

# Criterios de aceptación (overridable)
export MIN_PNL_PCT=${MIN_PNL_PCT:-0}                # PnL% mínimo absoluto
export MAX_MDD_PCT=${MAX_MDD_PCT:-15.0}             # MDD% máximo (valor absoluto)
export MIN_WEEKS_POSITIVE_PCT=${MIN_WEEKS_POSITIVE_PCT:-50}  # % weeks positive mínimo
export MAX_ECE=${MAX_ECE:-0.08}                     # ECE holdout máximo
export MAX_RELATIVE_DEGRADATION=${MAX_RELATIVE_DEGRADATION:-0.10}  # vs baseline: no peor de 10%

ARTIFACTS=artifacts/${RELEASE}/oof
NEW_DEPLOY_DIR=${ARTIFACTS}/${NEW_DEPLOY}
TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=reports/lockbox_validation
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

extract_metric() {
  local summary_file="$1"
  local key="$2"
  python3 -c "
import json
try:
    s = json.load(open('${summary_file}'))
    v = s.get('${key}', 0) or 0
    print(f'{float(v):.4f}')
except Exception:
    print('0.0')
"
}

# ─── Banner ──────────────────────────────────────────────────────────
log_section "FASE 4 — Validación LOCKBOX"
echo "  Release:           ${RELEASE}"
echo "  Nuevo deploy:      ${NEW_DEPLOY}"
echo "  Nuevo policy:      ${NEW_POLICY}"
echo "  Baseline deploy:   ${BASELINE_DEPLOY}"
echo "  Baseline policy:   ${BASELINE_POLICY}"
echo "  LOCKBOX window:    ${LOCKBOX_FROM} → ${LOCKBOX_TO}"
echo "  Warmup from:       ${WARMUP_FROM}"
echo "  Timestamp:         ${TS}"
echo ""
echo "  Criterios de aceptación:"
echo "    PnL% mínimo:                  ≥ ${MIN_PNL_PCT}"
echo "    MDD% máximo (abs):            ≤ ${MAX_MDD_PCT}"
echo "    Weeks positive mínimo:        ≥ ${MIN_WEEKS_POSITIVE_PCT}%"
echo "    ECE holdout máximo:           ≤ ${MAX_ECE}"
echo "    Degradación vs baseline:      ≤ ${MAX_RELATIVE_DEGRADATION}× peor"

# ─── 1. Pre-checks ─────────────────────────────────────────────────
log_section "1. Pre-checks"

[ -d "${NEW_DEPLOY_DIR}" ] || abort "Deploy no existe: ${NEW_DEPLOY_DIR}. Ejecuta 003 antes."
[ -f "config/${NEW_POLICY}.py" ] || abort "Policy no existe: config/${NEW_POLICY}.py"

# Verifica que hay datos OHLCV en BD para el rango LOCKBOX
python3 -c "
from datetime import datetime
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
db = Database()
dm = DataManager.from_database_historical_2(db, from_date='${LOCKBOX_FROM}', to_date='${LOCKBOX_TO}')
n = len(dm.df)
if n < 1000:
    raise SystemExit(f'❌ Solo {n} filas en LOCKBOX. Esperado >>1000 para validación útil.')
print(f'✅ LOCKBOX tiene {n:,} filas disponibles')
" || exit 1

# Verifica baseline si se quiere comparar
HAS_BASELINE=0
if [ -d "${ARTIFACTS}/${BASELINE_DEPLOY}" ] || \
   [ -d "artifacts/202500/oof/${BASELINE_DEPLOY}" ]; then
  if [ -f "config/${BASELINE_POLICY}.py" ]; then
    HAS_BASELINE=1
    echo "✅ Baseline disponible para comparativa: ${BASELINE_DEPLOY}"
  else
    echo "⚠️  Baseline deploy existe pero falta policy config/${BASELINE_POLICY}.py"
    echo "    → Comparativa baseline OMITIDA"
  fi
else
  echo "⚠️  Baseline deploy no encontrado (${BASELINE_DEPLOY})"
  echo "    → Comparativa baseline OMITIDA (validación contra criterios absolutos solamente)"
fi

# ─── 2. Replay nuevo deploy ─────────────────────────────────────────
log_section "2. Replay NUEVO deploy sobre LOCKBOX"

NEW_OUT=/tmp/replay_lockbox_new_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir ${NEW_DEPLOY} \
  --policy-config ${NEW_POLICY} \
  --warmup-from ${WARMUP_FROM} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --out ${NEW_OUT}

NEW_PNL=$(extract_metric ${NEW_OUT}/summary.json pnl_pct)
NEW_MDD=$(extract_metric ${NEW_OUT}/summary.json max_drawdown_pct)
NEW_TRADES=$(extract_metric ${NEW_OUT}/summary.json n_trades)
NEW_WIN_RATE=$(extract_metric ${NEW_OUT}/summary.json win_rate_pct)
NEW_AVG_R=$(extract_metric ${NEW_OUT}/summary.json ev_net_avg_R)

echo ""
echo "  📊 NUEVO deploy LOCKBOX:"
echo "     PnL%:       ${NEW_PNL}"
echo "     MDD%:       ${NEW_MDD}"
echo "     Trades:     ${NEW_TRADES}"
echo "     Win rate%:  ${NEW_WIN_RATE}"
echo "     Avg R/trade: ${NEW_AVG_R}"

# ─── 3. Diagnóstico drift sobre LOCKBOX ─────────────────────────────
log_section "3. Diagnóstico de drift (ghost_predict + analyzers)"

GHOST_LB=/tmp/ghost_lockbox_${RELEASE}_${TS}.parquet

python3 scripts/ghost_predict.py \
  --release ${RELEASE} \
  --deploy-subdir ${NEW_DEPLOY} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --include-tail \
  --out ${GHOST_LB}

CAL_OUT=${REPORT_DIR}/lockbox_cal_${RELEASE}_${TS}
PNL_OUT=${REPORT_DIR}/lockbox_score_pnl_${RELEASE}_${TS}

python3 -m mimo.oof.shift_analyzer.analyze_calibration_mimo \
  --input ${GHOST_LB} \
  --score-col oof_proba_cal --target-col signal \
  --state-col state --time-col time \
  --period-col period \
  --output-dir ${CAL_OUT}

python3 -m mimo.oof.shift_analyzer.analyze_score_to_pnl \
  --input ${GHOST_LB} \
  --score-col oof_proba_cal --target-col signal \
  --pnl-col R_multiple --state-col state --time-col time \
  --period-col period \
  --output-dir ${PNL_OUT}

# Extraer ECE del holdout (= ventana LOCKBOX en este caso)
NEW_ECE=$(python3 -c "
import json
try:
    s = json.load(open('${CAL_OUT}/summary.json'))
    holdout = next((p for p in s.get('overall_periods', [])
                   if p.get('period') == 'holdout'), {})
    print(f\"{float(holdout.get('ece', 0)):.4f}\")
except Exception:
    print('0.0')
")
NEW_AUC_PR=$(python3 -c "
import json
try:
    s = json.load(open('${CAL_OUT}/summary.json'))
    holdout = next((p for p in s.get('overall_periods', [])
                   if p.get('period') == 'holdout'), {})
    print(f\"{float(holdout.get('auc_pr', 0)):.4f}\")
except Exception:
    print('0.0')
")
NEW_POS_RATE=$(python3 -c "
import json
try:
    s = json.load(open('${CAL_OUT}/summary.json'))
    holdout = next((p for p in s.get('overall_periods', [])
                   if p.get('period') == 'holdout'), {})
    print(f\"{float(holdout.get('pos_rate', 0)):.4f}\")
except Exception:
    print('0.0')
")

echo ""
echo "  📊 Métricas LOCKBOX:"
echo "     ECE:        ${NEW_ECE}"
echo "     AUC-PR:     ${NEW_AUC_PR}"
echo "     pos_rate:   ${NEW_POS_RATE}"

# Identificar estados tóxicos (PnL/trade < -0.10R con n>=30)
TOXIC_STATES=$(python3 <<EOF
import pandas as pd
try:
    df = pd.read_csv("${PNL_OUT}/score_to_pnl_by_period_state.csv")
    hd = df[df["period"] == "holdout"]
    if hd.empty:
        print("")
    else:
        agg = hd.groupby("state").agg(n=("n", "sum"), total_pnl=("total_pnl", "sum"))
        agg["pnl_per_trade"] = agg["total_pnl"] / agg["n"].clip(lower=1)
        toxic = agg[(agg["n"] >= 30) & (agg["pnl_per_trade"] < -0.10)]
        print(",".join(toxic.index.tolist()) if not toxic.empty else "")
except Exception:
    print("")
EOF
)
if [ -n "${TOXIC_STATES}" ]; then
  echo "  ⚠️  Estados tóxicos detectados: ${TOXIC_STATES}"
else
  echo "  ✅ Sin estados tóxicos (todos PnL/trade >= -0.10R en n>=30)"
fi

# ─── 4. Replay baseline (si existe) ─────────────────────────────────
BASELINE_PNL="N/A"
BASELINE_MDD="N/A"
BASELINE_TRADES="N/A"
BASELINE_WIN_RATE="N/A"

if [ "${HAS_BASELINE}" = "1" ]; then
  log_section "4. Replay BASELINE (control) sobre LOCKBOX"
  
  # Detectar dónde está el baseline (en 202500 si es deploy antiguo, o en RELEASE actual)
  BASELINE_RELEASE=${RELEASE}
  if [ -d "artifacts/202500/oof/${BASELINE_DEPLOY}" ]; then
    BASELINE_RELEASE=202500
  fi
  
  BASELINE_OUT=/tmp/replay_lockbox_baseline_${RELEASE}_${TS}
  python3 scripts/replay_s2_202500.py \
    --release ${BASELINE_RELEASE} \
    --deploy-subdir ${BASELINE_DEPLOY} \
    --policy-config ${BASELINE_POLICY} \
    --warmup-from ${WARMUP_FROM} \
    --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
    --out ${BASELINE_OUT}

  BASELINE_PNL=$(extract_metric ${BASELINE_OUT}/summary.json pnl_pct)
  BASELINE_MDD=$(extract_metric ${BASELINE_OUT}/summary.json max_drawdown_pct)
  BASELINE_TRADES=$(extract_metric ${BASELINE_OUT}/summary.json n_trades)
  BASELINE_WIN_RATE=$(extract_metric ${BASELINE_OUT}/summary.json win_rate_pct)

  echo ""
  echo "  📊 BASELINE LOCKBOX:"
  echo "     PnL%:       ${BASELINE_PNL}"
  echo "     MDD%:       ${BASELINE_MDD}"
  echo "     Trades:     ${BASELINE_TRADES}"
  echo "     Win rate%:  ${BASELINE_WIN_RATE}"
fi

# ─── 5. Comparativa y veredicto ─────────────────────────────────────
log_section "5. COMPARATIVA Y VEREDICTO"

if [ "${HAS_BASELINE}" = "1" ]; then
  echo ""
  echo "                          NUEVO       BASELINE      Δ"
  echo "  ─────────────────────────────────────────────────────────"
  printf "  PnL%%:                 %+8.2f   %+8.2f    %+8.2f\n" \
    "${NEW_PNL}" "${BASELINE_PNL}" \
    "$(python3 -c "print(${NEW_PNL} - ${BASELINE_PNL})")"
  printf "  MDD%%:                 %+8.2f   %+8.2f\n" \
    "${NEW_MDD}" "${BASELINE_MDD}"
  printf "  Trades:               %8.0f   %8.0f\n" \
    "${NEW_TRADES}" "${BASELINE_TRADES}"
  printf "  Win rate%%:            %8.2f   %8.2f\n" \
    "${NEW_WIN_RATE}" "${BASELINE_WIN_RATE}"
fi

# Lógica de veredicto multi-criterio
VERDICT=$(python3 <<EOF
import sys

new_pnl = ${NEW_PNL}
new_mdd = abs(${NEW_MDD})
new_ece = ${NEW_ECE}
new_aucpr = ${NEW_AUC_PR}
new_pos_rate = ${NEW_POS_RATE}
new_trades = int(${NEW_TRADES})

baseline_pnl = ${BASELINE_PNL} if "${HAS_BASELINE}" == "1" else None

min_pnl = ${MIN_PNL_PCT}
max_mdd = ${MAX_MDD_PCT}
max_ece = ${MAX_ECE}
max_degradation = ${MAX_RELATIVE_DEGRADATION}

toxic_states = "${TOXIC_STATES}".split(",") if "${TOXIC_STATES}" else []

reasons_ok = []
reasons_warn = []
reasons_ko = []

# ─── Criterios absolutos ───
if new_pnl >= min_pnl + 5:
    reasons_ok.append(f"PnL {new_pnl:+.2f}% claramente > {min_pnl}")
elif new_pnl >= min_pnl:
    reasons_warn.append(f"PnL {new_pnl:+.2f}% justo en el límite ({min_pnl})")
else:
    reasons_ko.append(f"PnL {new_pnl:+.2f}% < umbral {min_pnl}")

if new_mdd < max_mdd * 0.7:
    reasons_ok.append(f"MDD {new_mdd:.2f}% holgadamente < {max_mdd}")
elif new_mdd <= max_mdd:
    reasons_warn.append(f"MDD {new_mdd:.2f}% cerca del límite {max_mdd}")
else:
    reasons_ko.append(f"MDD {new_mdd:.2f}% > {max_mdd}")

if new_ece < max_ece * 0.6:
    reasons_ok.append(f"ECE {new_ece:.4f} muy bajo")
elif new_ece <= max_ece:
    reasons_warn.append(f"ECE {new_ece:.4f} en zona aceptable")
else:
    reasons_ko.append(f"ECE {new_ece:.4f} > {max_ece}")

# AUC-PR: debe ser al menos pos_rate × 1.05 (mejor que random)
if new_pos_rate > 0:
    auc_floor = new_pos_rate * 1.05
    if new_aucpr >= auc_floor:
        reasons_ok.append(f"AUC-PR {new_aucpr:.4f} > floor {auc_floor:.4f}")
    else:
        reasons_warn.append(f"AUC-PR {new_aucpr:.4f} < floor {auc_floor:.4f} — modelo borderline")

if toxic_states and toxic_states != [""]:
    n_toxic = len(toxic_states)
    if n_toxic >= 3:
        reasons_ko.append(f"{n_toxic} estados tóxicos: {','.join(toxic_states)}")
    elif n_toxic == 2:
        reasons_warn.append(f"2 estados tóxicos: {','.join(toxic_states)}")
    else:
        reasons_warn.append(f"1 estado tóxico: {','.join(toxic_states)}")

# ─── Comparación con baseline (si existe) ───
if baseline_pnl is not None:
    if baseline_pnl > 0:
        relative_deg = (baseline_pnl - new_pnl) / baseline_pnl
        if relative_deg > max_degradation:
            reasons_ko.append(
                f"Degrada {relative_deg*100:.0f}% vs baseline ({new_pnl:+.2f}% vs {baseline_pnl:+.2f}%)"
            )
        elif relative_deg > 0:
            reasons_warn.append(
                f"Ligera degradación vs baseline ({new_pnl:+.2f}% vs {baseline_pnl:+.2f}%)"
            )
        else:
            reasons_ok.append(
                f"Iguala o mejora baseline ({new_pnl:+.2f}% vs {baseline_pnl:+.2f}%)"
            )

# ─── Veredicto final ───
if reasons_ko:
    verdict = "🔴 KO"
elif len(reasons_warn) >= 3:
    verdict = "🔴 KO"  # demasiadas warnings = KO
elif reasons_warn:
    verdict = "🟡 MARGINAL"
else:
    verdict = "✅ OK"

print(f"VERDICT={verdict}")
print(f"N_OK={len(reasons_ok)}")
print(f"N_WARN={len(reasons_warn)}")
print(f"N_KO={len(reasons_ko)}")
for r in reasons_ok:
    print(f"OK::{r}")
for r in reasons_warn:
    print(f"WARN::{r}")
for r in reasons_ko:
    print(f"KO::{r}")
EOF
)

# Parsear el output
VERDICT_LINE=$(echo "$VERDICT" | grep "^VERDICT=" | head -1 | cut -d= -f2-)
N_OK=$(echo "$VERDICT" | grep "^N_OK=" | head -1 | cut -d= -f2)
N_WARN=$(echo "$VERDICT" | grep "^N_WARN=" | head -1 | cut -d= -f2)
N_KO=$(echo "$VERDICT" | grep "^N_KO=" | head -1 | cut -d= -f2)

echo ""
echo "  📊 VEREDICTO: ${VERDICT_LINE}"
echo "     ✅ OK criteria:   ${N_OK}"
echo "     🟡 Warnings:      ${N_WARN}"
echo "     🔴 KO criteria:   ${N_KO}"
echo ""

# Imprimir razones
if [ "${N_OK}" -gt 0 ]; then
  echo "  Criterios cumplidos:"
  echo "$VERDICT" | grep "^OK::" | sed 's/OK::/    ✅ /'
fi
if [ "${N_WARN}" -gt 0 ]; then
  echo "  Warnings:"
  echo "$VERDICT" | grep "^WARN::" | sed 's/WARN::/    🟡 /'
fi
if [ "${N_KO}" -gt 0 ]; then
  echo "  Criterios incumplidos:"
  echo "$VERDICT" | grep "^KO::" | sed 's/KO::/    🔴 /'
fi

# ─── 6. Recomendación de próximo paso ───────────────────────────────
log_section "6. Recomendación"

case "${VERDICT_LINE}" in
  *OK*)
    echo "✅ DEPLOY VALIDADO. Procede a Fase 5 (production refit con datos full)."
    echo ""
    echo "   Siguiente comando:"
    echo "     bash scripts/005_production_refit.sh"
    echo ""
    echo "   Después:"
    echo "     bash scripts/006_promote_to_production.sh  (manual, requiere intervención)"
    ;;
  *MARGINAL*)
    echo "🟡 DEPLOY MARGINAL. Procede con CAUTELA o itera."
    echo ""
    echo "   Opciones:"
    echo "   A) Documentar warnings y proceder a Fase 5"
    echo "      - Registra los warnings en docs/RELEASES.md"
    echo "      - Monitor especialmente cerca de los criterios marginales"
    echo "      - bash scripts/005_production_refit.sh"
    echo ""
    echo "   B) Iterar para mejorar (recomendado si los warnings son por margen ajustado)"
    echo "      - Investiga si la policy tiene margen de tuning"
    echo "      - Considera re-Optuna ligero (30 trials) con --inherit-config-from"
    echo "      - Para policy tuning: replay con --policy-grid"
    ;;
  *KO*)
    echo "🔴 DEPLOY NO ADECUADO. NO promover a producción."
    echo ""
    echo "   Diagnóstico recomendado:"
    echo "   1. Revisa los criterios KO arriba"
    echo "   2. Mira el daily_breakdown.csv para identificar días anómalos:"
    echo "        cat ${NEW_OUT}/daily_breakdown.csv"
    echo "   3. Revisa estados tóxicos en analyze_score_to_pnl:"
    echo "        cat ${PNL_OUT}/score_to_pnl_by_period_state.csv"
    echo ""
    echo "   Acciones posibles según diagnóstico:"
    echo "   - Drift de calibración → 003B + 003C"
    echo "   - Drift estructural    → re-Optuna (Fase 1) con datos más recientes"
    echo "   - Estado tóxico persistente → endurecer gates en policy stub"
    echo "   - Régimen muy distinto a TRAIN → considerar widening del train period"
    echo ""
    echo "   ⚠️  NO uses el mismo LOCKBOX para iteraciones múltiples — overfit."
    echo "       Para iterar: usa una ventana de validación distinta o regenera."
    ;;
esac

# ─── 7. Persistir reporte estructurado ──────────────────────────────
log_section "7. Persistencia del veredicto"

REPORT_JSON=${REPORT_DIR}/validation_${RELEASE}_${TS}.json
python3 <<EOF
import json
from pathlib import Path

report = {
    "timestamp": "${TS}",
    "release": "${RELEASE}",
    "lockbox_window": ["${LOCKBOX_FROM}", "${LOCKBOX_TO}"],
    "new_deploy": "${NEW_DEPLOY}",
    "new_policy": "${NEW_POLICY}",
    "baseline_deploy": "${BASELINE_DEPLOY}" if "${HAS_BASELINE}" == "1" else None,
    "baseline_policy": "${BASELINE_POLICY}" if "${HAS_BASELINE}" == "1" else None,
    "verdict": "${VERDICT_LINE}",
    "metrics": {
        "new": {
            "pnl_pct": ${NEW_PNL},
            "mdd_pct": ${NEW_MDD},
            "n_trades": int(${NEW_TRADES}),
            "win_rate_pct": ${NEW_WIN_RATE},
            "avg_R_per_trade": ${NEW_AVG_R},
            "ece": ${NEW_ECE},
            "auc_pr": ${NEW_AUC_PR},
            "pos_rate": ${NEW_POS_RATE},
        },
    },
    "toxic_states": "${TOXIC_STATES}".split(",") if "${TOXIC_STATES}" else [],
    "n_criteria_ok": int(${N_OK}),
    "n_warnings": int(${N_WARN}),
    "n_criteria_ko": int(${N_KO}),
    "thresholds": {
        "min_pnl_pct": ${MIN_PNL_PCT},
        "max_mdd_pct": ${MAX_MDD_PCT},
        "max_ece": ${MAX_ECE},
        "max_relative_degradation": ${MAX_RELATIVE_DEGRADATION},
    },
    "outputs": {
        "new_replay": "${NEW_OUT}",
        "ghost_parquet": "${GHOST_LB}",
        "calibration_report": "${CAL_OUT}",
        "score_to_pnl_report": "${PNL_OUT}",
    },
}

if "${HAS_BASELINE}" == "1":
    report["metrics"]["baseline"] = {
        "pnl_pct": ${BASELINE_PNL},
        "mdd_pct": ${BASELINE_MDD},
        "n_trades": int(${BASELINE_TRADES}),
        "win_rate_pct": ${BASELINE_WIN_RATE},
    }
    report["outputs"]["baseline_replay"] = "${BASELINE_OUT}"

Path("${REPORT_JSON}").write_text(json.dumps(report, indent=2, default=str))
print(f"✅ Reporte: ${REPORT_JSON}")
EOF

# ─── 8. Resumen final ──────────────────────────────────────────────
log_section "VALIDACIÓN COMPLETADA"
echo "  Veredicto:           ${VERDICT_LINE}"
echo "  Timestamp:           ${TS}"
echo "  Reporte completo:    ${REPORT_JSON}"
echo ""
echo "  📁 Outputs detallados:"
echo "    - Replay nuevo:    ${NEW_OUT}/"
echo "    - Calibración:     ${CAL_OUT}/"
echo "    - Score → PnL:     ${PNL_OUT}/"
if [ "${HAS_BASELINE}" = "1" ]; then
  echo "    - Replay baseline: ${BASELINE_OUT}/"
fi
echo ""
echo "ℹ️  Este script NO modificó nada del deploy."
echo "   Toda decisión sobre promover/iterar es del operador humano basándose en el veredicto."