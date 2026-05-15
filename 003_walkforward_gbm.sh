#!/usr/bin/env bash
#
# 003_walkforward_gbm.sh — Fase 3 GBM: validación walk-forward.
#
# Reentrena el modelo ganador (LONG y SHORT por separado) sobre ventanas
# deslizantes y reporta estadísticas de robustez (mediana, p10/p90, PWR).
# Mucho más informativo que un holdout único.
#
# DEFAULTS:
#   - train_window: 12 meses
#   - test_window:  1 mes
#   - step:         1 mes
#   - walk_from:    2025-01-01 (primer test cubre Ene-2025 con train Ene-2024 → Dic-2024)
#   - walk_to:      2026-04-10 (último test cubre Mar-2026 con train Mar-2025 → Feb-2026)
#   → genera 15 ventanas walk-forward, una por mes
#
# ESPERAR ~5-15 min por ventana (refit LONG + SHORT). 15 ventanas ≈ 1-2h.
#
# MODO --raw-probs:
#   Para diagnóstico, lanza una variante que ignora el calibrador y
#   re-escanea threshold por ventana. Más defensible contra leak del
#   calibrator OOF (que vio todo el train durante el tuning).

set -euo pipefail

export RELEASE=${RELEASE:-202602_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}

# Walk-forward windowing
export TRAIN_MONTHS=${TRAIN_MONTHS:-12}
export TEST_MONTHS=${TEST_MONTHS:-1}
export STEP_MONTHS=${STEP_MONTHS:-1}
export WALK_FROM=${WALK_FROM:-2025-01-01}
export WALK_TO=${WALK_TO:-2026-04-10}

# Modo: si RAW_PROBS=1, agrega --raw-probs
export RAW_PROBS=${RAW_PROBS:-0}

export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json
SUFFIX=""
[ "${RAW_PROBS}" = "1" ] && SUFFIX="_raw"
OUT_JSON=${REPORTS_DIR}/walkforward_report${SUFFIX}.json
OUT_CSV=${REPORTS_DIR}/walkforward_windows${SUFFIX}.csv

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "FASE 3 GBM — Walk-forward validation"
echo "  Release:          ${RELEASE}"
echo "  Walk:             ${WALK_FROM} → ${WALK_TO}"
echo "  Train / test / step (meses):  ${TRAIN_MONTHS} / ${TEST_MONTHS} / ${STEP_MONTHS}"
echo "  Modo:             $([ "${RAW_PROBS}" = "1" ] && echo "RAW PROBS + thr scan" || echo "CALIBRATED + thr OOF")"

# Pre-checks
log_section "1. Pre-checks"
[ -f "${BEST_JSON}" ] || abort "${BEST_JSON} no existe. Corre fase 1 primero."
python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
study = 'oof_study_gbm_${RELEASE}_multitask'
if study not in names: raise SystemExit(f'Study {study} no existe')
print(f'✅ Study {study} accesible')
" || abort "Study no accesible"

# Lanzar
log_section "2. Walk-forward"

RAW_FLAG=""
[ "${RAW_PROBS}" = "1" ] && RAW_FLAG="--raw-probs"

python3 -m mimo.oof.main_oof_gbm_walkforward \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --best-json "${BEST_JSON}" \
  --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --walk-from ${WALK_FROM} --walk-to ${WALK_TO} \
  --train-months ${TRAIN_MONTHS} \
  --test-months  ${TEST_MONTHS} \
  --step-months  ${STEP_MONTHS} \
  --min-signals-window 15 \
  --cost-per-signal 0.05 \
  --max-drawdown-R 30 \
  ${RAW_FLAG} \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --out-json "${OUT_JSON}" \
  --out-csv  "${OUT_CSV}"

# Verificar
log_section "3. Reporte"
[ -f "${OUT_JSON}" ] || abort "No se generó ${OUT_JSON}"

python3 <<EOF
import json, math
r = json.load(open("${OUT_JSON}"))

def fnum(v, d=float("nan")): return d if v is None else float(v)
def fint(v, d=0): return d if v is None else int(v)

print(f"📋 Walk-forward summary | release={r['release']} | mode={r['mode']}")
print(f"   Windows: {r['walk_config']['walk_from']} → {r['walk_config']['walk_to']}")
print(f"   Train={r['walk_config']['train_months']}m  Test={r['walk_config']['test_months']}m  Step={r['walk_config']['step_months']}m")
print()
for side_key, side_label in (("summary_long", "LONG"), ("summary_short", "SHORT")):
    sm = r[side_key]
    if sm.get("n_windows", 0) == 0:
        print(f"  {side_label}: 0 ventanas válidas"); continue
    pwr = 100 * fnum(sm["pwr"], 0)
    print(f"  {side_label}:")
    print(f"    Ventanas válidas: {sm['n_windows']}  |  positivas (PWR): {sm['n_pos_windows']}/{sm['n_windows']} ({pwr:.0f}%)")
    print(f"    EV mediana:  {fnum(sm['ev_median']):+.4f}R   p10: {fnum(sm['ev_p10']):+.4f}R   p90: {fnum(sm['ev_p90']):+.4f}R")
    print(f"    Signals tot: {fint(sm['n_signals_total'])}  |  prec mediana: {fnum(sm['prec_median']):.3f}")
    print(f"    💰 R total:  {fnum(sm['R_total']):+.2f}R")
    print()
EOF

log_section "FASE 3 GBM COMPLETADA"
echo "  Reporte JSON:  ${OUT_JSON}"
echo "  CSV ventanas:  ${OUT_CSV}"
echo ""
echo "📋 LECTURA DEL REPORTE:"
echo "  · PWR (Positive Window Rate) > 60%   → modelo robusto al drift"
echo "  · EV mediana > 0 + p10 > -0.05R     → muy positivo, distribución sana"
echo "  · PWR ~50% pero EV mediana > 0       → señal débil pero presente"
echo "  · PWR < 40% o EV mediana < 0         → no generaliza; cambiar enfoque"
echo ""
echo "Variante alternativa (sin calibrador OOF):"
echo "  RAW_PROBS=1 bash 003_walkforward_gbm.sh"
