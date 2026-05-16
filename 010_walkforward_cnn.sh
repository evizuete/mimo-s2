#!/usr/bin/env bash
#
# 010_walkforward_cnn.sh — Walkforward CNN-LSTM refit-per-ventana.
# Apples-to-apples vs el walkforward GBM (003) para comparativa formal.
#
# Carga los best hyperparams del CNN del Optuna study existente y
# reentrena UN modelo CNN multitask por cada ventana mensual.
#
# COSTE: 15 ventanas × ~30-60min entrenamiento CNN = 8-15h GPU/CPU.
# Para reducir: usa --epochs 30 --patience 6 (menos epochs por ventana).
#
# SIMPLIFICACIONES vs producción CNN:
#   · Sin specialists (solo CNN base multitask)
#   · Sin RL gate
#   · Sin calibrator isotónico (raw probs + threshold scan)
#   · Mismos hyperparams del best Optuna trial
#
# USO:
#   bash 010_walkforward_cnn.sh
#   RELEASE=202500 CNN_STUDY=oof_study_202500_multitask bash 010_walkforward_cnn.sh

set -euo pipefail

export RELEASE=${RELEASE:-202500}
export CNN_STUDY=${CNN_STUDY:-oof_study_${RELEASE}_multitask}
export TAG=${TAG:-deploy_2026_04_combined_specialists_seed47}
export SEED=${SEED:-47}

# Arquitectura: original_v3 | mlp | hybrid | transformer | tcn
# Defaults razonables de epochs por arch:
#   original_v3 → 40 (modelo grande, más epochs)
#   mlp         → 30 (modelo pequeño, converge rápido)
#   hybrid      → 35
#   transformer → 30 (más sensible a overfit)
#   tcn         → 35
export ARCH=${ARCH:-original_v3}
case "${ARCH}" in
  mlp)         _DEFAULT_EPOCHS=30; _DEFAULT_PATIENCE=6 ;;
  hybrid)      _DEFAULT_EPOCHS=35; _DEFAULT_PATIENCE=7 ;;
  transformer) _DEFAULT_EPOCHS=30; _DEFAULT_PATIENCE=6 ;;
  tcn)         _DEFAULT_EPOCHS=35; _DEFAULT_PATIENCE=7 ;;
  *)           _DEFAULT_EPOCHS=40; _DEFAULT_PATIENCE=8 ;;
esac

# Walk-forward windowing (mismas defaults que GBM)
export WALK_FROM=${WALK_FROM:-2025-01-01}
export WALK_TO=${WALK_TO:-2026-04-10}
export TRAIN_MONTHS=${TRAIN_MONTHS:-12}
export TEST_MONTHS=${TEST_MONTHS:-1}
export STEP_MONTHS=${STEP_MONTHS:-1}

# Training acelerado por ventana (no 90 epochs como producción)
export EPOCHS=${EPOCHS:-${_DEFAULT_EPOCHS}}
export PATIENCE=${PATIENCE:-${_DEFAULT_PATIENCE}}

# Defaults del problema
export LH_LONG=${LH_LONG:-3}
export LH_SHORT=${LH_SHORT:-3}
export COST_PER_SIGNAL=${COST_PER_SIGNAL:-0.05}
export MAX_DD_R=${MAX_DD_R:-30.0}

export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
# Suffix por arch para no pisar reports si lanzas múltiples archs
_ARCH_SUFFIX=""
[ "${ARCH}" != "original_v3" ] && _ARCH_SUFFIX="_${ARCH}"
OUT_JSON=${OUT_JSON:-${REPORTS_DIR}/walkforward_report_cnn${_ARCH_SUFFIX}.json}
OUT_CSV=${OUT_CSV:-${REPORTS_DIR}/walkforward_windows_cnn${_ARCH_SUFFIX}.csv}

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "WALK-FORWARD CNN — refit per ventana"
echo "  Release:        ${RELEASE}"
echo "  CNN study:      ${CNN_STUDY}"
echo "  Walk:           ${WALK_FROM} → ${WALK_TO}"
echo "  Train/test/step:${TRAIN_MONTHS}m / ${TEST_MONTHS}m / ${STEP_MONTHS}m"
echo "  Epochs/patience:${EPOCHS} / ${PATIENCE}"
echo "  Arquitectura:   ${ARCH}"
echo "  Output:         ${OUT_JSON}"

log_section "1. Pre-checks"
python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
if '${CNN_STUDY}' not in names:
    matching = [n for n in names if '${RELEASE}' in n]
    raise SystemExit(\"Study '${CNN_STUDY}' no existe en Optuna. \"
                     f\"Studies con '${RELEASE}' en nombre: {matching[:10]}\")
s = optuna.load_study(study_name='${CNN_STUDY}', storage='${OPTUNA_STORAGE}')
n_done = sum(1 for t in s.trials if t.state.name == 'COMPLETE')
print(f\"✅ Study '${CNN_STUDY}' | {n_done} trials COMPLETE\")
" || abort "Pre-check fallido"

mkdir -p "${REPORTS_DIR}"

log_section "2. Walkforward (esto va a tardar ~8-15h)"
python3 -m mimo.oof.main_oof_cnn_walkforward \
  --release ${RELEASE} \
  --cnn-study-name "${CNN_STUDY}" \
  --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long ${LH_LONG} --label-horizon-short ${LH_SHORT} \
  --walk-from ${WALK_FROM} --walk-to ${WALK_TO} \
  --train-months ${TRAIN_MONTHS} --test-months ${TEST_MONTHS} --step-months ${STEP_MONTHS} \
  --epochs ${EPOCHS} --patience ${PATIENCE} \
  --arch ${ARCH} \
  --cost-per-signal ${COST_PER_SIGNAL} --max-drawdown-R ${MAX_DD_R} \
  --min-signals-window 15 \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --out-json "${OUT_JSON}" \
  --out-csv "${OUT_CSV}"

log_section "3. Resumen"
[ -f "${OUT_JSON}" ] || abort "No se generó ${OUT_JSON}"

python3 <<EOF
import json
import math
r = json.load(open("${OUT_JSON}"))
def fnum(v, d=float("nan")): return d if v is None else float(v)
def fint(v, d=0): return d if v is None else int(v)

print(f"📋 Walkforward CNN summary | release={r['release']} | mode={r['mode']}")
print(f"   CNN best trial: #{r['cnn_best_trial']}")
print(f"   Epochs/patience: {r['epochs']}/{r['patience']}")
print(f"   Windows: {r['walk_config']['walk_from']} → {r['walk_config']['walk_to']}")

for side_key, side_label in (("summary_long", "LONG"), ("summary_short", "SHORT")):
    sm = r.get(side_key, {})
    n = sm.get("n_windows", 0)
    if n == 0:
        print(f"  {side_label}: 0 ventanas válidas"); continue
    pwr = 100 * fnum(sm["pwr"], 0)
    print(f"\n  {side_label}:")
    print(f"    Ventanas válidas: {sm['n_windows']}  |  PWR: {sm['n_pos_windows']}/{sm['n_windows']} ({pwr:.0f}%)")
    print(f"    EV mediana:  {fnum(sm['ev_median']):+.4f}R   p10: {fnum(sm['ev_p10']):+.4f}R   p90: {fnum(sm['ev_p90']):+.4f}R")
    print(f"    Signals tot: {fint(sm['n_signals_total'])}  |  prec mediana: {fnum(sm['prec_median']):.3f}")
    print(f"    💰 R total:  {fnum(sm['R_total']):+.2f}R")
EOF

log_section "WALK-FORWARD CNN [${ARCH}] COMPLETADO"
echo "  JSON:  ${OUT_JSON}"
echo "  CSV:   ${OUT_CSV}"
echo ""
echo "📋 Para comparativa formal:"
echo "   python3 -m mimo.oof.diag_long_only_simulator \\"
echo "     --walkforward-json ${OUT_JSON} \\"
echo "     --out-json ${REPORTS_DIR}/long_only_sim_cnn${_ARCH_SUFFIX}.json"
echo ""
echo "🚀 Para correr otras arquitecturas (mismo script):"
echo "   ARCH=mlp         bash 010_walkforward_cnn.sh   # ~1-2h"
echo "   ARCH=hybrid      bash 010_walkforward_cnn.sh   # ~5-8h"
echo "   ARCH=tcn         bash 010_walkforward_cnn.sh   # ~7-10h"
echo "   ARCH=transformer bash 010_walkforward_cnn.sh   # ~3-5h"
