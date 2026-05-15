#!/usr/bin/env bash
#
# 001_hyperparams_tuning_gbm.sh — Fase 1 GBM: tuning de LightGBM con Optuna OOF.
#
# Paralelo a 001_hyperparams_tuning.sh (CNN-LSTM) pero entrenando árboles.
# Reusa todo el feature engineering, regime weights y barriers del CNN.
# Genera un Optuna study INDEPENDIENTE (oof_study_gbm_<release>_multitask)
# para no contaminar la búsqueda del CNN.
#
# CONSEJOS:
#   - Para BÚSQUEDA REAL: --optuna-trials 30-100. GBM es mucho más rápido
#     que la CNN (~30s-2min por trial vs 6+min) → 100 trials caben en ~3-4h.
#   - Para SMOKE TEST: --optuna-trials 1.
#
# Diferencias importantes vs 001_hyperparams_tuning.sh:
#   - Llama main_oof_gbm (no main_oof_regime_weights_v7)
#   - RELEASE default 202600_GBM (no 202601)
#   - Sin --use-grid (GBM trainer es TPE only)
#   - Sin --oof-epochs/--oof-patience (los maneja Optuna como n_estimators
#     y early_stopping_rounds)

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
# Default: 202604_GBM (barriers conservadoras tp=2.5/sl=1.5 + grid afinado
# tras los 58 trials del 202603). No hereda barriers (las trae propias) pero
# sí features (vol_invariant, reduced) de 202601.
#
# Otros releases disponibles:
#   RELEASE=202603_GBM INHERIT_FROM_RELEASE=202601 ...  (grid amplio default)
#   RELEASE=202602_GBM INHERIT_FROM_RELEASE=202601 ...  (grid baseline)
#   RELEASE=202600_GBM INHERIT_FROM_RELEASE=202500 ...  (barriers conservadoras + grid amplio)
export RELEASE=${RELEASE:-202604_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}

# OPTUNA_TRIALS = NÚMERO TOTAL OBJETIVO de trials COMPLETE en el study.
# El script calcula automáticamente cuántos NUEVOS lanzar (= TARGET - YA_COMPLETOS).
# Si el study ya está al target, el paso se salta.
# GBM es rápido (~5-15 min/trial CPU). 80 trials ~6-12h.
export OPTUNA_TRIALS=${OPTUNA_TRIALS:-80}

# Storage MySQL (mismo que CNN, distinto study name)
export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

REPORTS_DIR=artifacts/${RELEASE}/oof/${TAG}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json

# ─── Helpers ────────────────────────────────────────────────────────
log_section() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
}
abort() { echo "❌ $1"; exit 1; }

# ─── Banner ─────────────────────────────────────────────────────────
log_section "FASE 1 GBM — LightGBM tuning (paralelo al CNN)"
echo "  Release:           ${RELEASE}"
echo "  Inherit from:      ${INHERIT_FROM_RELEASE}"
echo "  Tag (exp):         ${TAG}"
echo "  Seed:              ${SEED}"
echo "  Train period:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Holdout period:    ${HOLDOUT_FROM} → ${HOLDOUT_TO}"
echo "  Optuna trials:     ${OPTUNA_TRIALS}"
if [ "${OPTUNA_TRIALS}" -lt 10 ]; then
  echo ""
  echo "  ⚠️  TRIALS BAJO (${OPTUNA_TRIALS}). Esto NO es búsqueda real."
fi

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

python3 -c "
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
db = Database()
dm = DataManager.from_database_historical_2(
    db, from_date='${TRAIN_FROM}', to_date='${HOLDOUT_TO}')
n = len(dm.df)
if n < 100_000:
    raise SystemExit(f'❌ Solo {n:,} filas en BD para el rango — esperado >>100k')
print(f'✅ {n:,} filas | rango {dm.df.time.min()} → {dm.df.time.max()}')
" || abort "Fallo conexión a BD"

python3 -c "
import optuna
try:
    optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
    print('✅ MySQL Optuna alcanzable')
except Exception as e:
    raise SystemExit(f'❌ No puedo conectar a Optuna storage: {e}')
" || abort "Fallo conexión a Optuna MySQL"

python3 -c "
try:
    import lightgbm
    print(f'✅ lightgbm versión {lightgbm.__version__}')
except ImportError:
    raise SystemExit('❌ lightgbm no instalado. pip install lightgbm')
" || abort "lightgbm no disponible"

echo "✅ Pre-checks OK"

# ─── 2. Calcular trials NUEVOS a lanzar (target - ya_completos) ───
log_section "2. Calcular delta de trials"

EXPECTED_STUDY="oof_study_gbm_${RELEASE}_multitask"
N_CURRENT=$(python3 -c "
import optuna
try:
    s = optuna.load_study(study_name='${EXPECTED_STUDY}', storage='${OPTUNA_STORAGE}')
    print(sum(1 for t in s.trials if t.state.name == 'COMPLETE'))
except KeyError:
    print(0)
except Exception as e:
    import sys
    print(f'ERR:{e}', file=sys.stderr); print(0)
")

N_DELTA=$((OPTUNA_TRIALS - N_CURRENT))
echo "  Trials completados en study : ${N_CURRENT}"
echo "  Target total                : ${OPTUNA_TRIALS}"
echo "  Nuevos a lanzar (delta)     : ${N_DELTA}"

if [ "${N_DELTA}" -le 0 ]; then
  echo "✅ Ya hay ${N_CURRENT} ≥ ${OPTUNA_TRIALS} trials COMPLETE. Saltando tuning."
  echo "   (Para forzar más, sube OPTUNA_TRIALS por encima de ${N_CURRENT})"
else
  log_section "3. main_oof_gbm (lanzando ${N_DELTA} nuevos trials)"
  python3 -m mimo.oof.main_oof_gbm \
    --release ${RELEASE} \
    --inherit-config-from ${INHERIT_FROM_RELEASE} \
    --base-tf 5min --target-type multitask --side both \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
    --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
    --use-tpe \
    --optuna-trials ${N_DELTA} \
    --objective ev_net --cost-per-signal 0.05 \
    --ev-min-signals 100 --max-drawdown-R 30 \
    --ev-thr-lo 0.10 --ev-thr-hi 0.40 \
    --optuna-storage "${OPTUNA_STORAGE}" \
    --seed ${SEED}
fi

# ─── 4. Verificar study en MySQL ───────────────────────────────────
log_section "4. Verificar study creado en Optuna"

STUDY_EXISTS=$(python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
print('YES' if '${EXPECTED_STUDY}' in names else 'NO')
")

if [ "${STUDY_EXISTS}" != "YES" ]; then
  echo "❌ Study '${EXPECTED_STUDY}' no se creó en MySQL"
  echo "   Studies GBM existentes:"
  python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
for n in names:
    if 'gbm' in n.lower():
        print(f'   · {n}')
"
  abort "Study no encontrado"
fi
echo "✅ Study '${EXPECTED_STUDY}' creado en MySQL"

N_COMPLETED=$(python3 -c "
import optuna
study = optuna.load_study(study_name='${EXPECTED_STUDY}', storage='${OPTUNA_STORAGE}')
n = sum(1 for t in study.trials if t.state.name == 'COMPLETE')
print(n)
")
echo "  ${N_COMPLETED} trials COMPLETE / ${OPTUNA_TRIALS} target"

# ─── 5. extract_best_per_side ──────────────────────────────────────
log_section "5. extract_best_per_side"

mkdir -p ${REPORTS_DIR}

python3 -m mimo.oof.extract_best_per_side \
  --release ${RELEASE} \
  --study-prefix oof_study_gbm \
  --out-json ${BEST_JSON}

# ─── 6. Verificar best_per_side.json ───────────────────────────────
log_section "6. Verificar best_per_side.json"

[ -f "${BEST_JSON}" ] || abort "No se generó ${BEST_JSON}"

python3 <<EOF
import json, sys
data = json.load(open("${BEST_JSON}"))
print("✅ best_per_side.json válido. Top trial por side:")
for side in ("top_long", "top_short"):
    if side not in data or not data[side]:
        print(f"  ⚠️  '{side}' vacío")
        continue
    top = data[side][0]
    ev_key = f"ev_{side.replace('top_', '')}"
    ev = top.get(ev_key, {})
    print(f"  ── {side.upper()} (trial #{top.get('trial', '?')}):")
    print(f"     ev_net:    {ev.get('ev_net', 'nan'):+.4f}R")
    print(f"     n_signals: {int(ev.get('n_signals', 0))}")
    print(f"     thr:       {ev.get('thr', 'nan'):.4f}")
    print(f"     prec_TP:   {ev.get('prec_TP', 'nan'):.3f}")
    print(f"     mdd_R:     {ev.get('mdd_R', 'nan'):.1f}R")
    print(f"     n_params:  {len(top.get('params', {}))}")
EOF

# ─── 6. Resumen final ──────────────────────────────────────────────
log_section "FASE 1 GBM COMPLETADA"
echo "  best_per_side.json:  ${BEST_JSON}"
echo "  Optuna study:        ${EXPECTED_STUDY}"
echo "  Trials completados:  ${N_COMPLETED} / ${OPTUNA_TRIALS}"
echo "  Output dir:          artifacts/${RELEASE}/oof/${TAG}/"
echo ""
echo "📋 Próximo paso: comparar contra el CNN (oof_study_<release>_multitask)"
echo "   y decidir qué modelo va a producción (o si mantener ambos en ensemble)."
