#!/usr/bin/env bash
#
# 001_optuna_search.sh — Fase 1: Búsqueda de hyperparams con Optuna OOF.
#
# Lanza main_oof_regime_weights_v7 con N trials de Optuna sobre el espacio
# definido en GRID_BY_RELEASE (o heredado vía --inherit-config-from).
# Después extrae los top trials por side a un best_per_side.json.
#
# CONSEJOS:
#   - Para BÚSQUEDA REAL: --optuna-trials 30-100 (tarda 3-10h con 1 GPU)
#   - Para SMOKE TEST del pipeline: --optuna-trials 1 (no es búsqueda, solo
#     verifica que la cadena 001→002→...→006 funciona end-to-end)
#
# OVERRIDE DE GRID HP (sin tocar release real):
#   CNN_LSTM_GRID=202500_v4 bash 001_hyperparams_tuning.sh
#     → usa el grid HP de 202500_v4 (espacio ampliado: continuous lr/dropouts,
#       loss_weight_* libres desde 0.0, HPs estructurales nuevos como
#       activation, kernel_size_short/long, gru_units, attn_num_heads).
#     → mantiene release=202500 para barriers/feature_masks/artifact paths.
#   IMPORTANTE: usar STUDY_NAME distinto al de v3 (espacios incompatibles).
#     Ej. export OPTUNA_STUDY_OVERRIDE=oof_study_202500_v4_multitask antes
#     de relanzar (o borrar el existente con --reset-study si tu workflow lo
#     soporta).
#
# Si solo quieres reusar best_per_side.json de otra release:
#   cp artifacts/<source_release>/oof/<TAG>/reports/best_per_side.json \
#      artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json
#   # y salta este script

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202500}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202500}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}

# Trials de Optuna — overridable. Default 40 = búsqueda productiva con TPE
# + HyperbandPruner. Hyperband corta trials malos tras fold 1-2, así que
# el coste medio por trial es menor que el teórico (n_splits folds × epochs).
# Para smoke test del pipeline: OPTUNA_TRIALS=1 bash 001_hyperparams_tuning.sh
# Para búsqueda exhaustiva sobre 202600 (espacio ampliado): 80-120.
export OPTUNA_TRIALS=${OPTUNA_TRIALS:-40}

# Storage MySQL (debe coincidir con extract_best_per_side default)
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
log_section "FASE 1 — Optuna search"
echo "  Release:           ${RELEASE}"
echo "  Inherit from:      ${INHERIT_FROM_RELEASE}"
echo "  Tag (exp):         ${TAG}"
echo "  Seed:              ${SEED}"
echo "  Train period:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Holdout period:    ${HOLDOUT_FROM} → ${HOLDOUT_TO}"
echo "  Optuna trials:     ${OPTUNA_TRIALS}"
if [ -n "${CNN_LSTM_GRID:-}" ]; then
  echo "  Grid override:     CNN_LSTM_GRID=${CNN_LSTM_GRID}"
  echo "                     (release real sigue siendo ${RELEASE} para barriers/features)"
fi
if [ "${OPTUNA_TRIALS}" -lt 10 ]; then
  echo ""
  echo "  ⚠️  TRIALS BAJO (${OPTUNA_TRIALS}). Esto NO es búsqueda real."
  echo "     - Para smoke test del pipeline: OK"
  echo "     - Para Fase 1 productiva: usa OPTUNA_TRIALS=40 (~4h con Hyperband)"
  echo "       o OPTUNA_TRIALS=100 (~10h) para espacio ampliado 202600."
fi

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

# Verificar conexión a BD y disponibilidad de datos
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

# Verificar conexión a MySQL Optuna
python3 -c "
import optuna
try:
    optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
    print('✅ MySQL Optuna alcanzable')
except Exception as e:
    raise SystemExit(f'❌ No puedo conectar a Optuna storage: {e}')
" || abort "Fallo conexión a Optuna MySQL"

echo "✅ Pre-checks OK"

# ─── 2. Lanzar Optuna ──────────────────────────────────────────────
log_section "2. main_oof_regime_weights_v7 (${OPTUNA_TRIALS} trials)"

python3 -m mimo.oof.main_oof_regime_weights_v7 \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --base-tf 5min --target-type multitask --side both \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
  --use-tpe \
  --optuna-trials ${OPTUNA_TRIALS} \
  --objective ev_net --cost-per-signal 0.05 \
  --ev-min-signals 100 --max-drawdown-R 30 \
  --ev-thr-lo 0.10 --ev-thr-hi 0.40 \
  --oof-epochs 120 --oof-patience 15

# ─── 3. Verificar que el study existe en MySQL ─────────────────────
log_section "3. Verificar study creado en Optuna"

EXPECTED_STUDY="oof_study_${RELEASE}_multitask"
STUDY_EXISTS=$(python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
print('YES' if '${EXPECTED_STUDY}' in names else 'NO')
")

if [ "${STUDY_EXISTS}" != "YES" ]; then
  echo "❌ Study '${EXPECTED_STUDY}' no se creó en MySQL"
  echo "   Studies que sí existen para esta release:"
  python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
for n in names:
    if '${RELEASE}' in n:
        print(f'   · {n}')
"
  abort "Study no encontrado"
fi
echo "✅ Study '${EXPECTED_STUDY}' creado en MySQL"

# Verificar trials completados
N_COMPLETED=$(python3 -c "
import optuna
study = optuna.load_study(study_name='${EXPECTED_STUDY}', storage='${OPTUNA_STORAGE}')
n = sum(1 for t in study.trials if t.state.name == 'COMPLETE')
print(n)
")
echo "  ${N_COMPLETED} trials COMPLETE de ${OPTUNA_TRIALS} solicitados"

if [ "${N_COMPLETED}" -lt "${OPTUNA_TRIALS}" ]; then
  echo "  ⚠️  Algunos trials no completaron (failed/pruned). Revisa los logs si la diferencia es grande."
fi

# ─── 4. extract_best_per_side ──────────────────────────────────────
log_section "4. extract_best_per_side"

mkdir -p ${REPORTS_DIR}

python3 -m mimo.oof.extract_best_per_side \
  --release ${RELEASE} \
  --study-prefix oof_study \
  --out-json ${BEST_JSON}

# ─── 5. Verificar best_per_side.json ───────────────────────────────
log_section "5. Verificar best_per_side.json"

[ -f "${BEST_JSON}" ] || abort "No se generó ${BEST_JSON}"

python3 <<EOF
import json
import sys

try:
    data = json.load(open("${BEST_JSON}"))
except Exception as e:
    print(f"❌ JSON inválido: {e}")
    sys.exit(1)

errors = []
for side in ("top_long", "top_short"):
    if side not in data or not data[side]:
        errors.append(f"falta '{side}' o está vacío")
        continue
    top = data[side][0]
    if "params" not in top or not top["params"]:
        errors.append(f"'{side}[0]' sin 'params'")
    ev_key = f"ev_{side.replace('top_', '')}"
    if ev_key not in top:
        errors.append(f"'{side}[0]' sin '{ev_key}'")
    else:
        ev_net = top[ev_key].get("ev_net", None)
        if ev_net is None:
            errors.append(f"'{side}[0].{ev_key}' sin ev_net")

if errors:
    print("❌ best_per_side.json tiene problemas:")
    for e in errors:
        print(f"   · {e}")
    sys.exit(1)

# Resumen
print("✅ best_per_side.json válido. Resumen del top trial por side:")
print()
for side in ("top_long", "top_short"):
    top = data[side][0]
    ev_key = f"ev_{side.replace('top_', '')}"
    ev = top.get(ev_key, {})
    print(f"  ── {side.upper()} (trial #{top.get('trial', '?')}):")
    print(f"     ev_net:    {ev.get('ev_net', 'nan'):+.4f}R")
    print(f"     n_signals: {int(ev.get('n_signals', 0))}")
    print(f"     thr:       {ev.get('thr', 'nan'):.4f}")
    print(f"     prec_TP:   {ev.get('prec_TP', 'nan'):.3f}")
    print(f"     mdd_R:     {ev.get('mdd_R', 'nan'):.1f}R")
    print(f"     params:")
    for k in sorted(top.get("params", {})):
        print(f"       {k:>22s}: {top['params'][k]}")
    print()
EOF

if [ $? -ne 0 ]; then abort "best_per_side.json malformado"; fi

# ─── 6. Resumen final ──────────────────────────────────────────────
log_section "FASE 1 COMPLETADA"
echo "  best_per_side.json:  ${BEST_JSON}"
echo "  Optuna study:        ${EXPECTED_STUDY}"
echo "  Trials completados:  ${N_COMPLETED} / ${OPTUNA_TRIALS}"
echo ""

if [ "${OPTUNA_TRIALS}" -lt 30 ]; then
  echo "  ⚠️  RECORDATORIO: ${OPTUNA_TRIALS} trials no es una búsqueda real."
  echo "     Los 'best' hyperparams son arbitrarios. Para Fase 1 real:"
  echo "        OPTUNA_TRIALS=40 bash 001_hyperparams_tuning.sh   # default"
  echo "        OPTUNA_TRIALS=100 bash 001_hyperparams_tuning.sh  # espacio ampliado"
  echo ""
fi

echo "📋 Siguiente paso: Fase 2 (train_specialists)"
echo "    bash 002_train_specialists.sh"