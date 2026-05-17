#!/usr/bin/env bash
#
# 012_optuna_arch.sh — Optuna tuning específico por arch alternativa.
#
# COMPLEMENTARIO a 010_walkforward_cnn.sh:
#   · 010 hereda hps del CNN-LSTM study (oof_study_<RELEASE>_multitask).
#     Eso sesga la comparación a favor del CNN-LSTM.
#   · 012 tunea cada arch sobre SU PROPIO espacio de hps con Optuna
#     antes de lanzar el walkforward, así la comparación es justa.
#
# COSTE estimado por arch (n-trials=30, epochs=15, 1 split):
#   mlp         ~1.5h
#   hybrid      ~3-5h
#   transformer ~2-3h
#   tcn         ~4-6h
#
# FLUJO recomendado (TCN — release 202500):
#   Tras observar asimetría LONG/SHORT en tcn_v3 (LONG +60R, SHORT +11R con
#   PWR 40%), v4 cambia la estrategia por defecto: tunear cada side por
#   separado. SIDE=both queda como opción legacy o para multi-objective.
#
#   1. SIDE=long  bash 012_optuna_arch.sh tcn 40   # ~5-6h
#   2. SIDE=short bash 012_optuna_arch.sh tcn 40   # ~5-6h
#   3. Walkforward 2-study combinando ambos:
#      CNN_STUDY_LONG=oof_study_202500_tcn_v4_long_only \
#      CNN_STUDY_SHORT=oof_study_202500_tcn_v4_short_only \
#      SPLIT_MODELS=1 SPLIT_ZERO_OTHER_LOSS=1 ARCH=tcn \
#      bash 010_walkforward_cnn.sh
#
# FLUJO clásico (otros archs o SIDE=both para TCN):
#   1. bash 012_optuna_arch.sh mlp           # tune (~1.5h)
#   2. CNN_STUDY=oof_study_202500_mlp_multitask \
#      ARCH=mlp bash 010_walkforward_cnn.sh  # walkforward (~1-2h)
#   3. Comparar long_only_sim_cnn_mlp.json vs el del GBM
#
# OBJETIVO Y MULTI-OBJ:
#   OBJECTIVE=auc_pr  (default) — max val AUC-PR, ranking puro, estable.
#   OBJECTIVE=ev_net  — max EV_net via threshold sweep en val, alineado con
#                       walkforward (en R, tp/sl/cost iguales al deploy).
#   MULTI_OBJECTIVE=1 (solo SIDE=both) — Pareto front (long, short) con NSGA-II.
#
# Si quieres tunear varios archs en cadena:
#   for a in mlp hybrid transformer tcn; do bash 012_optuna_arch.sh $a; done

set -euo pipefail

if [ $# -lt 1 ]; then
  echo "Uso: bash 012_optuna_arch.sh <arch> [n_trials]"
  echo "  archs: mlp | hybrid | transformer | tcn"
  echo "  n_trials default: 30"
  exit 1
fi

ARCH=$1
N_TRIALS=${2:-${N_TRIALS:-30}}

export RELEASE=${RELEASE:-202500}
export SEED=${SEED:-47}
# SIDE=both|long|short. Default both (multitask, comportamiento legacy).
# SIDE=long → single-side LONG (focal_alpha_short y loss_weight_short fijos
# a valores que apagan la head SHORT). SIDE=short → simétrico.
# Cuando SIDE!=both, el study default cambia a oof_study_..._<side>_only.
export SIDE=${SIDE:-both}

# Split train/val para tuning (12m train + 2m val por defecto)
# Distinto del rango de walkforward para no contaminar:
#   walkforward usa 2025-01-01 → 2026-04-10 con train 12m
#   tuning usa 2024-05-01 → 2025-07-01 (anterior al walk)
export TRAIN_FROM=${TRAIN_FROM:-2024-05-01}
export TRAIN_TO=${TRAIN_TO:-2025-05-01}
export VAL_FROM=${VAL_FROM:-2025-05-01}
export VAL_TO=${VAL_TO:-2025-07-01}

# Epochs por trial (acortados para velocidad — el tuning busca rankings, no convergencia perfecta)
export EPOCHS=${EPOCHS:-15}
export PATIENCE=${PATIENCE:-4}

export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

# Objective config (default ranking puro). Para TCN v4 es opcional alinear a EV.
export OBJECTIVE=${OBJECTIVE:-auc_pr}
export MULTI_OBJECTIVE=${MULTI_OBJECTIVE:-0}
export EV_TP_MULT=${EV_TP_MULT:-2.0}
export EV_SL_MULT=${EV_SL_MULT:-0.8}
export EV_COST=${EV_COST:-0.05}
export EV_MIN_SIGNALS=${EV_MIN_SIGNALS:-30}

# STUDY_NAME override permite separar espacios HP incompatibles (p.ej. tcn v3
# vs v4 donde se amplían y reducen rangos en distintos HPs). Default por arch:
#   · tcn  → sufijo v4 (espacio HP redefinido en main_oof_arch_tuning._suggest_hp).
#   · otros → sufijo canónico sin versión.
# Sub-sufijo según SIDE: _multitask para both, _<side>_only para single-side.
if [ "${ARCH}" = "tcn" ]; then
  ARCH_VER_SUFFIX="_v4"
else
  ARCH_VER_SUFFIX=""
fi
if [ "${SIDE}" = "both" ]; then
  STUDY_NAME=${STUDY_NAME:-"oof_study_${RELEASE}_${ARCH}${ARCH_VER_SUFFIX}_multitask"}
else
  STUDY_NAME=${STUDY_NAME:-"oof_study_${RELEASE}_${ARCH}${ARCH_VER_SUFFIX}_${SIDE}_only"}
fi

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }

log_section "OPTUNA TUNING — arch=${ARCH} side=${SIDE}"
echo "  Release:    ${RELEASE}"
echo "  Study:      ${STUDY_NAME}"
echo "  Side:       ${SIDE}"
echo "  N trials:   ${N_TRIALS}"
echo "  Objective:  ${OBJECTIVE}  (multi_obj=${MULTI_OBJECTIVE})"
if [ "${OBJECTIVE}" = "ev_net" ]; then
  echo "  EV cfg:     tp=${EV_TP_MULT}R sl=${EV_SL_MULT}R cost=${EV_COST}R min_signals=${EV_MIN_SIGNALS}"
fi
echo "  Train:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Val:        ${VAL_FROM} → ${VAL_TO}"
echo "  Epochs/pat: ${EPOCHS} / ${PATIENCE}"

# Aviso para TCN: la recomendación v4 es lanzar SIDE=long y SIDE=short
# por separado (la asimetría observada en v3 hace que el multitask sacrifique
# SHORT). SIDE=both queda como opción legacy o para uso con MULTI_OBJECTIVE=1.
if [ "${ARCH}" = "tcn" ] && [ "${SIDE}" = "both" ] && [ "${MULTI_OBJECTIVE}" != "1" ]; then
  echo ""
  echo "  ⚠️  TCN v4: con SIDE=both en single-obj, LONG y SHORT comparten cabeza"
  echo "      y se penalizan mutuamente. Considera:"
  echo "        SIDE=long  bash 012_optuna_arch.sh tcn ${N_TRIALS}"
  echo "        SIDE=short bash 012_optuna_arch.sh tcn ${N_TRIALS}"
  echo "      o si insistes en SIDE=both, MULTI_OBJECTIVE=1 (Pareto NSGA-II)."
fi

# Construir flags opcionales del objective
OBJ_FLAGS=(--objective "${OBJECTIVE}")
if [ "${OBJECTIVE}" = "ev_net" ]; then
  OBJ_FLAGS+=(--ev-tp-mult "${EV_TP_MULT}"
              --ev-sl-mult "${EV_SL_MULT}"
              --ev-cost "${EV_COST}"
              --ev-min-signals "${EV_MIN_SIGNALS}")
fi
if [ "${MULTI_OBJECTIVE}" = "1" ]; then
  OBJ_FLAGS+=(--multi-objective)
fi

python3 -m mimo.oof.main_oof_arch_tuning \
  --arch ${ARCH} \
  --release ${RELEASE} \
  --side ${SIDE} \
  --study-name "${STUDY_NAME}" \
  --n-trials ${N_TRIALS} \
  --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
  --val-from ${VAL_FROM} --val-to ${VAL_TO} \
  --epochs ${EPOCHS} --patience ${PATIENCE} \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  "${OBJ_FLAGS[@]}"

if [ "${SIDE}" = "both" ]; then
  SIDE_SUFFIX="multitask"
else
  SIDE_SUFFIX="${SIDE}_only"
fi

log_section "TUNING COMPLETADO — arch=${ARCH} side=${SIDE}"
echo ""
echo "📋 Best params en: artifacts/${RELEASE}/oof/tuning/best_params_${ARCH}_${SIDE_SUFFIX}.json"
echo ""
if [ "${SIDE}" = "both" ]; then
  echo "🚀 Ahora lanza el walkforward con la arch tuneada:"
  echo "   CNN_STUDY=${STUDY_NAME} ARCH=${ARCH} bash 010_walkforward_cnn.sh"
  if [ "${MULTI_OBJECTIVE}" = "1" ]; then
    echo ""
    echo "   ⚠️  Multi-objective: el JSON contiene 'pareto_front' con todos los"
    echo "       trials no dominados. Elige uno manualmente según el trade-off"
    echo "       LONG/SHORT que prefieras antes del walkforward."
  fi
else
  echo "🚀 Ahora lanza el walkforward 2-study combinando este side con el opuesto:"
  echo "   CNN_STUDY_LONG=oof_study_${RELEASE}_${ARCH}${ARCH_VER_SUFFIX}_long_only \\"
  echo "   CNN_STUDY_SHORT=oof_study_${RELEASE}_${ARCH}${ARCH_VER_SUFFIX}_short_only \\"
  echo "   SPLIT_MODELS=1 SPLIT_ZERO_OTHER_LOSS=1 ARCH=${ARCH} \\"
  echo "   bash 010_walkforward_cnn.sh"
fi
