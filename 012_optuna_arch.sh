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
# FLUJO recomendado:
#   1. bash 012_optuna_arch.sh mlp           # tune (~1.5h)
#   2. CNN_STUDY=oof_study_202500_mlp_multitask \
#      ARCH=mlp bash 010_walkforward_cnn.sh  # walkforward (~1-2h)
#   3. Comparar long_only_sim_cnn_mlp.json vs el del GBM
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

# STUDY_NAME override permite separar espacios HP incompatibles (p.ej. tcn v2
# vs v3 donde el rango de learning_rate cambia). Default = nombre canónico del
# arch, con sufijo según SIDE.
if [ "${SIDE}" = "both" ]; then
  STUDY_NAME=${STUDY_NAME:-"oof_study_${RELEASE}_${ARCH}_multitask"}
else
  STUDY_NAME=${STUDY_NAME:-"oof_study_${RELEASE}_${ARCH}_${SIDE}_only"}
fi

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }

log_section "OPTUNA TUNING — arch=${ARCH} side=${SIDE}"
echo "  Release:    ${RELEASE}"
echo "  Study:      ${STUDY_NAME}"
echo "  Side:       ${SIDE}"
echo "  N trials:   ${N_TRIALS}"
echo "  Train:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Val:        ${VAL_FROM} → ${VAL_TO}"
echo "  Epochs/pat: ${EPOCHS} / ${PATIENCE}"

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
  --seed ${SEED}

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
else
  echo "🚀 Ahora lanza el walkforward 2-study combinando este side con el opuesto:"
  echo "   CNN_STUDY_LONG=oof_study_${RELEASE}_${ARCH}_long_only \\"
  echo "   CNN_STUDY_SHORT=oof_study_${RELEASE}_${ARCH}_short_only \\"
  echo "   SPLIT_MODELS=1 SPLIT_ZERO_OTHER_LOSS=1 ARCH=${ARCH} \\"
  echo "   bash 010_walkforward_cnn.sh"
fi
