#!/usr/bin/env bash
#
# 002_train_specialists.sh — Fase 2: Entrena specialists por side con locked params.
#
# Lanza train_specialist (wrapper de main_oof_regime_weights_v7) para LONG y SHORT
# usando los hyperparams del best_per_side.json producido por Fase 1.
#
# PRE-REQUISITO:
#   - best_per_side.json debe existir en artifacts/${RELEASE}/oof/${TAG}/reports/
#   - Si no existe, ejecuta 001_optuna_search.sh primero (o copia de otra release)

set -euo pipefail

# ─── Configuración ─────────────────────────────────────────────────
export RELEASE=${RELEASE:-202601}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202500}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
# ARCH: arquitectura del modelo a entrenar. Por defecto 'original_v3' (CNN-LSTM
# nativo, compat hacia atrás con todos los runs previos). Cambiar a 'tcn' para
# entrenar specialists TCN con HPs cargados desde best_per_side.json. El JSON
# debe contener los HPs específicos del arch (best trial de un study TCN, p.ej.
# oof_study_202500_tcn_long_only_v4 Trial 38).
export ARCH=${ARCH:-original_v3}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}

ARTIFACTS=artifacts/${RELEASE}/oof
BEST_JSON=${ARTIFACTS}/${TAG}/reports/best_per_side.json

# ─── Helpers ────────────────────────────────────────────────────────
log_section() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
}
abort() { echo "❌ $1"; exit 1; }

# ─── Banner ─────────────────────────────────────────────────────────
log_section "FASE 2 — Train specialists"
echo "  Release:           ${RELEASE}"
echo "  Inherit from:      ${INHERIT_FROM_RELEASE}"
echo "  Tag:               ${TAG}"
echo "  Seed:              ${SEED}"
echo "  Arch:              ${ARCH}"
echo "  best_per_side:     ${BEST_JSON}"
echo "  Train period:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Holdout period:    ${HOLDOUT_FROM} → ${HOLDOUT_TO}"

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

# Existe best_per_side.json
[ -f "${BEST_JSON}" ] || abort "Falta ${BEST_JSON}. Ejecuta 001 antes o copia de otra release."

# Es JSON válido con top_long y top_short
python3 <<EOF
import json
import sys
try:
    data = json.load(open("${BEST_JSON}"))
except Exception as e:
    print(f"❌ JSON inválido: {e}")
    sys.exit(1)

for side in ("top_long", "top_short"):
    if side not in data or not data[side]:
        print(f"❌ '{side}' falta o vacío en ${BEST_JSON}")
        sys.exit(1)
    if "params" not in data[side][0]:
        print(f"❌ '{side}[0].params' falta")
        sys.exit(1)

print(f"✅ best_per_side.json válido")
print(f"   top_long  trial: #{data['top_long'][0].get('trial', '?')}, "
      f"ev_net: {data['top_long'][0].get('ev_long', {}).get('ev_net', 'nan'):+.4f}R")
print(f"   top_short trial: #{data['top_short'][0].get('trial', '?')}, "
      f"ev_net: {data['top_short'][0].get('ev_short', {}).get('ev_net', 'nan'):+.4f}R")
EOF

# Pre-verificar disponibilidad de datos
python3 -c "
from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
db = Database()
dm = DataManager.from_database_historical_2(
    db, from_date='${TRAIN_FROM}', to_date='${HOLDOUT_TO}')
print(f'✅ Datos BD: {len(dm.df):,} filas')
" || abort "Fallo conexión a BD"

echo "✅ Pre-checks OK"

# ─── 2. Archivar dirs previos (si existen) ─────────────────────────
log_section "2. Archivar specialist dirs previos"

ARCHIVED=0
for SIDE in long short; do
  OLD=${ARTIFACTS}/${TAG}_${SIDE}_specialist_seed${SEED}
  if [ -d "$OLD" ]; then
    NEW_NAME="${OLD}_$(date +%Y%m%d_%H%M%S)"
    mv "$OLD" "${NEW_NAME}"
    echo "  📦 Archivado: $(basename ${NEW_NAME})"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done

if [ "${ARCHIVED}" = "0" ]; then
  echo "  ℹ️  Sin specialist dirs previos a archivar"
fi

# ─── 3. Lanzar train_specialist ────────────────────────────────────
log_section "3. train_specialist (both sides)"

python3 -m mimo.oof.train_specialist \
  --best-per-side-json ${BEST_JSON} \
  --side both \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --base-tf 5min \
  --target-type multitask \
  --arch ${ARCH} \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
  --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30 \
  --oof-epochs 120 --oof-patience 15 \
  --seed ${SEED}

# ─── 4. Verificar outputs ──────────────────────────────────────────
log_section "4. Verificar artefactos generados"

ERRORS=0
for SIDE in long short; do
  SD=${ARTIFACTS}/${TAG}_${SIDE}_specialist_seed${SEED}
  echo ""
  echo "── ${SIDE} ──"
  
  # Modelo
  MODEL=${SD}/model_${RELEASE}_multitask.keras
  if [ -f "${MODEL}" ]; then
    SIZE=$(du -h "${MODEL}" | cut -f1)
    echo "  ✅ model:     ${MODEL} (${SIZE})"
  else
    echo "  ❌ FALTA model: ${MODEL}"
    ERRORS=$((ERRORS + 1))
  fi
  
  # Calibrator
  CAL=${SD}/oof_calibrator_${RELEASE}_multitask.joblib
  if [ -f "${CAL}" ]; then
    echo "  ✅ calibrator: $(basename ${CAL})"
  else
    echo "  ❌ FALTA calibrator: ${CAL}"
    ERRORS=$((ERRORS + 1))
  fi
  
  # Holdout predictions
  HP=${SD}/data/holdout_predictions_${RELEASE}_${SIDE}.parquet
  if [ -f "${HP}" ]; then
    N_ROWS=$(python3 -c "
import pandas as pd
print(len(pd.read_parquet('${HP}')))
" 2>/dev/null || echo "?")
    echo "  ✅ holdout_predictions: ${HP} (${N_ROWS} filas)"
  else
    echo "  ❌ FALTA holdout_predictions: ${HP}"
    ERRORS=$((ERRORS + 1))
  fi
  
  # Scalers
  SCALERS=${SD}/scalers_${RELEASE}
  if [ -d "${SCALERS}" ]; then
    echo "  ✅ scalers:    ${SCALERS}/"
  else
    echo "  ❌ FALTA scalers: ${SCALERS}"
    ERRORS=$((ERRORS + 1))
  fi
done

if [ "${ERRORS}" -gt 0 ]; then
  abort "${ERRORS} artefactos faltantes. Revisa logs de train_specialist."
fi

# ─── 5. Métricas rápidas del holdout ───────────────────────────────
log_section "5. Métricas rápidas del holdout (OOF)"

python3 <<EOF
import pandas as pd

for side in ("long", "short"):
    hp = "${ARTIFACTS}/${TAG}_" + side + "_specialist_seed${SEED}/data/holdout_predictions_${RELEASE}_" + side + ".parquet"
    df = pd.read_parquet(hp)
    
    raw_col = "y_pred_raw" if "y_pred_raw" in df.columns else "oof_proba_raw"
    cal_col = "y_pred_cal" if "y_pred_cal" in df.columns else "oof_proba_cal"
    sig_col = "y_true" if "y_true" in df.columns else "signal"
    
    if cal_col in df.columns and sig_col in df.columns:
        pos_rate = df[sig_col].mean()
        score_p99 = df[cal_col].quantile(0.99)
        score_max = df[cal_col].max()
        n_unique = df[cal_col].nunique()
        
        print(f"  ── {side.upper()} (n={len(df):,}):")
        print(f"     pos_rate (base rate): {pos_rate:.4f}")
        print(f"     score_cal p99:        {score_p99:.4f}")
        print(f"     score_cal max:        {score_max:.4f}")
        print(f"     score_cal unique:     {n_unique}")
        
        if score_max <= 0.10:
            print(f"     ⚠️  score_max muy bajo — el modelo casi no discrimina")
        if n_unique < 10:
            print(f"     ⚠️  pocos valores únicos en cal — isotonic saturada")
EOF

# ─── 6. Resumen final ──────────────────────────────────────────────
log_section "FASE 2 COMPLETADA"
echo "  Specialist LONG:   ${ARTIFACTS}/${TAG}_long_specialist_seed${SEED}/"
echo "  Specialist SHORT:  ${ARTIFACTS}/${TAG}_short_specialist_seed${SEED}/"
echo ""
echo "📋 Siguiente paso: Fase 3 (resume_deploy_v6 + merge + thresholds + policy)"
echo "    bash 003_calibrate_and_policy.sh"