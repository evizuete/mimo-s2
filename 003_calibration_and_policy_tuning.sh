export RELEASE=202600
export INHERIT_FROM_RELEASE=202500
export TAG=rw_both_Lvol_boost_td_down_h3_Svol_boost_h3
export SEED=47
export TRAIN_FROM=2024-01-01
export TRAIN_TO=2025-10-30
export HOLDOUT_FROM=2025-11-01
export HOLDOUT_TO=2026-04-10      # ← validación
export PROD_HOLDOUT_TO=2026-05-10  # ← refit final
export LOCKBOX_FROM=2026-04-11
export LOCKBOX_TO=2026-05-10
export TAIL_DAYS=21

BEST_JSON=artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json
if [ ! -f "${BEST_JSON}" ]; then
  echo "❌ Falta ${BEST_JSON}. Ejecuta scripts 001+002 antes."
  exit 1
fi

for SIDE in long short; do
  SPEC_DIR=artifacts/${RELEASE}/oof/${TAG}_${SIDE}_specialist_seed${SEED}
  if [ ! -d "${SPEC_DIR}" ]; then
    echo "❌ Falta specialist dir: ${SPEC_DIR}"
    exit 1
  fi
done
echo "✅ Pre-requisitos OK"

# Production deploy training x 2
for SIDE in long short; do
  python3 -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} \
    --inherit-config-from ${INHERIT_FROM_RELEASE} \
    --target-type multitask \
    --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} \
    --holdout-from ${HOLDOUT_FROM} \
    --holdout-to ${HOLDOUT_TO} \
    --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED} \
    --deploy-subdir deploy_validation_${SIDE}_specialist_seed${SEED} \
    --locked-params-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
    --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done

# Merge
python3 -m mimo.oof.merge_specialists \
  --release ${RELEASE} \
  --long-dir  artifacts/${RELEASE}/oof/deploy_validation_long_specialist_seed${SEED} \
  --short-dir artifacts/${RELEASE}/oof/deploy_validation_short_specialist_seed${SEED} \
  --out-dir   artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED}

# Exploring different calibrators
python3 scripts/ghost_predict.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_validation_combined_seed${SEED} \
  --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} \
  --include-tail \
  --out /tmp/ghost_holdout.parquet

python3 scripts/simulate_calibrators.py \
  --release ${RELEASE} \
  --specialist-tag ${TAG} \
  --seed ${SEED} \
  --test-parquet /tmp/ghost_holdout.parquet \
  --out-report reports/cal_sim.csv

# Selecting thresholds & policy
python3 -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

python3 -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED} \
  --emit-config-stub \
  --out-stub config/decision_policies_config_${RELEASE}_validation.py