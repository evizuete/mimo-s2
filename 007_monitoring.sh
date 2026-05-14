export RELEASE=202500
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

# Diaria — semana corta
python scripts/monitor_health.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_PROD \
  --from $(date -d "7 days ago" +%F) --to $(date +%F)

# Semanal — 30 días
python scripts/monitor_health.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_PROD

# Regenerar dashboard
python scripts/dashboard_health.py \
  --reports-dir reports/monitor \
  --out reports/dashboard.html
