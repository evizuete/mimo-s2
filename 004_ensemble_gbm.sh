#!/usr/bin/env bash
#
# 004_ensemble_gbm.sh — Fase 4 GBM: ensemble de top-N trials.
#
# Combina los N mejores trials por side (LONG y SHORT por separado) en
# un ensemble que reduce la varianza vs picar el "best único".
#
# Promedio de probs calibradas + thr mediana de los N trials → eval holdout.
#
# DEFAULTS:
#   TOP_N=5          (sweet spot empírico)
#   THR_STRATEGY=median  (más robusto que best_trial)
#   Holdout: 2025-11 → 2026-04 (mismo que fase 2)
#
# USO:
#   bash 004_ensemble_gbm.sh
#
# Variantes:
#   TOP_N=3 bash 004_ensemble_gbm.sh                  # ensemble más concentrado
#   THR_STRATEGY=rescan bash 004_ensemble_gbm.sh      # info: thr re-escaneado en holdout
#   THR_STRATEGY=best_trial bash 004_ensemble_gbm.sh  # usar thr del top-1

set -euo pipefail

export RELEASE=${RELEASE:-202602_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}

export TOP_N=${TOP_N:-5}
export THR_STRATEGY=${THR_STRATEGY:-median}

export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}

export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json
OUT_JSON=${REPORTS_DIR}/ensemble_report_top${TOP_N}_${THR_STRATEGY}.json

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }

log_section "FASE 4 GBM — Ensemble top-N"
echo "  Release:       ${RELEASE}"
echo "  Top-N:         ${TOP_N}"
echo "  thr strategy:  ${THR_STRATEGY}"
echo "  Holdout:       ${HOLDOUT_FROM} → ${HOLDOUT_TO}"

log_section "1. Pre-checks"
[ -f "${BEST_JSON}" ] || abort "${BEST_JSON} no existe"
python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
study = 'oof_study_gbm_${RELEASE}_multitask'
if study not in names: raise SystemExit(f'Study {study} no existe')
s = optuna.load_study(study_name=study, storage='${OPTUNA_STORAGE}')
n = sum(1 for t in s.trials if t.state.name == 'COMPLETE')
if n < ${TOP_N}: raise SystemExit(f'Solo {n} trials COMPLETE — TOP_N={${TOP_N}} insuficiente')
print(f'✅ {n} trials COMPLETE en {study}')
" || abort "Pre-checks fallaron"

log_section "2. Ensemble"

python3 -m mimo.oof.main_oof_gbm_ensemble \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --best-json "${BEST_JSON}" \
  --top-n ${TOP_N} \
  --thr-strategy ${THR_STRATEGY} \
  --base-tf 5min \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
  --cost-per-signal 0.05 \
  --ev-min-signals 30 \
  --max-drawdown-R 30 \
  --optuna-storage "${OPTUNA_STORAGE}" \
  --seed ${SEED} \
  --out-json "${OUT_JSON}"

log_section "3. Reporte"
[ -f "${OUT_JSON}" ] || abort "No se generó ${OUT_JSON}"

python3 <<EOF
import json
r = json.load(open("${OUT_JSON}"))

def fnum(v, d=float("nan")): return d if v is None else float(v)
def fint(v, d=0): return d if v is None else int(v)

print(f"📋 Ensemble report | top-{r['top_n']} | thr_strategy={r['thr_strategy']}")
print(f"   Trials LONG : {r['long_trials']}")
print(f"   Trials SHORT: {r['short_trials']}")
print()
print(f"─── Holdout {r['holdout_period'][0]} → {r['holdout_period'][1]} ───")
for side in ("long", "short"):
    h = r["holdout"][side]
    n = fint(h.get("n_signals"))
    if n == 0:
        print(f"  {side.upper():5s}: SIN SEÑALES")
        continue
    print(f"  {side.upper():5s}: ev_net={fnum(h.get('ev_net')):+.4f}R  sig={n}  "
          f"prec_TP={fnum(h.get('prec_TP')):.3f}  mdd={fnum(h.get('mdd_R')):.1f}R  "
          f"thr_used={fnum(h.get('thr')):.4f}")

total = fnum(r["holdout"]["total_R"], 0)
months = fnum(r["holdout"]["months"], 1)
print(f"\n  💰 R total ensemble: {total:+.2f}R en {months:.1f}m ({total/max(months,0.01):+.2f}R/mes)")
EOF

log_section "FASE 4 GBM COMPLETADA"
echo "  Reporte:  ${OUT_JSON}"
echo ""
echo "📋 INTERPRETACIÓN:"
echo "  · Si ensemble > best_único (fase 2) → reducción de varianza efectiva"
echo "  · Si ensemble ≈ best_único → los top-N son redundantes (TPE concentrado)"
echo "  · Si ensemble < best_único → algunos trials top-N son ruido; bajar TOP_N"
