#!/usr/bin/env bash
# 004_validate_lockbox.sh — FASE 4: Validación final sobre LOCKBOX.
# Veredicto OK / MARGINAL / KO multi-criterio.
set -euo pipefail

export RELEASE=${RELEASE:-202600}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export LOCKBOX_FROM=${LOCKBOX_FROM:-2026-04-11}
export LOCKBOX_TO=${LOCKBOX_TO:-2026-05-10}
export WARMUP_FROM=${WARMUP_FROM:-2026-03-25}
export NEW_DEPLOY=${NEW_DEPLOY:-deploy_validation_combined_seed${SEED}}
export NEW_POLICY=${NEW_POLICY:-decision_policies_config_${RELEASE}_validation}
export BASELINE_DEPLOY=${BASELINE_DEPLOY:-deploy_2026_04_combined_specialists_seed47}
export BASELINE_POLICY=${BASELINE_POLICY:-decision_policies_config_202500_mar31}
export MIN_PNL_PCT=${MIN_PNL_PCT:-0}
export MAX_MDD_PCT=${MAX_MDD_PCT:-15.0}
export MAX_ECE=${MAX_ECE:-0.08}
export MAX_REL_DEG=${MAX_REL_DEG:-0.10}

ARTIFACTS=artifacts/${RELEASE}/oof
NEW_DEPLOY_DIR=${ARTIFACTS}/${NEW_DEPLOY}
TS=$(date +%Y%m%d_%H%M%S)
REPORT_DIR=reports/lockbox_validation
mkdir -p ${REPORT_DIR}

log() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }
abort() { echo "❌ $1"; exit 1; }
extract() { python3 -c "import json; s=json.load(open('$1')); print(f\"{float(s.get('$2',0) or 0):.4f}\")"; }

log "FASE 4 — Validación LOCKBOX"
echo "  Release: ${RELEASE}  Deploy: ${NEW_DEPLOY}"
echo "  LOCKBOX: ${LOCKBOX_FROM} → ${LOCKBOX_TO}"

[ -d "${NEW_DEPLOY_DIR}" ] || abort "Deploy no existe: ${NEW_DEPLOY_DIR}"
[ -f "config/${NEW_POLICY}.py" ] || abort "Policy no existe: config/${NEW_POLICY}.py"

HAS_BASELINE=0
if [ -d "${ARTIFACTS}/${BASELINE_DEPLOY}" ] || [ -d "artifacts/202500/oof/${BASELINE_DEPLOY}" ]; then
  [ -f "config/${BASELINE_POLICY}.py" ] && HAS_BASELINE=1
fi
[ "${HAS_BASELINE}" = "1" ] && echo "✅ Baseline disponible" || echo "⚠️  Sin baseline"

log "2. Replay NUEVO sobre LOCKBOX"
NEW_OUT=/tmp/replay_lockbox_new_${RELEASE}_${TS}
python3 scripts/replay_s2_202500.py --release ${RELEASE} \
  --deploy-subdir ${NEW_DEPLOY} --policy-config ${NEW_POLICY} \
  --warmup-from ${WARMUP_FROM} --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} --out ${NEW_OUT}

NEW_PNL=$(extract ${NEW_OUT}/summary.json pnl_pct)
NEW_MDD=$(extract ${NEW_OUT}/summary.json max_drawdown_pct)
NEW_TRADES=$(extract ${NEW_OUT}/summary.json n_trades)
echo "  NUEVO: PnL=${NEW_PNL}% MDD=${NEW_MDD}% Trades=${NEW_TRADES}"

log "3. Diagnóstico drift"
GHOST_LB=/tmp/ghost_lockbox_${RELEASE}_${TS}.parquet
python3 scripts/ghost_predict.py --release ${RELEASE} \
  --deploy-subdir ${NEW_DEPLOY} --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --include-tail --out ${GHOST_LB}

CAL_OUT=${REPORT_DIR}/lockbox_cal_${RELEASE}_${TS}
PNL_OUT=${REPORT_DIR}/lockbox_score_pnl_${RELEASE}_${TS}
python3 -m mimo.oof.shift_analyzer.analyze_calibration_mimo --input ${GHOST_LB} \
  --score-col oof_proba_cal --target-col signal --state-col state --time-col time \
  --period-col period --output-dir ${CAL_OUT}
python3 -m mimo.oof.shift_analyzer.analyze_score_to_pnl --input ${GHOST_LB} \
  --score-col oof_proba_cal --target-col signal --pnl-col R_multiple \
  --state-col state --time-col time --period-col period --output-dir ${PNL_OUT}

NEW_ECE=$(python3 -c "
import json
try:
    s=json.load(open('${CAL_OUT}/summary.json'))
    h=next((p for p in s.get('overall_periods',[]) if p.get('period')=='holdout'),{})
    print(f\"{float(h.get('ece',0)):.4f}\")
except: print('0.0')")
NEW_AUC=$(python3 -c "
import json
try:
    s=json.load(open('${CAL_OUT}/summary.json'))
    h=next((p for p in s.get('overall_periods',[]) if p.get('period')=='holdout'),{})
    print(f\"{float(h.get('auc_pr',0)):.4f}\")
except: print('0.0')")
NEW_POSR=$(python3 -c "
import json
try:
    s=json.load(open('${CAL_OUT}/summary.json'))
    h=next((p for p in s.get('overall_periods',[]) if p.get('period')=='holdout'),{})
    print(f\"{float(h.get('pos_rate',0)):.4f}\")
except: print('0.0')")
echo "  ECE=${NEW_ECE} AUC-PR=${NEW_AUC} pos_rate=${NEW_POSR}"

TOXIC=$(python3 -c "
import pandas as pd
try:
    df=pd.read_csv('${PNL_OUT}/score_to_pnl_by_period_state.csv')
    hd=df[df['period']=='holdout']
    if hd.empty: print('')
    else:
        agg=hd.groupby('state').agg(n=('n','sum'),total_pnl=('total_pnl','sum'))
        agg['ppt']=agg['total_pnl']/agg['n'].clip(lower=1)
        t=agg[(agg['n']>=30)&(agg['ppt']<-0.10)]
        print(','.join(t.index.tolist()) if not t.empty else '')
except: print('')")
[ -n "${TOXIC}" ] && echo "  ⚠️  Tóxicos: ${TOXIC}" || echo "  ✅ Sin tóxicos"

BL_PNL="N/A"; BL_MDD="N/A"
if [ "${HAS_BASELINE}" = "1" ]; then
  log "4. Replay BASELINE (control)"
  BL_REL=${RELEASE}
  [ -d "artifacts/202500/oof/${BASELINE_DEPLOY}" ] && BL_REL=202500
  BL_OUT=/tmp/replay_lockbox_baseline_${RELEASE}_${TS}
  python3 scripts/replay_s2_202500.py --release ${BL_REL} \
    --deploy-subdir ${BASELINE_DEPLOY} --policy-config ${BASELINE_POLICY} \
    --warmup-from ${WARMUP_FROM} --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} --out ${BL_OUT}
  BL_PNL=$(extract ${BL_OUT}/summary.json pnl_pct)
  BL_MDD=$(extract ${BL_OUT}/summary.json max_drawdown_pct)
  echo "  BASELINE: PnL=${BL_PNL}% MDD=${BL_MDD}%"
fi

log "5. VEREDICTO"
V=$(python3 -c "
new_pnl=${NEW_PNL}; new_mdd=abs(${NEW_MDD}); new_ece=${NEW_ECE}
new_auc=${NEW_AUC}; new_posr=${NEW_POSR}
bl_pnl=${BL_PNL} if '${HAS_BASELINE}'=='1' else None
mp=${MIN_PNL_PCT}; mm=${MAX_MDD_PCT}; me=${MAX_ECE}; md=${MAX_REL_DEG}
tox='${TOXIC}'.split(',') if '${TOXIC}' else []

ok,wa,ko=[],[],[]
if new_pnl>=mp+5: ok.append(f'PnL OK')
elif new_pnl>=mp: wa.append(f'PnL en límite')
else: ko.append(f'PnL<{mp}')
if new_mdd<mm*0.7: ok.append(f'MDD OK')
elif new_mdd<=mm: wa.append(f'MDD cerca límite')
else: ko.append(f'MDD>{mm}')
if new_ece<me*0.6: ok.append(f'ECE muy bajo')
elif new_ece<=me: wa.append(f'ECE aceptable')
else: ko.append(f'ECE>{me}')
if new_posr>0:
    if new_auc>=new_posr*1.05: ok.append(f'AUC-PR OK')
    else: wa.append(f'AUC-PR<floor')
if tox and tox!=['']:
    if len(tox)>=3: ko.append(f'{len(tox)} tóxicos')
    elif len(tox)==2: wa.append(f'2 tóxicos')
    else: wa.append(f'1 tóxico')
if bl_pnl is not None and bl_pnl>0:
    rd=(bl_pnl-new_pnl)/bl_pnl
    if rd>md: ko.append(f'Degrada vs baseline')
    elif rd>0: wa.append(f'Ligera degradación')
    else: ok.append(f'Mejora baseline')

v='🔴 KO' if ko or len(wa)>=3 else ('🟡 MARGINAL' if wa else '✅ OK')
print(f'V={v}')
print(f'O={len(ok)}'); print(f'W={len(wa)}'); print(f'K={len(ko)}')
for r in ok: print(f'OK::{r}')
for r in wa: print(f'WA::{r}')
for r in ko: print(f'KO::{r}')")

VL=$(echo "$V"|grep '^V='|cut -d= -f2)
NO=$(echo "$V"|grep '^O='|cut -d= -f2)
NW=$(echo "$V"|grep '^W='|cut -d= -f2)
NK=$(echo "$V"|grep '^K='|cut -d= -f2)
echo "  📊 VEREDICTO: ${VL}  (OK=${NO} WARN=${NW} KO=${NK})"
echo "$V"|grep '^OK::'|sed 's/OK::/    ✅ /'
echo "$V"|grep '^WA::'|sed 's/WA::/    🟡 /'
echo "$V"|grep '^KO::'|sed 's/KO::/    🔴 /'

case "${VL}" in
  *OK*) echo "✅ Procede a Fase 5: bash scripts/005_production_refit.sh" ;;
  *MARGINAL*) echo "🟡 Documenta warnings o itera" ;;
  *KO*) echo "🔴 NO PROMOVER. Diagnóstico." ;;
esac

REPORT=${REPORT_DIR}/validation_${RELEASE}_${TS}.json
python3 -c "
import json
r={'timestamp':'${TS}','release':'${RELEASE}',
   'lockbox':['${LOCKBOX_FROM}','${LOCKBOX_TO}'],
   'verdict':'${VL}','metrics':{'new':{'pnl_pct':${NEW_PNL},'mdd_pct':${NEW_MDD},
   'n_trades':int(${NEW_TRADES}),'ece':${NEW_ECE},'auc_pr':${NEW_AUC},'pos_rate':${NEW_POSR}}},
   'toxic_states':'${TOXIC}'.split(',') if '${TOXIC}' else []}
open('${REPORT}','w').write(json.dumps(r,indent=2,default=str))
print(f'📁 ${REPORT}')"
