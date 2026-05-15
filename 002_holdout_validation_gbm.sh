#!/usr/bin/env bash
#
# 002_holdout_validation_gbm.sh — Fase 2 GBM: validación honesta en holdout.
#
# Reentrena los boosters ganadores (top_long, top_short) sobre TODO el train
# usando los hiperparámetros del mejor trial, y predice sobre holdout
# 2025-11 → 2026-04 (intacto durante tuning).
#
# Reporta:
#   - HONEST  : EV-net con thr fijo del OOF aplicado a holdout (= producción).
#   - OPTIMISTIC: EV-net con thr re-escaneado en holdout (info, no decisión).
#   - Erosión esperada vs OOF en %.
#
# USO:
#   bash 002_holdout_validation_gbm.sh
#
# Override de defaults:
#   RELEASE=202602_GBM TAG=... bash 002_holdout_validation_gbm.sh

set -euo pipefail

# ─── Configuración (mismos defaults que fase 1) ────────────────────
export RELEASE=${RELEASE:-202602_GBM}
export INHERIT_FROM_RELEASE=${INHERIT_FROM_RELEASE:-202601}
export TAG=${TAG:-rw_both_Lvol_boost_td_down_h3_Svol_boost_h3}
export SEED=${SEED:-47}
export TRAIN_FROM=${TRAIN_FROM:-2024-01-01}
export TRAIN_TO=${TRAIN_TO:-2025-10-30}
export HOLDOUT_FROM=${HOLDOUT_FROM:-2025-11-01}
export HOLDOUT_TO=${HOLDOUT_TO:-2026-04-10}
export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
BEST_JSON=${REPORTS_DIR}/best_per_side.json
OUT_JSON=${REPORTS_DIR}/holdout_report.json

log_section() {
  echo ""
  echo "═══════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "═══════════════════════════════════════════════════════════════"
}
abort() { echo "❌ $1"; exit 1; }

log_section "FASE 2 GBM — Holdout validation"
echo "  Release:           ${RELEASE}"
echo "  Tag:               ${TAG}"
echo "  Train period:      ${TRAIN_FROM} → ${TRAIN_TO}"
echo "  Holdout period:    ${HOLDOUT_FROM} → ${HOLDOUT_TO}"
echo "  best_per_side.json:${BEST_JSON}"

# ─── 1. Pre-checks ──────────────────────────────────────────────────
log_section "1. Pre-checks"

[ -f "${BEST_JSON}" ] || abort "${BEST_JSON} no existe. Corre fase 1 primero."

python3 -c "
import json
data = json.load(open('${BEST_JSON}'))
n_long = len(data.get('top_long', []))
n_short = len(data.get('top_short', []))
if n_long == 0 or n_short == 0:
    raise SystemExit(f'best_per_side.json sin trials: top_long={n_long} top_short={n_short}')
top_long = data['top_long'][0]
top_short = data['top_short'][0]
print(f'✅ best_per_side.json OK')
print(f'   LONG  trial #{top_long[\"trial\"]}: ev_net={top_long[\"ev_long\"][\"ev_net\"]:+.4f}R thr={top_long[\"ev_long\"][\"thr\"]:.4f}')
print(f'   SHORT trial #{top_short[\"trial\"]}: ev_net={top_short[\"ev_short\"][\"ev_net\"]:+.4f}R thr={top_short[\"ev_short\"][\"thr\"]:.4f}')
" || abort "best_per_side.json malformado"

EXPECTED_STUDY="oof_study_gbm_${RELEASE}_multitask"
python3 -c "
import optuna
names = optuna.get_all_study_names(storage='${OPTUNA_STORAGE}')
if '${EXPECTED_STUDY}' not in names:
    raise SystemExit(f'Study ${EXPECTED_STUDY} no existe en Optuna')
print(f'✅ Study {names[names.index(\"${EXPECTED_STUDY}\")]}: existe')
" || abort "Study Optuna no accesible"

# ─── 2. Lanzar holdout validation ──────────────────────────────────
log_section "2. Refit + Holdout prediction"

python3 -m mimo.oof.main_oof_gbm_holdout \
  --release ${RELEASE} \
  --inherit-config-from ${INHERIT_FROM_RELEASE} \
  --best-json "${BEST_JSON}" \
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

# ─── 3. Verificar reporte ──────────────────────────────────────────
log_section "3. Reporte"

[ -f "${OUT_JSON}" ] || abort "No se generó ${OUT_JSON}"

python3 <<EOF
import json
import math

r = json.load(open("${OUT_JSON}"))

def fnum(v, default=float("nan")):
    return default if v is None else float(v)

def fint(v, default=0):
    return default if v is None else int(v)

def _row(side_label, d, ref_oof=None, thr_field="thr"):
    h = d if not isinstance(d, dict) else d
    ev_h = fnum(h.get("ev_net"))
    n_sig = fint(h.get("n_signals"))
    thr = fnum(h.get(thr_field))
    if n_sig == 0 or math.isnan(ev_h):
        return f"  {side_label:5s}: SIN SEÑALES  (thr={thr:.4f} no produce >= min_signals)"
    prec = fnum(h.get("prec_TP"))
    mdd  = fnum(h.get("mdd_R"))
    eros = ""
    if ref_oof is not None:
        ev_o = fnum(ref_oof.get("ev_net"))
        if not math.isnan(ev_o) and ev_o != 0:
            eros = f"  erosión={(ev_h - ev_o) / abs(ev_o) * 100:+.0f}%"
    return f"  {side_label:5s}: thr={thr:.4f}  ev_net={ev_h:+.4f}R  sig={n_sig}  prec={prec:.3f}  mdd={mdd:.1f}R{eros}"

print("📋 RESUMEN HOLDOUT REPORT")
print(f"   Release:        {r['release']}")
months = fnum(r["holdout_honest"]["months"], 1)
print(f"   Holdout period: {r['holdout_period'][0]} → {r['holdout_period'][1]} (~{months:.1f} meses)")
print()

print("─── OOF (training-time) ──────────────────────────────────────────")
for side in ("long", "short"):
    o = r["oof"][side]
    print(f"  {side.upper():5s}: thr={fnum(o['thr']):.4f}  ev_net={fnum(o['ev_net']):+.4f}R  "
          f"sig={fint(o['n_signals'])}  prec={fnum(o['prec_TP']):.3f}  "
          f"sig_rate={fnum(o.get('sig_rate'), 0):.5f}")

modes = [
    ("HOLDOUT ABSOLUTE   (thr OOF directo, MAL si dist shift)",  "holdout_honest"),
    ("HOLDOUT TRAIN_QUANT (thr matchea sig_rate → PRODUCCIÓN)",  "holdout_train_quantile"),
    ("HOLDOUT HOLD_QUANT  (thr matchea sig_rate en holdout)",     "holdout_hold_quantile"),
    ("HOLDOUT OPTIMISTIC (thr re-escan en holdout, upper bound)", "holdout_optimistic"),
]
for title, key in modes:
    if key not in r: continue
    print(f"\n─── {title} ───")
    section = r[key]
    for side in ("long", "short"):
        ref = r["oof"][side] if key != "holdout_optimistic" else None
        thr_field = "thr"
        print(_row(side.upper(), section[side], ref_oof=ref, thr_field=thr_field))
    total = fnum(section.get("total_R"), 0)
    print(f"  💰 Total R: {total:+.2f}R  ({total/max(months,0.01):+.2f}R/mes)")
EOF

log_section "FASE 2 GBM COMPLETADA"
echo "  Reporte:  ${OUT_JSON}"
echo ""
echo "📋 INTERPRETACIÓN:"
echo "  · Si la EROSIÓN (honest vs OOF) está entre -20% y -50% → modelo robusto."
echo "  · Si la EROSIÓN < -70% o ev_net HOLDOUT negativo → overfit al OOF."
echo "  · Si HOLDOUT > OOF → puede ser ruido (holdout pequeño) o señal genuina."
echo "  · El thr OPTIMISTIC ≈ thr OOF indica que el modelo extrapola bien."
