#!/usr/bin/env bash
#
# 014_master_pipeline.sh — Pipeline maestro para evaluación rigurosa de
# arquitecturas alternativas + sensibilidad a costos.
# ═══════════════════════════════════════════════════════════════════════
#
# CONTEXTO
# ────────
# Hemos identificado dos sesgos en validaciones previas:
#   1) SOLAPE tuning↔walkforward: Optuna val (2025-05→2025-07) caía dentro
#      del rango walkforward (2025-01→2026-04) → inflaba Sharpe.
#   2) COSTO optimista: cost_per_signal=0.05R no contempla spread+slippage
#      adverso. Hay que validar sensibilidad.
#
# Este script ejecuta el pipeline correcto end-to-end:
#
# FASE 1 — Tuning Optuna SIN solape (val Sep-Dic 2024, anterior al walkforward)
#   Para cada arch en ARCHS_TO_TUNE: ~1.5h cada uno.
#   Output: oof_study_<R>_<arch>_multitask en MySQL + best_params_<arch>_multitask.json
#
# FASE 2 — Walkforward de cada arch con cost=0.05 (baseline cost)
#   Para cada arch: ~1.5h cada uno.
#   Output: walkforward_report_cnn_<arch>_cost005.json
#
# FASE 3 — Simulator de cada arch (Sharpe/Calmar/MaxDD)
#   Para cada arch: ~30s cada uno.
#   Output: long_only_sim_cnn_<arch>_cost005.json
#
# FASE 4 — Identificar ganador por Sharpe combined
#   Auto-pick del arch con mejor Sharpe combined.
#
# FASE 5 — Stress test costos del ganador (cost ∈ {0.10, 0.15})
#   2 walkforwards adicionales del ganador con costos elevados: ~3h.
#   Output: walkforward_report_cnn_<winner>_cost{010,015}.json
#
# FASE 6 — Tabla comparativa final
#   Output: artifacts/<R>/oof/<tag>/reports/master_pipeline_summary.txt
#
# TIEMPO TOTAL:
#   ARCHS_TO_TUNE=(mlp hybrid)        ~10h  ← default razonable
#   ARCHS_TO_TUNE=(mlp hybrid tcn)    ~15h
#   ARCHS_TO_TUNE=(mlp hybrid tcn transformer)  ~18h
#
# USO:
#   bash 014_master_pipeline.sh                      # default mlp + hybrid
#   ARCHS_TO_TUNE="mlp hybrid tcn" bash 014_master_pipeline.sh
#   FORCE_RETUNE=1 bash 014_master_pipeline.sh       # ignora studies existentes
#   SKIP_TUNE=1 bash 014_master_pipeline.sh          # usa studies existentes
#
# RECOMENDACIÓN: lanzar en screen/tmux para que sobreviva desconexión.

set -euo pipefail

# ═══ Configuración ════════════════════════════════════════════════════════

export RELEASE=${RELEASE:-202500}
export TAG=${TAG:-deploy_2026_04_combined_specialists_seed47}
export SEED=${SEED:-47}
export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

# Archs a evaluar (espacio-separados como string para parsing fácil)
ARCHS_TO_TUNE=${ARCHS_TO_TUNE:-"mlp hybrid"}

# Tuning sin solape (Sep-Dic 2024 → anterior al walkforward 2025-01→2026-04)
TUNE_TRAIN_FROM=${TUNE_TRAIN_FROM:-2024-01-01}
TUNE_TRAIN_TO=${TUNE_TRAIN_TO:-2024-09-01}
TUNE_VAL_FROM=${TUNE_VAL_FROM:-2024-09-01}
TUNE_VAL_TO=${TUNE_VAL_TO:-2024-12-01}
TUNE_TRIALS=${TUNE_TRIALS:-30}

# Niveles de cost para sensibilidad (el primero es el "baseline")
COSTS_BASELINE=0.05
COSTS_STRESS=(0.10 0.15)

# Position sizing para traducción monetaria
INITIAL_CAPITAL=${INITIAL_CAPITAL:-10000}    # EUR
RISK_PER_TRADE_PCT=${RISK_PER_TRADE_PCT:-1}  # % capital arriesgado por trade

# Flags de control
FORCE_RETUNE=${FORCE_RETUNE:-0}    # 1=borrar studies existentes y re-tune
SKIP_TUNE=${SKIP_TUNE:-0}          # 1=saltar tune, usar studies existentes

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
TUNING_DIR=artifacts/${RELEASE}/oof/tuning

mkdir -p "${REPORTS_DIR}" "${TUNING_DIR}"

log_section() {
  echo ""
  echo "════════════════════════════════════════════════════════════════════════════════"
  echo "  $1"
  echo "════════════════════════════════════════════════════════════════════════════════"
}

# Convert "0.05" → "005" para sufijos de archivo
_cost_tag() { echo "$1" | sed 's/\.//' ; }

log_section "MASTER PIPELINE — Evaluación rigurosa de arquitecturas"
echo "  Release:         ${RELEASE}"
echo "  Archs a evaluar: ${ARCHS_TO_TUNE}"
echo "  Tune val window: ${TUNE_VAL_FROM} → ${TUNE_VAL_TO} (sin solape con walkforward)"
echo "  Cost baseline:   ${COSTS_BASELINE}"
echo "  Cost stress:     ${COSTS_STRESS[@]}"
echo "  Skip tune:       ${SKIP_TUNE}"
echo "  Force re-tune:   ${FORCE_RETUNE}"

# ═══ FASE 1: Tuning por arch ══════════════════════════════════════════════

log_section "FASE 1/6 — Tuning Optuna sin solape"

for ARCH in ${ARCHS_TO_TUNE}; do
  STUDY_NAME="oof_study_${RELEASE}_${ARCH}_multitask"

  if [ "${SKIP_TUNE}" = "1" ]; then
    echo ""
    echo "⏭  Skip tune para arch=${ARCH} (SKIP_TUNE=1)"
    continue
  fi

  # Backup + delete si FORCE_RETUNE=1
  if [ "${FORCE_RETUNE}" = "1" ]; then
    echo ""
    echo "🗑  FORCE_RETUNE=1 → backup + reset study ${STUDY_NAME}"
    python3 <<EOF
import optuna
storage = "${OPTUNA_STORAGE}"
sname = "${STUDY_NAME}"
backup = f"{sname}_PRE_$(date +%Y%m%d_%H%M%S)"
try:
    names = optuna.get_all_study_names(storage=storage)
    if sname in names:
        optuna.copy_study(from_study_name=sname, from_storage=storage,
                          to_storage=storage, to_study_name=backup)
        optuna.delete_study(study_name=sname, storage=storage)
        print(f"✅ {sname} → backup {backup}, reset OK")
except Exception as e:
    print(f"⚠️  {e}")
EOF
  fi

  echo ""
  echo "▶  Tuning ${ARCH}..."
  TRAIN_FROM=${TUNE_TRAIN_FROM} TRAIN_TO=${TUNE_TRAIN_TO} \
  VAL_FROM=${TUNE_VAL_FROM} VAL_TO=${TUNE_VAL_TO} \
    bash 012_optuna_arch.sh ${ARCH} ${TUNE_TRIALS} \
    || { echo "❌ Tuning ${ARCH} falló, continuando con siguiente arch"; continue; }
done

# ═══ FASE 2: Walkforward baseline (cost=0.05) ═════════════════════════════

log_section "FASE 2/6 — Walkforward cost=${COSTS_BASELINE} para cada arch"

for ARCH in ${ARCHS_TO_TUNE}; do
  STUDY_NAME="oof_study_${RELEASE}_${ARCH}_multitask"
  COST_TAG=$(_cost_tag ${COSTS_BASELINE})
  WF_JSON="${REPORTS_DIR}/walkforward_report_cnn_${ARCH}_cost${COST_TAG}.json"
  WF_CSV="${REPORTS_DIR}/walkforward_windows_cnn_${ARCH}_cost${COST_TAG}.csv"

  echo ""
  echo "▶  Walkforward ${ARCH} cost=${COSTS_BASELINE}..."
  COST_PER_SIGNAL=${COSTS_BASELINE} \
  CNN_STUDY=${STUDY_NAME} \
  ARCH=${ARCH} \
  OUT_JSON="${WF_JSON}" \
  OUT_CSV="${WF_CSV}" \
    bash 010_walkforward_cnn.sh \
    || { echo "❌ Walkforward ${ARCH} falló"; continue; }
done

# ═══ FASE 3: Simulator para cada arch ═════════════════════════════════════

log_section "FASE 3/6 — Simulator por arch"

for ARCH in ${ARCHS_TO_TUNE}; do
  COST_TAG=$(_cost_tag ${COSTS_BASELINE})
  WF_JSON="${REPORTS_DIR}/walkforward_report_cnn_${ARCH}_cost${COST_TAG}.json"
  SIM_JSON="${REPORTS_DIR}/long_only_sim_cnn_${ARCH}_cost${COST_TAG}.json"

  if [ ! -f "${WF_JSON}" ]; then
    echo "⏭  ${WF_JSON} no existe, salto simulator"
    continue
  fi

  echo ""
  echo "▶  Simulator ${ARCH}..."
  python3 -m mimo.oof.diag_long_only_simulator \
    --walkforward-json "${WF_JSON}" \
    --out-json "${SIM_JSON}" \
    --initial-capital "${INITIAL_CAPITAL}" \
    --risk-per-trade-pct "${RISK_PER_TRADE_PCT}" \
    || echo "⚠️  Simulator falló para ${ARCH}"
done

# ═══ FASE 4: Identificar ganador por Sharpe combined ══════════════════════

log_section "FASE 4/6 — Identificar arquitectura ganadora"

WINNER=$(python3 <<EOF
import json
from pathlib import Path

reports = Path("${REPORTS_DIR}")
archs = "${ARCHS_TO_TUNE}".split()
cost_tag = "$(_cost_tag ${COSTS_BASELINE})"

results = []
for arch in archs:
    sim_path = reports / f"long_only_sim_cnn_{arch}_cost{cost_tag}.json"
    if not sim_path.exists():
        continue
    try:
        r = json.load(open(sim_path))
        combined = r["strategies"]["combined"]
        sharpe = combined.get("sharpe", -999) or -999
        r_total = combined.get("R_total", 0) or 0
        try:
            sharpe = float(sharpe)
        except: sharpe = -999
        results.append((arch, sharpe, r_total))
    except Exception as e:
        print(f"# warn: {arch} → {e}", file=__import__('sys').stderr)

results.sort(key=lambda r: r[1], reverse=True)
# Imprimir ranking a stderr para visibilidad
import sys
print("\n  Ranking por Sharpe combined:", file=sys.stderr)
for i, (a, s, r) in enumerate(results, 1):
    print(f"    {i}. {a:<14} Sharpe={s:+.3f}  R_total={r:+.2f}", file=sys.stderr)

if results:
    print(results[0][0])  # stdout = nombre del ganador
else:
    print("none")
EOF
)

if [ "${WINNER}" = "none" ] || [ -z "${WINNER}" ]; then
  echo "❌ No se pudo determinar ganador (ningún simulator generó output válido)"
  echo "   Saltando fase 5. Mira los simulator JSONs manualmente en ${REPORTS_DIR}"
  WINNER=""
else
  echo ""
  echo "🏆 GANADOR: ${WINNER}"
fi

# ═══ FASE 5: Stress test costos del ganador ═══════════════════════════════

if [ -n "${WINNER}" ]; then
  log_section "FASE 5/6 — Stress test costos para ganador (${WINNER})"

  STUDY_NAME="oof_study_${RELEASE}_${WINNER}_multitask"

  for COST in "${COSTS_STRESS[@]}"; do
    COST_TAG=$(_cost_tag ${COST})
    WF_JSON="${REPORTS_DIR}/walkforward_report_cnn_${WINNER}_cost${COST_TAG}.json"
    WF_CSV="${REPORTS_DIR}/walkforward_windows_cnn_${WINNER}_cost${COST_TAG}.csv"
    SIM_JSON="${REPORTS_DIR}/long_only_sim_cnn_${WINNER}_cost${COST_TAG}.json"

    echo ""
    echo "▶  Walkforward ${WINNER} cost=${COST}..."
    COST_PER_SIGNAL=${COST} \
    CNN_STUDY=${STUDY_NAME} \
    ARCH=${WINNER} \
    OUT_JSON="${WF_JSON}" \
    OUT_CSV="${WF_CSV}" \
      bash 010_walkforward_cnn.sh \
      || { echo "⚠️  Walkforward falló para cost=${COST}"; continue; }

    echo "▶  Simulator ${WINNER} cost=${COST}..."
    python3 -m mimo.oof.diag_long_only_simulator \
      --walkforward-json "${WF_JSON}" \
      --out-json "${SIM_JSON}" \
      || echo "⚠️  Simulator falló para cost=${COST}"
  done
fi

# ═══ FASE 6: Tabla comparativa final ══════════════════════════════════════

log_section "FASE 6/6 — Tabla comparativa final"

SUMMARY_FILE="${REPORTS_DIR}/master_pipeline_summary.txt"

python3 <<EOF | tee "${SUMMARY_FILE}"
import json
import math
from pathlib import Path

reports = Path("${REPORTS_DIR}")
archs = "${ARCHS_TO_TUNE}".split()
winner = "${WINNER}"
cost_baseline = "${COSTS_BASELINE}"
costs_stress = "${COSTS_STRESS[@]}".split()
all_costs = [cost_baseline] + costs_stress

def _fmt(v, fmt="{:+.3f}"):
    if v is None: return "  n/a"
    try:
        f = float(v)
        if math.isnan(f): return "  n/a"
        return fmt.format(f)
    except: return "  n/a"

def _load_sim(arch, cost):
    cost_tag = cost.replace(".", "")
    p = reports / f"long_only_sim_cnn_{arch}_cost{cost_tag}.json"
    if not p.exists(): return None
    try:
        return json.load(open(p))
    except: return None

print("")
print("═" * 105)
print("  MASTER PIPELINE — RESULTADO FINAL")
print("═" * 105)

# ─── Tabla 1: comparativa entre archs con cost baseline ───
print(f"\n  TABLA 1 — Comparativa entre arquitecturas (cost={cost_baseline}R)")
print(f"  {'─' * 100}")
print(f"  {'Arch':<14} {'Strat':<10} | {'R_total':>9} | {'Sharpe':>7} | "
      f"{'Calmar':>7} | {'MaxDD':>7} | {'WinRate':>7} | {'PF':>6}")
print(f"  {'─' * 100}")

for arch in archs:
    sim = _load_sim(arch, cost_baseline)
    if not sim:
        print(f"  {arch:<14} {'—':<10} | (sin datos)")
        continue
    for strat_key, strat_label in [("long_only", "LONG"),
                                     ("short_only", "SHORT"),
                                     ("combined", "COMBINED")]:
        d = sim["strategies"].get(strat_key, {})
        winner_mark = " 🏆" if arch == winner and strat_key == "combined" else ""
        print(f"  {arch:<14} {strat_label:<10} | "
              f"{_fmt(d.get('R_total'), '{:+9.2f}')} | "
              f"{_fmt(d.get('sharpe'), '{:+7.3f}')} | "
              f"{_fmt(d.get('calmar'), '{:+7.3f}')} | "
              f"{_fmt(d.get('max_dd'), '{:7.2f}')} | "
              f"{_fmt(d.get('win_rate'), '{:7.3f}')} | "
              f"{_fmt(d.get('profit_factor'), '{:6.1f}')}{winner_mark}")
    print()

# ─── Tabla 2: sensibilidad a cost del ganador ───
if winner:
    print(f"\n  TABLA 2 — Sensibilidad a costos para ganador ({winner})")
    print(f"  {'─' * 100}")
    print(f"  {'Cost':>6} {'Strat':<10} | {'R_total':>9} | {'Sharpe':>7} | "
          f"{'Calmar':>7} | {'MaxDD':>7} | {'WinRate':>7} | {'PF':>6}")
    print(f"  {'─' * 100}")

    for cost in all_costs:
        sim = _load_sim(winner, cost)
        if not sim:
            print(f"  {cost:>6} (sin datos)")
            continue
        for strat_key, strat_label in [("long_only", "LONG"),
                                         ("short_only", "SHORT"),
                                         ("combined", "COMBINED")]:
            d = sim["strategies"].get(strat_key, {})
            print(f"  {cost:>6} {strat_label:<10} | "
                  f"{_fmt(d.get('R_total'), '{:+9.2f}')} | "
                  f"{_fmt(d.get('sharpe'), '{:+7.3f}')} | "
                  f"{_fmt(d.get('calmar'), '{:+7.3f}')} | "
                  f"{_fmt(d.get('max_dd'), '{:7.2f}')} | "
                  f"{_fmt(d.get('win_rate'), '{:7.3f}')} | "
                  f"{_fmt(d.get('profit_factor'), '{:6.1f}')}")
        print()

# ─── Tabla 3: traducción monetaria ───
init_cap = ${INITIAL_CAPITAL}
risk_pct = ${RISK_PER_TRADE_PCT}
print(f"\n  TABLA 3 — Equivalencia monetaria  (capital inicial={init_cap:,.0f}€, riesgo/trade={risk_pct:.2f}% → 1R={init_cap*risk_pct/100:,.2f}€)")
print(f"  {'─' * 100}")
print(f"  {'Arch':<14} {'Cost':>5} {'Strat':<10} | {'%  LIN':>8} | {'EUR LIN':>11} | {'Final LIN':>11} | "
      f"{'% COMP':>7} | {'EUR COMP':>11} | {'Final COMP':>11}")
print(f"  {'─' * 100}")

# Para el ganador, mostrar 3 costs. Para los demás archs, sólo baseline.
table3_rows = []
if winner:
    for cost in all_costs:
        table3_rows.append((winner, cost))
for arch in archs:
    if arch != winner:
        table3_rows.append((arch, cost_baseline))

for arch, cost in table3_rows:
    sim = _load_sim(arch, cost)
    if not sim: continue
    money = sim.get("money_equivalence", {})
    for strat_key, strat_label in [("long_only", "LONG"),
                                     ("short_only", "SHORT"),
                                     ("combined", "COMBINED")]:
        m = money.get(strat_key, {})
        if not m:
            continue
        lin = m.get("linear", {})
        comp = m.get("compound", {})
        ruined = comp.get("ruined", False)
        comp_pct = "RUINED" if ruined else (f"{comp.get('pct_total', 0):+7.1f}%" if comp.get('pct_total') is not None else "  n/a")
        comp_eur = "  n/a" if ruined else (f"{comp.get('eur_total', 0):+11,.0f}€" if comp.get('eur_total') is not None else "  n/a")
        comp_fin = "  n/a" if ruined else (f"{comp.get('final_capital', 0):11,.0f}€" if comp.get('final_capital') is not None else "  n/a")
        print(f"  {arch:<14} {cost:>5} {strat_label:<10} | "
              f"{lin.get('pct_total', 0):+7.1f}% | "
              f"{lin.get('eur_total', 0):+10,.0f}€ | "
              f"{lin.get('final_capital', 0):10,.0f}€ | "
              f"{comp_pct:>7} | {comp_eur:>11} | {comp_fin:>11}")
    print()

# ─── Referencias ───
print(f"\n  REFERENCIA: GBM baseline LONG-only ~ R=+103R  Sharpe~1.15")
print(f"             → {init_cap*risk_pct/100*103:,.0f}€ LIN (+{103*risk_pct:.1f}%)")
print(f"  Nota:  LIN = position sizing fijo sobre capital INICIAL (sin compounding)")
print(f"        COMP = position sizing fijo sobre capital ACTUAL (con compounding)")
print("═" * 105)
EOF

log_section "MASTER PIPELINE COMPLETADO"
echo ""
echo "  📋 Summary final en: ${SUMMARY_FILE}"
echo "  📁 Reports en:       ${REPORTS_DIR}/"
echo "  📁 Tuning configs:   ${TUNING_DIR}/"
echo ""
echo "  Para inspección rápida del summary:"
echo "    cat ${SUMMARY_FILE}"
