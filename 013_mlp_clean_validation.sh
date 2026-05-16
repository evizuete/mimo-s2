#!/usr/bin/env bash
#
# 013_mlp_clean_validation.sh
# ═══════════════════════════════════════════════════════════════════════
# Validación rigurosa del MLP tuneado, eliminando dos posibles sesgos del
# resultado anterior (Sharpe combined 3.92):
#
#   1) Solape tuning-walkforward: val period del Optuna (2025-05 → 2025-07)
#      caía dentro del rango de test del walkforward (2025-01 → 2026-04).
#      Fix: re-tune con val=2024-09→2024-12 (anterior al walkforward).
#
#   2) Costo optimista: cost_per_signal=0.05R puede subestimar
#      spread+slippage real. Fix: walkforward con 3 niveles (0.05, 0.10, 0.15)
#      para sensibilidad. Cada nivel re-corre el walkforward completo
#      (el threshold scanner usa el cost para penalizar señales).
#
# TIEMPO TOTAL: ~7-8h en GPU. Lanzar en screen/tmux.
#
# OUTPUTS:
#   artifacts/202500/oof/tuning/best_params_mlp.json           ← tune limpio
#   …/reports/walkforward_report_cnn_mlp_clean_cost005.json    ← cost 0.05
#   …/reports/walkforward_report_cnn_mlp_clean_cost010.json    ← cost 0.10
#   …/reports/walkforward_report_cnn_mlp_clean_cost015.json    ← cost 0.15
#   …/reports/long_only_sim_cnn_mlp_clean_cost*.json           ← simulators
#   …/reports/mlp_clean_validation_summary.txt                 ← tabla comparativa

set -euo pipefail

export RELEASE=${RELEASE:-202500}
export SEED=${SEED:-47}
export TAG=${TAG:-deploy_2026_04_combined_specialists_seed47}
export OPTUNA_STORAGE=${OPTUNA_STORAGE:-mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db}

# Position sizing para traducción monetaria
INITIAL_CAPITAL=${INITIAL_CAPITAL:-10000}
RISK_PER_TRADE_PCT=${RISK_PER_TRADE_PCT:-1}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports
STUDY_NAME="oof_study_${RELEASE}_mlp_multitask"

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }

log_section "VALIDACIÓN RIGUROSA MLP — sin solape + stress costos"
echo "  Release:   ${RELEASE}"
echo "  Study:     ${STUDY_NAME}"
echo "  Outputs:   ${REPORTS_DIR}/walkforward_report_cnn_mlp_clean_*.json"
echo "  Tiempo:    ~7-8h estimado"

# ═══ Step 1: backup study actual y borrar para re-tuning limpio ═══════════

log_section "1/4 — Backup del study actual y reset"

python3 <<EOF
import optuna
storage = "${OPTUNA_STORAGE}"
study_name = "${STUDY_NAME}"
backup_name = f"{study_name}_OLD_solape"

names = optuna.get_all_study_names(storage=storage)

# Backup si no existe
if backup_name not in names:
    try:
        optuna.copy_study(
            from_study_name=study_name, from_storage=storage,
            to_storage=storage, to_study_name=backup_name,
        )
        print(f"✅ Study antiguo copiado a {backup_name}")
    except Exception as e:
        print(f"⚠️  No se pudo copiar (puede ser primera ejecución): {e}")
else:
    print(f"ℹ️  Backup {backup_name} ya existe, no se sobreescribe")

# Borrar el original para re-tune limpio
if study_name in names:
    optuna.delete_study(study_name=study_name, storage=storage)
    print(f"✅ Study {study_name} borrado para re-tune limpio")
else:
    print(f"ℹ️  Study {study_name} no existe, se creará en step 2")
EOF

# ═══ Step 2: Re-tuning con val sin solape ═════════════════════════════════

log_section "2/4 — Re-tuning con val 2024-09 → 2024-12 (sin solape)"

TRAIN_FROM=2024-01-01 TRAIN_TO=2024-09-01 \
VAL_FROM=2024-09-01   VAL_TO=2024-12-01 \
  bash 012_optuna_arch.sh mlp 30

# Verificar que generó best_params
BEST_PARAMS_JSON="artifacts/${RELEASE}/oof/tuning/best_params_mlp.json"
[ -f "${BEST_PARAMS_JSON}" ] || { echo "❌ Falta ${BEST_PARAMS_JSON}"; exit 1; }
echo ""
echo "✅ Best params nuevo:"
python3 -c "import json; d=json.load(open('${BEST_PARAMS_JSON}')); print(json.dumps(d, indent=2))"

# ═══ Step 3: 3 walkforwards con costs diferentes ══════════════════════════

log_section "3/4 — Walkforwards × 3 con cost_per_signal ∈ {0.05, 0.10, 0.15}"

for COST in 0.05 0.10 0.15; do
  TAG_COST="cost$(echo ${COST} | sed 's/\.//')"  # 0.05 → cost005
  WF_JSON="${REPORTS_DIR}/walkforward_report_cnn_mlp_clean_${TAG_COST}.json"
  WF_CSV="${REPORTS_DIR}/walkforward_windows_cnn_mlp_clean_${TAG_COST}.csv"

  echo ""
  echo "─── Walkforward con cost=${COST}R ────"
  COST_PER_SIGNAL=${COST} \
    CNN_STUDY=${STUDY_NAME} \
    ARCH=mlp \
    OUT_JSON="${WF_JSON}" \
    OUT_CSV="${WF_CSV}" \
    bash 010_walkforward_cnn.sh
done

# ═══ Step 4: Simulators × 3 ═══════════════════════════════════════════════

log_section "4/4 — Simulators × 3 + tabla comparativa final"

for COST in 0.05 0.10 0.15; do
  TAG_COST="cost$(echo ${COST} | sed 's/\.//')"
  WF_JSON="${REPORTS_DIR}/walkforward_report_cnn_mlp_clean_${TAG_COST}.json"
  SIM_JSON="${REPORTS_DIR}/long_only_sim_cnn_mlp_clean_${TAG_COST}.json"

  if [ ! -f "${WF_JSON}" ]; then
    echo "⚠️  ${WF_JSON} no existe, salto simulator"
    continue
  fi

  python3 -m mimo.oof.diag_long_only_simulator \
    --walkforward-json "${WF_JSON}" \
    --out-json "${SIM_JSON}" \
    --initial-capital "${INITIAL_CAPITAL}" \
    --risk-per-trade-pct "${RISK_PER_TRADE_PCT}" \
    || echo "⚠️  Simulator falló para cost=${COST}"
done

# ═══ Tabla comparativa final ══════════════════════════════════════════════

SUMMARY="${REPORTS_DIR}/mlp_clean_validation_summary.txt"

python3 <<EOF | tee "${SUMMARY}"
import json
from pathlib import Path

reports_dir = Path("${REPORTS_DIR}")
print("")
print("═" * 95)
print("  TABLA COMPARATIVA — MLP tuned (sin solape) vs sensibilidad a costos")
print("═" * 95)

def _fmt(v, fmt="{:+.3f}"):
    if v is None: return "  n/a"
    try:
        if str(v).lower() == "nan": return "  n/a"
        return fmt.format(float(v))
    except: return "  n/a"

results = []
for cost in ("005", "010", "015"):
    cost_label = cost[0] + "." + cost[1:]   # 005 → 0.05
    sim_path = reports_dir / f"long_only_sim_cnn_mlp_clean_cost{cost}.json"
    if not sim_path.exists():
        print(f"⚠️  Falta {sim_path.name}")
        continue
    try:
        r = json.load(open(sim_path))
        s = r["strategies"]
        money = r.get("money_equivalence", {})
        results.append({
            "cost":      cost_label,
            "long":      s.get("long_only", {}),
            "short":     s.get("short_only", {}),
            "combined":  s.get("combined",   {}),
            "money_long":     money.get("long_only", {}),
            "money_short":    money.get("short_only", {}),
            "money_combined": money.get("combined", {}),
        })
    except Exception as e:
        print(f"⚠️  Error leyendo {sim_path}: {e}")

if not results:
    print("Sin resultados que mostrar.")
else:
    for strat in ("long_only", "short_only", "combined"):
        print(f"\n  {strat.upper()}")
        print(f"  {'cost':>6} | {'R_total':>9} | {'Sharpe':>7} | {'Sortino':>8} | "
              f"{'Calmar':>7} | {'MaxDD':>7} | {'WinRate':>7} | {'PF':>5} | verdict")
        print("  " + "─" * 92)
        # Mapeo de strat externo → key en results
        key = {"long_only": "long", "short_only": "short", "combined": "combined"}[strat]
        for row in results:
            d = row[key] or {}
            verdict = ""  # simplified
            print(f"  {row['cost']:>6} | {_fmt(d.get('R_total'),'{:+9.2f}')} | "
                  f"{_fmt(d.get('sharpe'),'{:+7.3f}')} | {_fmt(d.get('sortino'),'{:+8.3f}')} | "
                  f"{_fmt(d.get('calmar'),'{:+7.3f}')} | {_fmt(d.get('max_dd'),'{:7.2f}')} | "
                  f"{_fmt(d.get('win_rate'),'{:7.3f}')} | {_fmt(d.get('profit_factor'),'{:5.1f}')}")

    # ─── Tabla EUR/% ───
    init_cap = ${INITIAL_CAPITAL}
    risk_pct = ${RISK_PER_TRADE_PCT}
    print(f"\n  EQUIVALENCIA MONETARIA  (capital inicial={init_cap:,.0f}€, riesgo/trade={risk_pct:.2f}% → 1R={init_cap*risk_pct/100:,.2f}€)")
    for strat_label, key_money in (("LONG_ONLY", "money_long"),
                                     ("SHORT_ONLY", "money_short"),
                                     ("COMBINED", "money_combined")):
        print(f"\n  {strat_label}")
        print(f"  {'cost':>6} | {'%  LIN':>8} | {'EUR LIN':>11} | {'Final LIN':>11} | "
              f"{'%  COMP':>8} | {'EUR COMP':>11} | {'Final COMP':>11}")
        print("  " + "─" * 80)
        for row in results:
            m = row.get(key_money, {})
            if not m:
                print(f"  {row['cost']:>6} | (sin datos)")
                continue
            lin = m.get("linear", {})
            comp = m.get("compound", {})
            ruined = comp.get("ruined", False)
            comp_pct = "RUINED" if ruined else (f"{comp.get('pct_total',0):+7.1f}%" if comp.get('pct_total') is not None else "n/a")
            comp_eur = "    n/a   " if ruined else (f"{comp.get('eur_total',0):+11,.0f}€" if comp.get('eur_total') is not None else "n/a")
            comp_fin = "    n/a   " if ruined else (f"{comp.get('final_capital',0):11,.0f}€" if comp.get('final_capital') is not None else "n/a")
            print(f"  {row['cost']:>6} | "
                  f"{lin.get('pct_total',0):+7.2f}% | {lin.get('eur_total',0):+10,.0f}€ | {lin.get('final_capital',0):10,.0f}€ | "
                  f"{comp_pct:>8} | {comp_eur:>11} | {comp_fin:>11}")

    # Comparativa contra resultado anterior (solape, cost=0.05)
    print("\n" + "─" * 95)
    print("  REFERENCIA: MLP tuned CON SOLAPE (cost=0.05):")
    print("    LONG:      R=+120.7  Sharpe=2.48  Calmar=7.42   MaxDD=13.0")
    print("    SHORT:     R=+116.4  Sharpe=4.18  Calmar=18.75  MaxDD=4.96")
    print("    COMBINED:  R=+237.0  Sharpe=3.92  Calmar=62.26  MaxDD=3.05")
    print("\n  REFERENCIA: GBM baseline (LONG only):")
    print(f"    LONG:      R=~+103   Sharpe~1.15  → ~{int(init_cap*risk_pct/100*103):,}€ LIN (+{103*risk_pct:.0f}%)")

print("\n" + "═" * 95)
EOF

log_section "VALIDACIÓN COMPLETADA"
echo ""
echo "  📋 Summary completo en: ${SUMMARY}"
echo "  📁 Reports en: ${REPORTS_DIR}/"
echo ""
echo "  Archivos generados:"
ls -1 "${REPORTS_DIR}/" | grep -E "mlp_clean|mlp_clean_validation" | sort | sed 's/^/    /'
