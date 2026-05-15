#!/usr/bin/env bash
#
# 011_walkforward_all_archs.sh
# ═══════════════════════════════════════════════════════════════════════
# Lanza walkforward CNN para las 5 arquitecturas en serie + comparativa
# final con diag_long_only_simulator.py.
#
# Genera 5 walkforward_report_cnn_<arch>.json + 5 long_only_sim_cnn_<arch>.json
# + tabla comparativa.
#
# COSTE TOTAL ESTIMADO: ~18-30h CPU/GPU.
# Lanzar en screen/tmux y dejar varias noches.
#
# SI quieres saltarte alguna arch, comenta su línea en ARCHS_TO_RUN.

set -euo pipefail

# Lista de archs a evaluar. Comenta las que no quieras correr.
ARCHS_TO_RUN=(
  "original_v3"     # ~8-15h — baseline CNN-LSTM jerárquico
  "mlp"             # ~1-2h  — MLP-tabular puro (hipótesis: el problema es tabular)
  "hybrid"          # ~5-8h  — Conv1D + GAP + MLP fuerte (sin LSTM)
  "tcn"             # ~7-10h — Temporal Conv Network (dilated convs)
  "transformer"     # ~3-5h  — Transformer encoder ligero
)

# Configuración compartida
export RELEASE=${RELEASE:-202500}
export CNN_STUDY=${CNN_STUDY:-oof_study_${RELEASE}_multitask}
export TAG=${TAG:-deploy_2026_04_combined_specialists_seed47}
export SEED=${SEED:-47}
export WALK_FROM=${WALK_FROM:-2025-01-01}
export WALK_TO=${WALK_TO:-2026-04-10}

ARTIFACT_DIR=artifacts/${RELEASE}/oof/${TAG}
REPORTS_DIR=${ARTIFACT_DIR}/reports

log_section() { echo ""; echo "═══════════════════════════════════════════════════════════════"; echo "  $1"; echo "═══════════════════════════════════════════════════════════════"; }

log_section "WALK-FORWARD ALL ARCHS"
echo "  Release:  ${RELEASE}"
echo "  Archs:    ${ARCHS_TO_RUN[@]}"
echo "  Total esperado: ~18-30h CPU/GPU"

# Loop sobre archs
for ARCH in "${ARCHS_TO_RUN[@]}"; do
  log_section "▶  Lanzando walkforward con arch=${ARCH}"
  ARCH=${ARCH} bash 010_walkforward_cnn.sh || {
    echo "❌ Walkforward fallido para arch=${ARCH}, sigo con el siguiente"
    continue
  }

  # Simulator post walkforward para tener Sharpe/Calmar/PWR
  _SUFFIX=""
  [ "${ARCH}" != "original_v3" ] && _SUFFIX="_${ARCH}"
  WF_JSON="${REPORTS_DIR}/walkforward_report_cnn${_SUFFIX}.json"
  SIM_JSON="${REPORTS_DIR}/long_only_sim_cnn${_SUFFIX}.json"

  if [ -f "${WF_JSON}" ]; then
    echo ""
    echo "🧮 Simulator post walkforward para arch=${ARCH}"
    python3 -m mimo.oof.diag_long_only_simulator \
      --walkforward-json "${WF_JSON}" \
      --out-json "${SIM_JSON}" || true
  fi
done

# Tabla comparativa final
log_section "TABLA COMPARATIVA ARCHS"

python3 <<EOF
import json
import os
from pathlib import Path

reports_dir = Path("${REPORTS_DIR}")
archs = ["${ARCHS_TO_RUN[0]}", "${ARCHS_TO_RUN[1]:-}", "${ARCHS_TO_RUN[2]:-}", "${ARCHS_TO_RUN[3]:-}", "${ARCHS_TO_RUN[4]:-}"]
archs = [a for a in archs if a]

def _fnum(v, d=float('nan')):
    if v is None: return d
    try: return float(v)
    except: return d

rows = []
for arch in archs:
    suffix = "" if arch == "original_v3" else f"_{arch}"
    sim_path = reports_dir / f"long_only_sim_cnn{suffix}.json"
    if not sim_path.exists():
        print(f"⚠️  {sim_path.name} no existe, saltando arch={arch}")
        continue
    try:
        r = json.load(open(sim_path))
        long_s = r["strategies"]["long_only"]
        rows.append({
            "arch":         arch,
            "R_total":      _fnum(long_s.get("R_total")),
            "sharpe":       _fnum(long_s.get("sharpe")),
            "sortino":      _fnum(long_s.get("sortino")),
            "calmar":       _fnum(long_s.get("calmar")),
            "max_dd":       _fnum(long_s.get("max_dd")),
            "win_rate":     _fnum(long_s.get("win_rate")),
            "profit_factor":_fnum(long_s.get("profit_factor")),
            "verdict":      r.get("verdicts", {}).get("long_only", "?"),
        })
    except Exception as e:
        print(f"⚠️  Error leyendo {sim_path}: {e}")

if not rows:
    print("Sin reports válidos.")
else:
    rows.sort(key=lambda r: r["sharpe"] if r["sharpe"] is not None else -999, reverse=True)
    cols = ("arch", "R_total", "sharpe", "sortino", "calmar", "max_dd", "win_rate", "profit_factor")
    print(f"\n  TABLA COMPARATIVA — LONG-only por arch (rankeada por Sharpe):\n")
    print(f"  {'arch':<14} {'R_total':>10} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'MaxDD':>8} {'WinRate':>8} {'PF':>6}")
    print("  " + "─" * 80)
    for r in rows:
        print(f"  {r['arch']:<14} {r['R_total']:>+10.2f} {r['sharpe']:>+8.3f} {r['sortino']:>+8.3f} "
              f"{r['calmar']:>+8.3f} {r['max_dd']:>8.1f} {r['win_rate']:>8.3f} {r['profit_factor']:>6.2f}")
    print()
    for r in rows:
        print(f"  {r['arch']:<14} → {r['verdict']}")

    best = rows[0]
    print(f"\n🏆 BEST: {best['arch']} (Sharpe={best['sharpe']:+.3f}, R={best['R_total']:+.2f}R)")
EOF

log_section "ALL ARCHS DONE"
echo "Reports en: ${REPORTS_DIR}/"
echo ""
echo "📋 Para comparar también con GBM:"
echo "   Asumiendo que tienes long_only_sim.json del GBM en"
echo "   artifacts/202603_GBM/oof/<gbm_tag>/reports/, puedes hacer un diff manual"
echo "   entre los Sharpe/Calmar/R_total de cada modelo."
