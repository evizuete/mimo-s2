#!/usr/bin/env bash
#
# 000_daily_pnl_summary.sh
# ════════════════════════════════════════════════════════════════════
# REPORTE — PnL diario y agregado desde logs S3.
#
# Reconstruye el ciclo de vida de cada trade desde los logs estructurados de
# s3_service.py (s3_events_YYYYMMDD.jsonl), calcula PnL real (en puntos y
# USD) por trade y emite un reporte con stats agregadas.
#
# CUÁNDO USARLO
# ─────────────
#   · Diario (post-cierre NY): rápido check de cómo fue el día.
#   · Semanal: ver tendencia con --recent 7.
#   · Mensual: ver tendencia larga con --recent 30 o rango específico.
#   · Ad-hoc: comparar dos días concretos.
#
# USO
# ───
#   # Hoy (default)
#   bash 000_daily_pnl_summary.sh
#
#   # Día específico
#   DATE=20260521 bash 000_daily_pnl_summary.sh
#
#   # Últimos 7 días (con tabla detallada por día)
#   RECENT=7 VERBOSE=1 bash 000_daily_pnl_summary.sh
#
#   # Últimos 30 días (solo resumen agregado)
#   RECENT=30 bash 000_daily_pnl_summary.sh
#
#   # Rango específico
#   FROM=20260520 TO=20260522 bash 000_daily_pnl_summary.sh
#
#   # Logs en otra ubicación
#   LOGS_DIR=/otra/ruta bash 000_daily_pnl_summary.sh
#
# VARIABLES DE ENTORNO
# ────────────────────
#   DATE          (default: hoy)         — día único YYYYMMDD
#   FROM, TO      (default: -)           — rango de fechas YYYYMMDD
#   RECENT        (default: -)           — últimos N días con log
#   VERBOSE       (default: 0)           — 1 para mostrar tabla detallada
#   LOGS_DIR      (default: ver abajo)   — directorio con s3_events_*.jsonl
#   PYBIN         (default: boti python) — interprete Python
#
# OUTPUT
# ──────
#   Stdout. Formato legible con emojis (🟢/🔴) y barras ASCII.
# ════════════════════════════════════════════════════════════════════

set -euo pipefail

# ─── Configuración por defecto ─────────────────────────────────────
export DATE=${DATE:-}
export FROM=${FROM:-}
export TO=${TO:-}
export RECENT=${RECENT:-}
export VERBOSE=${VERBOSE:-0}
export LOGS_DIR=${LOGS_DIR:-/mnt/c/Users/Usuario/Documents/Proyectos/TradingCo_s1_y_s3/logs}
export PYBIN=${PYBIN:-/home/evizuete/boti/bin/python3}

# ─── Resolver script Python ───────────────────────────────────────
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PY_SCRIPT="${SCRIPT_DIR}/scripts/daily_pnl_summary.py"
if [ ! -f "${PY_SCRIPT}" ]; then
  echo "❌ No se encontró ${PY_SCRIPT}" >&2
  exit 2
fi

# ─── Construir args ───────────────────────────────────────────────
ARGS=()
if [ -n "${DATE}" ]; then
  ARGS+=(--date "${DATE}")
fi
if [ -n "${FROM}" ]; then
  ARGS+=(--from "${FROM}")
fi
if [ -n "${TO}" ]; then
  ARGS+=(--to "${TO}")
fi
if [ -n "${RECENT}" ]; then
  ARGS+=(--recent "${RECENT}")
fi
if [ "${VERBOSE}" = "1" ]; then
  ARGS+=(--verbose)
fi
ARGS+=(--logs-dir "${LOGS_DIR}")

# ─── Banner ────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  DAILY PnL SUMMARY"
echo "═══════════════════════════════════════════════════════════════"
echo "  Logs:    ${LOGS_DIR}"
if [ -n "${RECENT}" ]; then
  echo "  Modo:    últimos ${RECENT} días"
elif [ -n "${FROM}" ] && [ -n "${TO}" ]; then
  echo "  Modo:    rango ${FROM} → ${TO}"
elif [ -n "${FROM}" ]; then
  echo "  Modo:    día ${FROM}"
elif [ -n "${DATE}" ]; then
  echo "  Modo:    día ${DATE}"
else
  echo "  Modo:    hoy"
fi
[ "${VERBOSE}" = "1" ] && echo "  Verbose: ON (tabla detallada por día)"

# ─── Lanzar ────────────────────────────────────────────────────────
"${PYBIN}" "${PY_SCRIPT}" "${ARGS[@]}"
