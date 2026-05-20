#!/usr/bin/env python3
"""
monitor_blocker_filters.py
==========================

Monitor de filtros bloqueadores: agrega por porcentaje todas las razones de
bloqueo (block_reason, no_signal_reason) observadas en una ventana móvil
de N horas, y alerta si algún filtro está bloqueando más del % configurable.

Motivación
----------
El incident INC-2026-05-20 reveló que un único filtro (TRANSITION_WEAK_SIGNAL
con umbral mal calibrado) bloqueaba el 100% de las señales en TRANSITION sin
que nadie lo notara. Este monitor está diseñado para detectar ese patrón
proactivamente: cuando un filtro pasa a dominar (>50% por defecto), avisa.

Casos de uso
------------
  - Cron horario que alerta si un filtro empieza a dominar.
  - Tras un cambio de policy/threshold, verificar que el reparto de
    bloqueos cambió en la dirección esperada.
  - Diagnóstico cuando el operador sospecha que algo está mal pero
    el sistema parece "normal" superficialmente.

Veredictos
----------
  - OK:     ningún filtro individual bloquea > alert_threshold_pct.
  - ALERT:  uno o más filtros bloquean > alert_threshold_pct.
  - EMPTY:  no hay eventos en la ventana (sistema parado o sin logs).

Uso
---
  # Snapshot interactivo (últimas 6h, alerta si filtro > 50%)
  python scripts/monitor_blocker_filters.py

  # Modo cron: silencioso si OK, output completo si ALERT
  python scripts/monitor_blocker_filters.py --quiet
  # exit: 0=OK | 1=ALERT | 2=EMPTY | 3=error

  # Ventana más larga / threshold más estricto
  python scripts/monitor_blocker_filters.py --window-hours 24 --alert-threshold-pct 80

  # Día específico (default: hoy)
  python scripts/monitor_blocker_filters.py --date 20260520
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "main" / "logs"


# Mapping de filtro → sugerencia de causa raíz / acción correctiva.
# Si un filtro identificado aquí supera el threshold, se imprime la
# sugerencia correspondiente. Añadir entradas según se identifiquen
# nuevos filtros aguas abajo.
FILTER_SUGGESTIONS = {
    "TRANSITION_WEAK_SIGNAL":
        "umbral transition_min_proba_delta puede ser demasiado alto. "
        "Validar con: python scripts/validate_calibrator_thresholds.py",
    "BLOCK_CHOP":
        "el detector chop está bloqueando entradas. Revisar "
        "config.strategy_gate.chop_block / chop_size_mult en s2_config.py.",
    "BLOCK_EXHAUSTION_REENTRY":
        "el detector de agotamiento está bloqueando reentradas. Revisar "
        "config.strategy_gate.exhaustion_blocks_reentry.",
    "REVERSAL_GUARD_LONG_TREND_DOWN":
        "reversal_guard bloqueando LONG en TREND_DOWN. Revisar min_proba_edge "
        "o long_min_rsi en config.reversal_guard.",
    "REVERSAL_GUARD_SHORT_TREND_UP":
        "reversal_guard bloqueando SHORT en TREND_UP. Revisar min_proba_edge "
        "o short_max_rsi en config.reversal_guard.",
    "DECISION_ENGINE_NONE":
        "el engine NO produce decisión (probas demasiado bajas/empatadas). "
        "Verificar calibrador con validate_calibrator_thresholds.py.",
    "COUNTER_TREND_BLOCKED":
        "señales contra-tendencia bloqueadas totalmente. Revisar "
        "config.counter_trend.block_total y los regímenes listados.",
    "RSI_OVERBOUGHT":
        "RSI > umbral en LONG. Revisar config.rsi_overbought_threshold.",
    "RSI_OVERSOLD":
        "RSI < umbral en SHORT. Revisar config.rsi_oversold_threshold.",
    "POST_CLOSE_COOLDOWN":
        "cooldown post-cierre dominante — quizá pocas oportunidades reales, "
        "o cooldown demasiado largo. Revisar config.keepalive.",
    "SIGNAL_INTER_COOLDOWN":
        "cooldown entre señales dominante — revisar signal_cooldown_bars.",
    "ENTRY_GAP_TOO_LARGE":
        "gap precio→entry excesivo. Revisar config.open_guard.max_entry_gap_pts.",
    "ANOMALY_BLOCK":
        "anomaly_score por encima del umbral. Revisar anomaly_block_threshold.",
}


def _scan_window(signals_log: Path, since_ts: float) -> dict:
    """Cuenta eventos por tipo y razón desde `since_ts`.

    Distingue entre:
      - n_no_reason_logged: NO_SIGNAL sin block_reason ni no_signal_reason.
        Suelen ser ticks donde el engine ni siquiera evaluó (mercado parado,
        startup_grace, datos insuficientes). NO los contamos como bloqueos
        ejecutados — son ruido contextual.
      - n_evaluated: NO_SIGNAL con razón explícita + SIGNAL_SENT. Es el
        denominador relevante para calcular % de bloqueo de cada filtro.
    """
    stats = {
        "n_total": 0,
        "n_no_signal": 0,
        "n_signal_sent": 0,
        "n_no_reason_logged": 0,
        "n_with_diag": 0,
        "reasons": Counter(),  # solo razones explícitas (block_reason o no_signal_reason)
        "states": Counter(),
    }
    if not signals_log.exists():
        return stats

    with signals_log.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except (json.JSONDecodeError, AttributeError):
                continue
            ts = d.get("ts")
            if ts is None or ts < since_ts:
                continue

            stats["n_total"] += 1
            ev = d.get("event", "?")
            md = d.get("model_diag") or {}
            if md:
                stats["n_with_diag"] += 1
                if st := md.get("state"):
                    stats["states"][st] += 1

            if ev == "NO_SIGNAL":
                stats["n_no_signal"] += 1
                if br := d.get("block_reason"):
                    stats["reasons"][br.split("(")[0]] += 1
                elif nsr := md.get("no_signal_reason"):
                    stats["reasons"][nsr.split("(")[0]] += 1
                else:
                    # Sin razón — el engine ni siquiera evaluó este tick
                    stats["n_no_reason_logged"] += 1
            elif ev == "SIGNAL_SENT":
                stats["n_signal_sent"] += 1

    return stats


def _format_window(hours: float) -> str:
    if hours < 1:
        return f"{hours*60:.0f}min"
    return f"{hours:.1f}h"


def _print_report(
    date_str: str, window_hours: float, stats: dict, alert_threshold_pct: float,
    quiet: bool,
) -> int:
    # n_engine_evaluated = eventos donde el engine SÍ tomó decisión
    # (NO_SIGNAL con razón explícita + SIGNAL_SENT). Es el denominador
    # relevante para % de bloqueo por filtro.
    n_engine_evaluated = (
        (stats["n_no_signal"] - stats["n_no_reason_logged"])
        + stats["n_signal_sent"]
    )
    n_blocked = stats["n_no_signal"] - stats["n_no_reason_logged"]

    if n_engine_evaluated == 0:
        if not quiet:
            print(f"\n⚠️  Sin evaluaciones del engine en ventana de {_format_window(window_hours)} ({date_str}).")
            print(f"   Total eventos: {stats['n_total']}, todos sin razón loggeada.")
            print(f"   Posible: mercado parado, fin de semana, o sistema sin logs útiles.")
        return 2  # EMPTY

    # Identificar filtros con >threshold sobre eventos evaluados
    alerting = []
    for reason, count in stats["reasons"].items():
        pct = count / n_engine_evaluated * 100
        if pct > alert_threshold_pct:
            alerting.append((reason, count, pct))
    alerting.sort(key=lambda x: -x[2])

    verdict = "ALERT" if alerting else "OK"

    if quiet and verdict == "OK":
        return 0

    print(f"\n{'='*78}")
    print(f"  MONITOR DE FILTROS BLOQUEADORES — {date_str}")
    print(f"  Ventana: últimas {_format_window(window_hours)}")
    print(f"  Threshold de alerta: > {alert_threshold_pct:.0f}% del bloqueo total")
    print(f"{'='*78}")

    print(f"\n  Eventos totales:           {stats['n_total']}")
    print(f"  · sin razón loggeada:      {stats['n_no_reason_logged']} (ticks no evaluados — mercado parado, datos insuficientes, etc.)")
    print(f"  · evaluados por engine:    {n_engine_evaluated}")
    print(f"     - SIGNAL_SENT:          {stats['n_signal_sent']}")
    print(f"     - bloqueados con razón: {n_blocked}")
    pct_sent = stats['n_signal_sent'] / n_engine_evaluated * 100
    print(f"  Tasa SIGNAL_SENT sobre evaluados:  {pct_sent:.1f}%")
    print()

    if stats["states"]:
        total_state = sum(stats["states"].values())
        print(f"  Distribución de estados (sobre {total_state} ticks con diag):")
        for state, c in stats["states"].most_common(8):
            pct = c / total_state * 100
            bar = "█" * int(pct / 2)
            print(f"    {state:25s} {c:5d}  {pct:5.1f}%  {bar}")
        print()

    if stats["reasons"]:
        print(f"  Razones de bloqueo (% sobre {n_engine_evaluated} eventos evaluados):")
        for reason, c in stats["reasons"].most_common():
            pct = c / n_engine_evaluated * 100
            marker = "🚨 " if pct > alert_threshold_pct else "   "
            bar = "█" * int(pct / 2)
            print(f"  {marker}{c:5d}  {pct:5.1f}%  {reason:35s}  {bar}")
        print()

    # Sugerencias para filtros que alertan
    if alerting:
        print(f"  ╔════════════════════════════════════════════════════════════════╗")
        print(f"  ║  🚨 FILTROS DOMINANTES (>{alert_threshold_pct:.0f}%) — POSIBLES CAUSAS:           ║")
        print(f"  ╚════════════════════════════════════════════════════════════════╝")
        for reason, count, pct in alerting:
            print(f"\n  • {reason} ({pct:.1f}%, {count} eventos)")
            suggestion = FILTER_SUGGESTIONS.get(reason)
            if suggestion:
                print(f"    💡 {suggestion}")
            else:
                print(f"    💡 Filtro no catalogado — investigar en código fuente.")
        print()

    # Veredicto
    emoji = {"OK": "✅", "ALERT": "🚨"}.get(verdict, "?")
    print(f"  {'─'*70}")
    print(f"  Veredicto: {emoji} {verdict}")
    if verdict == "OK":
        print(f"  Ningún filtro bloquea > {alert_threshold_pct:.0f}% individualmente.")
    print(f"  {'─'*70}\n")

    return 1 if verdict == "ALERT" else 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date", default=None,
        help="Fecha YYYYMMDD a evaluar (default: hoy)",
    )
    parser.add_argument(
        "--window-hours", type=float, default=6.0,
        help="Ventana móvil de evaluación en horas (default 6.0)",
    )
    parser.add_argument(
        "--alert-threshold-pct", type=float, default=50.0,
        help="Si un filtro individual bloquea > este %% → ALERT (default 50.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Output solo si veredicto != OK",
    )
    args = parser.parse_args()

    date_str = args.date or dt.datetime.now().strftime("%Y%m%d")
    signals_log = LOGS_DIR / f"signals_{date_str}.jsonl"

    if not signals_log.exists():
        print(f"❌ signals log no encontrado: {signals_log}", file=sys.stderr)
        return 3

    since_ts = dt.datetime.now().timestamp() - (args.window_hours * 3600)
    stats = _scan_window(signals_log, since_ts=since_ts)

    return _print_report(
        date_str=date_str, window_hours=args.window_hours, stats=stats,
        alert_threshold_pct=args.alert_threshold_pct, quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
