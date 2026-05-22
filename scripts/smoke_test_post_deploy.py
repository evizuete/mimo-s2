#!/usr/bin/env python3
"""
smoke_test_post_deploy.py
=========================

Verifica que el sistema emite señales tras un restart de s2. Reporta:
  - Cuánto tiempo lleva el proceso vivo desde el último restart.
  - Cuántos eventos ha procesado, cuántos NO_SIGNAL y cuántos SIGNAL_SENT.
  - Breakdown de las razones de bloqueo (block_reason).
  - Distribución de estados clasificados por el StateDetector.
  - Veredicto: PASS / FAIL / WAITING.

Motivación
----------
El incident INC-2026-05-20 mantuvo el sistema 7 días sin trades sin que
nadie lo detectara — los logs por separado parecían normales, pero el
sistema NO operaba. Este smoke test debería disparar una alerta dentro
de las primeras horas tras cualquier deploy/restart si la tasa de
trades es cero.

Veredicto
---------
  · PASS:    SIGNAL_SENT >= 1 y al menos `min_uptime_minutes` de uptime.
  · WAITING: SIGNAL_SENT == 0 y uptime < `fail_after_hours`.
  · FAIL:    SIGNAL_SENT == 0 y uptime >= `fail_after_hours`.

Defaults:
  · fail_after_hours = 6     (si en 6h no opera, alerta)
  · min_uptime_minutes = 5   (necesita al menos 5min vivo para PASS)

Uso
---
  # Snapshot del estado actual (interactivo)
  python scripts/smoke_test_post_deploy.py

  # Modo CI/cron: silencioso si OK, output completo si FAIL
  python scripts/smoke_test_post_deploy.py --quiet
  # exit code: 0 = PASS, 1 = FAIL, 2 = WAITING (uptime < fail_after), 3 = error

  # Ajustar umbral de alerta (ej: alertar si en 2h no opera)
  python scripts/smoke_test_post_deploy.py --fail-after-hours 2

  # Mirar día específico (default: hoy)
  python scripts/smoke_test_post_deploy.py --date 2026-05-20
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LOGS_DIR = PROJECT_ROOT / "main" / "logs"


def _now_ts() -> float:
    return dt.datetime.now().timestamp()


def _find_latest_restart(system_log: Path) -> Optional[float]:
    """Devuelve el ts del último SYSTEM_LOGGER_STARTED (restart de s2)."""
    if not system_log.exists():
        return None
    latest = None
    with system_log.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
                if d.get("event") == "SYSTEM_LOGGER_STARTED":
                    ts = d.get("ts")
                    if ts is not None and (latest is None or ts > latest):
                        latest = ts
            except (json.JSONDecodeError, AttributeError):
                continue
    return latest


def _scan_signals(signals_log: Path, since_ts: float) -> dict:
    """Cuenta eventos por tipo y razón desde `since_ts`."""
    stats = {
        "n_no_signal": 0,
        "n_signal_sent": 0,
        "n_with_model_diag": 0,
        "block_reasons": Counter(),
        "no_signal_reasons": Counter(),
        "states_seen": Counter(),
        "signal_sent_details": [],
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

            ev = d.get("event", "?")
            md = d.get("model_diag") or {}
            if md:
                stats["n_with_model_diag"] += 1
                if st := md.get("state"):
                    stats["states_seen"][st] += 1

            if ev == "NO_SIGNAL":
                stats["n_no_signal"] += 1
                if br := d.get("block_reason"):
                    stats["block_reasons"][br.split("(")[0]] += 1
                elif nsr := md.get("no_signal_reason"):
                    stats["no_signal_reasons"][nsr.split("(")[0]] += 1
            elif ev == "SIGNAL_SENT":
                stats["n_signal_sent"] += 1
                stats["signal_sent_details"].append({
                    "ts": ts,
                    "state": d.get("state") or md.get("state"),
                    "side": d.get("side"),
                    "qty": d.get("qty"),
                    "entry": d.get("entry"),
                    "score": d.get("score"),
                })
    return stats


def _format_uptime(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds/60:.1f}min"
    return f"{seconds/3600:.2f}h"


def _verdict(
    uptime_secs: float, n_signal_sent: int,
    fail_after_hours: float, min_uptime_minutes: float,
) -> str:
    if n_signal_sent >= 1 and uptime_secs >= min_uptime_minutes * 60:
        return "PASS"
    if n_signal_sent == 0 and uptime_secs >= fail_after_hours * 3600:
        return "FAIL"
    return "WAITING"


def _print_report(
    date_str: str, restart_ts: float, stats: dict,
    fail_after_hours: float, min_uptime_minutes: float,
    quiet: bool,
) -> int:
    uptime_secs = _now_ts() - restart_ts
    verdict = _verdict(uptime_secs, stats["n_signal_sent"], fail_after_hours, min_uptime_minutes)

    # En --quiet, sólo emitir output cuando FAIL. PASS y WAITING son estados
    # esperables tras un restart y no deben spamear logs si está en cron.
    if quiet and verdict != "FAIL":
        return {"PASS": 0, "WAITING": 2}[verdict]

    print(f"\n{'='*78}")
    print(f"  SMOKE TEST POST-DEPLOY — {date_str}")
    print(f"{'='*78}")

    restart_dt = dt.datetime.fromtimestamp(restart_ts).strftime("%Y-%m-%d %H:%M:%S")
    print(f"  Último restart de s2: {restart_dt}")
    print(f"  Uptime:               {_format_uptime(uptime_secs)}")
    print(f"  Eventos NO_SIGNAL:    {stats['n_no_signal']}")
    print(f"  Eventos SIGNAL_SENT:  {stats['n_signal_sent']}")
    print(f"  Con model_diag:       {stats['n_with_model_diag']}")
    print()

    if stats["states_seen"]:
        total_diag = sum(stats["states_seen"].values())
        print(f"  Distribución de estados (sobre {total_diag} ticks con model_diag):")
        for state, c in stats["states_seen"].most_common():
            pct = c / total_diag * 100
            bar = "█" * int(pct / 2)
            print(f"    {state:20s} {c:5d}  {pct:5.1f}%  {bar}")
        print()

    if stats["block_reasons"]:
        print(f"  Razones de bloqueo (block_reason):")
        for reason, c in stats["block_reasons"].most_common():
            print(f"    {c:5d}  {reason}")
        print()
    if stats["no_signal_reasons"]:
        print(f"  Otras razones (no_signal_reason):")
        for reason, c in stats["no_signal_reasons"].most_common():
            print(f"    {c:5d}  {reason}")
        print()

    if stats["signal_sent_details"]:
        print(f"  Señales enviadas (últimas 10):")
        for s in stats["signal_sent_details"][-10:]:
            time_str = dt.datetime.fromtimestamp(s["ts"]).strftime("%H:%M:%S")
            print(f"    [{time_str}] {s['state']} {s['side']} qty={s['qty']} entry={s['entry']} score={s['score']:.4f}")
        print()

    # Veredicto
    emoji = {"PASS": "✅", "FAIL": "❌", "WAITING": "⏳"}.get(verdict, "?")
    print(f"  {'─'*60}")
    print(f"  Veredicto: {emoji} {verdict}")
    if verdict == "WAITING":
        remaining_hours = fail_after_hours - uptime_secs / 3600
        print(f"  Aún tolerable. Alerta si en {remaining_hours:.2f}h más no hay SIGNAL_SENT.")
    elif verdict == "FAIL":
        print(f"  ⚠️  Sin trades tras {_format_uptime(uptime_secs)} de uptime.")
        print(f"  Posibles causas (mirar block_reasons arriba):")
        if "TRANSITION_WEAK_SIGNAL" in stats["block_reasons"]:
            print(f"    · TRANSITION_WEAK_SIGNAL bloqueando — verificar transition_min_proba_delta")
        if "BLOCK_CHOP" in stats["block_reasons"]:
            print(f"    · BLOCK_CHOP bloqueando — verificar StrategyGate config")
        if "REVERSAL_GUARD" in str(stats["block_reasons"]):
            print(f"    · REVERSAL_GUARD bloqueando — verificar min_proba_edge")
        if "DECISION_ENGINE_NONE" in stats["block_reasons"]:
            print(f"    · El modelo no produce decisión — verificar calibrador/scores")
        if stats["states_seen"].get("VOLATILE", 0) > sum(stats["states_seen"].values()) * 0.4:
            print(f"    · >40% del tiempo en VOLATILE — verificar threshold injection (StateDetector)")
        print(f"\n  Comandos de diagnóstico:")
        print(f"    grep -i 'Regime thresholds' main/logs/s2_*.log")
        print(f"    python scripts/validate_calibrator_thresholds.py")
    print(f"  {'─'*60}\n")

    return {"PASS": 0, "FAIL": 1, "WAITING": 2}[verdict]


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--date", default=None,
        help="Fecha YYYYMMDD a evaluar (default: hoy)",
    )
    parser.add_argument(
        "--fail-after-hours", type=float, default=6.0,
        help="Horas tras el restart sin SIGNAL_SENT para emitir FAIL (default 6.0)",
    )
    parser.add_argument(
        "--min-uptime-minutes", type=float, default=5.0,
        help="Uptime mínimo antes de poder emitir PASS (default 5.0)",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Output solo si veredicto != PASS",
    )
    args = parser.parse_args()

    date_str = args.date or dt.datetime.now().strftime("%Y%m%d")
    system_log = LOGS_DIR / f"s2_system_{date_str}.jsonl"
    signals_log = LOGS_DIR / f"signals_{date_str}.jsonl"

    if not system_log.exists():
        print(f"❌ system log no encontrado: {system_log}", file=sys.stderr)
        return 3
    if not signals_log.exists():
        print(f"❌ signals log no encontrado: {signals_log}", file=sys.stderr)
        return 3

    restart_ts = _find_latest_restart(system_log)
    if restart_ts is None:
        print(f"❌ No se encontró SYSTEM_LOGGER_STARTED en {system_log}", file=sys.stderr)
        return 3

    stats = _scan_signals(signals_log, since_ts=restart_ts)
    return _print_report(
        date_str=date_str, restart_ts=restart_ts, stats=stats,
        fail_after_hours=args.fail_after_hours,
        min_uptime_minutes=args.min_uptime_minutes,
        quiet=args.quiet,
    )


if __name__ == "__main__":
    sys.exit(main())
