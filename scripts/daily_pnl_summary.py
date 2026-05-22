#!/usr/bin/env python3
"""
daily_pnl_summary.py
====================

Reporte diario (y agregado) de PnL real desde los logs estructurados de S3.

Lee los eventos de s3_events_YYYYMMDD.jsonl y reconstruye el ciclo de vida
de cada trade (POSITION_OPENED → VIRTUAL_SL_TRIGGERED / ADAPTIVE_TP_CLOSE /
TIME_FORCE_CLOSE_TRIGGERED / etc.) para calcular PnL en puntos y en USD.

Modos
-----
  1. Día único (default: hoy)
  2. Rango de fechas (--from, --to)
  3. Últimos N días (--recent N)

Para cada modo emite:
  - Tabla detallada por trade del día (si --verbose)
  - Stats agregadas: win_rate, PnL total, max win/loss, avg trade
  - Histórico acumulado y running PnL (en modo rango/recent)

Uso
---
  # Hoy (default)
  python scripts/daily_pnl_summary.py

  # Día específico
  python scripts/daily_pnl_summary.py --date 20260521

  # Últimos 7 días con detalle por día
  python scripts/daily_pnl_summary.py --recent 7 --verbose

  # Rango específico (sólo resumen agregado)
  python scripts/daily_pnl_summary.py --from 20260520 --to 20260522

  # Logs en otra ubicación
  python scripts/daily_pnl_summary.py --logs-dir /otra/ruta
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

# XAUUSD: 1 unidad de precio = $100 por lote estándar (100 oz/lot).
# En este script trabajamos con "puntos centesimales": 1 pt = 0.01 precio = $1/lote.
# Por tanto: pnl_money = pnl_pts × vol  (en USD, asumiendo cuenta USD).
DEFAULT_LOGS_DIR = Path("/mnt/c/Users/Usuario/Documents/Proyectos/TradingCo_s1_y_s3/logs")

EXIT_EVENTS = {
    "VIRTUAL_SL_TRIGGERED": "VSL",
    "TIME_FORCE_CLOSE_TRIGGERED": "TIME_FORCE",
    "ADAPTIVE_TP_CLOSE": "ADAPTIVE_TP",
    "ADAPTIVE_EXTENSION_SL_TRIGGERED": "EXT_SL",
    "PARTIAL_CLOSE_TRIGGERED": "PARTIAL",
}


# ---------------------------------------------------------------------------
# Carga + parseo
# ---------------------------------------------------------------------------

def analyze_day(log_path: Path) -> List[Dict]:
    """Reconstruye trades desde un log s3_events_YYYYMMDD.jsonl."""
    tickets: Dict[int, dict] = defaultdict(dict)

    with log_path.open(encoding="utf-8") as f:
        for line in f:
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = d.get("ticket")
            ev = d.get("event")
            if t is None:
                continue

            if ev == "POSITION_OPENED":
                ts = float(d.get("ts", 0))
                tickets[t]["entry_time"] = dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S")
                tickets[t]["entry_ts"] = ts
                tickets[t]["side"] = d.get("side")
                tickets[t]["volume"] = float(d.get("volume", 0) or 0)
                tickets[t]["entry"] = float(d.get("entry_price", 0) or 0)
                tickets[t]["vsl"] = d.get("virtual_sl_price")
                tickets[t]["vtp"] = d.get("virtual_tp")
                tickets[t]["regime"] = d.get("regime", "?")
                tickets[t]["score"] = float(d.get("score", 0) or 0)

            elif ev in EXIT_EVENTS:
                ts = float(d.get("ts", 0))
                tickets[t].setdefault("exits", []).append({
                    "time": dt.datetime.fromtimestamp(ts).strftime("%H:%M:%S"),
                    "ts": ts,
                    "event": ev,
                    "price": d.get("price") or d.get("close_price") or d.get("exit_price"),
                })

    trades: List[Dict] = []
    for ticket, data in tickets.items():
        side = data.get("side", "?")
        vol = data.get("volume", 0)
        entry = data.get("entry", 0)
        exits = data.get("exits", [])

        if not exits or not data.get("side"):
            # Posición abierta o sin datos completos
            trades.append({
                **data,
                "ticket": ticket,
                "closed": False,
                "exit_price": None,
                "last_exit_event": None,
                "pnl_pts": None,
                "pnl_money": None,
                "hold_seconds": None,
            })
            continue

        last_exit = exits[-1]
        exit_price = float(last_exit["price"] or 0)

        if side == "SELL":
            pnl_pts = (entry - exit_price) * 100.0
        else:  # BUY/long
            pnl_pts = (exit_price - entry) * 100.0
        pnl_money = pnl_pts * vol  # $1/punto/lote para XAUUSD

        hold_seconds = float(last_exit.get("ts", 0)) - float(data.get("entry_ts", 0))

        trades.append({
            **data,
            "ticket": ticket,
            "closed": True,
            "exit_price": exit_price,
            "last_exit_event": last_exit["event"],
            "exit_chain": [e["event"] for e in exits],
            "pnl_pts": pnl_pts,
            "pnl_money": pnl_money,
            "hold_seconds": hold_seconds,
        })

    # Ordenar por hora de entrada
    trades.sort(key=lambda t: t.get("entry_ts", 0))
    return trades


def compute_stats(trades: List[Dict]) -> Optional[Dict]:
    closed = [t for t in trades if t["closed"]]
    if not closed:
        return None
    wins = [t for t in closed if t["pnl_pts"] > 0]
    losses = [t for t in closed if t["pnl_pts"] <= 0]
    pnls = [t["pnl_money"] for t in closed]
    pts = [t["pnl_pts"] for t in closed]
    holds = [t.get("hold_seconds", 0) or 0 for t in closed]

    return {
        "n_total": len(trades),
        "n_closed": len(closed),
        "n_open": len(trades) - len(closed),
        "n_wins": len(wins),
        "n_losses": len(losses),
        "win_rate": len(wins) / len(closed) * 100,
        "pnl_total_money": sum(pnls),
        "pnl_total_pts": sum(pts),
        "max_win": max(pnls) if pnls else 0,
        "max_loss": min(pnls) if pnls else 0,
        "avg_trade": sum(pnls) / len(pnls),
        "avg_hold_min": (sum(holds) / len(holds)) / 60.0 if holds else 0,
        "n_long": sum(1 for t in closed if t["side"] == "BUY"),
        "n_short": sum(1 for t in closed if t["side"] == "SELL"),
    }


# ---------------------------------------------------------------------------
# Output formateado
# ---------------------------------------------------------------------------

def print_day_table(date_str: str, trades: List[Dict]) -> None:
    print(f"\n{'─'*135}")
    print(f"  Detalle de trades — {date_str}")
    print(f"{'─'*135}")
    if not trades:
        print("  Sin trades.")
        return

    header = f"{'Ticket':<12} {'Open':<10} {'Side':<5} {'Vol':<6} {'Entry':<10} {'Exit':<10} {'PnL pts':>10} {'PnL $':>10} {'Hold':>8} {'Regime':<18} {'Score':<7} Exit"
    print(header)
    print("-" * 135)
    for t in trades:
        ticket = str(t["ticket"])
        if not t["closed"]:
            print(f"{ticket:<12} {t.get('entry_time','?'):<10} {t.get('side','?'):<5} "
                  f"{t.get('volume',0):<6.2f} {t.get('entry',0):<10.2f} {'(open)':<10}")
            continue
        ev_short = EXIT_EVENTS.get(t["last_exit_event"], t["last_exit_event"][:12])
        hold_min = (t.get("hold_seconds") or 0) / 60.0
        hold_str = f"{hold_min:.1f}m" if hold_min < 60 else f"{hold_min/60:.1f}h"
        emoji = "🟢" if t["pnl_pts"] > 0 else "🔴" if t["pnl_pts"] < 0 else "⚪"
        print(f"{ticket:<12} {t['entry_time']:<10} {t['side']:<5} "
              f"{t['volume']:<6.2f} {t['entry']:<10.2f} {t['exit_price']:<10.2f} "
              f"{t['pnl_pts']:>+10.1f} {t['pnl_money']:>+10.2f} {hold_str:>8} "
              f"{t.get('regime','?'):<18} {t['score']:<7.3f} {emoji} {ev_short}")


def print_day_stats(date_str: str, stats: Dict) -> None:
    if stats is None:
        print(f"\n  ⚪ {date_str}: sin trades cerrados")
        return
    emoji = "🟢" if stats["pnl_total_money"] > 0 else "🔴" if stats["pnl_total_money"] < 0 else "⚪"
    print(f"\n  {emoji} {date_str} | trades={stats['n_closed']}({stats['n_wins']}W/{stats['n_losses']}L) "
          f"wr={stats['win_rate']:.0f}% L/S={stats['n_long']}/{stats['n_short']} | "
          f"PnL={stats['pnl_total_money']:+.2f}$ "
          f"(max+{stats['max_win']:.2f}/{stats['max_loss']:.2f}) avg={stats['avg_trade']:+.2f}$ "
          f"hold≈{stats['avg_hold_min']:.1f}min")


def print_aggregate(per_day_stats: List[tuple]) -> None:
    if not per_day_stats:
        print("\n  Sin días con datos.")
        return

    total_trades = sum(s["n_closed"] for _, s in per_day_stats if s)
    total_wins = sum(s["n_wins"] for _, s in per_day_stats if s)
    total_losses = sum(s["n_losses"] for _, s in per_day_stats if s)
    total_pnl = sum(s["pnl_total_money"] for _, s in per_day_stats if s)
    total_pts = sum(s["pnl_total_pts"] for _, s in per_day_stats if s)
    positive_days = sum(1 for _, s in per_day_stats if s and s["pnl_total_money"] > 0)
    negative_days = sum(1 for _, s in per_day_stats if s and s["pnl_total_money"] < 0)
    n_days = sum(1 for _, s in per_day_stats if s)

    max_day = max(per_day_stats, key=lambda x: x[1]["pnl_total_money"] if x[1] else -1e9, default=None)
    min_day = min(per_day_stats, key=lambda x: x[1]["pnl_total_money"] if x[1] else 1e9, default=None)

    print(f"\n{'='*78}")
    print(f"  AGREGADO — {n_days} días con datos ({len(per_day_stats)} solicitados)")
    print(f"{'='*78}")
    print(f"  Total trades cerrados:  {total_trades}")
    print(f"  Wins / Losses:          {total_wins} / {total_losses}")
    if total_trades > 0:
        print(f"  Win rate agregado:      {total_wins/total_trades*100:.1f}%")
    print(f"  Días positivos:         {positive_days}")
    print(f"  Días negativos:         {negative_days}")
    if n_days > 0:
        print(f"  Pct días positivos:     {positive_days/n_days*100:.0f}%")
    print()
    emoji = "🟢" if total_pnl > 0 else "🔴" if total_pnl < 0 else "⚪"
    print(f"  PnL acumulado:          {emoji} {total_pts:+.1f} pts | {total_pnl:+.2f} USD")
    if n_days > 0:
        print(f"  Promedio diario:        {total_pnl/n_days:+.2f} USD/día")
    if max_day and max_day[1]:
        print(f"  Mejor día:              {max_day[0]} con {max_day[1]['pnl_total_money']:+.2f} USD")
    if min_day and min_day[1]:
        print(f"  Peor día:               {min_day[0]} con {min_day[1]['pnl_total_money']:+.2f} USD")

    # Running PnL ASCII
    print(f"\n  Running PnL día a día:")
    running = 0.0
    for date_str, stats in per_day_stats:
        if stats:
            running += stats["pnl_total_money"]
            bar_len = int(abs(stats["pnl_total_money"]) / 5)  # cada char = $5
            bar_len = min(bar_len, 40)
            bar = ("█" * bar_len)
            sign = "+" if stats["pnl_total_money"] >= 0 else "-"
            print(f"    {date_str}  {stats['pnl_total_money']:>+8.2f}$  → acum {running:>+8.2f}$  {sign}{bar}")
        else:
            print(f"    {date_str}  (sin trades)")


# ---------------------------------------------------------------------------
# Resolución de fechas
# ---------------------------------------------------------------------------

def parse_date(s: str) -> str:
    """Acepta YYYYMMDD o YYYY-MM-DD, devuelve YYYYMMDD."""
    s = str(s).replace("-", "").replace("/", "")
    if not re.match(r"^\d{8}$", s):
        raise argparse.ArgumentTypeError(f"Fecha inválida: {s} (esperado YYYYMMDD o YYYY-MM-DD)")
    return s


def date_range(from_str: str, to_str: str) -> List[str]:
    d_from = dt.datetime.strptime(from_str, "%Y%m%d").date()
    d_to = dt.datetime.strptime(to_str, "%Y%m%d").date()
    if d_from > d_to:
        d_from, d_to = d_to, d_from
    dates = []
    cur = d_from
    while cur <= d_to:
        dates.append(cur.strftime("%Y%m%d"))
        cur += dt.timedelta(days=1)
    return dates


def recent_dates(logs_dir: Path, n: int) -> List[str]:
    """Últimos N días con log existente (más recientes primero, sorted asc)."""
    pattern = re.compile(r"s3_events_(\d{8})\.jsonl$")
    available = []
    for p in logs_dir.glob("s3_events_*.jsonl"):
        m = pattern.search(p.name)
        if m:
            available.append(m.group(1))
    available.sort()
    return available[-n:]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--date", type=parse_date, default=None,
                        help="Día específico (YYYYMMDD o YYYY-MM-DD). Default: hoy.")
    parser.add_argument("--from", dest="from_date", type=parse_date, default=None,
                        help="Inicio del rango (incluido).")
    parser.add_argument("--to", dest="to_date", type=parse_date, default=None,
                        help="Fin del rango (incluido).")
    parser.add_argument("--recent", type=int, default=None,
                        help="Últimos N días con log disponible.")
    parser.add_argument("--logs-dir", default=str(DEFAULT_LOGS_DIR),
                        help=f"Directorio con logs s3_events_*.jsonl (default {DEFAULT_LOGS_DIR}).")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Mostrar tabla detallada de cada trade.")
    args = parser.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print(f"❌ Directorio de logs no existe: {logs_dir}", file=sys.stderr)
        return 2

    # Determinar lista de fechas
    if args.recent is not None:
        dates = recent_dates(logs_dir, args.recent)
        if not dates:
            print(f"❌ No se encontraron logs en {logs_dir}", file=sys.stderr)
            return 2
    elif args.from_date and args.to_date:
        dates = date_range(args.from_date, args.to_date)
    elif args.from_date:
        dates = [args.from_date]
    else:
        dates = [args.date or dt.datetime.now().strftime("%Y%m%d")]

    # Procesar
    per_day_stats: List[tuple] = []
    for date_str in dates:
        log_path = logs_dir / f"s3_events_{date_str}.jsonl"
        if not log_path.exists():
            per_day_stats.append((date_str, None))
            continue

        trades = analyze_day(log_path)
        stats = compute_stats(trades)

        if args.verbose or len(dates) == 1:
            print_day_table(date_str, trades)
        print_day_stats(date_str, stats)
        per_day_stats.append((date_str, stats))

    # Agregado si más de un día
    if len(dates) > 1:
        print_aggregate(per_day_stats)

    return 0


if __name__ == "__main__":
    sys.exit(main())
