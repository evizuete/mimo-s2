#!/usr/bin/env python3
"""
post_filter_replay_trades.py
============================

Post-filtro de trades.parquet del replay aplicando los filtros del
s2_service_v2 que NO están en el simulator. Devuelve un veredicto de
qué trades habrían sido enviados a MT5 en producción real.

Motivación
----------
El replay (scripts/replay_s2_202500.py) corre `simulator.backtest()`
directamente. Esto NO pasa por la lógica de `S2Service.process_tick()`,
que contiene ~8 filtros defensivos adicionales (TRANSITION_WEAK_SIGNAL,
REVERSAL_GUARD, RSI guards, cooldowns, etc.). Resultado: el replay puede
sobreestimar el n_trades de 3-6× respecto a la producción real.

Este script reconstruye el contexto tick a tick (via simulator.predict)
y aplica los filtros del service para cada trade del replay. Output: un
trades.parquet filtrado + summary recalculado, comparable al comportamiento
de la producción real.

Filtros aplicados
-----------------
1. TRANSITION_WEAK_SIGNAL  — delta cal_lado-cal_opuesto >= transition_min_proba_delta
2. REVERSAL_GUARD_LONG_TREND_DOWN  — long en trend_down requiere RSI + MACD + edge
3. REVERSAL_GUARD_SHORT_TREND_UP   — short en trend_up análogo
4. RSI_OVERBOUGHT  — long con RSI > rsi_overbought_threshold
5. RSI_OVERSOLD    — short con RSI < rsi_oversold_threshold
6. COUNTER_TREND_BLOCKED  — counter_trend.block_total
7. POST_CLOSE_COOLDOWN  — cooldown post-cierre (aproximado)
8. SIGNAL_INTER_COOLDOWN  — cooldown entre señales (aproximado)

NO aplicados (info no disponible o no determinista):
  - ENTRY_GAP_TOO_LARGE (requiere precio bid/ask en vivo)
  - ANOMALY_BLOCK del service (criterio similar al del simulator)

Uso
---
  python scripts/post_filter_replay_trades.py \\
    --release 202500 \\
    --deploy-subdir deploy_validation_combined_seed47 \\
    --policy-config decision_policies_config_202500_validation_v2 \\
    --replay-dir /tmp/replay_lockbox_rolling \\
    --from 2026-05-11 --to 2026-05-17 \\
    --warmup-from 2026-04-25
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import numpy as np

# Reutilizar build_simulator del replay
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from replay_s2_202500 import build_simulator, load_ohlcv  # type: ignore


# ---------------------------------------------------------------------------
# Filtros del s2_service_v2 reimplementados
# ---------------------------------------------------------------------------

def filter_transition_weak(ctx: pd.Series, side: str, min_delta: float) -> Optional[str]:
    """TRANSITION_WEAK_SIGNAL: en TRANSITION_*, delta cal_lado-cal_opuesto >= min_delta."""
    state = str(ctx.get("state", "")).lower()
    if "transition" not in state:
        return None
    pl = float(ctx.get("pred_long_cal", 0.0))
    ps = float(ctx.get("pred_short_cal", 0.0))
    delta = (ps - pl) if side in ("sell", "short") else (pl - ps)
    if delta < min_delta:
        return f"TRANSITION_WEAK_SIGNAL(delta={delta:.3f}<{min_delta})"
    return None


def filter_reversal_guard(ctx: pd.Series, side: str, cfg: dict) -> Optional[str]:
    """REVERSAL_GUARD_*: contra-tendencia fuerte requiere RSI + MACD + edge confirmando."""
    state = str(ctx.get("state", "")).lower()
    if not cfg.get("enabled", True):
        return None

    rsi = ctx.get("rsi")
    macd_hist = ctx.get("macd_hist")
    pl = float(ctx.get("pred_long_cal", 0.0))
    ps = float(ctx.get("pred_short_cal", 0.0))
    min_edge = float(cfg.get("min_proba_edge", 0.015))
    require_macd = bool(cfg.get("require_macd_flip", True))

    if side in ("buy", "long") and state == "trend_down" and cfg.get("long_in_trend_down", True):
        if rsi is None or float(rsi) < float(cfg.get("long_min_rsi", 46.0)):
            return f"REVERSAL_GUARD_LONG_TREND_DOWN(rsi={rsi})"
        if require_macd and macd_hist is not None and float(macd_hist) < 0.0:
            return f"REVERSAL_GUARD_LONG_TREND_DOWN(macd_hist={macd_hist:.4f}<0)"
        edge = pl - ps
        if edge < min_edge:
            return f"REVERSAL_GUARD_LONG_TREND_DOWN(edge={edge:.4f}<{min_edge})"

    if side in ("sell", "short") and state == "trend_up" and cfg.get("short_in_trend_up", True):
        if rsi is None or float(rsi) > float(cfg.get("short_max_rsi", 54.0)):
            return f"REVERSAL_GUARD_SHORT_TREND_UP(rsi={rsi})"
        if require_macd and macd_hist is not None and float(macd_hist) > 0.0:
            return f"REVERSAL_GUARD_SHORT_TREND_UP(macd_hist={macd_hist:.4f}>0)"
        edge = ps - pl
        if edge < min_edge:
            return f"REVERSAL_GUARD_SHORT_TREND_UP(edge={edge:.4f}<{min_edge})"

    return None


def filter_rsi_overbought_oversold(
    ctx: pd.Series, side: str,
    rsi_overbought: float, rsi_oversold: float
) -> Optional[str]:
    rsi = ctx.get("rsi")
    if rsi is None:
        return None
    rsi = float(rsi)
    if side in ("buy", "long") and rsi > rsi_overbought:
        return f"RSI_OVERBOUGHT(rsi={rsi:.1f}>{rsi_overbought})"
    if side in ("sell", "short") and rsi < rsi_oversold:
        return f"RSI_OVERSOLD(rsi={rsi:.1f}<{rsi_oversold})"
    return None


def filter_counter_trend(ctx: pd.Series, side: str, cfg: dict) -> Optional[str]:
    """COUNTER_TREND_BLOCKED si cfg.block_total y side va contra-trend del state."""
    if not cfg.get("block_total", False):
        return None
    state = str(ctx.get("state", "")).lower()
    short_block = cfg.get("regimes_short_block", set())
    long_block = cfg.get("regimes_long_block", set())
    if side in ("sell", "short") and state in short_block:
        return f"COUNTER_TREND_BLOCKED(state={state})"
    if side in ("buy", "long") and state in long_block:
        return f"COUNTER_TREND_BLOCKED(state={state})"
    return None


def apply_cooldowns(
    trades: pd.DataFrame, post_close_secs: float, signal_inter_secs: float,
) -> pd.DataFrame:
    """Aplica POST_CLOSE_COOLDOWN y SIGNAL_INTER_COOLDOWN como segunda pasada.

    Estos filtros son temporales: rechazan trades que ocurren demasiado pronto
    tras el cierre del trade anterior o tras la última señal.

    Atención: este es un approx pesimista. En producción los cooldowns
    aplican sobre 'señales enviadas', no sobre 'trades efectivamente abiertos'.
    Aquí asumimos que cada trade del replay corresponde a una señal enviada.
    """
    trades = trades.copy()
    if "cooldown_reason" not in trades.columns:
        trades["cooldown_reason"] = None

    last_exit_time = None
    last_signal_time = None

    for idx in trades.index:
        t = trades.loc[idx]
        # Solo aplicar a trades que NO han sido ya bloqueados
        if t.get("service_filter") and t["service_filter"] != "OK":
            continue

        entry_t = pd.Timestamp(t["entry_time"]).to_pydatetime()
        # post-close cooldown
        if last_exit_time is not None:
            delta_close = (entry_t - last_exit_time).total_seconds()
            if delta_close < post_close_secs:
                trades.at[idx, "cooldown_reason"] = (
                    f"POST_CLOSE_COOLDOWN(remaining={post_close_secs-delta_close:.0f}s)"
                )
                continue
        # inter-signal cooldown
        if last_signal_time is not None:
            delta_sig = (entry_t - last_signal_time).total_seconds()
            if delta_sig < signal_inter_secs:
                trades.at[idx, "cooldown_reason"] = (
                    f"SIGNAL_INTER_COOLDOWN(remaining={signal_inter_secs-delta_sig:.0f}s)"
                )
                continue
        # Si pasa cooldowns → este se envía
        last_signal_time = entry_t
        if pd.notna(t.get("exit_time")):
            last_exit_time = pd.Timestamp(t["exit_time"]).to_pydatetime()

    return trades


# ---------------------------------------------------------------------------
# Pipeline principal
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--release", default="202500")
    parser.add_argument("--deploy-subdir", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--replay-dir", required=True, help="Directorio con trades.parquet del replay")
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--warmup-from", default=None)
    parser.add_argument("--base-tf", default="5min")
    parser.add_argument("--transition-min-delta", type=float, default=0.04,
                        help="(s2_config.transition_min_proba_delta) — default 0.04")
    parser.add_argument("--rsi-overbought", type=float, default=75.0)
    parser.add_argument("--rsi-oversold", type=float, default=25.0)
    parser.add_argument("--reversal-min-edge", type=float, default=0.015,
                        help="(s2_config.reversal_guard.min_proba_edge) — default 0.015")
    parser.add_argument("--reversal-long-min-rsi", type=float, default=46.0)
    parser.add_argument("--reversal-short-max-rsi", type=float, default=54.0)
    parser.add_argument("--counter-trend-block-total", action="store_true",
                        help="(s2_config.counter_trend.block_total) — default False")
    parser.add_argument("--post-close-cooldown-secs", type=float, default=0.0,
                        help="POST_CLOSE_COOLDOWN: segundos tras cierre de trade anterior. Default 0 (off).")
    parser.add_argument("--signal-inter-cooldown-secs", type=float, default=0.0,
                        help="SIGNAL_INTER_COOLDOWN: segundos entre señales. Default 0 (off).")

    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    trades_path = replay_dir / "trades.parquet"
    summary_path = replay_dir / "summary.json"
    if not trades_path.exists():
        print(f"❌ No se encontró: {trades_path}", file=sys.stderr)
        return 2

    trades = pd.read_parquet(trades_path)
    print(f"\n📂 Cargados {len(trades)} trades del replay desde {trades_path}")

    # Construir simulator para llamar predict y obtener tick features
    print("🔧 Construyendo simulator (para predict de tick features)...")
    project_root = Path(__file__).resolve().parent.parent
    base_dir = project_root / "main"
    simulator = build_simulator(
        release=args.release,
        deploy_subdir=args.deploy_subdir,
        policy_module=f"config.{args.policy_config}",
        base_dir=base_dir,
        artifacts_root=project_root / "artifacts",
        score_low_quantile=80,
    )

    load_from = args.warmup_from or args.from_date
    print(f"📥 Cargando OHLCV {load_from} → {args.to_date} ...")
    df = load_ohlcv(load_from, args.to_date, args.base_tf)
    print(f"   {len(df)} filas")

    print("🔮 Ejecutando simulator.predict(simulation=True) para tick features...")
    df_pred = simulator.predict(df, simulation=True)
    print(f"   Predicciones generadas para {len(df_pred)} ticks")
    print(f"   Cols clave: pred_long_cal={'pred_long_cal' in df_pred.columns}, "
          f"pred_short_cal={'pred_short_cal' in df_pred.columns}, "
          f"state={'state' in df_pred.columns}, rsi={'rsi' in df_pred.columns}, "
          f"macd_hist={'macd_hist' in df_pred.columns}")

    # Asegurar que df_pred tiene un timestamp accesible
    if "time" in df_pred.columns:
        time_series = pd.to_datetime(df_pred["time"])
    elif isinstance(df_pred.index, pd.DatetimeIndex):
        time_series = df_pred.index.to_series()
    else:
        raise SystemExit("❌ No se pudo identificar columna time en df_pred")

    df_pred_sorted = df_pred.copy()
    df_pred_sorted = df_pred_sorted.reset_index(drop=True)
    df_pred_sorted["_time"] = time_series.values
    df_pred_sorted = df_pred_sorted.sort_values("_time").reset_index(drop=True)
    times_idx = pd.DatetimeIndex(df_pred_sorted["_time"])
    print(f"   df_pred rango temporal: {times_idx.min()} → {times_idx.max()}")

    # Aplicar filtros tick-context
    print("\n🛡  Aplicando filtros del s2_service_v2...")
    counter_trend_cfg = {
        "block_total": args.counter_trend_block_total,
        "regimes_short_block": {"trend_up"},
        "regimes_long_block": {"trend_down"},
    }
    reversal_guard_cfg = {
        "enabled": True,
        "long_in_trend_down": True,
        "short_in_trend_up": True,
        "long_min_rsi": args.reversal_long_min_rsi,
        "short_max_rsi": args.reversal_short_max_rsi,
        "require_macd_flip": True,
        "min_proba_edge": args.reversal_min_edge,
    }

    service_filters = []
    for _, t in trades.iterrows():
        # Mapear entry_time → contexto más reciente en df_pred (asof match)
        entry_t = pd.Timestamp(t["entry_time"])
        # buscar la fila con _time <= entry_t más reciente
        pos = times_idx.searchsorted(entry_t, side="right") - 1
        if pos < 0 or pos >= len(df_pred_sorted):
            service_filters.append(f"NO_MATCHING_TICK({entry_t})")
            continue
        ctx = df_pred_sorted.iloc[pos]
        # Verificar gap: si más de 10min entre tick y entry, sospechoso
        gap_min = (entry_t - times_idx[pos]).total_seconds() / 60
        if gap_min > 10:
            service_filters.append(f"TICK_GAP_TOO_LARGE({gap_min:.0f}min)")
            continue
        side = str(t["side"]).lower()

        for fn, args_fn in [
            (filter_transition_weak, (ctx, side, args.transition_min_delta)),
            (filter_reversal_guard, (ctx, side, reversal_guard_cfg)),
            (filter_rsi_overbought_oversold,
                (ctx, side, args.rsi_overbought, args.rsi_oversold)),
            (filter_counter_trend, (ctx, side, counter_trend_cfg)),
        ]:
            reason = fn(*args_fn)
            if reason:
                service_filters.append(reason)
                break
        else:
            service_filters.append("OK")

    trades["service_filter"] = service_filters

    # Cooldowns (segunda pasada — necesita orden temporal)
    if args.post_close_cooldown_secs > 0 or args.signal_inter_cooldown_secs > 0:
        print(f"⏱  Aplicando cooldowns: post_close={args.post_close_cooldown_secs}s, "
              f"signal_inter={args.signal_inter_cooldown_secs}s")
        trades = apply_cooldowns(
            trades.sort_values("entry_time").reset_index(drop=True),
            args.post_close_cooldown_secs,
            args.signal_inter_cooldown_secs,
        )

    # Resultado final
    trades["would_pass_service"] = (
        (trades["service_filter"] == "OK")
        & (trades.get("cooldown_reason").isna() if "cooldown_reason" in trades.columns else True)
    )

    n_total = len(trades)
    n_pass = int(trades["would_pass_service"].sum())
    n_block = n_total - n_pass

    # Reporte
    print(f"\n{'='*70}")
    print(f"  RESULTADO POST-FILTRO")
    print(f"{'='*70}")
    print(f"\n  Trades originales (replay sin filtros service): {n_total}")
    print(f"  Trades que pasan los filtros del service:        {n_pass} ({n_pass/n_total*100:.1f}%)")
    print(f"  Trades que NO se hubieran enviado en producción: {n_block} ({n_block/n_total*100:.1f}%)")

    # Razones de bloqueo
    print(f"\n  Razones de bloqueo:")
    reasons = Counter()
    for r in trades["service_filter"]:
        reasons[r.split("(")[0]] += 1
    if "cooldown_reason" in trades.columns:
        for r in trades["cooldown_reason"].dropna():
            reasons[r.split("(")[0]] += 1
    for reason, c in reasons.most_common():
        print(f"    {c:5d}  {reason}")

    # Métricas recalculadas sobre subset filtrado
    filt = trades[trades["would_pass_service"]].copy()
    print(f"\n  ── Métricas recalculadas (solo subset filtrado) ──")
    print(f"    n_trades:    {len(filt)}")
    if len(filt) > 0:
        print(f"    pnl_total:   {filt['pnl'].sum():.2f}")
        print(f"    win_rate:    {(filt['pnl'] > 0).mean()*100:.1f}%")
        print(f"    n_long:      {(filt['side']=='long').sum()}")
        print(f"    n_short:     {(filt['side']=='short').sum()}")

        # Comparación con summary original
        if summary_path.exists():
            orig = json.loads(summary_path.read_text())
            init_eq = orig.get("initial_equity", 10000)
            pnl_pct_filtered = filt["pnl"].sum() / init_eq * 100
            print(f"    pnl_pct (vs initial_equity {init_eq:.0f}): {pnl_pct_filtered:.2f}%")
            print(f"\n  ── Comparación replay original vs post-filtrado ──")
            print(f"    {'métrica':20s} {'original':>12s}  {'post-filtro':>12s}")
            print(f"    {'n_trades':20s} {orig.get('n_trades', 0):>12d}  {len(filt):>12d}")
            print(f"    {'pnl_pct':20s} {orig.get('pnl_pct', 0):>11.2f}%  {pnl_pct_filtered:>11.2f}%")
            print(f"    {'win_rate_pct':20s} {orig.get('win_rate_pct', 0):>11.1f}%  {(filt['pnl']>0).mean()*100:>11.1f}%")

    # Guardar
    out_trades = replay_dir / "trades_post_filtered.parquet"
    out_summary = replay_dir / "summary_post_filtered.json"
    trades.to_parquet(out_trades, index=False)

    out_summary_data = {
        "n_trades_original": n_total,
        "n_trades_post_filter": int(len(filt)),
        "pct_blocked_by_service": float(n_block / n_total * 100),
        "pnl_total_post_filter": float(filt["pnl"].sum()) if len(filt) else 0.0,
        "win_rate_post_filter_pct": float((filt["pnl"] > 0).mean() * 100) if len(filt) else 0.0,
        "block_reasons": dict(reasons),
        "config": {
            "transition_min_delta": args.transition_min_delta,
            "reversal_min_edge": args.reversal_min_edge,
            "rsi_overbought": args.rsi_overbought,
            "rsi_oversold": args.rsi_oversold,
            "counter_trend_block_total": args.counter_trend_block_total,
            "post_close_cooldown_secs": args.post_close_cooldown_secs,
            "signal_inter_cooldown_secs": args.signal_inter_cooldown_secs,
        },
    }
    out_summary.write_text(json.dumps(out_summary_data, indent=2, default=str))
    print(f"\n💾 Trades con flags: {out_trades}")
    print(f"💾 Summary post-filtro: {out_summary}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
