#!/usr/bin/env python3
"""
replay_s2_202500.py — Replay histórico del setup live de S2 sobre 2 meses
de OHLCV de la BD, sin necesitar paper trading wallclock.

Idea:
  Construir EXACTAMENTE el mismo TradingSimulator que `main/s2_main.py` levanta
  en producción (mismo deploy_dir, misma feature_config, mismo RiskConfig,
  mismas gates por régimen) y correr `simulator.backtest()` sobre el rango
  histórico que indiques. Esto te ahorra semanas de wallclock y produce
  métricas comparables a las que obtendrías en paper.

Diferencias respecto a paper trading real:
  - NO se simula slippage variable ni latencia ZMQ.
  - El spread se modela con `spread_price` (constante, igual que en s2_main).
  - No hay daily_halt por equity sino el del backtest.

Uso:
  python -m scripts.replay_s2_202500 \
    --release 202500 \
    --deploy-subdir deploy_full \
    --from 2026-03-01 --to 2026-05-01 \
    --initial-equity 10000 \
    --policy-config decision_policies_config_202500 \
    --out /tmp/replay_202500
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import ModelConfig, Config
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo.strategies.trading_simulator_v3 import TradingSimulator


# ─────────────────────────────────────────────────────────────────────────────
# Configuración del simulador (clon exacto de main/s2_main.py para release 202500)
# ─────────────────────────────────────────────────────────────────────────────

def build_simulator(
    release: str,
    deploy_subdir: str,
    policy_module: str,
    base_dir: Path,
    artifacts_root: Optional[Path] = None,
) -> TradingSimulator:
    """Construye el TradingSimulator con la misma config que s2_main.py."""

    # 1. Importar la policy del módulo elegido (con stub de 202500 ya generado)
    pol = importlib.import_module(policy_module)
    gate_by_action_and_state = pol.gate_by_action_and_state
    score_cap_by_state = pol.score_cap_by_state
    risk_mult_by_state = pol.risk_mult_by_state

    # 1.5. Detectar si el deploy es multitask: si existe `model_<release>_multitask.keras`,
    # el modelo único maneja ambas heads y necesita el contexto UNIÓN (25 cols).
    # En binary cada side tiene su propio modelo entrenado con su máscara (24 cols).
    if artifacts_root is None:
        _artifacts_root_for_detect = base_dir.parent / "artifacts"
    else:
        _artifacts_root_for_detect = artifacts_root
    _deploy_dir = _artifacts_root_for_detect / release / "oof" / deploy_subdir
    _multitask_keras = _deploy_dir / f"model_{release}_multitask.keras"
    is_multitask_deploy = _multitask_keras.exists()
    is_specialists_merged = False
    # merge_specialists produce un dir con archivos *_long.keras / *_short.keras
    # que internamente son modelos multitask renombrados; se detecta por el
    # meta.json con source_long_dir/source_short_dir.
    if not is_multitask_deploy:
        # merge_specialists escribe `merge_specialists_meta.json` con claves
        # `long_specialist` y `short_specialist` apuntando a los dirs origen.
        for _meta_name in ("merge_specialists_meta.json", "meta.json"):
            _meta_path = _deploy_dir / _meta_name
            if not _meta_path.exists():
                continue
            try:
                import json as _json
                with _meta_path.open("r", encoding="utf-8") as _f:
                    _meta = _json.load(_f)
                if (
                    ("long_specialist" in _meta and "short_specialist" in _meta)
                    or ("source_long_dir" in _meta and "source_short_dir" in _meta)
                ):
                    is_multitask_deploy = True
                    is_specialists_merged = True
                    print(f"🧠 Deploy specialists-merged detectado ({_meta_name}); usando máscaras UNIÓN.")
                    break
            except Exception as _e:
                print(f"⚠️  Error leyendo {_meta_name}: {_e}")

    if not is_multitask_deploy and not is_specialists_merged:
        # Fallback final: inspeccionar el shape del context scaler.
        # 25 cols → modelo multitask renombrado. 24 cols → binary single-side.
        try:
            import joblib as _joblib
            import glob as _glob
            _scaler_dir = _deploy_dir / f"scalers_{release}"
            _candidates = list(_scaler_dir.glob("*context*.pkl")) + list(_scaler_dir.glob("*context*.joblib"))
            for _ctx_scaler_path in _candidates:
                _ctx_scaler = _joblib.load(_ctx_scaler_path)
                _n = getattr(_ctx_scaler, "n_features_in_", None)
                if _n is None:
                    _n = getattr(_ctx_scaler, "n_features", None)
                if _n is not None and int(_n) >= 25:
                    is_multitask_deploy = True
                    is_specialists_merged = True
                    print(f"🧠 Deploy specialists-merged inferido por scaler {_ctx_scaler_path.name} (n_features={_n}); máscaras UNIÓN.")
                    break
        except Exception as _e:
            print(f"⚠️  Error inspeccionando context scaler: {_e}")

    if is_specialists_merged:
        pass  # ya se imprimió arriba
    elif is_multitask_deploy:
        print(f"🧠 Deploy multitask detectado ({_multitask_keras.name}); usando máscaras UNIÓN.")
    else:
        print(f"🧠 Deploy binary (no se encontró {_multitask_keras.name} ni meta de specialists ni scaler 25-col); máscaras por-side.")

    # Las direccionales se invierten en multitask: ambos sides ven todas las features.
    if is_multitask_deploy:
        _mask_long = {
            "ema_bull": True, "rsi_oversold": True, "macd_positive": True,
            "ema_bear": True, "rsi_overbought": True, "macd_negative": True,
        }
        _mask_short = {
            "ema_bull": True, "rsi_oversold": True, "macd_positive": True,
            "ema_bear": True, "rsi_overbought": True, "macd_negative": True,
        }
    else:
        _mask_long = {
            "ema_bull": True,  "rsi_oversold": True,  "macd_positive": True,
            "ema_bear": False, "rsi_overbought": False, "macd_negative": False,
        }
        _mask_short = {
            "ema_bear": True,  "rsi_overbought": True, "macd_negative": True,
            "ema_bull": False, "rsi_oversold": False,  "macd_positive": False,
        }

    # 2. Configs alineadas con el run de v6 deploy de 202500
    general_config = Config(
        release=release,
        oof_splits=5,
        oof_epochs=120,
    )

    model_config = ModelConfig(
        seq_len_short=24,
        seq_len_long=96,
        epochs=90,
        patience=12,
        use_hierarchical_fusion=True,
        ranking_loss_weight=0.0,
        target_type="multitask",
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier",
        label_horizon=3,
        tp_barrier=2.0,
        sl_barrier=0.8,
        label_method_long="triple_barrier",
        regime_barriers_long={
            "trending": {"tp": 2.0, "sl": 0.8},
            "ranging":  {"tp": 1.8, "sl": 0.8},
            "low_vol":  {"tp": 1.8, "sl": 0.8},
            "high_vol": {"tp": 2.2, "sl": 1.0},
        },
        label_method_short="triple_barrier",
        regime_barriers_short={
            "trending": {"tp": 2.0, "sl": 0.8},
            "ranging":  {"tp": 1.8, "sl": 0.8},
            "low_vol":  {"tp": 1.8, "sl": 0.8},
            "high_vol": {"tp": 2.2, "sl": 1.0},
        },
        tp_barrier_short=None,
        sl_barrier_short=None,
        use_vol_invariant_features=True,   # release 202500
        use_reduced_features=True,          # release 202500
        feature_masks={
            "long": _mask_long,
            "short": _mask_short,
        },
    )

    regime_config = RegimeConfig(adx_trend_threshold=25.0)

    decision_policy = DecisionPolicy(
        gate_by_action_and_state=gate_by_action_and_state["production"],
        score_cap_by_state=score_cap_by_state["production"],
        risk_mult_by_state=risk_mult_by_state["production"],
        score_low_quantile=80,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.20,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.0035,
        min_score_to_trade=0.0,
        max_risk_pct=0.02,
        max_positions=2,
    )

    if artifacts_root is None:
        artifacts_root = base_dir.parent / "artifacts"
    artifacts_path = str(
        (artifacts_root / release / "oof" / deploy_subdir).resolve()
    )

    print(f"📂 artifacts_path : {artifacts_path}")
    print(f"📂 policy_module  : {policy_module}")

    simulator = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=False,
        rl_config=None,
        rl_train=False,
        rl_eval_deterministic=True,
        rl_policy_path=None,
        spread_price=0.07,
        mtm_use_bid_ask=True,
        mtm_price_col="close",
        sizing_equity_mode="balance",
        max_daily_loss_pct=0.035,
        max_daily_profit_pct=None,
        compound=True,
        enable_live_scaler_updates=False,   # ← desactivado en replay (consistencia)
        anomaly_block_threshold=1.2,
        signal_cooldown_bars=3,
    )
    return simulator


# ─────────────────────────────────────────────────────────────────────────────
# Carga OHLCV histórico
# ─────────────────────────────────────────────────────────────────────────────

def load_ohlcv(from_date: str, to_date: str, base_tf: str = "1min") -> pd.DataFrame:
    print(f"\n📂 Cargando OHLCV desde BD: {from_date} → {to_date}  (tf={base_tf})")
    db = Database()
    resample_arg = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=from_date, to_date=to_date, resample=resample_arg
    )
    df = dm.df
    df["time"] = pd.to_datetime(df["time"])
    df = (
        df.dropna(subset=["close"])
        .sort_values("time")
        .drop_duplicates(subset="time", keep="first")
        .reset_index(drop=True)
    )
    print(f"   {len(df):,} filas | rango {df['time'].iloc[0]} → {df['time'].iloc[-1]}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Resumen de métricas
# ─────────────────────────────────────────────────────────────────────────────

def summarize(
    result: Dict[str, Any],
    initial_equity: float,
    out_dir: Path,
    eval_from: Optional[str] = None,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    trades = pd.DataFrame(result.get("trades", []))
    eq_curve_raw = result.get("equity_curve_mtm", None)
    eq_ts_raw = result.get("equity_timestamps", None)
    eq_curve = list(eq_curve_raw) if eq_curve_raw is not None and len(eq_curve_raw) > 0 else []
    eq_ts = list(eq_ts_raw) if eq_ts_raw is not None and len(eq_ts_raw) > 0 else []

    # Filtrado por ventana de evaluación: las barras anteriores a `eval_from`
    # son warmup (popular features) y no cuentan para PnL/equity reportados.
    if eval_from is not None:
        eval_from_ts = pd.to_datetime(eval_from)
        if eq_curve and eq_ts:
            eq_ts_dt = pd.to_datetime(eq_ts)
            mask_eq = eq_ts_dt >= eval_from_ts
            n_pre = int((~mask_eq).sum())
            if n_pre > 0 and mask_eq.any():
                # Rebase: el equity al inicio de la ventana de eval pasa a ser
                # el "initial_equity" reportado, para que pnl_pct refleje SOLO
                # la ventana evaluable.
                pre_eq = [e for e, m in zip(eq_curve, mask_eq) if not m]
                rebased_initial = float(pre_eq[-1]) if pre_eq else initial_equity
                eq_curve = [e for e, m in zip(eq_curve, mask_eq) if m]
                eq_ts = [t for t, m in zip(eq_ts, mask_eq) if m]
                print(f"\nℹ️  Warmup: descartadas {n_pre} barras de equity "
                      f"anteriores a {eval_from}.")
                print(f"   equity rebased: {initial_equity:,.2f} → {rebased_initial:,.2f} "
                      f"(equity al cierre del warmup)")
                initial_equity = rebased_initial

        if not trades.empty:
            entry_col = next(
                (c for c in ("entry_time", "open_time", "ts_open", "time") if c in trades.columns),
                None,
            )
            if entry_col is not None:
                trades[entry_col] = pd.to_datetime(trades[entry_col])
                n_trades_pre = len(trades)
                trades = trades[trades[entry_col] >= eval_from_ts].reset_index(drop=True)
                n_dropped = n_trades_pre - len(trades)
                if n_dropped > 0:
                    print(f"   trades descartados (warmup): {n_dropped} "
                          f"(quedan {len(trades)} en la ventana de eval)")

    final_eq = float(eq_curve[-1]) if eq_curve else initial_equity

    print("\n" + "═" * 78)
    print("  RESUMEN DEL REPLAY")
    print("═" * 78)
    print(f"  initial_equity        : {initial_equity:>12,.2f}")
    print(f"  final_equity          : {final_eq:>12,.2f}")
    print(f"  PnL total             : {final_eq - initial_equity:>+12,.2f}  "
          f"({(final_eq/initial_equity - 1)*100:+.2f}%)")

    if not trades.empty:
        n_trades = len(trades)
        n_long = int((trades["side"] == "long").sum()) if "side" in trades.columns else 0
        n_short = int((trades["side"] == "short").sum()) if "side" in trades.columns else 0
        wins = int((trades.get("pnl", 0) > 0).sum())
        losses = int((trades.get("pnl", 0) <= 0).sum())
        avg_pnl = float(trades["pnl"].mean()) if "pnl" in trades.columns else float("nan")
        med_pnl = float(trades["pnl"].median()) if "pnl" in trades.columns else float("nan")

        print(f"\n  trades                : {n_trades}  "
              f"(long={n_long}, short={n_short})")
        print(f"  win_rate              : {wins}/{n_trades} = "
              f"{100*wins/max(n_trades,1):.1f}%")
        print(f"  avg PnL/trade         : {avg_pnl:>+12,.2f}")
        print(f"  median PnL/trade      : {med_pnl:>+12,.2f}")

        if "r_multiple" in trades.columns:
            ev_net = float(trades["r_multiple"].mean())
            print(f"  EV_net medio          : {ev_net:>+12.4f}R/trade")
            total_R = float(trades["r_multiple"].sum())
            print(f"  R total               : {total_R:>+12.2f}R")

    if eq_curve and eq_ts:
        eq = pd.Series(eq_curve, index=pd.to_datetime(eq_ts))
        peak = eq.cummax()
        dd = (eq - peak) / peak
        max_dd_pct = float(dd.min() * 100)
        print(f"  max drawdown          : {max_dd_pct:>+12.2f}%")

        weekly = eq.resample("W-MON").last().dropna()
        weekly_pct = weekly.pct_change().dropna()
        if not weekly_pct.empty:
            pos_weeks = int((weekly_pct > 0).sum())
            tot_weeks = len(weekly_pct)
            print(f"  weeks positive        : {pos_weeks}/{tot_weeks} "
                  f"({100*pos_weeks/max(tot_weeks,1):.0f}%)")
            print(f"  best week             : {float(weekly_pct.max())*100:+.2f}%")
            print(f"  worst week            : {float(weekly_pct.min())*100:+.2f}%")

    # Persistir
    if not trades.empty:
        trades_path = out_dir / "trades.parquet"
        trades.to_parquet(trades_path, index=False)
        print(f"\n📁 trades  → {trades_path}")
    if eq_curve:
        eq_df = pd.DataFrame({"time": pd.to_datetime(eq_ts), "equity": eq_curve})
        eq_path = out_dir / "equity_curve.parquet"
        eq_df.to_parquet(eq_path, index=False)
        print(f"📁 equity → {eq_path}")

    max_dd_pct_val = 0.0
    if eq_curve and eq_ts:
        _eq = pd.Series(eq_curve, index=pd.to_datetime(eq_ts))
        _peak = _eq.cummax()
        _dd = (_eq - _peak) / _peak
        max_dd_pct_val = float(_dd.min() * 100)

    summary = {
        "initial_equity": initial_equity,
        "final_equity": final_eq,
        "pnl_total": final_eq - initial_equity,
        "pnl_pct": (final_eq / initial_equity - 1) * 100,
        "max_drawdown_pct": max_dd_pct_val,
        "n_trades": int(len(trades)),
        "win_rate_pct": 100 * float((trades.get("pnl", pd.Series([])) > 0).mean())
            if not trades.empty else 0.0,
    }
    if "r_multiple" in trades.columns and not trades.empty:
        summary["ev_net_avg_R"] = float(trades["r_multiple"].mean())
        summary["total_R"] = float(trades["r_multiple"].sum())
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"📁 summary → {summary_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--deploy-subdir", default="deploy_full",
                    help="Subdir bajo artifacts/<release>/oof/ donde están "
                         "los modelos y scalers (default: deploy_full).")
    ap.add_argument("--from", dest="from_date", required=True,
                    help="YYYY-MM-DD inicio de la ventana evaluable.")
    ap.add_argument("--to", dest="to_date", required=True,
                    help="YYYY-MM-DD fin.")
    ap.add_argument("--warmup-from", dest="warmup_from", default=None,
                    help="YYYY-MM-DD opcional. Si se pasa, se carga OHLCV "
                         "desde aquí pero las barras anteriores a --from "
                         "se usan SOLO para popular features multi-TF y NO "
                         "cuentan en PnL/equity/trades reportados. "
                         "Default: igual que --from (sin warmup).")
    ap.add_argument("--initial-equity", type=float, default=10_000.0)
    ap.add_argument("--base-tf", default="1min",
                    help="Timeframe del OHLCV cargado de la BD. El feature "
                         "builder espera 1min y resamplea internamente a "
                         "5m/15m/1h.")
    ap.add_argument("--policy-config", default="decision_policies_config_202500",
                    help="Módulo Python con gate_by_action_and_state "
                         "(default: decision_policies_config_202500). "
                         "Debe existir en config/.")
    ap.add_argument("--out", default="/tmp/replay_s2",
                    help="Dir donde persistir trades, equity y resumen.")
    args = ap.parse_args()

    base_dir = Path(__file__).resolve().parent
    repo_root = base_dir.parent

    # Asegurar que config/ es importable como módulo:
    config_dir = repo_root / "config"
    if config_dir.exists() and str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 1. Construir simulador
    print("\n🔧 Construyendo TradingSimulator...")
    simulator = build_simulator(
        release=args.release,
        deploy_subdir=args.deploy_subdir,
        policy_module=f"config.{args.policy_config}",
        base_dir=base_dir,
        artifacts_root=repo_root / "artifacts",
    )

    # 2. Cargar OHLCV
    load_from = args.warmup_from if args.warmup_from else args.from_date
    if args.warmup_from:
        print(f"\n🔥 Warmup activo: cargando desde {load_from} pero solo "
              f"contabilizando trades desde {args.from_date}.")
    df = load_ohlcv(load_from, args.to_date, args.base_tf)
    if len(df) < 8000:
        print(f"⚠️  Solo {len(df)} filas. Multi-TF features (1h) "
              "necesitan ≥7500 1m bars. Pasa un rango más amplio.")

    # 3. Backtest
    print("\n🏁 Ejecutando simulator.backtest()...")
    print("   (esto ejecuta predict() sobre todo el rango + decisión per-bar)")
    result = simulator.backtest(
        df_rates=df,
        initial_equity=float(args.initial_equity),
        df_is_predicted=False,
    )

    # 4. Resumen
    out_dir = Path(args.out)
    summarize(
        result,
        initial_equity=float(args.initial_equity),
        out_dir=out_dir,
        eval_from=args.from_date if args.warmup_from else None,
    )

    print("\n" + "═" * 78)
    print("✅ Replay completado.")
    print("═" * 78)
    print(f"\n📊 Compara con tus expectativas de holdout 202500:")
    print(f"   • EV_net LONG  esperado: +0.05R … +0.20R/sig "
          f"(fue +0.38R en tail)")
    print(f"   • EV_net SHORT esperado: +0.03R … +0.15R/sig "
          f"(fue +0.15R en tail)")
    print(f"   • Si EV_net replay ≥ +0.05R/sig en cada side → señal de viabilidad")
    print(f"     para mini-producción con tamaño 0.25× (base_risk_pct=0.0008).")


if __name__ == "__main__":
    main()
