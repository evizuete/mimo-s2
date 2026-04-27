"""
main_rl_train.py  — VERSIÓN OPTIMIZADA

OPTIMIZACIONES VS VERSIÓN ANTERIOR:
─────────────────────────────────────────────────────────────────────────────
1. WARM_CACHE MODE (--warm_cache):
   Nuevo flag que precalienta el caché de predicciones sin ejecutar ningún
   backtest RL. Se llama UNA VEZ antes de Optuna desde main_rl_staged.py.
   Todos los trials posteriores leen el caché en disco en lugar de llamar
   al modelo Keras → ahorro ~5-15s de inferencia por trial.

2. CACHE_ONLY: si el caché ya existe para el rango de fechas solicitado,
   el script lo detecta y salta load_artifacts() y la query a BD.
   Solo carga los datos y ejecuta el backtest RL puro.

3. SKIP_DB_LOAD (--predictions_cache_path):
   Permite pasarle directamente la ruta del caché pre-generado y evitar
   la query a la BD. Útil en paralelo cuando múltiples trials comparten BD.

4. FILTRO DE VOLATILIDAD (--rl_vol_filter):
   Controla si DecisionPolicy permite operar en régimen de alta volatilidad.
   Modos disponibles:
     - none        : sin filtro (comportamiento anterior, allow_volatile=True)
     - skip_high   : bloquea cuando vol > vol_high_threshold del régimen
     - skip_extreme: bloquea cuando vol > vol_high_threshold * rl_vol_filter_mult
   El threshold de referencia (vol_high) lo inyecta RegimeConfig automáticamente.
   Optimizable desde main_rl_staged.py como parámetro categórico de Optuna.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import logging
import os
import warnings
from datetime import datetime

import numpy as np

from config.decision_policies_config import gate_by_action_and_state, score_cap_by_state, risk_mult_by_state
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo.strategies.trading_simulator import TradingSimulator, load_rl_policy_npz, save_rl_policy_npz

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('optuna').setLevel(logging.WARNING)


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def safe_market_condition(row) -> str:
    if hasattr(row, "state") and row.state is not None:
        return str(row.state)
    if hasattr(row, "market_condition") and row.market_condition is not None:
        return str(row.market_condition)
    if hasattr(row, "regime") and row.regime is not None:
        return str(row.regime)
    return ""


def collect_p_take_distribution_ultrafast(sim: TradingSimulator, df_raw, sample_size=10_000) -> np.ndarray:
    if len(df_raw) > sample_size:
        step = len(df_raw) // sample_size
        sample_indices = np.arange(0, len(df_raw), step)[:sample_size]
        df_sample = df_raw.iloc[sample_indices].copy()
    else:
        df_sample = df_raw.copy()

    df = sim.predict(df_sample, simulation=True)
    df = df.reset_index(drop=True)

    pred_long_raw  = df['pred_long_raw'].values
    pred_short_raw = df['pred_short_raw'].values
    pred_long_cal  = df['pred_long_cal'].values
    pred_short_cal = df['pred_short_cal'].values

    p_takes = []
    for local_idx in range(len(df)):
        if pred_long_cal[local_idx] < 0.2 and pred_short_cal[local_idx] < 0.2:
            continue
        row = df.iloc[local_idx]
        decision = sim.decision_engine.decide_at_bar(
            p_buy_raw  = float(pred_long_raw[local_idx]),
            p_sell_raw = float(pred_short_raw[local_idx]),
            p_buy_cal  = float(pred_long_cal[local_idx]),
            p_sell_cal = float(pred_short_cal[local_idx]),
            market_condition = safe_market_condition(row),
            o    = float(row.open),
            h    = float(row.high),
            l    = float(row.low),
            c    = float(row.close),
            atr  = float(row.atr)  if getattr(row, 'atr',  None) is not None else None,
            adx14= float(row.adx)  if getattr(row, 'adx',  None) is not None else None,
            trend_dir = getattr(row, 'trend_dir', None),
        )
        if decision.action == "none":
            continue
        s     = sim.rl_wrapper.build_state(row, decision)
        probs = sim.rl_wrapper.policy.probs(s)
        p_takes.append(float(probs[1]))

    return np.array(p_takes, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# OPTIMIZACIÓN: detección de caché de predicciones existente
# ─────────────────────────────────────────────────────────────────────────────

def _cache_key(release: str, from_date: datetime, to_date: datetime) -> str:
    """Genera una clave de caché determinista para un rango de fechas."""
    return f"{release}_{from_date.strftime('%Y%m%d')}_{to_date.strftime('%Y%m%d')}"


def _cache_exists(cache_dir: str, release: str, from_date: datetime, to_date: datetime) -> bool:
    """Verifica si el caché de predicciones para este rango ya existe."""
    key = _cache_key(release, from_date, to_date)
    # El TradingSimulator guarda el caché como parquet o pkl con la key como nombre
    # Ajusta la extensión si tu implementación usa otra
    for ext in ['.parquet', '.pkl', '.joblib', '.npz']:
        if os.path.exists(os.path.join(cache_dir, f"{key}{ext}")):
            return True
    # También buscamos el índice si existe
    if os.path.exists(os.path.join(cache_dir, f"{key}_index.json")):
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--from", dest="from_date", required=True)
    ap.add_argument("--to", dest="to_date", required=True)
    ap.add_argument("--artifacts_path", default="./artifacts")
    ap.add_argument("--policy_out", default=None)
    ap.add_argument("--policy_init", default=None)
    ap.add_argument('--rl_seed', type=int, default=None)

    ap.add_argument("--eval_only", action="store_true", default=False)
    ap.add_argument("--rl_diag", action="store_true", default=False)

    # ── modo de gates de decisión ──────────────────────────────────────────
    # 'training'   : gates más permisivos, usados durante Optuna y prod_train
    # 'production' : gates más estrictos, representativos del despliegue real
    # El forward check debe correr con 'production' para que sus métricas
    # sean comparables a lo que verá el sistema en producción.
    ap.add_argument("--rl_mode", type=str, default="training",
                    choices=["training", "production"],
                    help="Modo de gates de decisión. Usar 'production' en forward check.")

    # ── NUEVO: modo precalentamiento de caché ──────────────────────────────
    ap.add_argument("--warm_cache", action="store_true", default=False,
                    help="[OPTIMIZACIÓN] Solo genera el caché de predicciones y sale. "
                         "No ejecuta backtest RL. Llamar una vez antes de Optuna.")

    # RL knobs
    ap.add_argument("--rl_lr", type=float, default=5e-3)
    ap.add_argument("--rl_entropy", type=float, default=1e-3)
    ap.add_argument("--rl_baseline_beta", type=float, default=0.90)
    ap.add_argument("--rl_max_grad_norm", type=float, default=5.0)
    ap.add_argument("--rl_trade_cost", type=float, default=0.0075)
    ap.add_argument("--rl_batch", type=int, default=64)
    ap.add_argument("--rl_chop_soft_thr", type=float, default=0.60)
    ap.add_argument("--rl_exhaustion_soft_thr", type=float, default=0.60)
    ap.add_argument("--rl_chop_penalty_coef", type=float, default=0.08)
    ap.add_argument("--rl_exhaustion_penalty_coef", type=float, default=0.06)
    ap.add_argument('--rl_take_threshold', type=float, default=0.28)
    ap.add_argument('--rl_eval_threshold_scale', type=float, default=0.25,
                    help='Factor de escala aplicado a rl_take_threshold en modo eval_only. '
                         'Default 0.25 (25%% del valor Optuna). Ajustar si p(TAKE) median >> threshold.')
    ap.add_argument('--rl_train_threshold', type=float, default=0.05)

    # ── FILTRO DE VOLATILIDAD ──────────────────────────────────────────────
    # none        → sin filtro (allow_volatile=True en DecisionPolicy)
    # skip_high   → bloquea entradas cuando vol > vol_high del régimen
    # skip_extreme→ bloquea entradas cuando vol > vol_high * rl_vol_filter_mult
    # Optimizable desde main_rl_staged.py como categórico en PARAM_SPACE.
    ap.add_argument("--rl_vol_filter", type=str, default="none",
                    choices=["none", "skip_high", "skip_extreme"],
                    help="Filtro de volatilidad para DecisionPolicy. "
                         "none=sin filtro, skip_high=bloquea vol>vol_high, "
                         "skip_extreme=bloquea vol>vol_high*rl_vol_filter_mult")
    ap.add_argument("--rl_vol_filter_mult", type=float, default=1.5,
                    help="Multiplicador sobre vol_high para el modo skip_extreme. "
                         "Default 1.5 (bloquea cuando vol supera 1.5x el umbral alto).")

    # Backtest / risk knobs
    ap.add_argument("--initial_equity", type=float, default=10_000.0)
    ap.add_argument("--spread_price", type=float, default=0.07)
    ap.add_argument("--max_daily_loss_pct", type=float, default=0.0350)
    ap.add_argument("--sizing_equity_mode", default="balance")
    ap.add_argument("--compound", action="store_true", default=True)

    # OOF/model knobs
    ap.add_argument("--use_oof", action="store_true", default=True)
    ap.add_argument("--oof_splits", type=int, default=3)
    ap.add_argument("--oof_epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--seq_len_short", type=int, default=64)
    ap.add_argument("--seq_len_long", type=int, default=256)

    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--debug_max_rows", type=int, default=200_000)

    args = ap.parse_args()

    from_date = parse_date(args.from_date)
    to_date   = parse_date(args.to_date)

    artifacts_path = args.artifacts_path
    ensure_dir(artifacts_path)

    # ── OPTIMIZACIÓN: path de caché compartido por release ─────────────────
    cache_dir = f'./cache/{args.release}/predictions'
    ensure_dir(cache_dir)

    policy_out = args.policy_out or os.path.join(artifacts_path, f"rl_policy_gate_{args.release}.npz")

    if args.eval_only and not args.warm_cache:
        if not args.policy_init:
            raise ValueError("--eval_only requiere --policy_init")
        if not os.path.exists(args.policy_init):
            raise FileNotFoundError(f"--policy_init no existe: {args.policy_init}")
        print(f"[EVAL_ONLY] policy_init verificado: {args.policy_init}")

    # -----------------------
    # Configs
    # -----------------------
    general_config = Config(
        release=str(args.release),
        use_oof=bool(args.use_oof),
        oof_splits=int(args.oof_splits),
        oof_epochs=int(args.oof_epochs),
        save_oof_artifacts=True,
    )

    model_config = ModelConfig(
        seq_len_short=int(args.seq_len_short),
        seq_len_long=int(args.seq_len_long),
        batch_size=int(args.batch_size),
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        label_method="adaptative",
        feature_masks={
            'long':  {'ema_bull': True,  'rsi_oversold': True,   'macd_positive': True,
                      'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False},
            'short': {'ema_bear': True,  'rsi_overbought': True,  'macd_negative': True,
                      'ema_bull': False, 'rsi_oversold': False,   'macd_positive': False},
        },
    )

    regime_config = RegimeConfig(adx_trend_threshold=25.0)

    mode = args.rl_mode  # 'training' (default/Optuna) o 'production' (forward check)

    # ── FILTRO DE VOLATILIDAD ──────────────────────────────────────────────
    # Determina allow_volatile para DecisionPolicy según --rl_vol_filter.
    # - "none"        → allow_volatile=True  (sin restricción)
    # - "skip_high"   → allow_volatile=False (usa el umbral vol_high del régimen)
    # - "skip_extreme"→ allow_volatile=False (umbral = vol_high * rl_vol_filter_mult,
    #                   pasado como vol_filter_threshold a DecisionPolicy si lo acepta;
    #                   en caso contrario actúa igual que skip_high como fallback seguro)
    _vol_filter      = getattr(args, 'rl_vol_filter', 'none')
    _vol_filter_mult = float(getattr(args, 'rl_vol_filter_mult', 1.5))
    _allow_volatile  = (_vol_filter == 'none')

    # Argumentos opcionales para DecisionPolicy — solo se pasan si el modo
    # lo requiere, para no romper versiones anteriores de la clase.
    _dp_extra: dict = {}
    if _vol_filter == 'skip_extreme':
        _dp_extra['vol_filter_multiplier'] = _vol_filter_mult
    if _vol_filter != 'none':
        _dp_extra['vol_filter_mode'] = _vol_filter

    if args.debug and _vol_filter != 'none':
        print(f"[VOL_FILTER] modo={_vol_filter}  allow_volatile={_allow_volatile}"
              + (f"  mult={_vol_filter_mult}" if _vol_filter == 'skip_extreme' else ""))

    decision_policy = DecisionPolicy(
        gate_by_action_and_state=gate_by_action_and_state[mode],
        score_cap_by_state=score_cap_by_state[mode],
        risk_mult_by_state=risk_mult_by_state[mode],
        score_low_quantile=75,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.15,
        allow_volatile=_allow_volatile,
        **_dp_extra,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.005,
        min_score_to_trade=0.10,
        max_risk_pct=0.02,
        max_positions=1,
    )

    rl_config = {
        "lr": float(args.rl_lr),
        "entropy_coef": float(args.rl_entropy),
        "baseline_beta": float(args.rl_baseline_beta),
        "max_grad_norm": float(args.rl_max_grad_norm),
        "trade_cost_money": float(args.rl_trade_cost),
        "batch_size": int(args.rl_batch),
        "chop_soft_thr": float(args.rl_chop_soft_thr),
        "exhaustion_soft_thr": float(args.rl_exhaustion_soft_thr),
        "chop_penalty_coef": float(args.rl_chop_penalty_coef),
        "exhaustion_penalty_coef": float(args.rl_exhaustion_penalty_coef),
        'seed': args.rl_seed if args.rl_seed is not None else 42,
        'update_frequency': int(args.rl_batch),
        'rl_take_threshold': float(args.rl_take_threshold),
        'rl_train_threshold': float(args.rl_train_threshold),
    }

    rl_config_diag = {
        "lr": 1e-4, "entropy_coef": 1e-4, "baseline_beta": 0.85,
        "max_grad_norm": 10.0, "trade_cost_money": 0.5, "batch_size": 256,
        "chop_soft_thr": 1.0, "exhaustion_soft_thr": 1.0,
        "chop_penalty_coef": 0.0, "exhaustion_penalty_coef": 0.0,
    }

    rl_cfg = rl_config_diag if args.rl_diag else rl_config
    if args.rl_seed is not None:
        rl_cfg['seed'] = args.rl_seed

    # -----------------------
    # Calibración del threshold para eval
    # -----------------------
    # El rl_take_threshold de Optuna se optimiza sobre distribuciones estocásticas de
    # training. En eval determinístico la policy colapsa (entropía baja -> p_take << 0.5),
    # por lo que aplicar ese mismo threshold produce 0 señales.
    # Escalamos al 25% del valor de Optuna como punto de partida conservador.
    # Se puede ajustar con --rl_eval_threshold_scale (default 0.25).
    if args.eval_only:
        eval_threshold_scale = getattr(args, 'rl_eval_threshold_scale', 0.25)
        effective_take_threshold = float(args.rl_take_threshold) * float(eval_threshold_scale)
        if args.debug:
            print(f"[EVAL_ONLY] rl_take_threshold escalado: "
                  f"{args.rl_take_threshold:.4f} x {eval_threshold_scale} = {effective_take_threshold:.4f}")
    else:
        effective_take_threshold = float(args.rl_take_threshold)

    # -----------------------
    # Simulator
    # -----------------------
    sim = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=True,
        rl_config=rl_cfg,
        rl_train=(not args.eval_only and not args.warm_cache),
        rl_eval_deterministic=args.eval_only,
        rl_take_threshold=effective_take_threshold,
        rl_train_threshold=float(args.rl_train_threshold),
        rl_policy_path=args.policy_init if args.policy_init else policy_out,
        spread_price=float(args.spread_price),
        mtm_use_bid_ask=True,
        mtm_price_col="close",
        sizing_equity_mode=str(args.sizing_equity_mode),
        max_daily_loss_pct=float(args.max_daily_loss_pct),
        max_daily_profit_pct=None,
        compound=bool(args.compound),
        use_prediction_cache=True,
        cache_dir=cache_dir,
    )

    # Cargar policy si procede (no en warm_cache)
    if args.policy_init and not args.warm_cache:
        ok = load_rl_policy_npz(sim.rl_wrapper, args.policy_init)
        print(f"[policy_init] loaded={ok} from {args.policy_init}")
        if not ok:
            raise RuntimeError(f"Falló al cargar policy desde: {args.policy_init}")
        if sim.rl_wrapper and sim.rl_wrapper.policy:
            policy = sim.rl_wrapper.policy
            print(f"[policy_init] W_sum={policy.W.sum() if hasattr(policy, 'W') else 0:.6f}  "
                  f"b_sum={policy.b.sum() if hasattr(policy, 'b') else 0:.6f}  "
                  f"baseline={policy.baseline if hasattr(policy, 'baseline') else 0:.6f}")
        else:
            raise RuntimeError("sim.rl_wrapper.policy es None después de cargar")

    # Cargar artifacts (modelo Keras, scalers, calibradores)
    sim.load_artifacts()

    # Cargar datos históricos
    df = sim.helper.load_from_database_historical(from_date=from_date, to_date=to_date)

    # ── OPTIMIZACIÓN: modo warm_cache ─────────────────────────────────────────
    # Solo genera el caché de predicciones sin ejecutar RL.
    # Lanza sim.predict() sobre todo el df y el TradingSimulator lo persiste.
    if args.warm_cache:
        print(f"[WARM_CACHE] Generando caché de predicciones para {from_date.date()} → {to_date.date()}...")
        _ = sim.predict(df, simulation=True)
        print(f"[WARM_CACHE] ✅ Caché generado en {cache_dir}")
        print("=== WARM_CACHE DONE ===")
        return
    # ─────────────────────────────────────────────────────────────────────────

    # Ejecutar backtest
    mode_str = "EVAL_ONLY" if args.eval_only else "TRAIN"
    print(f"Starting backtesting... mode={mode_str}")
    res = sim.backtest(df, initial_equity=float(args.initial_equity))

    if not args.eval_only:
        save_rl_policy_npz(sim.rl_wrapper, policy_out)

    # Resumen
    pnl      = res.get("net_pnl", 0.0)
    n_trades = res.get("n_trades", 0)
    summ     = res.get("summary", {}) or {}
    dd       = (res.get("drawdown_stats", {}) or {}).get("max_drawdown", None)
    dd_pct   = (res.get("drawdown_stats", {}) or {}).get("max_drawdown_pct", None)
    pf       = summ.get("profit_factor", None)
    wr       = summ.get("win_rate", None)

    print("=== RL RUN DONE ===")
    print(f"Mode       : {mode_str}")
    if not args.eval_only:
        print(f"Policy saved: {policy_out}")
    else:
        print(f"Policy used : {args.policy_init}")
    print(f"Period     : {from_date.date()} -> {to_date.date()}")
    print(f"VolFilter  : {_vol_filter}" + (f" (x{_vol_filter_mult})" if _vol_filter == 'skip_extreme' else ""))
    print(f"Trades     : {n_trades}")
    print(f"NetPnL     : {pnl:.2f}")
    if pf  is not None: print(f"ProfitFactor: {pf}")
    if wr  is not None: print(f"WinRate    : {wr}")
    if dd  is not None: print(f"MaxDD      : {dd}")
    if dd_pct is not None: print(f"MaxDD (%)  : {dd_pct:.4f}")

    if args.debug:
        reload_path = policy_out if (not args.eval_only) else args.policy_init
        ok = load_rl_policy_npz(sim.rl_wrapper, reload_path) if reload_path else False
        if not ok:
            print("WARNING: no se pudo cargar la policy para debug_ptake.")
        else:
            p_takes = collect_p_take_distribution_ultrafast(sim, df)
            if len(p_takes) == 0:
                print("No hay candidatos (action != none) en el muestreo.")
            else:
                print("\n== p(TAKE) distribution (sample) ==")
                print(f"n candidates: {len(p_takes)}")
                print(f"min/max    : {p_takes.min():.4f} / {p_takes.max():.4f}")
                qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
                print(f"percentiles: {dict(zip(qs, np.percentile(p_takes, qs)))}")
                print(f"mean/std   : {p_takes.mean():.4f} / {p_takes.std():.4f}")


if __name__ == "__main__":
    main()