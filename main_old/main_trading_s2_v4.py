import json
import os
import time
from typing import Dict, Any, Optional

import pandas as pd
import zmq

from config.trading_policies_config import trading_policies
from mimo_old.databases import Database
from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import ModelConfig, Config
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.trading_simulator import TradingSimulator, load_rl_policy_npz

def price_to_points(price_a: float, price_b: float, point: float = 0.01) -> int:
    return int(round(abs(price_a - price_b) / point))

def send_order(push_socket, payload: Dict[str, Any],):
    push_socket.send_json(payload)
    print(f"\t[ZMQ->S3] ORDER_REQUEST sent: {payload}\n")

def create_order_request(
        order: Dict[str, Any],
        emergency_atr_mult: float = 4.0,
        be_trigger_ratio: float = 0.40,
        be_offset_ratio: float = 0.02,
        trail_sl_ratio: float = 1.0,  # trailing como ratio del virtual_sl_points (1.0 = 1R)
        trail_step_ratio: float = 0.15,  # paso mínimo como ratio del virtual_sl_points
        max_hold_seconds: int = 900,
        close_fraction: float = 0.50,
        partial_trigger_pct: float = 60.0,
        point: float = 0.01,
        comment: str = '',
) -> Dict[str, Any]:
    entry = float(order['entry'])
    virtual_tp_price = float(order['tp'])
    atr = float(order['atr_at_entry'])
    side_str = order['side']
    qty = float(order['qty'])

    mt5_side = 'BUY' if side_str == 'long' else 'SELL'

    # Calculamos los dos niveles de SL
    # 1. SL de emergencias - Va al broker
    emergency_sl_distance = emergency_atr_mult * atr
    if side_str == 'long':
        emergency_sl_price = entry - emergency_sl_distance
    else:
        emergency_sl_price = entry + emergency_sl_distance

    emergency_points = price_to_points(entry, emergency_sl_price, point)

    # 2. SL Virtual (señal). Para calcular R y ratios
    virtual_sl_price = order['sl']
    virtual_sl_points = price_to_points(entry, virtual_sl_price, point)

    # Garantizar que emergency_points > virtual_sl_points (es un paracaídas, no un SL normal)
    if emergency_points <= virtual_sl_points:
        emergency_points = int(virtual_sl_points * 1.5)  # mínimo 50% más amplio

    # Calcular ratios basados en virtual_sl (no en emergency_sl)
    # Distancia al TP en puntos
    tp_points = price_to_points(entry, virtual_tp_price, point)

    # Break-even: activar al X% del camino hacia TP
    be_trigger_points = int(round(be_trigger_ratio * tp_points))

    # Break-even offset: donde se coloca el SL virtual al activar BE
    be_offset_points = int(round(be_offset_ratio * tp_points))

    # Trailing: proporcional al riesgo inicial (virtual_sl_points = 1R)
    # trail_sl_ratio=1.0 → trailing de 1R desde el máximo favorable
    # Así se adapta automáticamente a cada señal sin importar la magnitud
    trail_points = max(20, int(round(trail_sl_ratio * virtual_sl_points)))

    # Paso mínimo para actualizar trailing: ratio del mismo riesgo
    trail_step_points = max(10, int(round(trail_step_ratio * virtual_sl_points)))

    # Construir solicitud
    request = {
        'action': 'OPEN',
        'symbol': 'XAUUSD.r',
        'side': mt5_side,
        'volume': qty,
        'magic': int(release),
        'comment': comment,
        'deviation': 10,

        # TP virtual (NO se envía al broker, solo gestión interna)
        'virtual_tp': virtual_tp_price,
        'virtual_sl': virtual_sl_price,

        # Metadatos para logging / analisis
        'metadata': {
            'atr': atr,
            'entry': entry,
            'emergency_sl_price': emergency_sl_price,
            'emergency_sl_points': emergency_points,
            'virtual_sl_price': virtual_sl_price,
            'virtual_sl_points': virtual_sl_points,
            'tp_points': tp_points,
            'risk_R': virtual_sl_points,
            'trail_points': trail_points,
            'trail_step_points': trail_step_points,
            'be_trigger_points': be_trigger_points,
            'be_offset_points': be_offset_points,
        },

        'risk': {
            'profile': 'scalping',
            'emergency_points': emergency_points,  # sobreescribe el fijo del perfil con el valor ATR-based
            'trail_points': trail_points,  # 1R del virtual_sl → proporcional al riesgo
            'trail_step_points': trail_step_points,  # 0.15R → paso mínimo para mover trailing
            'be_trigger_points': be_trigger_points,  # 40% del TP
            'be_offset_points': be_offset_points,  # 2% del TP
            'close_confirmation': {
                'exit_mode': 'hard',
                'tp_mode': 'touch',  # no dejes escapar el TP
                'tp_confirm_count': 0,
                'sl_mode': 'close',  # espera cierre de vela M1 antes de ejecutar SL
                'sl_confirm_count': 1 if live_order.get('state') not in ('TREND_UP', 'TREND_DOWN') else 0,  # una vela confirmando cierre mas alla del nivel
            }
        }
    }

    return request

def trading_engine(release: str, trading_policy: str = 'aggressive'):
    general_config = Config(
        release=release,
        oof_splits=3,
        oof_epochs=60
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        batch_size=8192
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        tp_barrier=2.5,
        sl_barrier=1.0,
        label_method='adaptive',
        feature_masks={
            'long': {'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True,
                     'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False},
            'short': {'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True,
                      'ema_bull': False, 'rsi_oversold': False, 'macd_positive': False},
        },
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    decision_policy = DecisionPolicy(
        gate_by_action_and_state=trading_policies[trading_policy],
        score_cap_by_state={
            'trend_up': 1.5,
            'trend_down': 1.5,
            'transition': 1.25,
            'range': 1.0,
            'volatile': 0.75,
        },
        risk_mult_by_state={
            'trend_up': 1.0,
            'trend_down': 1.0,
            'transition': 0.75,
            'range': 0.50,
            'breakout': 0.50,
            'volatile': 0.25
        },
        score_low_quantile=75,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.15,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.0035,  # 0.5% equity
        min_score_to_trade=0.10, #0.2898742140685939
        max_risk_pct=0.02,
        max_positions=2
    )

    artifacts_path = f'./artifacts/{release}/oof/deploy_full'
    policy_path = f'./artifacts/{release}/rl/final/rl_policy_gate_{release}.npz'
    rl_config = {
        "lr": 0.001282007021137776,
        "entropy_coef": 0.024488467486753904,
        "baseline_beta": 0.9437265320016441,
        "max_grad_norm": 5.0,
        "trade_cost_money": 0.0075,
        "batch_size": 64,
        "chop_soft_thr": 0.55,
        "exhaustion_soft_thr": 0.40514612357395063,
        'rl_take_threshold': 0.2898742140685939,
        "chop_penalty_coef": 0.004,
        "exhaustion_penalty_coef": 0.004344263114297182,
    }

    engine = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=False,
        rl_config=rl_config,
        rl_train=False,
        rl_eval_deterministic=True,
        rl_take_threshold=0.2898742140685939, #0.62,
        rl_policy_path=policy_path,
        spread_price=0.07,
        mtm_use_bid_ask=True,
        mtm_price_col='close',
        sizing_equity_mode='balance',
        max_daily_loss_pct=0.035,
        max_daily_profit_pct=None,
        compound=True,
        enable_live_scaler_updates=True
    )

    engine.load_artifacts()
    if engine.use_rl:
        ok = load_rl_policy_npz(engine.rl_wrapper, policy_path)
        if not ok:
            raise RuntimeError(f'WARNING! RL Policy was not loaded from {policy_path}')

    return engine

# ── AÑADIR al bloque de imports (arriba del todo) ──────────────────────────
import logging
from pathlib import Path

# ── AÑADIR como función, junto a price_to_points / send_order ──────────────

def setup_signal_logger(log_dir: str = './logs') -> logging.Logger:
    """
    Crea un logger dedicado a señales del modelo.
    Escribe en JSONL rotando diariamente: signals_YYYYMMDD.jsonl
    """
    Path(log_dir).mkdir(parents=True, exist_ok=True)

    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    log_path = Path(log_dir) / f'signals_{date_str}.jsonl'

    logger = logging.getLogger('mimo_signals')
    logger.setLevel(logging.DEBUG)

    # Evitar duplicar handlers si se reinicia el engine en el mismo proceso
    if not logger.handlers:
        fh = logging.FileHandler(log_path, mode='a', encoding='utf-8')
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))  # solo el JSON, sin prefijos
        logger.addHandler(fh)

    logger.info(json.dumps({'event': 'LOGGER_STARTED', 'log_file': str(log_path),
                             'ts': time.time()}))
    return logger


def log_signal(
    logger: logging.Logger,
    event: str,                        # 'SIGNAL_SENT' | 'SIGNAL_BLOCKED' | 'NO_SIGNAL'
    bar_data: dict,                    # datos del tick/vela recibidos
    live_order: Optional[Dict] = None, # la señal del modelo (puede ser None)
    block_reason: Optional[str] = None,# motivo de bloqueo si aplica
    request: Optional[Dict] = None,    # el request construido, si se envió
    n_positions: int = 0,
    equity: float = 0.0,
    balance: float = 0.0,
) -> None:
    """
    Registra en JSONL cada decisión del motor: señal enviada, bloqueada o ausente.

    Estructura del registro:
      - ts / bar_time / event
      - regime / side / score / proba_long / proba_short / delta_rel
      - entry / sl / tp / qty / atr
      - sl_points / tp_points / rr_ratio
      - n_positions / equity / balance
      - block_reason (solo si SIGNAL_BLOCKED)
      - request (solo si SIGNAL_SENT, para trazabilidad completa)
    """
    record: Dict[str, Any] = {
        'event':       event,
        'ts':          time.time(),
        'bar_time':    str(bar_data.get('time', '')),
        'open':        bar_data.get('open'),
        'high':        bar_data.get('high'),
        'low':         bar_data.get('low'),
        'close':       bar_data.get('close'),
        'n_positions': n_positions,
        'equity':      round(equity, 2),
        'balance':     round(balance, 2),
    }

    if live_order:
        entry = float(live_order.get('entry', 0) or 0)
        sl    = float(live_order.get('sl',    0) or 0)
        tp    = float(live_order.get('tp',    0) or 0)
        atr   = float(live_order.get('atr_at_entry', 0) or 0)
        qty   = float(live_order.get('qty',   0) or 0)

        sl_pts = price_to_points(entry, sl) if entry and sl else None
        tp_pts = price_to_points(entry, tp) if entry and tp else None
        rr     = round(tp_pts / sl_pts, 2) if sl_pts and tp_pts and sl_pts > 0 else None

        record.update({
            'side':        live_order.get('side'),
            'regime':      live_order.get('state'),
            'score':       round(float(live_order.get('score', 0) or 0), 6),
            'proba_long':  round(float(live_order.get('proba_long',  0) or 0), 6),
            'proba_short': round(float(live_order.get('proba_short', 0) or 0), 6),
            'delta_rel':   round(float(live_order.get('delta_rel',   0) or 0), 6),
            'entry':       entry,
            'sl':          sl,
            'tp':          tp,
            'qty':         qty,
            'atr':         round(atr, 4),
            'sl_points':   sl_pts,
            'tp_points':   tp_pts,
            'rr_ratio':    rr,
        })

    if block_reason:
        record['block_reason'] = block_reason

    if request:
        # Solo guarda los campos más relevantes del request, no el dict completo
        # para mantener el log legible. Cambia a `record['request'] = request`
        # si quieres la traza completa.
        record['req_side']      = request.get('side')
        record['req_volume']    = request.get('volume')
        record['req_virtual_sl']= request.get('virtual_sl')
        record['req_virtual_tp']= request.get('virtual_tp')
        record['req_emergency_pts'] = request.get('risk', {}).get('emergency_points')

    logger.info(json.dumps(record, ensure_ascii=False))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logs de TensorFlow
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

context = zmq.Context.instance()
subscription = context.socket(zmq.SUB)
subscription.setsockopt(zmq.SUBSCRIBE, b'XAUUSD.r')
subscription.setsockopt(zmq.RCVHWM, 10000)
subscription.connect(f'tcp://10.1.21.25:5555')
time.sleep(1.0)

# --- S3 (Execution & Monitoring) via ZeroMQ ---
S3_ORDERS_ADDR = "tcp://10.1.21.25:5557"  # s3_executor PULL bind
S3_EVENTS_ADDR = "tcp://10.1.21.25:5558"  # s3_executor PUB bind

orders_push = context.socket(zmq.PUSH)
orders_push.setsockopt(zmq.SNDHWM, 10000)
orders_push.setsockopt(zmq.LINGER, 0)
orders_push.connect(S3_ORDERS_ADDR)

db = Database()
db.connect()

release='200345'
trading_policy='aggressive'

engine = trading_engine(release=release, trading_policy=trading_policy)
signal_logger = setup_signal_logger(log_dir='../logs')
while True:
    topic, raw = subscription.recv_multipart()
    data = json.loads(raw.decode("utf-8"))

    print(f'Data from MT5: {data}')
    symbol = data.pop('symbol')
    free_margin = data.pop('free_margin')
    balance = data.pop('balance')
    equity = data.pop('equity')
    n_positions = data.pop('n_positions')

    df = pd.DataFrame.from_dict([data])
    df['time'] = pd.to_datetime(df['time'] - 2 * 3600, unit='s', utc=True)
    df = df.drop(columns=['real_volume'])
    df = df.rename(columns={"tick_volume": 'volume'})
    db.save(df, table_name='rates')

    bar_snapshot = {
        'time': data.get('time'),
        'open': data.get('open'),
        'high': data.get('high'),
        'low': data.get('low'),
        'close': data.get('close'),
    }

    df_rates = engine.helper.load_from_database_real(db, last_n_rates=2048)
    live_order = engine.decide_live(df_rates=df_rates, equity=equity, current_positions=n_positions, max_positions=2)

    if hasattr(engine, 'pipeline') and hasattr(engine.pipeline, 'live_update_scalers_from_df'):
        df_prepared = engine.pipeline.prepare_data(df_rates)
        engine.pipeline.live_update_scalers_from_df(df_prepared.tail(1))

    # Completa parámetros de gestión de salida para S3 si el simulador aún no los añade.
    if live_order:
        score = live_order.get('score', 0)
        state = live_order.get('state', '?')
        side = live_order['side']

        print(f"\t[SCORE] {score:.4f}  side={side}  regime={state}")

        request = create_order_request(live_order, comment=trading_policy)
        send_order(orders_push, request)
        log_signal(signal_logger, event='SIGNAL_SENT',
                   bar_data=bar_snapshot, live_order=live_order,
                   request=request, n_positions=n_positions, equity=equity, balance=balance)

    else:
        log_signal(signal_logger, event='NO_SIGNAL',
                   bar_data=bar_snapshot, n_positions=n_positions, equity=equity, balance=balance)

print('')