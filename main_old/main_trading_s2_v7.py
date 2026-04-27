import json
import os
import time
from typing import Dict, Any, Optional

import pandas as pd
import zmq

from config.trading_policies_config import trading_policies
from mimo_old.databases import Database
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import ModelConfig, Config
from mimo.strategies.regime_detector import RegimeConfig
from mimo.strategies.trading_simulator import TradingSimulator, load_rl_policy_npz

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
        trail_step_ratio: float = 0.50,  # paso mínimo como ratio del virtual_sl_points (era 0.15, subido a 0.50 para reducir ruido)
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
            # FIX v6.0: sl_mode='touch' en todos los regímenes para evitar slippage
            # El modo 'close' añadía hasta 60s de retraso en scalping (ver S3 v7.0)
            'close_confirmation': {
                'exit_mode': 'hard',
                'tp_mode': 'touch',
                'tp_confirm_count': 0,
                'sl_mode': 'touch',
                'sl_confirm_count': 0,
            }
        }
    }

    return request

# ============================================================================
# S3 STATE SYNC  (v6.0)
# Mantiene un estado local sincronizado con los eventos que publica S3.
# Permite a S2 tomar decisiones de riesgo basadas en la realidad de S3
# (posiciones cerradas, parciales ejecutados) en lugar de confiar solo en
# el n_positions que llega con cada tick de MT5.
# ============================================================================

import threading
from dataclasses import dataclass


@dataclass
class S3Position:
    ticket: int
    side: str           # 'BUY' | 'SELL'
    volume: float
    entry_price: float
    partial_done: bool = False
    be_armed: bool = False
    virtual_sl: float = 0.0
    virtual_tp: float = 0.0


class S3State:
    """
    Estado local sincronizado con eventos de S3.

    Thread-safe: todas las lecturas/escrituras usan self._lock.
    Se actualiza en background por s3_event_listener.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.positions: Dict[int, S3Position] = {}
        self.last_event: Optional[Dict] = None
        self.last_slippage_pts: Optional[float] = None

    # ── Lecturas ─────────────────────────────────────────────────────────────

    def n_positions(self) -> int:
        with self._lock:
            return len(self.positions)

    def get_positions(self) -> Dict[int, S3Position]:
        with self._lock:
            return dict(self.positions)

    def total_volume(self) -> float:
        with self._lock:
            return sum(p.volume for p in self.positions.values())

    def sides_open(self) -> list:
        """Devuelve lista de sides ('BUY'/'SELL') de posiciones abiertas."""
        with self._lock:
            return [p.side for p in self.positions.values()]

    # ── Escrituras (llamadas desde el listener) ───────────────────────────────

    def on_event(self, msg: Dict) -> None:
        event = msg.get('event', '')
        with self._lock:
            self.last_event = msg

            if event == 'POSITION_OPENED':
                ticket = int(msg['ticket'])
                self.positions[ticket] = S3Position(
                    ticket=ticket,
                    side=msg.get('side', ''),
                    volume=float(msg.get('volume', 0)),
                    entry_price=float(msg.get('entry_price', 0)),
                    virtual_sl=float(msg.get('virtual_sl_price', 0)),
                    virtual_tp=float(msg.get('virtual_tp', 0)),
                )

            elif event in ('VIRTUAL_SL_TRIGGERED', 'VIRTUAL_TP_TRIGGERED',
                           'MAX_HOLD_TIME_TRIGGERED', 'PARTIAL_CLOSE_TO_FULL',
                           'POSITION_CLOSED_EXTERNAL', 'MANUAL_CLOSE_TRIGGERED'):
                ticket = int(msg.get('ticket', -1))
                self.positions.pop(ticket, None)
                # Guardar último slippage observado
                if 'slippage_pts' in msg:
                    self.last_slippage_pts = float(msg['slippage_pts'])

            elif event == 'PARTIAL_CLOSE_TRIGGERED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions and msg.get('volume_updated'):
                    self.positions[ticket].volume = float(msg.get('remaining_volume', 0))
                    self.positions[ticket].partial_done = True

            elif event == 'BE_ARMED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    self.positions[ticket].be_armed = True
                    self.positions[ticket].virtual_sl = float(msg.get('new_virtual_sl', 0))

            elif event == 'TRAILING_UPDATED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    self.positions[ticket].virtual_sl = float(msg.get('new_virtual_sl', 0))


def s3_event_listener(sub_socket: zmq.Socket, s3_state: S3State) -> None:
    """
    Hilo background que consume eventos publicados por S3 y actualiza S3State.
    No bloquea el loop principal de S2.
    """
    while True:
        try:
            raw = sub_socket.recv_string()
            msg = json.loads(raw)
            s3_state.on_event(msg)
        except zmq.Again:
            pass
        except Exception:
            pass


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
        sl_barrier=1.5, # Para evitar que en aquellos casos en que el Virtual SL es muy justo, cierre a perdidas
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
        enable_live_scaler_updates=True,
        anomaly_block_threshold=0.8,
        signal_cooldown_bars=3
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
    Escribe en JSONL rotando automáticamente a medianoche: signals_YYYYMMDD.jsonl

    FIX v6.0: sustituido FileHandler (fecha fija en arranque) por
    TimedRotatingFileHandler (rotación real a medianoche sin reiniciar el servicio).
    El sufijo se fuerza a YYYYMMDD para mantener el formato de nombre existente.
    """
    from logging.handlers import TimedRotatingFileHandler

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    # Fichero base con la fecha de hoy: el handler rota a medianoche creando uno nuevo
    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    log_path = Path(log_dir) / f'signals_{date_str}.jsonl'

    logger = logging.getLogger('mimo_signals')
    logger.setLevel(logging.DEBUG)

    # Evitar duplicar handlers si se reinicia el engine en el mismo proceso
    if not logger.handlers:
        fh = TimedRotatingFileHandler(
            filename=str(log_path),
            when='midnight',    # rotar a medianoche
            interval=1,          # cada 1 día
            backupCount=30,      # conservar 30 días de histórico
            encoding='utf-8',
            utc=False            # hora local, igual que el resto del sistema
        )
        # Forzar sufijo YYYYMMDD en ficheros rotados (por defecto sería YYYY-MM-DD)
        fh.suffix = '%Y%m%d'
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
    model_diag: Optional[Dict] = None, # diagnóstico del modelo aunque no haya señal
) -> None:
    """
    Registra en JSONL cada decisión del motor: señal enviada, bloqueada o ausente.

    Estructura del registro:
      - ts / bar_time / event
      - regime / side / score / proba_long / proba_short / delta_rel
      - entry / sl / tp / qty / atr
      - sl_points / tp_points / rr_ratio
      - n_positions / equity / balance
      - block_reason (si SIGNAL_BLOCKED)
      - model_diag (en NO_SIGNAL: score/regime/side del modelo aunque se haya filtrado)
      - request (si SIGNAL_SENT)
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
            # 'state' es la clave correcta en live_order (market_condition es el alias deprecado)
            'regime':      live_order.get('state') or live_order.get('market_condition'),
            'regime3':     live_order.get('regime3', ''),
            'score':       round(float(live_order.get('score', 0) or 0), 6),
            # proba_long / proba_short: ahora expuestos directamente por _build_order
            'proba_long':  round(float(live_order.get('proba_long',  live_order.get('proba_cal', 0)) or 0), 6),
            'proba_short': round(float(live_order.get('proba_short', 0) or 0), 6),
            'delta_rel':   round(float(live_order.get('delta_rel',   0) or 0), 6),
            'score_long':  round(float(live_order.get('score_long',  0) or 0), 6),
            'score_short': round(float(live_order.get('score_short', 0) or 0), 6),
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

    # Diagnóstico del modelo en ticks sin señal: permite ver qué estaba viendo
    # el modelo (score, regime, side candidato) aunque la señal se haya filtrado
    if model_diag:
        record['model_diag'] = model_diag

    if request:
        record['req_side']          = request.get('side')
        record['req_volume']        = request.get('volume')
        record['req_virtual_sl']    = request.get('virtual_sl')
        record['req_virtual_tp']    = request.get('virtual_tp')
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

# v6.0: suscribirse a eventos S3 para mantener estado local sincronizado
events_sub = context.socket(zmq.SUB)
events_sub.setsockopt(zmq.SUBSCRIBE, b'')   # todos los eventos (no hay topic prefix en S3)
events_sub.setsockopt(zmq.RCVHWM, 10000)
events_sub.setsockopt(zmq.LINGER, 0)
events_sub.connect(S3_EVENTS_ADDR)

# Estado local sincronizado con S3
s3_state = S3State()

# Lanzar listener en background
_s3_listener_thread = threading.Thread(
    target=s3_event_listener,
    args=(events_sub, s3_state),
    daemon=True,
    name='s3-event-listener'
)
_s3_listener_thread.start()
print(f"[S2] S3 event listener started → {S3_EVENTS_ADDR}")

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

    floating = equity - balance

    # ── Posiciones según S3 (fuente de verdad) vs MT5 ───────────────────────
    # s3_n: lo que S3 tiene realmente trackeado (incluye parciales, cierres, etc.)
    # n_positions: lo que MT5 reporta en el tick (puede ir con retraso)
    # Usamos el máximo como medida conservadora: si alguno ve más posiciones, creemos ese.
    s3_n = s3_state.n_positions()
    effective_positions = max(n_positions, s3_n)

    if s3_n != n_positions:
        print(f"\t[S3_SYNC] s3={s3_n} vs mt5={n_positions} → using effective={effective_positions}")

    if live_order:
        score = live_order.get('score', 0)
        state = live_order.get('state', '?')
        side  = live_order['side']
        mt5_side = 'BUY' if side == 'long' else 'SELL'

        print(f"\t[SCORE] {score:.4f}  side={side}  regime={state}")

        # ── FILTROS DE SEGURIDAD PRE-ENVÍO ───────────────────────────────────
        block_reason = None

        # 1. No acumular contra la dirección con drawdown significativo.
        #    v6.0: usamos sides_open() de S3 para saber el lado REAL abierto,
        #    en lugar de inferirlo del último side enviado (que podía ya estar cerrado).
        if effective_positions > 0 and floating < -50:
            sides = s3_state.sides_open()
            if sides and mt5_side not in sides:
                # La señal actual va en dirección opuesta a lo que S3 tiene abierto
                block_reason = (
                    f'OPPOSITE_SIDE_IN_DRAWDOWN('
                    f'floating={floating:.2f}, open_sides={sides}, new={mt5_side})'
                )

        # 2. Score muy bajo con posición ya abierta: no acumular con señales débiles.
        #    v6.0: usamos effective_positions en lugar de n_positions
        if effective_positions > 0 and score < 0.20 and block_reason is None:
            block_reason = f'LOW_SCORE_WITH_OPEN_POS(score={score:.4f}, s3_n={s3_n})'

        if block_reason:
            log_signal(signal_logger, event='SIGNAL_BLOCKED',
                       bar_data=bar_snapshot, live_order=live_order,
                       block_reason=block_reason,
                       n_positions=effective_positions, equity=equity, balance=balance)
            print(f"\t[BLOCKED] {block_reason}")
        else:
            request = create_order_request(live_order, comment=trading_policy)
            send_order(orders_push, request)
            log_signal(signal_logger, event='SIGNAL_SENT',
                       bar_data=bar_snapshot, live_order=live_order,
                       request=request, n_positions=effective_positions, equity=equity, balance=balance)

    else:
        # Sin señal: leer el diagnóstico completo que decide_live dejó en _last_diag
        model_diag = getattr(engine, '_last_diag', None)
        log_signal(signal_logger, event='NO_SIGNAL',
                   bar_data=bar_snapshot, n_positions=effective_positions,
                   equity=equity, balance=balance, model_diag=model_diag)

print('')