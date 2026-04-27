import json
import time

import pandas as pd
import requests
import zmq

from mimo_old.databases import Database
from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import ModelConfig, Config
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.trading_simulator import TradingSimulator, load_rl_policy_npz

API = "http://10.1.21.25:8001"
HEAD = {"x-api-key": "changeme", "Content-Type": "application/json"}

def send_order(symbol, side, volume=0.10, sl=None, tp=None, deviation=10, magic=12345, comment="api", result=None):
    payload = {
        "symbol": symbol,
        "type": side,            # "buy" o "sell"
        "volume": volume,
        "sl": sl,                # precio absoluto o None
        "tp": tp,                # precio absoluto o None
        "deviation": deviation,  # slippage máx en puntos del broker
        "magic": magic,
        "comment": comment,
        'rsi': result['rsi'] if result is not None else None,
        'atr': result['atr'] if result is not None else None,
        'adx': result['adx'] if result is not None else None,
        'dm_plus': result['dm_plus'] if result is not None else None,
        'dm_minus': result['dm_minus'] if result is not None else None,
        'bbp': result['bbp'] if result is not None else None,
        'bb_upper': result['bb_upper'] if result is not None else None,
        'bb_lower': result['bb_lower'] if result is not None else None,
        'buy': result['buy'] if result is not None else None,
        'sell': result['sell'] if result is not None else None,
        'mfe': result['mfe'] if result is not None else None,
        'mae': result['mae'] if result is not None else None
    }
    try:
        request = requests.post(f"{API}/order", headers=HEAD, json=payload, timeout=10)
    except requests.RequestException as e:
        print(repr(e))
        print('Payload: ', payload)
        raise

    ct = request.headers.get('Content-Type')
    body = request.text
    try:
        body_json = request.json()
    except Exception:
        body_json = None

    print(f'[HTTP] POST /order -> {request.status_code}')
    print('Payload: ', payload)
    if body_json is not None:
        print('Respuesta JSON: ', body_json)
    else:
        print('Respuesta: ', body[:1000])

    try:
        request.raise_for_status()
    except requests.HTTPError as e:
        raise requests.HTTPError(f'{e}\nDetalle backend: {body[:2000]}') from None

    return body_json if body_json is not None else body

def trading_engine():
    general_config = Config(
        release="200291",
        oof_splits=8,
        oof_epochs=25
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        batch_size=4096
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        label_method='adaptive'
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    decision_policy = DecisionPolicy(
        gate_by_action_and_state={
            "long": {
                'trend_up': 95,
                'transition': 97,
                'range': 99,
                'breakout': 97,
                'volatile': 99,
                "_global": 98,
            },
            "short": {
                'trend_down': 95,
                'transition': 97,
                'range': 99,
                'breakout': 97,
                'volatile': 99,
                "_global": 98,
            },
        },
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
        score_low_quantile=50,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.15,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.005,  # 0.5% equity
        min_score_to_trade=0.25,
        max_risk_pct=0.02,
        max_positions=2
    )

    artifacts_path = "../artifacts"
    policy_path = f'{artifacts_path}/rl_policies/mimo_gate_{general_config.release}_live.npz'
    #policy_path = f'./artifacts/rl_policy_gate_{general_config.release}.npz'
    rl_config = {
        'lr': 5e-3,  # 5e-4,
        'entropy_coef': 1e-3,
        'baseline_beta': 0.90,  # 0.95,
        'max_grad_norm': 10.0,
        'trade_cost_money': 0.5,
        'batch_size': 32,  # 256
        'chop_soft_thr': 0.60,
        'exhaustion_soft_thr': 0.60,
        'chop_penalty_coef': 0.08,
        'exhaustion_penalty_coef': 0.06,
    }

    engine = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=True,
        rl_config=rl_config,
        rl_train=False,
        rl_eval_deterministic=True,
        rl_take_threshold=0.62,
        rl_policy_path=policy_path,
        spread_price=0.07,
        mtm_use_bid_ask=True,
        mtm_price_col='close',
        sizing_equity_mode='balance',
        max_daily_loss_pct=0.0570,
        max_daily_profit_pct=None,
        compound=True,
    )

    engine.load_artifacts()
    ok = load_rl_policy_npz(engine.rl_wrapper, policy_path)
    if not ok:
        raise f'WARNING! RL Policy was not loaded from {policy_path}'

    return engine

context = zmq.Context.instance()
subscription = context.socket(zmq.SUB)
subscription.setsockopt(zmq.SUBSCRIBE, b'XAUUSD')
subscription.setsockopt(zmq.RCVHWM, 10000)
subscription.bind(f'tcp://localhost:5555')
time.sleep(1.0)

db = Database()
db.connect()

engine = trading_engine()
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
    df['time'] = pd.to_datetime(df['time'] - 2*3600, unit='s', utc=True)
    df = df.drop(columns=['real_volume'])
    df = df.rename(columns={"tick_volume": 'volume'})
    db.save(df, table_name='rates')

    df = engine.helper.load_from_database_real(db, last_n_rates=2048)
    live_order = engine.decide_live(df_rates=df, equity=equity, current_positions=n_positions, max_positions=2)
    if live_order:
        send_order(symbol,
                   side=live_order['side'],
                   volume=round(live_order['qty'], 2),
                   magic=200291,
                   tp=live_order['tp'],
                   sl=live_order['sl']
        )

print('')
