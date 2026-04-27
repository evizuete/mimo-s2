import json
import time

import zmq

from mimo_old.databases import Database
from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import ModelConfig, Config
from mimo_old.trading_simulator import TradingSimulator, load_rl_policy_npz


context = zmq.Context.instance()

S3_ORDERS_ADDR = "tcp://10.1.21.25:5557"  # s3_executor PULL bind
S3_EVENTS_ADDR = "tcp://10.1.21.25:5558"  # s3_executor PUB bind

orders_push = context.socket(zmq.PUSH)
orders_push.setsockopt(zmq.SNDHWM, 20000)
orders_push.setsockopt(zmq.LINGER, 0)
orders_push.connect(S3_ORDERS_ADDR)

events_pull = context.socket(zmq.PULL)
events_pull.setsockopt(zmq.LINGER, 0)
events_pull.connect(S3_EVENTS_ADDR)

order = {
  "action":"OPEN",
  "symbol":"XAUUSD.r",
  "side":"BUY",
  "volume":0.10,
  "magic":200290,
  "catastrophe_points":500,
  "risk":{
    "emergency_points":160,
    "be_trigger_points":120,
    "be_offset_points":10,
    "trail_points":200,
    "trail_step_points":10,
    "max_hold_seconds":900,
    "partial_close":{
      "enabled": True,
      "trigger_profit_pct": 60,
      "close_fraction": 0.5,
      "basis": "R"
    }
  }
}

time.sleep(1.0)



try:
    orders_push.send_string(
        json.dumps(order)
    )

    while True:
        msg = events_pull.recv_string()
        print(f'Received event: {msg}')

except KeyboardInterrupt:
    print('Stopping listener...')

finally:
    orders_push.close(0)
    events_pull.close(0)

    context.term()

