import json
import time
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path

import pandas as pd
import zmq
import logging

from config.decision_policies_config import gate_by_action_and_state, score_cap_by_state, risk_mult_by_state
from s2_config import S2Config
from s2_service_v2 import S2Service

from mimo.data_managers.databases import Database
from mimo.strategies.trading_simulator_v3 import TradingSimulator
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import ModelConfig, Config
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig
from mimo.strategies.regime_detector import RegimeConfig

def setup_signal_logger(log_dir: str = None):
    if log_dir is None:
        log_dir = str(Path(__file__).parent / "logs")

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    def _make_logger(name: str, base_stem: str) -> logging.Logger:
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger

        date_str = pd.Timestamp.now().strftime('%Y%m%d')
        log_path = Path(log_dir) / f'{base_stem}_{date_str}.jsonl'

        fh = TimedRotatingFileHandler(
            filename=str(log_path),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            utc=False,
        )

        def _namer(default_name: str) -> str:
            return default_name

        _log_dir_path = Path(log_dir)
        _base_stem_ref = base_stem

        def _rotator(source: str, dest: str) -> None:
            import os
            import shutil

            if os.path.exists(source):
                shutil.move(source, dest)

            new_date = pd.Timestamp.now().strftime('%Y%m%d')
            new_path = str(_log_dir_path / f'{_base_stem_ref}_{new_date}.jsonl')
            fh.baseFilename = new_path

        fh.namer = _namer
        fh.rotator = _rotator
        fh.suffix = '%Y%m%d'
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))

        logger.addHandler(fh)
        logger.propagate = False
        return logger

    signal_logger = _make_logger('mimo_signals', 'signals')
    system_logger = _make_logger('s2_system', 's2_system')

    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    log_path = Path(log_dir) / f'signals_{date_str}.jsonl'
    sys_log_path = Path(log_dir) / f's2_system_{date_str}.jsonl'

    signal_logger.info(json.dumps({
        'event': 'LOGGER_STARTED',
        'log_file': str(log_path),
        'ts': time.time(),
    }))

    system_logger.info(json.dumps({
        'event': 'SYSTEM_LOGGER_STARTED',
        'log_file': str(sys_log_path),
        'ts': time.time(),
    }))

    return signal_logger, system_logger


def build_sockets_and_loggers():
    context = zmq.Context.instance()

    # ── ticks_sub: antes llamado `subscription` ─────────────────────────────
    ticks_sub = context.socket(zmq.SUB)
    ticks_sub.setsockopt(zmq.SUBSCRIBE, b'XAUUSD.r')
    ticks_sub.setsockopt(zmq.RCVHWM, 10000)
    ticks_sub.setsockopt(zmq.RCVTIMEO, 30_000)   # 30s timeout
    ticks_sub.connect('tcp://10.1.21.25:5555')
    time.sleep(1.0)

    # ── orders_push ─────────────────────────────────────────────────────────
    S3_ORDERS_ADDR = "tcp://10.1.21.25:5557"
    orders_push = context.socket(zmq.PUSH)
    orders_push.setsockopt(zmq.SNDHWM, 10000)
    orders_push.setsockopt(zmq.LINGER, 0)
    orders_push.connect(S3_ORDERS_ADDR)

    # ── events_sub ──────────────────────────────────────────────────────────
    S3_EVENTS_ADDR = "tcp://10.1.21.25:5558"
    events_sub = context.socket(zmq.SUB)
    events_sub.setsockopt(zmq.SUBSCRIBE, b'')    # todos los eventos
    events_sub.setsockopt(zmq.RCVHWM, 10000)
    events_sub.setsockopt(zmq.LINGER, 0)
    events_sub.setsockopt(zmq.RCVTIMEO, 100)     # casi tiempo real
    events_sub.connect(S3_EVENTS_ADDR)

    # recomendable por tu propio diagnóstico histórico de handshake
    time.sleep(0.5)

    # ── loggers ─────────────────────────────────────────────────────────────
    signal_logger, system_logger = setup_signal_logger()

    sockets = {
        "ticks_sub": ticks_sub,
        "orders_push": orders_push,
        "events_sub": events_sub,
    }

    loggers = {
        "signals": signal_logger,
        "system": system_logger,
    }

    return sockets, loggers


def build_db():
    db = Database()
    db.connect()
    return db

def main(release: str, mode: str = 'production'):
    config = S2Config()
    config.counter_trend.block_total = False
    config.reversal_guard.enabled = True

    general_config = Config(
        release=release,
        oof_splits=5,
        oof_epochs=80
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
            "long": {
                "ema_bull": True,  "rsi_oversold": True,  "macd_positive": True,
                "ema_bear": False, "rsi_overbought": False, "macd_negative": False,
            },
            "short": {
                "ema_bear": True,  "rsi_overbought": True, "macd_negative": True,
                "ema_bull": False, "rsi_oversold": False,  "macd_positive": False,
            },
        },
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    decision_policy = DecisionPolicy(
        gate_by_action_and_state=gate_by_action_and_state[mode],
        score_cap_by_state=score_cap_by_state[mode],
        risk_mult_by_state=risk_mult_by_state[mode],
        score_low_quantile=80,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.20,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.0035,  # 0.5% equity
        min_score_to_trade=0.0,  # 0.2898742140685939
        max_risk_pct=0.02,
        max_positions=2
    )

    base_dir = Path(__file__).resolve().parent
    artifacts_path = str((base_dir / ".." / "artifacts" / release / "oof" / "deploy_full").resolve())
    policy_path = str((base_dir / ".." / "artifacts" / release / "rl" / "final" / f"rl_policy_gate_{release}.npz").resolve())
    rl_config = {
        "lr": 0.002,  # FINETUNE_LR del freeze
        "entropy_coef": 0.1,  # igual que staged
        "baseline_beta": 0.88,  # igual que staged
        "chop_soft_thr": 0.3670136046832346,  # trial 121
        "exhaustion_soft_thr": 0.4805561001350015,  # trial 121
        "rl_take_threshold": 0.036, #0.1444771151557441,  # trial 121
        "chop_penalty_coef": 0.004458284077367999,  # trial 121
        "exhaustion_penalty_coef": 0.003,  # fijo staged
        "max_grad_norm": 5.0,
        "trade_cost_money": 0.0075,
        "batch_size": 64,
    }
    simulator = TradingSimulator(
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
        rl_take_threshold=rl_config['rl_take_threshold'],
        rl_policy_path=policy_path,
        spread_price=0.07,
        mtm_use_bid_ask=True,
        mtm_price_col='close',
        sizing_equity_mode='balance',
        max_daily_loss_pct=0.035,
        max_daily_profit_pct=None,
        compound=True,
        enable_live_scaler_updates=True,
        anomaly_block_threshold=1.2,  # FIX v10.1: subido de 0.8 → 1.2 (0.8 bloqueaba 3h en sesión europea XAUUSD)
        signal_cooldown_bars=3
    )

    db = build_db()
    sockets, loggers = build_sockets_and_loggers()

    service = S2Service(
        config=config,
        simulator=simulator,
        db=db,
        sockets=sockets,
        loggers=loggers,
    )
    service.run()

if __name__ == "__main__":
    main(release='202500')