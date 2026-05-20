from datetime import datetime

import pandas as pd
from sqlalchemy import Column, BigInteger, String, Float, DateTime, JSON, Integer, Boolean
from sqlalchemy import inspect as sa_inspect
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class TradeEvent(Base):
    __tablename__ = 'trade_events'

    id = Column(BigInteger, primary_key=True)
    order_id = Column(String(64), index=True)
    symbol = Column(String(16))
    volume = Column(Float)
    direction = Column(String(16))
    open_price = Column(Float)
    close_price = Column(Float, nullable=True)
    pnl = Column(Float, nullable=True)

    virtual_sl = Column(Float)
    virtual_tp = Column(Float)

    status = Column(String(32))
    created_at = Column(DateTime, default=datetime.now())
    closed_at = Column(DateTime, nullable=True)

    meta = Column(JSON)

class RateFeatures(Base):
    __tablename__ = 'rate_features'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    rate_id = Column(BigInteger, index=True)

    atr = Column(Float)
    atr_norm = Column(Float)
    rsi = Column(Float)
    rsi_norm = Column(Float)
    adx = Column(Float)
    dmp = Column(Float)
    dmn = Column(Float)
    adx_smooth = Column(Float)
    dm_diff = Column(Float)
    dm_ratio = Column(Float)
    macd = Column(Float)
    macd_signal = Column(Float)
    macd_hist = Column(Float)
    macd_hist_norm = Column(Float)
    macd_divergence = Column(Float)
    bb_upper = Column(Float)
    bb_middle = Column(Float)
    bb_lower = Column(Float)
    bb_width = Column(Float)
    bb_position = Column(Float)
    ema_9 = Column(Float)
    ema_9_dist = Column(Float)
    ema_9_slope = Column(Float)
    ema_21 = Column(Float)
    ema_21_dist = Column(Float)
    ema_21_slope = Column(Float)
    ema_50 = Column(Float)
    ema_50_dist = Column(Float)
    ema_50_slope = Column(Float)
    ema_cross_9_21 = Column(Float)
    ema_cross_9_21_norm = Column(Float)
    trend_dir = Column(Integer)
    body = Column(Float)
    upper_wick = Column(Float)
    lower_wick = Column(Float)
    range_hl = Column(Float)
    body_rel = Column(Float)
    upper_wick_rel = Column(Float)
    lower_wick_rel = Column(Float)
    range_hl_rel = Column(Float)
    ret_1 = Column(Float)
    ret_1_norm = Column(Float)
    ret_3 = Column(Float)
    ret_3_norm = Column(Float)
    ret_5 = Column(Float)
    ret_5_norm = Column(Float)
    ret_10 = Column(Float)
    ret_10_norm = Column(Float)
    ret_20 = Column(Float)
    ret_20_norm = Column(Float)
    ret_60 = Column(Float)
    ret_60_norm = Column(Float)
    price_velocity = Column(Float)
    price_acceleration = Column(Float)
    efficiency_5 = Column(Float)
    realized_vol_5 = Column(Float)
    autocorr_5 = Column(Float)
    avg_range_5 = Column(Float)
    direction_bias_5 = Column(Float)
    efficiency_10 = Column(Float)
    realized_vol_10 = Column(Float)
    autocorr_10 = Column(Float)
    avg_range_10 = Column(Float)
    direction_bias_10 = Column(Float)
    efficiency_20 = Column(Float)
    realized_vol_20 = Column(Float)
    autocorr_20 = Column(Float)
    avg_range_20 = Column(Float)
    direction_bias_20 = Column(Float)
    efficiency_60 = Column(Float)
    realized_vol_60 = Column(Float)
    autocorr_60 = Column(Float)
    avg_range_60 = Column(Float)
    direction_bias_60 = Column(Float)
    range_expansion = Column(Float)
    consecutive_ups = Column(Integer)
    consecutive_downs = Column(Integer)
    doji = Column(Integer)
    hammer = Column(Integer)
    shooting_star = Column(Integer)
    bullish_engulfing = Column(Integer)
    bearish_engulfing = Column(Integer)
    pin_bar = Column(Integer)
    hour = Column(Integer)
    minute = Column(Integer)
    day_of_week = Column(Integer)
    hour_sin = Column(Float)
    hour_cos = Column(Float)
    minute_sin = Column(Float)
    minute_cos = Column(Float)
    dow_sin = Column(Float)
    dow_cos = Column(Float)
    is_asia = Column(Integer)
    is_london = Column(Integer)
    is_ny = Column(Integer)
    is_overlap = Column(Integer)
    is_open = Column(Integer)
    is_close = Column(Integer)
    dist_high_60 = Column(Float)
    dist_low_60 = Column(Float)
    position_range_60 = Column(Float)
    dist_high_240 = Column(Float)
    dist_low_240 = Column(Float)
    position_range_240 = Column(Float)
    dist_high_480 = Column(Float)
    dist_low_480 = Column(Float)
    position_range_480 = Column(Float)
    open_norm = Column(Float)
    high_norm = Column(Float)
    low_norm = Column(Float)
    close_norm = Column(Float)
    ema_slow = Column(Float)
    ema_slow_slope = Column(Float)
    regime = Column(String(64))
    regime_low_volatility = Column(Integer)
    regime_ranging = Column(Integer)
    regime_trending_up = Column(Integer)
    regime_trending_down = Column(Integer)
    regime_high_volatility = Column(Integer)
    regime_weight = Column(Float)
    chop_score = Column(Float)
    exhaustion_score = Column(Float)
    is_chop = Column(Boolean)
    is_exhaustion = Column(Boolean)

    @classmethod
    def from_row(cls, row: pd.Series):
        col_names = set(sa_inspect(cls).columns.keys())
        data = {}
        for k in col_names:
            if k == 'rate_id':
                v = row['id']
            elif k in row:
                v = row[k]
            else:
                continue

            data[k] = None if pd.isna(v) else v

        return cls(**data)

class DecisionEvent(Base):
    __tablename__ = 'decision_events'

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    rate_id = Column(BigInteger, index=True)

    action = Column(String(16))
    state = Column(String(64))
    state_group = Column(String(32))
    macro_regime = Column(String(64))
    p_buy_raw = Column(Float)
    p_sell_raw = Column(Float)
    p_buy_cal = Column(Float)
    p_sell_cal = Column(Float)
    score_buy = Column(Float)
    score_sell = Column(Float)
    chosen_score = Column(Float)
    risk_pct = Column(Float)
    reason = Column(String(64))
    buy_gate = Column(Integer)
    sell_gate = Column(Integer)
    buy_gate_val = Column(Float)
    sell_gate_val = Column(Float)
    score_cap = Column(Float)
    risk_mult = Column(Float)
    penalty = Column(Float)
    delta_rel = Column(Float)
    min_score = Column(Float)
    hard_spike = Column(Boolean)
    cooldown_left = Column(Integer)
    anomaly_score = Column(Float)

    @classmethod
    def from_decision(cls, d, *, rate_id=None):
        debug = d.debug or {}

        # Sanitize NaN → None: MySQL Float NOT accepts NaN literal.
        # Caso típico: con policy v2 (gate=100 = NO_TRADE), sell_gate_val sale
        # NaN al pedir el percentil 100 sobre una distribución vacía o sin
        # rango. Sin esta sanitización el INSERT crashea con:
        #   pymysql.err.ProgrammingError: nan can not be used with MySQL
        import math
        def _safe(v):
            if v is None:
                return None
            try:
                f = float(v)
                if math.isnan(f) or math.isinf(f):
                    return None
                return f
            except (TypeError, ValueError):
                return None

        return cls(
            rate_id=rate_id,
            action=d.action,
            state=d.state,
            macro_regime=d.macro_regime,
            p_buy_raw=_safe(d.p_buy_raw),
            p_sell_raw=_safe(d.p_sell_raw),
            p_buy_cal=_safe(d.p_buy_cal),
            p_sell_cal=_safe(d.p_sell_cal),
            score_buy=_safe(d.score_buy),
            score_sell=_safe(d.score_sell),
            chosen_score=_safe(d.chosen_score),
            risk_pct=_safe(d.risk_pct),

            reason=debug.get('reason'),
            state_group=debug.get('state_group'),
            delta_rel=_safe(debug.get('delta_rel')),
            min_score=_safe(debug.get('min_score')),
            buy_gate=debug.get('buy_gate'),
            sell_gate=debug.get('sell_gate'),
            buy_gate_val=_safe(debug.get('buy_gate_val')),
            sell_gate_val=_safe(debug.get('sell_gate_val')),
            score_cap=_safe(debug.get('score_cap')),
            risk_mult=_safe(debug.get('risk_mult')),
            penalty=_safe(debug.get('penalty')),
            hard_spike=debug.get('hard_spike'),
            cooldown_left=debug.get('cooldown_left'),
            anomaly_score=_safe(debug.get('anomaly_score')),
        )



