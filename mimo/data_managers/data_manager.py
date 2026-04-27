from typing import List

import joblib
import numpy as np
import pandas as pd
import pandas_ta_classic as ta
from keras.src.utils import to_categorical
from numpy.lib._stride_tricks_impl import sliding_window_view
from pandas import DataFrame
from sklearn.metrics import confusion_matrix, precision_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder
from sklearn.utils import compute_class_weight


class DataManager:
    def __init__(self, df: DataFrame) -> None:
        self.train_end = None
        self.val_start = None
        self.starts = None
        self.feature_cols = None
        self.market_state_classes = None
        self.df_features = None
        self.df_signals = pd.DataFrame()
        self.df_target = None
        self.scaler = None
        self.df = df

    @classmethod
    def from_database_historical(cls, database, from_date, to_date):
        query = f"SELECT r.id, r.time, r.open, r.high, r.low, r.close, r.volume as ticks_volume from historical_rates r WHERE r.time BETWEEN '{from_date}' AND '{to_date}'"
        rates = database.read(query=query)

        if rates is None or rates.empty:
            return None

        time_range = pd.date_range(start=rates.time.min().normalize(), end=rates.time.max(), freq='min')
        df = pd.DataFrame({'time': time_range})
        df = pd.merge(df, rates, on='time', how='left')
        return cls(df)

    @classmethod
    def from_database_historical_2(cls, database, from_date, to_date):
        query = f"SELECT r.id, r.time, r.open, r.high, r.low, r.close, r.volume as ticks_volume from historical_rates r WHERE r.time BETWEEN '{from_date}' AND '{to_date}'"
        rates = database.read(query=query)

        if rates is None or rates.empty:
            return None

        return cls(rates)

    @classmethod
    def from_database_real(cls, database, last_n_rates=256):
        query = f"SELECT r.id, r.time, r.open, r.high, r.low, r.close, r.volume as ticks_volume FROM rates r ORDER BY r.id DESC LIMIT {last_n_rates}"
        rates = database.read(query=query)

        if rates is None or rates.empty:
            return None

        rates = rates[::-1]
        time_range = pd.date_range(start=rates.time.min().normalize(), end=rates.time.max(), freq='min')
        df = pd.DataFrame({'time': time_range})
        df = pd.merge(df, rates, on='time', how='left')
        return cls(df.tail(last_n_rates))

    @classmethod
    def from_database_real_2(cls, database, last_n_rates=256):
        query = f"SELECT r.id, r.time, r.open, r.high, r.low, r.close, r.volume as ticks_volume FROM rates r ORDER BY r.id DESC LIMIT {last_n_rates}"
        rates = database.read(query=query)

        if rates is None or rates.empty:
            return None

        rates = rates[::-1]
        #time_range = pd.date_range(start=rates.time.min().normalize(), end=rates.time.max(), freq='min')
        #df = pd.DataFrame({'time': time_range})
        #df = pd.merge(df, rates, on='time', how='left')
        return cls(rates.tail(last_n_rates))

    @classmethod
    def from_file(cls, filename):
        cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'spread']
        df = pd.read_csv(filename, header=1, names=cols)
        df['time'] = pd.to_datetime(df.datetime) - pd.Timedelta(hours=1)
        df = df.drop('datetime', axis=1)

        return cls(df)

    @staticmethod
    def normalize_ohlc_by_range(df: DataFrame) -> DataFrame:
        hl_range = (df.high - df.low).replace(0, np.nan)
        out = pd.DataFrame(index=df.index)

        out['open_norm'] = (df.open - df.low) / hl_range
        out['high_norm'] = (df.high - df.low) / hl_range        #1.0
        out['low_norm'] = (df.low - df.high) / hl_range         #0.0
        out['close_norm'] = (df.close - df.low) / hl_range

        return out

    @staticmethod
    def normalize_ohlc_by_close(df: DataFrame) -> DataFrame:
        close = df.close.replace(0, np.nan)
        out = pd.DataFrame(index=df.index)

        out['open_norm_rel_close'] = (df.open - close) / close
        out['high_norm_rel_close'] = (df.high - close) / close
        out['low_norm_rel_close'] = (df.low - close) / close
        out['close_norm_rel_close'] = 0.0

        return out

    def get_market_state(self):
        self.df['state'] = np.select(
            [
                (self.df.adx > 40),
                (self.df.adx > 25) & (self.df.dm_plus > self.df.dm_minus + 5),
                (self.df.adx > 25) & (self.df.dm_minus > self.df.dm_plus + 5),
                (self.df.atr < self.df.atr.quantile(0.25)) & (self.df.bbp.between(0.4, 0.6))
            ],
            ['volatile', 'trending_up', 'trending_down', 'low_vol'],
            default='ranging'
        )

        label_encoder = LabelEncoder()
        self.df['state_id'] = label_encoder.fit_transform(self.df['state'])

        state_onehot = to_categorical(self.df['state_id'], num_classes=len(label_encoder.classes_))
        df_states = pd.DataFrame(state_onehot,
                                 index=self.df.index,
                                 columns=label_encoder.classes_)
        ok = np.all(self.df['state_id'].to_numpy() == df_states.to_numpy().argmax(axis=1))
        self.df = self.df.join(df_states)

        return label_encoder.classes_.tolist()

    def get_market_state2(self):
        df = self.df.copy()

        # Umbrales dinámicos por percentiles
        win_thr = 240
        atr_low = df["atr_norm"].rolling(win_thr).quantile(0.20)
        atr_high = df["atr_norm"].rolling(win_thr).quantile(0.80)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
        bb_width_low = df["bb_width"].rolling(win_thr).quantile(0.20)
        bb_width_high = df["bb_width"].rolling(win_thr).quantile(0.80)

        ADX_IN_TREND = 22  # entrar a "trend" si ADX supera esto
        ADX_OUT_TREND = 18  # salir de "trend" si ADX cae por debajo (menor)
        SLOPE_THR_IN = df["ema_20"].pct_change(20).abs().rolling(200).median() * 0.5 + 1e-6
        SLOPE_THR_OUT = SLOPE_THR_IN * 0.7

        is_low_vol = (df["atr_norm"] < atr_low) & (df["bb_width"] < bb_width_low) & (df["adx_s"] < ADX_OUT_TREND)
        is_volatile = (df["atr_norm"] > atr_high) | (df["bb_width"] > bb_width_high)

        trend_up_base = (df["adx_s"] > ADX_IN_TREND) & ((df["dm_plus"] - df["dm_minus"]) >= 5) & (df["ema_slow_slope"] > 0)
        trend_down_base = (df["adx_s"] > ADX_IN_TREND) & ((df["dm_minus"] - df["dm_plus"]) >= 5) & (df["ema_slow_slope"] < 0)

        near_mid = df["bbp"].between(0.35, 0.65)
        flat_slope = df["ema_fast_slope"].abs() < SLOPE_THR_OUT
        is_ranging_base = (df["adx_s"] < ADX_OUT_TREND) & flat_slope & near_mid & ~is_volatile

        states_base = np.select(
            [
                is_volatile,
                trend_up_base,
                trend_down_base,
                is_low_vol
            ],
            ["volatile", "trending_up", "trending_down", "low_vol"],
            default="ranging"
        )

        df["state_base"] = states_base

        persist = 3
        df["state"] = df["state_base"].copy()
        for s in ["trending_up", "trending_down", "low_vol", "volatile", "ranging"]:
            mask = (df["state_base"] == s).astype(int).rolling(persist).sum() >= persist
            df.loc[~mask & (df["state_base"] == s), "state"] = np.nan

        # forward-fill para mantener el estado previo si no hay persistencia suficiente
        df["state"] = df["state"].ffill().fillna("ranging")

        self.df["state"] = df["state"]

        label_encoder = LabelEncoder()
        self.df['state_id'] = label_encoder.fit_transform(self.df['state'])

        state_onehot = to_categorical(self.df['state_id'], num_classes=len(label_encoder.classes_))
        df_states = pd.DataFrame(state_onehot,
                                 index=self.df.index,
                                 columns=label_encoder.classes_)
        ok = np.all(self.df['state_id'].to_numpy() == df_states.to_numpy().argmax(axis=1))
        self.df = self.df.join(df_states)

        return label_encoder.classes_.tolist()


    def build_features(self,
                       use_range_norm: bool = True,
                       include_rel_close_norm: bool = True,
                       ema_fast: int = 20,
                       ema_slow: int = 50,
                       rsi_period: int = 14,
                       atr_period: int = 14,
                       adx_period: int = 14,
                       bb_period: int = 20,
                       bb_nstd: float = 2.0,
                       add_time_encoding: bool = True):

        df = self.df.copy()

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['atr_mean'] = df.atr.rolling(window=50).mean()
        df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
        df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
        df['rsi'] = ta.rsi(df['close'], period=rsi_period)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

        adx = ta.adx(df.high, df.low, df.close, length=adx_period)
        df = df.join(adx)
        df.rename(columns={f'ADX_{adx_period}': 'adx',
                           f'DMP_{adx_period}': 'dm_plus',
                           f'DMN_{adx_period}': 'dm_minus'
                           }, inplace=True)

        # Bollinger Bands
        bb = ta.bbands(df.close, length=bb_period, std=bb_nstd)
        df = df.join(bb)
        df.rename(columns={f'BBL_{bb_period}_2.0': 'bb_lower',
                           f'BBM_{bb_period}_2.0': 'bb_mid',
                           f'BBU_{bb_period}_2.0': 'bb_upper',
                           f'BBB_{bb_period}_2.0': 'bbb',
                           f'BBP_{bb_period}_2.0': 'bbp'
                           }, inplace=True)

        if use_range_norm:
            ohlc_norm = self.normalize_ohlc_by_range(df)
            df = pd.concat([df, ohlc_norm], axis=1)

        if include_rel_close_norm:
            ohlc_rel = self.normalize_ohlc_by_close(df)
            df = pd.concat([df, ohlc_rel], axis=1)

        df['return_1'] = df.close.pct_change(1)
        df['return_3'] = df.close.pct_change(3)
        df['return_5'] = df.close.pct_change(5)
        df['return_10'] = df.close.pct_change(10)

        if add_time_encoding:
            hours = df.time.dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

        for col in ['open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'bb_lower', 'bb_mid', 'bb_upper']:
            df[f'log_{col}'] = np.log(df[col] / df[col].shift(1))

        base_cols = [
            #'low', 'close', 'open', 'high',
            #'ema_fast', 'ema_slow', 'rsi', 'atr', 'adx', 'dm_plus', 'dm_minus', 'bb_lower', 'bb_mid', 'bb_upper', 'bbb',
            'log_open', 'log_high', 'log_low', 'log_close', 'log_ema_fast', 'log_ema_slow',
            'rsi', 'atr', 'adx', 'dm_plus', 'dm_minus', 'bb_lower', 'bb_mid', 'bb_upper', 'bbb',
            'bbp', #'return_1', 'return_3', 'return_5', 'return_10'
        ]

        time_cols = ['hour_sin', 'hour_cos'] if add_time_encoding else []
        ohlc_rel_cols = ['open_norm_rel_close', 'high_norm_rel_close', 'low_norm_rel_close', 'close_norm_rel_close'] if include_rel_close_norm else []
        ohlc_norm_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm'] if use_range_norm else []

        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df.dropna().copy()

        return base_cols + time_cols + ohlc_norm_cols + ohlc_rel_cols

    @staticmethod
    def slope(series, win=20):
        x = np.arange(win)
        return pd.Series(series).rolling(win).apply(
            lambda y: np.polyfit(x, y, 1)[0], raw=True
        )

    def build_features2(self,
                       use_range_norm: bool = True,
                       include_rel_close_norm: bool = True,
                       ema_fast: int = 20,
                       ema_slow: int = 50,
                       rsi_period: int = 14,
                       atr_period: int = 14,
                       adx_period: int = 14,
                       bb_period: int = 20,
                       bb_nstd: float = 2.0,
                       add_time_encoding: bool = True):

        df = self.df.copy()

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['atr_norm'] = df['atr'] / df['close']
        df['atr_mean'] = df.atr.rolling(window=50).mean()
        df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
        df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
        df['ema_fast_slope'] = self.slope(df['ema_fast'], win=20)
        df['ema_slow_slope'] = self.slope(df['ema_slow'], win=50)

        df['rsi'] = ta.rsi(df['close'], period=rsi_period)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)

        adx = ta.adx(df.high, df.low, df.close, length=adx_period)
        df = df.join(adx)
        df.rename(columns={f'ADX_{adx_period}': 'adx',
                           f'DMP_{adx_period}': 'dm_plus',
                           f'DMN_{adx_period}': 'dm_minus'
                           }, inplace=True)

        df['adx_s'] = df['adx'].rolling(5).median()

        # Bollinger Bands
        bb = ta.bbands(df.close, length=bb_period, std=bb_nstd)
        df = df.join(bb)
        df.rename(columns={f'BBL_{bb_period}_2.0': 'bb_lower',
                           f'BBM_{bb_period}_2.0': 'bb_mid',
                           f'BBU_{bb_period}_2.0': 'bb_upper',
                           f'BBB_{bb_period}_2.0': 'bbb',
                           f'BBP_{bb_period}_2.0': 'bbp'
                           }, inplace=True)

        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        if use_range_norm:
            ohlc_norm = self.normalize_ohlc_by_range(df)
            df = pd.concat([df, ohlc_norm], axis=1)

        if include_rel_close_norm:
            ohlc_rel = self.normalize_ohlc_by_close(df)
            df = pd.concat([df, ohlc_rel], axis=1)

        df['return_1'] = df.close.pct_change(1)
        df['return_3'] = df.close.pct_change(3)
        df['return_5'] = df.close.pct_change(5)
        df['return_10'] = df.close.pct_change(10)

        if add_time_encoding:
            hours = df.time.dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

        for col in ['open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'bb_lower', 'bb_mid', 'bb_upper']:
            df[f'log_{col}'] = np.log(df[col] / df[col].shift(1))

        base_cols = [
            #'low', 'close', 'open', 'high',
            #'ema_fast', 'ema_slow', 'rsi', 'atr', 'adx', 'dm_plus', 'dm_minus', 'bb_lower', 'bb_mid', 'bb_upper', 'bbb',
            'log_open', 'log_high', 'log_low', 'log_close', 'log_ema_fast', 'log_ema_slow',
            'rsi', 'atr', 'adx', 'dm_plus', 'dm_minus', 'bb_lower', 'bb_mid', 'bb_upper', 'bbb',
            'bbp', #'return_1', 'return_3', 'return_5', 'return_10'
        ]

        time_cols = ['hour_sin', 'hour_cos'] if add_time_encoding else []
        ohlc_rel_cols = ['open_norm_rel_close', 'high_norm_rel_close', 'low_norm_rel_close', 'close_norm_rel_close'] if include_rel_close_norm else []
        ohlc_norm_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm'] if use_range_norm else []

        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df.dropna().copy()

        return base_cols + time_cols + ohlc_norm_cols + ohlc_rel_cols

    def build_features3(self,
                       use_range_norm: bool = True,
                       include_rel_close_norm: bool = True,
                       ema_fast: int = 20,
                       ema_slow: int = 50,
                       rsi_period: int = 14,
                       atr_period: int = 14,
                       adx_period: int = 14,
                       bb_period: int = 20,
                       bb_nstd: float = 2.0,
                       add_time_encoding: bool = True):

        df = self.df.copy()

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['atr_norm'] = df['atr'] / df['close']
        df['atr_mean'] = df.atr.rolling(window=50).mean()
        df['atr_slope'] = self.slope(df['atr_norm'], win=min(10, atr_period))

        df['ema_fast'] = ta.ema(df['close'], length=ema_fast)
        df['ema_slow'] = ta.ema(df['close'], length=ema_slow)
        df['ema_spread'] = df['ema_fast'] - df['ema_slow']
        df['ema_spread_norm'] = df['ema_spread'] / df['close']
        df['ema_fast_slope'] = self.slope(df['ema_fast'], win=20)
        df['ema_slow_slope'] = self.slope(df['ema_slow'], win=50)

        df['rsi'] = ta.rsi(df['close'], period=rsi_period)
        df['rsi_slope'] = self.slope(df['rsi'], win=min(10, rsi_period))

        adx = ta.adx(df.high, df.low, df.close, length=adx_period)
        df = df.join(adx)
        df.rename(columns={f'ADX_{adx_period}': 'adx',
                           f'DMP_{adx_period}': 'dm_plus',
                           f'DMN_{adx_period}': 'dm_minus'
                           }, inplace=True)

        df['adx_s'] = df['adx'].rolling(5).median()

        # Bollinger Bands
        bb = ta.bbands(df.close, length=bb_period, std=bb_nstd)
        df = df.join(bb)
        df.rename(columns={f'BBL_{bb_period}_2.0': 'bb_lower',
                           f'BBM_{bb_period}_2.0': 'bb_mid',
                           f'BBU_{bb_period}_2.0': 'bb_upper',
                           f'BBB_{bb_period}_2.0': 'bbb',
                           f'BBP_{bb_period}_2.0': 'bbp'
                           }, inplace=True)

        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

        if use_range_norm:
            ohlc_norm = self.normalize_ohlc_by_range(df)
            df = pd.concat([df, ohlc_norm], axis=1)

        if include_rel_close_norm:
            ohlc_rel = self.normalize_ohlc_by_close(df)
            df = pd.concat([df, ohlc_rel], axis=1)

        df['return_1'] = df.close.pct_change(1)
        df['return_3'] = df.close.pct_change(3)
        df['return_5'] = df.close.pct_change(5)
        df['return_10'] = df.close.pct_change(10)

        sma20 = df['close'].rolling(20).mean()
        std20 = df['close'].rolling(20).std()
        df['zscore_20'] = (df['close'] - sma20) / std20.replace(0, np.nan)

        if add_time_encoding:
            hours = df.time.dt.hour
            df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
            df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

            weekday = df.time.dt.dayofweek
            df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7.0)
            df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7.0)

        for col in ['open', 'high', 'low', 'close', 'ema_fast', 'ema_slow', 'bb_lower', 'bb_mid', 'bb_upper']:
            df[f'log_{col}'] = np.log(df[col] / df[col].shift(1))

        base_cols = [
            'ema_fast', 'ema_slow', 'ema_spread_norm','ema_fast_slope', 'ema_slow_slope',
            'adx', 'adx_s', 'dm_plus', 'dm_minus',
            'atr_norm', 'bb_width',
            'bbp', 'zscore_20',
            'rsi', 'rsi_slope', 'return_1', 'return_3', 'return_5', 'return_10',
            'atr_slope'
        ]

        time_cols = ['hour_sin', 'hour_cos', 'weekday_sin', 'weekday_cos'] if add_time_encoding else []
        ohlc_rel_cols = ['open_norm_rel_close', 'high_norm_rel_close', 'low_norm_rel_close', 'close_norm_rel_close'] if include_rel_close_norm else []
        ohlc_norm_cols = ['open_norm', 'high_norm', 'low_norm', 'close_norm'] if use_range_norm else []

        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df.dropna().copy()

        return base_cols + time_cols + ohlc_norm_cols + ohlc_rel_cols

    def build_features_4(self,
                         lookback_only: bool = True,
                         reduce_noise: bool = True):

        df = self.df.copy()

        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=14)
        if lookback_only:
            df['close_sma'] = df['close'].rolling(20, min_periods=20).mean()
            df['atr_norm'] = df['atr'] / df['close_sma']
        else:
            df['atr_norm'] = df['atr'] / df['close']

        df['rsi'] = ta.rsi(df['close'], period=14)
        if reduce_noise:
            df['rsi_smooth'] = df['rsi'].ewm(span=3, adjust=False).mean()

        adx = ta.adx(df.high, df.low, df.close, length=14)
        df = df.join(adx)
        df.rename(columns={f'ADX_14': 'adx',
                           f'DMP_14': 'dm_plus',
                           f'DMN_14': 'dm_minus'}, inplace=True)
        df['adx_s'] = df['adx'].rolling(5, min_periods=5).median()

        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        df['ema_spread_pct'] = (df['ema_20'] - df['ema_50']) / df['ema_50'] * 100

        bb = ta.bbands(df.close, length=20, std=2)
        df = df.join(bb)
        df.rename(columns={f'BBL_20_2.0': 'bb_lower',
                           f'BBM_20_2.0': 'bb_mid',
                           f'BBU_20_2.0': 'bb_upper',
                           f'BBB_20_2.0': 'bbb',
                           f'BBP_20_2.0': 'bbp'
                           }, inplace=True)

        df['bb_width_pct'] = (df['bb_upper'] - df['bb_lower']) / df['ema_20'] * 100

        df['return_5'] = df['close'].pct_change(5)
        df['return_20'] = df['close'].pct_change(20)

        if reduce_noise:
            df['return_5'] = df['return_5'].clip(lower=-0.05, upper=0.05)
            df['return_20'] = df['return_20'].clip(lower=-0.10, upper=0.10)

        df['zscore'] = (df['close'] - df['close'].rolling(20, min_periods=20).mean()) / df['close'].rolling(20, min_periods=20).std()
        df['zscore'] = df['zscore'].clip(-3, 3)

        df['roc_10'] = ta.roc(df['close'], length=10)
        if reduce_noise:
            df['momentum'] = df['close'].pct_change(10).ewm(span=5, adjust=False).mean()
        else:
            df['momentum'] = df['close'].pct_change(10)

        df['volatility'] = df['return_5'].rolling(20, min_periods=20).std()
        df['volatility_norm'] = df['volatility'] / df['volatility'].rolling(50, min_periods=50).mean()
        df['volatility_norm'] = df['volatility_norm'].clip(0.5, 2.0)

        hours = df['time'].dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hours / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * hours / 24.0)

        feature_cols = [
            'atr_norm', 'rsi_smooth' if reduce_noise else 'rsi',
            'adx', 'dm_plus', 'dm_minus',
            'bbp', 'bb_width_pct', 'ema_spread_pct',
            'zscore', 'momentum', 'roc_10',
            'return_5', 'return_20',
            'volatility_norm',
            'hour_sin', 'hour_cos'
        ]

        df = df.replace([np.inf, -np.inf], np.nan)
        self.df = df.dropna()

        return feature_cols

    @staticmethod
    def make_sequences(data, seq_len: int = 32) -> np.ndarray:
        #datasets = self.df[feature_cols].values.astype(np.float32)

        T, F = data.shape
        if T < seq_len:
            raise ValueError(f'Not enough datasets to generate sequences ({T}) for timestamp={seq_len}')

        N = T - seq_len + 1
        X = np.zeros((N, seq_len, F), dtype = np.float32)
        for i in range(N):
            X[i] = data[i:i+seq_len]

        return X

    def add_basic_indicators(self, rsi_period=14, ema_short_period=20, ema_long_period=50, atr_period=14, adx_period=14, bb_period=20):
        df = self.df.copy()

        # Adding RSI, EMA and ATR indicators
        df['rsi'] = ta.rsi(df['close'], period=rsi_period)
        df['ema20'] = ta.ema(df['close'], length=ema_short_period)
        df['ema50'] = ta.ema(df['close'], length=ema_long_period)
        df['atr'] = ta.atr(df['high'], df['low'], df['close'], length=atr_period)
        df['atr_mean'] = df.atr.rolling(window=50).mean()

        adx = ta.adx(df.high, df.low, df.close, length=adx_period)
        df = df.join(adx)
        df.rename(columns={f'ADX_{adx_period}': 'adx',
                                f'DMP_{adx_period}': 'dm_plus',
                                f'DMN_{adx_period}': 'dm_minus'
                                }, inplace=True)

        # Estimating CLOSE changes in 1 and 5 rates
        df['return_1'] = df['close'].pct_change(1, fill_method=None)
        df['return_3'] = df['close'].pct_change(3, fill_method=None)
        df['return_5'] = df['close'].pct_change(5, fill_method=None)
        df['return_10'] = df['close'].pct_change(10, fill_method=None)

        # Bollinger Bands
        bb = ta.bbands(df.close, length=bb_period, std=2)
        df = df.join(bb)
        df.rename(columns={f'BBL_{bb_period}_2.0': 'bb_lower',
                                f'BBM_{bb_period}_2.0': 'bb_middle',
                                f'BBU_{bb_period}_2.0': 'bb_upper',
                                f'BBB_{bb_period}_2.0': 'bbb',
                                f'BBP_{bb_period}_2.0': 'bbp'
                                }, inplace=True)

        hour = df.time.dt.hour
        df['hour_sin'] = np.sin(2 * np.pi * hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * hour / 24)

        weekday = df.time.dt.dayofweek
        df['weekday_sin'] = np.sin(2 * np.pi * weekday / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * weekday / 7)

        self.df = df

        return ['open','high','low','close','rsi','ema20','ema50','atr','return_1', 'return_3', 'return_5', 'return_10',
                'adx', 'dm_plus', 'dm_minus', 'bb_lower', 'bb_middle', 'bb_upper', 'bbb', 'bbp',
                'hour_sin', 'hour_cos']

    def add_relative_indicators(self):
        df = self.df.copy()

        df['body_ratio'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8)
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['upper_shadow_ratio'] = (df['high'] - np.maximum(df['open'], df['close'])) / (df['high'] - df['low'] + 1e-8)
        df['lower_shadow_ratio'] = (np.minimum(df['open'], df['close']) - df['low']) / (df['high'] - df['low'] + 1e-8)
        df['shadow_symmetry'] = abs(df['upper_shadow_ratio'] - df['lower_shadow_ratio'])
        df['is_bullish'] = (df['close'] > df['open']).astype(int)
        df['is_doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low'] + 1e-8) < 0.1).astype(int)

        df['momentum_acceleration'] = df['return_1'] - df['return_1'].shift(1)
        df['momentum_vs_avg'] = df['return_5'] / (df['return_5'].rolling(20).std() + 1e-8)
        df['momentum_ratio'] = df['return_3'] / (df['return_10'] + 1e-8)

        df['distance_to_high'] = (df['high'].rolling(20).max() - df['close']) / df['close']
        df['distance_to_low'] = (df['close'] - df['low'].rolling(20).min()) / df['close']
        df['gap_ratio'] = abs(df['open'] - df['close'].shift(1)) / df['close'].shift(1)

        price_move = abs(df['close'] - df['open'])
        total_range = df['high'] - df['low']
        df['movement_efficiency'] = price_move / (total_range + 1e-8)

        if df['close'].iloc[-1] > df['open'].iloc[-1]:  # Vela alcista
            retracement = (df['high'] - df['close']) / (df['high'] - df['open'] + 1e-8)
        else:  # Vela bajista
            retracement = (df['close'] - df['low']) / (df['open'] - df['low'] + 1e-8)
        df['retracement_ratio'] = retracement

        self.df = df
        return ['body_ratio', 'close_position', 'upper_shadow_ratio', 'lower_shadow_ratio', 'shadow_symmetry', 'is_bullish',
                'is_doji', 'momentum_acceleration', 'momentum_vs_avg', 'momentum_ratio', 'distance_to_high', 'distance_to_low',
                'gap_ratio', 'movement_efficiency', 'retracement_ratio']

    def add_targets(self, horizon=15, use_atr_norm=True, clip_mult=10.0):
        """
        MFE = max( High_{t+1..t+H} - Close_t )
        MAE = max( Close_t - Low_{t+1..t+H} )   (la devolución es positiva)
        Si use_atr_norm: divide ambos por ATR_t (en unidades de precio).
        """
        fut_max_high = self.df['high'].shift(-1).rolling(horizon, min_periods=horizon).max().shift(-horizon+1)
        fut_min_low = self.df['low'].shift(-1).rolling(horizon, min_periods=horizon).min().shift(-horizon+1)

        base = self.df['close']
        raw_mfe = (fut_max_high - base).clip(lower=0)
        raw_mae_pos = (base - fut_min_low).clip(lower=0)

        if use_atr_norm:
            denom = self.df['atr'].replace(0, np.nan)
            mfe = (raw_mfe / denom).clip(upper=clip_mult)
            mae_pos = (raw_mae_pos / denom).clip(upper=clip_mult)
        else:
            # normalizar por precio (retorno relativo)
            mfe = ((fut_max_high - base) / base).clip(lower=0, upper=clip_mult / 100)
            mae_pos = ((base - fut_min_low) / base).clip(lower=0, upper=clip_mult / 100)

        self.df['mfe'] = mfe
        self.df['mae'] = mae_pos

        return ['mfe', 'mae']

        #self.y_df = pd.DataFrame({'mfe': mfe, 'mae_pos': mae_pos})

    def check_signals(self):
        self.df['sell'] = np.where(self.df.buy == 0, self.df.sell, 0)

    @staticmethod
    def get_class_weight(y_train):
        y2 = np.asarray(y_train).astype(int)
        y_idx = y2.argmax(axis=1)
        pos_rate = (y_idx == 1).mean()

        classes = np.array([0, 1])
        cw = compute_class_weight('balanced', classes=classes, y=y_idx)
        class_weight = dict(zip(classes, cw))  # {0: w0, 1: w1}
        sample_weight = np.vectorize(class_weight.get)(y_idx)

        #class_weight = {int(c): float(w) for c, w in zip(classes, cw)}
        return class_weight

    def add_signals(self, df, holding_period=15, k1=2.0, k2=1.0, spread=0.21):
        df = df.copy()

        future_high = df.high.shift(-1).rolling(holding_period, min_periods=holding_period).max().shift(-holding_period+1)
        future_low = df.low.shift(-1).rolling(holding_period, min_periods=holding_period).min().shift(-holding_period+1)

        actions = ['check_buy','check_sell']
        for action in actions:
            if action == 'check_buy':
                close = df.close + spread
                take_profit = (future_high - close)
                stop_loss = (close - future_low)
            else:
                close = df.close
                take_profit = (close - future_low)
                stop_loss = (future_high - close)

            target = ((take_profit >= k1 * df.atr) & (stop_loss <= k2 * df.atr)).astype(int)
            df[action] = target

        df['action'] = np.select(
            [
                (df.check_buy == 0) & (df.check_sell == 1),
                (df.check_buy == 1) & (df.check_sell == 0),
                (df.check_buy == 1) & (df.check_sell == 1),
            ], ['sell', 'buy', 'both'],
            default='hold'
        )

        label_encoder = LabelEncoder()
        y_int = label_encoder.fit_transform(df['action'])
        y_onehot = to_categorical(y_int, num_classes=len(label_encoder.classes_))
        y_df = pd.DataFrame(y_onehot,
                            index=df.index,
                            columns=label_encoder.classes_)

        df = df.join(y_df)
        return df, label_encoder.classes_.tolist()

    @staticmethod
    def get_stats(y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        report = classification_report(y_true, y_pred)

        return cm, precision, report

    def add_market_state(self):
        market_state_map = {
            'trending_up': 0,
            'trending_down': 1,
            'ranging': 2,
            'volatile': 3,
            'low_volatility': 4,
            'neutral': 5
        }

        df = self.df.copy()
        df['market_state'] = df.apply(self.classify_market_state, axis=1)

        data = to_categorical(df['market_state'].map(market_state_map))
        df_market_state = pd.DataFrame(
            data, columns=[f'state_{k}' for k in market_state_map.keys()]
        )

        self.df = pd.concat([df, df_market_state], axis=1)

        return df_market_state.columns.to_list()

    @staticmethod
    def classify_market_state(row):
        atr_is_low = row.atr < 0.7 * row.atr_mean
        atr_is_high = row.atr > 1.5 * row.atr_mean
        atr_is_medium_or_high = row.atr >= row.atr_mean
        dm_diff_small = abs(row.dm_plus - row.dm_minus) < 5

        if row.adx > 25 and row.dm_plus > row.dm_minus and row.rsi > 60 and row.bbp > 0.6 and atr_is_medium_or_high:
            market_state = 'trending_up'
        elif row.adx > 25 and row.dm_minus > row.dm_plus and row.rsi < 40 and row.bbp < 0.4 and atr_is_medium_or_high:
            market_state = 'trending_down'
        elif row.adx < 25 and 40 <= row.rsi <= 60 and dm_diff_small and 0.4 <= row.bbp <= 0.6 and row.atr < row.atr_mean:
            market_state = 'ranging'
        elif atr_is_high and row.adx < 20:
            market_state = 'volatile'
        elif atr_is_low and row.adx < 15: # and 45 <= row.rsi <= 55 and 0.45 <= row.bbp <= 0.55:
            market_state = 'low_volatility'
        else:
            market_state = 'neutral'

        return market_state

    def purge(self):
        self.df.dropna(inplace=True)

    def temporal_splits(self, feature_cols, val_size: float=0.2):
        self.df.dropna(inplace=True)

        x_data = self.df[feature_cols].values.astype(np.float32)
        y_buy = self.df.buy.values.astype(np.int32)
        y_sell = self.df.sell.values.astype(np.int32)
        y_mfe = self.df.mfe.values.astype(np.float32)
        y_mae = self.df.mae.values.astype(np.float32)

        split = int(len(self.df) * (1-val_size))

        return (x_data[:split], x_data[split:],
                y_buy[:split], y_buy[split:], y_sell[:split], y_sell[split:],
                y_mfe[:split], y_mfe[split:], y_mae[:split], y_mae[split:]
        )

    def temporal_split(self, feature_cols, val_size: float=0.2):
        data = pd.concat([self.df, self.y_df], axis=1).dropna().reset_index(drop=True)

        X = data[feature_cols].values.astype(np.float32)
        y_mfe = data['mfe'].values.astype(np.float32)  # >=0 (en ATRs)
        y_mae = data['mae_pos'].values.astype(np.float32)  # >=0 (en ATRs)

        # --- Split temporal ---
        split = int(len(X) * (1-val_size))

        return X[:split], X[split:], y_mfe[:split], y_mae[:split], y_mfe[split:], y_mae[split:]

    @staticmethod
    def make_windows(X, y_mfe, y_mae, seq_len=32, stride=1):
        """
        X: np.array [T, F]
        y_*: np.array [T,]
        Genera ventanas causales: la etiqueta del índice i usa la ventana [i-seq_len+1..i]
        """
        T, F = X.shape
        idx_end = np.arange(seq_len - 1, T, stride)
        Xw = np.zeros((len(idx_end), seq_len, F), dtype=np.float32)
        for j, i in enumerate(idx_end):
            Xw[j] = X[i - seq_len + 1:i + 1, :]

        if y_mfe is not None:
            yw_mfe = y_mfe[idx_end]
        else:
            yw_mfe = None

        if y_mae is not None:
            yw_mae = y_mae[idx_end]
        else:
            yw_mae = None

        return Xw, yw_mfe, yw_mae

    @staticmethod
    def make_window(X, seq_len=32, stride=1):
        T, F = X.shape
        idx_end = np.arange(seq_len - 1, T, stride)
        Xw = np.zeros((len(idx_end), seq_len, F), dtype=np.float32)
        for j, i in enumerate(idx_end):
            Xw[j] = X[i - seq_len + 1:i + 1, :]

        return Xw, idx_end

    @staticmethod
    def make_windows_for_signals(X, y_buy, y_sell, seq_len=64, stride=1):
        """
        X: np.array [T, F]
        y_*: np.array [T,]
        Genera ventanas causales: la etiqueta del índice i usa la ventana [i-seq_len+1..i]
        """
        T, F = X.shape
        idx_end = np.arange(seq_len - 1, T, stride)
        Xw = np.zeros((len(idx_end), seq_len, F), dtype=np.float32)
        for j, i in enumerate(idx_end):
            Xw[j] = X[i - seq_len + 1:i + 1, :]

        if y_buy is not None:
            Yw_buy = y_buy[idx_end]
            Yw_sell = y_sell[idx_end]
        else:
            Yw_buy = None
            Yw_sell = None

        return Xw, Yw_buy, Yw_sell


    def load_scaler(self, release):
        self.scaler = joblib.load(f'models/scaler_{release}.pkl')
        #self.scaler = joblib.load(f'models/scaler_{release}.pkl')

    def save_scaler(self, release):
        #joblib.dump(self.scaler, f'models/scaler_{release}.pkl')
        joblib.dump(self.scaler, f'models/scaler_{release}.pkl')

    def fit_scaler(self, x):
        self.scaler = MinMaxScaler() #StandardScaler() #RobustScaler()

        x_flat = x.reshape(-1, x.shape[-1])
        self.scaler.fit(x_flat)

    def fit_transform(self, x):
        self.scaler = RobustScaler()

        x_flat = x.reshape(-1, x.shape[-1])
        x_scaled = self.scaler.fit_transform(x_flat)

        return x_scaled.reshape(x.shape)

    def apply_scaler(self, x2d):
        x_flat = self.scaler.transform(x2d.reshape(-1, x2d.shape[-1]))
        return x_flat.reshape(x2d.shape)

    def save(self, db, table='rates'):
        self.df = self.df.rename(columns={"ticks_volume": "volume"})
        db.save(self.df, table_name=table)

    @staticmethod
    def decide_trade(db, rate_id, row, mfe, mae, r_min=1.5, spread=0.21):
        costs_atr = spread / row.atr.item()

        p_long = row.dm_plus.item() / (row.dm_plus.item() + row.dm_minus.item() + 1e-9)
        w = min(1.0, max(0.0, (row.adx.item() - 15) / 20.0))
        p_long = w * p_long + (1-w) * 0.5

        edge_long = mfe[0][0] - costs_atr
        risk_long = mae[0][0]
        profit_risk_long = edge_long / (risk_long + 1e-9)

        edge_short = mae[0][0]
        risk_short = mfe[0][0]
        profit_risk_short = edge_short / (risk_short + 1e-9)

        do_long = (p_long >= 0.45) and (edge_long > 0) and (profit_risk_long >= r_min)
        do_short = ((1-p_long) >= 0.45) and (edge_short > 0) and (profit_risk_short >= r_min)

        bbp = row.bbp.item()

        print(f'\tdo_long: {do_long}, Prob.: {p_long:.2f}, Edge: {edge_long:.2f}, Risk: {risk_long:.2f}, Ratio: {profit_risk_long:.2f}\n'
              f'\tdo_short: {do_short}, Prob.: {(1-p_long):.2f}, Edge: {edge_short:.2f}, Risk: {risk_short:.2f}, Ratio: {profit_risk_short:.2f}\n'
              f'\tATR: {row.atr.item():.2f}. ADX: {row.adx.item():.2f}. DM(+): {row.dm_plus.item():.2f}. DM(-): {row.dm_minus.item():.2f}. BBP: {bbp:.2f}\n')

        analysis = pd.DataFrame(
            [
                {
                    'rate_id': rate_id,
                    'do_long': do_long,
                    'p_long': p_long,
                    'edge_long': edge_long,
                    'risk_long': risk_long,
                    'profit_risk_long': profit_risk_long,
                    'do_short': do_short,
                    'p_short': 1-p_long,
                    'edge_short': edge_short,
                    'risk_short': risk_short,
                    'profit_risk_short': profit_risk_short,
                    'atr': row.atr.item(),
                    'adx': row.adx.item(),
                    'dm_plus': row.dm_plus.item(),
                    'dm_minus': row.dm_minus.item(),
                    'mae_50': mae[0][0],
                    'mae_80': mae[0][1],
                    'mae_95': mae[0][2],
                    'mfe_50': mfe[0][0],
                    'mfe_80': mfe[0][1],
                    'mfe_95': mfe[0][2]
                }
            ]
        )

        db.save(analysis, table_name='analysis')

        return do_long, p_long, edge_long, risk_long, profit_risk_long, do_short, (1-p_long), edge_short, risk_short, profit_risk_short, row.atr.item(), row.adx.item()

    @staticmethod
    def make_sequences_2(X, y_trade, y_state, window_size: int = 64):
        n_sequences, n_features = X.shape

        starts = np.arange(0, n_sequences - window_size + 1, 1, dtype=int)
        X_seq = np.stack([X[s:s + window_size] for s in starts], axis=0)

        idx = starts + (window_size - 1)
        if y_trade is not None:
            y_trade_seq = y_trade[idx]
        else:
            y_trade_seq = None

        if y_state is not None:
            y_state_seq = y_state[idx]
        else:
            y_state_seq = None

        return X_seq, y_trade_seq, y_state_seq

    @staticmethod
    def make_sequences_3(X, y_should_buy, y_should_sell, y_state, window_size: int = 64):
        n_sequences, n_features = X.shape

        starts = np.arange(0, n_sequences - window_size + 1, 1, dtype=int)
        X_seq = np.stack([X[s:s + window_size] for s in starts], axis=0)

        idx = starts + (window_size - 1)
        if y_should_buy is not None:
            y_should_buy_seq = y_should_buy[idx]
        else:
            y_should_buy_seq = None

        if y_should_sell is not None:
            y_should_sell_seq = y_should_sell[idx]
        else:
            y_should_sell_seq = None

        if y_state is not None:
            y_state_seq = y_state[idx]
        else:
            y_state_seq = None

        return X_seq, y_should_buy_seq, y_should_sell_seq, y_state_seq

    @staticmethod
    def make_sequences_4(X, y_mfe, y_mae, y_state, window_size: int = 64):
        n_sequences, n_features = X.shape

        starts = np.arange(0, n_sequences - window_size + 1, 1, dtype=int)
        X_seq = np.stack([X[s:s + window_size] for s in starts], axis=0)

        idx = starts + (window_size - 1)
        if y_mfe is not None:
            y_mfe_seq = y_mfe[idx]
        else:
            y_mfe_seq = None

        if y_mae is not None:
            y_mae_seq = y_mae[idx]
        else:
            y_mae_seq = None

        if y_state is not None:
            y_state_seq = y_state[idx]
        else:
            y_state_seq = None

        return X_seq, y_mfe_seq, y_mae_seq, y_state_seq

    @staticmethod
    def temporal_split_2(data, val_size, holding_period):
        n_sequences, window_size, n_features = data.shape
        gap = max(window_size - 1, holding_period)
        val_len = int(n_sequences * val_size)

        train_end = n_sequences - val_len
        val_start = n_sequences - val_len + gap

        X_train, X_test = data[:train_end], data[val_start:]
        return X_train, X_test, train_end, val_start


    @staticmethod
    def trade_plan(row, mfe, mae, spread=0.21):
        atr = row.atr.item()
        costs_atr = spread / atr

        profit_long = mfe[0][0] - costs_atr
        loss_long = mae[0][0]

        profit_short = mae[0][0]
        loss_short = mfe[0][0]

        adx = row.adx.item()
        dm_plus = row.dm_plus.item()
        dm_minus = row.dm_minus.item()
        bbp = row.bbp.item()

        print(f'ATR: {atr:.2f}. ')

        if adx < 25:
            if (bbp > 0.60) and (dm_minus > dm_plus + 5):
                return 'sell', profit_short, loss_short
            elif (bbp < 0.40) and (dm_plus > dm_minus + 5):
                return 'buy', profit_long, loss_long
            else:
                return 'none', None, None
        else:
            if dm_plus > dm_minus + 5:
                return 'buy', profit_long, loss_long
            elif dm_minus > dm_plus + 5:
                return 'sell', profit_short, loss_short
            else:
                return 'none', None, None

    def make_only_window(self, last=True, feature_cols=None, window=32):
        if last:
            F = self.df[feature_cols].tail(window).to_numpy()
        else:
            F = self.df[feature_cols].to_numpy()

        n_samples, n_features = F.shape
        n = n_samples - window + 1
        if n <= 0:
            raise ValueError('Data set length must be higher than window size')

        X_all = sliding_window_view(F, window_shape=(window, n_features)).squeeze(1)
        return self.apply_scaler(X_all)

    def make_windows_and_split(self, feature_cols, action_cols, state_cols, window, val_size, shuffle=False):
        F = self.df[feature_cols].to_numpy()
        A = self.df[action_cols].to_numpy()
        S = self.df[state_cols].to_numpy()

        n_samples, n_features = F.shape
        n = n_samples - window + 1
        if n <= 0:
            raise ValueError('Data set length must be higher than window size')

        X_all = sliding_window_view(F, window_shape=(window, n_features)).squeeze()
        y_state_all = S[window-1:]
        y_action_all = A[window-1:]

        X_train, X_val, y_state_train, y_state_val, y_action_train, y_action_val = train_test_split(
            X_all, y_state_all, y_action_all, test_size=val_size, shuffle=shuffle
        )

        X_train = self.fit_transform(X_train)
        X_val = self.apply_scaler(X_val)

        return X_train, X_val, y_state_train, y_state_val, y_action_train, y_action_val

    def prepare(self, seq_len: int=64,
                trade_signals: List[str]=None, trade_metrics: List[str]=None, feature_cols: List[str]=None,
                holding_period: int=5, persistence: int=3, k1: float=0.5, k2: float=2.0,
                spread: float=0.21, val_size: float=0.20, last: bool=False, mode: str='production'):

        feature_builder = FeatureBuilder()
        df, self.feature_cols = feature_builder.build(self.df, lookback_only=True, reduce_noise=True)
        if feature_cols is not None:
            self.feature_cols = feature_cols

        market_state_detector = MarketStateDetector()
        df, self.market_state_classes = market_state_detector.build(df, lookback=100, min_persistence=2)

        trade_signals_generator = TradeSignalsGenerator()
        if mode == 'training':
            df, _ = trade_signals_generator.add_regression_targets(df, holding_period=holding_period, use_atr_norm=True)
            for trade_signal in trade_signals:
                df = trade_signals_generator.add_signals(df, action=trade_signal, holding_period=holding_period, spread=spread, k1=k1, k2=k2)

        df = df.dropna()
        self.df = df

        if last:
            X = df[self.feature_cols].tail(seq_len).values.astype(np.float32)
        else:
            X = df[self.feature_cols].values.astype(np.float32)

        n_sequences, n_features = X.shape

        gap = max(seq_len - 1, holding_period)
        val_len = int(n_sequences * val_size)
        self.train_end = n_sequences - val_len
        self.val_start = n_sequences - val_len + gap

        self.starts = np.arange(0, n_sequences - seq_len + 1, 1, dtype=int)
        X_seq = np.stack([X[s:s + seq_len] for s in self.starts], axis=0)

        X_train, X_val = X_seq[:self.train_end], X_seq[self.val_start:]

        if mode == 'production':
            return X_train, X_val, None

        y_trades = {}
        for trade_signal in trade_signals:
            df[trade_signal] = trade_signals_generator.persistence_consecutive(df[trade_signal], n=persistence)
            y_trade = df[trade_signal].values.astype(np.float32)[seq_len-1:]
            p_pos = float(y_trade.mean())

            result = {}
            result['y_trade'] = y_trade
            result['p_pos'] = p_pos
            result['y_train'] = y_trade[:self.train_end]
            result['y_val'] = y_trade[self.val_start:]

            y_trades[trade_signal] = result

        for trade_metric in trade_metrics:
            y_trade = df[trade_metric].values.astype(np.float32)[seq_len-1:]
            p_pos = float(y_trade.mean())

            result = {}
            result['y_trade'] = y_trade
            result['p_pos'] = p_pos
            result['y_train'] = y_trade[:self.train_end]
            result['y_val'] = y_trade[self.val_start:]

            y_trades[trade_metric] = result

        return X_train, X_val, y_trades

    def prepare_2(self, seq_len: int=64,
                trade_signals: List[str]=None, trade_metrics: List[str]=None, feature_cols: List[str]=None,
                holding_period: int=5, persistence: int=3, k1: float=0.5, k2: float=2.0,
                spread: float=0.21, val_size: float=0.20, last: bool=False, mode: str='production', extra_features=None,
                shuffle: bool=False):

        feature_builder = FeatureBuilder()
        df, self.feature_cols = feature_builder.build(self.df, lookback_only=True, reduce_noise=True, extra_features=extra_features)
        if feature_cols is not None:
            self.feature_cols = feature_cols

        market_state_detector = MarketStateDetector()
        df, self.market_state_classes = market_state_detector.build(df, lookback=100, min_persistence=2)

        trade_signals_generator = TradeSignalsGenerator()
        if mode == 'training':
            df, _ = trade_signals_generator.add_regression_targets(df, holding_period=holding_period, use_atr_norm=True)
            for trade_signal in trade_signals:
                df = trade_signals_generator.add_signals(df, action=trade_signal, holding_period=holding_period, spread=spread, k1=k1, k2=k2)

            buy_pos = (df.buy > 0).sum()
            sell_pos = (df.sell > 0).sum()
            n_total = len(df)

            print(f'Buy ratio: {buy_pos/n_total:.4f}. Sell ratio: {sell_pos/n_total:.4f}')

        df = df.dropna()
        self.df = df

        X = df[self.feature_cols].values.astype(np.float32)
        n_sequences, n_features = X.shape
        X_all = sliding_window_view(X, window_shape=(seq_len, n_features)).squeeze()

        index = df.index[seq_len - 1:]
        if mode == 'training':
            y_cols = trade_signals + trade_metrics + ['state_id']
            y_all = df[y_cols][seq_len-1:].values

            idx_train, idx_val, X_train, X_val, y_train, y_val = train_test_split(index, X_all, y_all, test_size=val_size, shuffle=shuffle)
            return X_train, X_val, y_train, y_val, idx_train, idx_val

        elif val_size > 0.0:
            X_train, X_val, idx_train, idx_val = train_test_split(index, X_all, test_size=val_size, shuffle=shuffle)
            return X_train, X_val, idx_train, idx_val

        else:
            return X_all, index

    def split(self, seq_len, val_size, shuffle=False):
        index = self.dm.df.index[seq_len:]

        idx_train, idx_val = train_test_split(index, val_size=val_size, shuffle=shuffle)
        return idx_train, idx_val

    def prepare_final(self, predictions, seq_len: int,
                      trade_signals: List[str], trade_metrics: List[str], feature_cols: List[str],
                      holding_period: int=5, val_size: float=0.20, last: bool=False):

        df = self.df[63:].copy()
        df[['buy_pred', 'sell_pred', 'mfe_pred', 'mae_pred']] = predictions

        self.df = df

        if last:
            X = df[feature_cols].tail(seq_len).values.astype(np.float32)
        else:
            X = df[feature_cols].values.astype(np.float32)

        n_sequences, n_features = X.shape

        gap = max(seq_len - 1, holding_period)
        val_len = int(n_sequences * val_size)
        self.train_end = n_sequences - val_len
        self.val_start = n_sequences - val_len + gap

        self.starts = np.arange(0, n_sequences - seq_len + 1, 1, dtype=int)
        X_seq = np.stack([X[s:s + seq_len] for s in self.starts], axis=0)

        X_train, X_val = X_seq[:self.train_end], X_seq[self.val_start:]

        y_trades = {}
        for trade_signal in trade_signals:
            y_trade = df[trade_signal].values.astype(np.float32)[seq_len-1:]
            p_pos = float(y_trade.mean())

            result = {}
            result['p_pos'] = p_pos
            result['y_train'] = y_trade[:self.train_end]
            result['y_val'] = y_trade[self.val_start:]

            y_trades[trade_signal] = result

        for trade_metric in trade_metrics:
            y_trade = df[trade_metric].values.astype(np.float32)[seq_len-1:]
            p_pos = float(y_trade.mean())

            result = {}
            result['p_pos'] = p_pos
            result['y_train'] = y_trade[:self.train_end]
            result['y_val'] = y_trade[self.val_start:]

            y_trades[trade_metric] = result

        return X_train, X_val, y_trades

