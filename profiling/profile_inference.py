from contextlib import contextmanager
import time

from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.trading_simulator import TradingSimulator


@contextmanager
def timer(name):
    t0 = time.perf_counter()
    yield
    t1 = time.perf_counter()
    print(f'⏱️  {name}: {(t1-t0)*1000:.2f}ms')

def profile_engine_initialization(release):
    print('\n' + '='*120)
    print('PROFILIN: Starting engine...')
    print('='*120)

    with timer('1. Configuraciones'):
        general_config = Config(release=release, oof_splits=3, oof_epochs=60)
        model_config = ModelConfig(seq_len_short=64, seq_len_long=256, batch_size=8192)
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
        regime_config = RegimeConfig(adx_trend_threshold=25.0)
        decision_policy = DecisionPolicy(
            gate_by_action_and_state={
                "long": {'trend_up': 50, 'transition_up': 50, 'range': 50, '_global': 50},
                "short": {'trend_down': 50, 'transition_down': 50, 'range': 50, '_global': 50},
            }
        )
        risk_config = RiskConfig(
            base_risk_pct=0.005,  # 0.5% equity
            min_score_to_trade=0.00,  # 0.25,
            max_risk_pct=0.02,
            max_positions=2
        )

    with timer('2. Crear TradingSimulator'):
        engine = TradingSimulator(
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            regime_config=regime_config,
            decision_policy=decision_policy,
            risk_config=risk_config,
            artifacts_path=f'./artifacts/{release}/oof/deploy_full',
            use_rl=False,
            spread_price=0.07,
            enable_live_scaler_updates=True
        )

    with timer('3. Cargar artifacts (modelos + scalers)'):
        engine.load_artifacts()

    print('\n✅ Inicializacion completada\n')
    return engine

def profile_decide_live(engine, df_rates):
    print("\n" + "=" * 70)
    print("PROFILING: decide_live() - POR TICK")
    print("=" * 70)

    with timer("TOTAL decide_live"):
        live_order = engine.decide_live(
            df_rates=df_rates,
            equity=10000,
            current_positions=0,
            max_positions=20
        )

    print(f"\nResultado: {live_order}")
    return live_order

