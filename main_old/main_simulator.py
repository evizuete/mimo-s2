from datetime import datetime, timedelta

import numpy as np

from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo_old.trading_simulator import TradingSimulator, load_rl_policy_npz, save_rl_policy_npz

def collect_p_take(sim, df):
    p = []
    df = sim.predict(df, simulation=True)
    for _, row in df.iterrows():
        decision = sim.decision_engine.decide_at_bar(
            p_buy_raw=float(row.pred_long_raw),
            p_sell_raw=float(row.pred_short_raw),
            p_buy_cal=float(row.pred_long_cal),
            p_sell_cal=float(row.pred_short_cal),
            market_condition=str(row.market_condition),
            o=float(row.open),
            h=float(row.high),
            l=float(row.low),
            c=float(row.close),
            atr=float(row.atr),
            adx14=float(row.adx) if 'adx' in row else None,
            trend_dir=row.trend_dir if 'trend_dir' in row else None
        )

        if decision.action == 'none':
            continue
        s = sim.rl_wrapper.build_state(row, decision)
        p_take = float(sim.rl_wrapper.policy.probs(s)[1])
        p.append(p_take)
    return np.array(p, dtype=np.float32)

def print_results(res, mode, threshold):
    pnl = res['net_pnl']
    pnl_rel = pnl / res['summary']['initial_equity']
    n_trades = res['n_trades']
    win_rate = res['summary']['win_rate']
    profit_factor = res['summary']['profit_factor']
    dd = res['drawdown_stats']['max_drawdown']
    dd_rel = res['drawdown_stats']['max_drawdown_pct']

    print(f'Mode {mode:>8}. Threshold: {threshold:.4f}. Trades: {n_trades:>5}. PNL: {pnl:>8.2f}. PNL (%): {pnl_rel:.4f}. '
          f'WinRate: {win_rate:.4f}. ProfitFactor: {profit_factor:.4f}. Drawdown: {dd:.2f}. Drawdown (%): {dd_rel:.4f}')

def evaluate(from_date, to_date, mode, debug, max_daily_loss_pct, general_config, model_config, feature_config, regime_config, decision_policy, risk_config, artifacts_path,
             use_rl, rl_config, rl_train, rl_eval_deterministic, rl_take_threshold):

    policy_path = f'./artifacts/rl_policy_gate_{general_config.release}.npz'
    sim = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=use_rl,
        rl_config=rl_config,
        rl_train=rl_train,
        rl_eval_deterministic=rl_eval_deterministic,
        rl_take_threshold=rl_take_threshold,
        rl_policy_path=policy_path,
        spread_price=0.21,
        mtm_use_bid_ask=True,
        mtm_price_col='close',
        sizing_equity_mode='balance',
        max_daily_loss_pct=max_daily_loss_pct,
        max_daily_profit_pct=None,
        # max_risk_money=1e12,
        # max_qty=1e12,
        compound=True,
    )

    sim.load_artifacts()
    df = sim.helper.load_from_database_historical(from_date=from_date, to_date=to_date)
    p_takes = None

    if mode == 'eval':
        ok = load_rl_policy_npz(sim.rl_wrapper, policy_path)
        if not ok:
            raise f'WARNING! RL Policy was not loaded from {policy_path}'

        if not hasattr(evaluate, '_first_time'):
            evaluate._first_time = True

        if debug and evaluate._first_time:
            p_takes = collect_p_take(sim, df)
            print(f'n candidates: {len(p_takes)}')
            print(f'min/max: {p_takes.min()} / {p_takes.max()}')
            print(f'p5/p10/p25: {np.percentile(p_takes, [5, 10, 25])}')

            evaluate._first_time = False

    res = sim.backtest(df, initial_equity=10_000.0)
    if mode == 'train':
        save_rl_policy_npz(sim.rl_wrapper, policy_path)

    print_results(res, mode=mode, threshold=rl_take_threshold)

    return res

def calibrate_threshold(sim, df, percentile=50):
    all_probs = []

    for _, row in df.iterrows():
        decision = sim.decision_engine.decide_at_bar(
            p_buy_raw=float(row.pred_long_raw),
            p_sell_raw=float(row.pred_short_raw),
            p_buy_cal=float(row.pred_long_cal),
            p_sell_cal=float(row.pred_short_cal),
            market_condition=str(row.market_condition),
            o=float(row.open),
            h=float(row.high),
            l=float(row.low),
            c=float(row.close),
            atr=float(row.atr),
            adx14=float(row.adx) if 'adx' in row else None,
            trend_dir=row.trend_dir if 'trend_dir' in row else None
        )

        if decision.action != 'none':
            state = sim.rl_wrapper.build_state(row, decision)
            probs = sim.rl_wrapper.policy.probs(state)
            all_probs.append(probs[1])

    if all_probs:
        threshold = np.percentile(all_probs, percentile)
        print(f'Threshold sugerido: (percentil {percentile}: {threshold:.3f}')
        print(f"Rango de p(TAKE): [{min(all_probs):.3f}, {max(all_probs):.3f}]")
        print(f"Media de p(TAKE): {np.mean(all_probs):.3f}")
        return threshold
    return 0.50


def main():
    general_config = Config(
        release="200284",
        oof_splits=12,
        oof_epochs=20
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        batch_size=4096
    )

    feature_config = FeatureConfig()     # usa tus defaults
    regime_config = RegimeConfig()

    decision_policy = DecisionPolicy(
        gate_by_action_and_regime={
            'long': {
                'breakout_wait_up': 97,
                'breakout_wait_down': 99,
                'range': 98,
                'transition_up': 98,
                'transition_down': 99,
                'trend_up': 95,
                'trend_down': 99,
                'volatile': 99,
                '_global': 98
            },
            'short': {
                'breakout_wait_down': 95,
                'breakout_wait_up': 99,
                'range': 97,
                'transition_down': 97,
                'transition_up': 98,
                'trend_down': 95,
                'trend_up': 98,
                'volatile': 99,
                '_global': 97
            }
            #'long': {'range': 95, 'transition_up': 97, 'breakout_wait_up': 98, 'trend_up': 95, 'volatile': 99},
            #'short': {'range': 95, 'transition_down': 97, 'breakout_wait_down': 98, 'trend_down': 95, 'volatile': 99}
            #"long": {"trend": 90, "range": 95, "volatile": 99},
            #"short": {"trend": 95, "range": 90, "volatile": 99},
        },
        score_low_quantile=50,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.15,
        allow_volatile=False
    )

    risk_config = RiskConfig(
        base_risk_pct=0.005,  # 0.5% equity
        min_score_to_trade=0.25,
        max_risk_pct=0.02,
        max_positions=1
    )

    artifacts_path = "../artifacts"
    rl_config = {
        'lr': 5e-4,
        'entropy_coef': 1e-3,
        'baseline_beta': 0.95,
        'max_grad_norm': 10.0,
        'trade_cost_money': 0.5,
        'batch_size': 256
    }

    date2 = datetime(2025, 12, 28)
    date1 = date2 - timedelta(days=90)
    date0 = date1 - timedelta(days=90)

    res_train = evaluate(from_date=date0, to_date=date1, mode='train', max_daily_loss_pct=0.03, debug=True, general_config=general_config,
                         model_config=model_config, feature_config=feature_config, regime_config=regime_config,
                         decision_policy=decision_policy, risk_config=risk_config, artifacts_path=artifacts_path,
                         use_rl=True, rl_config=rl_config, rl_train=True, rl_eval_deterministic=False,
                         rl_take_threshold=0.52)

    start_num = 0.4900
    stop_num = 0.5100
    step: float = 0.001
    n_steps: int = round((stop_num - start_num) / step + 1.0)
    thresholds = np.linspace(start_num, stop_num, n_steps)
    for threshold in thresholds:
        threshold = round(threshold, 4)
        res_eval = evaluate(from_date=date1, to_date=date1 + timedelta(days=30), mode='eval', max_daily_loss_pct=0.03, debug=True, general_config=general_config,
                             model_config=model_config, feature_config=feature_config, regime_config=regime_config,
                             decision_policy=decision_policy, risk_config=risk_config, artifacts_path=artifacts_path,
                             use_rl=True, rl_config=rl_config, rl_train=False, rl_eval_deterministic=True,
                             rl_take_threshold=threshold)

    print('Done')

if __name__ == '__main__':
    main()
