# mimo_old/rl_experiment.py
# High-level reusable runner built from your main_kk_train.py logic.
#
# It encapsulates:
#   - building TradingSimulator with RL wrapper
#   - RL train run (saving policy .npz)
#   - RL eval run (loading policy .npz)
#   - candidate export for threshold sweeping (per-trade or per-candidate)
#
# IMPORTANT:
#   This module assumes your TradingSimulator / rl_wrapper APIs as in your project.

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from mimo_old.trading_simulator import TradingSimulator, load_rl_policy_npz, save_rl_policy_npz
from reinforcement_learning.rl_trade_log import TradeLogSpec, build_trade_log_for_sweep, save_trade_log


@dataclass
class RLRunConfig:
    # Policy storage
    policy_path: str

    # Simulator toggles
    use_rl: bool = True
    rl_train: bool = True
    rl_eval_deterministic: bool = False
    rl_take_threshold: float = 0.50

    # Trading / sim params
    initial_equity: float = 10_000.0
    spread_price: float = 0.21
    mtm_use_bid_ask: bool = True
    mtm_price_col: str = "close"
    sizing_equity_mode: str = "balance"
    compound: bool = True

    # Risk firewall
    max_daily_loss_pct: Optional[float] = 0.03
    max_daily_profit_pct: Optional[float] = None


class RLExperiment:
    def __init__(
        self,
        *,
        general_config: Any,
        model_config: Any,
        feature_config: Any,
        regime_config: Any,
        decision_policy: Any,
        risk_config: Any,
        artifacts_path: str,
        rl_config: Dict[str, Any],
    ):
        self.general_config = general_config
        self.model_config = model_config
        self.feature_config = feature_config
        self.regime_config = regime_config
        self.decision_policy = decision_policy
        self.risk_config = risk_config
        self.artifacts_path = artifacts_path
        self.rl_config = rl_config

    def _build_sim(self, run: RLRunConfig) -> TradingSimulator:
        sim = TradingSimulator(
            general_config=self.general_config,
            model_config=self.model_config,
            feature_config=self.feature_config,
            regime_config=self.regime_config,
            decision_policy=self.decision_policy,
            risk_config=self.risk_config,
            artifacts_path=self.artifacts_path,
            use_rl=run.use_rl,
            rl_config=self.rl_config,
            rl_train=run.rl_train,
            rl_eval_deterministic=run.rl_eval_deterministic,
            rl_take_threshold=run.rl_take_threshold,
            rl_policy_path=run.policy_path,
            spread_price=run.spread_price,
            mtm_use_bid_ask=run.mtm_use_bid_ask,
            mtm_price_col=run.mtm_price_col,
            sizing_equity_mode=run.sizing_equity_mode,
            max_daily_loss_pct=run.max_daily_loss_pct,
            max_daily_profit_pct=run.max_daily_profit_pct,
            compound=run.compound,
        )
        sim.load_artifacts()
        return sim

    def run_backtest(
        self,
        *,
        from_date: datetime,
        to_date: datetime,
        mode: str,  # 'train' or 'eval'
        run: RLRunConfig,
        debug: bool = False,
    ) -> Dict[str, Any]:
        if mode not in ("train", "eval"):
            raise ValueError("mode must be 'train' or 'eval'")

        run = RLRunConfig(**{**run.__dict__, "rl_train": (mode == "train")})
        sim = self._build_sim(run)

        df = sim.helper.load_from_database_historical(from_date=from_date, to_date=to_date)

        if mode == "eval":
            ok = load_rl_policy_npz(sim.rl_wrapper, run.policy_path)
            if not ok:
                raise RuntimeError(f"RL policy was not loaded from {run.policy_path}")

        # Predicts + features (so decision_engine sees probabilities)
        df = sim.predict(df, simulation=True)

        if debug:
            self._debug_candidate_distribution(sim, df)

        res = sim.backtest(df, initial_equity=run.initial_equity)

        if mode == "train":
            save_rl_policy_npz(sim.rl_wrapper, run.policy_path)

        return res

    def export_candidates(
        self,
        *,
        from_date: datetime,
        to_date: datetime,
        run: RLRunConfig,
        out_path: str,
        include_none: bool = False,
    ) -> str:
        """
        Exports candidate rows for sweeping thresholds.
        Output is bar-level candidates, one row per bar where DecisionEngine proposes an action.
        Columns:
          time,state,action,rl_score,price(o/h/l/c),p_buy/p_sell (raw+cal),atr,adx,trend_dir
        """
        sim = self._build_sim(run)
        ok = load_rl_policy_npz(sim.rl_wrapper, run.policy_path)
        if not ok:
            raise RuntimeError(f"RL policy was not loaded from {run.policy_path}")

        df = sim.helper.load_from_database_historical(from_date=from_date, to_date=to_date)
        df = sim.predict(df, simulation=True)

        rows = []
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
                adx14=float(row.adx) if "adx" in row else None,
                trend_dir=row.trend_dir if "trend_dir" in row else None,
            )

            if (decision.action == "none") and (not include_none):
                continue

            if decision.action == "none":
                rl_score = float("nan")
            else:
                s = sim.rl_wrapper.build_state(row, decision)
                rl_score = float(sim.rl_wrapper.policy.probs(s)[1])  # p(TAKE)

            rows.append({
                "time": row.time if "time" in row else row.get("timestamp", None),
                "state": str(getattr(decision, "state", getattr(decision, "regime", ""))),
                "action": str(decision.action),
                "rl_score": rl_score,
                "open": float(row.open),
                "high": float(row.high),
                "low": float(row.low),
                "close": float(row.close),
                "pred_long_raw": float(row.pred_long_raw),
                "pred_short_raw": float(row.pred_short_raw),
                "pred_long_cal": float(row.pred_long_cal),
                "pred_short_cal": float(row.pred_short_cal),
                "atr": float(row.atr) if "atr" in row else float("nan"),
                "adx": float(row.adx) if "adx" in row else float("nan"),
                "trend_dir": row.trend_dir if "trend_dir" in row else None,
            })

        out_df = pd.DataFrame(rows)
        if out_path.lower().endswith(".parquet"):
            out_df.to_parquet(out_path, index=False)
        else:
            out_df.to_csv(out_path, index=False)

        return out_path


    def export_trade_log(
        self,
        *,
        from_date: datetime,
        to_date: datetime,
        run: RLRunConfig,
        out_path: str,
        spec: TradeLogSpec = TradeLogSpec(),
        score_field: str = "p_take",
    ) -> str:
        """
        Builds a TRADE-level dataset for sweeping RL thresholds by state.

        It runs a backtest (with RL enabled as configured in run), then attaches an RL score
        to each trade by recomputing the RL gate score at the trade entry bar.

        Output columns (minimum):
          time, state, action, rl_score, pnl

        Notes:
        - This is a FAST, approximate dataset if you later filter trades without re-simulating.
          Recommended workflow:
            1) use this to sweep and pick top thresholds per state
            2) validate the best few thresholds by re-running full backtests.
        """
        sim = self._build_sim(run)

        # Load data & predict bars (needed to recompute RL score at entry)
        df = sim.helper.load_from_database_historical(from_date=from_date, to_date=to_date)
        df_pred = sim.predict(df, simulation=True)

        # Run backtest to obtain trades and PnL
        res = sim.backtest(df_pred, initial_equity=run.initial_equity)

        trade_log = build_trade_log_for_sweep(
            backtest_result=res,
            df_bars_predicted=df_pred,
            sim=sim,
            spec=spec,
        )

        return save_trade_log(trade_log, out_path)

    def _debug_candidate_distribution(self, sim: TradingSimulator, df: pd.DataFrame) -> None:
        p_takes = []
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
                adx14=float(row.adx) if "adx" in row else None,
                trend_dir=row.trend_dir if "trend_dir" in row else None,
            )
            if decision.action == "none":
                continue
            s = sim.rl_wrapper.build_state(row, decision)
            p_take = float(sim.rl_wrapper.policy.probs(s)[1])
            p_takes.append(p_take)

        if not p_takes:
            print("[RLExperiment] No candidates found for debug distribution.")
            return

        p = np.array(p_takes, dtype=np.float32)
        print(f"[RLExperiment] n candidates: {len(p)}")
        print(f"[RLExperiment] min/max p(TAKE): {p.min():.4f} / {p.max():.4f}")
        print(f"[RLExperiment] p5/p10/p25: {np.percentile(p, [5, 10, 25])}")
