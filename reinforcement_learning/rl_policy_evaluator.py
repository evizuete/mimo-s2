# rl_policy_evaluator.py
from __future__ import annotations

import json
import os
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

import numpy as np

from reinforcement_learning.rl_experiment import RLExperiment, RLRunConfig


@dataclass
class EvalObjective:
    """
    Define cómo puntuar un trial.

    score = net_pnl
            - lambda_dd * abs(max_dd_pct) * initial_equity
            - lambda_stop * stop_days (si lo añades más adelante)
    """
    lambda_dd: float = 2.0
    # reservado para futuras penalizaciones
    lambda_stop: float = 0.0


def _safe_float(x, default=np.nan) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


def compute_profit_factor_and_winrate(trades: Any) -> Tuple[float, float]:
    """
    ProfitFactor = sum(gains)/sum(losses_abs)
    WinRate = wins / total
    Acepta trades como list[dict] o DataFrame-like.
    """
    pnls = []
    if isinstance(trades, list):
        for t in trades:
            if isinstance(t, dict) and "pnl" in t:
                pnls.append(_safe_float(t["pnl"], default=np.nan))
    else:
        # pandas DataFrame
        try:
            if hasattr(trades, "columns") and "pnl" in trades.columns:
                pnls = [float(x) for x in trades["pnl"].astype(float).tolist()]
        except Exception:
            pnls = []

    pnls = [p for p in pnls if np.isfinite(p)]
    if not pnls:
        return (float("nan"), float("nan"))

    pnl = np.array(pnls, dtype=np.float64)
    gp = pnl[pnl > 0].sum()
    gl = -pnl[pnl < 0].sum()
    pf = (gp / gl) if gl > 1e-12 else (float("inf") if gp > 0 else float("nan"))
    wr = float((pnl > 0).mean())
    return float(pf), float(wr)


def compute_max_drawdown_pct_from_equity(equity_curve: Any) -> float:
    eq = np.asarray(equity_curve, dtype=np.float64)
    if eq.size < 2:
        return 0.0
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / np.maximum(peak, 1e-12)
    return float(np.min(dd))


class RLPolicyEvaluator:
    """
    Clase “evaluador”:
      - entrena una policy en un periodo train
      - evalúa esa policy en un periodo eval (out-of-sample)
      - devuelve métricas y score (objetivo)
      - guarda artefactos por trial (json de métricas, policy npz, etc.)
    """

    def __init__(
        self,
        *,
        experiment: RLExperiment,
        train_from: datetime,
        train_to: datetime,
        eval_from: datetime,
        eval_to: datetime,
        outdir: str,
        objective: Optional[EvalObjective] = None,
    ) -> None:
        self.exp = experiment
        self.train_from = train_from
        self.train_to = train_to
        self.eval_from = eval_from
        self.eval_to = eval_to
        self.outdir = outdir
        self.objective = objective or EvalObjective()

        os.makedirs(self.outdir, exist_ok=True)

    def train_and_eval(
        self,
        *,
        run: RLRunConfig,
        tag: str,
        export_trade_log: bool = False,
    ) -> Dict[str, Any]:
        """
        Ejecuta:
          1) TRAIN: exp.run_backtest(mode="train") -> guarda policy npz
          2) EVAL : exp.run_backtest(mode="eval")  -> métricas
        """
        trial_dir = os.path.join(self.outdir, tag)
        os.makedirs(trial_dir, exist_ok=True)

        # Ajusta policy_path dentro del trial_dir si el usuario pasó una ruta genérica
        policy_path = run.policy_path
        if not policy_path or os.path.dirname(policy_path) in ("", ".", None):
            policy_path = os.path.join(trial_dir, "rl_policy.npz")

        run_train = RLRunConfig(**{**run.__dict__, "policy_path": policy_path, 'rl_train': True})
        run_eval = RLRunConfig(**{**run.__dict__, "policy_path": policy_path, 'rl_train': False})

        '''
        print("EVAL policy_path:", run_eval.policy_path)
        print("exists:", os.path.exists(run_eval.policy_path), "size:", os.path.getsize(run_eval.policy_path))
        print("npz keys:", list(np.load(run_eval.policy_path).keys()))
        print("EVAL rl_train:", run_eval.rl_train, "use_rl:", run_eval.use_rl)
        print("EVAL take_thr:", run_eval.rl_take_threshold)
        '''

        # 1) TRAIN (guarda policy npz)
        train_res = self.exp.run_backtest(
            from_date=self.train_from,
            to_date=self.train_to,
            mode="train",
            run=run_train,
            debug=False,
        )

        # 2) EVAL
        eval_res = self.exp.run_backtest(
            from_date=self.eval_from,
            to_date=self.eval_to,
            mode="eval",
            run=run_eval,
            debug=True,
        )

        metrics = self._summarize(eval_res, run_eval)

        # Score (objetivo)
        score = self.score(metrics, run_eval)
        metrics["score"] = float(score)

        payload = {
            "tag": tag,
            "run": asdict(run_eval),
            "train_period": {"from": self.train_from.isoformat(), "to": self.train_to.isoformat()},
            "eval_period": {"from": self.eval_from.isoformat(), "to": self.eval_to.isoformat()},
            "metrics": metrics,
            "policy_path": policy_path,
        }

        # (Opcional) export trade-log/candidates para análisis
        if export_trade_log:
            # Llamada robusta: no asumimos parámetros extra
            try:
                trade_log_path = os.path.join(trial_dir, "trade_log.csv")
                self.exp.export_trade_log(
                    from_date=self.eval_from,
                    to_date=self.eval_to,
                    run=run_eval,
                    out_path=trade_log_path,
                )
                payload["trade_log_path"] = trade_log_path
            except TypeError:
                # Algunas versiones pueden tener firma distinta (p.ej. positional args)
                try:
                    self.exp.export_trade_log(self.eval_from, self.eval_to, run_eval, trade_log_path)
                    payload["trade_log_path"] = trade_log_path
                except Exception as e2:
                    payload["trade_log_error"] = f"{type(e2).__name__}: {e2}"
            except Exception as e:
                payload["trade_log_error"] = f"{type(e).__name__}: {e}"

        # Guardar artefacto del trial
        out_json = os.path.join(trial_dir, "metrics.json")
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        return payload

    def _summarize(self, res: Dict[str, Any], run: RLRunConfig) -> Dict[str, Any]:
        n_trades = int(res.get("n_trades", 0))
        net_pnl = _safe_float(res.get("net_pnl", 0.0), default=0.0)
        final_balance = _safe_float(res.get("final_balance", np.nan))
        final_equity_mtm = _safe_float(res.get("final_equity_mtm", np.nan))

        # ProfitFactor / WinRate
        pf, wr = compute_profit_factor_and_winrate(res.get("trades", []))

        # MaxDD (%): preferimos equity_curve_mtm si está
        eq = res.get("equity_curve_mtm") if res.get("equity_curve_mtm") is not None else res.get("balance_curve")
        mdd_pct = compute_max_drawdown_pct_from_equity(eq) if eq is not None else float("nan")

        # Sharpe/Sortino si vienen
        ss = res.get("sharpe_sortino_metrics") or {}
        sharpe = float("nan")
        sortino = float("nan")
        if isinstance(ss, dict):
            sharpe = _safe_float(ss.get("sharpe_daily_ann", np.nan))
            sortino = _safe_float(ss.get("sortino_daily_ann", np.nan))

        return {
            "n_trades": n_trades,
            "net_pnl": float(net_pnl),
            "profit_factor": float(pf) if np.isfinite(pf) else float(pf),
            "win_rate": float(wr) if np.isfinite(wr) else float(wr),
            "max_dd_pct": float(mdd_pct),
            "final_balance": float(final_balance) if np.isfinite(final_balance) else float("nan"),
            "final_equity_mtm": float(final_equity_mtm) if np.isfinite(final_equity_mtm) else float("nan"),
            "sharpe_daily_ann": float(sharpe),
            "sortino_daily_ann": float(sortino),
            # Eco de parámetros clave para lectura rápida
            "max_daily_loss_pct": float(run.max_daily_loss_pct) if run.max_daily_loss_pct is not None else None,
            "rl_take_threshold": float(run.rl_take_threshold),
            "compound": bool(run.compound),
            "sizing_equity_mode": str(run.sizing_equity_mode),
        }

    def score(self, metrics: Dict[str, Any], run: RLRunConfig) -> float:
        """
        Score por defecto: NetPnL penalizando drawdown.
        (max_dd_pct es negativo; tomamos abs).
        """
        net = _safe_float(metrics.get("net_pnl", 0.0), default=0.0)
        mdd_pct = _safe_float(metrics.get("max_dd_pct", 0.0), default=0.0)

        dd_pen = self.objective.lambda_dd * abs(mdd_pct) * float(run.initial_equity)
        return float(net - dd_pen)
