from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass
from typing import Optional, Dict, Any, Tuple, Union, List

import numpy as np
import pandas as pd

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.databases import Database
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig, DecisionOutput, AntiNaturalConfig, DecisionEngine
from mimo.data_managers.entities import DecisionEvent
from mimo.features.feature_builder import FeatureConfig
from mimo.helpers.helper import Helper
from mimo.helpers.instruments import INSTRUMENT_SPECS
from mimo.helpers.margin import required_margin
from mimo.models.model_builder import Config, ModelConfig
from mimo.models.model_evaluator import ModelEvaluator
from mimo.strategies.regime_detector import RegimeConfig
from mimo.strategies.strategy_gates import StrategyGate
from mimo.strategies.prediction_cache import PredictionCache
from mimo.models.numba_utils import (
    check_exit_simple_numba,
    check_exit_advanced_numba,
    batch_unrealized_pnl_numba,
    batch_check_exits_simple_numba
)

# =============================
# RL on top of backtest (per-trade closed reward)
# =============================

# =============================
# RL gate-only (trade cerrado) — decide TAKE / SKIP, NO cambia el lado
# =============================

A_SKIP = 0  # no operar
A_TAKE = 1  # ejecutar la señal base (buy/sell) tal cual


def debug_validate_rl_config_dict(rl_config: dict | None):
    """
    Comprueba que el dict que llega a TradingSimulator(rl_config=...) usa las keys correctas
    (las de RLConfig: lr, entropy_coef, baseline_beta, trade_cost_money, batch_size, etc.)
    y NO las de argparse (rl_lr, rl_entropy, ...), porque esas NO casan con RLConfig(**dict).
    """
    if not rl_config:
        print("[RL][WARN] rl_config is empty/None -> RLConfig defaults will be used.")
        return

    valid = set(RLConfig.__annotations__.keys())
    incoming = set(rl_config.keys())

    unknown = sorted(incoming - valid)
    missing = sorted(valid - incoming)

    if unknown:
        print("[RL][WARN] rl_config has unknown keys (will crash if passed to RLConfig(**dict) "
              "or will be ignored if someone filtered earlier):", unknown)

    # No es obligatorio incluir todas, pero ayuda a ver qué estás pasando realmente.
    print("[RL][DEBUG] rl_config keys:", sorted(incoming))
    print("[RL][DEBUG] rl_config missing keys (not necessarily a problem):", missing[:10],
          "..." if len(missing) > 10 else "")


@dataclass
class RLConfig:
    # Optimización REINFORCE (bandit) sobre reward en R-multiple
    lr: float = 2e-3
    entropy_coef: float = 1e-3
    baseline_beta: float = 0.99
    max_grad_norm: float = 5.0

    # Penalización fija por trade (en dinero). Se resta del pnl antes de pasar a R.
    trade_cost_money: float = 0.0

    # Actualización por batch para reducir varianza
    batch_size: int = 256

    chop_soft_thr: float = 0.60
    exhaustion_soft_thr: float = 0.60
    chop_penalty_coef: float = 0.08
    exhaustion_penalty_coef: float = 0.06
    rl_take_threshold: float = 0.25
    rl_train_threshold: float = 0.05
    seed: int = 42
    update_frequency: int = 32
    # Reward asignado a A_SKIP para evitar colapso hacia "nunca operar".
    # Valor pequeño positivo: la policy aprende que skipear no es gratis pero
    # tampoco tan bueno como un trade ganador (~2.5R).
    skip_reward: float = 0.1


import os


def save_rl_policy_npz(wrapper: 'RLDecisionWrapperGate', path: str) -> None:
    """Persist RL gate policy (W, b, baseline) to .npz."""
    if wrapper is None or getattr(wrapper, "policy", None) is None:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    import numpy as _np
    _np.savez(
        path,
        W=wrapper.policy.W,
        b=wrapper.policy.b,
        baseline=_np.array([float(wrapper.policy.baseline)], dtype=_np.float32),
    )


def load_rl_policy_npz(wrapper: 'RLDecisionWrapperGate', path: str) -> bool:
    """Load RL gate policy from .npz. Returns True if loaded."""
    if wrapper is None or (not os.path.exists(path)):
        return False
    import numpy as _np
    data = _np.load(path, allow_pickle=False)
    W = data["W"].astype(_np.float32)
    b = data["b"].astype(_np.float32)
    baseline = float(data["baseline"][0])

    if getattr(wrapper, "policy", None) is None:
        state_dim = int(W.shape[0])
        wrapper.policy = SoftmaxBanditPolicy(
            state_dim=state_dim,
            n_actions=2,
            cfg=wrapper.cfg,
            seed=wrapper.seed,
        )
        print(f"\t[load_rl_policy_npz] Created policy with state_dim={state_dim}")

    wrapper.policy.W = W
    wrapper.policy.b = b
    wrapper.policy.baseline = baseline

    print(f"\t[load_rl_policy_npz] Loaded W_sum={W.sum():.6f}, baseline={baseline:.6f}")

    return True


class SoftmaxBanditPolicy:
    """Política softmax lineal entrenada con REINFORCE (bandit). Soporta n_actions."""

    def __init__(self, state_dim: int, n_actions: int, cfg: RLConfig, seed: int = 7):
        self.cfg = cfg
        self.n_actions = int(n_actions)
        self.rng = np.random.default_rng(seed)
        self.W = (0.01 * self.rng.standard_normal((state_dim, self.n_actions))).astype(np.float32)
        self.b = np.zeros((self.n_actions,), dtype=np.float32)
        self.baseline = 0.0

        self.update_step = 0
        self.entropy_annealing_steps = 1000  # Reducir de 5000 a 1000
        self.entropy_coef_initial = float(cfg.entropy_coef)
        # Mínimo del 10% del valor inicial — evita colapso de exploración
        self.entropy_coef_min = float(cfg.entropy_coef) * 0.10

    @staticmethod
    def _softmax(z: np.ndarray) -> np.ndarray:
        z = z - np.max(z)
        e = np.exp(z)
        return e / (np.sum(e) + 1e-12)

    def probs(self, s: np.ndarray) -> np.ndarray:
        z = s @ self.W + self.b
        return self._softmax(z)

    def sample(self, s: np.ndarray) -> tuple[int, np.ndarray]:
        p = self.probs(s)
        a = int(self.rng.choice(self.n_actions, p=p))
        return a, p

    def update(self, s: np.ndarray, a: int, reward: float) -> Dict[str, float]:
        # baseline EMA
        reward = float(reward)
        baseline_old = float(self.baseline)
        adv = reward - baseline_old

        self.baseline = float(self.cfg.baseline_beta) * baseline_old + (1.0 - float(self.cfg.baseline_beta)) * reward

        p = self.probs(s)
        one = np.zeros_like(p);
        one[int(a)] = 1.0
        dlogp_dz = (one - p)

        gW = np.outer(s, dlogp_dz) * adv
        gb = dlogp_dz * adv

        # entropy bonus: empuja suavemente hacia uniforme (exploración)
        progress = min(1.0, self.update_step / self.entropy_annealing_steps)
        entropy_coef = self.entropy_coef_initial * (1.0 - 0.9 * progress)
        # Floor: no bajar del 10% del valor inicial para evitar colapso de exploración
        entropy_coef = max(entropy_coef, self.entropy_coef_min)

        if entropy_coef > 1e-6:
            target = np.ones_like(p) / float(len(p))
            g_ent = (target - p)
            gW += entropy_coef * np.outer(s, g_ent)
            gb += entropy_coef * g_ent

        # clip
        gn = float(np.sqrt(np.sum(gW * gW) + np.sum(gb * gb)) + 1e-12)
        clipped = False
        if float(self.cfg.max_grad_norm) and gn > float(self.cfg.max_grad_norm):
            scale = float(self.cfg.max_grad_norm) / gn
            gW *= scale
            gb *= scale
            clipped = True

        # SGD ascent
        self.W += float(self.cfg.lr) * gW.astype(np.float32)
        self.b += float(self.cfg.lr) * gb.astype(np.float32)

        ent = float(-np.sum(p * np.log(p + 1e-12)))

        self.update_step += 1
        if self.update_step % 100 == 0:  # cada 100 updates
            print(f"[Policy] step={self.update_step}, entropy_coef={entropy_coef:.6f}, entropy={ent:.4f}")

        return {
            'reward': float(reward),
            'adv': float(adv),
            'entropy': ent,
            'baseline': float(self.baseline),
            'baseline_old': float(baseline_old),
            'entropy_coef': float(entropy_coef),
            'grad_norm': float(gn),
            'grad_clipped': bool(clipped)
        }


class RLDecisionWrapperGate:
    """
    RL SOLO como 'gate':
      - Decide TAKE / SKIP sobre la señal propuesta por el DecisionEngine.
      - NO cambia la dirección (buy/sell).
      - Reward al cierre del trade en R-multiple: (pnl - trade_cost_money) / risk_money.
      - Updates por batch (cfg.batch_size) para reducir varianza.
    """

    def __init__(self, cfg: RLConfig):
        self.cfg = cfg
        self.seed = int(cfg.seed)
        self.policy: Optional[SoftmaxBanditPolicy] = None
        self.pending: Dict[int, Dict[str, Any]] = {}
        self._next_id = 1
        self._buf: List[Tuple[np.ndarray, int, float]] = []
        self.logs: List[Dict[str, Any]] = []

        self.update_frequency = int(getattr(cfg, 'update_frequency', cfg.batch_size))
        self.steps_since_update = 0
        print(f'[RL] Update frequency: {self.update_frequency}')

        self.trade_history = deque(maxlen=20)
        self.equity_peak = None
        self.initial_equity = None
        self.current_equity = None
        self.total_trades = 0
        self.winning_trades = 0

    def _update_equity_tracking(self, current_equity: float) -> None:
        """Actualiza tracking de equity para calcular drawdown."""
        if self.initial_equity is None:
            self.initial_equity = float(current_equity)

        self.current_equity = float(current_equity)
        if self.equity_peak is None or current_equity > self.equity_peak:
            self.equity_peak = float(current_equity)

    def _calculate_current_dd_pct(self) -> float:
        """Calcula drawdown actual desde el peak."""
        if self.equity_peak is None or self.current_equity is None:
            return 0.0

        if self.equity_peak <= 0:
            return 0.0

        current_dd = (self.current_equity - self.equity_peak) / self.equity_peak

        return float(max(0.0, -current_dd))  # retornar positivo

    def _calculate_equity_ratio(self) -> float:
        """Ratio equity actual / equity inicial."""
        if self.initial_equity is None or self.initial_equity <= 0:
            return 1.0

        if self.current_equity is None:
            return 1.0

        return float(self.current_equity / self.initial_equity)

    def _calculate_recent_win_rate(self, n: int = 10) -> float:
        """Win rate de los últimos N trades."""
        if len(self.trade_history) == 0:
            return 0.5  # neutral

        recent = list(self.trade_history)[-n:]
        wins = sum(1 for t in recent if t['pnl'] > 0)

        return float(wins / len(recent))

    def _calculate_recent_pnl_trend(self, n: int = 5) -> float:
        """Promedio de PnL de últimos N trades (normalizado por riesgo)."""
        if len(self.trade_history) == 0:
            return 0.0

        recent = list(self.trade_history)[-n:]

        # Calcular R-multiple promedio
        r_multiples = []
        for t in recent:
            r = t['pnl'] / max(t.get('risk', 1.0), 1.0)
            r_multiples.append(r)

        return float(np.mean(r_multiples))

    def _get_equity_context(self) -> Dict[str, float]:
        """Construye diccionario con contexto de equity para el estado."""
        return {
            'current_dd_pct': self._calculate_current_dd_pct(),
            'equity_ratio': self._calculate_equity_ratio(),
            'recent_win_rate': self._calculate_recent_win_rate(n=10),
            'recent_pnl_trend': self._calculate_recent_pnl_trend(n=5),
        }

    def build_state(self, row: pd.Series, dec: 'DecisionOutput',
                    equity_context: Optional[Dict[str, float]] = None) -> np.ndarray:
        """
        Estado compacto para el gate (normalizado suave).
        Importante: solo features 'estables' que ya tenías en decide_at_bar/predict:
          - probabilidades calibradas buy/sell
          - scores buy/sell
          - régimen (one-hot)
          - atr/adx/trend_dir si existen
        """
        # === FEATURES ORIGINALES ===
        p_buy = float(getattr(dec, 'p_buy_cal', 0.0))
        p_sell = float(getattr(dec, 'p_sell_cal', 0.0))
        s_buy = float(getattr(dec, 'score_buy', 0.0))
        s_sell = float(getattr(dec, 'score_sell', 0.0))

        reg = str(getattr(dec, 'regime', '') or '').strip().lower()
        reg_trend = 1.0 if 'trend' in reg else 0.0
        reg_range = 1.0 if 'range' in reg else 0.0
        reg_vol = 1.0 if 'volatile' in reg or 'volatil' in reg else 0.0

        atr = float(row.get('atr', 0.0))
        adx = float(row.get('adx', 0.0))
        td = float(row.get('trend_dir', 0.0))

        atr_n = float(np.tanh(atr / 10.0))
        adx_n = float(np.tanh(adx / 50.0))
        td_n = float(np.tanh(td))

        chop = float(row.get("chop_score", 0.0))
        exh = float(row.get("exhaustion_score", 0.0))

        if equity_context is None:
            equity_context = self._get_equity_context()

        current_dd_pct = float(equity_context.get('current_dd_pct', 0.0))
        equity_ratio = float(equity_context.get('equity_ratio', 1.0))
        recent_wr = float(equity_context.get('recent_win_rate', 0.5))
        recent_pnl_trend = float(equity_context.get('recent_pnl_trend', 0.0))

        score_spread = abs(s_buy - s_sell)
        score_max = max(s_buy, s_sell, 0.01)
        score_confidence = score_max / (score_spread + 0.01)

        if 'high' in row and 'low' in row and 'close' in row:
            close = float(row['close'])
            if close > 0:
                bar_range_pct = (float(row['high']) - float(row['low'])) / close
            else:
                bar_range_pct = 0.0
        else:
            bar_range_pct = 0.0

        bar_range_pct = float(np.tanh(bar_range_pct * 10.0))  # normalizar

        if 'time' in row:
            try:
                import pandas as pd
                ts = pd.to_datetime(row['time'])
                hour = ts.hour
                hour_sin = float(np.sin(2 * np.pi * hour / 24.0))
                hour_cos = float(np.cos(2 * np.pi * hour / 24.0))
            except:
                hour_sin, hour_cos = 0.0, 0.0
        else:
            hour_sin, hour_cos = 0.0, 0.0

        state = np.array([
            # Probabilidades y scores (4)
            p_buy, p_sell, s_buy, s_sell,

            # Régimen (3)
            reg_trend, reg_range, reg_vol,

            # Indicadores técnicos (3)
            atr_n, adx_n, td_n,

            # Condiciones adversas (2)
            chop, exh,

            # === NUEVAS: Contexto de equity (4) ===
            current_dd_pct,
            equity_ratio - 1.0,  # centrar en 0 (0 = breakeven)
            recent_wr - 0.5,  # centrar en 0 (0 = 50% WR)
            recent_pnl_trend,  # ya está en R-multiple

            # === NUEVAS: Confianza y timing (4) ===
            score_confidence,
            bar_range_pct,
            hour_sin,
            hour_cos,
        ], dtype=np.float32)

        if not np.all(np.isfinite(state)):
            print(f"[ERROR] State contains NaN or Inf!")
            print(f"State: {state}")
            for i, val in enumerate(state):
                if not np.isfinite(val):
                    print(f"  Feature {i} is invalid: {val}")

        return state

    def _compute_reward(self, pnl_money: float, risk_money: float, chop_score: float, exh_score: float) -> Dict[
        str, float]:
        """
        Reward en R-multiple con clipping y penalizaciones sigmoid.

        Returns:
            Dict con 'reward', 'pnl_net', 'pen_chop', 'pen_exh'
        """
        # Reward base
        trade_cost = float(getattr(self.cfg, "trade_cost_money", 0.0))
        pnl_net = float(pnl_money - trade_cost)
        r_mult = pnl_net / max(float(risk_money), 1e-12)

        # Clip en [-3, +3]R
        r_mult = np.clip(r_mult, -3.0, +3.0)

        # Penalizaciones sigmoid
        cfg = self.cfg
        chop_thr = float(getattr(cfg, "chop_soft_thr", 0.60))
        exh_thr = float(getattr(cfg, "exhaustion_soft_thr", 0.60))
        k_chop = float(getattr(cfg, "chop_penalty_coef", 0.08))
        k_exh = float(getattr(cfg, "exhaustion_penalty_coef", 0.06))

        def sigmoid_soft(x: float, thr: float) -> float:
            return 1.0 / (1.0 + np.exp(-10.0 * (x - thr)))

        pen_chop = k_chop * sigmoid_soft(chop_score, chop_thr)
        pen_exh = k_exh * sigmoid_soft(exh_score, exh_thr)

        reward = r_mult - pen_chop - pen_exh

        # Devolver dict con todas las variables
        return {
            'reward': float(reward),
            'pnl_net': float(pnl_net),
            'pen_chop': float(pen_chop),
            'pen_exh': float(pen_exh),
            'r_mult_base': float(r_mult),  # opcional: reward antes de penalizaciones
        }

    def decide(
            self,
            row: pd.Series,
            dec: 'DecisionOutput',
            *,
            train: bool = True,
            deterministic: bool = False,
            take_threshold: float = 0.55,
            train_threshold: float = 0.05,
            current_equity: Optional[float] = None,
    ) -> tuple[bool, int, float]:

        if current_equity is not None:
            self._update_equity_tracking(current_equity)

        equity_context = self._get_equity_context()

        s = self.build_state(row, dec, equity_context)
        state_dim = int(s.shape[0])

        '''
        if self.policy is None:
            self.policy = SoftmaxBanditPolicy(state_dim=s.shape[0], n_actions=2, cfg=self.cfg, seed=self.seed)
        else:
            if getattr(self.policy, 'W', None) is None or self.policy.W.shape[0] != state_dim:
                old_dim = None if getattr(self.policy, 'W', None) is None else int(self.policy.W.shape[0])
                print(f'[RL] WARNING: policy state_dim mismatch (policy={old_dim}, current={state_dim}. Reinitializing...')
                self.policy = SoftmaxBanditPolicy(state_dim=state_dim, n_actions=2, cfg=self.cfg, seed=self.seed)

        '''

        # Inicializar policy solo si NO existe o NO tiene W
        if self.policy is None or getattr(self.policy, 'W', None) is None:
            self.policy = SoftmaxBanditPolicy(state_dim=s.shape[0], n_actions=2, cfg=self.cfg, seed=self.seed)
        # Verificar dimensiones si policy ya existe
        elif self.policy.W.shape[0] != state_dim:
            old_dim = int(self.policy.W.shape[0])
            print(f'[RL] WARNING: policy state_dim mismatch (policy={old_dim}, current={state_dim}). Reinitializing...')
            self.policy = SoftmaxBanditPolicy(state_dim=state_dim, n_actions=2, cfg=self.cfg, seed=self.seed)

        p = self.policy.probs(s)
        p_take = float(p[A_TAKE])

        if deterministic:
            a = A_TAKE if (p_take >= float(take_threshold)) else A_SKIP
        else:
            if p_take < float(train_threshold):
                a = A_SKIP
            else:
                a, _P = self.policy.sample(s)

        if int(a) == A_SKIP:
            # Reward por skip: pequeño valor positivo para que la policy tenga
            # contexto de A_SKIP y no colapse hacia "siempre skipear".
            # Sin este feedback, la policy solo ve rewards negativos de trades
            # malos y aprende racionalmente a nunca operar.
            if train:
                skip_reward = float(getattr(self.cfg, 'skip_reward', 0.1))
                self._buf.append((s, A_SKIP, skip_reward))
                self._maybe_update()
            return False, -1, p_take

        trade_id = self._next_id
        self._next_id += 1

        chop = float(row.get("chop_score", 0.0)) if hasattr(row, "get") else float(
            getattr(row, "chop_score", 0.0) or 0.0)
        exh = float(row.get("exhaustion_score", 0.0)) if hasattr(row, "get") else float(
            getattr(row, "exhaustion_score", 0.0) or 0.0)
        self.pending[int(trade_id)] = {
            "s": s,
            "chop_score": chop,
            "exhaustion_score": exh,
        }

        return True, int(trade_id), p_take

    def on_trade_closed(self, trade_id: int, pnl_money: float, risk_money: float, meta: Optional[Dict[str, Any]] = None,
                        *, train: bool = True) -> Optional[Dict[str, float]]:
        """
        Reward por trade cerrado en R-multiple:
            reward_R = (pnl_money - trade_cost_money) / risk_money

        - Si train=False: NO actualiza la policy, pero registra log igualmente.
        - meta puede incluir: i_exit, exit_reason, i_entry, etc.
        """
        if trade_id is None or int(trade_id) < 0 or self.policy is None:
            return None

        trade_id = int(trade_id)

        # recuperar estado
        # s = self.pending.pop(trade_id, None)
        item = self.pending.pop(trade_id, None)
        if item is None:
            return None
        s = item.get("s", None)
        if s is None:
            return None

        chop_score = float(item.get("chop_score", 0.0))
        exh_score = float(item.get("exhaustion_score", 0.0))

        if s is None:
            return None

        # riesgo monetario
        try:
            risk_money = float(risk_money)
        except Exception:
            return None
        if (not np.isfinite(risk_money)) or (risk_money <= 0.0):
            return None

        # pnl monetario neto (resta coste fijo si aplica)
        try:
            pnl_money = float(pnl_money)
        except Exception:
            return None

        reward_info = self._compute_reward(pnl_money, risk_money, chop_score, exh_score)
        reward_R = reward_info['reward']
        pnl_net = reward_info['pnl_net']
        pen_chop = reward_info['pen_chop']
        pen_exh = reward_info['pen_exh']

        print(f"[RL] trade#{trade_id}: pnl={pnl_money:.2f}, reward={reward_R:.4f}")

        # (Opcional) shaping muy suave: penaliza SL muy rápido si meta trae i_entry
        if meta:
            try:
                exit_reason = str(meta.get("exit_reason", "") or "")
                i_entry = int(meta.get("i_entry", -1))
                i_exit = int(meta.get("i_exit", -1))
                if ("SL" in exit_reason) and (i_entry >= 0) and (i_exit >= 0):
                    dur = i_exit - i_entry
                    if dur >= 0 and dur < 10:
                        reward_R -= 0.15
            except Exception:
                pass

        out: Optional[Dict[str, float]] = None
        if train:
            self._buf.append((s, A_TAKE, reward_R))
            out = self._maybe_update()

        # log (siempre)
        rec: Dict[str, Any] = {
            "trade_id": int(trade_id),
            "reward_R": float(reward_R),
            "pnl_money": float(pnl_money),
            "pnl_net": float(pnl_net),
            "risk_money": float(risk_money),
            "trained": bool(train),
        }

        rec.update({
            "chop_score_entry": chop_score,
            "exhaustion_score_entry": exh_score,
            "pen_chop": pen_chop,
            "pen_exh": pen_exh,
        })

        self.trade_history.append({
            'pnl': float(pnl_money),
            'risk': float(risk_money),
            'trade_id': int(trade_id),
        })

        self.total_trades += 1
        if pnl_money > 0:
            self.winning_trades += 1

        if meta:
            rec.update(meta)
        if out:
            rec.update(out)
        self.logs.append(rec)

        return out

    '''
    def _maybe_update(self) -> Optional[Dict[str, float]]:
        if int(len(self._buf)) < int(self.cfg.batch_size):
            return None
        # actualiza con todo el batch acumulado
        batch = self._buf[:]
        self._buf.clear()

        metrics_last: Optional[Dict[str, float]] = None
        for s, a, r in batch:
            metrics_last = self.policy.update(s, int(a), float(r))
        return metrics_last
    '''

    def _maybe_update(self) -> Optional[Dict[str, float]]:
        self.steps_since_update += 1

        if self.steps_since_update < self.update_frequency:
            return None

        if len(self._buf) < self.cfg.batch_size:
            return None

        batch = self._buf[:]
        self._buf.clear()
        self.steps_since_update = 0

        metrics_last: Optional[Dict[str, float]] = None
        for s, a, r in batch:
            metrics_last = self.policy.update(s, int(a), float(r))

        print(f'[RL] Updated with {len(batch)} experiences')
        return metrics_last

    def flush_updates(self) -> Optional[Dict[str, float]]:
        if len(self._buf) == 0:
            return None

        print(f'[RL] Flushing {len(self._buf)} pending experiences...')

        batch = self._buf[:]
        self._buf.clear()
        self.steps_since_update = 0

        metrics_last: Optional[Dict[str, float]] = None
        for s, a, r in batch:
            metrics_last = self.policy.update(s, int(a), float(r))

        return metrics_last


@dataclass
class MTMPosition:
    """Posición abierta para backtest con valoración MTM."""
    entry_time: Any
    side: str  # "long" / "short"
    entry: float
    sl: float
    tp: float
    qty: float

    req_margin: float = 0.0
    risk_pct: float = 0.0
    proba_cal: float = 0.0
    score: float = 0.0
    market_condition: Any = None
    regime3: str = ""
    i_entry: int = -1
    trade_id: int = -1
    symbol: str = ''

    # --- NUEVO: configuración de salida por posición ---
    exit_mode: str = 'close_confirm'  # 'hard' | 'close_confirm'
    close_confirm_bars: int = 1
    firewall_atr_mult: float = 0.0
    tp_mode: str = 'wick'  # 'wick' | 'close'

    # --- NUEVO: estado interno para close-confirm / sweeps ---
    confirm_count: int = 0
    last_sweep_i: int = -1
    sweep_level: float = np.nan


@dataclass
class AccountState:
    balance: float
    equity: float
    used_margin: float = 0.0

    @property
    def free_margin(self) -> float:
        return float(self.equity - self.used_margin)

    @property
    def margin_level_pct(self) -> float:
        if self.used_margin <= 0:
            return float('inf')

        return float(self.equity / self.used_margin * 100.0)


def round_lot(x: float, lot_step: float, lot_min: float, lot_max: float) -> float:
    try:
        x = float(x)
    except Exception:
        return 0.0

    if not np.isfinite(x) or x <= 0:
        return 0.0

    x = max(float(lot_min), min(float(lot_max), x))
    lot_step = float(lot_step) if lot_step else 0.0
    if lot_step > 0:
        x = np.floor(x / lot_step) * lot_step

    return float(max(float(lot_min), min(float(lot_max), x)))


class TradingSimulator:
    def __init__(
            self,
            *,
            general_config: Config,
            model_config: ModelConfig = ModelConfig(),
            feature_config: FeatureConfig = FeatureConfig(),
            regime_config: RegimeConfig = RegimeConfig(),
            decision_policy: Optional[DecisionPolicy] = None,
            risk_config: RiskConfig = RiskConfig(),
            artifacts_path: str = ".",
            batch_size: Optional[int] = None,

            # --- sizing / pnl units ---
            value_per_price_unit: float = 100.0,  # $ por 1.0 de precio y 1.0 qty (ajusta a tu broker)
            max_risk_money: float = 200.0,
            max_qty: float = 5.0,

            # --- anti-compounding ---
            compound: bool = False,  # False => sizing con equity fija (initial_equity)
            sizing_equity_mode: str = "fixed",  # "fixed" | "balance" | "mtm"
            # fixed: initial_equity (recomendado)
            # balance: balance realizado actual
            # mtm: equity MTM actual

            # --- mtm ---
            spread_price: float = 0.0,  # spread en PRECIO (no pips)
            mtm_use_bid_ask: bool = True,
            mtm_price_col: str = "close",

            # --- daily caps (muy recomendables) ---
            max_daily_loss_pct: float = 0.015,  # 1.5% del initial_equity
            max_daily_profit_pct: Optional[float] = None,  # p.ej. 0.03 para +3% (opcional)
            max_trades_per_day: Optional[int] = None,

            # --- instruments / margin ---
            symbol: Optional[str] = None,
            account_currency: str = 'USD',
            enforce_margin: bool = True,
            min_margin_level_pct: float = 200.0,

            # --- metrics ---
            # --- RL fine-tuning (opcional) ---
            use_rl: bool = False,
            rl_config: Optional[Dict[str, Any]] = None,
            # --- RL controls ---
            rl_train: bool = True,
            rl_eval_deterministic: bool = False,
            rl_take_threshold: float = 0.55,
            rl_train_threshold: float = 0.05,
            rl_policy_path: Optional[str] = None,

            # --- exits / risk management (NUEVO) ---
            exit_mode: str = "close_confirm",  # 'hard' | 'close_confirm'
            close_confirm_bars: int = 1,
            firewall_atr_mult: float = 0.0,  # 0 desactiva
            tp_mode: str = "wick",  # 'wick' | 'close'
            emergency_atr_mult: float = 4.0,
            emergency_buffer_points: int = 30,

            risk_free_rate_annual: float = 0.0,
            enable_live_scaler_updates: bool = False,
            use_prediction_cache: bool = False,
            cache_dir: Optional[str] = None,

            # --- antinat / anomaly (v2) ---
            anomaly_block_threshold: float = 1.0,   # anomaly_score >= umbral → bloquea señal
            signal_cooldown_bars: int = 3,           # velas de silencio tras bloqueo por anomalía
    ):
        self.general_config = general_config
        self.model_config = model_config
        self.feature_config = feature_config
        self.regime_config = regime_config
        self.risk_config = risk_config
        self.artifacts_path = artifacts_path

        self.value_per_price_unit = float(value_per_price_unit)
        self.max_risk_money = float(max_risk_money)
        self.max_qty = float(max_qty)

        self.compound = bool(compound)
        self.sizing_equity_mode = str(sizing_equity_mode)

        self.spread_price = float(spread_price)
        self.mtm_use_bid_ask = bool(mtm_use_bid_ask)
        self.mtm_price_col = str(mtm_price_col)

        # exits
        self.exit_mode = str(exit_mode)
        self.close_confirm_bars = int(close_confirm_bars)
        self.firewall_atr_mult = float(firewall_atr_mult)
        self.tp_mode = str(tp_mode)
        self.emergency_atr_mult = float(emergency_atr_mult)
        self.emergency_buffer_points = int(emergency_buffer_points)

        self.max_daily_loss_pct = float(max_daily_loss_pct)
        self.max_daily_profit_pct = None if max_daily_profit_pct is None else float(max_daily_profit_pct)
        self.max_trades_per_day = None if max_trades_per_day is None else int(max_trades_per_day)

        self.risk_free_rate_annual = float(risk_free_rate_annual)

        # --- RL wrapper (opcional) ---
        self.use_rl = bool(use_rl)
        self.rl_config = rl_config or {}
        self.rl_train = rl_train
        self.rl_eval_deterministic = rl_eval_deterministic
        self.rl_take_threshold = float(rl_take_threshold)
        self.rl_train_threshold = float(rl_train_threshold)
        self.rl_policy_path = rl_policy_path

        self.rl_wrapper: Optional[RLDecisionWrapperGate] = None
        if self.use_rl:
            debug_validate_rl_config_dict(rl_config)

            cfg = RLConfig(**(rl_config or {}))
            self.rl_wrapper = RLDecisionWrapperGate(cfg)
            print('[RL CONFIG]', self.rl_wrapper.cfg)

        self.symbol = str(symbol) if symbol is not None else str(getattr(general_config, 'symbol', 'XAUUSD.r'))
        self.account_currency = str(account_currency)
        self.enforce_margin = bool(enforce_margin)
        self.min_margin_level_pct = float(min_margin_level_pct)

        self.enable_live_scaler_updates = enable_live_scaler_updates

        self.use_prediction_cache = use_prediction_cache
        if use_prediction_cache:
            cache_path = cache_dir or f'./cache/{self.general_config.release}/predictions'
            self.prediction_cache = PredictionCache(cache_dir=cache_path, enabled=True)
            print(f'[TradingSimulator] ✅ Prediction cache ENABLED: {cache_path}')
        else:
            self.prediction_cache = PredictionCache(enabled=False)
            print(f'[TradingSimulator] ⚠️  Prediction cache DISABLED')

        self.pipeline = DataPipeline(
            general_config=general_config,
            feature_config=feature_config,
            model_config=model_config,
            regime_config=regime_config,
        )

        self.helper = Helper(general_config=general_config, path=artifacts_path)
        self.strategy_gate = StrategyGate(
            chop_block=True,
            chop_size_mult=0.0,
            exhaustion_blocks_reentry=True
        )

        # Policy por defecto (si no pasas una)
        if decision_policy is None:
            decision_policy = DecisionPolicy(
                gate_by_action_and_state={
                    "long": {
                        'trend_up': 90,
                        'transition': 95,
                        'range': 95,
                        'breakout': 97,
                        'volatile': 99,
                        "_global": 98,
                    },
                    "short": {
                        'trend_down': 90,
                        'transition': 95,
                        'range': 95,
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

        self.decision_policy = decision_policy
        self.decision_engine = DecisionEngine(
            general_config=general_config,
            policy=decision_policy,
            risk_config=risk_config,
            path=artifacts_path,
            antinat_config=AntiNaturalConfig(
                cooldown_bars=5,
                penalty_lambda=1.25,
                use_strong_trend_gate=True,
                strong_trend_adx=30.0,
                anomaly_block_threshold=float(anomaly_block_threshold),  # v2
                signal_cooldown_bars=int(signal_cooldown_bars),           # v2
            )
        )

        self.evaluator = ModelEvaluator()

        self.models: Dict[str, Any] = {}
        self.calibrators: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.loaded: bool = False

        self.batch_size = int(batch_size) if batch_size is not None else int(getattr(model_config, "batch_size", 256))
        self._initial_equity_for_bt: Optional[float] = None

        self.db = Database()

        '''
        self.calibrator_adaptative = TradingCalibratorAdaptive(
            min_calibration_ratio=0.2,
            extreme_threshold=0.98,
            extreme_discount=0.70,
            enable_logging=True
        )
        '''

    # -------------------
    # Artifacts / predict
    # -------------------
    def load_artifacts(self) -> None:
        self.models, self.calibrators, self.scalers = self.helper.load_everything(self.pipeline)
        self.loaded = True

    def _log_scaler_state(self, tag='LIVE'):
        for name, sc in self.pipeline.scalers.items():
            if hasattr(sc, 'get_stats'):
                stats = sc.get_stats()
                print(
                    f"[{tag}] scaler={name} "
                    f"fitted={stats['is_fitted']} "
                    f"buf={stats['buffer_size']} "
                    f"upd={stats['n_updates']} "
                    f"med[0]={stats['median'][0]:.4f} "
                    f"iqr[0]={stats['scale'][0]:.4f}"
                )

    @staticmethod
    def _log_feature_ranges(X, name="features"):
        """
        Log de rangos para sanity-check.
        Acepta np.ndarray o pd.DataFrame.
        Si es DataFrame, usa SOLO columnas numéricas.
        """
        if isinstance(X, pd.DataFrame):
            # Quedarnos solo con numéricas (evita Timestamp, strings, etc.)
            Xn = X.select_dtypes(include=[np.number])
            if Xn.shape[1] == 0:
                print(f"[FEAT] {name} -> no numeric columns to log")
                return
            x = Xn.to_numpy(dtype=np.float32, copy=False)
        else:
            x = np.asarray(X, dtype=np.float32)

        if x.ndim == 3:
            x = x.reshape(-1, x.shape[-1])
        elif x.ndim == 1:
            x = x.reshape(-1, 1)
        elif x.ndim != 2:
            raise ValueError(f"Unexpected ndim in _log_feature_ranges: {x.ndim}")

        mn = float(np.nanmin(x))
        p50 = float(np.nanmedian(x))
        p99 = float(np.nanpercentile(x, 99))
        mx = float(np.nanmax(x))

        print(f"[FEAT] {name} min={mn:.2f} p50={p50:.2f} p99={p99:.2f} max={mx:.2f}")

    def _prepare_live(self, df_rates: pd.DataFrame, simulation: bool = True) -> pd.DataFrame:
        df = self.helper.load_from_dataframe(df_rates)
        df_p = self.pipeline.prepare_data(df, labels=False, side=None, set_market_condition=True, ensure_regime=True)

        self._log_scaler_state(tag="LIVE_PRE")

        if self.enable_live_scaler_updates:
            self.pipeline.live_update_scalers_from_df(df_p)

        '''
        if not simulation:
            last = df_p.iloc[-1]
            rate_features = RateFeatures.from_row(last)

            with self.db.session() as session:
                session.add(rate_features)
                session.commit()
        '''

        return df_p

    def predict_side_pack(self, side_pack: dict, df: pd.DataFrame, side: str):
        X_seq_short = side_pack[side]['seq_short']
        X_seq_long = side_pack[side]['seq_long']
        X_context = side_pack[side]['context']
        X_time = side_pack[side]['time']

        n = len(X_seq_long)
        if n == 0:
            return None, None, None

        model = self.models[side]
        proba_raw = model.predict(
            [X_seq_short, X_seq_long, X_context, X_time], verbose=1
        ).reshape(-1)

        if side in self.calibrators:
            proba_cal = self.calibrators[side].predict(proba_raw)
        else:
            proba_cal = proba_raw

        proba_cal = np.clip(proba_cal, 0.0, 1.0)

        L = self.model_config.seq_len_long
        context_offset = L - 1
        df_aligned = df.iloc[context_offset:context_offset + len(proba_raw)].copy()

        return proba_raw, proba_cal, df_aligned

    def _predict_side(self, df_prepared: pd.DataFrame, side: str) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        L = int(getattr(self.model_config, "seq_len_long", 64))
        if len(df_prepared) < L:
            raise ValueError(f"Filas insuficientes: len(df)={len(df_prepared)} < seq_len_long={L}")

        data = self.pipeline.create_sequences(df_prepared, fit_scalers=False, train=False)
        X = [data["seq_short"], data["seq_long"], data["context"], data["time"]]

        model = self.models[side]
        p_raw = model.predict(X, batch_size=self.batch_size, verbose=0).reshape(-1).astype(np.float32)

        cal = self.calibrators[side]
        try:
            p_cal = cal.predict(p_raw).astype(np.float32)
        except Exception:
            p_cal = cal.predict(p_raw.reshape(-1, 1)).astype(np.float32)

        p_cal = np.clip(p_cal, 0.0, 1.0)

        context_offset = L - 1
        df_aligned = df_prepared.iloc[context_offset:context_offset + len(p_raw)].copy()
        return p_raw, p_cal, df_aligned

    def print_last(self, df):
        row = df.iloc[-1]

        pred_long_raw = row.pred_long_raw.item()
        pred_short_raw = row.pred_short_raw.item()
        pred_long_cal = row.pred_long_cal.item()
        pred_short_cal = row.pred_short_cal.item()
        macro_regime = getattr(row, 'macro_regime', getattr(row, 'regime', None))
        state = row.state

        print(
            f'\tState: {state}. MacroRegime: {macro_regime}. Raw probs: {pred_long_raw:.4f} - {pred_short_raw:.4f}. Cal probs: {pred_long_cal:.4f} - {pred_short_cal:.4f}')

        # print(f"\tMacroRegime: {macro_regime}. State: {state}. Long (prob): {pred_long_raw:.4f}. Long (cal): {pred_long_cal:.4f}. Short (prob): {pred_short_raw:.4f}. Short (cal): {pred_short_cal:.4f}")

    def predict(self, df_rates: pd.DataFrame, simulation: bool = True, verbose: bool = False) -> pd.DataFrame:
        '''
        Genera predicciones del modelo con cache opcional

        Args:
            df_rates: DataFrame con datos OHLVC
            simulation: si True, permite usar cache (en live siempre debe de ser False)
            verbose: logging detallado

        Returns:
            DataFrame con predicciones (pred_long_raw, pred_long_cal, pred_short_raw, pred_short_cal)

        Importante:
            - Solo usar cache en simulation=True durante el backtesting
            - En live trading (simulation=False) siempre computa fresh predictions
        '''

        from_date = None
        to_date = None

        if simulation and self.use_prediction_cache:
            # Extraemos las fechas del dataframe
            try:
                if isinstance(df_rates.index, pd.DatetimeIndex):
                    from_date = df_rates.index[0].normalize().strftime('%Y-%m-%d')
                    to_date = df_rates.index[-1].normalize().strftime('%Y-%m-%d')
                elif 'time' in df_rates.columns:
                    from_date = pd.to_datetime(df_rates['time'].iloc[0]).normalize().strftime('%Y-%m-%d')
                    to_date = pd.to_datetime(df_rates['time'].iloc[-1]).normalize().strftime('%Y-%m-%d')
            except Exception as e:
                print(f'[Cache] ⚠️  Error extracting dates: {e}')
                from_date = None
                to_date = None

        n_rows = len(df_rates)

        if simulation and self.use_prediction_cache and from_date and to_date:
            cached = self.prediction_cache.get(
                release=self.general_config.release,
                from_date=from_date,
                to_date=to_date,
                n_rows=n_rows
            )

            if cached is not None:
                if len(cached) + 734 == len(df_rates):
                    return cached
                else:
                    print(f'[Cache] ⚠️  Size mismatch (cached={len(cached)}, expected={len(df_rates)}), recomputing')

        if not self.loaded:
            # Permite usar predict() sin load explícito si tu Helper ya lo gestiona
            self.load_artifacts()

        df_p = self.pipeline.prepare_data(df_rates, labels=False, side='both', set_market_condition=True,
                                          ensure_regime=True)
        df_p.dropna(inplace=True)

        sequences = self.pipeline.create_sequences_by_side(df_p, sides=('long', 'short'), fit_scalers=False,
                                                           train=False)
        if verbose:
            for side in ('long', 'short'):
                data = sequences[side]

                print(f'\n[DECIDE_LIVE] feature ranges after scaling:')
                self._log_feature_ranges(data['seq_short'], 'seq_short_scaled')
                self._log_feature_ranges(data['seq_long'], 'seq_long_scaled')
                self._log_feature_ranges(data['context'], 'context_scaled')

        p_raw_long, p_cal_long, df_long = self.predict_side_pack(sequences, df_p, 'long')
        p_raw_short, p_cal_short, df_short = self.predict_side_pack(sequences, df_p, 'short')

        # self._log_feature_ranges(df_p, 'df_p')

        n = min(len(df_long), len(df_short), len(p_cal_long), len(p_cal_short))
        out = df_long.iloc[:n].copy()
        out["pred_long_raw"] = p_raw_long[:n]
        out["pred_long_cal"] = p_cal_long[:n]
        out["pred_short_raw"] = p_raw_short[:n]
        out["pred_short_cal"] = p_cal_short[:n]

        # Normaliza campos esperados por el backtest
        if "time" not in out.columns and isinstance(out.index, pd.DatetimeIndex):
            out["time"] = out.index

        # Normalización de nomenclatura:
        # - macro_regime: salida del detector de 5 clases (contexto/diagnóstico)
        # - state: salida de 8 clases (driver canónico para decisión)
        # - market_condition: alias DEPRECATED (siempre apuntando a state cuando exista)
        if "macro_regime" not in out.columns and "regime" in out.columns:
            out["macro_regime"] = out["regime"].astype(str)

        if "state" in out.columns:
            out["market_condition"] = out["state"].astype(str)
        elif "macro_regime" in out.columns:
            out["market_condition"] = out["macro_regime"].astype(str)
        elif "regime" in out.columns:
            out["market_condition"] = out["regime"].astype(str)
        else:
            out["market_condition"] = ""

        if simulation and self.use_prediction_cache and from_date and to_date:
            try:
                self.prediction_cache.put(
                    release=self.general_config.release,
                    from_date=from_date,
                    to_date=to_date,
                    df_predictions=out,
                    n_rows=n_rows
                )
            except Exception as e:
                print(f'[Cache] ⚠️  Error saving to cache: {e}')

        return out

    # -----------
    # Decisions
    # -----------
    def decide_at(self, df_pred: pd.DataFrame, i: int, equity_for_sizing: float, *,
                  has_position: bool = False, pos_side: Optional[str] = None, ) -> Tuple[
        DecisionOutput, Optional[Dict[str, Any]]]:

        row = df_pred.iloc[i]

        '''
        result = self.calibrator_adaptative.adjust_probabilities(
            raw_p_buy=row.pred_long_raw,
            raw_p_sell=row.pred_short_raw,
            cal_p_buy=row.pred_long_cal,
            cal_p_sell=row.pred_short_cal,
            market_data={
                'price': row.close,
                'hist_max': 4_960.19,
                'hist_min': 2_730.53,
                'atr': row.atr,
                'hist_atr_avg': 1.46,
                'macro_regime': 'trending'
            }
        )

        print(f'\tResult: {result}\n')
        '''

        decision = self.decision_engine.decide_at_bar(
            p_buy_raw=float(row.pred_long_raw),
            p_sell_raw=float(row.pred_short_raw),
            p_buy_cal=float(row.pred_long_cal),
            p_sell_cal=float(row.pred_short_cal),
            state_raw=str(getattr(row, 'state', getattr(row, 'market_condition', 'range'))),
            macro_regime=str(getattr(row, 'macro_regime', getattr(row, 'regime', ''))),
            market_condition=str(getattr(row, 'market_condition', getattr(row, 'state', ''))),  # DEPRECATED
            o=float(row.open),
            h=float(row.high),
            l=float(row.low),
            c=float(row.close),
            atr=float(row.atr),
            adx14=float(row.adx) if 'adx' in row else None,
            trend_dir=row.trend_dir if 'trend_dir' in row else None
        )

        if getattr(self, 'strategy_gate', None) is not None and decision.action in ('buy', 'sell'):
            gated_decision = self.strategy_gate.apply(
                base_action=str(decision.action),
                row=row.to_dict(),
                has_position=bool(has_position),
                pos_side=pos_side
            )

            decision.debug = dict(decision.debug or {})
            decision.debug['strategy_gate_reason'] = gated_decision.reason
            decision.debug['strategy_gate_mult'] = float(gated_decision.size_mult)

            decision.action = gated_decision.action

            if gated_decision.action != 'none':
                decision.risk_pct = float(decision.risk_pct) * float(gated_decision.size_mult)

        # print(decision)
        if decision.action == "none":
            return decision, None

        # --- NUEVO: exit_mode efectivo por trade (solo si reason=trade) ---
        # Se obtiene del DecisionEngine (exit_mode_suggested) y, si no existe,
        # se usa el default global del simulador.
        exit_mode_eff = str(getattr(self, "exit_mode", "close_confirm"))
        try:
            if decision.action in ("buy", "sell") and isinstance(getattr(decision, "debug", None), dict):
                if "exit_mode_suggested" in decision.debug:
                    exit_mode_eff = str(decision.debug["exit_mode_suggested"])
        except Exception:
            pass

        order = self._build_order(row, decision, equity_for_sizing, exit_mode_eff=exit_mode_eff)
        if order is None:
            return decision, None
        return decision, order

    def _build_order(self, row: pd.Series, decision: DecisionOutput, equity_for_sizing: float, *, exit_mode_eff: str) -> \
    Optional[Dict[str, Any]]:
        close = float(row["close"].item())
        atr = float(row["atr"].item()) if "atr" in row else np.nan
        if not np.isfinite(atr) or atr <= 0:
            return None

        sl_mult = float(getattr(self.feature_config, "sl_barrier", 1.0))
        tp_mult = float(getattr(self.feature_config, "tp_barrier", 1.0))

        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr
        if sl_dist <= 0 or tp_dist <= 0:
            return None

        if decision.action == "buy":
            side = "long"
            sl = close - sl_dist
            tp = close + tp_dist
            proba = float(getattr(decision, "p_buy_cal", 0.0))
            score = float(getattr(decision, "score_buy", 0.0))
        else:
            side = "short"
            sl = close + sl_dist
            tp = close - tp_dist
            proba = float(getattr(decision, "p_sell_cal", 0.0))
            score = float(getattr(decision, "score_sell", 0.0))

        risk_pct = float(getattr(decision, "risk_pct", 0.01))
        risk_base = float(equity_for_sizing)
        risk_money = min(risk_base * risk_pct, self.max_risk_money)

        # CAMBIO CRÍTICO: Obtener el contract_size del instrumento
        symbol = self._get_symbol_from_row(row)
        spec = self._get_spec(symbol)

        # Calcular value_per_lot basado en el instrumento específico
        if spec is not None:
            contract_size = float(getattr(spec, 'contract_size', 1.0))
            # Para forex/metales, el value_per_lot es el tamaño del contrato
            value_per_lot = contract_size
        else:
            # Fallback al valor por defecto
            value_per_lot = self.value_per_price_unit

        # Cálculo correcto del tamaño del lote
        # risk_money = sl_dist * qty * value_per_lot
        # Por lo tanto: qty = risk_money / (sl_dist * value_per_lot)
        qty = risk_money / max(sl_dist * value_per_lot, 1e-12)

        # Aplicar redondeo según especificaciones del instrumento
        if spec is not None:
            qty = round_lot(
                qty,
                getattr(spec, 'lot_step', 0.01),
                getattr(spec, 'lot_min', 0.01),
                getattr(spec, 'lot_max', self.max_qty)
            )
        else:
            qty = float(min(qty, self.max_qty))

        if qty <= 0 or not np.isfinite(qty):
            return None

        # sanity checks
        if side == "long" and not (sl < close < tp):
            return None
        if side == "short" and not (tp < close < sl):
            return None

        # Probabilidades calibradas de ambos lados (no solo el del trade)
        proba_long  = float(getattr(decision, "p_buy_cal",  0.0))
        proba_short = float(getattr(decision, "p_sell_cal", 0.0))

        # Delta relativo: conviccion del modelo en [-1, 1]
        denom     = max(proba_long + proba_short, 1e-9)
        delta_rel = (proba_long - proba_short) / denom

        # state canonico (8 clases, driver de decision)
        state = str(getattr(row, "state", getattr(row, "market_condition", "")))

        return {
            "entry_time": row["time"],
            "symbol": symbol,
            "side": side,
            "entry": close,
            "atr_at_entry": atr,
            "sl": float(sl),
            "tp": float(tp),
            "qty": qty,
            "risk_pct": risk_pct,

            # Probabilidades — ambos lados expuestos para diagnostico completo
            "proba_cal":   proba,
            "proba_long":  proba_long,
            "proba_short": proba_short,
            "delta_rel":   round(delta_rel, 6),

            # Scores — ambos lados
            "score":       score,
            "score_long":  float(getattr(decision, "score_buy",  0.0)),
            "score_short": float(getattr(decision, "score_sell", 0.0)),

            # Regimen y estado del mercado
            "state":            state,
            "market_condition": row.get("market_condition", None),
            "regime3":          getattr(decision, "regime", ""),

            "contract_size": value_per_lot,

            # Configuracion de salida por trade
            "exit_mode":          str(exit_mode_eff),
            "close_confirm_bars": int(getattr(self, "close_confirm_bars", 1)),
            "firewall_atr_mult":  float(getattr(self, "firewall_atr_mult", 0.0)),
            "tp_mode":            str(getattr(self, "tp_mode", "wick")),
        }

    # ----------------
    # Trading mechanics
    # ----------------
    @staticmethod
    def _check_exit_in_bar(side: str, high: float, low: float, tp: float, sl: float) -> Tuple[
        Optional[str], Optional[float]]:

        side_is_long = 1 if side == 'long' else 0
        exit_type, exit_price = check_exit_simple_numba(
            side_is_long, float(high), float(low), float(tp), float(sl)
        )

        exit_type_map = {
            0: (None, None),
            1: ('SL', exit_price),
            2: ('TP', exit_price),
            3: ('SL and TP on same bar', exit_price)
        }

        return exit_type_map[exit_type]

    @staticmethod
    def _check_exit_in_bar_v2(pos: MTMPosition, *, o, h, l, c, atr, i) -> Tuple[Optional[str], Optional[float]]:

        side = str(pos.side).lower()
        exit_mode = str(getattr(pos, 'exit_mode', 'close_confirm') or 'close_confirm').strip().lower()
        tp_mode = str(getattr(pos, 'tp_mode', 'wick') or 'wich').strip().lower()

        side_is_long = 1 if side == 'long' else 0
        tp_mode_is_wick = 1 if tp_mode == 'wick' else 0
        exit_mode_is_hard = 1 if exit_mode == 'hard' else 0

        tp = float(pos.tp)
        sl = float(pos.sl)
        atr_val = float(atr) if np.isfinite(atr) else 0.0

        firewall_mult = float(getattr(pos, 'firewall_atr_mult', 0.0) or 0.0)
        confirm_count = int(getattr(pos, 'confirm_count', 0))
        confirm_bars = max(1, int(getattr(pos, 'close_confirm_bars', 1)))
        entry_price = float(pos.entry)

        last_sweep_was_sl = getattr(pos, 'last_sweep_i', -999) > 0

        exit_type, exit_price, new_confirm, swept_sl = check_exit_advanced_numba(
            side_is_long, float(o), float(h), float(l), float(c),
            tp, sl, atr_val,
            tp_mode_is_wick, exit_mode_is_hard,
            firewall_mult, confirm_count, confirm_bars,
            entry_price, 1 if last_sweep_was_sl else 0
        )

        pos.confirm_count = new_confirm
        if swept_sl:
            pos.last_sweep_i = int(i)
            pos.sweep_level = float(sl)

        exit_type_map = {
            0: (None, None),
            1: ('TP', exit_price),
            2: ('SL', exit_price),
            3: ('SL_CLOSE_CONFIRM', exit_price),
            4: ('FIREWALL', exit_price)
        }

        return exit_type_map[exit_type]

    def _unrealized_pnl(self, positions: List[MTMPosition], *, mid_price: float) -> float:
        """Calcular PnL no realizado - OPTIMIZADO con Numba."""
        if not positions:
            return 0.0

        n = len(positions)

        # Extraer datos a arrays numpy
        entry_prices = np.empty(n)
        sides_long = np.empty(n, dtype=np.int8)
        qtys = np.empty(n)
        value_per_lots = np.empty(n)

        for idx, p in enumerate(positions):
            entry_prices[idx] = float(p.entry)
            sides_long[idx] = 1 if p.side == "long" else 0
            qtys[idx] = float(p.qty)

            # Obtener contract_size para este símbolo
            symbol = getattr(p, 'symbol', self.symbol)
            spec = self._get_spec(symbol)
            if spec:
                value_per_lots[idx] = float(getattr(spec, 'contract_size', self.value_per_price_unit))
            else:
                value_per_lots[idx] = self.value_per_price_unit

        # Calcular bid/ask
        spr = self.spread_price if self.mtm_use_bid_ask else 0.0
        bid = mid_price - 0.5 * spr
        ask = mid_price + 0.5 * spr

        # Usar Numba para calcular PnL
        total_pnl, _ = batch_unrealized_pnl_numba(
            entry_prices, sides_long, qtys, value_per_lots, bid, ask
        )

        return float(total_pnl)

    def _batch_check_exits_for_positions(
            self,
            positions: List[MTMPosition], o: float, h: float, l: float, c: float, atr: float, i: int
    ) -> List[Tuple[MTMPosition, Optional[str], Optional[float]]]:
        """
        Chequear exits para múltiples posiciones en batch - OPTIMIZADO.

        Returns:
            Lista de (position, exit_type, exit_price) para cada posición
        """
        if not positions:
            return []

        # Determinar si usar versión simple o avanzada
        use_advanced = any(
            getattr(p, "exit_mode", "close_confirm") != "hard" or
            getattr(p, "firewall_atr_mult", 0.0) > 0
            for p in positions
        )

        if use_advanced:
            # Procesar uno por uno (la versión avanzada tiene estado)
            results = []
            for pos in positions:
                exit_type, exit_price = self._check_exit_in_bar_v2(
                    pos, o=o, h=h, l=l, c=c, atr=atr, i=i
                )
                results.append((pos, exit_type, exit_price))
            return results

        # Versión simple - batch processing
        n = len(positions)
        sides_long = np.empty(n, dtype=np.int8)
        highs = np.full(n, h)
        lows = np.full(n, l)
        tps = np.empty(n)
        sls = np.empty(n)

        for idx, p in enumerate(positions):
            sides_long[idx] = 1 if p.side == "long" else 0
            tps[idx] = float(p.tp)
            sls[idx] = float(p.sl)

        # Batch check con Numba
        exit_types, exit_prices = batch_check_exits_simple_numba(
            sides_long, highs, lows, tps, sls
        )

        # Mapear resultados
        exit_type_map = {0: None, 1: "SL", 2: "TP", 3: "SL and TP on same bar"}

        results = []
        for idx, pos in enumerate(positions):
            et = exit_type_map[exit_types[idx]]
            ep = float(exit_prices[idx]) if et is not None else None
            results.append((pos, et, ep))

        return results

    # ----------------------------
    # Instruments / margin helpers
    # ----------------------------
    def _get_symbol_from_row(self, row: pd.Series) -> str:
        # Si el dataframe trae una columna 'symbol', úsala; si no, usa self.symbol
        if isinstance(row, pd.Series) and ("symbol" in row.index):
            try:
                s = str(row["symbol"])
                if s:
                    return s
            except Exception:
                pass
        return str(self.symbol)

    def _get_spec(self, symbol: str):
        if not INSTRUMENT_SPECS:
            return None
        return INSTRUMENT_SPECS.get(symbol)

    def compute_risk_money(self, entry_price: float, sl_price: float, qty: float, symbol: str = None) -> float:
        """
        Riesgo monetario del trade (en dinero):
          |entry - SL| * qty * contract_size
        """
        try:
            entry_price = float(entry_price)
            sl_price = float(sl_price)
            qty = float(qty)
        except Exception:
            return 0.0

        if (not np.isfinite(entry_price)) or (not np.isfinite(sl_price)) or (qty <= 0):
            return 0.0

        # Obtener contract_size del símbolo
        spec = self._get_spec(symbol or self.symbol) if symbol or self.symbol else None
        if spec:
            value_per_lot = float(getattr(spec, 'contract_size', self.value_per_price_unit))
        else:
            value_per_lot = self.value_per_price_unit

        price_distance = abs(entry_price - sl_price)
        risk_money = price_distance * qty * value_per_lot
        return float(max(risk_money, 1e-9))

    def _required_margin_cash(self, *, symbol: str, price: float, lots: float) -> float:
        spec = self._get_spec(symbol)
        if spec is None:
            # Sin specs no podemos calcular margen; asumimos 0 para no bloquear.
            return 0.0
        return float(required_margin(
            price=float(price),
            lots=float(lots),
            contract_size=float(getattr(spec, "contract_size", 1.0)),
            leverage=float(getattr(spec, "leverage", 1.0)),
            margin_rate=getattr(spec, "margin_rate", None),
        ))

    def _calc_used_margin_from_open_positions(self, open_positions: List[Dict[str, Any]], *,
                                              fallback_price: float) -> float:
        used = 0.0
        for p in open_positions or []:
            try:
                if "req_margin" in p and p["req_margin"] is not None:
                    used += float(p["req_margin"])
                    continue
                sym = str(p.get("symbol", self.symbol))
                lots = float(p.get("qty", 0.0))
                price = float(p.get("entry", fallback_price))
                used += self._required_margin_cash(symbol=sym, price=price, lots=lots)
            except Exception:
                continue
        return float(max(0.0, used))

    def _can_open_by_margin(
            self,
            *,
            symbol: str,
            lots: float,
            price: float,
            equity: float,
            used_margin: float,
    ) -> Tuple[bool, float]:
        """Devuelve (ok, req_margin)."""
        if not self.enforce_margin:
            return True, 0.0

        req = self._required_margin_cash(symbol=symbol, price=price, lots=lots)
        free = float(equity - used_margin)

        if req > free:
            return False, req

        projected_used = float(used_margin + req)
        projected_ml = (float(equity) / projected_used * 100.0) if projected_used > 0 else float("inf")
        if projected_ml < float(self.min_margin_level_pct):
            return False, req

        return True, req

    # ----------------
    # Metrics helpers
    # ----------------
    @staticmethod
    def sharpe_sortino_from_equity_daily(
            equity_curve: Union[np.ndarray, list],
            timestamps: Union[np.ndarray, list, pd.DatetimeIndex, pd.Series],
            *,
            risk_free_rate_annual: float = 0.0,
            method: str = "simple",
            trading_days_per_year: int = 252,
            align: str = "tail",
    ) -> Dict[str, Any]:
        eq = np.asarray(equity_curve, dtype=np.float64)
        idx = pd.to_datetime(timestamps)

        if eq.size < 3 or len(idx) < 3:
            return {"sharpe_daily_ann": np.nan, "sortino_daily_ann": np.nan, "n_days": 0,
                    "note": "Insuficientes puntos."}

        n = int(min(eq.size, len(idx)))
        if n < 3:
            return {"sharpe_daily_ann": np.nan, "sortino_daily_ann": np.nan, "n_days": 0,
                    "note": "Insuficientes puntos tras recorte."}

        if eq.size != len(idx):
            if align == "tail":
                eq = eq[-n:]
                idx = idx[-n:]
            elif align == "head":
                eq = eq[:n]
                idx = idx[:n]
            else:
                raise ValueError("align debe ser 'tail' o 'head'")

        s_eq = pd.Series(eq, index=idx).sort_index()
        s_eq_daily = s_eq.resample("1D").last().dropna()
        if s_eq_daily.size < 3:
            return {"sharpe_daily_ann": np.nan, "sortino_daily_ann": np.nan, "n_days": int(s_eq_daily.size),
                    "note": "Insuficientes días con equity."}

        if method == "log":
            s_ret_daily = np.log(s_eq_daily.clip(lower=1e-12)).diff().dropna()
        else:
            s_ret_daily = s_eq_daily.pct_change().dropna()

        if s_ret_daily.size < 3:
            return {"sharpe_daily_ann": np.nan, "sortino_daily_ann": np.nan, "n_days": int(s_ret_daily.size),
                    "note": "Insuficientes retornos diarios."}

        rf_daily = risk_free_rate_annual / float(trading_days_per_year)
        s_excess = s_ret_daily - rf_daily

        mu = float(s_excess.mean())
        sigma = float(s_excess.std(ddof=1))
        sharpe_ann = (mu / (sigma + 1e-12)) * np.sqrt(trading_days_per_year)

        downside = s_excess[s_excess < 0]
        downside_std = float(downside.std(ddof=1)) if downside.size >= 2 else 0.0
        sortino_ann = (mu / (downside_std + 1e-12)) * np.sqrt(trading_days_per_year)

        return {
            "sharpe_daily_ann": np.float64(sharpe_ann),
            "sortino_daily_ann": np.float64(sortino_ann),
            "n_days": int(s_ret_daily.size),
            "note": f"Alineado por recorte ({align}). eq={len(eq)}, ts={len(idx)}",
        }

    @staticmethod
    def _make_summary(trades: list, initial_equity: float, final_equity: float) -> dict:
        pnls = np.array([t["pnl"] for t in trades], dtype=np.float64) if trades else np.array([], dtype=np.float64)
        wins = pnls[pnls > 0]
        losses = pnls[pnls < 0]
        profit_factor = float(wins.sum() / max(1e-12, -losses.sum())) if pnls.size else 0.0
        win_rate = float((pnls > 0).mean()) if pnls.size else 0.0
        avg_pnl = float(pnls.mean()) if pnls.size else 0.0
        return {
            "n_trades": int(len(trades)),
            "initial_equity": float(initial_equity),
            "final_equity": float(final_equity),
            "net_pnl": float(final_equity - initial_equity),
            "win_rate": win_rate,
            "avg_pnl": avg_pnl,
            "profit_factor": profit_factor,
        }

    # -----------
    # BACKTEST MTM
    # -----------
    def _select_sizing_equity(self, *, initial_equity: float, balance: float, equity_mtm: float) -> float:
        # anti-compounding fuerte: fixed
        if not self.compound:
            return float(initial_equity)

        mode = self.sizing_equity_mode.lower().strip()
        if mode == "fixed":
            return float(initial_equity)
        if mode == "balance":
            return float(balance)
        if mode == "mtm":
            return float(equity_mtm)
        # fallback
        return float(balance)

    def _position_risk_cash(self, pos: Dict[str, Any]) -> float:
        entry = float(pos.get('entry', np.nan))
        sl = float(pos.get('sl', np.nan))
        qty = float(pos.get('qty', 0.0))
        sym = str(pos.get('symbol', self.symbol))
        if not np.isfinite(entry) or not np.isfinite(sl) or qty <= 0:
            return 0.0

        spec = self._get_spec(sym)
        value_per_lot = float(getattr(spec, 'contract_size', self.value_per_price_unit)) if spec else float(
            self.value_per_price_unit)

        return abs(entry - sl) * qty * value_per_lot

    def _compute_qty_from_risk_cash(
            self,
            *,
            risk_cash: float,
            entry: float,
            sl: float,
            min_qty: float = 0.0,
            qty_step: float = 0.0,
            max_qty: float = float("inf"),
    ) -> float:
        """
        Calcula qty para que el riesgo hasta el SL sea aproximadamente risk_cash.

        risk_cash: riesgo monetario objetivo (EUR)
        entry/sl: precios
        value_per_price_unit: EUR por 1.0 de movimiento de precio y por 1 qty (lote/contrato)
            Ejemplo: si qty=1 y el precio se mueve 1.0, el PnL = 1.0 * value_per_price_unit
        """
        risk_cash = float(risk_cash)
        entry = float(entry)
        sl = float(sl)

        if not np.isfinite(risk_cash) or risk_cash <= 0:
            return 0.0
        if not np.isfinite(entry) or not np.isfinite(sl):
            return 0.0

        dist = abs(entry - sl)
        if dist <= 0 or not np.isfinite(dist):
            return 0.0

        vppu = float(getattr(self, "value_per_price_unit", 1.0))
        if not np.isfinite(vppu) or vppu <= 0:
            return 0.0

        risk_per_qty = dist * vppu  # EUR de riesgo si qty=1 hasta el SL
        if risk_per_qty <= 0:
            return 0.0

        qty = risk_cash / risk_per_qty

        # caps
        qty = max(float(min_qty), float(qty))
        qty = min(float(max_qty), float(qty))

        # redondeo a step si aplica (ej: 0.01 lot)
        if qty_step and qty_step > 0:
            qty = np.floor(qty / qty_step) * qty_step

        return float(qty)

    def _compute_qty_from_risk_pct(
            self,
            *,
            equity: float,
            risk_pct: float,
            entry: float,
            sl: float,
            min_qty: float = 0.0,
            qty_step: float = 0.0,
            max_qty: float = float("inf"),
    ) -> float:
        """
        Wrapper: calcula qty usando risk_pct sobre equity.
        """
        equity = float(equity)
        risk_pct = float(risk_pct)
        if not np.isfinite(equity) or equity <= 0:
            return 0.0
        if not np.isfinite(risk_pct) or risk_pct <= 0:
            return 0.0

        risk_cash = equity * risk_pct
        return self._compute_qty_from_risk_cash(
            risk_cash=risk_cash,
            entry=entry,
            sl=sl,
            min_qty=min_qty,
            qty_step=qty_step,
            max_qty=max_qty,
        )

    def decide_live(self,
                    *,
                    df_rates: pd.DataFrame,
                    symbol: Optional[str] = None,
                    equity: float,
                    open_positions: Optional[List[Dict[str, Any]]] = None,
                    i: Optional[int] = None,
                    max_positions: Optional[int] = None,
                    current_positions: Optional[int] = None,
                    risk_cash_cap: Optional[float] = None,
                    portfolio_risk_cap: Optional[float] = None,
                    max_qty: Optional[float] = None,
                    ) -> Optional[Dict[str, Any]]:
        """
        Decide si abrir posición en la vela actual (1 minuto).
        Devuelve una orden lista para ejecutar o None.

        Cuando devuelve None, popula self._last_diag con el diagnóstico completo
        del modelo para que el caller pueda loguearlo (NO_SIGNAL enriquecido).
        """

        # ── helper interno: construye y guarda el diagnóstico antes de retornar None ──
        def _reject(reason: str, row=None, dec=None, rl_take_prob: float = None) -> None:
            """Popula self._last_diag con contexto diagnóstico completo."""
            diag: Dict[str, Any] = {"no_signal_reason": reason}

            # Campos de la fila predicha (probas raw/cal, estado, scores de chop/exhaustion)
            if row is not None:
                try:
                    diag["proba_long_raw"] = round(float(getattr(row, "pred_long_raw", 0) or 0), 6)
                    diag["proba_short_raw"] = round(float(getattr(row, "pred_short_raw", 0) or 0), 6)
                    diag["proba_long_cal"] = round(float(getattr(row, "pred_long_cal", 0) or 0), 6)
                    diag["proba_short_cal"] = round(float(getattr(row, "pred_short_cal", 0) or 0), 6)
                    diag["state"] = str(getattr(row, "state", getattr(row, "market_condition", "")))
                    diag["macro_regime"] = str(getattr(row, "macro_regime", getattr(row, "regime", "")))
                    diag["chop_score"] = round(float(getattr(row, "chop_score", 0) or 0), 4)
                    diag["exhaustion_score"] = round(float(getattr(row, "exhaustion_score", 0) or 0), 4)
                    diag["atr"] = round(float(getattr(row, "atr", 0) or 0), 4)
                except Exception:
                    pass

            # Campos del DecisionOutput (scores, acción decidida, debug del engine)
            if dec is not None:
                try:
                    diag["action"] = str(getattr(dec, "action", ""))
                    diag["score_long"] = round(float(getattr(dec, "score_buy", 0) or 0), 6)
                    diag["score_short"] = round(float(getattr(dec, "score_sell", 0) or 0), 6)
                    diag["proba_long_cal_dec"] = round(float(getattr(dec, "p_buy_cal", 0) or 0), 6)
                    diag["proba_short_cal_dec"] = round(float(getattr(dec, "p_sell_cal", 0) or 0), 6)
                    diag["regime3"] = str(getattr(dec, "regime", ""))

                    # delta_rel: convicción del modelo en [-1, 1]
                    pl = diag.get("proba_long_cal_dec", 0)
                    ps = diag.get("proba_short_cal_dec", 0)
                    denom = max(pl + ps, 1e-9)
                    diag["delta_rel"] = round((pl - ps) / denom, 6)

                    # Debug del engine: strategy_gate_reason, no_trade_reason, etc.
                    debug = getattr(dec, "debug", None)
                    if isinstance(debug, dict) and debug:
                        diag["engine_debug"] = {
                            k: v for k, v in debug.items()
                            if k not in ("exit_mode_suggested",)  # omitir campos de salida
                        }
                except Exception:
                    pass

            # Probabilidad del RL gate si fue el filtro
            if rl_take_prob is not None:
                diag["rl_take_prob"] = round(float(rl_take_prob), 6)
                diag["rl_take_threshold"] = float(getattr(self, "rl_take_threshold", 0))

            self._last_diag = diag

        # ── inicio de la lógica principal ────────────────────────────────────────────

        self._last_diag = None  # resetear en cada llamada

        df_pred = self.predict(df_rates, simulation=False, verbose=True)
        self.print_last(df_pred)

        open_positions = open_positions or []

        symbol_eff = str(symbol) if symbol is not None else str(self.symbol)

        used_margin_open = self._calc_used_margin_from_open_positions(open_positions, fallback_price=float(
            df_rates.iloc[-1]['close'].item()))
        eq = float(equity)

        max_positions = int(
            max_positions if max_positions is not None else getattr(self.risk_config, 'max_positions', 1))
        current_positions = int(current_positions if current_positions is not None else len(open_positions))
        risk_pct_default = float(getattr(self, 'risk_pct', 0.01))
        risk_cash_cap = float(risk_cash_cap) if risk_cash_cap is not None else float('inf')
        portfolio_risk_cap = float(portfolio_risk_cap) if portfolio_risk_cap is not None else 1.0
        max_qty = float(max_qty) if max_qty is not None else float('inf')

        if i is None:
            i = -1

        row = df_pred.iloc[i]  # fila predicha, disponible desde aquí para el diagnóstico

        if current_positions >= max_positions:
            print(f'\tOpen positions: {current_positions} is higher (or equal) than {max_positions}')
            _reject(f'MAX_POSITIONS({current_positions}/{max_positions})', row=row)
            return None

        open_risk_cash = sum(self._position_risk_cash(p) for p in open_positions)
        portfolio_risk_cash_cap = eq * portfolio_risk_cap
        if open_risk_cash >= portfolio_risk_cash_cap:
            _reject('PORTFOLIO_RISK_CAP', row=row)
            return None

        has_pos = len(open_positions) > 0
        pos_side = open_positions[0].get('side') if has_pos else None

        dec, order = self.decide_at(df_pred, i, eq, has_position=has_pos, pos_side=pos_side)
        print(f'\tDecision: {dec}')
        print(f'\tOrder: {order}')

        event = DecisionEvent.from_decision(
            dec,
            rate_id=df_rates.iloc[-1]['id']
        )

        with self.db.session() as session:
            session.add(event)
            session.commit()

        if order is None:
            # La acción del decision engine fue 'none': score bajo, régimen filtrado,
            # delta_rel insuficiente, strategy_gate bloqueó, etc.
            # El motivo específico está en dec.debug['strategy_gate_reason'] si existe.
            gate_reason = ""
            debug = getattr(dec, "debug", None)
            if isinstance(debug, dict):
                gate_reason = str(debug.get("strategy_gate_reason", ""))
            _reject(f'DECISION_ENGINE_NONE{(":" + gate_reason) if gate_reason else ""}',
                    row=row, dec=dec)
            return None

        # --- RL gate-only (live): decide TAKE / SKIP sobre señal base ---
        if self.use_rl and (self.rl_wrapper is not None):
            policy_path = self.rl_policy_path
            if not policy_path:
                raise RuntimeError('LIVE: rl_policy_path no configurado')

            load_rl_policy_npz(self.rl_wrapper, policy_path)

            take, trade_id, take_prob = self.rl_wrapper.decide(
                row=df_pred.iloc[i],
                dec=dec,
                deterministic=True,
                take_threshold=float(self.rl_take_threshold),
                train_threshold=float(self.rl_train_threshold),
                train=False
            )

            print(f'\tTake prob: {take_prob:.4f}. Take: {take}')
            if not take:
                _reject('RL_GATE_SKIP', row=row, dec=dec, rl_take_prob=take_prob)
                return None
            order["trade_id"] = int(trade_id)

        side = order.get('side')
        entry = float(order.get('entry', np.nan))
        sl = float(order.get('sl', np.nan))
        tp = float(order.get('tp', np.nan))

        if side not in ('long', 'short'):
            _reject('INVALID_SIDE', row=row, dec=dec)
            return None

        if not np.isfinite(entry) or not np.isfinite(sl) or not np.isfinite(tp):
            _reject('NON_FINITE_PRICES', row=row, dec=dec)
            return None

        if side == 'long' and not (sl < entry < tp):
            _reject(f'PRICE_SANITY_LONG(sl={sl:.2f},e={entry:.2f},tp={tp:.2f})', row=row, dec=dec)
            return None

        if side == 'short' and not (tp < entry < sl):
            _reject(f'PRICE_SANITY_SHORT(tp={tp:.2f},e={entry:.2f},sl={sl:.2f})', row=row, dec=dec)
            return None

        risk_pct = float(order.get('risk_pct', risk_pct_default))
        desired_risk_cash = eq * risk_pct
        risk_cash = min(desired_risk_cash, risk_cash_cap)

        remaining_risk_cash = max(0.0, portfolio_risk_cash_cap - open_risk_cash)
        risk_cash = min(risk_cash, remaining_risk_cash)
        if risk_cash <= 0:
            _reject('RISK_CASH_ZERO', row=row, dec=dec)
            return None

        eff_risk_pct = risk_cash / eq if eq > 0 else 0.0

        sym = str(order.get('symbol', symbol_eff))
        spec = self._get_spec(sym)
        value_per_lot = float(getattr(spec, 'contract_size', self.value_per_price_unit)) if spec is not None else float(
            self.value_per_price_unit)

        dist = abs(entry - sl)
        if dist <= 0 or not np.isfinite(dist):
            _reject('INVALID_SL_DIST', row=row, dec=dec)
            return None

        qty = risk_cash / max(dist * value_per_lot, 1e-12)
        if spec is not None:
            qty = round_lot(
                qty,
                getattr(spec, 'lot_step', 0.01),
                getattr(spec, 'lot_min', 0.01),
                getattr(spec, 'lot_max', max_qty)
            )
        else:
            qty = min(float(qty), max_qty)

        if not np.isfinite(qty) or qty <= 0:
            _reject('INVALID_QTY', row=row, dec=dec)
            return None

        # --- control de margen / apalancamiento (live) ---
        ok_margin, req_margin = self._can_open_by_margin(
            symbol=str(order.get("symbol", symbol_eff)),
            lots=float(qty),
            price=float(entry),
            equity=float(eq),
            used_margin=float(used_margin_open),
        )
        if not ok_margin:
            _reject(f'MARGIN_INSUFFICIENT(req={req_margin:.2f})', row=row, dec=dec)
            return None

        live_order = dict(order)
        live_order['qty'] = qty
        live_order['risk_pct'] = eff_risk_pct
        live_order['risk_cash'] = risk_cash

        return live_order

    def backtest(self, df_rates: pd.DataFrame, initial_equity: float = 10_000.0, df_is_predicted: bool = False) -> Dict[
        str, Any]:

        # print(f'max_daily_loss_pct: {self.max_daily_loss_pct}')

        """
        Backtest con:
          - múltiples posiciones (si tu política lo permite)
          - equity MTM por barra
          - anti-compounding (compound=False por defecto)
          - daily loss cap / profit cap (opcional)
        """

        action_counts = Counter()
        base_counts = Counter()
        gate_counts = Counter()
        n_steps = 0
        n_trade_actions = 0
        n_flips = 0
        prev_side = 0

        if df_is_predicted:
            df_pred = df_rates
        else:
            df_pred = self.predict(df_rates, simulation=True)

        if self.use_rl and (self.rl_wrapper is not None):
            self.rl_wrapper.logs = []
            if hasattr(self.rl_wrapper, "_buf"):
                self.rl_wrapper._buf = []
            if hasattr(self.rl_wrapper, "pending"):
                self.rl_wrapper.pending = {}

        if self.use_rl and (self.rl_wrapper is not None) and (not bool(getattr(self, "rl_train", True))):
            policy_path = self.rl_policy_path or f"{self.artifacts_path}/rl_policy_gate.npz"
            try:
                load_rl_policy_npz(self.rl_wrapper, policy_path)
            except Exception:
                pass

        balance = float(initial_equity)
        equity_mtm = float(initial_equity)
        used_margin = 0.0
        self._initial_equity_for_bt = float(initial_equity)

        positions: List[MTMPosition] = []
        trades: List[Dict[str, Any]] = []

        balance_curve: List[float] = []
        equity_curve_mtm: List[float] = []
        equity_timestamps: List[Any] = []

        # daily controls
        current_day = None
        daily_realized_pnl = 0.0
        daily_trade_count = 0
        trading_halted_today = False

        max_positions = int(getattr(self.risk_config, "max_positions", 1)) if hasattr(self.risk_config,
                                                                                      "max_positions") else 1

        for i in range(len(df_pred)):
            row = df_pred.iloc[i]
            t = row["time"]
            day = pd.to_datetime(t).date()

            if current_day != day:
                current_day = day
                daily_realized_pnl = 0.0
                daily_trade_count = 0
                trading_halted_today = False

            # 1) cierre por SL/TP (realizado)
            high = float(row["high"].item()) if "high" in row else float(row["close"].item())
            low = float(row["low"].item()) if "low" in row else float(row["close"].item())
            o_ = float(row["open"].item()) if "open" in row else float(row["close"].item())
            c_ = float(row["close"].item()) if "close" in row else float(row[self.mtm_price_col].item())
            atr_ = float(row["atr"].item()) if "atr" in row else np.nan

            # Batch check de exits (MUCHO MÁS RÁPIDO)
            exit_results = self._batch_check_exits_for_positions(positions, o_, high, low, c_, atr_, i)

            # Procesar resultados
            still_open: List[MTMPosition] = []
            for pos, reason, exit_price in exit_results:
                if reason:
                    exit_price = float(exit_price)

                    delta = (exit_price - pos.entry) if pos.side == "long" else (pos.entry - exit_price)
                    spec = self.symbol_specs.get(pos.symbol) if hasattr(self, 'symbol_specs') else None
                    value_per_lot = float(getattr(spec, "contract_size", 0.0) or self.value_per_price_unit)
                    pnl = float(delta * pos.qty * value_per_lot)

                    balance += pnl
                    daily_realized_pnl += pnl
                    daily_trade_count += 1

                    used_margin = max(0.0, float(used_margin) - float(getattr(pos, 'req_margin', 0.0)))

                    trades.append({
                        **asdict(pos),
                        "exit_price": exit_price,
                        "exit_time": t,
                        "pnl": pnl,
                        "exit_reason": reason,
                        "i_exit": i,
                        'exit_mode': getattr(pos, 'exit_mode', None),
                        'sl_confirm_count': getattr(pos, 'confirm_count', None),
                        'last_sweep_i': getattr(pos, 'last_sweep_i', None),
                        'sweep_level': getattr(pos, 'sweep_level', None),
                        'sweep_before_exit': bool(pos.last_sweep_i >= 0),
                    })

                    # --- RL update (reward por trade cerrado) ---
                    if self.use_rl and (self.rl_wrapper is not None):
                        self.rl_wrapper.on_trade_closed(
                            trade_id=int(getattr(pos, 'trade_id', -1)),
                            pnl_money=float(pnl),
                            risk_money=float(self.compute_risk_money(pos.entry, pos.sl, pos.qty,
                                                                     symbol=getattr(pos, 'symbol', None))),
                            meta={"i_exit": int(i), "exit_reason": str(reason),
                                  "i_entry": int(getattr(pos, "i_entry", -1))},
                            train=bool(getattr(self, "rl_train", True)),
                        )
                else:
                    still_open.append(pos)

            positions = still_open

            # 2) aplicar daily caps (sobre realizado)
            if self.max_daily_loss_pct is not None and self.max_daily_loss_pct > 0:
                if daily_realized_pnl <= -self.max_daily_loss_pct * float(initial_equity):
                    trading_halted_today = True

            if (self.max_daily_profit_pct is not None) and (self.max_daily_profit_pct > 0):
                if daily_realized_pnl >= self.max_daily_profit_pct * float(initial_equity):
                    trading_halted_today = True

            if self.max_trades_per_day is not None:
                if daily_trade_count >= self.max_trades_per_day:
                    trading_halted_today = True

            # 3) aperturas (anti-compounding: sizing con equity fija si compound=False)
            # Primero actualizamos equity_mtm para sizing opcional en modo compuesto
            mid = float(row[self.mtm_price_col].item()) if self.mtm_price_col in row else float(row["close"].item())
            unreal = self._unrealized_pnl(positions, mid_price=mid)
            equity_mtm = float(balance + unreal)

            if (not trading_halted_today) and (len(positions) < max_positions):
                sizing_equity = self._select_sizing_equity(
                    initial_equity=float(initial_equity),
                    balance=float(balance),
                    equity_mtm=float(equity_mtm),
                )

                has_pos = len(positions) > 0
                pos_side = positions[0].side if has_pos else None

                dec, order = self.decide_at(df_pred, i, sizing_equity, has_position=has_pos, pos_side=pos_side)
                if dec is None:
                    base_action = 'none'
                else:
                    base_action = dec.action

                # --- RL gate-only: decide TAKE / SKIP sobre la señal base (reward al cierre) ---
                if self.use_rl and (self.rl_wrapper is not None) and (order is not None):
                    take, trade_id, take_prob = self.rl_wrapper.decide(row=row, dec=dec,
                                                                       deterministic=bool(self.rl_eval_deterministic),
                                                                       take_threshold=float(self.rl_take_threshold),
                                                                       train_threshold=float(self.rl_train_threshold),
                                                                       train=bool(self.rl_train))
                    if not take:
                        order = None
                    else:
                        order['trade_id'] = int(trade_id)
                if order is not None:
                    sym = str(order.get('symbol', self.symbol))
                    spec = self._get_spec(sym)
                    if spec is not None:
                        open_sym = sum(1 for p in positions if p.symbol == sym)
                        if open_sym >= int(getattr(spec, 'max_positions', 10 ** 9)):
                            order = None

                if order is not None:
                    ok_margin, req_margin = self._can_open_by_margin(
                        symbol=str(order.get('symbol', self.symbol)),
                        lots=float(order.get('qty', 0.0)),
                        price=float(order.get('entry', np.nan)),
                        equity=float(equity_mtm),
                        used_margin=float(used_margin),
                    )
                    if ok_margin:
                        used_margin += float(req_margin)
                        positions.append(MTMPosition(
                            symbol=str(order.get("symbol", self.symbol)),
                            entry_time=order["entry_time"],
                            side=order["side"],
                            entry=float(order["entry"]),
                            sl=float(order["sl"]),
                            tp=float(order["tp"]),
                            qty=float(order["qty"]),
                            req_margin=float(req_margin),
                            risk_pct=float(order.get("risk_pct", 0.0)),
                            proba_cal=float(order.get("proba_cal", 0.0)),
                            score=float(order.get("score", 0.0)),
                            market_condition=order.get("market_condition", None),
                            regime3=str(order.get("regime3", "")),
                            i_entry=i,
                            trade_id=int(order.get('trade_id', -1)),

                            # --- NUEVO: exits por posición ---
                            exit_mode=str(order.get("exit_mode", getattr(self, "exit_mode", "close_confirm"))),
                            close_confirm_bars=int(
                                order.get("close_confirm_bars", getattr(self, "close_confirm_bars", 1))),
                            firewall_atr_mult=float(
                                order.get("firewall_atr_mult", getattr(self, "firewall_atr_mult", 0.0))),
                            tp_mode=str(order.get("tp_mode", getattr(self, "tp_mode", "wick"))),
                        ))

                    n_steps += 1
                    if dec is None:
                        final_action = 'none'
                    else:
                        final_action = dec.action

                    action_counts[final_action] += 1
                    base_counts[base_action] += 1

                    if final_action in ('buy', 'sell'):
                        n_trade_actions += 1

                    side = +1 if final_action == 'buy' else -1 if final_action == 'sell' else prev_side
                    if prev_side != 0 and side != 0 and side != prev_side:
                        n_flips += 1

                    prev_side = side

                    if base_action in ('buy', 'sell'):
                        if final_action == 'none':
                            gate_counts['blocked_by_rl'] += 1
                        elif final_action == base_action:
                            gate_counts['passed_by_rl'] += 1
                        else:
                            gate_counts['changed_by_rl'] += 1

                    if base_action == 'none' and final_action in ('buy', 'sell'):
                        gate_counts['added_by_rl'] += 1

            # 4) registrar curvas (SIEMPRE alineadas)
            equity_timestamps.append(t)
            balance_curve.append(float(balance))

            # recalcular MTM tras posibles entradas
            mid = float(row[self.mtm_price_col].item()) if self.mtm_price_col in row else float(row["close"].item())
            unreal = self._unrealized_pnl(positions, mid_price=mid)
            equity_mtm = float(balance + unreal)
            equity_curve_mtm.append(float(equity_mtm))

        # métricas
        sharpe_sortino = self.sharpe_sortino_from_equity_daily(
            equity_curve=equity_curve_mtm,
            timestamps=equity_timestamps,
            risk_free_rate_annual=self.risk_free_rate_annual,
            method="simple",
            align="tail",
        )

        summary = self._make_summary(trades, float(initial_equity), float(balance))
        trade_stats_by_period = self._aggregate_trade_stats_by_periods_v2(trades, initial_equity)
        equity_curve = np.array(equity_curve_mtm, dtype=np.float64)
        dd_stats = self._compute_drawdown_stats(equity_curve)

        # --- RL: guardar policy tras entrenamiento ---
        if self.use_rl and (self.rl_wrapper is not None) and bool(getattr(self, "rl_train", True)):
            policy_path = self.rl_policy_path or f"{self.artifacts_path}/rl_policy_gate.npz"
            try:
                save_rl_policy_npz(self.rl_wrapper, policy_path)
            except Exception:
                pass

        if self.use_rl and self.rl_wrapper:
            self.rl_wrapper.flush_updates()

        return {
            "trades": trades,
            "balance_curve": np.array(balance_curve, dtype=np.float64),
            "equity_curve_mtm": np.array(equity_curve_mtm, dtype=np.float64),
            "equity_timestamps": equity_timestamps,
            "final_balance": float(balance),
            "final_equity_mtm": float(equity_mtm),
            "net_pnl": float(balance - float(initial_equity)),
            "n_trades": int(len(trades)),
            "sharpe_sortino_metrics": sharpe_sortino,
            "rl": (self.rl_wrapper.logs if (self.use_rl and self.rl_wrapper is not None) else None),
            "mtm": {
                "spread_price": float(self.spread_price),
                "mtm_use_bid_ask": bool(self.mtm_use_bid_ask),
                "mtm_price_col": str(self.mtm_price_col),
            },
            "anti_compounding": {
                "compound": bool(self.compound),
                "sizing_equity_mode": str(self.sizing_equity_mode),
                "max_daily_loss_pct": float(self.max_daily_loss_pct),
                "max_daily_profit_pct": None if self.max_daily_profit_pct is None else float(self.max_daily_profit_pct),
                "max_trades_per_day": self.max_trades_per_day,
            },
            "summary": summary,
            'trade_stats_by_period': trade_stats_by_period,
            'drawdown_stats': dd_stats,
            'debug_RL_stats': {
                'n_steps': n_steps,
                'n_trade_actions': n_trade_actions,
                'n_flips': n_flips,
                'actions': dict(action_counts),
                'gate': dict(gate_counts)
            }
        }

    def _aggregate_trade_stats_by_period(self, df, *, period: str, initial_equity: float) -> dict:
        if df is None:
            return {}

        df = df.copy()
        if df.empty:
            return {}

        if 'exit_time' not in df.columns or 'pnl' not in df.columns:
            return {}

        df['time'] = pd.to_datetime(df['exit_time'])
        df['pnl'] = df['pnl'].astype(float)

        if period == 'H':
            df['_period'] = df['time'].dt.hour
        else:
            df['_period'] = df['time'].dt.to_period(period)

        out: dict = {}
        for key, g in df.groupby('_period', observed=True):
            pnl = g['pnl']
            wins = pnl[pnl > 0]
            losses = pnl[pnl < 0]

            out[key] = {
                'n_trades': int(len(g)),
                'n_win': int(len(wins)),
                'n_loss': int(len(losses)),
                'win_rate': float(len(wins) / len(g)) if len(g) else 0.0,
                'avg_win': float(wins.mean()) if len(wins) else 0.0,
                'avg_loss': float(losses.mean()) if len(losses) else 0.0,
                'pnl_sum': float(g['pnl'].sum()),
                'pnl_relative': float(g['pnl'].sum() / initial_equity),
            }

        return out

    def _aggregate_trade_stats_by_periods_v2(self, trades: list[dict], initial_equity) -> dict:
        """
        Agrupa PnL y número de operaciones por día, semana, mes y hora.
        Además, desglosa por tipo de orden (buy/sell) si existe en los trades.
        Devuelve:
          {
            "daily":  {"total": ..., "by_side": {"buy": ..., "sell": ...}},
            "weekly": {"total": ..., "by_side": {...}},
            ...
          }
        """
        import pandas as pd

        df = pd.DataFrame(trades)

        period_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'hourly': 'H',
        }

        # Detectar columna de lado/tipo de orden
        side_col = None
        for cand in ("side", "type", "order_type", "direction"):
            if cand in df.columns:
                side_col = cand
                break

        # Normalizar valores buy/sell si existe side_col
        if side_col is not None and not df.empty:
            df[side_col] = (
                df[side_col]
                .astype(str)
                .str.lower()
                .replace({
                    "long": "buy",
                    "short": "sell",
                    "1": "buy",
                    "-1": "sell",
                })
            )

        stats_by_period = {}

        for name, code in period_map.items():
            out = {}

            # Total
            out["total"] = self._aggregate_trade_stats_by_period(
                df, period=code, initial_equity=initial_equity
            )

            # Por tipo de orden
            out["by_side"] = {}
            if side_col is not None and not df.empty:
                for side in ("buy", "sell"):
                    df_side = df[df[side_col] == side]
                    # si no hay trades de ese tipo, devuelve estructura vacía coherente
                    if df_side.empty:
                        out["by_side"][side] = self._aggregate_trade_stats_by_period(
                            df_side, period=code, initial_equity=initial_equity
                        )
                    else:
                        out["by_side"][side] = self._aggregate_trade_stats_by_period(
                            df_side, period=code, initial_equity=initial_equity
                        )

            # Por condicion del mercado
            market_condition_col = 'market_condition'
            out["by_market_condition"] = {}
            if market_condition_col is not None and not df.empty:
                for market_condition in ('low_volatility', 'ranging', 'trending_up', 'trending_down',
                                         'high_volatility'):
                    df_market_condition = df[df[market_condition_col] == market_condition]
                    # si no hay trades de ese tipo, devuelve estructura vacía coherente
                    if df_market_condition.empty:
                        out["by_market_condition"][market_condition] = self._aggregate_trade_stats_by_period(
                            df_market_condition, period=code, initial_equity=initial_equity
                        )
                    else:
                        out["by_market_condition"][market_condition] = self._aggregate_trade_stats_by_period(
                            df_market_condition, period=code, initial_equity=initial_equity
                        )

            stats_by_period[name] = out

        return stats_by_period

    def _aggregate_trade_stats_by_periods(self, trades: list[dict], initial_equity) -> dict:
        """
        Agrupa PnL y número de operaciones por día, semana, mes y hora, y calcula estadísticas.
        """

        df = pd.DataFrame(trades)

        period_map = {
            'daily': 'D',
            'weekly': 'W',
            'monthly': 'M',
            'hourly': 'H',
        }

        stats_by_period = {}
        for name, code in period_map.items():
            stats_by_period[name] = self._aggregate_trade_stats_by_period(
                df, period=code, initial_equity=initial_equity
            )

        return stats_by_period

    def _compute_drawdown_stats(self, equity_curve: np.ndarray) -> Dict[str, Any]:
        if equity_curve.size == 0:
            return {
                "drawdown_curve": np.array([], dtype=np.float64),
                "max_drawdown": 0.0,
                "max_drawdown_pct": 0.0,
                "max_drawdown_start_i": None,
                "max_drawdown_end_i": None,
            }

        peak = np.maximum.accumulate(equity_curve)
        # evitar divide-by-zero por si acaso
        peak_safe = np.where(peak <= 0, np.nan, peak)
        dd = equity_curve / peak_safe - 1.0
        dd = np.nan_to_num(dd, nan=0.0)

        max_dd_pct = float(np.min(dd))  # negativo
        end_i = int(np.argmin(dd))

        # inicio: último pico antes del end_i
        start_i = int(np.argmax(equity_curve[: end_i + 1])) if end_i >= 0 else 0

        max_dd_abs = float(peak[start_i] - equity_curve[end_i])

        return {
            "drawdown_curve": dd.astype(np.float64),
            "max_drawdown": max_dd_abs,
            "max_drawdown_pct": max_dd_pct,
            "max_drawdown_start_i": start_i,
            "max_drawdown_end_i": end_i,
        }