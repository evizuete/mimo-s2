# -*- coding: utf-8 -*-
"""
decision_engine.py - CORREGIDO
Soporta tanto el sistema de 3 regímenes como el de 8 estados
"""

from __future__ import annotations
import json
from dataclasses import dataclass, field
from typing import Dict, Any, Tuple, Optional
import math
import numpy as np
from mimo.models.model_builder import Config


@dataclass
class DecisionPolicy:
    """Política de decisión (normalizada a *state*).

    - `state` es el driver canónico (8 estados).
    - Campos con sufijo `_regime` quedan como aliases *deprecated* para compatibilidad.
    """
    # Canonical (NEW)
    gate_by_action_and_state: Dict[str, Dict[str, int]] = field(default_factory=dict)
    score_low_quantile: int = 50
    score_high_quantile: int = 99
    require_delta_rel: bool = True
    min_delta_rel: float = 0.15
    score_cap_by_state: Optional[Dict[str, float]] = None
    risk_mult_by_state: Optional[Dict[str, float]] = None
    allow_volatile: bool = False

@dataclass
class RiskConfig:
    base_risk_pct: float = 0.005
    min_score_to_trade: float = 0.15   # v2: subido de 0.10 → 0.15 para filtrar señales débiles
    max_risk_pct: float = 0.02
    eps: float = 1e-9
    max_positions: int = None


# MAPEO ACTUALIZADO - Soporta 8 estados + backward compatibility
STATE_TO_REGIME = {
    # Estados nuevos del state machine
    'range': 'range',
    'transition_up': 'transition_up',
    'transition_down': 'transition_down',
    'breakout_wait_up': 'breakout_wait_up',
    'breakout_wait_down': 'breakout_wait_down',
    'trend_up': 'trend_up',
    'trend_down': 'trend_down',
    'low_vol': 'low_vol',
    'volatile': 'volatile',

    # Backward compatibility con nomenclatura antigua
    'high_volatility': 'volatile',
    'trending_up': 'trend_up',
    'trending_down': 'trend_down',
    'ranging': 'range',
    'low_volatility': 'range',

    # Mapeo genérico de 3 estados (si alguien usa 'trend')
    'trend': 'trend_up',  # Default a trend_up si no especifica dirección
}

# Agrupación para caps y multiplicadores (cuando no están definidos por estado)
STATE_GROUPS = {
    'range': 'range',
    'transition_up': 'transition',
    'transition_down': 'transition',
    'breakout_wait_up': 'breakout',
    'breakout_wait_down': 'breakout',
    'trend_up': 'trend',
    'trend_down': 'trend',
    'volatile': 'volatile',
    'low_vol': 'low_vol'
}


def normalize_state_label(label: str) -> str:
    """Mapea market_condition/regime/state a estado normalizado de 8 tipos"""
    mc = (label or '').strip().lower()
    # Primero buscar mapeo directo
    if mc in STATE_TO_REGIME:
        return STATE_TO_REGIME[mc]
    # Si no existe, buscar con guiones bajos
    mc_underscore = mc.replace('-', '_')
    if mc_underscore in STATE_TO_REGIME:
        return STATE_TO_REGIME[mc_underscore]
    # Default conservador
    print(f"⚠️ Warning: Unknown market condition '{mc}', defaulting to 'range'")
    return 'range'



def map_market_condition_to_state(mc: str) -> str:
    """DEPRECATED: usar `normalize_state_label`.
    Mantiene compatibilidad con versiones anteriores que pasaban market_condition/regime/state.
    """
    return normalize_state_label(mc)

def get_state_group(state: str) -> str:
    """Obtiene el grupo del estado para caps/multiplicadores"""
    return STATE_GROUPS.get(state, 'range')


def clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))


def get_pct(pcts: Dict[str, Any],
            state: str,
            key: int,
            allow_group: bool = False,
            allow_global: bool = False,
) -> Optional[float]:
    """
    Busca percentiles con fallback inteligente:
    1. Intenta con el estado específico (ej: 'breakout_wait_up')
    2. Si no existe, intenta con el grupo (ej: 'trend' para 'trend_up')
    3. Si no existe, intenta con '_global'
    4. Si nada funciona, lanza excepción
    """
    # Intentar estado específico
    if state in pcts:
        try:
            return float(pcts[state]['percentiles'][f'p{key}'])
        except (KeyError, TypeError):
            return None

    # Intentar grupo del estado
    if allow_group:
        group = get_state_group(state)
        if group != state and group in pcts:
            try:
                return float(pcts[group]['percentiles'][f'p{key}'])
            except (KeyError, TypeError):
                return None

    # Intentar _global
    if allow_global and "_global" in pcts:
        try:
            return float(pcts['_global']['percentiles'][f'p{key}'])
        except (KeyError, TypeError):
            return None

    # Si llegamos aquí, no encontramos nada
    return None

def score_from_percentiles(p: float, p_low: float, p_high: float, eps: float = 1e-9) -> float:
    return clip01((p - p_low) / (p_high - p_low + eps))


@dataclass
class DecisionOutput:
    action: str
    state: str  # Cambiado de 'regime' a 'state' para claridad
    macro_regime: str
    p_buy_raw: float
    p_sell_raw: float
    p_buy_cal: float
    p_sell_cal: float
    score_buy: float
    score_sell: float
    chosen_score: float
    risk_pct: float
    debug: Dict[str, Any]

    # Propiedad para backward compatibility
    @property
    def regime(self) -> str:
        return self.state

    def __str__(self) -> str:
        debug = PrettyDict(self.debug)
        return (
            f'\tDecision(action={self.action}, state={self.state}, p_buy={self.p_buy_cal:.3f}, '
            f'p_sell={self.p_sell_cal:.3f}, score_buy={self.score_buy:.2f}, score_sell={self.score_sell:.2f}, '
            f'chosen_score={self.chosen_score:.2f}, risk_pct={self.risk_pct:.2f}, \n'
            f'\t\t\t\tdebug={debug}'
        )


class PrettyDict(dict):
    def __repr__(self) -> str:
        parts = []
        for k, v in self.items():
            if isinstance(v, float):
                parts.append(f'{k}={v:.3f}')
            else:
                parts.append(f'{k}={v}')
        return '{' + ', '.join(parts) + '}'


@dataclass
class AntiNaturalConfig:
    k_body_atr: float = 1.4
    k_range_atr: float = 2.2
    max_body_ratio: float = 0.35
    cooldown_bars: int = 5          # cooldown tras hard_spike (velas antinat)
    z_window: int = 200
    z_range_hi: float = 3.0
    z_body_hi: float = 3.0
    penalty_lambda: float = 1.25
    use_strong_trend_gate: bool = True
    strong_trend_adx: float = 30.0

    # v2: bloqueo explícito por anomaly_score
    # Si anomaly_score >= anomaly_block_threshold → acción bloqueada + cooldown de señal
    anomaly_block_threshold: float = 1.0    # umbral para bloquear la señal (0 = desactivado)
    signal_cooldown_bars: int = 3           # velas de espera tras un bloqueo por anomalía


class DecisionEngine:
    def __init__(
            self,
            general_config: Config,
            policy: DecisionPolicy,
            risk_config: RiskConfig = RiskConfig(),
            path: str = ".",
            antinat_config: AntiNaturalConfig = AntiNaturalConfig(),
    ):
        self.general_config = general_config
        self.policy = policy
        self.risk_config = risk_config
        self.path = path or "."
        self.antinat_config = antinat_config

        self.percentiles = {}
        for action in ['long', 'short']:
            self.percentiles[action] = self._load_percentiles(action)

        # Inicializar caps y multiplicadores con defaults inteligentes
        if self.policy.score_cap_by_state is None:
            self.policy.score_cap_by_state = {
            'trend_up': 1.5,
            'trend_down': 1.5,
            'transition': 1.25,
            'range': 1.0,
            'volatile': 0.75,
        }

        if self.policy.risk_mult_by_state is None:
            self.policy.risk_mult_by_state = {
            'trend_up': 1.0,
            'trend_down': 1.0,
            'transition': 0.75,
            'range': 0.50,
            'breakout': 0.50,
            'volatile': 0.25
        }

        # AntiNatural state
        self.cooldown_left = 0          # cooldown tras hard_spike
        self.signal_cooldown_left = 0   # v2: cooldown tras bloqueo por anomaly_score
        self._ranges = []
        self._bodies = []

    def _load_percentiles(self, action: str, verbose: bool = False) -> Dict[str, Any]:
        filename = f'{self.path}/percentiles_{self.general_config.release}_{action}.json'
        if verbose:
            print(f'Loading percentiles: {filename}')
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                #data = json.load(f)
                data = {k.lower(): v for k, v in json.load(f).items()}
                if verbose:
                    print(f"  States found: {list(data.keys())}")
                return data
        except FileNotFoundError:
            print(f"  ⚠️ File not found, using empty percentiles")
            return {}

    def _check_pass_and_get_score(
            self, p_cal: float, state: str, action: str
    ) -> Tuple[bool, float, int, float, float, float]:
        """
        Verifica gate y calcula score usando el estado específico
        """
        # Obtener gate key del estado específico o fallback
        gate_key = None

        # Buscar en orden: estado específico -> _global -> default
        if state in self.policy.gate_by_action_and_state[action]:
            gate_key = self.policy.gate_by_action_and_state[action][state]
        elif '_global' in self.policy.gate_by_action_and_state[action]:
            gate_key = self.policy.gate_by_action_and_state[action]['_global']
        else:
            gate_key = 95  # Default conservador

        gate_val = get_pct(self.percentiles[action], state, gate_key, allow_group=True, allow_global=False)
        if gate_val is None:
            return False, 0.0, gate_key, np.nan, np.nan, np.nan

        action_pass = float(p_cal) >= gate_val

        if not action_pass:
            return False, 0.0, gate_key, gate_val, np.nan, np.nan

        p_low = get_pct(self.percentiles[action], state, self.policy.score_low_quantile, allow_group=True, allow_global=False)
        p_high = get_pct(self.percentiles[action], state, self.policy.score_high_quantile, allow_group=True, allow_global=False)

        if p_low is None or p_high is None:
            return False, 0.0, gate_key, gate_val, p_low, p_high

        action_score = score_from_percentiles(float(p_cal), p_low, p_high, self.risk_config.eps)
        return True, action_score, gate_key, gate_val, p_low, p_high

    def _update_anomaly(self, o: float, h: float, l: float, c: float, atr: float) -> Dict[str, Any]:
        rng = max(0.0, h - l)
        body = abs(c - o)
        body_ratio = (body / (rng + 1e-12)) if rng > 0 else 1.0

        self._ranges.append(rng)
        self._bodies.append(body)
        if len(self._ranges) > self.antinat_config.z_window:
            self._ranges.pop(0)
        if len(self._bodies) > self.antinat_config.z_window:
            self._bodies.pop(0)

        hard_spike = False
        if np.isfinite(atr) and atr > 0:
            hard_spike = (
                (body > self.antinat_config.k_body_atr * atr) and
                (rng > self.antinat_config.k_range_atr * atr) and
                (body_ratio < self.antinat_config.max_body_ratio)
            )

        anomaly_score = 0.0
        if len(self._ranges) >= 60:
            r = np.asarray(self._ranges, dtype=float)
            b = np.asarray(self._bodies, dtype=float)
            r_mu, r_sd = float(r.mean()), float(r.std(ddof=0) + 1e-12)
            b_mu, b_sd = float(b.mean()), float(b.std(ddof=0) + 1e-12)

            z_range = (rng - r_mu) / r_sd
            z_body = (body - b_mu) / b_sd

            anomaly_score += 0.8 * max(0.0, z_range / self.antinat_config.z_range_hi)
            anomaly_score += 0.6 * max(0.0, z_body / self.antinat_config.z_body_hi)

        if np.isfinite(atr) and atr > 0:
            anomaly_score += 0.25 * max(0.0, (rng / atr) - 1.0)
            anomaly_score += 0.20 * max(0.0, (body / atr) - 0.8)

        # cooldown hard_spike
        if self.cooldown_left > 0:
            self.cooldown_left -= 1
        if hard_spike:
            self.cooldown_left = max(self.cooldown_left, self.antinat_config.cooldown_bars)

        # cooldown por anomalía (decrementar independientemente)
        if self.signal_cooldown_left > 0:
            self.signal_cooldown_left -= 1

        return {
            "hard_spike": bool(hard_spike),
            "cooldown_left": int(self.cooldown_left),
            "signal_cooldown_left": int(self.signal_cooldown_left),
            "anomaly_score": float(anomaly_score),
        }

    # -----------------------------
    # Exit-mode suggestion (teaser):
    # -----------------------------
    # En XAUUSD M1 es habitual el "sweep" (mechas que barren stops). Para reducir
    # cierres prematuros en rango/transición/breakout, sugerimos usar stop por
    # confirmación de cierre (close_confirm). En tendencias limpias, permitir
    # stop duro puede ser razonable.
    def suggest_exit_mode(self, state: str) -> str:
        g = get_state_group(state)
        if g in ("range", "transition", "breakout", "volatile"):
            return "close_confirm"
        if g == "trend":
            return "hard"
        return "close_confirm"

    def decide_at_bar(
            self,
            *,
            p_buy_raw: float,
            p_sell_raw: float,
            p_buy_cal: float,
            p_sell_cal: float,
            state_raw: Optional[str] = None,
            macro_regime: Optional[str] = None,
            market_condition: Optional[str] = None,  # DEPRECATED alias

            o: float,
            h: float,
            l: float,
            c: float,
            atr: float,
            adx14: Optional[float] = None,
            trend_dir: Optional[int] = None,
            enable_antinat: bool = True,
    ) -> DecisionOutput:
        """
        Decisión usando el estado específico de 8 tipos
        """
        # Normalización canónica:
        # - `state_raw` es el input preferido (state de 8 clases).
        # - `market_condition` se mantiene solo por compatibilidad (DEPRECATED).
        if state_raw is None:
            state_raw = market_condition
        if state_raw is None:
            state_raw = "range"
        state = normalize_state_label(state_raw)
        state_group = get_state_group(state)
        macro_regime_norm = (macro_regime or '').strip().lower()

        # Bloqueo de estados no operables
        if state in {"low_vol", "volatile"}:
            return DecisionOutput(
                action='none',
                state=state,
                macro_regime=macro_regime_norm,
                p_buy_raw=float(p_buy_raw),
                p_sell_raw=float(p_sell_raw),
                p_buy_cal=float(p_buy_cal),
                p_sell_cal=float(p_sell_cal),
                score_buy=0.0,
                score_sell=0.0,
                chosen_score=0.0,
                risk_pct=0.0,
                debug={
                    'reason': f'{state}_blocked',
                    'state_group': state_group
                }
            )

        # Anti-natural layer (sin cambios)
        an = {"hard_spike": False, "cooldown_left": 0, "anomaly_score": 0.0}
        if enable_antinat:
            an = self._update_anomaly(o=o, h=h, l=l, c=c, atr=float(atr) if atr is not None else np.nan)

            if an["hard_spike"]:
                return DecisionOutput(
                    action="none",
                    state=state,
                    macro_regime=macro_regime_norm,
                    p_buy_raw=float(p_buy_raw),
                    p_sell_raw=float(p_sell_raw),
                    p_buy_cal=float(p_buy_cal),
                    p_sell_cal=float(p_sell_cal),
                    score_buy=0.0,
                    score_sell=0.0,
                    chosen_score=0.0,
                    risk_pct=0.0,
                    debug={
                        "reason": "HARD_SPIKE",
                        'state_group': state_group,
                        **an
                    },
                )

            if an["cooldown_left"] > 0:
                return DecisionOutput(
                    action="none",
                    state=state,
                    macro_regime=macro_regime_norm,
                    p_buy_raw=float(p_buy_raw),
                    p_sell_raw=float(p_sell_raw),
                    p_buy_cal=float(p_buy_cal),
                    p_sell_cal=float(p_sell_cal),
                    score_buy=0.0,
                    score_sell=0.0,
                    chosen_score=0.0,
                    risk_pct=0.0,
                    debug={
                        "reason": "COOLDOWN",
                        'state_group': state_group,
                        **an
                    },
                )

            # v2: bloqueo explícito por anomaly_score
            # Si anomaly_score supera el umbral, bloquear y armar signal_cooldown
            _anomaly_thr = float(self.antinat_config.anomaly_block_threshold)
            if _anomaly_thr > 0 and float(an["anomaly_score"]) >= _anomaly_thr:
                # Armar cooldown de señal para las siguientes velas
                self.signal_cooldown_left = max(
                    self.signal_cooldown_left,
                    self.antinat_config.signal_cooldown_bars
                )
                return DecisionOutput(
                    action="none",
                    state=state,
                    macro_regime=macro_regime_norm,
                    p_buy_raw=float(p_buy_raw),
                    p_sell_raw=float(p_sell_raw),
                    p_buy_cal=float(p_buy_cal),
                    p_sell_cal=float(p_sell_cal),
                    score_buy=0.0,
                    score_sell=0.0,
                    chosen_score=0.0,
                    risk_pct=0.0,
                    debug={
                        "reason": "ANOMALY_BLOCKED",
                        "state_group": state_group,
                        "anomaly_block_threshold": _anomaly_thr,
                        **an,
                    },
                )

            # v2: cooldown de señal post-anomalía
            if self.signal_cooldown_left > 0:
                return DecisionOutput(
                    action="none",
                    state=state,
                    macro_regime=macro_regime_norm,
                    p_buy_raw=float(p_buy_raw),
                    p_sell_raw=float(p_sell_raw),
                    p_buy_cal=float(p_buy_cal),
                    p_sell_cal=float(p_sell_cal),
                    score_buy=0.0,
                    score_sell=0.0,
                    chosen_score=0.0,
                    risk_pct=0.0,
                    debug={
                        "reason": "SIGNAL_COOLDOWN",
                        "state_group": state_group,
                        **an,
                    },
                )

        # Verificar gates y calcular scores usando estado específico
        buy_pass, score_buy, buy_gate_key, buy_gate_val, buy_p_low, buy_p_high = self._check_pass_and_get_score(
            p_buy_cal, state, 'long'
        )
        sell_pass, score_sell, sell_gate_key, sell_gate_val, sell_p_low, sell_p_high = self._check_pass_and_get_score(
            p_sell_cal, state, 'short'
        )

        # Strong-trend contra gate
        if enable_antinat and self.antinat_config.use_strong_trend_gate:
            try:
                adx = float(adx14) if adx14 is not None else float("nan")
            except Exception:
                adx = float("nan")
            try:
                td = int(trend_dir) if trend_dir is not None else 0
            except Exception:
                td = 0

            if np.isfinite(adx) and adx >= self.antinat_config.strong_trend_adx and td != 0:
                if td == -1:
                    buy_pass = False
                elif td == +1:
                    sell_pass = False

        # Aplicar caps por estado (buscar específico primero, luego grupo)
        cap = self.policy.score_cap_by_state.get(
            state,
            self.policy.score_cap_by_state.get(state_group, 1.0)
        )
        score_buy = float(min(score_buy, cap))
        score_sell = float(min(score_sell, cap))

        # Si ninguno pasa el gate
        if (not buy_pass) and (not sell_pass):
            return DecisionOutput(
                action="none",
                state=state,
                macro_regime=macro_regime_norm,
                p_buy_raw=float(p_buy_raw),
                p_sell_raw=float(p_sell_raw),
                p_buy_cal=float(p_buy_cal),
                p_sell_cal=float(p_sell_cal),
                score_buy=float(score_buy),
                score_sell=float(score_sell),
                chosen_score=0.0,
                risk_pct=0.0,
                debug={
                    "reason": "no_gate_pass",
                    "state_group": state_group,
                    "buy_gate": buy_gate_key,
                    "sell_gate": sell_gate_key,
                    "buy_gate_val": float(buy_gate_val),
                    "sell_gate_val": float(sell_gate_val),
                    **an,
                },
            )

        # Penalización por anomalía
        penalty = 1.0
        if enable_antinat:
            penalty = math.exp(-self.antinat_config.penalty_lambda * float(an.get("anomaly_score", 0.0)))
            score_buy = float(score_buy * penalty)
            score_sell = float(score_sell * penalty)

        # Determinar acción
        action = "none"
        chosen_score = 0.0

        if buy_pass and not sell_pass:
            action = "buy"
            chosen_score = score_buy
        elif sell_pass and not buy_pass:
            action = "sell"
            chosen_score = score_sell
        else:
            mx = max(score_buy, score_sell) + self.risk_config.eps
            delta_rel = abs(score_buy - score_sell) / mx

            if self.policy.require_delta_rel and (delta_rel < self.policy.min_delta_rel):
                return DecisionOutput(
                    action="none",
                    state=state,
                    macro_regime=macro_regime_norm,
                    p_buy_raw=float(p_buy_raw),
                    p_sell_raw=float(p_sell_raw),
                    p_buy_cal=float(p_buy_cal),
                    p_sell_cal=float(p_sell_cal),
                    score_buy=float(score_buy),
                    score_sell=float(score_sell),
                    chosen_score=0.0,
                    risk_pct=0.0,
                    debug={
                        "reason": "delta_rel_low",
                        'state_group': state_group,
                        "delta_rel": float(delta_rel),
                        "penalty": float(penalty),
                        **an
                    },
                )

            if score_buy >= score_sell:
                action = "buy"
                chosen_score = score_buy
            else:
                action = "sell"
                chosen_score = score_sell

        # Verificar score mínimo
        if chosen_score < self.risk_config.min_score_to_trade:
            return DecisionOutput(
                action="none",
                state=state,
                macro_regime=macro_regime_norm,
                p_buy_raw=float(p_buy_raw),
                p_sell_raw=float(p_sell_raw),
                p_buy_cal=float(p_buy_cal),
                p_sell_cal=float(p_sell_cal),
                score_buy=float(score_buy),
                score_sell=float(score_sell),
                chosen_score=float(chosen_score),
                risk_pct=0.0,
                debug={
                    "reason": "score_below_min",
                    'state_group': state_group,
                    "min_score": self.risk_config.min_score_to_trade,
                    **an
                },
            )

        # Calcular riesgo dinámico
        risk_mult = self.policy.risk_mult_by_state.get(
            state,
            self.policy.risk_mult_by_state.get(state_group, 1.0)
        )
        risk_pct = self.risk_config.base_risk_pct * chosen_score * risk_mult
        risk_pct = float(min(risk_pct, self.risk_config.max_risk_pct))

        return DecisionOutput(
            action=action,
            state=state,
            macro_regime=macro_regime_norm,
            p_buy_raw=float(p_buy_raw),
            p_sell_raw=float(p_sell_raw),
            p_buy_cal=float(p_buy_cal),
            p_sell_cal=float(p_sell_cal),
            score_buy=float(score_buy),
            score_sell=float(score_sell),
            chosen_score=float(chosen_score),
            risk_pct=float(risk_pct),
            debug={
                'reason': 'trade',
                'state_group': state_group,
                "score_cap": cap,
                "risk_mult": risk_mult,
                "penalty": float(penalty),
                "exit_mode_suggested": self.suggest_exit_mode(state),
                **an,
            },
        )

    def decide(
        self,
        p_buy_cal: float,
        p_sell_cal: float,
        *,
        state_raw: Optional[str] = None,
        market_condition: Optional[str] = None,  # DEPRECATED alias
    ) -> DecisionOutput:
        """Backward compatibility (usa `state_raw` como canónico)."""
        if state_raw is None:
            state_raw = market_condition
        return self.decide_at_bar(
            p_buy_raw=p_buy_cal,
            p_sell_raw=p_sell_cal,
            p_buy_cal=p_buy_cal,
            p_sell_cal=p_sell_cal,
            state_raw=state_raw,
            market_condition=market_condition,
            o=np.nan, h=np.nan, l=np.nan, c=np.nan, atr=np.nan,
            enable_antinat=False,
        )