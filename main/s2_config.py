from dataclasses import dataclass, field
from typing import FrozenSet

TRAINING_SL_BARRIER_R = 1.5
TRAINING_TP_BARRIER_R = 2.5
TRAINING_LABEL_RR = TRAINING_TP_BARRIER_R / TRAINING_SL_BARRIER_R

# SHORT usa label_method='adaptive' — barriers más pequeños y consistentes
# con los movimientos bajistas reales en M1.
# Ajustar tras analizar holdout del release actual.
TRAINING_SL_BARRIER_R_SHORT = 0.8
TRAINING_TP_BARRIER_R_SHORT = 1.2
TRAINING_LABEL_RR_SHORT = TRAINING_TP_BARRIER_R_SHORT / TRAINING_SL_BARRIER_R_SHORT

MIN_RR_BY_REGIME = {
    "trend_up": 1.0,
    "trend_down": 1.0,
    "transition_up": 1.0,
    "transition_down": 1.0,
    "range": 1.0,
    "breakout_wait_up": 1.0,
    "breakout_wait_down": 1.0,
    "volatile": 1.0,
    "low_vol": 1.0,
    "_default": 1.0,
}

MAX_RR_BY_REGIME = {
    "trend_up": TRAINING_LABEL_RR,
    "trend_down": TRAINING_LABEL_RR,
    "transition_up": 1.2,
    "transition_down": 1.2,
    # FIX v1.1: subido de 1.0 -> TRAINING_LABEL_RR (1.667).
    # Con rr_max=1.0 el TP quedaba simetrico al vSL (1:1). El trailing
    # BE_ARMED (trigger=40pts) interrumpia sistematicamente los ganadores
    # antes de llegar al TP, cerrandolos a avg +0.22R mientras los
    # perdedores completaban -1.05R. EV observado: -0.25R por operacion.
    # El modelo propone consistentemente 1.665-1.668R (training label exacto)
    # -> dejarlo correr hasta ese nivel alinea inferencia con ejecucion.
    "range": TRAINING_LABEL_RR,
    "breakout_wait_up": 1.2,
    "breakout_wait_down": 1.2,
    "volatile": 0.8,
    # FIX v1.1: idem para low_vol (mismo razonamiento que range).
    "low_vol": 1.2,
    "_default": 1.0,
}

MIN_RR_BY_REGIME_SHORT = {
    "trend_up":           0.8,   # contra-tendencia SHORT — RR mínimo más permisivo
    "trend_down":         1.0,
    "transition_up":      0.8,
    "transition_down":    1.0,
    "range":              1.0,
    "breakout_wait_up":   0.8,
    "breakout_wait_down": 1.0,
    "volatile":           0.8,   # más permisivo en volatilidad
    "low_vol":            1.0,
    "_default":           1.0,
}

MAX_RR_BY_REGIME_SHORT = {
    "trend_up":           TRAINING_LABEL_RR_SHORT,
    "trend_down":         TRAINING_LABEL_RR_SHORT,
    "transition_up":      1.0,
    "transition_down":    1.0,
    "range":              TRAINING_LABEL_RR_SHORT,
    "breakout_wait_up":   1.0,
    "breakout_wait_down": TRAINING_LABEL_RR_SHORT,
    "volatile":           0.8,
    "low_vol":            1.0,
    "_default":           TRAINING_LABEL_RR_SHORT,
}


# ============================================================================
# KEEPALIVE
# ============================================================================

@dataclass
class KeepAliveConfig:
    interval_secs: int = 30
    on_new_bar: bool = True


# ============================================================================
# COUNTER-TREND POLICY
# ============================================================================

@dataclass
class CounterTrendConfig:
    """Política de bloqueo/penalización de señales contra-tendencia.

    block_total = True  → rechazar completamente señales contra-tendencia en
                          los regímenes listados (no importa el score).
    block_total = False → permitir si score >= min_score_to_trade + score_penalty.
    """
    block_total: bool = False       ## False or True
    score_penalty: float = 0.25
    regimes_short_block: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"trend_up"})
    )
    regimes_long_block: FrozenSet[str] = field(
        default_factory=lambda: frozenset({"trend_down"})
    )


# ============================================================================
# STRICT REVERSAL GUARDS
# ============================================================================

@dataclass
class ReversalGuardConfig:
    """Guardias extra para entradas contra tendencia fuerte.

    No bloquea por completo el contra-trend, pero exige señales mínimas de
    giro cuando el régimen es claramente adverso.
    """
    enabled: bool = True
    long_in_trend_down: bool = True
    short_in_trend_up: bool = True
    long_min_rsi: float = 46.0
    short_max_rsi: float = 54.0
    require_macd_flip: bool = True
    # 2026-05-20: bajado de 0.02 a 0.015 tras swap calibrador iso→Platt.
    # Análisis OOF (n=3251) con el calibrador nuevo:
    #   · LONG_in_TREND_DOWN edge=|cal_long-cal_short|>=0.02 pasaba 13.1% (era 27.7% con iso)
    #   · LONG_in_TREND_DOWN edge>=0.015 pasa 21% (recupera proximidad al rate iso)
    #   · SHORT_in_TREND_UP edge>=0.015 pasa 55% (apenas cambia)
    # El rango cal_probs con Platt está comprimido (LONG max 0.35 vs iso 1.0),
    # así que el edge típico también es menor. 0.015 mantiene el espíritu del
    # filtro (rechazar reversals tibios) sin estrangular el flow.
    min_proba_edge: float = 0.015
    require_indicators_present: bool = True


# ============================================================================
# OPEN GUARDS
# ============================================================================

@dataclass
class OpenGuardConfig:
    """Guards que S2 evalúa antes de enviar una orden OPEN a S3.

    open_guard_secs:    cooldown mínimo entre dos envíos de OPEN (evita ráfagas
                        si dos ticks llegan en la misma barra).
    max_signal_age_bars: señales más viejas que N barras se descartan.
    max_entry_gap_pts:  gap máximo (en puntos) entre model_entry y precio real
                        (bid/ask) para aceptar la señal. Si ATR está disponible,
                        el threshold es max(max_entry_gap_pts, atr_pts).
    """
    open_guard_secs: float = 5.0
    max_signal_age_bars: int = 1
    max_entry_gap_pts: int = 300


# ============================================================================
# STRATEGY GATE (chop / exhaustion)
# ============================================================================

@dataclass
class StrategyGateConfig:
    """Configuración del StrategyGate (filtros chop/exhaustion sobre la decisión).

    Antes hardcoded en `mimo/strategies/trading_simulator_v3.py:920-924`. Expuesto
    aquí tras incident INC-2026-05-20 para poder ajustar sin tocar código del
    simulator.

    chop_block:        si True, bloquea entradas cuando is_chop=1 (NO_TRADE).
                       Si False, permite entrada con tamaño chop_size_mult.
    chop_size_mult:    multiplicador de tamaño cuando is_chop=1.
                       0.0 = bloqueo total (equivalente a chop_block=True).
                       0.5 = entrada con la mitad del tamaño (defensa parcial).
                       1.0 = sin penalización.
    exhaustion_blocks_reentry: si True, bloquea reentrada en el mismo trend
                       cuando is_exhaustion=1 (defensivo).

    Nota: el detector `is_chop` (feature_builder.py:983-985) usa percentil 85
    móvil, así que por construcción ~15% de las barras tienen is_chop=1.
    Con chop_block=True + chop_size_mult=0.0 (config original), eso bloquea
    ~15% de las entradas potenciales independientemente del régimen.
    """
    chop_block: bool = False
    chop_size_mult: float = 0.5
    exhaustion_blocks_reentry: bool = True


# ============================================================================
# S2 CONFIG (raíz)
# ============================================================================

@dataclass
class S2Config:
    keepalive: KeepAliveConfig = field(default_factory=KeepAliveConfig)
    counter_trend: CounterTrendConfig = field(default_factory=CounterTrendConfig)
    reversal_guard: ReversalGuardConfig = field(default_factory=ReversalGuardConfig)
    open_guard: OpenGuardConfig = field(default_factory=OpenGuardConfig)
    strategy_gate: StrategyGateConfig = field(default_factory=StrategyGateConfig)

    # ── Geometry / order builder ────────────────────────────────────────────
    min_vsl_points: int = 20

    # ── Startup ─────────────────────────────────────────────────────────────
    # ── RSI entry filter (FIX 17/04/2026 BUG-2) ────────────────────────────
    rsi_overbought_threshold: float = 75.0
    rsi_oversold_threshold: float = 25.0

    # ── Transition weak-signal filter (FIX 17/04/2026 BUG-3) ───────────────
    # 2026-05-20: bajado de 0.10 a 0.04 tras análisis empírico.
    # En datos OOF (n=72 muestras en TRANSITION):
    #   · P50=0.021  P85=0.036  P90=0.040  P95=0.050  P99=0.056 (Platt actual)
    #   · P50=0.029  P90=0.055  P99=0.090 (isotónico antiguo)
    # El valor original 0.10 estaba por ENCIMA del P99 de ambos calibradores
    # → bloqueo del 100% de señales en TRANSITION (confirmado en runtime
    # 20/05/2026: 3 SELL consecutivos en TRANSITION_DOWN con deltas 0.017,
    # 0.030, 0.037 todos bloqueados pese a tener scores 0.54-0.65).
    # El caso histórico del 17/04/2026 (3 SELL con delta≈0.088 que perdieron
    # -675pts) era P99 del isotónico — un outlier raro, no la masa de la
    # distribución. Con threshold=0.04 (P90 Platt) bloqueamos 89% del ruido
    # y dejamos pasar señales realmente fuertes para el rango natural del
    # calibrador Platt.
    transition_min_proba_delta: float = 0.04

    startup_grace_bars: int = 1