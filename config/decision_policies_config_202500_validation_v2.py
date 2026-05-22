# config/decision_policies_config_202500_validation_v2.py
# Iteración v2 sobre decision_policies_config_202500_validation.py (auto-gen
# por compute_state_percentiles).
#
# Motivación: el primer Lockbox del TCN v4 dio veredicto KO con PnL=-17.20%
# y MDD=-25.61%. El análisis post-mortem (trades.parquet by state+side)
# reveló:
#
#   Bleed concentrado en TREND_UP (ambos sides) y TREND_DOWN:
#     · TREND_UP   LONG  : 278 trades, -1,210R (win_rate 23%)  ← killer principal
#     · TREND_UP   SHORT : 198 trades,   -337R (win_rate 27%)
#     · TREND_DOWN LONG  : 124 trades,   -220R (win_rate 27%)
#     · TREND_DOWN SHORT :  78 trades,   -157R (win_rate 27%)
#   Ganadores:
#     · TRANSITION_DOWN LONG : 81 trades, +492R (win 46%) ← mejor estado
#     · TRANSITION_UP   LONG : 20 trades,  +35R (win 40%)
#     · RANGE           LONG : 27 trades,  +25R (win 44%)
#
# Si eliminamos TREND_UP + TREND_DOWN (ambos sides), proyección ≈ +440R neto.
# SHORT entero se desactiva — todas las celdas SHORT sangraron o tienen n<5.
#
# Cambios vs v1:
#   1. LONG: gate=100 en TREND_UP y TREND_DOWN (no_trade efectivo).
#   2. SHORT: gate=100 en TODOS los estados (no_trade global del side).
#   3. risk_mult/score_cap añaden defensa adicional (0.0 en trend_* y short).
#
# Caveat metodológico: estos cambios se derivan del propio Lockbox, así que
# el rigor unbiased queda parcialmente quemado. Re-validar en lockbox NUEVO
# (periodo distinto) o mediante paper trading antes de promoción real.
#
# selected_threshold LONG  = 0.1659  (heredado del v1)
# selected_threshold SHORT = 0.1051  (heredado, irrelevante porque SHORT off)

# Gate=100 actúa como "no_trade": ningún score percentil supera p100 dentro de
# la distribución del state, así que la señal nunca se materializa.
NO_TRADE = 100

gate_by_action_and_state = {
    "training": {
        "long": {
            "trend_up": NO_TRADE,         # ← v2: disable (era 94)
            "trend_down": NO_TRADE,       # ← v2: disable (era 93)
            "transition_up": 94,
            "transition_down": 94,
            "range": 91,
            "breakout_wait_up": 94,
            "breakout_wait_down": 94,
            "volatile": 93,
            "low_vol": 99,
            "_global": 94,
        },
        "short": {
            "trend_up": NO_TRADE,         # ← v2: disable SHORT entero
            "trend_down": NO_TRADE,
            "transition_up": NO_TRADE,
            "transition_down": NO_TRADE,
            "range": NO_TRADE,
            "breakout_wait_up": NO_TRADE,
            "breakout_wait_down": NO_TRADE,
            "volatile": NO_TRADE,
            "low_vol": NO_TRADE,
            "_global": NO_TRADE,
        },
    },
    "production": {
        "long": {
            "trend_up": NO_TRADE,         # ← v2: disable (era 99)
            "trend_down": NO_TRADE,       # ← v2: disable (era 98)
            "transition_up": 99,
            "transition_down": 99,
            "range": 96,
            "breakout_wait_up": 99,
            "breakout_wait_down": 99,
            "volatile": 98,               # bloqueado por score_cap
            "low_vol": 99,                # bloqueado por score_cap
            "_global": 99,
        },
        "short": {
            "trend_up": NO_TRADE,         # ← v2: disable SHORT entero
            "trend_down": NO_TRADE,
            "transition_up": NO_TRADE,
            "transition_down": NO_TRADE,
            "range": NO_TRADE,
            "breakout_wait_up": NO_TRADE,
            "breakout_wait_down": NO_TRADE,
            "volatile": NO_TRADE,
            "low_vol": NO_TRADE,
            "_global": NO_TRADE,
        },
    },
}

# score_cap_by_state: defensa adicional. Con score_cap=0.0 el score efectivo
# se clipea a 0, así que ningún gate (incluso p50) lo deja pasar.
score_cap_by_state = {
    "training": {
        "trend_up": 0.00,                 # ← v2: defensa anti-bleed
        "trend_down": 0.00,               # ← v2: defensa anti-bleed
        "transition": 1.25, "range": 1.00, "breakout": 1.00,
        "volatile": 0.50, "low_vol": 0.25,
    },
    "production": {
        "trend_up": 0.00,                 # ← v2
        "trend_down": 0.00,               # ← v2
        "transition": 1.05, "range": 0.75, "breakout": 0.90,
        "volatile": 0.00, "low_vol": 0.00,
    },
}

risk_mult_by_state = {
    "training": {
        "trend_up": 0.00,                 # ← v2: tamaño 0 (no_trade defensivo)
        "trend_down": 0.00,               # ← v2
        "transition": 0.75, "range": 0.50, "breakout": 0.50,
        "volatile": 0.00, "low_vol": 0.00,
    },
    "production": {
        "trend_up": 0.00,                 # ← v2
        "trend_down": 0.00,               # ← v2
        "transition": 0.55, "range": 0.25, "breakout": 0.40,
        "volatile": 0.00, "low_vol": 0.00,
    },
}
