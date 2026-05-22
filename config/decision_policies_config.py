# =========================================================
# DECISION POLICIES CONFIG — Release 200383
# =========================================================
# LONG:  label_method='triple_barrier', regime_barriers_long
# SHORT: label_method='adaptive',       regime_barriers_short
#
# Gates expresados como percentiles OOF por régimen.
# Los percentiles de referencia están en:
#   percentiles_200383_long.json
#   percentiles_200383_short.json
#
# Umbrales globales de referencia (deploy_calibration_tail):
#   LONG  → selected_threshold = 0.1429
#   SHORT → selected_threshold = 0.1726
#
# Nota importante:
#   - LOW_VOL viene marcado como no_trade=True en ambos percentiles.
#   - BREAKOUT_WAIT_* no aparece en el tail reciente; se usan proxies
#     prudentes tomados de regímenes cercanos.
#   - VOLATILE y LOW_VOL siguen bloqueados más abajo por score_cap/risk_mult,
#     así que sus gates quedan documentados pero sin efecto operativo mientras
#     ese bloqueo siga activo.
# =========================================================

gate_by_action_and_state = {

    "training": {
        "long": {
            "trend_up":           95,
            "trend_down":         97,
            "breakout_wait_up":   99,
            "breakout_wait_down": 99,
            "range":              95,
            "transition_up":      97,
            "transition_down":    95,
            "_global":            95,
        },
        "short": {
            "trend_up":           98,
            "trend_down":         97,
            "breakout_wait_up":   99,
            "breakout_wait_down": 99,
            "range":              98,
            "transition_up":      99,
            "transition_down":    97,
            "_global":            98,
        },
    },
    'production': {
        "long": {
        "trend_up":           90,   # mantener
        "trend_down":         95,   # mantener
        "transition_up":      60,   # antes 50
        "transition_down":    60,   # antes 50
        "range":              80,   # mantener; 80/90/95 casi no cambia
        "breakout_wait_up":   90,   # antes 85
        "breakout_wait_down": 95,   # mantener
        "volatile":           99,   # sin efecto operativo real
        "low_vol":            99,   # sin operativa
        "_global":            97,   # antes 75
        },
        "short": {
            "trend_down":         90,   # antes 85
            "trend_up":           98,   # mantener
            "transition_down":    95,   # antes 80
            "transition_up":      97,   # antes 80
            "range":              95,   # antes 90
            "breakout_wait_down": 90,   # antes 85
            "breakout_wait_up":   95,   # antes 90
            "volatile":           99,   # sin efecto operativo real
            "low_vol":            99,   # sin operativa
            "_global":            99,   # antes 85
        }
    }
}


# =========================================================
# LÍMITES DE SCORE POR ESTADO
# =========================================================
# score_cap: limita el score máximo que puede generar un régimen.
#   0.00 → bloqueo total independientemente del score del modelo.
# risk_mult: multiplica el riesgo base por trade.
#   0.00 → posición de tamaño cero (bloqueo efectivo).
# =========================================================

score_cap_by_state = {
    "training": {
        "trend_up":    1.50,
        "trend_down":  1.50,
        "transition":  1.25,
        "range":       1.00,
        "breakout":    1.00,
        "volatile":    0.50,
        "low_vol":     0.25,
    },
    "production": {
        "trend_up":    1.50,
        "trend_down":  1.50,
        "transition":  1.05,
        "range":       0.75,
        "breakout":    0.90,
        # Volátil y low_vol: bloqueados
        "volatile":    0.00,
        "low_vol":     0.00,
    },
}

risk_mult_by_state = {
    "training": {
        "trend_up":    1.00,
        "trend_down":  1.00,
        "transition":  0.75,
        "range":       0.50,
        "breakout":    0.50,
        "volatile":    0.00,
        "low_vol":     0.00,
    },
    "production": {
        "trend_up":    1.00,
        "trend_down":  1.00,
        "transition":  0.55,
        "range":       0.25,
        "breakout":    0.40,
        # Volátil y low_vol: bloqueados
        "volatile":    0.00,
        "low_vol":     0.00,
    },
}
