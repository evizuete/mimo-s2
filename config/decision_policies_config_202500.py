# config/decision_policies_config_202500.py
# Auto-generado por compute_state_percentiles.py
# selected_threshold LONG  = 0.1899
# selected_threshold SHORT = 0.1500

gate_by_action_and_state = {
    "training": {
        "long": {
            "trend_up": 93,
            "trend_down": 91,
            "transition_up": 91,
            "transition_down": 90,
            "range": 90,
            "breakout_wait_up": 94,
            "breakout_wait_down": 94,
            "volatile": 94,
            "low_vol": 90,
            "_global": 92,
        },
        "short": {
            "trend_up": 94,
            "trend_down": 94,
            "transition_up": 92,
            "transition_down": 90,
            "range": 91,
            "breakout_wait_up": 94,
            "breakout_wait_down": 94,
            "volatile": 94,
            "low_vol": 94,
            "_global": 93,
        },
    },
    'production': {
        "long": {
            "trend_down":         85,   # ÚNICO setup permitido
            # Bloqueos
            "trend_up":           99,
            "range":              99,
            "transition_up":      99,
            "transition_down":    99,
            "breakout_wait_up":   99,
            "breakout_wait_down": 99,
            "volatile":           99,
            "low_vol":            99,
            "_global":            99,
        },
        "short": {
            "trend_up":           85,   # ÚNICO setup permitido
            # Bloqueos
            "trend_down":         99,
            "range":              99,
            "transition_up":      99,
            "transition_down":    99,
            "breakout_wait_up":   99,
            "breakout_wait_down": 99,
            "volatile":           99,
            "low_vol":            99,
            "_global":            99,
        },
    }
}

# score_cap y risk_mult: punto de partida copiado del 200383.
# Ajusta según riesgo aceptable per régimen.
score_cap_by_state = {
    "training": {
        "trend_up": 1.50, "trend_down": 1.50,
        "transition": 1.25, "range": 1.00, "breakout": 1.00,
        "volatile": 0.50, "low_vol": 0.25,
    },
    "production": {
        "trend_up": 1.50, "trend_down": 1.50,
        "transition": 1.05, "range": 0.75, "breakout": 0.90,
        "volatile": 0.00, "low_vol": 0.00,
    },
}

risk_mult_by_state = {
    "training": {
        "trend_up": 1.00, "trend_down": 1.00,
        "transition": 0.75, "range": 0.50, "breakout": 0.50,
        "volatile": 0.00, "low_vol": 0.00,
    },
    "production": {
        "trend_up": 1.00,
        "trend_down": 1.00,
        "transition": 0.00,  # bloqueo total
        "range": 0.00,  # bloqueo total
        "breakout": 0.00,  # bloqueo total
        "volatile": 0.00,
        "low_vol": 0.00,
    },
}