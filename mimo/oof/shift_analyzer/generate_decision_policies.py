import json
import numpy as np
import pandas as pd


ALLOWED_PERCENTILES = [75, 80, 85, 90, 95, 99]
NO_TRADE_STATES = {"low_vol", 'volatile'}   # añade "volatile" si decides bloquearlo también


def snap_percentile(p: int | float) -> int:
    p = int(round(float(p)))
    return min(ALLOWED_PERCENTILES, key=lambda x: abs(x - p))


def build_gate_config(summary_df: pd.DataFrame) -> dict:
    """
    Convierte threshold_recalibration_summary.csv en un dict:
      {
        'trend_up': 90,
        'range': 95,
        ...
        '_global': 90,
      }
    """
    states = {}

    for _, row in summary_df.iterrows():
        state = str(row["state"]).lower()

        if state in NO_TRADE_STATES:
            continue

        percentile = snap_percentile(row["recommended_quantile"])
        states[state] = percentile

    if not states:
        raise ValueError("No hay estados operables para construir gate config")

    vals = list(states.values())
    states["_global"] = snap_percentile(np.median(vals))

    return states


def build_gate_by_action_and_state(
    summary_long_training: pd.DataFrame,
    summary_short_training: pd.DataFrame,
    summary_long_production: pd.DataFrame,
    summary_short_production: pd.DataFrame,
) -> dict:
    """
    Construye la estructura completa:
      gate_by_action_and_state = {
          'training': {...},
          'production': {...},
      }
    """
    return {
        "training": {
            "long": build_gate_config(summary_long_training),
            "short": build_gate_config(summary_short_training),
        },
        "production": {
            "long": build_gate_config(summary_long_production),
            "short": build_gate_config(summary_short_production),
        },
    }


def export_gate_config_py(gate_cfg: dict, output_path: str):
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("gate_by_action_and_state=")
        f.write(json.dumps(gate_cfg, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    # Ejemplo:
    # training y production pueden ser iguales si aún no separas ambos procesos
    summary_long = pd.read_csv("threshold_recalibration_report_long/threshold_recalibration_summary.csv")
    summary_short = pd.read_csv("threshold_recalibration_report_short/threshold_recalibration_summary.csv")

    gate_cfg = build_gate_by_action_and_state(
        summary_long_training=summary_long,
        summary_short_training=summary_short,
        summary_long_production=summary_long,
        summary_short_production=summary_short,
    )

    export_gate_config_py(gate_cfg, "../decision_engine_percentiles.py")

    print("✅ Config generada en decision_engine_percentiles.py")
    print(json.dumps(gate_cfg, indent=4, ensure_ascii=False))