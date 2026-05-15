"""
s2_main_gbm.py
═══════════════════════════════════════════════════════════════════════════
Entry point del servicio S2 corriendo el modelo GBM LONG-only.

PARALELO A s2_main.py (CNN). Comparten:
  · S2Service (s2_service_v2.py)
  · S2Config, sockets ZMQ, loggers
  · Database, DecisionPolicy base, RiskConfig

DIFERENCIA:
  · Usa GBMTradingSimulator en lugar de TradingSimulator
  · No requiere ModelConfig (GBM no usa secuencias)
  · No requiere artifacts del CNN deploy — usa production/<release>/

CADENCIA:
  · Relanzar el servicio cuando se refresque el modelo (mensual típicamente,
    via `bash 008_deploy_long_only_gbm.sh`)

USO:
  python s2_main_gbm.py [release] [production_dir]

EJEMPLO:
  python s2_main_gbm.py 202603_GBM production/202603_GBM
"""
import json
import sys
import time
from pathlib import Path

import pandas as pd
import zmq

# Reuse infraestructura común del CNN main
from s2_config import S2Config
from s2_service_v2 import S2Service
from s2_main import build_db, build_sockets_and_loggers

# Mimo imports
from mimo.data_managers.databases import Database  # noqa: F401  (re-export)
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config
from mimo.strategies.decision_engine import RiskConfig
from mimo.strategies.gbm_trading_simulator import GBMTradingSimulator
from mimo.strategies.regime_detector import RegimeConfig


def _load_metadata(production_dir: str) -> dict:
    """Lee production_metadata.json para extraer release, barriers, horizons."""
    meta_path = Path(production_dir) / "production_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} no existe. Ejecuta primero:\n"
            f"  RELEASE=<release> bash 008_deploy_long_only_gbm.sh"
        )
    with open(meta_path) as fh:
        return json.load(fh)


def main(release: str, production_dir: str, mode: str = "production"):
    print(f"\n═══ s2_main_gbm ═══")
    print(f"  release         = {release}")
    print(f"  production_dir  = {production_dir}")
    print(f"  mode            = {mode}")

    # Cargar metadata para construir feature_config consistente con lo que
    # fue tuneado/deployado.
    meta = _load_metadata(production_dir)
    print(f"  ↳ trial deployed  = #{meta.get('trial', '?')}")
    print(f"  ↳ as-of           = {meta.get('as_of', '?')}")
    print(f"  ↳ horizon         = {meta.get('horizon', '?')}")
    print(f"  ↳ tp/sl mult      = {meta.get('tp_mult', '?')}/{meta.get('sl_mult', '?')}")

    config = S2Config()
    config.counter_trend.block_total = False
    config.reversal_guard.enabled = True

    general_config = Config(
        release=release,
        oof_splits=5,
        oof_epochs=80,   # ignorado por GBM
    )

    horizon = int(meta.get("horizon", 3))
    tp_mult = float(meta.get("tp_mult", 2.0))
    sl_mult = float(meta.get("sl_mult", 0.8))

    # FeatureConfig: replica el del entrenamiento. El GBM no usa secuencias
    # pero la pipeline necesita FeatureConfig para construir features.
    # label_method="triple_barrier_dual" porque así se entrenó el GBM
    # (multitask, aunque solo usamos LONG en producción).
    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier_dual",
        label_horizon=horizon,
        tp_barrier=tp_mult,
        sl_barrier=sl_mult,
        label_method_long="triple_barrier_dual",
        regime_barriers_long={
            "trending": {"tp": 2.0,  "sl": 0.8},
            "ranging":  {"tp": 1.8,  "sl": 0.8},
            "low_vol":  {"tp": 1.8,  "sl": 0.8},
            "high_vol": {"tp": 2.2,  "sl": 1.0},
        },
        label_method_short="triple_barrier_dual",
        regime_barriers_short={
            "trending": {"tp": 2.0,  "sl": 0.8},
            "ranging":  {"tp": 1.8,  "sl": 0.8},
            "low_vol":  {"tp": 1.8,  "sl": 0.8},
            "high_vol": {"tp": 2.2,  "sl": 1.0},
        },
        tp_barrier_short=None,
        sl_barrier_short=None,
        use_vol_invariant_features=True,
        use_reduced_features=True,
        feature_masks={
            "long": {
                "ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                "ema_bear": True, "rsi_overbought": True, "macd_negative": True,
            },
            "short": {
                "ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                "ema_bear": True, "rsi_overbought": True, "macd_negative": True,
            },
        },
    )

    regime_config = RegimeConfig(adx_trend_threshold=25.0)

    # RiskConfig: misma estructura que el CNN main, sin RL ni decision_policy
    # complejos — el GBM aplica threshold + simple sizing internamente.
    risk_config = RiskConfig(
        base_risk_pct=0.0035,  # 0.35% equity
        min_score_to_trade=0.0,
        max_risk_pct=0.02,
        max_positions=2,
    )

    # GBM simulator (carga artifacts en __init__)
    simulator = GBMTradingSimulator(
        general_config=general_config,
        feature_config=feature_config,
        regime_config=regime_config,
        production_dir=production_dir,
        risk_config=risk_config,
        spread_price=0.07,
        sizing_equity_mode="balance",
        symbol="XAUUSD.r",
    )

    db = build_db()
    sockets, loggers = build_sockets_and_loggers()

    print(f"\n🚀 Iniciando S2Service con GBM LONG-only backend...")
    service = S2Service(
        config=config,
        simulator=simulator,
        db=db,
        sockets=sockets,
        loggers=loggers,
    )
    service.run()


if __name__ == "__main__":
    # CLI: python s2_main_gbm.py [release] [production_dir]
    release = sys.argv[1] if len(sys.argv) > 1 else "202603_GBM"

    if len(sys.argv) > 2:
        production_dir = sys.argv[2]
    else:
        # Default: production/<release>/ relativo al root del repo
        base_dir = Path(__file__).resolve().parent.parent
        production_dir = str(base_dir / "production" / release)

    main(release=release, production_dir=production_dir)
