"""
gbm_trading_simulator.py
═══════════════════════════════════════════════════════════════════════════
Simulador GBM LONG-only para producción. Drop-in para S2Service:
expone la MISMA interfaz contractual que TradingSimulator (CNN) sin
herencia, manteniendo el CNN 100% intacto y coexistente.

INTERFAZ EXPUESTA A S2Service:
  · simulator.pipeline.prepare_data(df_rates)
  · simulator.helper.load_from_database_real(...)
  · simulator.decide_live(df_rates, equity, current_positions, max_positions)
  · simulator._last_diag  (dict con diagnóstico cuando devuelve None)

LÓGICA DE DECISIÓN:
  1. prepare_data sobre df_rates → df_prepared
  2. Última fila con features válidas → X (shape 1×N)
  3. Predict booster → p_raw
  4. Calibrate si calibrator está disponible → p_cal (else p_cal = p_raw)
  5. Threshold check (raw, porque deploy usa raw probs por default):
       · si p_raw >= threshold AND current_positions < max_positions → LONG order
       · sino → reject con diag detallado en _last_diag

ARTIFACTS REQUERIDOS en production_dir/:
  · production_booster_long.joblib       (LightGBM Booster)
  · production_threshold.json            (thr + scan stats)
  · production_features.json             (feat_cols esperados)
  · production_metadata.json             (release, trial, fechas, tp/sl)
  · production_calibrator_long.joblib    (opcional)

NO soporta SHORT — por diseño. SHORT se deshabilitó tras 2 holdouts colapsados
(-221R en walkforward de 15 ventanas, real regime shift).
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.helpers.helper import Helper
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig


class GBMTradingSimulator:
    """Simulador GBM LONG-only standalone, compatible con S2Service."""

    def __init__(
        self,
        *,
        general_config: Config,
        feature_config: FeatureConfig,
        regime_config,
        production_dir: str,
        risk_config=None,
        spread_price: float = 0.07,
        sizing_equity_mode: str = "balance",
        symbol: str = "XAUUSD.r",
        # Reservado para compatibilidad futura — no se usa en GBM aún
        decision_policy=None,
        **_unused_kwargs,
    ):
        self.general_config = general_config
        self.feature_config = feature_config
        self.regime_config = regime_config
        self.production_dir = Path(production_dir)
        self.risk_config = risk_config
        self.spread_price = float(spread_price)
        self.sizing_equity_mode = str(sizing_equity_mode)
        self.symbol = str(symbol)
        self.decision_policy = decision_policy  # placeholder

        # Pipeline + helper (mismos que el CNN, model-agnostic)
        self.pipeline = DataPipeline(
            general_config=general_config,
            feature_config=feature_config,
            model_config=ModelConfig(
                seq_len_short=64, seq_len_long=256, target_type="multitask",
            ),  # ignored por GBM pero requerido por DataPipeline init
            regime_config=regime_config,
        )
        self.helper = Helper(general_config=general_config, path=str(self.production_dir))

        # GBM artifacts (load on init para fail-fast)
        self.booster = None
        self.calibrator = None
        self.threshold = None
        self.feat_cols: List[str] = []
        self.metadata: Dict[str, Any] = {}
        self.loaded = False
        self._last_diag: Optional[Dict[str, Any]] = None

        self.load_artifacts()

    # ─── Public API ─────────────────────────────────────────────────

    def load_artifacts(self) -> None:
        prod = self.production_dir
        booster_path  = prod / "production_booster_long.joblib"
        thr_path      = prod / "production_threshold.json"
        feat_path     = prod / "production_features.json"
        meta_path     = prod / "production_metadata.json"
        cal_path      = prod / "production_calibrator_long.joblib"

        for p in (booster_path, thr_path, feat_path, meta_path):
            if not p.exists():
                raise FileNotFoundError(f"GBM artifact missing: {p}")

        self.booster = joblib.load(booster_path)
        with open(thr_path) as fh:
            thr_data = json.load(fh)
            self.threshold = float(thr_data["threshold"])
            self.use_calibrator_at_train = bool(thr_data.get("use_calibrator", False))
        with open(feat_path) as fh:
            self.feat_cols = list(json.load(fh)["feat_cols"])
        with open(meta_path) as fh:
            self.metadata = json.load(fh)

        # Calibrator: solo si existe Y se usó al deployar
        if cal_path.exists() and self.use_calibrator_at_train:
            self.calibrator = joblib.load(cal_path)
            print(f"[GBM-SIM] Calibrator cargado: {cal_path}")
        else:
            self.calibrator = None
            if cal_path.exists():
                print(f"[GBM-SIM] Calibrator presente pero deshabilitado (raw probs)")

        self.loaded = True
        print(f"[GBM-SIM] ✅ Artifacts cargados desde {prod}")
        print(f"[GBM-SIM]   threshold = {self.threshold:.4f}")
        print(f"[GBM-SIM]   features  = {len(self.feat_cols)}")
        print(f"[GBM-SIM]   release   = {self.metadata.get('release', '?')}")
        print(f"[GBM-SIM]   trial     = #{self.metadata.get('trial', '?')}")
        print(f"[GBM-SIM]   as-of     = {self.metadata.get('as_of', '?')}")

    def decide_live(
        self,
        *,
        df_rates: pd.DataFrame,
        equity: float,
        current_positions: Optional[int] = 0,
        max_positions: Optional[int] = 2,
        # Parámetros aceptados por compat con TradingSimulator pero no usados
        symbol: Optional[str] = None,
        open_positions: Optional[List[Dict[str, Any]]] = None,
        i: Optional[int] = None,
        risk_cash_cap: Optional[float] = None,
        portfolio_risk_cap: Optional[float] = None,
        max_qty: Optional[float] = None,
        **_unused,
    ) -> Optional[Dict[str, Any]]:
        """Decide LONG order o None. Si None, popula self._last_diag."""
        self._last_diag = None

        # 0. Position limit (chequeo barato primero)
        cur = int(current_positions or 0)
        mx  = int(max_positions or 2)
        if cur >= mx:
            self._reject({
                "no_signal_reason": "POSITION_LIMIT",
                "current_positions": cur, "max_positions": mx,
            })
            return None

        # 1. prepare_data
        try:
            df_prep = self.pipeline.prepare_data(
                df_rates, labels=False, side="both",
                set_market_condition=False, ensure_regime=True,
            )
        except Exception as e:
            self._reject({"no_signal_reason": "PREPARE_DATA_FAILED",
                          "error": str(e)[:200]})
            return None

        # 2. Última fila con features válidas
        miss = [c for c in self.feat_cols if c not in df_prep.columns]
        if miss:
            self._reject({"no_signal_reason": "FEATURES_MISSING",
                          "missing": miss[:5], "n_missing": len(miss)})
            return None
        valid = df_prep[self.feat_cols].notna().all(axis=1)
        if not valid.any():
            self._reject({"no_signal_reason": "NO_VALID_ROWS"})
            return None
        last_row = df_prep.loc[valid].iloc[-1]

        # 3. Predict
        X = last_row[self.feat_cols].astype(np.float32).values.reshape(1, -1)
        p_raw = float(self.booster.predict(X)[0])
        if self.calibrator is not None:
            p_cal = float(self.calibrator.predict(np.array([p_raw]))[0])
        else:
            p_cal = p_raw

        # 4. Construir diag base (poblado siempre)
        base_diag = {
            "proba_long_raw":   round(p_raw, 6),
            "proba_long_cal":   round(p_cal, 6),
            "proba_short_raw":  0.0,
            "proba_short_cal":  0.0,
            "threshold":        round(float(self.threshold), 6),
            "state":            str(last_row.get("state", "")),
            "macro_regime":     str(last_row.get("macro_regime", last_row.get("regime", ""))),
            "atr":              round(float(last_row.get("atr", 0) or 0), 4),
            "rsi":              float(last_row.get("rsi", float("nan"))) if "rsi" in last_row else None,
            "macd_hist":        float(last_row.get("macd_hist", float("nan"))) if "macd_hist" in last_row else None,
            "model_backend":    "gbm",
            "release":          self.metadata.get("release", ""),
        }

        # 5. Threshold check (raw vs threshold seleccionado en deploy)
        if p_raw < self.threshold:
            self._reject({
                "no_signal_reason": "BELOW_THRESHOLD",
                **base_diag,
            })
            return None

        # 6. ATR check (necesario para sizing)
        atr = float(last_row.get("atr", 0) or 0)
        if atr <= 0:
            self._reject({
                "no_signal_reason": "ATR_INVALID",
                **base_diag,
            })
            return None

        # 7. Risk sizing
        sl_mult = float(self.metadata.get("sl_mult", 0.8))
        tp_mult = float(self.metadata.get("tp_mult", 2.0))
        sl_dist = sl_mult * atr
        tp_dist = tp_mult * atr

        risk_pct = 0.005   # default 0.5% equity
        max_risk_pct = 0.02
        if self.risk_config is not None:
            risk_pct = float(getattr(self.risk_config, "base_risk_pct", risk_pct))
            max_risk_pct = float(getattr(self.risk_config, "max_risk_pct", max_risk_pct))
        risk_pct = min(risk_pct, max_risk_pct)

        risk_cash = float(equity) * risk_pct
        if risk_cash_cap is not None:
            risk_cash = min(risk_cash, float(risk_cash_cap))
        qty = risk_cash / max(sl_dist, 1e-9)
        if max_qty is not None:
            qty = min(qty, float(max_qty))
        if qty <= 0:
            self._reject({
                "no_signal_reason": "QTY_ZERO",
                **base_diag,
                "equity": float(equity), "atr": atr, "sl_dist": sl_dist,
            })
            return None

        # 8. Build order
        close_price = float(last_row.get("close", 0) or 0)
        entry_time  = pd.to_datetime(last_row.get("time")) if "time" in last_row.index else pd.Timestamp.utcnow()

        return {
            "side":          "long",
            "symbol":        self.symbol,
            "entry":         close_price,
            "sl":            close_price - sl_dist,
            "tp":            close_price + tp_dist,
            "qty":           float(qty),
            "state":         str(last_row.get("state", "")),
            "score":         float(p_raw),  # raw prob como score
            "proba_long":    float(p_cal),
            "proba_short":   0.0,
            "rsi":           float(last_row.get("rsi", float("nan"))) if "rsi" in last_row else None,
            "macd_hist":     float(last_row.get("macd_hist", float("nan"))) if "macd_hist" in last_row else None,
            "atr":           atr,
            "atr_at_entry":  atr,
            "entry_time":    entry_time,
            "model_backend": "gbm",
            "release":       self.metadata.get("release", ""),
            "threshold_used": float(self.threshold),
            "tp_mult":       tp_mult,
            "sl_mult":       sl_mult,
        }

    # ─── Internal helpers ───────────────────────────────────────────

    def _reject(self, diag: Dict[str, Any]) -> None:
        """Popula _last_diag con campos consistentes con CNN simulator."""
        # Asegurar campos mínimos para que S2Service no crashee
        diag.setdefault("proba_long_raw", None)
        diag.setdefault("proba_long_cal", None)
        diag.setdefault("proba_short_raw", 0.0)
        diag.setdefault("proba_short_cal", 0.0)
        diag.setdefault("state", "")
        diag.setdefault("atr", None)
        diag.setdefault("model_backend", "gbm")
        self._last_diag = diag
