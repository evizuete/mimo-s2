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
import logging
import os
from datetime import datetime
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.helpers.helper import Helper
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig


# Features clave a loguear por inferencia. Selección basada en feature
# importance + SHAP del best trial 202603 (top-25 por gain/SHAP) +
# básicas OHLCV. Si una feature no está en el df, se loguea como None.
_MARKET_FEATURES_TO_LOG: List[str] = [
    # OHLCV básicas (siempre presentes)
    "open", "high", "low", "close", "atr",
    # SHAP top (volatilidad / range / volume)
    "range_hl_rel", "atr_norm_bps_z", "realized_vol_20_bps_z",
    "vol_trend_1h", "vol_spike", "vol_z_1h", "bb_width_bps_z",
    # SHAP top (momentum / direccional)
    "ema_9_dist_atr", "ema_21_dist_atr", "ema_50_dist_atr",
    "ret_3_atr", "ret_10_atr", "ret_60_atr",
    "macd_hist_atr_log", "macd_hist", "macd_hist_5m_atr",
    "rsi", "rsi_norm", "rsi_5m_norm",
    # SHAP top (regime indicators)
    "adx_smooth_norm", "adx_1h_norm", "adx_5m_norm",
    "dm_diff_1h_norm", "trend_dir",
    "position_range_240", "dist_high_60",
    "chop_score", "is_exhaustion",
    # Time features (contexto, no decisión)
    "minute_of_day_sin", "minute_of_day_cos",
    "hour_sin", "hour_cos",
    "dow_sin", "dow_cos",
    "is_asia", "is_london", "is_ny", "is_overlap",
    # VWAP / structure
    "vwap_dist_atr", "vwap_band_pos", "bb_position",
]


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
        # Inference logging (jsonl por bar / decisión)
        log_dir: Optional[str] = None,
        enable_inference_log: bool = True,
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

        # Inference logger (después de load_artifacts para conocer release/trial)
        self.enable_inference_log = bool(enable_inference_log)
        if self.enable_inference_log:
            _log_dir = Path(log_dir) if log_dir else (self.production_dir / "logs")
            _log_dir.mkdir(parents=True, exist_ok=True)
            self._inference_logger = self._setup_inference_logger(_log_dir)
            print(f"[GBM-SIM] 📋 Inference log: {_log_dir}/gbm_inference_YYYYMMDD.jsonl")
        else:
            self._inference_logger = None

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
        """Decide LONG order o None. Si None, popula self._last_diag.
        Loguea cada decisión (signal o no) en gbm_inference_YYYYMMDD.jsonl."""
        self._last_diag = None
        cur = int(current_positions or 0)
        mx  = int(max_positions or 2)

        # 0. Position limit (chequeo barato primero) — antes de prepare_data
        if cur >= mx:
            diag = {"no_signal_reason": "POSITION_LIMIT",
                    "current_positions": cur, "max_positions": mx}
            self._reject(diag)
            self._log_inference(bar_row=None, p_raw=None, p_cal=None,
                                decision="NO_SIGNAL", reason="POSITION_LIMIT",
                                order=None, equity=equity, cur=cur, mx=mx,
                                extra=diag)
            return None

        # 1. prepare_data
        try:
            df_prep = self.pipeline.prepare_data(
                df_rates, labels=False, side="both",
                set_market_condition=False, ensure_regime=True,
            )
        except Exception as e:
            diag = {"no_signal_reason": "PREPARE_DATA_FAILED",
                    "error": str(e)[:200]}
            self._reject(diag)
            self._log_inference(bar_row=None, p_raw=None, p_cal=None,
                                decision="NO_SIGNAL", reason="PREPARE_DATA_FAILED",
                                order=None, equity=equity, cur=cur, mx=mx,
                                extra=diag)
            return None

        # 2. Última fila con features válidas
        miss = [c for c in self.feat_cols if c not in df_prep.columns]
        if miss:
            diag = {"no_signal_reason": "FEATURES_MISSING",
                    "missing": miss[:5], "n_missing": len(miss)}
            self._reject(diag)
            self._log_inference(bar_row=None, p_raw=None, p_cal=None,
                                decision="NO_SIGNAL", reason="FEATURES_MISSING",
                                order=None, equity=equity, cur=cur, mx=mx,
                                extra=diag)
            return None
        valid = df_prep[self.feat_cols].notna().all(axis=1)
        if not valid.any():
            diag = {"no_signal_reason": "NO_VALID_ROWS"}
            self._reject(diag)
            self._log_inference(bar_row=None, p_raw=None, p_cal=None,
                                decision="NO_SIGNAL", reason="NO_VALID_ROWS",
                                order=None, equity=equity, cur=cur, mx=mx,
                                extra=diag)
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
            self._reject({"no_signal_reason": "BELOW_THRESHOLD", **base_diag})
            self._log_inference(bar_row=last_row, p_raw=p_raw, p_cal=p_cal,
                                decision="NO_SIGNAL", reason="BELOW_THRESHOLD",
                                order=None, equity=equity, cur=cur, mx=mx)
            return None

        # 6. ATR check (necesario para sizing)
        atr = float(last_row.get("atr", 0) or 0)
        if atr <= 0:
            self._reject({"no_signal_reason": "ATR_INVALID", **base_diag})
            self._log_inference(bar_row=last_row, p_raw=p_raw, p_cal=p_cal,
                                decision="NO_SIGNAL", reason="ATR_INVALID",
                                order=None, equity=equity, cur=cur, mx=mx)
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
            self._reject({"no_signal_reason": "QTY_ZERO", **base_diag,
                          "equity": float(equity), "atr": atr, "sl_dist": sl_dist})
            self._log_inference(bar_row=last_row, p_raw=p_raw, p_cal=p_cal,
                                decision="NO_SIGNAL", reason="QTY_ZERO",
                                order=None, equity=equity, cur=cur, mx=mx,
                                extra={"sl_dist": sl_dist, "risk_cash": risk_cash})
            return None

        # 8. Build order
        close_price = float(last_row.get("close", 0) or 0)
        entry_time  = pd.to_datetime(last_row.get("time")) if "time" in last_row.index else pd.Timestamp.utcnow()

        order = {
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

        # Log signal emitted
        self._log_inference(
            bar_row=last_row, p_raw=p_raw, p_cal=p_cal,
            decision="LONG", reason=None,
            order=order, equity=equity, cur=cur, mx=mx,
            extra={"sl_dist": sl_dist, "tp_dist": tp_dist,
                   "risk_cash": risk_cash, "risk_pct": risk_pct},
        )
        return order

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

    # ─── Inference logging (JSONL) ──────────────────────────────────

    def _setup_inference_logger(self, log_dir: Path) -> logging.Logger:
        """Logger dedicado a inferencias GBM, rotación diaria, no propaga al root."""
        logger_name = f"gbm_inference.{self.metadata.get('release', 'unknown')}.{id(self)}"
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)
        if logger.handlers:
            return logger  # idempotente
        log_path = log_dir / f"gbm_inference_{pd.Timestamp.now().strftime('%Y%m%d')}.jsonl"
        fh = TimedRotatingFileHandler(
            filename=str(log_path), when="midnight", interval=1,
            backupCount=60, encoding="utf-8", utc=False,
        )
        # Mantener nombre fijo del archivo activo + sufijo de fecha en rotación
        def _namer(default_name: str) -> str: return default_name
        _log_dir_ref = log_dir
        def _rotator(source: str, dest: str) -> None:
            import os, shutil
            if os.path.exists(source):
                shutil.move(source, dest)
            new_date = pd.Timestamp.now().strftime("%Y%m%d")
            fh.baseFilename = str(_log_dir_ref / f"gbm_inference_{new_date}.jsonl")
        fh.namer = _namer
        fh.rotator = _rotator
        fh.suffix = "%Y%m%d"
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(fh)
        logger.propagate = False
        # Header opcional indicando que arranca
        logger.info(json.dumps({
            "event": "GBM_LOGGER_STARTED",
            "ts": datetime.utcnow().isoformat() + "Z",
            "log_file": str(log_path),
            "release": self.metadata.get("release"),
            "trial": self.metadata.get("trial"),
            "as_of": self.metadata.get("as_of"),
            "threshold": self.threshold,
            "n_features": len(self.feat_cols),
            "use_calibrator": self.calibrator is not None,
        }))
        return logger

    @staticmethod
    def _safe_float(v) -> Optional[float]:
        """Convierte a float o None si NaN/None/no-numeric."""
        if v is None: return None
        try:
            f = float(v)
            return f if np.isfinite(f) else None
        except Exception:
            return None

    def _extract_market_features(self, bar_row: Optional[pd.Series]) -> Dict[str, Any]:
        """Subset de features del bar_row con valores nulos seguros."""
        if bar_row is None: return {}
        out: Dict[str, Any] = {}
        for k in _MARKET_FEATURES_TO_LOG:
            if k in bar_row.index:
                out[k] = self._safe_float(bar_row[k])
            else:
                out[k] = None
        # Añadir time y state aparte (no son features pero son contexto)
        out["state"] = str(bar_row.get("state", "")) if "state" in bar_row.index else ""
        out["macro_regime"] = str(bar_row.get("macro_regime",
                                              bar_row.get("regime", ""))) \
            if "macro_regime" in bar_row.index or "regime" in bar_row.index else ""
        return out

    def _log_inference(
        self,
        *,
        bar_row: Optional[pd.Series],
        p_raw: Optional[float],
        p_cal: Optional[float],
        decision: str,                       # "LONG" o "NO_SIGNAL"
        reason: Optional[str],               # razón si NO_SIGNAL
        order: Optional[Dict[str, Any]],     # orden completa si LONG
        equity: float,
        cur: int,
        mx: int,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Escribe una línea JSONL con todos los datos relevantes de la decisión."""
        if not self.enable_inference_log or self._inference_logger is None:
            return

        # bar_time: del bar_row si está disponible, sino now()
        bar_time = None
        if bar_row is not None and "time" in bar_row.index:
            try:
                bar_time = pd.to_datetime(bar_row["time"]).isoformat()
            except Exception:
                bar_time = str(bar_row.get("time"))

        entry: Dict[str, Any] = {
            "ts":           datetime.utcnow().isoformat() + "Z",
            "bar_time":     bar_time,
            "decision":     decision,
            "reason":       reason,
            "release":      self.metadata.get("release"),
            "trial":        self.metadata.get("trial"),
            "as_of":        self.metadata.get("as_of"),
            "model": {
                "proba_long_raw":  self._safe_float(p_raw),
                "proba_long_cal":  self._safe_float(p_cal),
                "threshold":       self._safe_float(self.threshold),
                "crosses_thr":     (p_raw >= self.threshold) if p_raw is not None else None,
                "use_calibrator":  self.calibrator is not None,
                "n_features":      len(self.feat_cols),
            },
            "market":   self._extract_market_features(bar_row),
            "context": {
                "equity":            self._safe_float(equity),
                "current_positions": int(cur),
                "max_positions":     int(mx),
            },
        }

        if order is not None:
            entry["order"] = {
                "side":         order.get("side"),
                "symbol":       order.get("symbol"),
                "entry":        self._safe_float(order.get("entry")),
                "sl":           self._safe_float(order.get("sl")),
                "tp":           self._safe_float(order.get("tp")),
                "qty":          self._safe_float(order.get("qty")),
                "atr_at_entry": self._safe_float(order.get("atr_at_entry")),
                "tp_mult":      self._safe_float(order.get("tp_mult")),
                "sl_mult":      self._safe_float(order.get("sl_mult")),
            }
        else:
            entry["order"] = None

        if extra:
            entry["extra"] = {k: self._safe_float(v) if isinstance(v, (int, float, np.floating))
                                else v for k, v in extra.items()}

        try:
            self._inference_logger.info(json.dumps(entry, default=str))
        except Exception as e:
            # Nunca fallar la decisión por un error de logging
            print(f"[GBM-SIM] ⚠️  Error escribiendo inference log: {e}")
