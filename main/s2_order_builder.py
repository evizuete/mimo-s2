from typing import Any, Dict, Optional

from s2_config import (
    MIN_RR_BY_REGIME, MAX_RR_BY_REGIME,
    MIN_RR_BY_REGIME_SHORT, MAX_RR_BY_REGIME_SHORT,
)


class OrderBuilder:
    """
    Builder de geometría de órdenes para S2.

    Objetivos:
      - resolver ATR con fallback robusto
      - reanclar la geometría al precio efectivo de fill (bid/ask)
      - calcular virtual_sl y virtual_tp desde entry_ref
      - aplicar MIN_RR y MAX_RR por régimen (diferenciado por side LONG/SHORT)
      - opcionalmente ajustar TP con Bollinger en range/low_vol
      - construir metadata auditable
    """

    BOLLINGER_TP_REGIMES = {"range", "low_vol"}
    BOLLINGER_TP_MIN_RR = 1.0

    def __init__(self, config):
        self.config = config

    @staticmethod
    def _safe_float(x, default: float = 0.0) -> float:
        try:
            return float(x)
        except Exception:
            return float(default)

    @staticmethod
    def _norm_state(state: Any) -> str:
        return str(state or "").strip().lower()

    @staticmethod
    def _price_to_points(a: float, b: float, point: float = 0.01) -> int:
        try:
            return int(round(abs(float(a) - float(b)) / float(point)))
        except Exception:
            return 0

    def resolve_atr(self, order: dict, point: float) -> tuple[float, str, float]:
        atr_min_valid = self.config.min_vsl_points * point

        atr_signal = self._safe_float(order.get("atr_at_entry"))
        atr_order = self._safe_float(order.get("atr"))
        atr_pipe = self._safe_float(order.get("_atr_fallback"))

        atr = atr_signal
        source = "atr_at_entry"

        if atr <= atr_min_valid:
            atr = atr_order
            source = "order.atr"
        if atr <= atr_min_valid:
            atr = atr_pipe
            source = "_atr_fallback"
        if atr <= atr_min_valid:
            atr = self.config.min_vsl_points * 4 * point
            source = "synthetic_min_safe"

        return atr, source, atr_min_valid

    def _apply_rr_caps(
        self, *, state: str, rr_model: float, side: str = "long"
    ) -> tuple[float, float, float, bool, bool]:
        """Aplica caps de RR mínimo y máximo según régimen y side.

        Args:
            state:    régimen normalizado (lowercase).
            rr_model: RR propuesto por el modelo.
            side:     'long' o 'short' — determina qué tablas de RR usar.
        """
        if side == "short":
            rr_min = float(MIN_RR_BY_REGIME_SHORT.get(state, MIN_RR_BY_REGIME_SHORT["_default"]))
            rr_max = float(MAX_RR_BY_REGIME_SHORT.get(state, MAX_RR_BY_REGIME_SHORT["_default"]))
        else:
            rr_min = float(MIN_RR_BY_REGIME.get(state, MIN_RR_BY_REGIME["_default"]))
            rr_max = float(MAX_RR_BY_REGIME.get(state, MAX_RR_BY_REGIME["_default"]))

        rr_after = float(rr_model)
        min_applied = False
        max_applied = False

        if rr_after < rr_min:
            rr_after = rr_min
            min_applied = True

        if rr_max > 0 and rr_after > rr_max:
            rr_after = rr_max
            max_applied = True

        return rr_after, rr_min, rr_max, min_applied, max_applied

    def _maybe_adjust_tp_with_bollinger(
        self,
        *,
        side_s3: str,
        state: str,
        entry_ref: float,
        virtual_tp_price: float,
        risk_distance: float,
        rr_after_caps: float,
        bb_upper: Optional[float],
        bb_lower: Optional[float],
    ) -> tuple[float, bool, Optional[float], float]:
        bb_applied = False
        bb_value = None
        rr_final = rr_after_caps

        if state not in self.BOLLINGER_TP_REGIMES:
            return virtual_tp_price, bb_applied, bb_value, rr_final

        if risk_distance <= 0:
            return virtual_tp_price, bb_applied, bb_value, rr_final

        if side_s3 == "BUY":
            band = self._safe_float(bb_upper, 0.0)
            if band > entry_ref and band < virtual_tp_price:
                rr_band = abs(band - entry_ref) / risk_distance
                if rr_band >= self.BOLLINGER_TP_MIN_RR:
                    virtual_tp_price = band
                    bb_applied = True
                    bb_value = band
                    rr_final = rr_band
        else:
            band = self._safe_float(bb_lower, 0.0)
            if band < entry_ref and band > virtual_tp_price:
                rr_band = abs(entry_ref - band) / risk_distance
                if rr_band >= self.BOLLINGER_TP_MIN_RR:
                    virtual_tp_price = band
                    bb_applied = True
                    bb_value = band
                    rr_final = rr_band

        return virtual_tp_price, bb_applied, bb_value, rr_final

    def build(
        self,
        *,
        order: dict,
        side: str,
        entry: float,
        point: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        spread_points: Optional[int] = None,
        bb_upper: Optional[float] = None,
        bb_lower: Optional[float] = None,
    ) -> Dict[str, Any]:
        side_s3 = str(side).strip().upper()
        if side_s3 not in {"BUY", "SELL"}:
            raise ValueError(f"Unsupported side: {side}")

        # side normalizado para lógica interna ('long' / 'short')
        side_norm = "short" if side_s3 == "SELL" else "long"

        model_entry = self._safe_float(entry)
        bid_f = self._safe_float(bid, model_entry)
        ask_f = self._safe_float(ask, model_entry)

        # 1) Entry efectivo reanclado a bid/ask
        if side_s3 == "BUY":
            entry_ref = ask_f if ask is not None else model_entry
            fill_ref_src = "ask" if ask is not None else "model_entry"
        else:
            entry_ref = bid_f if bid is not None else model_entry
            fill_ref_src = "bid" if bid is not None else "model_entry"

        fill_ref_gap_pts = self._price_to_points(entry_ref, model_entry, point=point)

        # 2) Resolver ATR y vSL
        atr, atr_source, atr_min_valid = self.resolve_atr(order, point)

        raw_sl = self._safe_float(order.get("sl"))
        if raw_sl <= 0:
            raw_sl = model_entry - atr if side_s3 == "BUY" else model_entry + atr

        vsl_points = self._price_to_points(model_entry, raw_sl, point=point)
        if vsl_points < int(self.config.min_vsl_points):
            vsl_points = int(self.config.min_vsl_points)

        # 3) virtual_sl SIEMPRE desde entry_ref
        if side_s3 == "BUY":
            virtual_sl_price = entry_ref - vsl_points * point
        else:
            virtual_sl_price = entry_ref + vsl_points * point

        # 4) Riesgo real SIEMPRE desde entry_ref a virtual_sl
        risk_distance = abs(entry_ref - virtual_sl_price)

        # 5) RR del modelo y caps por régimen
        raw_tp = self._safe_float(order.get("tp"))
        tp_model_distance = abs(raw_tp - model_entry) if raw_tp > 0 else 0.0
        rr_model = (tp_model_distance / risk_distance) if risk_distance > 0 else 0.0
        if rr_model <= 0:
            # Fallback: usar rr_max del régimen según side
            _state_fb = self._norm_state(order.get("state"))
            if side_norm == "short":
                rr_model = MAX_RR_BY_REGIME_SHORT.get(_state_fb, MAX_RR_BY_REGIME_SHORT["_default"])
            else:
                rr_model = MAX_RR_BY_REGIME.get(_state_fb, MAX_RR_BY_REGIME["_default"])

        state = self._norm_state(order.get("state"))
        rr_after_caps, rr_min, rr_max, rr_min_applied, rr_max_applied = self._apply_rr_caps(
            state=state,
            rr_model=rr_model,
            side=side_norm,
        )

        # 6) virtual_tp SIEMPRE desde entry_ref, NUNCA desde virtual_sl
        if side_s3 == "BUY":
            virtual_tp_price = entry_ref + rr_after_caps * risk_distance
        else:
            virtual_tp_price = entry_ref - rr_after_caps * risk_distance

        # 7) Ajuste opcional por Bollinger
        virtual_tp_price, bb_tp_applied, bb_tp_band_value, rr_final = self._maybe_adjust_tp_with_bollinger(
            side_s3=side_s3,
            state=state,
            entry_ref=entry_ref,
            virtual_tp_price=virtual_tp_price,
            risk_distance=risk_distance,
            rr_after_caps=rr_after_caps,
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )

        tp_points = self._price_to_points(entry_ref, virtual_tp_price, point=point)

        # 8) Emergency SL: paracaídas más allá del vSL
        emergency_points = max(int(vsl_points * 1.5), 50)
        if side_s3 == "BUY":
            emergency_sl_price = virtual_sl_price - emergency_points * point
        else:
            emergency_sl_price = virtual_sl_price + emergency_points * point

        metadata = {
            "entry_ref_price": entry_ref,
            "fill_ref_price": entry_ref,
            "fill_ref_src": fill_ref_src,
            "model_entry": model_entry,
            "fill_ref_gap_pts": fill_ref_gap_pts,
            "raw_sl_from_model": raw_sl,
            "raw_tp_from_model": raw_tp,
            "virtual_sl_price": virtual_sl_price,
            "virtual_tp_price": virtual_tp_price,
            "virtual_sl_points": int(vsl_points),
            "tp_points": int(tp_points),
            "risk_distance": risk_distance,
            "rr_model": rr_model,
            "rr_target": rr_after_caps,
            "rr_final": rr_final,
            "rr_min_regime": rr_min,
            "rr_max_regime": rr_max,
            "tp_rr_min_applied": rr_min_applied,
            "tp_rr_capped": rr_max_applied,
            "bb_tp_applied": bb_tp_applied,
            "bb_tp_band_value": bb_tp_band_value,
            "atr_used": atr,
            "atr_source": atr_source,
            "atr_min_valid": atr_min_valid,
            "emergency_points": int(emergency_points),
            "emergency_sl_price": emergency_sl_price,
            "spread_points": spread_points,
            "state": state,
            "side_norm": side_norm,
        }

        return {
            "entry_ref_price": entry_ref,
            "virtual_sl_price": virtual_sl_price,
            "virtual_tp_price": virtual_tp_price,
            "virtual_sl_points": int(vsl_points),
            "tp_points": int(tp_points),
            "risk_distance": risk_distance,
            "rr_target": rr_after_caps,
            "rr_final": rr_final,
            "emergency_points": int(emergency_points),
            "emergency_sl_price": emergency_sl_price,
            "metadata": metadata,
        }