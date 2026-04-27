# mimo_old/strategy_gates.py

from dataclasses import dataclass
from typing import Dict, Any, Optional

@dataclass
class GatedDecision:
    action: str          # "buy" | "sell" | "none"
    size_mult: float
    reason: str
    meta: Optional[Dict[str, Any]] = None


class StrategyGate:
    def __init__(self,
                 chop_block: bool = True,
                 chop_size_mult: float = 0.0,
                 exhaustion_blocks_reentry: bool = True):
        self.chop_block = chop_block
        self.chop_size_mult = chop_size_mult
        self.exhaustion_blocks_reentry = exhaustion_blocks_reentry

    @staticmethod
    def _norm_action(a: str) -> str:
        a = (a or "").strip().lower()
        if a in ("long", "buy"):
            return "buy"
        if a in ("short", "sell"):
            return "sell"
        return "none"

    @staticmethod
    def _norm_pos_side(pos_side: Optional[str]) -> Optional[str]:
        if pos_side is None:
            return None
        s = str(pos_side).strip().lower()
        if s in ("long", "buy"):
            return "buy"
        if s in ("short", "sell"):
            return "sell"
        return None

    def apply(self,
              base_action: str,
              row: Dict[str, Any],
              has_position: bool,
              pos_side: Optional[str]) -> GatedDecision:

        base_action = self._norm_action(base_action)
        pos_side = self._norm_pos_side(pos_side)

        chop = bool(row.get("is_chop", False))
        exhaustion = bool(row.get("is_exhaustion", False))
        trend_dir = row.get("trend_dir", None)

        # 1) Bloqueo en chop para nuevas entradas
        if self.chop_block and chop and base_action in ("buy", "sell") and not has_position:
            return GatedDecision("none", 0.0, "BLOCK_CHOP")

        # 2) Bloqueo de reentrada en agotamiento (en la misma dirección del trend_dir)
        if self.exhaustion_blocks_reentry and exhaustion and base_action in ("buy", "sell") and not has_position:
            if (base_action == "buy" and trend_dir == +1) or (base_action == "sell" and trend_dir == -1):
                return GatedDecision("none", 0.0, "BLOCK_EXHAUSTION_REENTRY")

        # 3) Permitir “micro size” en chop si lo configuras (chop_size_mult > 0)
        if chop and base_action in ("buy", "sell") and not has_position and self.chop_size_mult > 0:
            return GatedDecision(base_action, float(self.chop_size_mult), "ALLOW_MIN_IN_CHOP")

        return GatedDecision(base_action, 1.0, "OK")
