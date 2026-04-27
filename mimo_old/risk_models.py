from enum import Enum, auto
from typing import Sequence


class Side(Enum):
    LONG = auto()
    SHORT = auto()


def compute_structural_sl(
    side: Side,
    lows: Sequence[float],
    highs: Sequence[float],
    atr_value: float,
    i: int,
    lookback: int,
    buffer_atr_mult: float
) -> float:

    start = max(0, i - lookback)
    if side == Side.LONG:
        swing = min(lows[start:i])
        return swing - buffer_atr_mult * atr_value
    else:
        swing = max(highs[start:i])
        return swing + buffer_atr_mult * atr_value
