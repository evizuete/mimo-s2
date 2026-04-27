from dataclasses import dataclass
from typing import Dict, Optional

@dataclass(frozen=True)
class InstrumentSpec:
    symbol: str
    contract_size: float
    quote_ccy: str
    lot_min: float = 0.01
    lot_step: float = 0.01
    lot_max: float = 100.0
    max_positions: int = 20
    leverage: float = 500.0
    margin_rate: Optional[float] = None  # alternativa a leverage

INSTRUMENT_SPECS: Dict[str, InstrumentSpec] = {
    "XAUUSD": InstrumentSpec(
        symbol="XAUUSD",
        contract_size=100.0,
        quote_ccy="USD",
        lot_min=0.01,
        lot_step=0.01,
        max_positions=10,
        leverage=500.0
    ),
    "XAUUSD.r": InstrumentSpec(
        symbol="XAUUSD.r",
        contract_size=100.0,
        quote_ccy="USD",
        lot_min=0.01,
        lot_step=0.01,
        max_positions=10,
        leverage=100.0
    ),
    "EURUSD": InstrumentSpec(
        symbol="EURUSD",
        contract_size=100000.0,
        quote_ccy="USD",
        leverage=100.0
    ),
}