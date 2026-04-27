from dataclasses import dataclass, field
from typing import Dict, Any, Set
import threading

@dataclass
class RuntimeState:
    keepalive_last_sent: Dict[int, float] = field(default_factory=dict)
    keepalive_last_bar_time: Dict[int, int] = field(default_factory=dict)
    keepalive_last_indicators: Dict[int, dict] = field(default_factory=dict)
    s1_open_tickets: Set[int] = field(default_factory=set)
    startup_bars_seen: Set[int] = field(default_factory=set)
    last_diag: dict | None = None

_CLOSE_EVENTS = {
    "VIRTUAL_SL_TRIGGERED",
    "VIRTUAL_TP_TRIGGERED",
    "TIME_FORCE_CLOSE_TRIGGERED",
    "ADAPTIVE_TP_CLOSE",
    "ADAPTIVE_SL_CLOSE",
    "ADAPTIVE_VIRTUAL_SL_TRIGGERED",
    "ADAPTIVE_EXTENSION_SL_TRIGGERED",
    "ADAPTIVE_EXPANSION_SL_TRIGGERED",
    "ADAPTIVE_HARD_SL_TRIGGERED",
    "POSITION_CLOSED_EXTERNAL",
    "POSITION_CLOSED",
    "MAX_HOLD_TIME_TRIGGERED",
}

# Eventos que implican que la posición sigue viva aunque S2 se haya perdido
# el POSITION_OPENED inicial por race/handshake del SUB.
_OPEN_LIKE_EVENTS = {
    "POSITION_OPENED",
    "POSITION_MODIFIED",
    "BE_ARMED",
    "BE_SKIPPED",
    "TIME_PROFIT_FLOOR_ARMED",
    "TRAILING_DELAY_ACTIVE",
    "TRAILING_DELAY_RELEASED",
    "TRAILING_UPDATED",
    "RUNNER_VTP_DISABLED",
    "PARTIAL_CLOSE_DONE",
    "PARTIAL_CLOSE_SKIPPED",
    "TP_CONFIRMED",
    "SPREAD_INHIBIT_ACTIVE",
    "SPREAD_INHIBIT_TIMEOUT",
    "TIME_FORCE_CLOSE_EXTENDED",
    "ADAPTIVE_SL_CREATED",
    "ADAPTIVE_TP_CREATED",
    "ADAPTIVE_TP_EXTENSION_OVERRIDDEN",
    "ADAPTIVE_INDICATORS_STALE",
}

class S3State:
    def __init__(self):
        self._lock = threading.RLock()
        self.positions: Dict[int, Any] = {}
        self.last_sl_close_by_side = {"long": [], "short": []}
        self._on_position_opened_cb = None
        self._on_position_closed_cb = None

    def set_on_position_opened(self, callback):
        self._on_position_opened_cb = callback

    def set_on_position_closed(self, callback):
        self._on_position_closed_cb = callback

    def get_positions(self):
        with self._lock:
            return dict(self.positions)

    @staticmethod
    def _extract_ticket(msg: dict) -> int:
        try:
            return int(float(msg.get("ticket", -1)))
        except Exception:
            return -1

    def _merge_position_event(self, ticket: int, msg: dict) -> bool:
        """Mergea un evento de S3 en positions.

        Devuelve True si el ticket se ha creado ahora (apertura sintética o real),
        False si solo ha sido una actualización.
        """
        existing = self.positions.get(ticket)
        if existing is None:
            base = {
                "ticket": ticket,
                "symbol": msg.get("symbol"),
                "side": msg.get("side"),
                "volume": msg.get("volume"),
                "entry_price": msg.get("entry_price") or msg.get("open_price"),
                "virtual_tp": msg.get("virtual_tp"),
                "virtual_sl": msg.get("virtual_sl") or msg.get("virtual_sl_price"),
                "state": msg.get("state") or msg.get("regime"),
                "score": msg.get("score"),
                "last_event": msg.get("event"),
                "synthetic_open": msg.get("event") != "POSITION_OPENED",
            }
            self.positions[ticket] = base
            return True

        # update in place with any fresh non-null values
        updates = {
            "symbol": msg.get("symbol"),
            "side": msg.get("side"),
            "volume": msg.get("volume"),
            "entry_price": msg.get("entry_price") or msg.get("open_price"),
            "virtual_tp": msg.get("virtual_tp"),
            "virtual_sl": msg.get("virtual_sl") or msg.get("virtual_sl_price"),
            "state": msg.get("state") or msg.get("regime"),
            "score": msg.get("score"),
        }
        for k, v in updates.items():
            if v is not None:
                existing[k] = v
        existing["last_event"] = msg.get("event")
        return False

    def on_event(self, msg: dict):
        event = msg.get("event")
        ticket = self._extract_ticket(msg)
        opened_cb = None
        closed_cb = None
        opened_ticket = -1
        closed_ticket = -1

        with self._lock:
            if ticket >= 0 and event in _OPEN_LIKE_EVENTS:
                created = self._merge_position_event(ticket, msg)
                if created and self._on_position_opened_cb:
                    opened_cb = self._on_position_opened_cb
                    opened_ticket = ticket

            if ticket >= 0 and event in _CLOSE_EVENTS:
                _closed = self.positions.pop(ticket, None)
                if self._on_position_closed_cb:
                    closed_cb = self._on_position_closed_cb
                    closed_ticket = ticket
                if _closed and _closed.get("side"):
                    side = str(_closed.get("side", "")).lower()
                    if side in ("buy", "long"):
                        self.last_sl_close_by_side["long"].append(ticket)
                    elif side in ("sell", "short"):
                        self.last_sl_close_by_side["short"].append(ticket)

        if opened_cb and opened_ticket >= 0:
            try:
                opened_cb(opened_ticket)
            except Exception:
                pass
        if closed_cb and closed_ticket >= 0:
            try:
                closed_cb(closed_ticket)
            except Exception:
                pass

    def n_positions(self) -> int:
        with self._lock:
            return len(self.positions)

    def total_volume(self) -> float:
        with self._lock:
            total = 0.0
            for p in self.positions.values():
                try:
                    total += float(p.get("volume", 0.0))
                except Exception:
                    pass
            return total

    def sides_open(self) -> list:
        with self._lock:
            out = []
            for p in self.positions.values():
                side = p.get("side")
                if side is not None:
                    out.append(side)
            return out
