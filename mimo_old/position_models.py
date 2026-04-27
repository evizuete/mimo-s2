from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional, Tuple


# =========================
# Enums
# =========================

class Side(Enum):
    LONG = auto()
    SHORT = auto()


class ExitMode(Enum):
    HARD = auto()          # SL duro por extremos de vela (o ticks si más adelante)
    CLOSE_CONFIRM = auto() # SL solo si hay cierre más allá del nivel
    STRUCTURAL = auto()    # Igual que CLOSE_CONFIRM pero pensado para SL estructural


# =========================
# Configuración de salida
# =========================

@dataclass
class ExitConfig:
    # Modo principal de salida
    exit_mode: ExitMode = ExitMode.CLOSE_CONFIRM

    # --- Stop por cierre / zona ---
    close_confirm_bars: int = 1           # nº de cierres consecutivos más allá del SL

    # --- Firewall duro ---
    firewall_atr_mult: float = 2.0        # pérdida máxima desde entrada (en ATR)

    # --- Reentrada post-sweep ---
    allow_reentry: bool = True
    reentry_max: int = 1                  # máx reentradas por trade
    reentry_size_mult: float = 0.50       # tamaño relativo de la reentrada
    reentry_reclaim_bars: int = 1         # cierres de reclaim requeridos
    reentry_cooldown_bars: int = 10       # ventana máxima tras sweep


# =========================
# Posición virtual
# =========================

@dataclass
class VirtualPosition:
    side: Side
    entry_price: float
    size: float

    sl_level: float
    tp_level: Optional[float] = None

    exit_cfg: ExitConfig = ExitConfig()

    # Estado temporal
    open_i: int = 0
    close_i: Optional[int] = None
    is_open: bool = True

    # --- Estado interno ---
    _confirm_count: int = 0

    # Sweep / reentrada
    _last_sweep_i: Optional[int] = None
    _sweep_level: Optional[float] = None
    _reentry_count: int = 0

    # =========================
    # Helpers internos
    # =========================

    def _is_stop_touched_intrabar(self, low_i: float, high_i: float) -> bool:
        if self.side == Side.LONG:
            return low_i <= self.sl_level
        else:
            return high_i >= self.sl_level

    def _is_close_beyond_sl(self, close_i: float) -> bool:
        if self.side == Side.LONG:
            return close_i < self.sl_level
        else:
            return close_i > self.sl_level

    def _firewall_level(self, atr_value: float) -> float:
        dist = self.exit_cfg.firewall_atr_mult * atr_value
        if self.side == Side.LONG:
            return self.entry_price - dist
        else:
            return self.entry_price + dist

    # =========================
    # Lógica de cierre
    # =========================

    def should_close_on_bar(
        self,
        i: int,
        o: float,
        h: float,
        l: float,
        c: float,
        atr_value: float
    ) -> Tuple[bool, Optional[str], Optional[float]]:
        """
        Decide si la posición debe cerrarse en la vela i.

        Devuelve:
          (close?, reason, exit_price)
        """

        if not self.is_open:
            return False, None, None

        # -------- Firewall duro (siempre activo) --------
        fw = self._firewall_level(atr_value)
        if self.side == Side.LONG:
            if l <= fw:
                return True, "FIREWALL", fw
        else:
            if h >= fw:
                return True, "FIREWALL", fw

        # -------- Take Profit (por extremos de vela) --------
        if self.tp_level is not None:
            if self.side == Side.LONG and h >= self.tp_level:
                return True, "TP", self.tp_level
            if self.side == Side.SHORT and l <= self.tp_level:
                return True, "TP", self.tp_level

        # -------- Stop según modo --------
        stop_touched = self._is_stop_touched_intrabar(l, h)

        if self.exit_cfg.exit_mode == ExitMode.HARD:
            if stop_touched:
                return True, "SL_HARD", self.sl_level

        else:
            # CLOSE_CONFIRM / STRUCTURAL

            # Marca sweep si hubo toque intrabar
            if stop_touched:
                self._last_sweep_i = i
                self._sweep_level = self.sl_level

            # Confirmación por cierre
            if self._is_close_beyond_sl(c):
                self._confirm_count += 1
                if self._confirm_count >= self.exit_cfg.close_confirm_bars:
                    return True, "SL_CLOSE_CONFIRM", c
            else:
                self._confirm_count = 0

        return False, None, None

    # =========================
    # Lógica de reentrada
    # =========================

    def should_reenter_on_bar(self, i: int, c: float) -> bool:
        """
        Reentrada tras sweep:
          - hubo sweep reciente
          - reclaim del nivel
          - dentro de ventana temporal
        """

        if not self.exit_cfg.allow_reentry:
            return False

        if self._reentry_count >= self.exit_cfg.reentry_max:
            return False

        if self._last_sweep_i is None or self._sweep_level is None:
            return False

        if (i - self._last_sweep_i) > self.exit_cfg.reentry_cooldown_bars:
            return False

        # reclaim
        if self.side == Side.LONG:
            return c > self._sweep_level
        else:
            return c < self._sweep_level

    def mark_closed(self, i: int):
        self.is_open = False
        self.close_i = i

    def create_reentry(
        self,
        i: int,
        entry_price: float,
        new_sl: float,
        tp: Optional[float]
    ) -> "VirtualPosition":

        self._reentry_count += 1

        new_size = self.size * self.exit_cfg.reentry_size_mult

        vp = VirtualPosition(
            side=self.side,
            entry_price=entry_price,
            size=new_size,
            sl_level=new_sl,
            tp_level=tp,
            exit_cfg=self.exit_cfg,
            open_i=i
        )

        vp._reentry_count = self._reentry_count
        return vp
