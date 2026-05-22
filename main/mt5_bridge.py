from typing import Optional, Dict, Any, Tuple
import MetaTrader5 as mt5

MT5_BRIDGE_VERSION = "30.2"


def norm_side(side: Optional[str]) -> str:
    s = (side or "").strip().upper()
    if s in ("BUY", "B"):
        return "BUY"
    if s in ("SELL", "S"):
        return "SELL"
    raise ValueError(f"Invalid side: {side}")


def side_to_order_type(side: str) -> int:
    side = norm_side(side)
    return mt5.ORDER_TYPE_BUY if side == "BUY" else mt5.ORDER_TYPE_SELL


def opposite_side(side: str) -> str:
    side = norm_side(side)
    return "SELL" if side == "BUY" else "BUY"


class MT5Bridge:
    def __init__(
        self,
        path: Optional[str] = None,
        login: Optional[str] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
    ) -> None:
        self.path = path
        self.login = login
        self.password = password
        self.server = server

    def initialize_or_raise(self) -> None:
        kwargs: Dict[str, Any] = {}
        if self.path:
            kwargs["path"] = self.path

        if not mt5.initialize(**kwargs):
            raise RuntimeError(f"mt5.initialize() failed: {mt5.last_error()}")

        if self.login and self.password and self.server:
            login_i = int(self.login)
            if not mt5.login(login=login_i, password=self.password, server=self.server):
                raise RuntimeError(f"mt5.login() failed: {mt5.last_error()}")

    def shutdown(self) -> None:
        try:
            mt5.shutdown()
        except Exception:
            pass

    def copy_rates_from_pos(
        self,
        symbol: str,
        start_pos: int,
        count: int,
        period: int = mt5.TIMEFRAME_M1,
    ):
        rates = mt5.copy_rates_from_pos(symbol, period, start_pos, count)
        if rates is None or len(rates) < 2:
            return None
        return rates

    @staticmethod
    def select(symbol: str, enable: bool = True):
        mt5.symbol_select(symbol, enable)

    def info(self):
        info = mt5.account_info()
        if info is None:
            raise RuntimeError(f"mt5.info() failed: {mt5.last_error()}")
        return info

    def ensure_symbol(self, symbol: str) -> None:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) is None: {mt5.last_error()}")
        if not info.visible:
            if not mt5.symbol_select(symbol, True):
                raise RuntimeError(f"symbol_select({symbol}, True) failed: {mt5.last_error()}")

    def point(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) is None: {mt5.last_error()}")
        return float(info.point)

    def volume_constraints(self, symbol: str) -> Tuple[float, float]:
        info = mt5.symbol_info(symbol)
        if info is None:
            raise RuntimeError(f"symbol_info({symbol}) is None: {mt5.last_error()}")
        return float(info.volume_min), float(info.volume_step)

    def quantize_volume(self, symbol: str, vol: float) -> float:
        vmin, step = self.volume_constraints(symbol)
        if step <= 0:
            return float(vol)
        q = round(vol / step) * step
        if 0 < q < vmin:
            q = vmin
        if abs(q) < 1e-12:
            q = 0.0
        return float(q)

    def tick_bid_ask(self, symbol: str) -> Tuple[bool, float, float]:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            raise RuntimeError(f"symbol_info_tick({symbol}) is None: {mt5.last_error()}")
        return True, float(tick.bid), float(tick.ask)

    def positions_get(self, ticket: Optional[int] = None, symbol: Optional[str] = None):
        if ticket is not None:
            return mt5.positions_get(ticket=int(ticket)) or []
        if symbol is not None:
            return mt5.positions_get(symbol=str(symbol)) or []
        return mt5.positions_get() or []

    @staticmethod
    def _result_to_dict(result, action_name: str) -> Dict[str, Any]:
        if result is None:
            return {
                "ok": False,
                "retcode": -1,
                "error": f"order_send({action_name}) returned None: {mt5.last_error()}",
            }
        d = result._asdict()
        d["ok"] = (result.retcode == mt5.TRADE_RETCODE_DONE)
        return d

    def send_market_order(
        self,
        symbol: str,
        side: str,
        volume: float,
        magic: int,
        comment: str,
        deviation: int,
        sl: float = 0.0,
        tp: float = 0.0,
    ) -> Dict[str, Any]:
        side = norm_side(side)
        self.ensure_symbol(symbol)
        ok, bid, ask = self.tick_bid_ask(symbol)
        price = ask if side == "BUY" else bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": side_to_order_type(side),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": (comment or ""),
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        if sl and sl > 0:
            request["sl"] = float(sl)
        if tp and tp > 0:
            request["tp"] = float(tp)

        result = mt5.order_send(request)
        d = self._result_to_dict(result, "open")
        d["bid_at_send"] = bid
        d["ask_at_send"] = ask
        d["price_sent"] = price
        d["request_side"] = side
        d["request_volume"] = float(volume)
        return d

    def modify_sl(self, ticket: int, symbol: str, sl: float) -> Dict[str, Any]:
        """
        Modifica el SL de una posición existente sin tocar TP ni volumen.
        """
        self.ensure_symbol(symbol)
        positions = mt5.positions_get(ticket=int(ticket))
        current_tp = float(positions[0].tp) if positions else 0.0

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": symbol,
            "position": int(ticket),
            "sl": round(float(sl), 5),
            "tp": current_tp,
        }
        result = mt5.order_send(request)
        d = self._result_to_dict(result, "modify_sl")
        d["requested_sl"] = round(float(sl), 5)
        d["current_tp"] = current_tp
        return d

    def close_position_market(
        self,
        ticket: int,
        symbol: str,
        side: str,
        volume: float,
        magic: int,
        deviation: int,
        reason: str,
    ) -> Dict[str, Any]:
        side = norm_side(side)
        self.ensure_symbol(symbol)
        ok, bid, ask = self.tick_bid_ask(symbol)
        close_side = opposite_side(side)
        price = ask if close_side == "BUY" else bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": int(ticket),
            "volume": float(volume),
            "type": side_to_order_type(close_side),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": "",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)
        d = self._result_to_dict(result, "close")
        d["bid_at_send"] = bid
        d["ask_at_send"] = ask
        d["price_sent"] = price
        d["request_side"] = close_side
        d["request_volume"] = float(volume)
        d["close_reason"] = reason
        return d

    def close_position_partial(
        self,
        ticket: int,
        symbol: str,
        side: str,
        volume_to_close: float,
        magic: int,
        deviation: int,
        reason: str,
    ) -> Dict[str, Any]:
        side = norm_side(side)
        self.ensure_symbol(symbol)
        ok, bid, ask = self.tick_bid_ask(symbol)
        close_side = opposite_side(side)
        price = ask if close_side == "BUY" else bid

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "position": int(ticket),
            "volume": float(volume_to_close),
            "type": side_to_order_type(close_side),
            "price": float(price),
            "deviation": int(deviation),
            "magic": int(magic),
            "comment": "",
            "type_filling": mt5.ORDER_FILLING_IOC,
            "type_time": mt5.ORDER_TIME_GTC,
        }
        result = mt5.order_send(request)
        d = self._result_to_dict(result, "close_partial")
        d["bid_at_send"] = bid
        d["ask_at_send"] = ask
        d["price_sent"] = price
        d["request_side"] = close_side
        d["request_volume"] = float(volume_to_close)
        d["close_reason"] = reason
        return d
