import concurrent.futures
import json
import math
import time
import threading
from collections import deque
from typing import Any, Dict, Optional, Set

import pandas as pd
import zmq

from s2_state import RuntimeState, S3State
from s2_listener import S3EventListener
from s2_order_builder import OrderBuilder


class IncrementalSMA:
    def __init__(self, period: int):
        self.period = int(period)
        self.buf = deque(maxlen=self.period)
        self._sum = 0.0

    def reset(self):
        self.buf.clear()
        self._sum = 0.0

    def push(self, value: Optional[float]):
        if value is None:
            return
        try:
            v = float(value)
        except Exception:
            return
        if not math.isfinite(v):
            return
        if len(self.buf) == self.period:
            self._sum -= float(self.buf[0])
        self.buf.append(v)
        self._sum += v

    @property
    def value(self) -> Optional[float]:
        if len(self.buf) < self.period:
            return None
        return float(self._sum / len(self.buf))


class S2Service:
    _MT5_SUB_ADDR = "tcp://10.1.21.25:5555"
    _MT5_SUB_TOPIC = b"XAUUSD.r"
    _ZMQ_RECONNECT_AFTER_TIMEOUTS = 40
    _NO_TICK_WARN_SECS = 90

    # ── Modelo entrenado a 5-min (release 202500 y posteriores) ────────────
    # MT5/S1 publica ticks 1m. Resamplear 1m→BASE_TF antes de pasar al
    # simulador, y solo invocar decide_live() en cierre de barra base_tf.
    # Si el modelo es 1-min (releases legacy 200xxx), poner BASE_TF_MINUTES=1
    # y el resample es no-op.
    BASE_TF_MINUTES = 5

    # 1m bars desde BD (~5.5 días con 8000): suficiente para que multi-TF
    # features (15m/1h) tengan ≥60 bars con warmup ADX_14/EMA50.
    LAST_N_RATES = 8000

    VOLUME_MA_PERIOD = 20
    OPEN_BE_OFFSET_POINTS = 15
    RUNNER_TIGHT_TRAIL_PTS = 20

    def __init__(self, config, simulator, db, sockets, loggers):
        self.config = config
        self.simulator = simulator
        self.db = db

        self.ticks_sub = sockets["ticks_sub"]
        self.orders_push = sockets["orders_push"]
        self.events_sub = sockets["events_sub"]

        self.signal_logger = loggers["signals"]
        self.system_logger = loggers["system"]

        self.runtime = RuntimeState()
        self.s3_state = S3State()
        self.order_builder = OrderBuilder(config)

        self._context = zmq.Context.instance()
        self._last_tick_ts = time.time()
        self._no_tick_count = 0
        self._reconnect_tries = 0
        self._open_guard = {"bar_time": 0, "sent_ts": 0.0}
        self._keepalive_last_sent: Dict[int, float] = {}
        self._keepalive_last_bar_time: Dict[int, int] = {}
        self._keepalive_last_indicators: Dict[int, Dict[str, Any]] = {}

        # Tracker para idempotencia del cierre de barra base_tf.
        # Solo invocamos decide_live una vez por barra (no por cada tick 1m
        # que llega dentro de la barra). Almacena pd.Timestamp del último
        # bar base_tf procesado.
        self._last_processed_basetf_bar_ts = None
        self._keepalive_last_context: Dict[int, Dict[str, Any]] = {}
        self._startup_bars_seen: Set[int] = set()
        self._volume_sma = IncrementalSMA(period=self.VOLUME_MA_PERIOD)
        self._db_executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="db-save",
        )

        self.s3_state.set_on_position_opened(self._on_position_opened)
        self._s3_open_tickets_local: Set[int] = set()

        # FIX: cooldown post-cierre para evitar ráfagas de re-entrada.
        # Cuando S3 confirma un cierre, se registra el timestamp y se bloquean
        # nuevas órdenes OPEN durante POST_CLOSE_COOLDOWN_SECS (3 barras M1).
        # Motivación: log 13/04/2026 muestra bursts de 5-7 posiciones en 8
        # minutos, todas con EV=-0.25R, amplificando pérdidas innecesariamente.
        self.POST_CLOSE_COOLDOWN_SECS: float = 180.0   # 3 barras M1
        self._last_close_ts: float = 0.0

        # FIX 17/04/2026 BUG-1: cooldown entre señales OPEN consecutivas.
        # El OPEN_GUARD (check por bar_time) solo bloquea duplicados dentro de
        # la MISMA barra. Cuando el modelo emite señal en dos barras seguidas
        # (gap=60s), ambas pasan el guard y se abren dos posiciones en el mismo
        # contexto de mercado, duplicando la exposición sin nueva información.
        # Observado en logs: pares (07:29/07:30), (08:45/08:46), (10:14/10:15),
        # (10:26/10:27) — todos con gap=60s. Impacto: -521pts evitables.
        # SIGNAL_INTER_COOLDOWN_SECS: mínimo entre dos envíos OPEN. Se elige
        # 120s (2 barras M1) para que el primer trade tenga al menos una barra
        # de vida antes de poder añadir exposición en la misma dirección.
        self.SIGNAL_INTER_COOLDOWN_SECS: float = 120.0  # 2 barras M1
        self._last_open_sent_ts: float = 0.0
        self._last_close_ticket: int = -1
        self._closed_ticket_tombstones: Dict[int, float] = {}
        self.TICKET_TOMBSTONE_SECS: float = 900.0
        self.s3_state.set_on_position_closed(self._on_position_closed)

    def start_listener(self):
        listener = S3EventListener(
            sub_socket=self.events_sub,
            s3_state=self.s3_state,
            logger=self.system_logger,
        )
        t = threading.Thread(target=listener.run_forever, daemon=True)
        t.start()

    def _on_position_opened(self, ticket: int):
        try:
            ticket = int(ticket)
        except Exception:
            return
        self._closed_ticket_tombstones.pop(ticket, None)
        self._s3_open_tickets_local.add(ticket)
        self._keepalive_last_sent[ticket] = 0.0

    def _on_position_closed(self, ticket: int):
        """Callback invocado por S3State cuando S3 confirma el cierre de una posicion.

        Registra el timestamp de cierre para el cooldown post-cierre.
        Llamado desde el hilo del S3EventListener (no desde el hilo principal),
        por lo que solo escribe atributos simples (float/int) — operacion atomica
        en CPython por el GIL, sin necesidad de lock adicional.
        """
        try:
            t = int(ticket)
            self._last_close_ts = time.time()
            self._last_close_ticket = t
            self._closed_ticket_tombstones[t] = self._last_close_ts
            self._s3_open_tickets_local.discard(t)
            self._drop_ticket_state(t)
        except Exception:
            pass

    def _drop_ticket_state(self, ticket: int):
        t = int(ticket)
        self._keepalive_last_sent.pop(t, None)
        self._keepalive_last_bar_time.pop(t, None)
        self._keepalive_last_indicators.pop(t, None)
        self._keepalive_last_context.pop(t, None)

    def _prune_ticket_tombstones(self):
        now_ts = time.time()
        self._closed_ticket_tombstones = {
            int(t): ts
            for t, ts in self._closed_ticket_tombstones.items()
            if (now_ts - float(ts)) <= self.TICKET_TOMBSTONE_SECS
        }
        if self._closed_ticket_tombstones:
            for t in list(self._closed_ticket_tombstones.keys()):
                self._s3_open_tickets_local.discard(int(t))

    @staticmethod
    def _safe_float(x, default=0.0) -> float:
        try:
            v = float(x)
            if math.isfinite(v):
                return v
        except Exception:
            pass
        return float(default)

    @staticmethod
    def _first_not_none(*vals):
        for v in vals:
            if v is not None:
                return v
        return None

    @staticmethod
    def _side_to_s3(side: str) -> str:
        s = str(side).strip().lower()
        return "BUY" if s in ("buy", "long") else "SELL"

    @staticmethod
    def _price_to_points(a: float, b: float, point: float = 0.01) -> int:
        try:
            return int(round(abs(float(a) - float(b)) / float(point)))
        except Exception:
            return 0

    @staticmethod
    def _norm_state(state: Any) -> str:
        return str(state or "").strip().lower()

    def _db_save_async(self, df_to_save: pd.DataFrame):
        self.db.save(df_to_save, table_name="rates")

    def _reconnect_ticks_sub(self):
        try:
            self.ticks_sub.setsockopt(zmq.LINGER, 0)
            self.ticks_sub.close()
        except Exception:
            pass

        s = self._context.socket(zmq.SUB)
        s.setsockopt(zmq.SUBSCRIBE, self._MT5_SUB_TOPIC)
        s.setsockopt(zmq.RCVHWM, 10000)
        s.setsockopt(zmq.RCVTIMEO, 30_000)
        s.connect(self._MT5_SUB_ADDR)
        self.ticks_sub = s

    def _log_signal(
        self,
        *,
        event: str,
        bar_data: Optional[dict] = None,
        live_order: Optional[dict] = None,
        request: Optional[dict] = None,
        n_positions: Optional[int] = None,
        equity: Optional[float] = None,
        balance: Optional[float] = None,
        indicators: Optional[dict] = None,
        model_diag: Optional[dict] = None,
        block_reason: Optional[str] = None,
    ):
        record: Dict[str, Any] = {"event": event, "ts": time.time()}

        if bar_data:
            record.update({
                "time": bar_data.get("time"),
                "open": bar_data.get("open"),
                "high": bar_data.get("high"),
                "low": bar_data.get("low"),
                "close": bar_data.get("close"),
                "spread": bar_data.get("spread"),
            })

        if live_order:
            record.update({
                "side": live_order.get("side"),
                "entry": live_order.get("entry"),
                "sl": live_order.get("sl"),
                "tp": live_order.get("tp"),
                "qty": live_order.get("qty"),
                "state": live_order.get("state"),
                "score": live_order.get("score"),
                "proba_long": live_order.get("proba_long"),
                "proba_short": live_order.get("proba_short"),
            })

        if request:
            record.update({
                "req_action": request.get("action"),
                "req_side": request.get("side"),
                "req_symbol": request.get("symbol"),
                "req_volume": request.get("volume"),
                "req_virtual_sl": request.get("virtual_sl"),
                "req_virtual_tp": request.get("virtual_tp"),
            })
            _meta = request.get("metadata", {})
            if isinstance(_meta, dict):
                record["metadata"] = _meta

        if indicators:
            record["indicators"] = indicators
        if model_diag:
            record["model_diag"] = model_diag
        if n_positions is not None:
            record["n_positions"] = n_positions
        if equity is not None:
            record["equity"] = equity
        if balance is not None:
            record["balance"] = balance
        if block_reason is not None:
            record["block_reason"] = block_reason

        self.signal_logger.info(json.dumps(record, ensure_ascii=False))

    def _send_order(self, request: dict):
        self.orders_push.send_string(json.dumps(request, ensure_ascii=False))

    def _ensure_db_tick_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        tick_cols = ["bid", "ask", "tick_ok"]
        has_tick_cols = getattr(self.db, "_rates_has_tick_cols", None)

        if has_tick_cols is None:
            try:
                import sqlalchemy
                with self.db.engine.connect():
                    insp = sqlalchemy.inspect(self.db.engine)
                    cols = {c["name"] for c in insp.get_columns("rates")}
                    has_tick_cols = all(c in cols for c in tick_cols)
            except Exception:
                has_tick_cols = False

            self.db._rates_has_tick_cols = has_tick_cols
            print(f"[DB] rates tick columns (bid/ask/tick_ok) present: {has_tick_cols}")

        if not has_tick_cols:
            return df.drop(columns=tick_cols, errors="ignore")
        return df

    def _prepare_bar_df(self, data: dict) -> pd.DataFrame:
        row = dict(data)
        row.pop("symbol", None)
        row.pop("balance", None)
        row.pop("equity", None)
        row.pop("free_margin", None)
        row.pop("n_positions", None)
        row.pop("open_tickets", None)
        row.pop("topic", None)

        df = pd.DataFrame.from_dict([row])
        df["time"] = pd.to_datetime(df["time"] - 2 * 3600, unit="s", utc=True)
        df = df.drop(columns=["real_volume"], errors="ignore")
        df = df.rename(columns={"tick_volume": "volume", "ticks_volume": "volume"})
        df = self._ensure_db_tick_cols(df)
        return df

    def _extract_pipeline_snapshot(self, df_rates: pd.DataFrame):
        df_prepared = getattr(self.simulator, "_last_df_prepared", None)
        if df_prepared is None and hasattr(self.simulator, "pipeline"):
            try:
                df_prepared = self.simulator.pipeline.prepare_data(df_rates)
            except Exception as e:
                print(f"[WARN][S2] pipeline.prepare_data fallback failed: {e}")
                df_prepared = None

        pip_last = df_prepared.iloc[-1] if df_prepared is not None and len(df_prepared) >= 1 else None
        pip_prev = df_prepared.iloc[-2] if df_prepared is not None and len(df_prepared) >= 2 else None

        def _pip(row, col):
            if row is None:
                return None
            try:
                v = row.get(col) if hasattr(row, "get") else (row[col] if col in row.index else None)
            except Exception:
                return None
            if v is None:
                return None
            try:
                f = float(v)
            except Exception:
                return None
            return None if math.isnan(f) else f

        bb_upper = _pip(pip_last, "bb_upper")
        bb_lower = _pip(pip_last, "bb_lower")

        return df_prepared, pip_last, pip_prev, _pip, bb_upper, bb_lower

    def _build_current_indicators(
        self,
        data: dict,
        live_order: Optional[dict],
        pip_last,
        pip_prev,
        pip_get,
        model_diag: Optional[dict],
    ) -> Dict[str, Any]:
        indicators: Dict[str, Any] = {}

        def put(k: str, *vals):
            v = self._first_not_none(*vals)
            if v is not None:
                indicators[k] = v

        put(
            "rsi",
            pip_get(pip_last, "rsi") if pip_get else None,
            live_order.get("rsi") if live_order else None,
            model_diag.get("rsi") if isinstance(model_diag, dict) else None,
        )
        put(
            "macd_hist",
            pip_get(pip_last, "macd_hist") if pip_get else None,
            live_order.get("macd_hist") if live_order else None,
            model_diag.get("macd_hist") if isinstance(model_diag, dict) else None,
        )
        put(
            "macd_hist_prev",
            pip_get(pip_prev, "macd_hist") if pip_get else None,
            live_order.get("macd_hist_prev") if live_order else None,
            model_diag.get("macd_hist_prev") if isinstance(model_diag, dict) else None,
        )
        put(
            "atr",
            pip_get(pip_last, "atr") if pip_get else None,
            live_order.get("atr_at_entry") if live_order else None,
            live_order.get("atr") if live_order else None,
            model_diag.get("atr") if isinstance(model_diag, dict) else None,
        )
        put(
            "proba_long",
            # FIX BUG-4 (17/04/2026): prioridad correcta para proba_long.
            # 1. model_diag.proba_long_cal — fresco cada barra (ahora también poblado
            #    en el path de éxito gracias al fix en trading_simulator_v2.py).
            # 2. model_diag.proba_long_cal_dec — alias desde dec.p_buy_cal.
            # 3. model_diag.proba_long_raw — raw del modelo, también fresco.
            # 4. pip_last['pred_long_cal'] — nombre real en _last_df_prepared.
            #    NOTA: el pipeline escribe 'pred_long_cal', NO 'proba_long'.
            #    pip_get(pip_last, 'proba_long') devuelve siempre None por este motivo.
            # 5. live_order.proba_long — último recurso, puede estar stale.
            model_diag.get("proba_long_cal") if isinstance(model_diag, dict) else None,
            model_diag.get("proba_long_cal_dec") if isinstance(model_diag, dict) else None,
            model_diag.get("proba_long_raw") if isinstance(model_diag, dict) else None,
            pip_get(pip_last, "pred_long_cal") if pip_get else None,
            live_order.get("proba_long") if live_order else None,
        )
        put(
            "proba_short",
            model_diag.get("proba_short_cal") if isinstance(model_diag, dict) else None,
            model_diag.get("proba_short_cal_dec") if isinstance(model_diag, dict) else None,
            model_diag.get("proba_short_raw") if isinstance(model_diag, dict) else None,
            pip_get(pip_last, "pred_short_cal") if pip_get else None,
            live_order.get("proba_short") if live_order else None,
        )

        volume = self._first_not_none(data.get("volume"), data.get("tick_volume"))
        if volume is not None:
            indicators["volume"] = volume
        if self._volume_sma.value is not None:
            indicators["volume_ma"] = self._volume_sma.value

        return indicators

    def _build_current_context(self, data: dict, live_order: Optional[dict], model_diag: Optional[dict]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}

        regime = self._first_not_none(
            live_order.get("state") if live_order else None,
            model_diag.get("state") if isinstance(model_diag, dict) else None,
        )
        if regime is not None:
            ctx["regime"] = regime

        score = self._first_not_none(
            live_order.get("score") if live_order else None,
            model_diag.get("score") if isinstance(model_diag, dict) else None,
            model_diag.get("score_long") if isinstance(model_diag, dict) else None,
        )
        if score is not None:
            ctx["score"] = score

        if data.get("spread") is not None:
            ctx["spread"] = data.get("spread")

        return ctx

    def _build_minimal_indicators_from_pipeline(self, pip_last, pip_prev, pip_get) -> Dict[str, Any]:
        out: Dict[str, Any] = {}

        def put(k: str, v):
            if v is not None:
                out[k] = v

        if pip_get:
            put("rsi", pip_get(pip_last, "rsi"))
            put("macd_hist", pip_get(pip_last, "macd_hist"))
            put("macd_hist_prev", pip_get(pip_prev, "macd_hist"))
            put("atr", pip_get(pip_last, "atr"))
            put("proba_long", pip_get(pip_last, "proba_long"))
            put("proba_short", pip_get(pip_last, "proba_short"))

        if self._volume_sma.value is not None:
            out["volume_ma"] = self._volume_sma.value

        return out

    def _build_open_request(
        self,
        live_order: dict,
        data: dict,
        current_indicators: dict,
        current_context: dict,
        bb_upper=None,
        bb_lower=None,
    ) -> dict:
        enriched_order = dict(live_order)

        if "atr_at_entry" not in enriched_order or not enriched_order.get("atr_at_entry"):
            enriched_order["_atr_fallback"] = current_indicators.get("atr")

        enriched_order["bb_upper"] = bb_upper
        enriched_order["bb_lower"] = bb_lower

        side_s3 = self._side_to_s3(live_order.get("side"))
        volume = self._safe_float(live_order.get("qty"))

        geom = self.order_builder.build(
            order=enriched_order,
            side=side_s3,
            entry=self._safe_float(live_order.get("entry")),
            point=0.01,
            bid=data.get("bid"),
            ask=data.get("ask"),
            spread_points=data.get("spread"),
            bb_upper=bb_upper,
            bb_lower=bb_lower,
        )

        metadata = dict(geom["metadata"])
        metadata["requested_be_offset_points"] = int(self.OPEN_BE_OFFSET_POINTS)
        metadata["requested_runner_tight_trail_pts"] = int(self.RUNNER_TIGHT_TRAIL_PTS)

        request = {
            "action": "OPEN",
            "symbol": live_order.get("symbol", "XAUUSD.r"),
            "side": side_s3,
            "volume": volume,
            "magic": 0,
            "comment": "production",
            "virtual_sl": geom["virtual_sl_price"],
            "virtual_tp": geom["virtual_tp_price"],
            "risk": {
                "profile": "scalping",
                "emergency_points": int(geom["emergency_points"]),
                "max_hold_seconds": 900,
                "be_offset_points": int(self.OPEN_BE_OFFSET_POINTS),
                "partial_close": {
                    "runner_tight_trail_pts": int(self.RUNNER_TIGHT_TRAIL_PTS),
                },
            },
            "indicators": current_indicators or {},
            "context": current_context or {},
            "metadata": metadata,
            "sent_ts": time.time(),
        }

        return request

    def _send_modify_for_ticket(self, ticket: int, indicators: dict, context: dict, *, keepalive: bool = False):
        if not indicators and not context:
            return

        cmd = {
            "action": "MODIFY",
            "ticket": int(ticket),
            "indicators": indicators or {},
            "context": context or {},
        }
        if keepalive:
            cmd["keepalive"] = True

        self._send_order(cmd)
        self._keepalive_last_sent[int(ticket)] = time.time()

        if indicators:
            self._keepalive_last_indicators[int(ticket)] = dict(indicators)
        if context:
            self._keepalive_last_context[int(ticket)] = dict(context)

    def _compute_active_tickets(self) -> Set[int]:
        # FIX: fuentes de verdad con prioridad clara.
        # 1. s3_state.positions         = eventos ZMQ de S3 (snapshot)
        # 2. _s3_open_tickets_local     = espejo local por callbacks on_position_opened/closed
        # 3. s1_open_tickets            = tickets abiertos en MT5 según S1 (fuente real)
        # 4. _keepalive_last_sent       = fallback SOLO si S1 también reporta tickets
        #
        # Además, si S3 ya confirmó el cierre de un ticket, se deja un tombstone
        # temporal para que ese ticket no vuelva a ser tratado como activo aunque
        # S1 o los dicts de keepalive lo sigan arrastrando 1-2 ciclos.
        self._prune_ticket_tombstones()
        try:
            s3_positions = self.s3_state.get_positions()
            active = set(int(t) for t in s3_positions.keys())
        except Exception:
            active = set()

        active |= set(int(t) for t in self._s3_open_tickets_local)

        # Fallback a keepalive SOLO si S1 también reporta tickets abiertos.
        # Si S1 dice 0 tickets, no inflar active con keepalive stale.
        if not active and self.runtime.s1_open_tickets:
            active |= set(int(t) for t in self._keepalive_last_sent.keys())

        if not active:
            active |= set(int(t) for t in (self.runtime.s1_open_tickets or set()))

        if self._closed_ticket_tombstones:
            active = {int(t) for t in active if int(t) not in self._closed_ticket_tombstones}

        return active

    def _cleanup_closed_tickets(self, active_tickets: Set[int]):
        # FIX: S1 es la fuente de verdad absoluta (lee MT5 en cada tick).
        # Si S1 dice que no hay tickets abiertos Y active_tickets está vacío
        # (S3 tampoco reporta posiciones), limpiar TODOS los dicts de keepalive.
        #
        # Además, cualquier ticket con tombstone de cierre se elimina siempre de
        # los dicts auxiliares aunque S1 tarde 1-2 ciclos en dejar de reportarlo.
        _keepalive_dicts = (
            self._keepalive_last_sent,
            self._keepalive_last_bar_time,
            self._keepalive_last_indicators,
            self._keepalive_last_context,
        )

        self._prune_ticket_tombstones()

        if not active_tickets and not self.runtime.s1_open_tickets:
            # No hay posiciones en ningún sitio → limpiar todos los dicts
            for d in _keepalive_dicts:
                d.clear()
            return

        # Caso normal: limpiar tickets que no están ni en active ni en S1
        truth = set(active_tickets) | set(self.runtime.s1_open_tickets) | set(self._s3_open_tickets_local)
        truth = {int(t) for t in truth if int(t) not in self._closed_ticket_tombstones}
        for d in _keepalive_dicts:
            to_drop = [int(t) for t in d.keys() if int(t) not in truth]
            for t in to_drop:
                d.pop(t, None)

    def _send_modifys_and_keepalive(
        self,
        data: dict,
        current_indicators: dict,
        current_context: dict,
        pip_last=None,
        pip_prev=None,
        pip_get=None,
    ):
        active_tickets = self._compute_active_tickets()
        self._cleanup_closed_tickets(active_tickets)
        if not active_tickets:
            return

        bar_time = int(data.get("time") or 0)
        now_ts = time.time()
        modified_this_tick: Set[int] = set()

        minimal_pipeline_indicators = self._build_minimal_indicators_from_pipeline(pip_last, pip_prev, pip_get)

        effective_current_indicators = dict(current_indicators or {})
        if len(effective_current_indicators) < 3:
            for k, v in minimal_pipeline_indicators.items():
                effective_current_indicators.setdefault(k, v)

        has_pipeline_data = any(k in effective_current_indicators for k in ("rsi", "macd_hist", "atr"))

        s3_positions = self.s3_state.get_positions()

        for ticket in sorted(active_tickets):
            # Guard: no enviar MODIFY a tickets que s3_state ya marcó como cerrados.
            # Evita MODIFY_FAILED: NOT_TRACKED cuando S2 procesa la siguiente barra
            # tras un cierre por vSL (la posición desaparece de s3_state pero aún
            # puede estar en s1_open_tickets durante 1-2 ciclos).
            if ticket in self._closed_ticket_tombstones:
                continue

            if ticket not in s3_positions and ticket not in self.runtime.s1_open_tickets:
                self.system_logger.warning(json.dumps({
                    "event": "MODIFY_SKIP_CLOSED",
                    "ticket": ticket,
                    "reason": "not_in_s3_state_nor_s1",
                    "ts": time.time(),
                }))
                self._drop_ticket_state(ticket)
                continue

            last_bar = self._keepalive_last_bar_time.get(ticket)
            if last_bar != bar_time and (has_pipeline_data or len(effective_current_indicators) > 1):
                self._send_modify_for_ticket(
                    ticket,
                    effective_current_indicators,
                    current_context,
                    keepalive=False,
                )
                self._keepalive_last_bar_time[ticket] = bar_time
                modified_this_tick.add(ticket)

        for ticket in sorted(active_tickets):
            if ticket in modified_this_tick:
                continue

            # Guard: mismo check que el bucle de MODIFY por barra
            if ticket in self._closed_ticket_tombstones:
                continue

            if ticket not in s3_positions and ticket not in self.runtime.s1_open_tickets:
                self._drop_ticket_state(ticket)
                continue

            last_sent = float(self._keepalive_last_sent.get(ticket, 0.0))
            if (now_ts - last_sent) >= self.config.keepalive.interval_secs:
                ka_indicators = dict(self._keepalive_last_indicators.get(ticket) or {})
                ka_context = dict(self._keepalive_last_context.get(ticket) or {})

                if len(ka_indicators) < 3:
                    for k, v in effective_current_indicators.items():
                        ka_indicators.setdefault(k, v)

                if not ka_context:
                    ka_context = dict(current_context or {})

                if ka_indicators or ka_context:
                    self._send_modify_for_ticket(
                        ticket,
                        ka_indicators,
                        ka_context,
                        keepalive=True,
                    )

    def _signal_age_bars(self, bar_time: int, live_order: dict) -> Optional[int]:
        entry_time = live_order.get("entry_time")
        if entry_time is None:
            return None
        try:
            return int(round((int(bar_time) - int(entry_time)) / 60))
        except Exception:
            return None

    def _entry_gap_threshold_pts(self, live_order: dict, current_indicators: dict) -> int:
        atr_price = self._first_not_none(
            live_order.get("atr_at_entry") if live_order else None,
            live_order.get("atr") if live_order else None,
            current_indicators.get("atr") if current_indicators else None,
        )
        atr_price = self._safe_float(atr_price, 0.0)
        atr_pts = self._price_to_points(0.0, atr_price, point=0.01) if atr_price > 0 else 0
        return max(self.config.open_guard.max_entry_gap_pts, atr_pts)

    def _check_reversal_guard(
        self,
        *,
        live_order: Optional[dict],
        current_indicators: dict,
        current_context: dict,
    ) -> Optional[str]:
        guard = getattr(self.config, "reversal_guard", None)
        if not guard or not getattr(guard, "enabled", False) or not live_order:
            return None

        state = self._norm_state(self._first_not_none(
            live_order.get("state"),
            current_context.get("regime") if current_context else None,
        ))
        side_l = str(live_order.get("side") or "").strip().lower()

        rsi = self._first_not_none(
            current_indicators.get("rsi") if current_indicators else None,
            live_order.get("rsi"),
        )
        macd_hist = self._first_not_none(
            current_indicators.get("macd_hist") if current_indicators else None,
            live_order.get("macd_hist"),
        )
        proba_long = self._first_not_none(
            current_indicators.get("proba_long") if current_indicators else None,
            live_order.get("proba_long"),
        )
        proba_short = self._first_not_none(
            current_indicators.get("proba_short") if current_indicators else None,
            live_order.get("proba_short"),
        )

        def _missing(*pairs):
            return [name for name, value in pairs if value is None]

        if side_l in ("buy", "long") and state == "trend_down" and getattr(guard, "long_in_trend_down", False):
            missing = _missing(("rsi", rsi), ("macd_hist", macd_hist), ("proba_long", proba_long), ("proba_short", proba_short))
            if missing and getattr(guard, "require_indicators_present", True):
                return f"REVERSAL_GUARD_LONG_TREND_DOWN(missing={','.join(missing)})"
            if rsi is not None and float(rsi) < float(guard.long_min_rsi):
                return f"REVERSAL_GUARD_LONG_TREND_DOWN(rsi={float(rsi):.2f}<{float(guard.long_min_rsi):.2f})"
            if getattr(guard, "require_macd_flip", True) and macd_hist is not None and float(macd_hist) < 0.0:
                return f"REVERSAL_GUARD_LONG_TREND_DOWN(macd_hist={float(macd_hist):.6f}<0)"
            if proba_long is not None and proba_short is not None:
                edge = float(proba_long) - float(proba_short)
                if edge < float(guard.min_proba_edge):
                    return f"REVERSAL_GUARD_LONG_TREND_DOWN(edge={edge:.4f}<{float(guard.min_proba_edge):.4f})"

        if side_l in ("sell", "short") and state == "trend_up" and getattr(guard, "short_in_trend_up", False):
            missing = _missing(("rsi", rsi), ("macd_hist", macd_hist), ("proba_long", proba_long), ("proba_short", proba_short))
            if missing and getattr(guard, "require_indicators_present", True):
                return f"REVERSAL_GUARD_SHORT_TREND_UP(missing={','.join(missing)})"
            if rsi is not None and float(rsi) > float(guard.short_max_rsi):
                return f"REVERSAL_GUARD_SHORT_TREND_UP(rsi={float(rsi):.2f}>{float(guard.short_max_rsi):.2f})"
            if getattr(guard, "require_macd_flip", True) and macd_hist is not None and float(macd_hist) > 0.0:
                return f"REVERSAL_GUARD_SHORT_TREND_UP(macd_hist={float(macd_hist):.6f}>0)"
            if proba_long is not None and proba_short is not None:
                edge = float(proba_short) - float(proba_long)
                if edge < float(guard.min_proba_edge):
                    return f"REVERSAL_GUARD_SHORT_TREND_UP(edge={edge:.4f}<{float(guard.min_proba_edge):.4f})"

        return None

    def _check_open_guards(
        self,
        *,
        data: dict,
        live_order: Optional[dict],
        current_indicators: dict,
        current_context: dict,
        effective_positions: int,
    ) -> Optional[str]:
        if not live_order:
            return None

        bar_time = int(data.get("time") or 0)
        now_ts = time.time()

        # 0) startup grace
        if len(self._startup_bars_seen) <= self.config.startup_grace_bars:
            return "STARTUP_OPEN_BLOCKED"

        # 0b) cooldown post-cierre: evita re-entrada inmediata tras un vSL.
        # Motivacion: bursts de 5-7 posiciones en 8 minutos observados en el
        # log 13/04/2026. S2 enviaba OPEN en cada barra M1 porque el modelo
        # seguia generando señal con el mercado moviendose en la misma
        # direccion, pero cada posicion cerraba por vSL antes de que S1
        # actualizara n_positions, permitiendo que el guard de max_positions
        # no actuara. Cooldown de 3 barras (180s) da tiempo a que el mercado
        # se estabilice y a que S1/S3 sincronicen su estado.
        secs_since_close = now_ts - self._last_close_ts
        if self._last_close_ts > 0 and secs_since_close < self.POST_CLOSE_COOLDOWN_SECS:
            remaining = int(self.POST_CLOSE_COOLDOWN_SECS - secs_since_close)
            return f"POST_CLOSE_COOLDOWN(ticket={self._last_close_ticket},remaining={remaining}s)"

        # 1) anti-duplicado por tiempo/barra
        if (
            self._open_guard["bar_time"] == bar_time
            or (now_ts - float(self._open_guard["sent_ts"])) < self.config.open_guard.open_guard_secs
        ):
            return "OPEN_GUARD"

        # 1b) cooldown inter-señal: bloquea si el último OPEN se envió hace
        # menos de SIGNAL_INTER_COOLDOWN_SECS, independientemente de la barra.
        # Esto cubre el caso de señales en barras consecutivas (gap=60s) que
        # el OPEN_GUARD por bar_time no captura.
        secs_since_last_open = now_ts - self._last_open_sent_ts
        if self._last_open_sent_ts > 0 and secs_since_last_open < self.SIGNAL_INTER_COOLDOWN_SECS:
            remaining_cd = int(self.SIGNAL_INTER_COOLDOWN_SECS - secs_since_last_open)
            return f"SIGNAL_INTER_COOLDOWN(remaining={remaining_cd}s)"

        # 2) señal demasiado vieja
        age_bars = self._signal_age_bars(bar_time, live_order)
        if age_bars is not None and age_bars > self.config.open_guard.max_signal_age_bars:
            return f"SIGNAL_TOO_OLD(age_bars={age_bars})"

        # 3) gap entry vs fill_ref demasiado grande
        side_s3 = self._side_to_s3(live_order.get("side"))
        model_entry = self._safe_float(live_order.get("entry"))
        fill_ref = data.get("ask") if side_s3 == "BUY" else data.get("bid")
        if fill_ref is None:
            fill_ref = data.get("close")
        gap_pts = self._price_to_points(model_entry, fill_ref, point=0.01)
        threshold = self._entry_gap_threshold_pts(live_order, current_indicators)
        if gap_pts > threshold:
            return f"ENTRY_GAP_TOO_LARGE(gap_pts={gap_pts},threshold={threshold})"

        # 4) bloqueo counter-trend fuerte
        state = self._norm_state(self._first_not_none(
            live_order.get("state"),
            current_context.get("regime") if current_context else None,
        ))
        score = self._safe_float(self._first_not_none(
            live_order.get("score"),
            current_context.get("score") if current_context else None,
        ), 0.0)
        side_l = str(live_order.get("side") or "").strip().lower()

        # 3b) filtro RSI: bloquea entradas en zonas de sobrecompra/sobreventa.
        # Umbral 75/25 calibrado sobre log 17/04/2026:
        #   - BUY con RSI=77.2 (08:45) → -255pts evitables.
        #   - Umbral 70 también lo capturaría pero produce más falsos positivos
        #     en tendencias fuertes; 75 es conservador y suficiente.
        # El filtro es independiente del régimen: incluso en TREND_UP un RSI>75
        # indica precio extendido con mayor riesgo de pullback inmediato.
        _rsi_val = self._first_not_none(
            current_indicators.get("rsi") if current_indicators else None,
            live_order.get("rsi"),
        )
        if _rsi_val is not None:
            _rsi_f = float(_rsi_val)
            if side_l in ("buy", "long") and _rsi_f > self.config.rsi_overbought_threshold:
                return f"RSI_OVERBOUGHT(rsi={_rsi_f:.1f}>{self.config.rsi_overbought_threshold})"
            if side_l in ("sell", "short") and _rsi_f < self.config.rsi_oversold_threshold:
                return f"RSI_OVERSOLD(rsi={_rsi_f:.1f}<{self.config.rsi_oversold_threshold})"

        # 4) bloqueo weak-transition: en estados TRANSITION_*, exige que la
        # diferencia de probabilidad entre el lado señalado y el contrario supere
        # un mínimo. Una transición con delta<0.10 es más ruido que señal real.
        # Observado en log 17/04/2026: 3 SELL en TRANSITION_DOWN con delta≈0.088
        # cerraron todos en TREND_UP → -675pts. El 4º SELL (RANGE, delta=0.099)
        # cerró en +5pts — por debajo del umbral pero en régimen diferente.
        if "transition" in state:
            _proba_long_v = self._first_not_none(
                current_indicators.get("proba_long") if current_indicators else None,
                live_order.get("proba_long"),
            )
            _proba_short_v = self._first_not_none(
                current_indicators.get("proba_short") if current_indicators else None,
                live_order.get("proba_short"),
            )
            if _proba_long_v is not None and _proba_short_v is not None:
                if side_l in ("sell", "short"):
                    _delta_tr = float(_proba_short_v) - float(_proba_long_v)
                else:
                    _delta_tr = float(_proba_long_v) - float(_proba_short_v)
                if _delta_tr < self.config.transition_min_proba_delta:
                    return (
                        f"TRANSITION_WEAK_SIGNAL("
                        f"delta={_delta_tr:.3f}<{self.config.transition_min_proba_delta},"
                        f"state={state})"
                    )

        # 5) bloqueo counter-trend fuerte
        if self.config.counter_trend.block_total:
            if side_l in ("sell", "short") and state in self.config.counter_trend.regimes_short_block:
                return f"COUNTER_TREND_BLOCKED(state={state})"
            if side_l in ("buy", "long") and state in self.config.counter_trend.regimes_long_block:
                return f"COUNTER_TREND_BLOCKED(state={state})"
        else:
            min_score = self._safe_float(getattr(getattr(self.simulator, "risk_config", None), "min_score_to_trade", 0.0), 0.0)
            if side_l in ("sell", "short") and state in self.config.counter_trend.regimes_short_block:
                if score < (min_score + self.config.counter_trend.score_penalty):
                    return f"COUNTER_TREND_LOW_SCORE(score={score:.4f},state={state})"
            if side_l in ("buy", "long") and state in self.config.counter_trend.regimes_long_block:
                if score < (min_score + self.config.counter_trend.score_penalty):
                    return f"COUNTER_TREND_LOW_SCORE(score={score:.4f},state={state})"

        # 5b) guardia estricta de reversión en contra-tendencia fuerte
        strict_reversal_reason = self._check_reversal_guard(
            live_order=live_order,
            current_indicators=current_indicators,
            current_context=current_context,
        )
        if strict_reversal_reason:
            return strict_reversal_reason

        # 6) max_positions
        max_positions = int(getattr(getattr(self.simulator, "risk_config", None), "max_positions", 2))
        if effective_positions >= max_positions:
            return f"MAX_POSITIONS({effective_positions}/{max_positions})"

        return None

    # ─────────────────────────────────────────────────────────────────────
    # Gating del base_tf del modelo (5min para release 202500+)
    # ─────────────────────────────────────────────────────────────────────

    def _resample_to_base_tf(self, df_1m: pd.DataFrame) -> pd.DataFrame:
        """Resamplea OHLCV de 1-min al BASE_TF_MINUTES configurado.

        Si BASE_TF_MINUTES == 1 → devuelve el df tal cual (no-op).

        Para releases entrenados con base_tf=5min (202200, 202300, 202500,
        ...) hay que pasar bars 5m al modelo, no 1m, porque las sequencias
        seq_short/seq_long se construyen sobre el TF nativo del modelo y
        sus features (ATR, EMA, etc.) también.
        """
        if self.BASE_TF_MINUTES == 1:
            return df_1m

        df = df_1m.copy()
        df["time"] = pd.to_datetime(df["time"])
        df = df.set_index("time").sort_index()

        agg_spec = {"open": "first", "high": "max", "low": "min", "close": "last"}
        if "ticks_volume" in df.columns:
            agg_spec["ticks_volume"] = "sum"
        if "real_volume" in df.columns:
            agg_spec["real_volume"] = "sum"
        if "spread" in df.columns:
            # spread es ruidoso; usamos el último del bin
            agg_spec["spread"] = "last"

        rule = f"{self.BASE_TF_MINUTES}min"
        agg = (
            df.resample(rule, label="right", closed="right")
            .agg(agg_spec)
            .dropna(subset=["close"])
            .reset_index()
        )
        return agg

    def _should_process_basetf_close(self, tick_time_epoch) -> bool:
        """Decide si el tick actual cierra una nueva barra base_tf.

        Returns True solo cuando:
          1. El minuto del tick es múltiplo de BASE_TF_MINUTES (e.g. 0,5,10,...).
          2. No hemos procesado ya esta barra (idempotencia, por si llega más
             de un tick con el mismo timestamp del bar de cierre).

        Si BASE_TF_MINUTES == 1, siempre devuelve True (todo tick es cierre
        de barra 1m).
        """
        if tick_time_epoch is None:
            return False
        try:
            tick_time_epoch = int(tick_time_epoch)
        except (TypeError, ValueError):
            return False

        ts = pd.Timestamp(tick_time_epoch, unit="s")
        if self.BASE_TF_MINUTES == 1:
            bar_key = ts.replace(second=0, microsecond=0)
            if self._last_processed_basetf_bar_ts == bar_key:
                return False
            self._last_processed_basetf_bar_ts = bar_key
            return True

        if ts.minute % self.BASE_TF_MINUTES != 0:
            return False

        bar_key = ts.replace(second=0, microsecond=0)
        if self._last_processed_basetf_bar_ts == bar_key:
            return False
        self._last_processed_basetf_bar_ts = bar_key
        return True

    def process_tick(self, data: dict):
        print(f"Data from MT5: {data}")

        balance = data.get("balance")
        equity = data.get("equity")
        n_positions = int(data.get("n_positions", 0) or 0)
        self.runtime.s1_open_tickets = set(data.get("open_tickets", []) or [])

        vol = self._first_not_none(data.get("tick_volume"), data.get("volume"))
        self._volume_sma.push(vol)

        df = self._prepare_bar_df(data)
        future = self._db_executor.submit(self._db_save_async, df.copy())
        try:
            future.result()
        except Exception as db_err:
            print(f"[ERROR][db-save] Fallo al guardar barra en BD — saltando tick: {db_err}")
            return

        bar_snapshot = {
            "time": data.get("time"),
            "open": data.get("open"),
            "high": data.get("high"),
            "low": data.get("low"),
            "close": data.get("close"),
            "spread": data.get("spread"),
        }

        # ── 1) Carga 1m bars (cubre warmup multi-TF: 8000 1m ≈ 5.5 días)
        df_rates_1m = self.simulator.helper.load_from_database_real(
            self.db, last_n_rates=self.LAST_N_RATES
        )

        # ── 2) Resamplea 1m → base_tf SIEMPRE (coste despreciable)
        # Necesario para que _extract_pipeline_snapshot trabaje con bars
        # del mismo TF que el modelo, y para que indicators reflejen la
        # granularidad correcta en el keepalive a S3.
        df_rates = self._resample_to_base_tf(df_rates_1m)
        if len(df_rates) < 100:
            print(f"[S2][WARN] solo {len(df_rates)} bars {self.BASE_TF_MINUTES}m tras resample. "
                  f"Aborto este tick (warmup insuficiente).")
            return

        # ── 3) Trigger del modelo solo en cierre de barra base_tf
        # En ticks intermedios (minute % 5 != 0 para BASE_TF=5m) NO llamamos
        # a decide_live, pero SÍ corremos keepalive + guards + position
        # management en bloques posteriores de esta función. Esto evita que
        # S3 marque posiciones como NOT_TRACKED por timeout (el keepalive
        # se envía cada tick, no cada cierre 5m).
        tick_time_epoch = data.get("time")
        should_invoke_model = self._should_process_basetf_close(tick_time_epoch)
        ts_dbg = (pd.Timestamp(int(tick_time_epoch), unit="s")
                  if tick_time_epoch else "n/a")

        s3_n_pre = self.s3_state.n_positions()
        s3_n_local = len(self._s3_open_tickets_local)
        effective_positions = max(n_positions, s3_n_pre, s3_n_local)
        if max(s3_n_pre, s3_n_local) != n_positions:
            print(
                f"\t[S3_SYNC] s3_snapshot={s3_n_pre} s3_local={s3_n_local} vs mt5={n_positions} "
                f"→ using effective={effective_positions}"
            )

        live_order = None
        model_diag = None
        if should_invoke_model:
            print(f"[S2][TF] 1m → {self.BASE_TF_MINUTES}m: {len(df_rates_1m)} → {len(df_rates)} bars  "
                  f"cierre @ {df_rates['time'].iloc[-1]}")
            live_order = self.simulator.decide_live(
                df_rates=df_rates,
                equity=float(equity),
                current_positions=effective_positions,
                max_positions=2,
            )
            model_diag = getattr(self.simulator, "_last_diag", None)
        else:
            print(f"[S2][SKIP-MODEL] tick @ {ts_dbg} no cierra bar "
                  f"{self.BASE_TF_MINUTES}m → keepalive only, decide_live skipped")
        _, pip_last, pip_prev, pip_get, bb_upper, bb_lower = self._extract_pipeline_snapshot(df_rates)

        current_indicators = self._build_current_indicators(
            data,
            live_order,
            pip_last,
            pip_prev,
            pip_get,
            model_diag,
        )
        current_context = self._build_current_context(data, live_order, model_diag)

        self._send_modifys_and_keepalive(
            data,
            current_indicators,
            current_context,
            pip_last=pip_last,
            pip_prev=pip_prev,
            pip_get=pip_get,
        )

        bar_time = int(data.get("time") or 0)
        self._startup_bars_seen.add(bar_time)

        block_reason = self._check_open_guards(
            data=data,
            live_order=live_order,
            current_indicators=current_indicators,
            current_context=current_context,
            effective_positions=effective_positions,
        )

        # Enriquecer model_diag con estado del cooldown para visibilidad en
        # NO_SIGNAL aunque el modelo no haya generado señal (live_order=None).
        _now_ts = time.time()
        _secs_since_close = _now_ts - self._last_close_ts if self._last_close_ts > 0 else None
        _cooldown_active = (
            self._last_close_ts > 0
            and _secs_since_close is not None
            and _secs_since_close < self.POST_CLOSE_COOLDOWN_SECS
        )
        if _cooldown_active and isinstance(model_diag, dict):
            model_diag = dict(model_diag)
            model_diag["post_close_cooldown_remaining"] = int(
                self.POST_CLOSE_COOLDOWN_SECS - _secs_since_close
            )
            model_diag["post_close_last_ticket"] = self._last_close_ticket

        if live_order and not block_reason:
            request = self._build_open_request(
                live_order,
                data,
                current_indicators,
                current_context,
                bb_upper=bb_upper,
                bb_lower=bb_lower,
            )
            self._send_order(request)
            self._open_guard["bar_time"] = bar_time
            self._open_guard["sent_ts"] = time.time()
            self._last_open_sent_ts = time.time()   # FIX BUG-1: registrar ts para inter-signal cooldown
            effective_positions += 1

            _meta = request.get("metadata", {})
            # FIX BUG-4: enriquecer el model_diag de SIGNAL_SENT con probas frescas.
            # El diag inline solo tenía campos de RR. Añadimos proba_long_cal y
            # proba_short_cal del model_diag real (ahora poblado en el path de éxito
            # por el fix en trading_simulator_v2.py decide_live).
            _signal_diag = {
                "action": request.get("action"),
                "regime": self._norm_state(self._first_not_none(
                    live_order.get("state"),
                    current_context.get("regime") if current_context else None,
                )) or None,
                "rr_target": _meta.get("rr_target"),
                "rr_final": _meta.get("rr_final"),
                "rr_model": _meta.get("rr_model"),
                "tp_rr_capped": _meta.get("tp_rr_capped"),
                "atr_source": _meta.get("atr_source"),
            }
            if isinstance(model_diag, dict):
                for _k in ("proba_long_cal", "proba_short_cal",
                           "proba_long_cal_dec", "proba_short_cal_dec",
                           "proba_long_raw", "proba_short_raw",
                           "score_long", "score_short", "state",
                           "chop_score", "exhaustion_score"):
                    if _k in model_diag:
                        _signal_diag[_k] = model_diag[_k]
            self._log_signal(
                event="SIGNAL_SENT",
                bar_data=bar_snapshot,
                live_order=live_order,
                request=request,
                n_positions=effective_positions,
                equity=equity,
                balance=balance,
                indicators=current_indicators,
                model_diag=_signal_diag,
            )
        else:
            # FIX 13/04/2026: preservar regime y score del model_diag en NO_SIGNAL.
            # Antes, cuando live_order existía y había block_reason, model_diag se
            # reemplazaba por {"no_signal_reason": block_reason} perdiendo el regime
            # y score que el simulador había calculado. Con 210/222 NO_SIGNAL siendo
            # DECISION_ENGINE_NONE, no podíamos saber en qué régimen ni con qué score
            # se filtraban las señales — imposibilitaba análisis de si los gates son
            # demasiado restrictivos en ciertos regímenes.
            if live_order and block_reason:
                _base_diag = dict(model_diag) if isinstance(model_diag, dict) else {}
                _base_diag["no_signal_reason"] = block_reason
                # Propagar regime y score si están disponibles en live_order
                if "state" not in _base_diag and live_order.get("state"):
                    _base_diag["regime"] = live_order.get("state")
                if "score" not in _base_diag and live_order.get("score") is not None:
                    _base_diag["score"] = live_order.get("score")
                model_diag = _base_diag
            self._log_signal(
                event="NO_SIGNAL",
                bar_data=bar_snapshot,
                n_positions=effective_positions,
                equity=equity,
                balance=balance,
                model_diag=model_diag,
                block_reason=block_reason,
            )

        print("=" * 160)

    def run(self):
        self.start_listener()
        print("[S2] Servicio arrancado. Esperando mensajes de S1...")

        while True:
            try:
                topic, raw = self.ticks_sub.recv_multipart()
            except zmq.Again:
                elapsed = time.time() - self._last_tick_ts
                if elapsed > self._NO_TICK_WARN_SECS:
                    print(f"[WARN][S2] Sin ticks desde hace {elapsed:.0f}s — MT5 desconectado o mercado cerrado")

                self._no_tick_count += 1
                if self._no_tick_count >= self._ZMQ_RECONNECT_AFTER_TIMEOUTS:
                    self._no_tick_count = 0
                    self._reconnect_tries += 1
                    backoff = min(2 ** (self._reconnect_tries - 1) * 5, 300)
                    print(f"[WARN][S2] Reconectando socket MT5 (intento #{self._reconnect_tries}, backoff={backoff}s)…")
                    time.sleep(backoff)
                    self._reconnect_ticks_sub()
                    print(f"[INFO][S2] Socket MT5 recreado → {self._MT5_SUB_ADDR}")
                    self._volume_sma.reset()
                continue
            except Exception as e:
                print(f"[S2][ERROR][recv] {e}")
                time.sleep(0.2)
                continue

            self._last_tick_ts = time.time()
            self._no_tick_count = 0
            self._reconnect_tries = 0

            try:
                data = json.loads(raw.decode("utf-8"))
                data["topic"] = topic.decode("utf-8", errors="ignore")
                self.process_tick(data)
            except Exception as e:
                print(f"[S2][ERROR][process_tick] {e}")
                time.sleep(0.2)