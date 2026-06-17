import json
import time

import zmq

from mt5_bridge import MT5Bridge

bridge = MT5Bridge()
bridge.initialize_or_raise()

context = zmq.Context.instance()
rates_socket = context.socket(zmq.PUB)
rates_socket.setsockopt(zmq.LINGER, 0)
rates_socket.setsockopt(zmq.SNDHWM, 1000)
rates_socket.bind(f'tcp://10.1.21.25:5555')

time.sleep(1.0)

symbols = ['XAUUSD.r']
for symbol in symbols:
    bridge.select(symbol, True)

last_time = {s: 0 for s in symbols}
try:
    while True:
        for symbol in symbols:
            rates = bridge.copy_rates_from_pos(symbol, start_pos=0, count=2)
            if rates is None:
                continue

            last_closed_rate = rates[-2]
            bar_time = last_closed_rate['time']
            if last_time.get(symbol) != bar_time:
                last_time[symbol] = bar_time

                rate = {}
                for name in last_closed_rate.dtype.names:
                    rate[name] = last_closed_rate[name].item()

                info = bridge.info()
                _all_positions = bridge.positions_get()
                n_positions = len(_all_positions)
                # Lista de tickets abiertos — S2 la usa para enviar MODIFYs
                # incluso cuando s3_state no ha recibido el POSITION_OPENED
                # (p.ej. tras un restart de S3 donde el evento ZMQ se pierde).
                open_tickets = [int(p.ticket) for p in _all_positions]

                # Bid/ask del tick actual (precio de mercado real en el momento
                # de publicar la barra). S2 los usa para validar señales y
                # calcular geometría de SL anclada al precio real, no al close
                # de la barra cerrada (que puede diferir cientos de puntos en
                # mercados rápidos).
                tick_ok, tick_bid, tick_ask = bridge.tick_bid_ask(symbol)

                payload = {
                    'symbol': symbol,
                    'balance': info.balance,
                    'equity': info.equity,
                    'free_margin': info.margin_free,
                    'n_positions': n_positions,
                    'open_tickets': open_tickets,
                    **rate,
                    # Precio real de mercado (tick actual, no close de barra)
                    'bid':       round(tick_bid, 5) if tick_ok else None,
                    'ask':       round(tick_ask, 5) if tick_ok else None,
                    'tick_ok':   tick_ok,
                }

                rates_socket.send_multipart([
                    symbol.encode('utf-8'),
                    json.dumps(payload, ensure_ascii=False).encode('utf-8')
                ])

                print(f'Topic ({symbol}): {payload}')

        time.sleep(0.10)

finally:
    rates_socket.close(0)