import MetaTrader5 as mt5
from datetime import datetime
import numpy as np
import pandas as pd

mt5.initialize()
ticks_arr = mt5.copy_ticks_range(
    "XAUUSD.r",
    datetime(2025, 10, 8, 12, 0),   # martes 12:00 UTC = London/NY overlap
    datetime(2025, 10, 8, 16, 0),   # 4 horas de máxima actividad
    mt5.COPY_TICKS_ALL,
)

mt5.shutdown()

# Opción A — Convertir desde el structured array crudo (más rápido y seguro)
df = pd.DataFrame({
    "time": ticks_arr["time"],
    "bid": ticks_arr["bid"].astype(np.float64),
    "ask": ticks_arr["ask"].astype(np.float64),
    "last": ticks_arr["last"].astype(np.float64),
    "volume": ticks_arr["volume"].astype(np.int64),
    "volume_real": ticks_arr["volume_real"].astype(np.float64),
    "flags": np.array([int(f) for f in ticks_arr["flags"]], dtype=np.int64),
    "time_msc": ticks_arr["time_msc"].astype(np.int64),
})

print(f"Total ticks 1 día: {len(df):,}")
print(f"\nDistribución de flags (top 10):")
print(df["flags"].value_counts().head(10))

F_BID    = 0x02
F_ASK    = 0x04
F_LAST   = 0x08
F_VOLUME = 0x10
F_BUY    = 0x20
F_SELL   = 0x40

print(f"\nfracción con BID flag : {((df.flags & F_BID)    > 0).mean():.3f}")
print(f"fracción con ASK flag : {((df.flags & F_ASK)    > 0).mean():.3f}")
print(f"fracción con LAST flag: {((df.flags & F_LAST)   > 0).mean():.3f}")
print(f"fracción con BUY flag : {((df.flags & F_BUY)    > 0).mean():.3f}")
print(f"fracción con SELL flag: {((df.flags & F_SELL)   > 0).mean():.3f}")

print(f"\nvolume vs volume_real (describe):")
print(df[["volume", "volume_real"]].describe().round(4))

print(f"\nticks con volume_real > 0 : {(df.volume_real > 0).mean():.3f}")
print(f"ticks con last > 0        : {(df.last > 0).mean():.3f}")
print(f"\nvolume_real unique sample : {df.volume_real.unique()[:20]}")
