"""
analyze_timing.py
═══════════════════════════════════════════════════════════════════════════════
Análisis de cuándo se toca el SL en las muestras con signal=0.

Objetivo: determinar si SL=1xATR está dentro del ruido natural de la vela,
midiendo en qué barra (1,2,3,...,horizon) se tocó el SL para los label=0.

Uso:
    python analyze_timing.py \
        --data  /ruta/a/tu/csv_o_parquet \
        --side  long \
        --horizon 5 \
        --tp    2.5 \
        --sl    1.0 \
        --atr_period 14 \
        --output ./sl_timing_report
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from numba import njit


# ── Funciones Numba ────────────────────────────────────────────────────────────

@njit
def atr_numba(high, low, close, period=14):
    n = len(close)
    atr = np.full(n, np.nan)
    tr = np.zeros(n)
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)
    atr[period - 1] = np.mean(tr[:period])
    alpha = 1.0 / period
    for i in range(period, n):
        atr[i] = atr[i - 1] * (1 - alpha) + tr[i] * alpha
    return atr


@njit
def find_sl_touch_bar(close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long):
    """
    Para cada barra i devuelve:
      - signal[i]: 1 si TP tocado antes que SL, 0 si SL tocado primero o timeout
      - sl_bar[i]: barra (1..horizon) en que se tocó el SL (-1 si no se tocó)
      - tp_bar[i]: barra (1..horizon) en que se tocó el TP (-1 si no se tocó)
      - outcome[i]: 0=timeout, 1=TP, 2=SL
    """
    n = len(close)
    signal  = np.zeros(n, dtype=np.int8)
    sl_bar  = np.full(n, -1, dtype=np.int32)
    tp_bar  = np.full(n, -1, dtype=np.int32)
    outcome = np.zeros(n, dtype=np.int8)  # 0=timeout, 1=TP, 2=SL

    for i in range(n - horizon):
        if np.isnan(atr[i]) or atr[i] <= 0:
            continue

        entry = close[i]
        tp_dist = tp_mult * atr[i]
        sl_dist = sl_mult * atr[i]

        if side_is_long:
            tp_level = entry + tp_dist
            sl_level = entry - sl_dist
        else:
            tp_level = entry - tp_dist
            sl_level = entry + sl_dist

        hit_tp = -1
        hit_sl = -1

        for k in range(1, horizon + 1):
            j = i + k
            if j >= n:
                break

            if side_is_long:
                if hit_tp == -1 and high[j] >= tp_level:
                    hit_tp = k
                if hit_sl == -1 and low[j] <= sl_level:
                    hit_sl = k
            else:
                if hit_tp == -1 and low[j] <= tp_level:
                    hit_tp = k
                if hit_sl == -1 and high[j] >= sl_level:
                    hit_sl = k

            if hit_tp != -1 and hit_sl != -1:
                break

        tp_bar[i] = hit_tp
        sl_bar[i] = hit_sl

        if hit_tp != -1 and (hit_sl == -1 or hit_tp <= hit_sl):
            signal[i] = 1
            outcome[i] = 1
        elif hit_sl != -1 and (hit_tp == -1 or hit_sl < hit_tp):
            signal[i] = 0
            outcome[i] = 2
        else:
            signal[i] = 0
            outcome[i] = 0  # timeout

    return signal, sl_bar, tp_bar, outcome


# ── Carga de datos ─────────────────────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.parquet', '.pq'):
        df = pd.read_parquet(path)
    elif ext == '.csv':
        df = pd.read_csv(path, parse_dates=True, index_col=0)
    elif ext in ('.pkl', '.pickle'):
        df = pd.read_pickle(path)
    else:
        raise ValueError(f"Formato no soportado: {ext}. Usa .parquet, .csv o .pkl")

    # Normalizar nombres de columnas
    df.columns = [c.lower() for c in df.columns]
    required = {'open', 'high', 'low', 'close'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas OHLC: {missing}")

    # Ordenar cronológicamente
    df = df.sort_index()
    return df


# ── Análisis principal ─────────────────────────────────────────────────────────

def run_analysis(
    df: pd.DataFrame,
    side: str,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    atr_period: int,
    output_dir: str,
    sl_variants: list = None,
):
    os.makedirs(output_dir, exist_ok=True)

    close = df['close'].values.astype(np.float64)
    high  = df['high'].values.astype(np.float64)
    low   = df['low'].values.astype(np.float64)

    atr = atr_numba(high, low, close, period=atr_period)
    side_is_long = (side == 'long')

    print(f"\n{'='*60}")
    print(f"  Análisis SL timing | side={side} | horizon={horizon}")
    print(f"  TP={tp_mult}×ATR  SL={sl_mult}×ATR  ATR_period={atr_period}")
    print(f"  Barras totales: {len(df):,}")
    print(f"{'='*60}")

    signal, sl_bar, tp_bar, outcome = find_sl_touch_bar(
        close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long
    )

    # ── Stats generales ────────────────────────────────────────────────────────
    valid = outcome != 0  # excluir barras sin suficiente forward data
    # Las últimas `horizon` barras no tienen señal válida
    valid[-horizon:] = False

    total      = int(valid.sum())
    n_tp       = int((outcome[valid] == 1).sum())
    n_sl       = int((outcome[valid] == 2).sum())
    n_timeout  = int((outcome[valid] == 0).sum())  # será 0 aquí por el filtro
    base_rate  = n_tp / total if total > 0 else 0

    print(f"\n📊 RESUMEN GENERAL")
    print(f"  Total muestras válidas : {total:,}")
    print(f"  TP tocado (signal=1)   : {n_tp:,}  ({100*n_tp/total:.1f}%)")
    print(f"  SL tocado (signal=0)   : {n_sl:,}  ({100*n_sl/total:.1f}%)")
    print(f"  Timeout   (signal=0)   : {n_timeout:,}  ({100*n_timeout/total:.1f}%)")
    print(f"  Base rate (TP%)        : {base_rate:.3f}")

    # ── Distribución de sl_bar para signal=0 ──────────────────────────────────
    sl_only_mask = valid & (outcome == 2)
    sl_bars_arr  = sl_bar[sl_only_mask]

    counts = np.bincount(sl_bars_arr, minlength=horizon + 1)[1:]  # barras 1..horizon
    pct    = counts / counts.sum() * 100 if counts.sum() > 0 else counts

    cum_pct = np.cumsum(pct)

    print(f"\n📊 DISTRIBUCIÓN DE BARRA EN QUE SE TOCA EL SL (solo signal=0 por SL)")
    print(f"  {'Barra':>6}  {'N':>8}  {'%':>7}  {'Acum%':>7}")
    print(f"  {'-'*35}")
    for b in range(1, horizon + 1):
        idx = b - 1
        print(f"  {b:>6}  {counts[idx]:>8,}  {pct[idx]:>6.1f}%  {cum_pct[idx]:>6.1f}%")

    early_pct = float(cum_pct[0]) if horizon >= 1 else 0  # % tocados en barra 1
    early2_pct = float(cum_pct[1]) if horizon >= 2 else 0  # % tocados en barras 1-2

    print(f"\n  ⚠️  SL tocado en barra 1     : {early_pct:.1f}%")
    print(f"  ⚠️  SL tocado en barras 1-2  : {early2_pct:.1f}%")

    if early_pct > 35:
        print(f"\n  🔴 DIAGNÓSTICO: SL demasiado ajustado.")
        print(f"     Más del 35% de los stops saltan en la primera vela.")
        print(f"     El SL={sl_mult}×ATR está dentro del ruido natural.")
    elif early_pct > 20:
        print(f"\n  🟡 DIAGNÓSTICO: SL posiblemente ajustado.")
        print(f"     {early_pct:.0f}% de stops en barra 1 es elevado para M1.")
    else:
        print(f"\n  🟢 DIAGNÓSTICO: Distribución de SL razonable.")

    # ── Comparativa con variantes de SL ───────────────────────────────────────
    if sl_variants is None:
        sl_variants = [0.75, 1.0, 1.5, 2.0, 2.5]

    print(f"\n📊 COMPARATIVA BASE RATE vs SL MULTIPLIER (tp={tp_mult}×ATR fijo)")
    print(f"  {'SL mult':>8}  {'Base rate':>10}  {'% SL en bar1':>14}  {'Ratio TP/SL':>12}")
    print(f"  {'-'*50}")

    variant_results = []
    for sl_v in sl_variants:
        sig_v, sl_b_v, _, out_v = find_sl_touch_bar(
            close, high, low, atr, horizon, tp_mult, sl_v, side_is_long
        )
        v_valid = np.ones(len(out_v), dtype=bool)
        v_valid[-horizon:] = False
        v_total = int(v_valid.sum())
        v_tp = int((out_v[v_valid] == 1).sum())
        v_sl = int((out_v[v_valid] == 2).sum())
        v_br = v_tp / v_total if v_total > 0 else 0

        # % SL en barra 1
        v_sl_mask = v_valid & (out_v == 2)
        v_sl_bars = sl_b_v[v_sl_mask]
        v_early = float((v_sl_bars == 1).sum() / len(v_sl_bars) * 100) if len(v_sl_bars) > 0 else 0

        marker = " ◄ actual" if abs(sl_v - sl_mult) < 1e-6 else ""
        print(f"  {sl_v:>8.2f}  {v_br:>10.3f}  {v_early:>13.1f}%  {tp_mult/sl_v:>12.2f}{marker}")
        variant_results.append({
            'sl_mult': sl_v, 'base_rate': v_br, 'pct_bar1': v_early,
            'ratio': tp_mult / sl_v, 'n_tp': v_tp, 'n_sl': v_sl
        })

    # ── Gráficos ───────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Análisis SL Timing | side={side} | TP={tp_mult}×ATR | SL={sl_mult}×ATR',
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    # 1. Distribución barras SL
    ax1 = fig.add_subplot(gs[0, 0])
    bars = np.arange(1, horizon + 1)
    colors = ['#e74c3c' if b <= 2 else '#3498db' for b in bars]
    ax1.bar(bars, pct, color=colors, edgecolor='white', linewidth=0.5)
    ax1.set_xlabel('Barra en que se toca el SL')
    ax1.set_ylabel('% de casos')
    ax1.set_title(f'Distribución SL por barra (signal=0 por SL)\nRojo=primeras 2 barras')
    ax1.set_xticks(bars)
    for b, p in zip(bars, pct):
        ax1.text(b, p + 0.3, f'{p:.1f}%', ha='center', fontsize=8)

    # 2. Acumulado
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(bars, cum_pct, 'o-', color='#e74c3c', linewidth=2, markersize=6)
    ax2.axhline(50, color='gray', linestyle='--', alpha=0.5, label='50%')
    ax2.axhline(80, color='gray', linestyle=':', alpha=0.5, label='80%')
    ax2.fill_between(bars, cum_pct, alpha=0.15, color='#e74c3c')
    ax2.set_xlabel('Barra')
    ax2.set_ylabel('% acumulado')
    ax2.set_title('% acumulado de SL tocados hasta barra N')
    ax2.set_xticks(bars)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)

    # 3. Base rate vs SL mult
    vr = pd.DataFrame(variant_results)
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(vr['sl_mult'], vr['base_rate'], 's-', color='#2ecc71', linewidth=2, markersize=8)
    ax3.axvline(sl_mult, color='#e74c3c', linestyle='--', label=f'SL actual={sl_mult}')
    ax3.set_xlabel('SL multiplier (×ATR)')
    ax3.set_ylabel('Base rate (% TP)')
    ax3.set_title('Base rate vs SL multiplier\n(TP fijo)')
    ax3.legend(fontsize=9)
    for _, row in vr.iterrows():
        ax3.annotate(f"{row['base_rate']:.3f}",
                     (row['sl_mult'], row['base_rate']),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    # 4. % SL en barra 1 vs SL mult
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(vr['sl_mult'], vr['pct_bar1'], 'D-', color='#e67e22', linewidth=2, markersize=8)
    ax4.axvline(sl_mult, color='#e74c3c', linestyle='--', label=f'SL actual={sl_mult}')
    ax4.axhline(35, color='gray', linestyle=':', alpha=0.7, label='Umbral crítico 35%')
    ax4.axhline(20, color='gray', linestyle='--', alpha=0.5, label='Umbral aviso 20%')
    ax4.set_xlabel('SL multiplier (×ATR)')
    ax4.set_ylabel('% SL tocados en barra 1')
    ax4.set_title('% SL prematuros (barra 1) vs SL multiplier')
    ax4.legend(fontsize=9)
    for _, row in vr.iterrows():
        ax4.annotate(f"{row['pct_bar1']:.1f}%",
                     (row['sl_mult'], row['pct_bar1']),
                     textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8)

    plot_path = os.path.join(output_dir, f'sl_timing_{side}.png')
    fig.savefig(plot_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\n  📈 Gráfico guardado: {plot_path}")

    # ── CSV de resultados ──────────────────────────────────────────────────────
    csv_path = os.path.join(output_dir, f'sl_timing_{side}.csv')
    vr.to_csv(csv_path, index=False)
    print(f"  📄 CSV guardado: {csv_path}")

    # ── Recomendación final ────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  RECOMENDACIÓN")
    print(f"{'='*60}")

    # Buscar el SL donde % bar1 cae por debajo del 20%
    below_20 = vr[vr['pct_bar1'] < 20.0]
    if not below_20.empty:
        rec_sl = float(below_20.iloc[0]['sl_mult'])
        rec_br = float(below_20.iloc[0]['base_rate'])
        print(f"  El primer SL con <20% stops en barra 1: SL={rec_sl}×ATR")
        print(f"  Base rate resultante: {rec_br:.3f} (actual: {base_rate:.3f})")
        if rec_br > base_rate:
            print(f"  ✅ Base rate MEJORA en +{(rec_br-base_rate)*100:.1f}pp → labels más limpios")
        else:
            print(f"  ⚠️  Base rate baja en {(base_rate-rec_br)*100:.1f}pp → más difícil de clasificar")
    else:
        print(f"  Ningún SL testado consigue <20% stops en barra 1.")
        print(f"  Considera aumentar el rango de sl_variants o usar ATR de periodo mayor.")

    return vr


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Análisis de SL timing')
    parser.add_argument('--data',       required=True,  help='Ruta al fichero de datos (parquet/csv/pkl)')
    parser.add_argument('--side',       default='long', choices=['long', 'short'])
    parser.add_argument('--horizon',    type=int,   default=5)
    parser.add_argument('--tp',         type=float, default=2.5)
    parser.add_argument('--sl',         type=float, default=1.0)
    parser.add_argument('--atr_period', type=int,   default=14)
    parser.add_argument('--output',     default='./sl_timing_report')
    parser.add_argument('--sl_variants', nargs='+', type=float,
                        default=[0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0])
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Datos cargados: {len(df):,} filas | columnas: {list(df.columns)}")

    run_analysis(
        df=df,
        side=args.side,
        horizon=args.horizon,
        tp_mult=args.tp,
        sl_mult=args.sl,
        atr_period=args.atr_period,
        output_dir=args.output,
        sl_variants=args.sl_variants,
    )


if __name__ == '__main__':
    main()