#!/usr/bin/env python3
"""
Diagnóstico: Verificar si las probabilidades del modelo OOF son uniformes/altas

Este script analiza:
1. Distribución de p_buy_cal y p_sell_cal
2. Correlación entre p_buy y p_sell
3. Rango de valores
4. Si están calibradas correctamente
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Configuración
ARTIFACTS_PATH = "../artifacts/200333/oof/train_only"
RELEASE = "200333"

def analyze_oof_probabilities(artifacts_path: str, release: str):
    """Analiza las probabilidades OOF guardadas."""

    print("=" * 80)
    print("DIAGNÓSTICO DE PROBABILIDADES OOF")
    print("=" * 80)

    # Cargar OOF DataFrames
    oof_long_path = Path(artifacts_path) / f"oof_{release}_long.parquet"
    oof_short_path = Path(artifacts_path) / f"oof_{release}_short.parquet"

    if not oof_long_path.exists():
        print(f"❌ No encontrado: {oof_long_path}")
        return

    if not oof_short_path.exists():
        print(f"❌ No encontrado: {oof_short_path}")
        return

    print(f"✅ Cargando OOF DataFrames...")
    df_long = pd.read_parquet(oof_long_path)
    df_short = pd.read_parquet(oof_short_path)

    print(f"\n📊 ESTADÍSTICAS - MODELO LONG")
    print("-" * 80)
    analyze_probs_distribution(df_long, "oof_proba_raw", "oof_proba_cal")

    print(f"\n📊 ESTADÍSTICAS - MODELO SHORT")
    print("-" * 80)
    analyze_probs_distribution(df_short, "oof_proba_raw", "oof_proba_cal")

    # Análisis cruzado (durante predicción, ambos modelos se ejecutan)
    print(f"\n🔀 ANÁLISIS CRUZADO (Long vs Short)")
    print("-" * 80)

    # Merge solo con columnas necesarias (evitar duplicados)
    cols_long = ["time", "oof_proba_raw", "oof_proba_cal"]
    cols_short = ["time", "oof_proba_raw", "oof_proba_cal"]

    # Filtrar solo columnas que existen
    cols_long_available = [c for c in cols_long if c in df_long.columns]
    cols_short_available = [c for c in cols_short if c in df_short.columns]

    df_merged = df_long[cols_long_available].merge(
        df_short[cols_short_available],
        on="time",
        suffixes=("_long", "_short"),
        how="inner"
    )

    if len(df_merged) > 0:
        p_buy_cal = df_merged["oof_proba_cal_long"]
        p_sell_cal = df_merged["oof_proba_cal_short"]

        print(f"Samples válidos: {len(df_merged)}")
        print(f"\np_buy_cal (LONG calibrado):")
        print(f"  mean:  {p_buy_cal.mean():.6f}")
        print(f"  std:   {p_buy_cal.std():.6f}")
        print(f"  min:   {p_buy_cal.min():.6f}")
        print(f"  max:   {p_buy_cal.max():.6f}")
        print(f"  p01:   {p_buy_cal.quantile(0.01):.6f}")
        print(f"  p10:   {p_buy_cal.quantile(0.10):.6f}")
        print(f"  p50:   {p_buy_cal.quantile(0.50):.6f}")
        print(f"  p90:   {p_buy_cal.quantile(0.90):.6f}")
        print(f"  p99:   {p_buy_cal.quantile(0.99):.6f}")

        print(f"\np_sell_cal (SHORT calibrado):")
        print(f"  mean:  {p_sell_cal.mean():.6f}")
        print(f"  std:   {p_sell_cal.std():.6f}")
        print(f"  min:   {p_sell_cal.min():.6f}")
        print(f"  max:   {p_sell_cal.max():.6f}")
        print(f"  p01:   {p_sell_cal.quantile(0.01):.6f}")
        print(f"  p10:   {p_sell_cal.quantile(0.10):.6f}")
        print(f"  p50:   {p_sell_cal.quantile(0.50):.6f}")
        print(f"  p90:   {p_sell_cal.quantile(0.90):.6f}")
        print(f"  p99:   {p_sell_cal.quantile(0.99):.6f}")

        # Correlación
        corr = p_buy_cal.corr(p_sell_cal)
        print(f"\n📈 Correlación p_buy vs p_sell: {corr:.4f}")

        # ¿Ambas altas simultáneamente?
        both_high = ((p_buy_cal > 0.5) & (p_sell_cal > 0.5)).sum()
        pct_both_high = 100 * both_high / len(df_merged)
        print(f"\n⚠️  Ambas >0.5 simultáneamente: {both_high} ({pct_both_high:.1f}%)")

        # ¿Alguna baja?
        any_low = ((p_buy_cal < 0.3) | (p_sell_cal < 0.3)).sum()
        pct_any_low = 100 * any_low / len(df_merged)
        print(f"✅ Al menos una <0.3: {any_low} ({pct_any_low:.1f}%)")

    else:
        print("⚠️  No se pudieron hacer merge de los DataFrames por tiempo")


def analyze_probs_distribution(df: pd.DataFrame, raw_col: str, cal_col: str):
    """Analiza distribución de probabilidades raw y calibradas."""

    # Filtrar NaNs
    mask = df[cal_col].notna()
    p_raw = df.loc[mask, raw_col]
    p_cal = df.loc[mask, cal_col]

    print(f"Samples válidos: {len(p_cal)}")

    print(f"\n{raw_col} (RAW):")
    print(f"  mean:  {p_raw.mean():.6f}")
    print(f"  std:   {p_raw.std():.6f}")
    print(f"  min:   {p_raw.min():.6f}")
    print(f"  max:   {p_raw.max():.6f}")
    print(f"  p01:   {p_raw.quantile(0.01):.6f}")
    print(f"  p50:   {p_raw.quantile(0.50):.6f}")
    print(f"  p99:   {p_raw.quantile(0.99):.6f}")

    print(f"\n{cal_col} (CALIBRADO):")
    print(f"  mean:  {p_cal.mean():.6f}")
    print(f"  std:   {p_cal.std():.6f}")
    print(f"  min:   {p_cal.min():.6f}")
    print(f"  max:   {p_cal.max():.6f}")
    print(f"  p01:   {p_cal.quantile(0.01):.6f}")
    print(f"  p50:   {p_cal.quantile(0.50):.6f}")
    print(f"  p99:   {p_cal.quantile(0.99):.6f}")

    # Bins
    bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    hist, _ = np.histogram(p_cal, bins=bins)
    print(f"\nHistograma (calibrado):")
    for i in range(len(bins) - 1):
        pct = 100 * hist[i] / len(p_cal)
        bar = "█" * int(pct / 2)
        print(f"  [{bins[i]:.1f}-{bins[i + 1]:.1f}): {hist[i]:6d} ({pct:5.1f}%) {bar}")


if __name__ == "__main__":
    analyze_oof_probabilities(ARTIFACTS_PATH, RELEASE)
