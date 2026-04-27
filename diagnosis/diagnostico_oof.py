#!/usr/bin/env python3
"""
Análisis del calibrador isotónico para entender por qué el sistema funciona
a pesar de predicciones ultra-bajas
"""
import joblib
import numpy as np
import json

# Cargar calibrador
cal_path = '../artifacts/200312/oof/train_only/oof_calibrator_200312_long.joblib'
calibrator = joblib.load(cal_path)

print("=" * 80)
print("ANÁLISIS DEL CALIBRADOR ISOTÓNICO")
print("=" * 80)

print(f"\nTipo: {type(calibrator).__name__}")

# Rangos de entrada (X)
print(f"\n📊 RANGO DE ENTRADA (X - predicciones raw del modelo):")
print(f"  X_min: {calibrator.X_min_:.10f}")
print(f"  X_max: {calibrator.X_max_:.10f}")

# Para la salida, usar los thresholds
if hasattr(calibrator, 'y_thresholds_'):
    y_min = calibrator.y_thresholds_.min()
    y_max = calibrator.y_thresholds_.max()
    print(f"\n📊 RANGO DE SALIDA (y - predicciones calibradas):")
    print(f"  y_min: {y_min:.10f}")
    print(f"  y_max: {y_max:.10f}")

    # Análisis crítico
    if calibrator.X_max_ < 0.01:
        print(f"\n  ⚠️  CRÍTICO: X_max muy bajo ({calibrator.X_max_:.6f})")
        print(f"     → El modelo solo genera predicciones < 0.01")
        print(f"     → El modelo está colapsado")

    if y_max > 0.1:
        print(f"\n  ✓ RESCATE: El calibrador mapea valores bajos a valores razonables")
        print(f"     → X_max={calibrator.X_max_:.6f} → y_max={y_max:.6f}")
        print(f"     → Factor de amplificación: {y_max / calibrator.X_max_:.1f}x")

# Test con el valor que vimos en la predicción
print(f"\n" + "=" * 80)
print("TEST CON VALORES ULTRA-BAJOS")
print("=" * 80)

test_values = [
    0.0000001,
    0.0000005,
    0.0000009518,  # ⬅️ El valor exacto que vimos
    0.000001,
    0.000005,
    0.00001,
    0.00005,
    0.0001,
    0.0005,
    0.001,
    0.005,
    0.01,
]

print(f"\nMapeo de predicciones raw → calibradas:")
for val in test_values:
    try:
        cal_val = calibrator.predict([val])[0]
        ratio = cal_val / val if val > 0 else 0
        print(f"  raw={val:.10f} → cal={cal_val:.6f}  (amplificación: {ratio:.1f}x)")
    except Exception as e:
        print(f"  raw={val:.10f} → ERROR: {e}")

# Ver la función de mapeo completa (primeros y últimos puntos)
print(f"\n" + "=" * 80)
print("FUNCIÓN DE MAPEO COMPLETA")
print("=" * 80)

if hasattr(calibrator, 'X_thresholds_') and hasattr(calibrator, 'y_thresholds_'):
    n_points = len(calibrator.X_thresholds_)
    print(f"\nTotal de puntos en la función: {n_points}")

    print(f"\n📊 Primeros 20 puntos (valores más bajos):")
    for i in range(min(20, n_points)):
        x = calibrator.X_thresholds_[i]
        y = calibrator.y_thresholds_[i]
        ratio = y / x if x > 0 else 0
        print(f"  {i + 1:3d}. X={x:.10f} → Y={y:.6f} (amplif: {ratio:.1f}x)")

    print(f"\n📊 Últimos 10 puntos (valores más altos):")
    for i in range(max(0, n_points - 10), n_points):
        x = calibrator.X_thresholds_[i]
        y = calibrator.y_thresholds_[i]
        ratio = y / x if x > 0 else 0
        print(f"  {i + 1:3d}. X={x:.10f} → Y={y:.6f} (amplif: {ratio:.1f}x)")

# Estadísticas de la amplificación
print(f"\n" + "=" * 80)
print("ESTADÍSTICAS DE AMPLIFICACIÓN")
print("=" * 80)

if hasattr(calibrator, 'X_thresholds_') and hasattr(calibrator, 'y_thresholds_'):
    X = calibrator.X_thresholds_
    Y = calibrator.y_thresholds_

    # Evitar división por 0
    mask = X > 1e-15
    ratios = Y[mask] / X[mask]

    print(f"\nFactores de amplificación (Y/X):")
    print(f"  Min:    {ratios.min():.1f}x")
    print(f"  Max:    {ratios.max():.1f}x")
    print(f"  Mean:   {ratios.mean():.1f}x")
    print(f"  Median: {np.median(ratios):.1f}x")
    print(f"  Std:    {ratios.std():.1f}x")

    # Rangos de amplificación
    print(f"\nDistribución de amplificación:")
    for low, high in [(1, 10), (10, 100), (100, 1000), (1000, 10000), (10000, float('inf'))]:
        count = ((ratios >= low) & (ratios < high)).sum()
        pct = count / len(ratios) * 100
        print(f"  {low:5d}x - {high:5.0f}x: {count:4d} puntos ({pct:5.1f}%)")

# Cargar percentiles para contexto
print(f"\n" + "=" * 80)
print("PERCENTILES POR RÉGIMEN")
print("=" * 80)

pct_path = '../artifacts/200312/oof/train_only/percentiles_200312_long.json'
try:
    with open(pct_path, 'r') as f:
        percentiles = json.load(f)

    print(f"\nPercentiles cargados desde: {pct_path}")

    # Mostrar para cada régimen
    for regime, pcts in percentiles.items():
        if regime == '_meta':
            continue
        print(f"\n  {regime}:")
        if isinstance(pcts, dict):
            for p, val in sorted(pcts.items(), key=lambda x: float(x[0]) if x[0] != '_meta' else 0):
                if p != '_meta':
                    print(f"    P{p}: {val:.6f}")
except Exception as e:
    print(f"  ⚠️  No se pudo cargar percentiles: {e}")

print(f"\n" + "=" * 80)
print("CONCLUSIÓN")
print("=" * 80)

if hasattr(calibrator, 'X_thresholds_'):
    X_max = calibrator.X_max_
    y_max = calibrator.y_thresholds_.max()

    if X_max < 0.01 and y_max > 0.1:
        print(f"\n✓ CONFIRMADO: El calibrador está rescatando al modelo")
        print(f"  • Predicciones raw del modelo: muy bajas (max={X_max:.6f})")
        print(f"  • Predicciones calibradas: razonables (max={y_max:.6f})")
        print(f"  • El sistema funciona GRACIAS al calibrador, NO al modelo base")
        print(f"\n⚠️  RECOMENDACIÓN:")
        print(f"  • Reentrenar el modelo con bias limitado")
        print(f"  • El calibrador NO debería ser necesario para salvar predicciones")
        print(f"  • Un modelo bien entrenado + calibración = mejor performance")
    elif X_max > 0.1:
        print(f"\n✓ El modelo genera predicciones razonables")
        print(f"  • X_max = {X_max:.6f}")
        print(f"  • El calibrador hace ajustes finos, no rescates")
    else:
        print(f"\n⚠️  Estado ambiguo, revisar métricas OOF")

print("\n" + "=" * 80)
