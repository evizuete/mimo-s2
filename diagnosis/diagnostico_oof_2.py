import pandas as pd

# Cargar el CSV de predicciones
df = pd.read_csv('df_pred.csv')

print("=" * 80)
print("ANÁLISIS POR RÉGIMEN")
print("=" * 80)

for regime in df['macro_regime'].unique():
    subset = df[df['macro_regime'] == regime]
    print(f"\n{regime.upper()}:")
    print(f"  Filas: {len(subset)}")
    print(f"  pred_long_raw:")
    print(f"    Min:    {subset['pred_long_raw'].min():.10f}")
    print(f"    Max:    {subset['pred_long_raw'].max():.10f}")
    print(f"    Mean:   {subset['pred_long_raw'].mean():.10f}")
    print(f"    Median: {subset['pred_long_raw'].median():.10f}")

print("\n" + "=" * 80)
print("ANÁLISIS POR STATE")
print("=" * 80)

for state in df['state'].unique():
    subset = df[df['state'] == state]
    print(f"\n{state}:")
    print(f"  Filas: {len(subset)}")
    print(f"  pred_long_raw:")
    print(f"    Mean:   {subset['pred_long_raw'].mean():.10f}")
    print(f"    Max:    {subset['pred_long_raw'].max():.10f}")