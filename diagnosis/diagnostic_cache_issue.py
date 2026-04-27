# diagnostic_cache_issue.py - Diagnóstico de Múltiples Archivos de Caché

"""
Script para diagnosticar por qué se están generando múltiples archivos de caché
cuando debería ser solo uno.

PROBLEMA REPORTADO:
- Se generan múltiples archivos de caché para una única simulación
- Debería ser un solo archivo por release+fechas

CAUSAS POSIBLES:
1. El DataFrame se está dividiendo en chunks con diferentes fechas
2. Se está llamando predict() múltiples veces con diferentes rangos de fechas
3. Las fechas están cambiando entre llamadas (problema de timezone o formato)
4. El método load_from_database_historical está devolviendo datos fragmentados
"""
import joblib
import pandas as pd
from pathlib import Path
import json
import pickle


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN 1: Analizar archivos de caché existentes
# ═══════════════════════════════════════════════════════════════════════

def analyze_cache_files(cache_dir='./cache/predictions'):
    """
    Analiza los archivos de caché para ver qué contienen
    """
    cache_path = Path(cache_dir)

    if not cache_path.exists():
        print(f"❌ Directorio de caché no existe: {cache_dir}")
        return

    files = list(cache_path.glob("pred_*.pkl"))

    if not files:
        print(f"⚠️  No hay archivos de caché en {cache_dir}")
        return

    print(f"\n{'=' * 70}")
    print(f"ANÁLISIS DE ARCHIVOS DE CACHÉ")
    print(f"{'=' * 70}")
    print(f"Directorio: {cache_dir}")
    print(f"Total archivos: {len(files)}\n")

    file_info = []

    for i, filepath in enumerate(sorted(files), 1):
        try:
            # Cargar archivo
            with open(filepath, 'rb') as f:
                df = joblib.load(f)

            # Extraer información
            size_mb = filepath.stat().st_size / (1024 * 1024)

            # Fechas del DataFrame
            if isinstance(df.index, pd.DatetimeIndex):
                first_date = df.index[0].date()
                last_date = df.index[-1].date()
            elif 'time' in df.columns:
                first_date = pd.to_datetime(df['time'].iloc[0]).date()
                last_date = pd.to_datetime(df['time'].iloc[-1]).date()
            else:
                first_date = "N/A"
                last_date = "N/A"

            info = {
                'file': filepath.name,
                'size_mb': size_mb,
                'n_rows': len(df),
                'first_date': str(first_date),
                'last_date': str(last_date),
                'columns': list(df.columns)
            }
            file_info.append(info)

            # Mostrar
            print(f"[{i}] {filepath.name}")
            print(f"    Size: {size_mb:.2f} MB")
            print(f"    Rows: {len(df)}")
            print(f"    Date range: {first_date} → {last_date}")

            # Detectar solapamiento
            if i > 1:
                prev_info = file_info[i - 2]
                if prev_info['last_date'] >= info['first_date']:
                    print(f"    ⚠️  SOLAPAMIENTO detectado con archivo anterior!")

            print()

        except Exception as e:
            print(f"[{i}] {filepath.name}")
            print(f"    ❌ Error al leer: {e}\n")

    # Resumen
    print(f"{'=' * 70}")
    print(f"RESUMEN")
    print(f"{'=' * 70}")

    if len(file_info) > 1:
        # Buscar duplicados por rango de fechas
        date_ranges = {}
        for info in file_info:
            key = f"{info['first_date']}_{info['last_date']}"
            if key not in date_ranges:
                date_ranges[key] = []
            date_ranges[key].append(info['file'])

        duplicates = {k: v for k, v in date_ranges.items() if len(v) > 1}

        if duplicates:
            print("⚠️  DUPLICADOS DETECTADOS:")
            for date_range, files in duplicates.items():
                print(f"\n  Rango de fechas: {date_range}")
                print(f"  Archivos ({len(files)}):")
                for f in files:
                    print(f"    - {f}")
        else:
            print("✅ No hay duplicados exactos")
            print("\n⚠️  Pero hay múltiples archivos con diferentes rangos:")
            for info in file_info:
                print(f"  - {info['first_date']} → {info['last_date']} ({info['n_rows']} rows)")

    return file_info


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN 2: Interceptar llamadas a predict para ver qué fechas se pasan
# ═══════════════════════════════════════════════════════════════════════

def add_predict_logging(simulator):
    """
    Añade logging al método predict para ver qué fechas se están usando

    Uso:
        sim = TradingSimulator(...)
        add_predict_logging(sim)
        # Ahora cada llamada a predict logeará las fechas
    """
    original_predict = simulator.predict
    call_count = [0]  # Usar lista para poder modificar en closure

    def logged_predict(df_rates, simulation=True, verbose=False):
        call_count[0] += 1

        # Extraer fechas
        if isinstance(df_rates.index, pd.DatetimeIndex):
            first_date = df_rates.index[0].date()
            last_date = df_rates.index[-1].date()
        elif 'time' in df_rates.columns:
            first_date = pd.to_datetime(df_rates['time'].iloc[0]).date()
            last_date = pd.to_datetime(df_rates['time'].iloc[-1]).date()
        else:
            first_date = "N/A"
            last_date = "N/A"

        print(f"\n[PREDICT CALL #{call_count[0]}]")
        print(f"  Simulation: {simulation}")
        print(f"  Date range: {first_date} → {last_date}")
        print(f"  Rows: {len(df_rates)}")

        # Llamar al original
        result = original_predict(df_rates, simulation, verbose)

        print(f"  Result rows: {len(result)}")

        return result

    simulator.predict = logged_predict
    print("✅ Logging añadido a predict(). Cada llamada mostrará las fechas.")


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN 3: Verificar si el problema está en load_from_database_historical
# ═══════════════════════════════════════════════════════════════════════

def check_database_loading(helper, from_date, to_date):
    """
    Verifica si load_from_database_historical devuelve datos consistentes
    """
    print(f"\n{'=' * 70}")
    print(f"VERIFICACIÓN DE CARGA DE DATOS")
    print(f"{'=' * 70}")
    print(f"Rango solicitado: {from_date} → {to_date}\n")

    # Cargar 3 veces y comparar
    dfs = []
    for i in range(3):
        df = helper.load_from_database_historical(from_date=from_date, to_date=to_date)

        if isinstance(df.index, pd.DatetimeIndex):
            first = df.index[0].date()
            last = df.index[-1].date()
        elif 'time' in df.columns:
            first = pd.to_datetime(df['time'].iloc[0]).date()
            last = pd.to_datetime(df['time'].iloc[-1]).date()
        else:
            first = last = "N/A"

        print(f"Carga #{i + 1}:")
        print(f"  Filas: {len(df)}")
        print(f"  Fechas: {first} → {last}")

        dfs.append(df)

    # Comparar
    print(f"\n{'=' * 70}")
    if len(dfs[0]) == len(dfs[1]) == len(dfs[2]):
        print("✅ Las 3 cargas devuelven el mismo número de filas")
    else:
        print("⚠️  Las cargas devuelven diferentes números de filas!")
        print(f"   Carga 1: {len(dfs[0])} filas")
        print(f"   Carga 2: {len(dfs[1])} filas")
        print(f"   Carga 3: {len(dfs[2])} filas")

    # Verificar fechas
    for i in range(3):
        if isinstance(dfs[i].index, pd.DatetimeIndex):
            first_i = dfs[i].index[0]
            last_i = dfs[i].index[-1]
        elif 'time' in dfs[i].columns:
            first_i = pd.to_datetime(dfs[i]['time'].iloc[0])
            last_i = pd.to_datetime(dfs[i]['time'].iloc[-1])

        if i > 0:
            if isinstance(dfs[i - 1].index, pd.DatetimeIndex):
                first_prev = dfs[i - 1].index[0]
                last_prev = dfs[i - 1].index[-1]
            else:
                first_prev = pd.to_datetime(dfs[i - 1]['time'].iloc[0])
                last_prev = pd.to_datetime(dfs[i - 1]['time'].iloc[-1])

            if first_i != first_prev or last_i != last_prev:
                print(f"⚠️  Carga #{i + 1} tiene fechas diferentes a carga #{i}")


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN 4: Solución - Limpiar duplicados
# ═══════════════════════════════════════════════════════════════════════

def clean_duplicate_caches(cache_dir='./cache/predictions', dry_run=True):
    """
    Limpia archivos de caché duplicados, dejando solo el más reciente

    Args:
        cache_dir: Directorio del caché
        dry_run: Si True, solo muestra qué haría sin borrar
    """
    cache_path = Path(cache_dir)
    files = list(cache_path.glob("pred_*.pkl"))

    # Agrupar por rango de fechas
    by_date_range = {}

    for filepath in files:
        try:
            with open(filepath, 'rb') as f:
                df = joblib.load(f)

            if isinstance(df.index, pd.DatetimeIndex):
                first_date = str(df.index[0].date())
                last_date = str(df.index[-1].date())
            elif 'time' in df.columns:
                first_date = str(pd.to_datetime(df['time'].iloc[0]).date())
                last_date = str(pd.to_datetime(df['time'].iloc[-1]).date())
            else:
                continue

            key = f"{first_date}_{last_date}"

            if key not in by_date_range:
                by_date_range[key] = []

            by_date_range[key].append({
                'path': filepath,
                'mtime': filepath.stat().st_mtime,
                'size': filepath.stat().st_size
            })
        except:
            pass

    # Encontrar duplicados
    print(f"\n{'=' * 70}")
    print(f"LIMPIEZA DE DUPLICADOS")
    print(f"{'=' * 70}")
    print(f"Modo: {'DRY RUN (no se borrará nada)' if dry_run else 'REAL (se borrarán archivos)'}\n")

    duplicates_found = False
    total_to_delete = 0

    for date_range, files_info in by_date_range.items():
        if len(files_info) > 1:
            duplicates_found = True
            print(f"📅 Rango: {date_range}")
            print(f"   Duplicados: {len(files_info)} archivos\n")

            # Ordenar por fecha de modificación (más reciente primero)
            files_info.sort(key=lambda x: x['mtime'], reverse=True)

            # Mantener el más reciente, borrar los demás
            keep = files_info[0]
            to_delete = files_info[1:]

            print(f"   ✅ MANTENER: {keep['path'].name}")
            print(f"      Modified: {pd.Timestamp.fromtimestamp(keep['mtime'])}")
            print(f"      Size: {keep['size'] / (1024 * 1024):.2f} MB\n")

            for item in to_delete:
                print(f"   ❌ BORRAR: {item['path'].name}")
                print(f"      Modified: {pd.Timestamp.fromtimestamp(item['mtime'])}")
                print(f"      Size: {item['size'] / (1024 * 1024):.2f} MB")

                if not dry_run:
                    item['path'].unlink()
                    print(f"      → Borrado ✓")

                total_to_delete += 1
                print()

    if not duplicates_found:
        print("✅ No se encontraron duplicados")
    else:
        action = "se borrarían" if dry_run else "se borraron"
        print(f"{'=' * 70}")
        print(f"Total: {action} {total_to_delete} archivos duplicados")
        print(f"{'=' * 70}")

        if dry_run:
            print("\n💡 Para borrar realmente, ejecuta: clean_duplicate_caches(dry_run=False)")


# ═══════════════════════════════════════════════════════════════════════
# FUNCIÓN 5: Script completo de diagnóstico
# ═══════════════════════════════════════════════════════════════════════

def full_diagnostic(release='200343', cache_dir=None):
    """
    Ejecuta diagnóstico completo
    """
    if cache_dir is None:
        cache_dir = f'./cache/{release}/predictions'

    print(f"\n{'█' * 70}")
    print(f"DIAGNÓSTICO COMPLETO DEL CACHÉ")
    print(f"{'█' * 70}")
    print(f"Release: {release}")
    print(f"Cache dir: {cache_dir}")

    # Analizar archivos
    file_info = analyze_cache_files(cache_dir)

    if file_info and len(file_info) > 1:
        print(f"\n{'=' * 70}")
        print(f"DIAGNÓSTICO")
        print(f"{'=' * 70}")

        # Verificar solapamientos
        has_overlap = False
        for i in range(1, len(file_info)):
            prev = file_info[i - 1]
            curr = file_info[i]

            if prev['last_date'] >= curr['first_date']:
                has_overlap = True
                print(f"\n⚠️  Solapamiento detectado:")
                print(f"   {prev['file']}: {prev['first_date']} → {prev['last_date']}")
                print(f"   {curr['file']}: {curr['first_date']} → {curr['last_date']}")

        # Verificar si son exactamente iguales
        date_ranges = {}
        for info in file_info:
            key = f"{info['first_date']}_{info['last_date']}"
            if key not in date_ranges:
                date_ranges[key] = []
            date_ranges[key].append(info)

        exact_dupes = {k: v for k, v in date_ranges.items() if len(v) > 1}

        if exact_dupes:
            print(f"\n⚠️  DUPLICADOS EXACTOS encontrados:")
            for date_range, infos in exact_dupes.items():
                print(f"\n   Rango: {date_range}")
                print(f"   Archivos: {len(infos)}")
                for info in infos:
                    print(f"     - {info['file']} ({info['n_rows']} rows, {info['size_mb']:.2f} MB)")

            print(f"\n💡 CAUSA PROBABLE:")
            print(f"   Se está llamando a predict() múltiples veces con los mismos datos")
            print(f"   pero el hash del caché está generando keys diferentes.")

            print(f"\n🔧 SOLUCIÓN:")
            print(f"   Ejecutar: clean_duplicate_caches(cache_dir='{cache_dir}', dry_run=False)")

        elif has_overlap:
            print(f"\n💡 CAUSA PROBABLE:")
            print(f"   El DataFrame se está dividiendo en chunks con fechas solapadas.")
            print(f"   Posiblemente en el método load_from_database_historical.")

        else:
            print(f"\n💡 CAUSA PROBABLE:")
            print(f"   Se están procesando diferentes rangos de fechas en la misma ejecución.")
            print(f"   Esto es normal si estás haciendo TRAIN + VAL + TEST en una sola ejecución.")

            # Mostrar rangos
            print(f"\n   Rangos encontrados:")
            for info in file_info:
                print(f"     - {info['first_date']} → {info['last_date']} ({info['n_rows']} rows)")





# ═══════════════════════════════════════════════════════════════════════
# USO DEL SCRIPT
# ═══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys

    # Detectar release del directorio actual
    cache_dirs = list(Path('.././cache').glob('*/predictions'))

    if not cache_dirs:
        print("❌ No se encontró directorio de caché en ./cache/*/predictions")
        print("💡 Especifica manualmente: python diagnostic_cache_issue.py <cache_dir>")
        sys.exit(1)

    if len(sys.argv) > 1:
        cache_dir = sys.argv[1]
    else:
        cache_dir = str(cache_dirs[0])
        print(f"📁 Usando directorio de caché: {cache_dir}\n")

    # Ejecutar diagnóstico
    full_diagnostic(cache_dir=cache_dir)

    print("\n" + "=" * 70)
    print("COMANDOS ÚTILES")
    print("=" * 70)
    print()
    print("# Ver detalles de archivos:")
    print(f"python -c \"from diagnostic_cache_issue import *; analyze_cache_files('{cache_dir}')\"")
    print()
    print("# Limpiar duplicados (DRY RUN):")
    print(f"python -c \"from diagnostic_cache_issue import *; clean_duplicate_caches('{cache_dir}', dry_run=True)\"")
    print()
    print("# Limpiar duplicados (REAL):")
    print(f"python -c \"from diagnostic_cache_issue import *; clean_duplicate_caches('{cache_dir}', dry_run=False)\"")
    print()