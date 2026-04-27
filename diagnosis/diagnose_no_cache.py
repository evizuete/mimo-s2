# diagnostic_no_cache.py - Diagnóstico de Caché que no se Genera

"""
Script para diagnosticar por qué no se están generando archivos de caché
"""

import sys
from pathlib import Path


def check_cache_setup():
    """Verifica la configuración del sistema de caché"""

    print("=" * 70)
    print("DIAGNÓSTICO: CACHÉ NO SE GENERA")
    print("=" * 70)

    issues = []

    # 1. Verificar que prediction_cache.py existe
    print("\n[1] Verificando módulo prediction_cache.py...")
    try:
        from mimo.strategies.prediction_cache import PredictionCache
        print("    ✅ Módulo importado correctamente")
    except ImportError as e:
        print(f"    ❌ Error importando: {e}")
        issues.append("prediction_cache.py no está en mimo_old/")
        return issues

    # 2. Verificar que TradingSimulator tiene los parámetros
    print("\n[2] Verificando parámetros en TradingSimulator...")
    try:
        from mimo_old.trading_simulator import TradingSimulator
        import inspect

        sig = inspect.signature(TradingSimulator.__init__)

        if 'use_prediction_cache' not in sig.parameters:
            print("    ❌ Falta parámetro: use_prediction_cache")
            issues.append("TradingSimulator.__init__ no tiene use_prediction_cache")
        else:
            print("    ✅ Parámetro use_prediction_cache existe")

        if 'cache_dir' not in sig.parameters:
            print("    ❌ Falta parámetro: cache_dir")
            issues.append("TradingSimulator.__init__ no tiene cache_dir")
        else:
            print("    ✅ Parámetro cache_dir existe")

    except Exception as e:
        print(f"    ❌ Error verificando TradingSimulator: {e}")
        issues.append(str(e))
        return issues

    # 3. Verificar que predict tiene la lógica de caché
    print("\n[3] Verificando método predict...")
    try:
        import inspect
        source = inspect.getsource(TradingSimulator.predict)

        if 'use_prediction_cache' in source:
            print("    ✅ predict tiene lógica de caché")
        else:
            print("    ❌ predict NO tiene lógica de caché")
            issues.append("predict no fue modificado con la lógica de caché")

        if 'prediction_cache.get' in source:
            print("    ✅ predict llama a prediction_cache.get()")
        else:
            print("    ❌ predict NO llama a prediction_cache.get()")
            issues.append("predict no llama a get() del caché")

        if 'prediction_cache.put' in source:
            print("    ✅ predict llama a prediction_cache.put()")
        else:
            print("    ❌ predict NO llama a prediction_cache.put()")
            issues.append("predict no llama a put() del caché")

    except Exception as e:
        print(f"    ❌ Error inspeccionando predict: {e}")
        issues.append(str(e))

    # 4. Test de creación de TradingSimulator con caché
    print("\n[4] Test de creación de simulator con caché...")
    try:
        from mimo_old.model_builder import Config, ModelConfig
        from mimo_old.feature_builder import FeatureConfig
        from mimo.strategies.regime_detector import RegimeConfig
        from mimo_old.decision_engine import DecisionPolicy, RiskConfig

        sim = TradingSimulator(
            general_config=Config(release='test'),
            model_config=ModelConfig(),
            feature_config=FeatureConfig(),
            regime_config=RegimeConfig(),
            decision_policy=DecisionPolicy(),
            risk_config=RiskConfig(),
            use_prediction_cache=True,
            cache_dir='./test_cache'
        )

        if not hasattr(sim, 'use_prediction_cache'):
            print("    ❌ sim.use_prediction_cache no existe")
            issues.append("Atributo use_prediction_cache no se crea en __init__")
        elif not sim.use_prediction_cache:
            print("    ❌ sim.use_prediction_cache = False (debería ser True)")
            issues.append("use_prediction_cache no se inicializa correctamente")
        else:
            print(f"    ✅ sim.use_prediction_cache = {sim.use_prediction_cache}")

        if not hasattr(sim, 'prediction_cache'):
            print("    ❌ sim.prediction_cache no existe")
            issues.append("Atributo prediction_cache no se crea en __init__")
        else:
            print(f"    ✅ sim.prediction_cache existe")
            print(f"       Type: {type(sim.prediction_cache)}")
            print(f"       Enabled: {sim.prediction_cache.enabled}")

    except Exception as e:
        print(f"    ❌ Error creando simulator: {e}")
        issues.append(str(e))

    # 5. Verificar directorio de caché
    print("\n[5] Verificando directorio de caché...")
    cache_dirs = list(Path('./cache').glob('*/predictions'))

    if not cache_dirs:
        print("    ⚠️  No hay directorios de caché en ./cache/*/predictions")
        print("       (Normal si nunca se ha ejecutado)")
    else:
        for cache_dir in cache_dirs:
            files = list(cache_dir.glob("pred_*.pkl"))
            print(f"    📁 {cache_dir}")
            print(f"       Archivos: {len(files)}")

    # Resumen
    print("\n" + "=" * 70)
    print("RESUMEN")
    print("=" * 70)

    if not issues:
        print("✅ NO SE DETECTARON PROBLEMAS DE CONFIGURACIÓN")
        print("\nPosibles causas si aún no se genera caché:")
        print("  1. simulation=False en llamadas a predict")
        print("  2. use_prediction_cache=False al crear TradingSimulator")
        print("  3. Errores silenciosos (ver siguiente paso)")
    else:
        print("❌ PROBLEMAS DETECTADOS:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")

    return issues


def test_cache_creation():
    """Test real de creación de caché"""

    print("\n" + "=" * 70)
    print("TEST REAL DE CREACIÓN DE CACHÉ")
    print("=" * 70)

    try:
        from mimo_old.trading_simulator import TradingSimulator
        from mimo_old.model_builder import Config, ModelConfig
        from mimo_old.feature_builder import FeatureConfig
        from mimo.strategies.regime_detector import RegimeConfig
        from mimo_old.decision_engine import DecisionPolicy, RiskConfig
        import pandas as pd
        import numpy as np
        from datetime import datetime, timedelta

        print("\n[1] Creando simulator con caché habilitado...")
        sim = TradingSimulator(
            general_config=Config(release='test_cache'),
            model_config=ModelConfig(),
            feature_config=FeatureConfig(),
            regime_config=RegimeConfig(),
            decision_policy=DecisionPolicy(),
            risk_config=RiskConfig(),
            use_prediction_cache=True,
            cache_dir='./test_cache_diagnostic'
        )
        print(f"    ✅ Simulator creado")
        print(f"       use_prediction_cache: {sim.use_prediction_cache}")
        print(f"       cache enabled: {sim.prediction_cache.enabled}")
        print(f"       cache dir: {sim.prediction_cache.cache_dir}")

        print("\n[2] Creando DataFrame de prueba...")
        # Crear datos dummy
        dates = pd.date_range(start='2025-01-01', end='2025-01-31', freq='1H')
        df_test = pd.DataFrame({
            'time': dates,
            'open': np.random.randn(len(dates)) + 100,
            'high': np.random.randn(len(dates)) + 101,
            'low': np.random.randn(len(dates)) + 99,
            'close': np.random.randn(len(dates)) + 100,
            'volume': np.random.randint(1000, 10000, len(dates))
        })
        df_test.set_index('time', inplace=True)
        print(f"    ✅ DataFrame creado: {len(df_test)} filas")
        print(f"       Rango: {df_test.index[0]} → {df_test.index[-1]}")

        print("\n[3] Intentando llamar a predict...")
        print("    NOTA: Esto fallará si no tienes artifacts, pero debería intentar cachear")

        try:
            df_pred = sim.predict(df_test, simulation=True, verbose=False)
            print(f"    ✅ predict() ejecutado exitosamente")
            print(f"       Resultado: {len(df_pred)} filas")
        except Exception as e:
            print(f"    ⚠️  predict() falló (esperado si no hay artifacts)")
            print(f"       Error: {str(e)[:100]}...")

        print("\n[4] Verificando si se creó archivo de caché...")
        cache_dir = Path('./test_cache_diagnostic')

        if cache_dir.exists():
            files = list(cache_dir.glob("pred_*.pkl"))
            print(f"    📁 Directorio existe: {cache_dir}")
            print(f"       Archivos: {len(files)}")

            if files:
                print("    ✅ CACHÉ FUNCIONANDO - Se crearon archivos!")
                for f in files:
                    print(f"       - {f.name}")
            else:
                print("    ❌ NO SE CREARON ARCHIVOS DE CACHÉ")
                print("\n    CAUSA PROBABLE:")
                print("    - predict() falló antes de llegar al put()")
                print("    - Verificar que simulation=True")
                print("    - Verificar try-except que esté ocultando errores")
        else:
            print(f"    ❌ Directorio de caché NO EXISTE: {cache_dir}")
            print("       El caché está DESHABILITADO o hubo error en creación")

        # Verificar stats del caché
        print("\n[5] Estadísticas del caché...")
        stats = sim.prediction_cache.get_stats()
        print(f"    Enabled: {stats['enabled']}")
        print(f"    Hits: {stats['hits']}")
        print(f"    Misses: {stats['misses']}")
        print(f"    Files: {stats['n_files']}")

    except Exception as e:
        print(f"\n❌ ERROR EN TEST: {e}")
        import traceback
        traceback.print_exc()


def show_predict_calls():
    """Muestra cómo debería verse una llamada a predict con logging"""

    print("\n" + "=" * 70)
    print("CÓMO AÑADIR LOGGING PARA VER QUÉ PASA")
    print("=" * 70)

    code = '''
# Añade esto al inicio de tu script de entrenamiento
import logging
logging.basicConfig(level=logging.DEBUG)

# O añade prints en prediction_cache.py
# En el método get():
def get(self, release: str, from_date: str, to_date: str, n_rows: int = None):
    print(f"[Cache.get] Buscando: {release}_{from_date}_{to_date}")  # AÑADIR

    if not self.enabled:
        print(f"[Cache.get] Cache DISABLED")  # AÑADIR
        return None

    key = self._get_key(release, from_date, to_date, n_rows)
    print(f"[Cache.get] Key: {key[:8]}...")  # AÑADIR

    filepath = self._get_filepath(key)
    print(f"[Cache.get] Filepath: {filepath}")  # AÑADIR

    if not filepath.exists():
        print(f"[Cache.get] MISS - archivo no existe")  # AÑADIR
        self.misses += 1
        return None
    ...

# En el método put():
def put(self, release: str, from_date: str, to_date: str, df_predictions, n_rows=None):
    print(f"[Cache.put] Guardando: {release}_{from_date}_{to_date}")  # AÑADIR

    if not self.enabled:
        print(f"[Cache.put] Cache DISABLED - no se guarda")  # AÑADIR
        return

    key = self._get_key(release, from_date, to_date, n_rows)
    print(f"[Cache.put] Key: {key[:8]}...")  # AÑADIR

    filepath = self._get_filepath(key)
    print(f"[Cache.put] Guardando en: {filepath}")  # AÑADIR
    ...
'''

    print(code)


def check_common_issues():
    """Verifica problemas comunes"""

    print("\n" + "=" * 70)
    print("PROBLEMAS COMUNES - CHECKLIST")
    print("=" * 70)

    print("""
    □ 1. use_prediction_cache=False en creación de TradingSimulator
         Verificar en tu script:
         sim = TradingSimulator(..., use_prediction_cache=True)  # ← Debe ser True

    □ 2. simulation=False en llamadas a predict
         Verificar:
         df_pred = sim.predict(df, simulation=True)  # ← Debe ser True

    □ 3. Errores silenciosos en try-except
         En trading_simulator.py, método predict, buscar:
         try:
             self.prediction_cache.put(...)
         except Exception as e:
             pass  # ← Puede estar ocultando errores

         Cambiar a:
         except Exception as e:
             print(f"[Cache] ERROR guardando: {e}")  # ← Ver el error

    □ 4. from_date/to_date son None
         En predict(), verificar que se están extrayendo las fechas:
         print(f"[DEBUG] from_date={from_date}, to_date={to_date}")

    □ 5. DataFrame vacío o sin columna 'time'
         Verificar estructura del DataFrame:
         print(f"[DEBUG] df_rates.columns={df_rates.columns}")
         print(f"[DEBUG] df_rates.index type={type(df_rates.index)}")

    □ 6. Permisos de escritura en directorio
         Verificar:
         import os
         cache_dir = './cache/200343/predictions'
         os.makedirs(cache_dir, exist_ok=True)
         test_file = f'{cache_dir}/test.txt'
         with open(test_file, 'w') as f:
             f.write('test')
         os.remove(test_file)
         print(f"✅ Permisos OK en {cache_dir}")
    """)


if __name__ == "__main__":
    # Ejecutar todos los diagnósticos
    issues = check_cache_setup()

    if not issues:
        print("\n" + "=" * 70)
        print("¿Ejecutar test real de creación de caché? (s/n)")
        print("=" * 70)

        if len(sys.argv) > 1 and sys.argv[1] == '--test':
            test_cache_creation()
        else:
            print("Para ejecutar test: python diagnostic_no_cache.py --test")

    show_predict_calls()
    check_common_issues()

    print("\n" + "=" * 70)
    print("SIGUIENTE PASO")
    print("=" * 70)
    print("""
    1. Revisa los problemas detectados arriba
    2. Añade logging a prediction_cache.py (ver sección CÓMO AÑADIR LOGGING)
    3. Ejecuta tu script y comparte el output
    4. Si no ves prints de [Cache.get] o [Cache.put], el caché está deshabilitado
    """)