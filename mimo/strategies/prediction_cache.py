# mimo_old/prediction_cache.py - Sistema de Caché para Predicciones del Modelo

"""
Caché para predicciones del modelo base en backtests RL.

IMPORTANTE: Solo para backtest/simulation. NUNCA usar en live trading.

PROPÓSITO:
-----------
En optimización RL, las predicciones base (pred_long_raw, pred_short_raw) son
IDÉNTICAS en cada trial porque solo cambian parámetros RL, no el modelo.

Recalcular predicciones es MUY costoso (inferencia de red neural).

GANANCIA ESPERADA:
------------------
- Primer trial: Tiempo normal (genera caché)
- Trials 2-N: 50-70% más rápidos (usa caché)
- 40 trials: Ahorro de 15-25 horas

USO:
----
```python
sim = TradingSimulator(
    # ... params normales ...
    use_prediction_cache=True,
    cache_dir='./cache/predictions'
)
```
"""

import hashlib
import pickle
from pathlib import Path
from typing import Optional

import joblib
import pandas as pd
import numpy as np


class PredictionCache:
    """
    Caché para predicciones del modelo

    IMPORTANTE: Solo para backtest/simulation
    NO usar en live trading

    Características:
    - Caché en disco (pickle)
    - Key basada en release + fechas
    - Verificación de integridad (tamaño)
    - Estadísticas de hit/miss
    """

    def __init__(self, cache_dir: str = './cache/predictions', enabled: bool = True):
        self.cache_dir = Path(cache_dir)
        self.enabled = enabled
        self.hits = 0
        self.misses = 0

        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_key(self, release: str, from_date: str, to_date: str, n_rows: int = None) -> str:
        """
        Genera key única incluyendo número de filas para evitar subsets
        Genera key única para combinación release+fechas

        IMPORTANTE: Incluye release porque las predicciones cambian
        entre diferentes modelos entrenados
        """

        if n_rows is not None:
            key_str = f"{release}_{from_date}_{to_date}_{n_rows}"
        else:
            key_str = f"{release}_{from_date}_{to_date}"

        return hashlib.md5(key_str.encode()).hexdigest()

    def _get_filepath(self, key: str) -> Path:
        """Path al archivo de caché"""
        return self.cache_dir / f"pred_{key}.joblib"

    def get(self, release: str, from_date: str, to_date: str, n_rows: int = None) -> Optional[pd.DataFrame]:
        """
        Intenta cargar predicciones del caché

        Returns:
            DataFrame con predicciones si existe, None si no
        """
        if not self.enabled:
            return None

        key = self._get_key(release, from_date, to_date, n_rows)
        filepath = self._get_filepath(key)

        if not filepath.exists():
            self.misses += 1
            return None

        try:
            with open(filepath, 'rb') as f:
                df = joblib.load(f)

            # Verificación básica
            if not isinstance(df, pd.DataFrame):
                print(f"[Cache] ⚠️  Invalid format in cache, removing: {key[:8]}...")
                filepath.unlink()
                self.misses += 1
                return None

            self.hits += 1
            hit_rate = self.hits / (self.hits + self.misses)
            print(
                f"[Cache] ✅ HIT ({self.hits}/{self.hits + self.misses}, {hit_rate:.1%}): {key[:8]}... ({len(df)} rows)")

            return df

        except Exception as e:
            print(f"[Cache] ❌ Error loading: {e}")
            # Si hay error, eliminar archivo corrupto
            try:
                filepath.unlink()
            except:
                pass
            self.misses += 1
            return None

    def put(self, release: str, from_date: str, to_date: str, df_predictions: pd.DataFrame, n_rows: int = None):
        """
        Guarda predicciones en caché

        Args:
            release: Código de release del modelo
            from_date: Fecha inicio (str, formato YYYY-MM-DD)
            to_date: Fecha fin (str, formato YYYY-MM-DD)
            df_predictions: DataFrame con predicciones
        """
        if not self.enabled:
            return

        key = self._get_key(release, from_date, to_date, n_rows)
        filepath = self._get_filepath(key)

        try:
            with open(filepath, 'wb') as f:
                joblib.dump(df_predictions, f, protocol=pickle.HIGHEST_PROTOCOL)

            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            print(f"[Cache] 💾 SAVE: {key[:8]}... ({len(df_predictions)} rows, {file_size_mb:.2f} MB)")

        except Exception as e:
            print(f"[Cache] ❌ Error saving: {e}")
            # Si hay error al guardar, intentar eliminar archivo parcial
            try:
                if filepath.exists():
                    filepath.unlink()
            except:
                pass

    def clear(self):
        """Limpia todo el caché"""
        if not self.enabled:
            return

        files = list(self.cache_dir.glob("pred_*.joblib"))
        for f in files:
            f.unlink()

        print(f"[Cache] 🗑️  Cleared {len(files)} files from {self.cache_dir}")
        self.hits = 0
        self.misses = 0

    def clear_release(self, release: str):
        """Limpia caché para un release específico"""
        if not self.enabled:
            return

        # Limpiar todos los archivos que pertenecen a este release
        # Necesitamos buscar porque el key es un hash
        # Mejor estrategia: leer metadata de cada archivo

        removed = 0
        for filepath in self.cache_dir.glob("pred_*.joblib"):
            try:
                # Verificar si el archivo corresponde a este release
                # (simple: si el nombre contiene el release antes del hash)
                # Alternativa: cargar metadata del pickle
                with open(filepath, 'rb') as f:
                    df = joblib.load(f)
                    # Si el DataFrame tiene metadata de release, verificar
                    if hasattr(df, 'attrs') and df.attrs.get('release') == release:
                        filepath.unlink()
                        removed += 1
            except:
                pass

        print(f"[Cache] 🗑️  Cleared {removed} files for release {release}")

    def get_stats(self):
        """Estadísticas del caché"""
        if not self.enabled:
            return {
                'enabled': False,
                'hits': 0,
                'misses': 0,
                'hit_rate': 0.0,
                'n_files': 0,
                'total_size_mb': 0.0
            }

        files = list(self.cache_dir.glob("pred_*.joblib"))
        total_size = sum(f.stat().st_size for f in files)

        return {
            'enabled': True,
            'hits': self.hits,
            'misses': self.misses,
            'hit_rate': self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0.0,
            'n_files': len(files),
            'total_size_mb': total_size / (1024 * 1024),
            'cache_dir': str(self.cache_dir)
        }

    def list_files(self):
        """Lista archivos en el caché con información"""
        if not self.enabled:
            return []

        result = []
        for filepath in sorted(self.cache_dir.glob("pred_*.joblib")):
            try:
                size_mb = filepath.stat().st_size / (1024 * 1024)
                modified = filepath.stat().st_mtime

                # Intentar cargar metadata
                try:
                    with open(filepath, 'rb') as f:
                        df = joblib.load(f)
                        n_rows = len(df)
                except:
                    n_rows = None

                result.append({
                    'file': filepath.name,
                    'size_mb': size_mb,
                    'modified': modified,
                    'n_rows': n_rows
                })
            except:
                pass

        return result

    def invalidate_if_older_than(self, days: int = 30):
        """
        Invalida archivos de caché más antiguos que N días

        Útil para limpiar caché antiguo automáticamente
        """
        if not self.enabled:
            return

        import time
        cutoff = time.time() - (days * 86400)

        removed = 0
        for filepath in self.cache_dir.glob("pred_*.joblib"):
            if filepath.stat().st_mtime < cutoff:
                filepath.unlink()
                removed += 1

        if removed > 0:
            print(f"[Cache] 🗑️  Removed {removed} files older than {days} days")


# ═══════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════

def get_cache_stats_summary(cache: PredictionCache) -> str:
    """Genera un resumen legible de las estadísticas del caché"""
    stats = cache.get_stats()

    if not stats['enabled']:
        return "Cache: DISABLED"

    return (
        f"Cache Stats:\n"
        f"  Hit rate: {stats['hit_rate']:.1%} ({stats['hits']} hits, {stats['misses']} misses)\n"
        f"  Files: {stats['n_files']}\n"
        f"  Size: {stats['total_size_mb']:.2f} MB\n"
        f"  Dir: {stats['cache_dir']}"
    )