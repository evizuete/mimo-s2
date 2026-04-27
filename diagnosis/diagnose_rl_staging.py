#!/usr/bin/env python3
"""
Diagnóstico: ¿Por qué main_rl_staged.py devuelve siempre el mismo valor?
"""

import re
from pathlib import Path


# ============================================================================
# DIAGNÓSTICO 1: Revisar los logs de validación
# ============================================================================

def diagnose_val_logs():
    """
    Revisa los logs de validación para ver qué está pasando
    """
    print("=" * 80)
    print("DIAGNÓSTICO 1: LOGS DE VALIDACIÓN")
    print("=" * 80)

    logs_dir = Path("../artifacts/200316/rl/staged/logs")

    if not logs_dir.exists():
        print(f"❌ No se encuentra: {logs_dir}")
        return

    # Buscar logs de validation
    val_logs = sorted(logs_dir.glob("*/val.log"))

    if not val_logs:
        print(f"❌ No hay logs de validación en: {logs_dir}")
        return

    print(f"\n✓ Encontrados {len(val_logs)} logs de validación\n")

    # Analizar primeros 3 logs
    for i, log_path in enumerate(val_logs[:5], 1):
        print(f"\n{'=' * 60}")
        print(f"TRIAL {i}: {log_path.parent.name}")
        print(f"{'=' * 60}")

        try:
            content = log_path.read_text()

            # Buscar métricas en el output
            patterns = {
                "Trades": r"Trades\s*:\s*(\d+)",
                "NetPnL": r"NetPnL\s*:\s*([-+]?\d+\.?\d*)",
                "ProfitFactor": r"ProfitFactor\s*:\s*([-+]?\d+\.?\d*)",
                "WinRate": r"WinRate\s*:\s*([-+]?\d+\.?\d*)",
                "MaxDD": r"MaxDD\s*:\s*([-+]?\d+\.?\d*)",
            }

            metrics_found = {}
            for name, pattern in patterns.items():
                match = re.search(pattern, content)
                if match:
                    metrics_found[name] = match.group(1)

            if metrics_found:
                print("Métricas encontradas:")
                for k, v in metrics_found.items():
                    print(f"  {k:15s}: {v}")
            else:
                print("⚠️  NO SE ENCONTRARON MÉTRICAS EN EL LOG")
                print("\nPrimeras 500 chars del log:")
                print("-" * 60)
                print(content[:500])
                print("-" * 60)

            # Revisar si hay errores
            if "error" in content.lower() or "exception" in content.lower():
                print("\n⚠️  ERRORES DETECTADOS:")
                error_lines = [line for line in content.split('\n')
                               if 'error' in line.lower() or 'exception' in line.lower()]
                for line in error_lines[:5]:
                    print(f"    {line}")

            # Revisar returncode
            if "RETURN CODE" in content:
                rc_match = re.search(r"=== RETURN CODE ===\s*(\d+)", content)
                if rc_match:
                    rc = int(rc_match.group(1))
                    if rc != 0:
                        print(f"\n❌ Return code: {rc} (ERROR)")
                    else:
                        print(f"\n✓ Return code: {rc} (OK)")

        except Exception as e:
            print(f"❌ Error leyendo log: {e}")


# ============================================================================
# DIAGNÓSTICO 2: Revisar los JSONs de métricas
# ============================================================================

def diagnose_metrics_json():
    """
    Revisa los JSONs de métricas guardados
    """
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO 2: JSONs DE MÉTRICAS")
    print("=" * 80)

    logs_dir = Path("../artifacts/200316/rl/staged/logs")

    if not logs_dir.exists():
        print(f"❌ No se encuentra: {logs_dir}")
        return

    # Buscar metrics_val.json
    metric_jsons = sorted(logs_dir.glob("*/metrics_val.json"))

    if not metric_jsons:
        print(f"❌ No hay metrics_val.json en: {logs_dir}")
        return

    print(f"\n✓ Encontrados {len(metric_jsons)} archivos de métricas\n")

    import json

    all_values = []

    for i, json_path in enumerate(metric_jsons[:5], 1):
        print(f"\nTRIAL {i}: {json_path.parent.name}")
        print("-" * 60)

        try:
            data = json.loads(json_path.read_text())
            print(f"  Métricas: {data}")

            if 'net_pnl' in data:
                all_values.append(data['net_pnl'])
        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Verificar si todos son iguales
    if all_values:
        print("\n" + "=" * 60)
        print("ANÁLISIS DE VALORES:")
        print("=" * 60)
        print(f"  Valores de net_pnl: {all_values}")

        if len(set(all_values)) == 1:
            print(f"\n  ⚠️  TODOS LOS VALORES SON IGUALES: {all_values[0]}")
            print("  → PROBLEMA CONFIRMADO")
        else:
            print(f"\n  ✓ Los valores son diferentes")
            print(f"    Min: {min(all_values)}")
            print(f"    Max: {max(all_values)}")


# ============================================================================
# DIAGNÓSTICO 3: Revisar las policies generadas
# ============================================================================

def diagnose_policies():
    """
    Verifica si las policies se están generando correctamente
    """
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO 3: POLICIES GENERADAS")
    print("=" * 80)

    logs_dir = Path("../artifacts/200316/rl/staged/logs")

    if not logs_dir.exists():
        print(f"❌ No se encuentra: {logs_dir}")
        return

    # Buscar archivos .npz (policies)
    policies = sorted(logs_dir.glob("*/rl_policy_*.npz"))

    print(f"\n✓ Policies encontradas: {len(policies)}\n")

    import numpy as np

    policy_sizes = []

    for i, policy_path in enumerate(policies[:5], 1):
        print(f"TRIAL {i}: {policy_path.name}")

        try:
            # Cargar policy
            data = np.load(policy_path)

            # Obtener tamaño
            total_params = sum(arr.size for arr in data.values())
            policy_sizes.append(total_params)

            print(f"  Parámetros: {total_params:,}")
            print(f"  Arrays: {list(data.keys())}")

            # Verificar si todas las policies son idénticas
            if i == 1:
                first_policy = {k: v.copy() for k, v in data.items()}
            else:
                # Comparar con primera policy
                identical = True
                for key in first_policy:
                    if key in data:
                        if not np.allclose(first_policy[key], data[key]):
                            identical = False
                            break

                if identical:
                    print(f"  ⚠️  IDÉNTICA a Trial 1")
                else:
                    print(f"  ✓ Diferente de Trial 1")

        except Exception as e:
            print(f"  ❌ Error: {e}")


# ============================================================================
# DIAGNÓSTICO 4: Verificar el comando ejecutado
# ============================================================================

def diagnose_commands():
    """
    Revisa los comandos ejecutados en cada trial
    """
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO 4: COMANDOS EJECUTADOS")
    print("=" * 80)

    logs_dir = Path("../artifacts/200316/rl/staged/logs")

    if not logs_dir.exists():
        print(f"❌ No se encuentra: {logs_dir}")
        return

    # Buscar config_val.json
    configs = sorted(logs_dir.glob("*/config_val.json"))

    if not configs:
        print(f"❌ No hay config_val.json en: {logs_dir}")
        return

    print(f"\n✓ Encontrados {len(configs)} configs\n")

    import json

    # Verificar parámetros RL
    rl_params_list = []

    for i, config_path in enumerate(configs[:5], 1):
        print(f"\nTRIAL {i}: {config_path.parent.name}")
        print("-" * 60)

        try:
            data = json.loads(config_path.read_text())

            # Extraer parámetros RL
            rl_params = {k: v for k, v in data.items() if k.startswith('rl_')}

            print("Parámetros RL:")
            for k, v in sorted(rl_params.items()):
                print(f"  {k:30s}: {v}")

            rl_params_list.append(rl_params)

        except Exception as e:
            print(f"  ❌ Error: {e}")

    # Verificar si los parámetros son diferentes
    if len(rl_params_list) >= 2:
        print("\n" + "=" * 60)
        print("COMPARACIÓN DE PARÁMETROS:")
        print("=" * 60)

        params_identical = all(
            rl_params_list[0] == params
            for params in rl_params_list[1:]
        )

        if params_identical:
            print("  ⚠️  TODOS LOS PARÁMETROS SON IDÉNTICOS")
            print("  → Optuna no está variando los parámetros correctamente")
        else:
            print("  ✓ Los parámetros son diferentes entre trials")


# ============================================================================
# DIAGNÓSTICO 5: Verificar el problema específico
# ============================================================================

def diagnose_specific_value():
    """
    Investiga de dónde viene el valor 3852.678309979098
    """
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO 5: ORIGEN DEL VALOR 3852.678309979098")
    print("=" * 80)

    logs_dir = Path("../artifacts/200316/rl/staged/logs")

    if not logs_dir.exists():
        print(f"❌ No se encuentra: {logs_dir}")
        return

    target_value = "3852.678309979098"
    target_short = "3852.68"

    # Buscar en todos los archivos
    all_files = list(logs_dir.rglob("*.*"))

    print(f"\nBuscando '{target_value}' en {len(all_files)} archivos...")

    matches = []

    for file_path in all_files:
        if file_path.suffix in ['.json', '.log', '.txt']:
            try:
                content = file_path.read_text()

                if target_value in content or target_short in content:
                    matches.append(file_path)

                    print(f"\n✓ Encontrado en: {file_path.relative_to(logs_dir)}")

                    # Mostrar contexto
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if target_value in line or target_short in line:
                            print(f"  Línea {i + 1}: {line.strip()[:100]}")

            except:
                pass

    if not matches:
        print(f"\n⚠️  Valor NO encontrado en los logs")
        print("  → El valor viene del composite score, no de las métricas")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """
    Ejecuta todos los diagnósticos
    """
    print("\n" + "=" * 80)
    print("DIAGNÓSTICO COMPLETO: ¿Por qué valores son siempre iguales?")
    print("=" * 80)

    diagnose_val_logs()
    diagnose_metrics_json()
    diagnose_policies()
    diagnose_commands()
    diagnose_specific_value()

    print("\n" + "=" * 80)
    print("CONCLUSIÓN Y RECOMENDACIONES")
    print("=" * 80)

    print("""
    Si el diagnóstico mostró:

    1. ❌ Métricas NO encontradas en logs:
       → El script main_rl_train.py no está imprimiendo correctamente
       → Verificar que el output tenga formato: "NetPnL: 1234.56"

    2. ❌ Todas las métricas son idénticas:
       → La evaluación no está usando los parámetros de cada trial
       → Verificar que policy_init se está cargando correctamente

    3. ❌ Parámetros idénticos entre trials:
       → Optuna no está variando correctamente
       → Problema en suggest_from_space()

    4. ❌ Policies idénticas:
       → El entrenamiento no está variando
       → Los parámetros no se están usando

    5. ✓ Todo diferente pero score igual:
       → Problema en CompositeScorer.score()
       → Los pesos están mal configurados
    """)


if __name__ == '__main__':
    main()
