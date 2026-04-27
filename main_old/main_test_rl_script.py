"""
Script para testear que main_rl_train.py funciona correctamente
y que el parseo de métricas está bien configurado.
"""

import subprocess
import sys
from pathlib import Path


def test_main_rl_script():
    """
    Ejecuta main_rl_train.py con parámetros básicos
    para verificar que funciona y devuelve métricas.
    """

    # Comando de prueba con parámetros básicos
    cmd = [
        sys.executable, 'main_rl_train.py',
        '--release', '200304',
        '--from', '2025-03-01',
        '--to', '2025-03-15',  # Solo 2 semanas para ir rápido
        '--artifacts_path', './artifacts',
        '--initial_equity', '10000.0',
        '--spread_price', '0.07',
        '--rl_lr', '0.005',
        '--rl_entropy', '0.001',
        '--rl_baseline_beta', '0.90',
        '--rl_max_grad_norm', '10.0',
        '--rl_trade_cost', '0.5',
        '--rl_batch', '32',
        '--rl_chop_soft_thr', '0.60',
        '--rl_exhaustion_soft_thr', '0.60',
        '--rl_chop_penalty_coef', '0.08',
        '--rl_exhaustion_penalty_coef', '0.06',
    ]

    print("=" * 80)
    print("TESTING main_rl_train.py")
    print("=" * 80)
    print(f"\nCommand: {' '.join(cmd)}\n")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minutos máximo
        )

        print("\n" + "=" * 80)
        print("RESULT")
        print("=" * 80)
        print(f"Return code: {result.returncode}")

        print("\n" + "-" * 80)
        print("STDOUT (last 2000 chars):")
        print("-" * 80)
        print(result.stdout[-2000:])

        if result.stderr:
            print("\n" + "-" * 80)
            print("STDERR (last 2000 chars):")
            print("-" * 80)
            print(result.stderr[-2000:])

        # Intentar parsear métricas
        print("\n" + "=" * 80)
        print("PARSING METRICS")
        print("=" * 80)

        import re
        patterns = {
            'n_trades': r'Trades\s*:\s*(\d+)',
            'net_pnl': r'NetPnL\s*:\s*([-+]?\d+\.?\d*)',
            'profit_factor': r'ProfitFactor\s*:\s*(\d+\.?\d*)',
            'win_rate': r'WinRate\s*:\s*(\d+\.?\d*)',
            'max_dd': r'MaxDD\s*:\s*([-+]?\d+\.?\d*)',
            'max_dd_pct': r'MaxDD\s*\(%\)\s*:\s*([-+]?\d+\.?\d*)',
        }

        metrics = {}
        for metric, pattern in patterns.items():
            match = re.search(pattern, result.stdout)
            if match:
                try:
                    metrics[metric] = float(match.group(1))
                    print(f"✓ {metric}: {metrics[metric]}")
                except ValueError:
                    print(f"✗ {metric}: failed to parse")
            else:
                print(f"✗ {metric}: not found in output")

        if not metrics:
            print("\n❌ NO METRICS WERE PARSED!")
            print("\nPossible issues:")
            print("1. Script crashed before printing results")
            print("2. Output format is different than expected")
            print("3. Script is not completing execution")

            # Guardar output completo para análisis
            log_file = Path('test_output.log')
            with open(log_file, 'w') as f:
                f.write("=== STDOUT ===\n")
                f.write(result.stdout)
                f.write("\n\n=== STDERR ===\n")
                f.write(result.stderr)
            print(f"\nFull output saved to: {log_file}")
        else:
            print(f"\n✓ Successfully parsed {len(metrics)} metrics!")

        return result.returncode == 0 and len(metrics) > 0

    except subprocess.TimeoutExpired:
        print("\n❌ ERROR: Script timed out after 10 minutes")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        print(traceback.format_exc())
        return False


if __name__ == '__main__':
    success = test_main_rl_script()

    if success:
        print("\n" + "=" * 80)
        print("✓ TEST PASSED - Script works correctly!")
        print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("✗ TEST FAILED - Check the output above")
        print("=" * 80)
        print("\nTroubleshooting steps:")
        print("1. Check if all dependencies are installed")
        print("2. Verify database connection works")
        print("3. Check if artifacts path exists and has required files")
        print("4. Look at ./test_output.log for detailed output")
        print("5. Try running main_rl_train.py manually to see errors")
