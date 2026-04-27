import json
from pathlib import Path

# deploy_full no tiene los umbrales — probar train_only
ARTIFACTS_PATH = Path('../../artifacts/200372/oof/train_only')

print("=== Todos los archivos en train_only ===")
for f in sorted(ARTIFACTS_PATH.rglob('*')):
    if f.is_file():
        print(f"  {f.relative_to(ARTIFACTS_PATH)}  ({f.stat().st_size/1024:.1f} KB)")

print("\n=== Buscando vol_low / vol_high ===")
for f in ARTIFACTS_PATH.rglob('*'):
    if f.is_file():
        try:
            text = f.read_text(encoding='utf-8', errors='ignore')
            if 'vol_low' in text or 'vol_high' in text:
                print(f"\n  Encontrado en: {f.relative_to(ARTIFACTS_PATH)}")
                # Mostrar contexto
                for line in text.splitlines():
                    if 'vol_low' in line or 'vol_high' in line:
                        print(f"    {line.strip()}")
        except:
            pass