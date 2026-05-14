#!/usr/bin/env python3
"""
rollback_deploy.py — Rollback seguro a un deploy anterior conocido.

Operaciones:
  1. Snapshot del estado ACTUAL (config + binario s2_main) antes de tocar nada
  2. Actualiza s2_main.py: artifacts_path → target_deploy
  3. Actualiza s2_main.py: import policy → target_policy
  4. Opcional: restaura calibradores desde .before_swap si existen

Por defecto: --dry-run (muestra qué pasaría, no toca nada).
Para aplicar: --apply

Uso:
  # Ver qué cambiaría sin tocar nada:
  python scripts/rollback_deploy.py \\
    --target-deploy deploy_2026_04_combined_specialists_seed47 \\
    --target-policy decision_policies_config_202500_mar31

  # Aplicar el rollback:
  python scripts/rollback_deploy.py \\
    --target-deploy deploy_2026_04_combined_specialists_seed47 \\
    --target-policy decision_policies_config_202500_mar31 \\
    --apply

  # Listar deploys disponibles para rollback:
  python scripts/rollback_deploy.py --list
"""
from __future__ import annotations
import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent.parent
S2_MAIN = REPO_ROOT / "main" / "s2_main.py"
SNAPSHOTS_DIR = REPO_ROOT / "artifacts" / "_rollback_snapshots"


def _detect_current_deploy() -> Optional[str]:
    """Lee s2_main.py y extrae el subdir actual."""
    if not S2_MAIN.exists():
        return None
    text = S2_MAIN.read_text()
    m = re.search(r'"artifacts"\s*/\s*release\s*/\s*"oof"\s*/\s*"([^"]+)"', text)
    return m.group(1) if m else None


def _detect_current_policy() -> Optional[str]:
    """Extrae el módulo de policy importado en s2_main.py."""
    if not S2_MAIN.exists():
        return None
    text = S2_MAIN.read_text()
    m = re.search(r"from\s+(config\.decision_policies_config[\w_]*)\s+import", text)
    if m:
        return m.group(1).removeprefix("config.")
    return None


def _list_available_deploys(release: str = "202500") -> list[str]:
    base = REPO_ROOT / "artifacts" / release / "oof"
    if not base.exists():
        return []
    deploys = []
    for p in sorted(base.iterdir()):
        if p.is_dir() and (p / f"model_{release}_long.keras").exists():
            deploys.append(p.name)
        elif p.is_dir() and (p / f"model_{release}_multitask.keras").exists():
            deploys.append(p.name)
    return deploys


def _list_available_policies() -> list[str]:
    config_dir = REPO_ROOT / "config"
    if not config_dir.exists():
        return []
    return [p.stem for p in sorted(config_dir.glob("decision_policies_config_*.py"))]


def _snapshot_state(reason: str) -> Path:
    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    snap_dir = SNAPSHOTS_DIR / f"snapshot_{ts}"
    snap_dir.mkdir()
    # Copiar s2_main.py
    if S2_MAIN.exists():
        shutil.copy2(S2_MAIN, snap_dir / "s2_main.py")
    # Metadata
    meta = {
        "timestamp": datetime.now().isoformat(),
        "reason": reason,
        "current_deploy": _detect_current_deploy(),
        "current_policy": _detect_current_policy(),
    }
    (snap_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return snap_dir


def _replace_in_text(text: str, pattern: str, replacement: str, flags: int = 0) -> tuple[str, int]:
    new_text, n = re.subn(pattern, replacement, text, count=1, flags=flags)
    return new_text, n


def _patch_s2_main(target_deploy: str, target_policy: str, apply: bool) -> dict:
    if not S2_MAIN.exists():
        raise SystemExit(f"❌ {S2_MAIN} no existe")
    text = S2_MAIN.read_text()
    original = text
    changes = []

    # 1. artifacts_path → target_deploy
    deploy_pat = r'("artifacts"\s*/\s*release\s*/\s*"oof"\s*/\s*")([^"]+)(")'
    text, n = _replace_in_text(text, deploy_pat, rf'\g<1>{target_deploy}\g<3>')
    if n:
        changes.append(f"artifacts subdir → '{target_deploy}'")

    # 2. import policy → target_policy
    policy_pat = r"from\s+(config\.decision_policies_config[\w_]*)\s+import"
    text, n = _replace_in_text(text, policy_pat, f"from config.{target_policy} import")
    if n:
        changes.append(f"policy import → 'config.{target_policy}'")

    if not changes:
        print("⚠️  No se detectaron patrones a cambiar en s2_main.py. "
              "Revisa manualmente.")
        return {"changed": False, "changes": []}

    if apply:
        S2_MAIN.write_text(text)
        return {"changed": True, "changes": changes, "applied": True}
    else:
        # Mostrar diff resumido
        for c in changes:
            print(f"  · {c}")
        return {"changed": True, "changes": changes, "applied": False}


def _restore_calibrator_backups(release: str, deploy_dir: Path,
                                 apply: bool) -> list[str]:
    """Si en deploy_dir hay archivos .joblib.before_swap, los restaura."""
    restored = []
    for backup in deploy_dir.glob("oof_calibrator_*.joblib.before_swap"):
        target = backup.with_suffix("").with_suffix(".joblib")  # quita .before_swap
        target = backup.parent / backup.name.replace(".before_swap", "")
        if apply:
            shutil.copy2(backup, target)
        restored.append(f"{backup.name} → {target.name}")
    return restored


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-deploy",
                    help="Subdir de artifacts/<release>/oof/ al que volver.")
    ap.add_argument("--target-policy",
                    help="Módulo de policy (sin extensión).")
    ap.add_argument("--release", default="202500")
    ap.add_argument("--restore-calibrator-backups", action="store_true",
                    help="Restaurar oof_calibrator_*.before_swap → .joblib")
    ap.add_argument("--apply", action="store_true",
                    help="Aplicar los cambios. Sin esto es dry-run.")
    ap.add_argument("--list", action="store_true",
                    help="Solo listar deploys y policies disponibles, no hacer nada.")
    args = ap.parse_args()

    if args.list:
        print(f"\n📦 DEPLOYS disponibles en artifacts/{args.release}/oof/:")
        for d in _list_available_deploys(args.release):
            print(f"  · {d}")
        print(f"\n⚙️  POLICY modules disponibles en config/:")
        for p in _list_available_policies():
            print(f"  · {p}")
        print(f"\n📍 Estado actual:")
        print(f"  · deploy:  {_detect_current_deploy() or '(no detectado)'}")
        print(f"  · policy:  {_detect_current_policy() or '(no detectado)'}")
        return

    if not args.target_deploy or not args.target_policy:
        raise SystemExit("❌ Falta --target-deploy y/o --target-policy. "
                         "Usa --list para ver opciones.")

    target_deploy_path = REPO_ROOT / "artifacts" / args.release / "oof" / args.target_deploy
    target_policy_path = REPO_ROOT / "config" / f"{args.target_policy}.py"

    print(f"\n🔄 ROLLBACK PLAN")
    print(f"   Modo:           {'APPLY' if args.apply else 'DRY-RUN (sin --apply)'}")
    print(f"   target_deploy:  {args.target_deploy}")
    print(f"   target_policy:  {args.target_policy}")
    print(f"   restore_cal:    {args.restore_calibrator_backups}")
    print()

    # Validación
    if not target_deploy_path.exists():
        raise SystemExit(f"❌ target_deploy no existe: {target_deploy_path}")
    if not target_policy_path.exists():
        raise SystemExit(f"❌ target_policy no existe: {target_policy_path}")

    current_deploy = _detect_current_deploy()
    current_policy = _detect_current_policy()
    print(f"   Estado actual:")
    print(f"     deploy: {current_deploy}")
    print(f"     policy: {current_policy}")

    if current_deploy == args.target_deploy and current_policy == args.target_policy:
        print("\n✅ Ya estás en el estado objetivo. Nada que hacer.")
        return

    print(f"\n📦 Snapshot del estado actual...")
    if args.apply:
        snap_dir = _snapshot_state(reason=f"pre_rollback_to_{args.target_deploy}")
        print(f"   ✅ snapshot guardado en {snap_dir}")
    else:
        print(f"   (dry-run) snapshot iría a {SNAPSHOTS_DIR}/snapshot_<TS>/")

    print(f"\n🔧 Patching s2_main.py...")
    patch_result = _patch_s2_main(args.target_deploy, args.target_policy, args.apply)
    if not patch_result.get("changed"):
        print("   ⚠️  Sin cambios efectivos.")
    else:
        if args.apply:
            print(f"   ✅ aplicados {len(patch_result['changes'])} cambios.")
        else:
            print(f"   (dry-run) cambiaría {len(patch_result['changes'])} cosas.")

    if args.restore_calibrator_backups:
        print(f"\n🔧 Restaurando calibrator backups en {args.target_deploy}...")
        restored = _restore_calibrator_backups(args.release, target_deploy_path, args.apply)
        if restored:
            for r in restored:
                pfx = "✅" if args.apply else "(dry-run)"
                print(f"   {pfx} {r}")
        else:
            print(f"   (no había .before_swap que restaurar)")

    print(f"\n{'─'*72}")
    if args.apply:
        print(f"✅ ROLLBACK COMPLETADO")
        print(f"   Reinicia el servicio s2 para que los cambios tengan efecto:")
        print(f"     sudo systemctl restart s2-service")
        print(f"     # o, si lo arrancas a mano:")
        print(f"     pkill -f s2_main.py && nohup python main/s2_main.py > logs/s2_$(date +%F).log 2>&1 &")
    else:
        print(f"⚠️  DRY-RUN — nada se ha modificado.")
        print(f"   Para aplicar: añade --apply al comando.")


if __name__ == "__main__":
    main()