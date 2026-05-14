#!/usr/bin/env python3
"""
diagnose_alpha_raw_space.py — Analiza la relación entre raw_proba y hit_rate
para entender DÓNDE vive el alpha antes de tocar el calibrador.

Casos posibles:
  - 🟢 CASO A: Alpha concentrada en cola alta → calibrador con cola alta resuelta
               (iso_full, beta_45d) ayudará
  - 🔴 CASO B: Alpha plana → modelo no discrimina, calibrador NO salvará
  - 🟠 CASO C: Alpha en sweet spot intermedio → la cola alta del raw es noise,
               calibrador que la expande PERJUDICA

Uso:
  python3 scripts/diagnose_alpha_raw_space.py \\
    --release 202500 \\
    --specialist-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
    --seed 47 \\
    --out-dir reports/alpha_diagnosis
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False


def diagnose_side(release, tag, seed, side, out_dir):
    hp_path = Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}/"
                   f"data/holdout_predictions_{release}_{side}.parquet")
    if not hp_path.exists():
        hp_path = Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}_cutoff_mar31/"
                       f"data/holdout_predictions_{release}_{side}.parquet")
    if not hp_path.exists():
        print(f"❌ No encuentro holdout_predictions para {side}")
        return None

    hp = pd.read_parquet(hp_path)
    raw_col = "y_pred_raw" if "y_pred_raw" in hp.columns else "oof_proba_raw"
    sig_col = "y_true" if "y_true" in hp.columns else "signal"

    if raw_col not in hp.columns or sig_col not in hp.columns:
        print(f"❌ Cols requeridas no en parquet: {raw_col}, {sig_col}")
        return None

    try:
        hp["raw_bucket"] = pd.qcut(hp[raw_col], q=20, labels=False, duplicates="drop")
    except Exception:
        hp["raw_bucket"] = pd.cut(hp[raw_col], bins=20, labels=False)

    bucket_stats = hp.groupby("raw_bucket").agg(
        n=(sig_col, "size"),
        hit_rate=(sig_col, "mean"),
        raw_mean=(raw_col, "mean"),
        raw_min=(raw_col, "min"),
        raw_max=(raw_col, "max"),
    ).reset_index()

    base_rate = hp[sig_col].mean()
    bucket_stats["lift"] = bucket_stats["hit_rate"] / base_rate

    print(f"\n{'═'*72}\n  SIDE = {side.upper()}  (base_rate={base_rate:.4f})\n{'═'*72}")
    print(bucket_stats.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    sorted_hp = hp.sort_values(raw_col, ascending=False).reset_index(drop=True)
    sorted_hp["cumhit"] = sorted_hp[sig_col].expanding().mean()
    sorted_hp["rank_pct"] = (sorted_hp.index + 1) / len(sorted_hp)

    print(f"\n  CUMULATIVE HIT RATE TOP-K:")
    for pct in [0.01, 0.02, 0.05, 0.10, 0.20]:
        k = max(int(len(sorted_hp) * pct), 1)
        chr_at_k = sorted_hp[sig_col].head(k).mean()
        lift = chr_at_k / base_rate
        print(f"     top-{pct*100:>4.1f}% (n={k:>5}): hit_rate={chr_at_k:.4f}  lift={lift:.2f}x")

    top_5pct_lift = sorted_hp.head(max(int(len(sorted_hp) * 0.05), 1))[sig_col].mean() / base_rate
    top_1pct_lift = sorted_hp.head(max(int(len(sorted_hp) * 0.01), 1))[sig_col].mean() / base_rate

    print(f"\n  📊 INTERPRETACIÓN:")
    if top_1pct_lift > 2.5 and top_5pct_lift > 1.8:
        print(f"     🟢 CASO A — Alpha CONCENTRADA en cola alta")
        print(f"        → Calibrador con cola alta resuelta (iso_full, beta_45d) ayudará")
        case = "A"
    elif top_5pct_lift < 1.3:
        print(f"     🔴 CASO B — Alpha PLANA")
        print(f"        → El modelo no discrimina bien. Calibrador NO te salvará.")
        case = "B"
    else:
        top_2pct_hit = sorted_hp.head(max(int(len(sorted_hp) * 0.02), 1))[sig_col].mean()
        top_10pct_hit = sorted_hp.head(max(int(len(sorted_hp) * 0.10), 1))[sig_col].mean()
        if top_2pct_hit < top_10pct_hit * 1.1:
            print(f"     🟠 CASO C — Alpha en SWEET SPOT intermedio")
            print(f"        → La cola alta del raw es NOISE. MEJOR mantener iso_21d")
            case = "C"
        else:
            print(f"     🟡 INTERMEDIO — Alpha decente pero no dominante")
            case = "MID"

    if HAS_MPL:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
        ax1.bar(bucket_stats["raw_bucket"], bucket_stats["hit_rate"])
        ax1.axhline(base_rate, color="red", linestyle="--", label=f"base rate = {base_rate:.4f}")
        ax1.set_xlabel("Raw proba bucket")
        ax1.set_ylabel("Hit rate")
        ax1.set_title(f"{side}: Hit rate por raw bucket")
        ax1.legend()
        ax1.grid(alpha=0.3)

        ax2.plot(sorted_hp["rank_pct"], sorted_hp["cumhit"])
        ax2.axhline(base_rate, color="red", linestyle="--", label="base rate")
        ax2.set_xlabel("Top-K (fracción sorted desc)")
        ax2.set_ylabel("Cumulative hit rate")
        ax2.set_title(f"{side}: Cumulative hit rate top-K")
        ax2.set_xscale("log")
        ax2.legend()
        ax2.grid(alpha=0.3)

        ax3.hist(hp[raw_col], bins=50, density=True, alpha=0.6, label="all")
        tp_rows = hp[hp[sig_col] == 1]
        ax3.hist(tp_rows[raw_col], bins=50, density=True, alpha=0.6, label="TP only")
        ax3.set_xlabel("raw_proba")
        ax3.set_ylabel("density")
        ax3.set_title(f"{side}: Distribución de raw_proba")
        ax3.legend()
        ax3.grid(alpha=0.3)

        fig.suptitle(f"Alpha diagnosis raw-space — {side.upper()} (release {release})")
        plot_path = out_dir / f"alpha_raw_{side}.png"
        fig.savefig(plot_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        print(f"  📊 Plot: {plot_path}")

    return {
        "side": side, "n": len(hp), "base_rate": float(base_rate),
        "top_1pct_lift": float(top_1pct_lift), "top_5pct_lift": float(top_5pct_lift),
        "case": case,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="202500")
    ap.add_argument("--specialist-tag", default="rw_both_Lvol_boost_td_down_h3_Svol_boost_h3")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-dir", default="reports/alpha_diagnosis")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = {}
    for side in ["long", "short"]:
        r = diagnose_side(args.release, args.specialist_tag, args.seed, side, out_dir)
        if r:
            results[side] = r

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\n{'═'*72}\n  RESUMEN\n{'═'*72}")
    for side, r in results.items():
        print(f"  {side.upper()}: caso {r['case']} | "
              f"top-1% lift={r['top_1pct_lift']:.2f}x | "
              f"top-5% lift={r['top_5pct_lift']:.2f}x")

    print(f"\n📁 Reporte JSON: {json_path}")


if __name__ == "__main__":
    main()
