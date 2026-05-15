"""
diag_diff_report_gbm.py
═══════════════════════════════════════════════════════════════════════════
Cross-release diff reporter — pone OOF / holdout / walk-forward / ensemble
de N releases en una sola tabla markdown.

Lee los JSONs persistidos por las fases 1-4:
  · reports/best_per_side.json       (fase 1, siempre presente)
  · reports/holdout_report.json      (fase 2, opcional)
  · reports/walkforward_report.json  (fase 3, opcional)
  · reports/ensemble_report_*.json   (fase 4, opcional — primer match)

NO accede a la BD ni a Optuna. Solo JSONs.

USO:
  python -m mimo.oof.diag_diff_report_gbm \\
    --releases 202602_GBM 202603_GBM \\
    --tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
    --out-md artifacts/_cross_release_report.md
"""

from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _fnum(v, default=float("nan")) -> float:
    if v is None: return default
    try: return float(v)
    except Exception: return default


def _fint(v, default=0) -> int:
    if v is None: return default
    try: return int(v)
    except Exception: return default


def _load(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path): return None
    try:
        with open(path) as fh: return json.load(fh)
    except Exception as e:
        print(f"⚠️  No pude leer {path}: {e}")
        return None


def _collect(release: str, tag: str, base: str = "artifacts") -> Dict[str, Any]:
    reports_dir = Path(base) / release / "oof" / tag / "reports"
    best     = _load(str(reports_dir / "best_per_side.json"))
    holdout  = _load(str(reports_dir / "holdout_report.json"))
    walkf    = _load(str(reports_dir / "walkforward_report.json"))
    # Ensemble: distintos nombres según strategy/top_n. Tomamos el primero.
    ens_files = sorted(glob.glob(str(reports_dir / "ensemble_report_*.json")))
    ensemble = _load(ens_files[0]) if ens_files else _load(str(reports_dir / "ensemble_report.json"))
    return {"release": release, "tag": tag, "best": best,
            "holdout": holdout, "walkf": walkf, "ensemble": ensemble,
            "ensemble_path": ens_files[0] if ens_files else None}


def _fmt_or_na(fmt: str, v) -> str:
    f = _fnum(v)
    return "n/a" if f != f else format(f, fmt)  # f != f → NaN


def _row(data: Dict[str, Any]) -> Dict[str, str]:
    r = {"release": data["release"]}
    best = data["best"]
    holdout = data["holdout"]
    walkf = data["walkf"]
    ens = data["ensemble"]

    # OOF best
    if best:
        tl = best.get("top_long",  [{}])[0].get("ev_long",  {}) or {}
        ts = best.get("top_short", [{}])[0].get("ev_short", {}) or {}
        r["OOF_L_ev"]   = _fmt_or_na("+.4f", tl.get("ev_net"))
        r["OOF_L_prec"] = _fmt_or_na(".3f",  tl.get("prec_TP"))
        r["OOF_L_sig"]  = str(_fint(tl.get("n_signals")))
        r["OOF_L_thr"]  = _fmt_or_na(".3f",  tl.get("thr"))
        r["OOF_S_ev"]   = _fmt_or_na("+.4f", ts.get("ev_net"))
        r["OOF_S_prec"] = _fmt_or_na(".3f",  ts.get("prec_TP"))
        r["OOF_S_sig"]  = str(_fint(ts.get("n_signals")))
        r["OOF_S_thr"]  = _fmt_or_na(".3f",  ts.get("thr"))
    else:
        for k in ("OOF_L_ev","OOF_L_prec","OOF_L_sig","OOF_L_thr",
                  "OOF_S_ev","OOF_S_prec","OOF_S_sig","OOF_S_thr"):
            r[k] = "—"

    # Holdout honest
    if holdout and "holdout_honest" in holdout:
        hl = holdout["holdout_honest"]["long"]  or {}
        hs = holdout["holdout_honest"]["short"] or {}
        r["HOLD_L_ev"]   = _fmt_or_na("+.4f", hl.get("ev_net"))
        r["HOLD_L_prec"] = _fmt_or_na(".3f",  hl.get("prec_TP"))
        r["HOLD_L_sig"]  = str(_fint(hl.get("n_signals")))
        r["HOLD_S_ev"]   = _fmt_or_na("+.4f", hs.get("ev_net"))
        r["HOLD_S_prec"] = _fmt_or_na(".3f",  hs.get("prec_TP"))
        r["HOLD_S_sig"]  = str(_fint(hs.get("n_signals")))
        r["HOLD_R"]      = _fmt_or_na("+.2f", holdout["holdout_honest"].get("total_R"))
    else:
        for k in ("HOLD_L_ev","HOLD_L_prec","HOLD_L_sig",
                  "HOLD_S_ev","HOLD_S_prec","HOLD_S_sig","HOLD_R"):
            r[k] = "—"

    # Walk-forward
    if walkf:
        sl = walkf.get("summary_long",  {}) or {}
        ss = walkf.get("summary_short", {}) or {}
        r["WF_L_pwr"]  = _fmt_or_na(".0%", sl.get("pwr"))
        r["WF_L_evm"]  = _fmt_or_na("+.4f", sl.get("ev_median"))
        r["WF_L_Rtot"] = _fmt_or_na("+.2f", sl.get("R_total"))
        r["WF_S_pwr"]  = _fmt_or_na(".0%", ss.get("pwr"))
        r["WF_S_evm"]  = _fmt_or_na("+.4f", ss.get("ev_median"))
        r["WF_S_Rtot"] = _fmt_or_na("+.2f", ss.get("R_total"))
    else:
        for k in ("WF_L_pwr","WF_L_evm","WF_L_Rtot","WF_S_pwr","WF_S_evm","WF_S_Rtot"):
            r[k] = "—"

    # Ensemble
    if ens and "holdout" in ens:
        el = ens["holdout"]["long"]  or {}
        es = ens["holdout"]["short"] or {}
        r["ENS_L_ev"]   = _fmt_or_na("+.4f", el.get("ev_net"))
        r["ENS_L_sig"]  = str(_fint(el.get("n_signals")))
        r["ENS_S_ev"]   = _fmt_or_na("+.4f", es.get("ev_net"))
        r["ENS_S_sig"]  = str(_fint(es.get("n_signals")))
        r["ENS_R"]      = _fmt_or_na("+.2f", ens["holdout"].get("total_R"))
        r["ENS_topN"]   = str(_fint(ens.get("top_n")))
    else:
        for k in ("ENS_L_ev","ENS_L_sig","ENS_S_ev","ENS_S_sig","ENS_R","ENS_topN"):
            r[k] = "—"
    return r


def _build_markdown(rows: List[Dict[str, str]]) -> str:
    """4 sub-tablas (OOF, HOLD, WF, ENS) cada una con LONG y SHORT, por
    release. Optimizado para legibilidad en GitHub/preview md."""
    if not rows:
        return "(no data)"

    def _tbl(title: str, cols: List[str], headers: List[str]) -> str:
        lines = [f"\n### {title}\n"]
        lines.append("| Release | " + " | ".join(headers) + " |")
        lines.append("|---|" + "|".join(["---"] * len(headers)) + "|")
        for r in rows:
            vals = [r.get(c, "—") for c in cols]
            lines.append(f"| `{r['release']}` | " + " | ".join(vals) + " |")
        return "\n".join(lines)

    out = ["# Cross-Release GBM Report\n",
           f"Releases comparados: {', '.join(r['release'] for r in rows)}\n"]
    out.append(_tbl("OOF (best per side — training-time, calibrated)",
                    ["OOF_L_ev","OOF_L_prec","OOF_L_sig","OOF_L_thr",
                     "OOF_S_ev","OOF_S_prec","OOF_S_sig","OOF_S_thr"],
                    ["L ev_net","L prec","L sig","L thr",
                     "S ev_net","S prec","S sig","S thr"]))
    out.append(_tbl("Holdout HONEST (thr OOF aplicado a holdout)",
                    ["HOLD_L_ev","HOLD_L_prec","HOLD_L_sig",
                     "HOLD_S_ev","HOLD_S_prec","HOLD_S_sig","HOLD_R"],
                    ["L ev_net","L prec","L sig",
                     "S ev_net","S prec","S sig","Total R"]))
    out.append(_tbl("Walk-forward (ventanas deslizantes)",
                    ["WF_L_pwr","WF_L_evm","WF_L_Rtot",
                     "WF_S_pwr","WF_S_evm","WF_S_Rtot"],
                    ["L PWR","L EV med","L R tot",
                     "S PWR","S EV med","S R tot"]))
    out.append(_tbl("Ensemble top-N (holdout)",
                    ["ENS_topN","ENS_L_ev","ENS_L_sig",
                     "ENS_S_ev","ENS_S_sig","ENS_R"],
                    ["N","L ev_net","L sig","S ev_net","S sig","Total R"]))
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--releases", nargs="+", required=True,
                    help="Lista de releases. Ej: 202602_GBM 202603_GBM")
    ap.add_argument("--tag", required=True,
                    help="Tag de experimento bajo artifacts/<release>/oof/<tag>/")
    ap.add_argument("--base", default="artifacts")
    ap.add_argument("--out-md", default=None)
    args = ap.parse_args()

    rows = []
    for rel in args.releases:
        print(f"📂 Leyendo {rel}/{args.tag}")
        data = _collect(rel, args.tag, base=args.base)
        rows.append(_row(data))
        # Resumen rápido stdout
        present = []
        for k, label in (("best","OOF"),("holdout","HOLD"),
                         ("walkf","WF"),("ensemble","ENS")):
            present.append(label if data[k] else f"{label}∅")
        print(f"   → {' '.join(present)}")

    md = _build_markdown(rows)
    print("\n" + md)
    if args.out_md:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_md)), exist_ok=True)
        with open(args.out_md, "w") as fh: fh.write(md)
        print(f"\n📁 Markdown: {args.out_md}")


if __name__ == "__main__":
    main()
