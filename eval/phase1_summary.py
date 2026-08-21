"""Every number the Phase 1 report quotes, computed from the artifacts.

A report with hand-typed numbers is a report that is wrong three commits later.
This produces the tables; the prose in `docs/PHASE-1-REPORT.md` interprets them.

    python -m eval.phase1_summary            # markdown to stdout
    python -m eval.phase1_summary --json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import numpy as np

from pipeline.core import ArtifactStore, Ledger, STAGE_ORDER, latency_report

from .holdout import load as load_split


def _pct(xs, q):
    return round(float(np.percentile(xs, q)), 2) if len(xs) else None


def collect(golden: Path, store_root: Path | None = None) -> dict:
    listings = json.loads((golden / "golden_set.json").read_text())["listings"]
    split = load_split(golden / "holdout_split.json")
    by_id = {l["listing_id"]: l for l in listings}
    have_plan = {l["listing_id"] for l in listings if l.get("has_floorplan")}

    stage_status: dict[str, Counter] = {s: Counter() for s in STAGE_ORDER}
    plan_rows, layout_rows, asm_rows, scale_rows, shell_rows = [], [], [], [], []
    heights, refusals = [], Counter()
    flags = Counter()
    ran = 0

    for lid in by_id:
        store = ArtifactStore(lid, store_root)
        run = Ledger(lid, store_root).latest()
        if run:
            ran += 1
            for s in run["stages"]:
                stage_status[s["stage"]][s["status"]] += 1
        plan = store.read("5-plan")
        if plan:
            plan_rows.append({
                "listing_id": lid, "scale_source": plan["scale_source"],
                "px_per_metre": plan["px_per_metre"],
                "n_rooms": len(plan["rooms"]),
                "n_labelled": sum(1 for r in plan["rooms"] if r.get("label")),
                "n_doors": sum(1 for a in plan["apertures"] if a["type"] == "door"),
                "n_adjacency": len(plan["adjacency"]),
                "confidence": plan["confidence"],
                "plan_area_m2": plan["totals"].get("plan_area_m2"),
                "stated_area_m2": plan["totals"].get("stated_area_m2"),
                "area_ratio": plan["totals"].get("area_ratio"),
                "not_to_scale": plan["ocr"].get("not_to_scale"),
            })
            flags.update(plan["qa_flags"])
        lay = store.read("4-layout")
        if lay:
            for r in lay["rooms"]:
                if r.get("room_height_m"):
                    heights.append(r["room_height_m"])
                else:
                    for f in r["qa_flags"]:
                        if "floor" in f or "ceiling" in f or "wider" in f:
                            refusals[f] += 1
            layout_rows.append({"listing_id": lid, "n_rooms": len(lay["rooms"]),
                                "engine": lay.get("engine"),
                                "confidence": lay["confidence"]})
        asm = store.read("6-assembly")
        if asm:
            margins = [m["margin"] for m in asm["matches"] if m["margin"] is not None]
            asm_rows.append({
                "listing_id": lid, "n_matched": len(asm["matches"]),
                "n_unmatched_rooms": len(asm["unmatched_rooms"]),
                "n_unmatched_plan": len(asm["unmatched_plan_rooms"]),
                "median_margin": round(float(np.median(margins)), 4) if margins else None,
                "mean_iou": asm["refinement"]["mean_iou_after"],
                "confidence": asm["confidence"],
            })
        sc = store.read("7-scale")
        if sc:
            scale_rows.append({
                "listing_id": lid, "scale": sc["scale"], "quality": sc["quality"],
                "residual_rms_pct": sc["residual_rms_pct"],
                "n_constraints": len(sc["constraints"]),
                "n_rejected": sum(1 for c in sc["constraints"] if not c["used"]),
                "self_consistency_pct": sc["self_consistency"]["median_abs_pct"],
                "within_10pct": sc["self_consistency"]["within_10pct_frac"],
                "ceiling_plausible_frac": sc["plausibility"]["ceiling_plausible_frac"],
                "cross_disagreement_pct": sc["cross_check"]["max_disagreement_pct"],
            })
        sh = store.read("8-shell")
        if sh:
            shell_rows.append({"listing_id": lid, "bytes": sh["glb"]["bytes"],
                               "triangles": sh["glb"]["triangles"],
                               "n_rooms": len(sh["rooms"]),
                               "omitted": len(sh.get("omitted_rooms", []))})
        scene = store.read("9-package")
        if scene:
            flags.update(scene["qa_flags"])

    def stat(rows, key):
        v = [r[key] for r in rows if r.get(key) is not None]
        if not v:
            return {"n": 0}
        return {"n": len(v), "median": round(float(np.median(v)), 3),
                "mean": round(statistics.fmean(v), 3),
                "p10": _pct(v, 10), "p90": _pct(v, 90),
                "min": round(min(v), 3), "max": round(max(v), 3)}

    return {
        "listings": len(by_id),
        "listings_with_plan": len(have_plan),
        "listings_run": ran,
        "split": {"dev": len(split["dev"]), "holdout": len(split["holdout"]),
                  "holdout_with_plan": split["holdout_with_plan"]},
        "stage_completion": {s: dict(c) for s, c in stage_status.items() if c},
        "plan_channel": {
            "listings": len(plan_rows),
            "scale_source": dict(Counter(r["scale_source"] for r in plan_rows)),
            "rooms_per_listing": stat(plan_rows, "n_rooms"),
            "labelled_fraction": round(
                sum(r["n_labelled"] for r in plan_rows) /
                max(1, sum(r["n_rooms"] for r in plan_rows)), 3),
            "doors_per_listing": stat(plan_rows, "n_doors"),
            "adjacency_edges": stat(plan_rows, "n_adjacency"),
            "confidence": stat(plan_rows, "confidence"),
            "area_ratio_vs_stated": stat(plan_rows, "area_ratio"),
            "not_to_scale_fraction": round(
                sum(1 for r in plan_rows if r["not_to_scale"]) / max(1, len(plan_rows)), 3),
            "rows": plan_rows,
        },
        "geometry": {
            "rooms_with_height": len(heights),
            "rooms_height_refused": sum(refusals.values()),
            "refusal_reasons": dict(refusals),
            "ceiling_m": {"median": round(float(np.median(heights)), 3) if heights else None,
                          "p10": _pct(heights, 10), "p90": _pct(heights, 90)},
            "ceiling_in_2.3_3.2": round(
                float(np.mean([(2.3 <= h <= 3.2) for h in heights])), 3) if heights else None,
            "listings": len(layout_rows),
        },
        "assembly": {
            "listings": len(asm_rows),
            "matched_per_listing": stat(asm_rows, "n_matched"),
            "unmatched_rooms": stat(asm_rows, "n_unmatched_rooms"),
            "unmatched_plan_polygons": stat(asm_rows, "n_unmatched_plan"),
            "median_margin": stat(asm_rows, "median_margin"),
            "mean_iou": stat(asm_rows, "mean_iou"),
            "rows": asm_rows,
        },
        "scale": {
            "listings": len(scale_rows),
            "scale_factor": stat(scale_rows, "scale"),
            "quality": stat(scale_rows, "quality"),
            "residual_rms_pct": stat(scale_rows, "residual_rms_pct"),
            "self_consistency_pct": stat(scale_rows, "self_consistency_pct"),
            "within_10pct": stat(scale_rows, "within_10pct"),
            "cross_disagreement_pct": stat(scale_rows, "cross_disagreement_pct"),
            "constraints_rejected": sum(r["n_rejected"] for r in scale_rows),
            "rows": scale_rows,
        },
        "shell": {
            "listings": len(shell_rows),
            "bytes": stat(shell_rows, "bytes"),
            "triangles": stat(shell_rows, "triangles"),
            "rooms": stat(shell_rows, "n_rooms"),
            "over_budget": sum(1 for r in shell_rows if r["bytes"] > 1_048_576),
        },
        "qa_flags": dict(flags.most_common()),
        "latency": latency_report(store_root),
    }


def render(d: dict) -> str:
    L = []
    A = L.append
    A(f"{d['listings_run']}/{d['listings']} listings have been run "
      f"({d['listings_with_plan']} carry a floor plan). "
      f"Split: {d['split']['dev']} dev / {d['split']['holdout']} holdout "
      f"({d['split']['holdout_with_plan']} of the holdout carry a plan).\n")

    A("\n### Stage completion\n\n| stage | ok | partial/skipped | failed |\n|---|---|---|---|")
    for s, c in d["stage_completion"].items():
        A(f"| {s} | {c.get('ok', 0)} | {c.get('skipped', 0)} | {c.get('failed', 0)} |")

    p = d["plan_channel"]
    A(f"\n### Plan channel (stage 5) — {p['listings']} listings\n")
    A("| | value |\n|---|---|")
    A(f"| scale source | {', '.join(f'{k} {v}' for k, v in p['scale_source'].items())} |")
    A(f"| rooms found per listing | median {p['rooms_per_listing'].get('median')} "
      f"(p10 {p['rooms_per_listing'].get('p10')}, p90 {p['rooms_per_listing'].get('p90')}) |")
    A(f"| rooms carrying a label | {100 * p['labelled_fraction']:.0f}% |")
    A(f"| doors found per listing | median {p['doors_per_listing'].get('median')} |")
    A(f"| adjacency edges per listing | median {p['adjacency_edges'].get('median')} |")
    A(f"| plan area / stated area | median {p['area_ratio_vs_stated'].get('median')} "
      f"(n={p['area_ratio_vs_stated'].get('n')}) |")
    A(f"| carry a 'not to scale' disclaimer | {100 * p['not_to_scale_fraction']:.0f}% |")
    A(f"| stage confidence | median {p['confidence'].get('median')} |")

    g = d["geometry"]
    A(f"\n### Geometry and layout (stages 3-4) — {g['listings']} listings\n")
    A("| | value |\n|---|---|")
    A(f"| rooms reporting a ceiling height | {g['rooms_with_height']} |")
    A(f"| rooms where we refused to report one | {g['rooms_height_refused']} "
      f"({', '.join(f'{k} {v}' for k, v in g['refusal_reasons'].items()) or '—'}) |")
    A(f"| ceiling height | median {g['ceiling_m']['median']} m "
      f"(p10 {g['ceiling_m']['p10']}, p90 {g['ceiling_m']['p90']}) |")
    A(f"| within 2.3-3.2 m | {100 * (g['ceiling_in_2.3_3.2'] or 0):.0f}% |")

    a = d["assembly"]
    A(f"\n### Assembly (stage 6) — {a['listings']} listings\n")
    A("| | value |\n|---|---|")
    A(f"| rooms matched per listing | median {a['matched_per_listing'].get('median')} |")
    A(f"| reconstructed rooms left unmatched | median {a['unmatched_rooms'].get('median')} |")
    A(f"| plan polygons left unmatched | median {a['unmatched_plan_polygons'].get('median')} |")
    A(f"| cost margin over the runner-up | median {a['median_margin'].get('median')} |")
    A(f"| polygon fit (IoU after SE(2)) | median {a['mean_iou'].get('median')} |")

    s = d["scale"]
    A(f"\n### Scale (stage 7) — {s['listings']} listings\n")
    A("| | value |\n|---|---|")
    A(f"| scale factor applied | median {s['scale_factor'].get('median')} "
      f"(p10 {s['scale_factor'].get('p10')}, p90 {s['scale_factor'].get('p90')}) |")
    A(f"| solve quality | median {s['quality'].get('median')} |")
    A(f"| residual RMS | median {s['residual_rms_pct'].get('median')}% |")
    A(f"| **self-consistency vs printed dimensions** | median "
      f"{s['self_consistency_pct'].get('median')}% (n={s['self_consistency_pct'].get('n')}) |")
    A(f"| rooms within ±10% of their printed size | median "
      f"{s['within_10pct'].get('median')} |")
    A(f"| cross-model scale disagreement | median "
      f"{s['cross_disagreement_pct'].get('median')}% |")
    A(f"| constraints rejected as outliers | {s['constraints_rejected']} total |")

    sh = d["shell"]
    A(f"\n### Shell (stages 8-9) — {sh['listings']} listings\n")
    A("| | value |\n|---|---|")
    A(f"| glTF size | median {sh['bytes'].get('median')} bytes "
      f"(max {sh['bytes'].get('max')}) |")
    A(f"| triangles | median {sh['triangles'].get('median')} |")
    A(f"| rooms in the shell | median {sh['rooms'].get('median')} |")
    A(f"| over the 1 MB budget | {sh['over_budget']} |")

    lat = d["latency"]
    A(f"\n### Latency (M12) — {lat['n_runs']} runs\n")
    if lat.get("caveat"):
        A(f"> {lat['caveat']}\n")
    A("| stage | p50 | p95 | max |\n|---|---|---|---|")
    for st, v in lat["stages"].items():
        A(f"| {st} | {v['p50']} | {v['p95']} | {v['max']} |")
    A(f"| **end to end** | **{lat['end_to_end']['p50']}** | "
      f"{lat['end_to_end']['p95']} | — |")

    A("\n### QA flags, by how many listings raised them\n\n| flag | listings |\n|---|---|")
    for f, n in list(d["qa_flags"].items())[:20]:
        A(f"| `{f}` | {n} |")
    return "\n".join(L) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("eval/results/phase1_summary.md"))
    a = ap.parse_args(argv)
    d = collect(a.golden, a.store)
    if a.json:
        print(json.dumps(d, indent=2))
        return 0
    md = render(d)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(md)
    a.out.with_suffix(".json").write_text(json.dumps(d, indent=2) + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
