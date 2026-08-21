"""The eval harness: run the metrics over a split and publish a scoreboard.

Harness rules (ARCHITECTURE §9), enforced here rather than remembered:

* **The frozen holdout is untouched by development.** ``--split dev`` is the
  default; asking for the holdout requires ``--split holdout`` and the report says
  loudly which split it measured.
* **Coverage is reported next to every score.** "M2 = 4%" over three listings and
  over twenty-four are different claims, and a scoreboard that hides which one it
  is invites the wrong one.
* **Plan-channel isolation.** ``--channel plan`` scores the plan channel on its own
  (plan → scaled polygons versus the plan's own printed numbers). ROADMAP S4 asks
  for this specifically, because when the end-to-end number is bad, G1 debugging
  needs to know which channel to blame.
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pipeline.core import ArtifactStore, latency_report

from . import annotations as ann
from . import metrics as M
from .holdout import load as load_split

log = logging.getLogger("harness")


def score_listing(listing_id: str, golden: Path, store_root: Path | None = None,
                  channel: str = "full") -> dict | None:
    store = ArtifactStore(listing_id, store_root)
    plan = store.read("5-plan")
    manifest = store.read("0-triage") or {}
    layouts = store.read("4-layout")
    assembly = store.read("6-assembly")
    scale = store.read("7-scale")
    shell = store.read("8-shell")
    scene = store.read("9-package")
    annotation = ann.load(golden, listing_id)

    if channel == "plan":
        return _score_plan_channel(listing_id, plan, manifest, annotation)
    if not (scene and scale and shell and assembly and plan and layouts):
        missing = [n for n, v in (("5-plan", plan), ("4-layout", layouts),
                                  ("6-assembly", assembly), ("7-scale", scale),
                                  ("8-shell", shell), ("9-package", scene)) if not v]
        return {"listing_id": listing_id, "complete": False, "missing_stages": missing}

    m1 = M.m1_total_area(scene, scale, manifest, plan)
    m2 = M.m2_room_area(scale)
    m3 = M.m3_adjacency(scene, plan, assembly, annotation)
    m4 = M.m4_layout_iou(scene, plan, assembly)
    m5 = M.m5_assignment(assembly, annotation)
    g1 = M.g1_criteria(scale, layouts, shell, assembly, m4, annotation)
    return {
        "listing_id": listing_id,
        "complete": True,
        "channel": "full",
        "metrics": {m.id: m.to_dict() for m in (m1, m2, m3, m4, m5)},
        "g1": g1,
        "confidence": scene.get("confidence"),
        "qa_flags": scene.get("qa_flags", []),
        "rooms": len(scene.get("rooms", [])),
        "shell_bytes": shell.get("glb", {}).get("bytes"),
        "scale": scale.get("scale"),
    }


def _score_plan_channel(listing_id: str, plan: dict | None, manifest: dict,
                        annotation: dict | None) -> dict:
    """Plan → scaled polygons, judged against the plan's own printed numbers.

    Isolating this from the photo channel is what makes a bad end-to-end number
    debuggable: if the plan channel is clean and the full pipeline is not, the
    fault is downstream of stage 5.
    """
    if not plan:
        return {"listing_id": listing_id, "complete": False, "channel": "plan",
                "missing_stages": ["5-plan"]}
    rooms = plan.get("rooms", [])
    errs = []
    for r in rooms:
        if r.get("ocr_dims_m") and len(r["ocr_dims_m"]) == 2 and r.get("area_m2"):
            printed = r["ocr_dims_m"][0] * r["ocr_dims_m"][1]
            if printed > 0:
                errs.append(100 * abs(r["area_m2"] - printed) / printed)
    total = plan.get("totals", {})
    ref = plan.get("ocr", {}).get("total_area_m2") or total.get("stated_area_m2")
    m1v = (100 * abs(total["plan_area_m2"] - ref) / ref
           if ref and total.get("plan_area_m2") else None)
    return {
        "listing_id": listing_id,
        "complete": True,
        "channel": "plan",
        "metrics": {
            "M1": M.Metric("M1", "plan total area error", m1v, "%",
                           "printed" if plan.get("ocr", {}).get("total_area_m2") else "stated",
                           1 if m1v is not None else 0).to_dict(),
            "M2": M.Metric("M2", "plan per-room area error",
                           float(np.median(errs)) if errs else None, "%", "printed",
                           len(errs),
                           {"within_10pct_frac": (round(float(np.mean([e <= 10 for e in errs])), 3)
                                                  if errs else None)}).to_dict(),
        },
        "px_per_metre": plan.get("px_per_metre"),
        "scale_source": plan.get("scale_source"),
        "n_rooms": len(rooms),
        "n_labelled": sum(1 for r in rooms if r.get("label")),
        "confidence": plan.get("confidence"),
        "qa_flags": plan.get("qa_flags", []),
    }


def aggregate(rows: list[dict]) -> dict:
    done = [r for r in rows if r.get("complete")]
    out: dict = {"listings": len(rows), "complete": len(done),
                 "incomplete": len(rows) - len(done)}
    ids = sorted({m for r in done for m in r.get("metrics", {})})
    out["metrics"] = {}
    for mid in ids:
        vals = [r["metrics"][mid]["value"] for r in done
                if r["metrics"].get(mid, {}).get("value") is not None]
        ns = [r["metrics"][mid]["n"] for r in done if r["metrics"].get(mid)]
        first = next((r["metrics"][mid] for r in done if r["metrics"].get(mid)), {})
        out["metrics"][mid] = {
            "name": first.get("name"), "unit": first.get("unit"),
            "reference": first.get("reference"),
            "listings_with_value": len(vals),
            "coverage": round(len(vals) / len(done), 3) if done else None,
            "median": round(float(np.median(vals)), 3) if vals else None,
            "mean": round(float(statistics.fmean(vals)), 3) if vals else None,
            "p90": round(float(np.percentile(vals, 90)), 3) if vals else None,
            "observations": int(sum(ns)),
        }
    g1_rows = [r["g1"] for r in done if r.get("g1")]
    if g1_rows:
        out["g1"] = M.summarise_g1(g1_rows)
    flags: dict[str, int] = {}
    for r in done:
        for f in r.get("qa_flags", []):
            flags[f] = flags.get(f, 0) + 1
    out["qa_flags"] = dict(sorted(flags.items(), key=lambda kv: -kv[1]))
    return out


def run(golden: Path, split: str = "dev", store_root: Path | None = None,
        channel: str = "full", listing_ids: list[str] | None = None) -> dict:
    if listing_ids:
        ids = listing_ids
        split_name = "explicit"
    else:
        p = load_split(golden / "holdout_split.json")
        ids = p[split]
        split_name = split
    rows = [score_listing(i, golden, store_root, channel) for i in ids]
    rows = [r for r in rows if r]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split": split_name,
        "channel": channel,
        "n_requested": len(ids),
        "summary": aggregate(rows),
        "latency": latency_report(store_root),
        "listings": rows,
        "caveat": ("No tape-measure ground truth exists for this set. Every number "
                   "here is plausibility and self-consistency, not accuracy — "
                   "ROADMAP §0b."),
    }


def render(report: dict) -> str:
    s = report["summary"]
    lines = [f"# Eval — {report['channel']} channel, {report['split']} split",
             "",
             f"{s['complete']}/{report['n_requested']} listings scored "
             f"({s['incomplete']} incomplete) · {report['generated_at']}",
             "", "| metric | median | p90 | coverage | reference | n obs |",
             "|---|---|---|---|---|---|"]
    for mid, m in s.get("metrics", {}).items():
        lines.append(f"| {mid} {m['name']} | {m['median']} {m['unit'] or ''} | {m['p90']} | "
                     f"{m['listings_with_value']}/{s['complete']} | {m['reference']} | "
                     f"{m['observations']} |")
    if s.get("g1"):
        lines += ["", "## G1 criteria", "",
                  "| criterion | pass rate | median | threshold | judged | unjudged |",
                  "|---|---|---|---|---|---|"]
        for k, v in s["g1"].items():
            lines.append(f"| {k} | {v['pass_rate']} | {v['median_value']} | "
                         f"{v['threshold']} | {v['listings_judged']} | "
                         f"{v['listings_unjudged']} |")
    if s.get("qa_flags"):
        lines += ["", "## QA flags", "", "| flag | listings |", "|---|---|"]
        for f, n in list(s["qa_flags"].items())[:15]:
            lines.append(f"| `{f}` | {n} |")
    lines += ["", f"> {report['caveat']}"]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--split", default="dev", choices=["dev", "holdout"])
    ap.add_argument("--channel", default="full", choices=["full", "plan"])
    ap.add_argument("--listing", nargs="*", default=None)
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.split == "holdout" and not a.listing:
        log.warning("scoring the FROZEN HOLDOUT — this is a gate measurement, "
                    "not a development loop")
    rep = run(a.golden, a.split, a.store, a.channel, a.listing)
    a.out.mkdir(parents=True, exist_ok=True)
    name = f"phase1_{a.channel}_{a.split}"
    (a.out / f"{name}.json").write_text(json.dumps(rep, indent=2) + "\n")
    (a.out / f"{name}.md").write_text(render(rep))
    print(render(rep))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
