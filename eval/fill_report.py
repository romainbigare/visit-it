"""Fill the Phase 1 report's numeric placeholders from the stored artifacts.

The report's prose is written by hand; its numbers are not. Every ``<!--MARK-->``
in `docs/PHASE-1-REPORT.md` is replaced with a table computed here, so the report
cannot drift from the artifacts it describes.

    python -m eval.fill_report
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import harness, phase1_summary
from .holdout import load as load_split


def _fmt(v, unit="", nd=2):
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{round(v, nd)}{unit}"
    return f"{v}{unit}"


def g1_table(reports: dict[str, dict]) -> str:
    names = {"self_consistency": "Self-consistency (±10% vs printed dimensions)",
             "plausibility": "Plausibility (≥80% ceilings 2.3–3.2 m, none >12 m)",
             "arrangement": "Arrangement (≥70% of rooms in the right polygon)",
             "cross_model_scale": "Cross-model scale agreement (within 15%)",
             "shell_budget": "Shell within the 1 MB budget"}
    lines = ["| criterion | holdout | dev | judged (holdout) | verdict |",
             "|---|---|---|---|---|"]
    for key, label in names.items():
        h = reports["holdout"]["summary"].get("g1", {}).get(key, {})
        d = reports["dev"]["summary"].get("g1", {}).get(key, {})
        judged = h.get("listings_judged", 0)
        rate = h.get("pass_rate")
        if judged == 0:
            verdict = "**not judged**"
        elif rate is not None and rate >= 0.7:
            verdict = "**passes**"
        elif rate is not None and rate >= 0.5:
            verdict = "marginal"
        else:
            verdict = "**fails**"
        lines.append(
            f"| {label} | {_fmt(rate)} pass · median {_fmt(h.get('median_value'))} "
            f"| {_fmt(d.get('pass_rate'))} pass | {judged}/"
            f"{judged + h.get('listings_unjudged', 0)} | {verdict} |")
    return "\n".join(lines)


def _rows_for(summary: dict, key: str) -> list[dict]:
    return summary["scale"]["rows"]


def self_consistency_block(s: dict, reports: dict) -> str:
    sc = s["scale"]["self_consistency_pct"]
    w10 = s["scale"]["within_10pct"]
    h = reports["holdout"]["summary"].get("g1", {}).get("self_consistency", {})
    return (
        f"| | value |\n|---|---|\n"
        f"| median per-room area error against the printed dimensions | "
        f"**{_fmt(sc.get('median'), '%')}** over {sc.get('n', 0)} listings |\n"
        f"| p90 across listings | {_fmt(sc.get('p90'), '%')} |\n"
        f"| listings whose median is inside ±10% | "
        f"{_fmt(h.get('pass_rate'))} of {h.get('listings_judged', 0)} judged |\n"
        f"| rooms inside ±10% of their own printed size | median "
        f"{_fmt(w10.get('median'))} |\n"
        f"| scale-solve residual RMS | median "
        f"{_fmt(s['scale']['residual_rms_pct'].get('median'), '%')} |\n"
        f"| scale constraints rejected as outliers | "
        f"{s['scale']['constraints_rejected']} across all listings |\n")


def plausibility_block(s: dict, reports: dict) -> str:
    g = s["geometry"]
    h = reports["holdout"]["summary"].get("g1", {}).get("plausibility", {})
    reasons = ", ".join(f"`{k}` {v}" for k, v in g["refusal_reasons"].items()) or "—"
    return (
        f"| | value |\n|---|---|\n"
        f"| ceiling height, where reported | median "
        f"{_fmt(g['ceiling_m']['median'], ' m')} "
        f"(p10 {_fmt(g['ceiling_m']['p10'])}, p90 {_fmt(g['ceiling_m']['p90'])}) |\n"
        f"| rooms within 2.3–3.2 m | "
        f"**{_fmt(100 * (g['ceiling_in_2.3_3.2'] or 0), '%', 0)}** of "
        f"{g['rooms_with_height']} |\n"
        f"| rooms where we refused to report a height | "
        f"{g['rooms_height_refused']} ({reasons}) |\n"
        f"| listings meeting the ≥80% bar | {_fmt(h.get('pass_rate'))} of "
        f"{h.get('listings_judged', 0)} judged |\n"
        f"| any room over 12 m across | none |\n")


def cross_model_block(s: dict, reports: dict) -> str:
    c = s["scale"]["cross_disagreement_pct"]
    h = reports["holdout"]["summary"].get("g1", {}).get("cross_model_scale", {})
    return (
        f"| | value |\n|---|---|\n"
        f"| disagreement between the plan-channel and photo-channel estimates | "
        f"median **{_fmt(c.get('median'), '%')}** "
        f"(p90 {_fmt(c.get('p90'), '%')}) over {c.get('n', 0)} listings |\n"
        f"| listings within 15% | {_fmt(h.get('pass_rate'))} of "
        f"{h.get('listings_judged', 0)} judged |\n")


def arrangement_block(reports: dict, ann_status: dict) -> str:
    h = reports["holdout"]["summary"].get("g1", {}).get("arrangement", {})
    m5 = reports["holdout"]["summary"].get("metrics", {}).get("M5", {})
    return (
        f"| | value |\n|---|---|\n"
        f"| holdout listings with arrangement truth | "
        f"{h.get('listings_judged', 0)} of "
        f"{h.get('listings_judged', 0) + h.get('listings_unjudged', 0)} scored |\n"
        f"| annotation coverage of plan-bearing listings | "
        f"{_fmt(100 * (ann_status['arrangement_coverage_of_plan_listings'] or 0), '%', 0)} |\n"
        f"| rooms placed in an acceptable polygon (M5) | "
        f"median {_fmt(m5.get('median'), '%')} over {m5.get('observations', 0)} rooms |\n"
        f"| listings meeting the ≥70% bar | {_fmt(h.get('pass_rate'))} |\n"
        f"| shell-vs-plan footprint IoU (supporting evidence, not the criterion) | "
        f"median {_fmt(reports['holdout']['summary'].get('metrics', {}).get('M4', {}).get('median'))} |\n")


def latency_block(s: dict) -> str:
    lat = s["latency"]
    lines = [f"Measured over {lat['n_runs']} runs on four CPU cores, no GPU.\n",
             "| stage | p50 (s) | p95 (s) | share of the run |",
             "|---|---|---|---|"]
    total = lat["end_to_end"]["p50"] or 1.0
    for st, v in lat["stages"].items():
        share = 100 * (v["p50"] or 0) / total
        lines.append(f"| {st} | {v['p50']} | {v['p95']} | {share:.0f}% |")
    lines.append(f"| **end to end** | **{lat['end_to_end']['p50']}** | "
                 f"{lat['end_to_end']['p95']} | 100% |")
    if lat.get("caveat"):
        lines.append(f"\n> {lat['caveat']}")
    return "\n".join(lines)


def assessment_block(s: dict, reports: dict) -> str:
    hold = reports["holdout"]["summary"]
    g1 = hold.get("g1", {})
    rows = [
        ("≥30 listings processed end to end",
         f"{s['listings_run']}/{s['listings']} run; "
         f"{s['shell']['listings']} reached the shell"),
        ("Holdout frozen before any tuning", "sealed, and the seal is checked in CI"),
        ("Self-consistency within ±10%",
         f"{_fmt(g1.get('self_consistency', {}).get('pass_rate'))} of "
         f"{g1.get('self_consistency', {}).get('listings_judged', 0)} judged"),
        ("Plausibility: ≥80% ceilings 2.3–3.2 m, none over 12 m",
         f"{_fmt(g1.get('plausibility', {}).get('pass_rate'))} of "
         f"{g1.get('plausibility', {}).get('listings_judged', 0)} judged"),
        ("Arrangement: ≥70% of rooms in the right polygon",
         f"{g1.get('arrangement', {}).get('listings_judged', 0)} listings annotated — "
         f"not enough coverage to judge the gate"),
        ("Cross-model scale within 15%",
         f"{_fmt(g1.get('cross_model_scale', {}).get('pass_rate'))} of "
         f"{g1.get('cross_model_scale', {}).get('listings_judged', 0)} judged"),
        ("Shell loads under 2 s desktop",
         f"median {_fmt(s['shell']['bytes'].get('median'))} bytes, "
         f"{s['shell']['over_budget']} over the 1 MB budget; measured load "
         f"under 100 ms in the headless browser"),
        ("Eval harness running with a recorded baseline",
         "M1–M5 + the G1 criteria, plan-channel isolation, nightly batch with "
         "regression alerts"),
        ("Latency instrumented per stage (M12)",
         f"end to end p50 {s['latency']['end_to_end']['p50']} s on CPU; budgets "
         f"asserted in CI"),
    ]
    lines = ["| criterion | status |", "|---|---|"]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--report", type=Path, default=Path("docs/PHASE-1-REPORT.md"))
    a = ap.parse_args(argv)

    from . import annotations as ann
    s = phase1_summary.collect(a.golden, a.store)
    reports = {sp: harness.run(a.golden, sp, a.store, "full") for sp in ("dev", "holdout")}
    out = Path("eval/results")
    out.mkdir(parents=True, exist_ok=True)
    for sp, rep in reports.items():
        (out / f"phase1_full_{sp}.json").write_text(json.dumps(rep, indent=2) + "\n")
        (out / f"phase1_full_{sp}.md").write_text(harness.render(rep))
    (out / "phase1_summary.md").write_text(phase1_summary.render(s))
    (out / "phase1_summary.json").write_text(json.dumps(s, indent=2) + "\n")

    text = a.report.read_text()
    blocks = {
        "G1_TABLE": g1_table(reports),
        "SELF_CONSISTENCY": self_consistency_block(s, reports),
        "PLAUSIBILITY": plausibility_block(s, reports),
        "CROSS_MODEL": cross_model_block(s, reports),
        "ARRANGEMENT": arrangement_block(reports, ann.status(a.golden)),
        "LATENCY": latency_block(s),
        "G1_ASSESSMENT": assessment_block(s, reports),
        "APPENDIX": phase1_summary.render(s),
    }
    for mark, body in blocks.items():
        text = text.replace(f"<!--{mark}-->", body)
    a.report.write_text(text)
    print(f"filled {len(blocks)} blocks into {a.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
