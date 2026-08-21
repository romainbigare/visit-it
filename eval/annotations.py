"""Ground truth for the metrics nothing else can supply.

Most of Phase 1's numbers need no annotation: the plan prints its own room
dimensions and its own total area, so self-consistency and plausibility come free
(ROADMAP §0b). Two things do not:

* **Arrangement** (G1's third criterion, and M5): which reconstructed room belongs
  in which plan polygon. Nothing in the listing states this — it is precisely what
  assembly has to work out — so a human has to say.
* **Adjacency** (M3), where the vectoriser's own adjacency graph is itself an
  estimate and cannot grade itself.

The format is a flat JSON file per listing under ``data/golden/annotations/``, with
a ``method`` field on every fact recording where it came from:

``human``
    Someone looked at the plan and typed it. The only reference good enough for
    the arrangement criterion.
``derived``
    Read off the plan's own printed text by ``derive``. Honest and free, but it is
    the plan channel grading its own homework for anything the vectoriser produced,
    so metrics say so rather than quoting it as ground truth.

Usage::

    python -m eval.annotations derive              # seed files from plan OCR
    python -m eval.annotations status              # coverage, by method
    python -m eval.annotations edit <listing_id>   # print the template to fill in
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path

log = logging.getLogger("annotations")

ANNOTATION_VERSION = "v1"


def annotation_dir(golden: Path) -> Path:
    d = golden / "annotations"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load(golden: Path, listing_id: str) -> dict | None:
    p = annotation_dir(golden) / f"{listing_id}.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_all(golden: Path) -> dict[str, dict]:
    return {p.stem: json.loads(p.read_text())
            for p in sorted(annotation_dir(golden).glob("*.json"))}


def save(golden: Path, listing_id: str, payload: dict) -> Path:
    p = annotation_dir(golden) / f"{listing_id}.json"
    payload["updated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    p.write_text(json.dumps(payload, indent=2) + "\n")
    return p


def blank(listing_id: str) -> dict:
    return {
        "version": ANNOTATION_VERSION,
        "listing_id": listing_id,
        "method": "human",
        "rooms": [],          # [{plan_room_id, label, area_m2, method}]
        "adjacency": [],      # [[plan_room_id, plan_room_id]]
        "assignment": {},     # {reconstructed_room_id: plan_room_id | null}
        "notes": "",
    }


def derive(golden: Path, store_root: Path | None = None, overwrite: bool = False) -> dict:
    """Seed annotations from what each plan prints on itself.

    Fills in room labels, printed areas and the vectoriser's adjacency, all marked
    ``derived``. It deliberately does **not** fill in ``assignment``: that is the
    one fact the plan cannot supply, and a derived value there would silently turn
    the arrangement criterion into a tautology.
    """
    from pipeline.core.artifacts import ArtifactStore
    listings = json.loads((golden / "golden_set.json").read_text())["listings"]
    made, skipped, no_plan = 0, 0, 0
    for lst in listings:
        lid = lst["listing_id"]
        existing = load(golden, lid)
        if existing and existing.get("method") == "human" and not overwrite:
            skipped += 1
            continue
        plan = ArtifactStore(lid, store_root).read("5-plan")
        if not plan:
            no_plan += 1
            continue
        rooms = []
        for r in plan.get("rooms", []):
            printed = None
            if r.get("ocr_dims_m") and len(r["ocr_dims_m"]) == 2:
                printed = round(r["ocr_dims_m"][0] * r["ocr_dims_m"][1], 2)
            elif r.get("ocr_area_m2"):
                printed = r["ocr_area_m2"]
            rooms.append({"plan_room_id": r["room_id"], "label": r.get("label"),
                          "printed_area_m2": printed,
                          "method": "derived" if printed else "unknown"})
        payload = existing or blank(lid)
        payload.update({
            "method": "derived",
            "rooms": rooms,
            "adjacency": [[a["a"], a["b"]] for a in plan.get("adjacency", [])],
            "plan_source": plan.get("source_image"),
            "derived_from_plan_sha": None,
            "notes": (payload.get("notes") or "") or
                     ("Seeded from the plan's own printed text. `assignment` is "
                      "deliberately empty — it is the one fact the plan cannot supply, "
                      "and the arrangement criterion is not judged without it."),
        })
        payload.setdefault("assignment", {})
        save(golden, lid, payload)
        made += 1
    return {"derived": made, "kept_human": skipped, "no_plan_artifact": no_plan}


def status(golden: Path) -> dict:
    all_ann = load_all(golden)
    listings = json.loads((golden / "golden_set.json").read_text())["listings"]
    with_plan = [l["listing_id"] for l in listings if l.get("has_floorplan")]
    human = [k for k, v in all_ann.items() if v.get("method") == "human"]
    with_assign = [k for k, v in all_ann.items() if v.get("assignment")]
    return {
        "listings": len(listings),
        "listings_with_plan": len(with_plan),
        "annotation_files": len(all_ann),
        "human_annotated": len(human),
        "with_arrangement_truth": len(with_assign),
        "arrangement_coverage_of_plan_listings": (
            round(len(with_assign) / len(with_plan), 3) if with_plan else None),
        "note": ("Only listings in `with_arrangement_truth` can judge G1's "
                 "arrangement criterion. Everything else reports it unjudged."),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["derive", "status", "edit"])
    ap.add_argument("listing_id", nargs="?")
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if a.cmd == "derive":
        print(json.dumps(derive(a.golden, a.store, a.overwrite), indent=2))
    elif a.cmd == "status":
        print(json.dumps(status(a.golden), indent=2))
    else:
        if not a.listing_id:
            print("edit needs a listing id")
            return 2
        print(json.dumps(load(a.golden, a.listing_id) or blank(a.listing_id), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
