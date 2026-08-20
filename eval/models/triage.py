"""Stage 0 validation: can a zero-shot VLM sort a bag of listing images?

Two questions, one rigorously answerable and one not:

  1. FLOOR-PLAN DETECTION - fully labelled. Rightmove tells us which asset is a
     floorplan, so we have 553 images with ground truth and can report real
     precision/recall. This is the gate the whole plan channel depends on.
  2. ROOM TYPE - unlabelled. We report the predicted distribution and dump a
     contact sheet for eyeballing; any accuracy number here would be invented.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

log = logging.getLogger("triage")

MODEL_ID = "google/siglip-base-patch16-224"

# Deliberately more than two classes: the real bag contains EPC charts, maps and
# marketing collateral, and forcing a binary choice hides those failures.
IMAGE_TYPE_PROMPTS = {
    "floorplan": "a 2D architectural floor plan drawing of an apartment",
    "interior": "a photograph of the inside of a room in a house",
    "exterior": "a photograph of the outside of a building",
    "epc": "an energy performance certificate rating chart",
    "map": "a street map or aerial map view",
    "other": "a logo, document, or marketing graphic",
}

ROOM_TYPE_PROMPTS = {
    "living_room": "a photograph of a living room with sofas",
    "bedroom": "a photograph of a bedroom with a bed",
    "kitchen": "a photograph of a kitchen with cabinets and worktops",
    "bathroom": "a photograph of a bathroom with a bath or shower",
    "dining_room": "a photograph of a dining room with a dining table",
    "hallway": "a photograph of a hallway, corridor or entrance",
    "garden": "a photograph of a garden, balcony or outdoor terrace",
    "other_room": "a photograph of a utility room, garage or storage space",
}


@dataclass
class Prediction:
    path: str
    listing_id: str
    truth_type: str
    pred_type: str
    type_score: float
    pred_room: str | None = None
    room_score: float | None = None


class Triage:
    def __init__(self, model_id: str = MODEL_ID, threads: int = 4):
        torch.set_num_threads(threads)
        t0 = time.time()
        self.proc = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModel.from_pretrained(model_id).eval()
        self.load_s = time.time() - t0
        self.model_id = model_id
        self.n_params = sum(p.numel() for p in self.model.parameters())

    @torch.no_grad()
    def _score(self, images: list[Image.Image], prompts: list[str]) -> torch.Tensor:
        inp = self.proc(text=prompts, images=images, padding="max_length",
                        return_tensors="pt")
        return torch.sigmoid(self.model(**inp).logits_per_image)

    def classify(self, paths: list[Path], prompts: dict[str, str],
                 batch_size: int = 16) -> list[tuple[str, float]]:
        keys, texts = list(prompts), list(prompts.values())
        out: list[tuple[str, float]] = []
        for i in range(0, len(paths), batch_size):
            batch = paths[i:i + batch_size]
            ims = [Image.open(p).convert("RGB") for p in batch]
            probs = self._score(ims, texts)
            for row in probs:
                j = int(row.argmax())
                out.append((keys[j], float(row[j])))
            log.info("  classified %d/%d", min(i + batch_size, len(paths)), len(paths))
        return out


def run(golden: Path, out_dir: Path, limit: int | None = None) -> dict:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data = json.loads((golden / "golden_set.json").read_text())
    items: list[tuple[Path, str, str]] = []
    for lst in data["listings"]:
        for ref in lst["photos"] + lst["floorplans"]:
            if ref.get("local_path"):
                items.append((golden / ref["local_path"], lst["listing_id"], ref["kind"]))
    if limit:
        items = items[:limit]
    paths = [p for p, _, _ in items]
    log.info("triage over %d images", len(paths))

    t = Triage()
    t0 = time.time()
    type_preds = t.classify(paths, IMAGE_TYPE_PROMPTS)
    type_s = time.time() - t0

    preds = [Prediction(path=str(p.relative_to(golden)), listing_id=lid,
                        truth_type=("floorplan" if kind == "floorplan" else "photo"),
                        pred_type=pt, type_score=sc)
             for (p, lid, kind), (pt, sc) in zip(items, type_preds)]

    # Room type only for images predicted interior.
    interior_idx = [i for i, pr in enumerate(preds) if pr.pred_type == "interior"]
    t0 = time.time()
    if interior_idx:
        room_preds = t.classify([paths[i] for i in interior_idx], ROOM_TYPE_PROMPTS)
        for i, (rt, rs) in zip(interior_idx, room_preds):
            preds[i].pred_room = rt
            preds[i].room_score = rs
    room_s = time.time() - t0

    # Floor-plan detection is the labelled question.
    tp = sum(1 for p in preds if p.truth_type == "floorplan" and p.pred_type == "floorplan")
    fn = sum(1 for p in preds if p.truth_type == "floorplan" and p.pred_type != "floorplan")
    fp = sum(1 for p in preds if p.truth_type == "photo" and p.pred_type == "floorplan")
    tn = sum(1 for p in preds if p.truth_type == "photo" and p.pred_type != "floorplan")
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0

    dist: dict[str, int] = {}
    for p in preds:
        dist[p.pred_type] = dist.get(p.pred_type, 0) + 1
    rooms: dict[str, int] = {}
    for p in preds:
        if p.pred_room:
            rooms[p.pred_room] = rooms.get(p.pred_room, 0) + 1

    result = {
        "stage": "0-triage",
        "model": t.model_id,
        "params_m": round(t.n_params / 1e6, 1),
        "device": "cpu",
        "n_images": len(paths),
        "load_seconds": round(t.load_s, 1),
        "type_seconds": round(type_s, 1),
        "ms_per_image_type": round(type_s / max(len(paths), 1) * 1000, 1),
        "room_seconds": round(room_s, 1),
        "floorplan_detection": {
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "accuracy": round((tp + tn) / len(preds), 4),
        },
        "predicted_type_distribution": dist,
        "predicted_room_distribution": rooms,
        "predictions": [asdict(p) for p in preds],
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stage0_triage.json").write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--out", type=Path, default=Path("eval/results"))
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    r = run(a.golden, a.out, a.limit)
    fd = r["floorplan_detection"]
    print(f"\nfloor-plan detection: P={fd['precision']:.3f} R={fd['recall']:.3f} "
          f"F1={fd['f1']:.3f} acc={fd['accuracy']:.3f}  (tp={fd['tp']} fp={fd['fp']} fn={fd['fn']})")
    print("types:", r["predicted_type_distribution"])
    print("rooms:", r["predicted_room_distribution"])
