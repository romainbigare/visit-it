"""Stage 0 — triage: what is each image, and what does the listing say about itself.

Phase 0 measured this at F1 0.96 with zero-shot SigLIP, and found something worth
carrying forward: **the model is more accurate than the portal's own metadata.**
Of six disagreements on the golden set, three were real floor plans Rightmove had
filed as photographs and one "floorplan" was an entirely black image. So portal
metadata is a *fallback*, not a reference — and never an evaluation reference.

The classifier is optional. Where it cannot be loaded (no weights on the box, or
the `instant` profile's ban on cold model loads) the stage falls back to portal
metadata and flags the artifact accordingly, because a listing that triages badly
should look different from one that triaged well.
"""
from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from ..core.stages import StageContext, StageResult, register_stage

log = logging.getLogger("triage")

MODEL_ID = "google/siglip-base-patch16-224"

#: Verbatim from the Phase 0 validation run that measured F1 0.96 (see
#: eval/results/VALIDATION-REPORT.md). Do not "improve" these without re-running
#: that measurement — the numbers the roadmap quotes belong to these strings.
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
#: Classes a floor plan is plausibly mistaken for. Used by the adjudication rule.
PLAN_LIKE = ("floorplan", "map", "other", "epc")

_AREA_RE = re.compile(
    r"([\d,]+(?:\.\d+)?)\s*(sq\.?\s*(?:ft|feet)|sqft|ft2|sq\.?\s*m|sqm|m2|m²)", re.I)


class _Classifier:
    """Lazily-loaded SigLIP. Cached on the class — a cold load per listing would
    cost more than the whole rest of the stage."""
    _model = None
    _proc = None

    @classmethod
    def available(cls) -> bool:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
        except ImportError:
            return False
        return True

    @classmethod
    def load(cls, threads: int = 4):
        if cls._model is not None:
            return cls._model, cls._proc
        import torch
        from transformers import AutoModel, AutoProcessor
        torch.set_num_threads(threads)
        cls._proc = AutoProcessor.from_pretrained(MODEL_ID)
        cls._model = AutoModel.from_pretrained(MODEL_ID).eval()
        return cls._model, cls._proc

    @classmethod
    def score(cls, paths: list[Path], prompts: dict[str, str], batch: int = 16):
        """Argmax label + sigmoid score per image.

        Phase 0 bug worth remembering: SigLIP's absolute scores are poorly
        calibrated, and a 0.02 confidence threshold cut room grouping from 26
        groups to 2. **The argmax is the signal; the magnitude is not.** Callers
        must not threshold on the returned score.
        """
        import torch
        from PIL import Image
        model, proc = cls.load()
        keys = list(prompts)
        texts = [prompts[k] for k in keys]
        out: list[tuple[str, float]] = []
        for i in range(0, len(paths), batch):
            imgs = []
            for p in paths[i:i + batch]:
                try:
                    imgs.append(Image.open(p).convert("RGB"))
                except Exception:  # noqa: BLE001
                    imgs.append(Image.new("RGB", (224, 224)))
            inputs = proc(text=texts, images=imgs, padding="max_length",
                          truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = model(**inputs).logits_per_image
            probs = torch.sigmoid(logits)
            for row in probs:
                j = int(row.argmax())
                out.append((keys[j], float(row[j])))
        return out


def phash(path: Path) -> str | None:
    """Cheap perceptual hash for the duplicate pass. 8x8 mean threshold — enough
    to catch a portal serving the same photo twice at different sizes."""
    try:
        from PIL import Image
        with Image.open(path) as im:
            g = im.convert("L").resize((8, 8), Image.LANCZOS)
        px = list(g.getdata())
        avg = sum(px) / len(px)
        bits = "".join("1" if v > avg else "0" for v in px)
        return f"{int(bits, 2):016x}"
    except Exception:  # noqa: BLE001
        return None


def parse_area(listing: dict) -> tuple[float | None, str | None]:
    if listing.get("floor_area_sqm"):
        return float(listing["floor_area_sqm"]), listing.get("floor_area_source")
    m = _AREA_RE.search(listing.get("description") or "")
    if not m:
        return None, None
    v = float(m.group(1).replace(",", ""))
    unit = m.group(2).lower().replace(" ", "").replace(".", "")
    if unit.startswith(("sqf", "sqft", "ft")):
        v *= 0.09290304
    return (round(v, 2), "description") if 8 <= v <= 1500 else (None, None)


def build_manifest(listing: dict, media_root: Path, *, use_model: bool = True) -> dict:
    refs = [(r, "photo") for r in listing.get("photos", [])] + \
           [(r, "floorplan") for r in listing.get("floorplans", [])]
    paths, metas = [], []
    for i, (ref, portal_kind) in enumerate(refs):
        if not ref.get("local_path"):
            continue
        p = media_root / ref["local_path"]
        if not p.exists():
            continue
        paths.append(p)
        metas.append({"image_id": f"i{i:03d}", "path": ref["local_path"],
                      "portal_kind": portal_kind})

    qa: list[str] = []
    types: list[tuple[str, float]] = []
    if use_model and paths and _Classifier.available():
        try:
            types = _Classifier.score(paths, IMAGE_TYPE_PROMPTS)
        except Exception as e:  # noqa: BLE001
            log.warning("triage classifier failed, falling back to portal metadata: %s", e)
            qa.append("classifier_unavailable")
    if not types:
        qa.append("portal_metadata_fallback")
        types = [("floorplan" if m["portal_kind"] == "floorplan" else "interior", 0.4)
                 for m in metas]

    # Adjudication (Phase 0 took recall to 1.00 with this step). Two asymmetric
    # rules, both grounded in what the golden set actually showed:
    #  * model says floorplan, portal says photo  -> believe the model. Three of
    #    the six Phase 0 disagreements were real plans the portal had misfiled.
    #  * model says map/other/epc, portal says floorplan -> believe the portal.
    #    Those classes are what a plan gets confused *with*, and the portal is
    #    right often enough here to be worth deferring to.
    adjudicated = 0
    for i, meta in enumerate(metas):
        t, sc = types[i]
        if meta["portal_kind"] == "floorplan" and t in PLAN_LIKE and t != "floorplan":
            types[i] = ("floorplan", sc)
            adjudicated += 1
    if adjudicated:
        qa.append(f"adjudicated_to_floorplan_{adjudicated}")

    interior_idx = [i for i, (t, _s) in enumerate(types) if t == "interior"]
    rooms: dict[int, tuple[str, float]] = {}
    if use_model and interior_idx and _Classifier.available() and "classifier_unavailable" not in qa:
        try:
            scored = _Classifier.score([paths[i] for i in interior_idx], ROOM_TYPE_PROMPTS)
            rooms = dict(zip(interior_idx, scored))
        except Exception as e:  # noqa: BLE001
            log.warning("room labelling failed: %s", e)

    from PIL import Image
    images, plans, seen_hash = [], [], {}
    disagreements = 0
    for i, (meta, (t, s)) in enumerate(zip(metas, types)):
        try:
            with Image.open(paths[i]) as im:
                w, h = im.size
        except Exception:  # noqa: BLE001
            w = h = None
        flags = []
        ph = phash(paths[i])
        if ph and ph in seen_hash:
            flags.append(f"duplicate_of_{seen_hash[ph]}")
        elif ph:
            seen_hash[ph] = meta["image_id"]
        if ph == "0000000000000000" or ph == "ffffffffffffffff":
            flags.append("blank_image")
        expected = "floorplan" if meta["portal_kind"] == "floorplan" else None
        if expected and t != expected:
            flags.append("portal_metadata_disagrees")
            disagreements += 1
        elif meta["portal_kind"] == "photo" and t == "floorplan":
            flags.append("portal_metadata_disagrees")
            disagreements += 1
        room, room_conf = rooms.get(i, (None, None))
        rec = {"image_id": meta["image_id"], "path": meta["path"], "type": t,
               "type_confidence": round(s, 4), "room_label": room,
               "room_confidence": round(room_conf, 4) if room_conf else None,
               "phash": ph, "quality_flags": flags, "provenance": "scraped",
               "width": w, "height": h}
        images.append(rec)
        if t == "floorplan" and "blank_image" not in flags:
            plans.append({"image_id": meta["image_id"], "path": meta["path"],
                          "max_side_px": max(w or 0, h or 0)})
    plans.sort(key=lambda p: -(p["max_side_px"] or 0))

    if disagreements:
        qa.append(f"portal_metadata_disagreements_{disagreements}")
    if not plans:
        qa.append("no_floorplan")
    if len([im for im in images if im["type"] == "interior"]) < 3:
        qa.append("few_interior_photos")

    area, area_src = parse_area(listing)
    return {
        "schema": "manifest/v1",
        "listing_id": listing["listing_id"],
        "images": images,
        "plans": plans,
        "listing": {
            "advertised_area_m2": area,
            "area_source": area_src,
            "room_count": listing.get("bedrooms"),
            "bedrooms": listing.get("bedrooms"),
            "bathrooms": listing.get("bathrooms"),
            "floor": None,
            "source_text_refs": [k for k in ("description", "key_features")
                                 if listing.get(k)],
        },
        "confidence": round(0.9 if "portal_metadata_fallback" not in qa else 0.45, 3),
        "qa_flags": sorted(set(qa)),
    }


@register_stage("0-triage", description="Classify every image; parse the listing's own numbers")
def run(ctx: StageContext) -> StageResult:
    # AD-17: the instant profile bans cold model loads on the hot path. The
    # fallback is portal metadata, which is worse and says so in the flags.
    use_model = ctx.options.get("triage_model", True)
    payload = build_manifest(ctx.listing, Path(ctx.golden_root or "."), use_model=use_model)
    return StageResult(payload=payload, engine=MODEL_ID if use_model else "portal_metadata")
