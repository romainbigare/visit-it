"""Read the text a floor plan prints on itself.

Phase 0 measured what is there: room labels on 83% of plans, printed dimensions
on 54% (62% of high-resolution ones), an explicit "not to scale" disclaimer on
nearly half. That text does three jobs for us:

1. It **names** the rooms, which is the cheapest room-type signal in the system.
2. It **scales** the plan — a room drawn 420 px wide and printed "3.96 m" gives
   px-per-metre directly, with no external measurement.
3. It **seeds** the vectoriser: every labelled room gets a segmentation seed
   exactly where a human would point.

Two practical lessons are baked in. Tesseract's default page-segmentation mode
assumes a page of prose; floor plans are sparse text scattered over a drawing, so
we also run PSM 11 and merge. And resolution matters more than any parameter —
upscaling before OCR is the single highest-yield knob.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

log = logging.getLogger("floorplan.ocr")

OCR_MIN_SIDE = 2400
MIN_CONF = 25.0

#: Canonical labels, in match priority order. Longest / most specific first, so
#: "en-suite" is not swallowed by "suite" and "utility room" beats "room".
LABEL_PATTERNS: tuple[tuple[str, str], ...] = (
    ("bathroom", r"bath\s*room|bathroom|family\s+bath"),
    ("bathroom", r"en[\s\-–]?suite|ensuite|shower\s*room"),
    ("wc", r"\bw\.?\s?c\.?\b|cloak\s*room|cloaks|separate\s+wc|toilet"),
    ("kitchen", r"kitchen|galley"),
    ("dining_room", r"dining|breakfast\s*room"),
    ("living_room", r"living|lounge|sitting|family\s*room|snug|drawing\s*room"),
    ("reception", r"reception"),
    ("bedroom", r"bed\s*room|bedroom|\bbed\s*\d|master\s+bed|principal\s+bed|guest\s+room"),
    ("study", r"study|office|home\s*office|cinema\s*room|media\s*room"),
    ("utility", r"utility|laundry|plant\s*room|boiler"),
    ("storage", r"store|storage|cupboard|pantry|larder|airing|dress(ing)?\s*room|wardrobe|closet"),
    ("hallway", r"hall(way)?|landing|corridor|lobby|entrance|vestibule|passage"),
    ("balcony", r"balcony|terrace|roof\s*terrace|veranda"),
    ("garden", r"garden|patio|yard|decking"),
)
_LABEL_RE = [(lab, re.compile(pat, re.I)) for lab, pat in LABEL_PATTERNS]

#: Any of these words means "this text block sits in a room".
ROOM_WORD = re.compile("|".join(p for _, p in LABEL_PATTERNS), re.I)

#: "3.96 x 3.66" or "3,96m x 3,66 m" — the metric pair printed under a room name.
METRIC_DIM = re.compile(
    r"(\d{1,2}[.,]\d{1,2})\s*m?\s*[a-z]?\s*[xX×]\s*(\d{1,2}[.,]\d{1,2})\s*m?", re.I)
#: The same pair after OCR has dropped the "x". Two plausible decimals side by
#: side under a room name are its dimensions on every plan style we have seen.
METRIC_PAIR = re.compile(r"\b(\d{1,2}[.,]\d{1,2})\s*m?\s{0,3}(\d{1,2}[.,]\d{1,2})\s*m?\b", re.I)
#: "5.8m" printed, "58m" recognised. The decimal point is a single pixel cluster
#: and it is the character tesseract drops most often on plan captions. We repair
#: it only when the repaired value is plausible and the raw one is not.
LOST_POINT = re.compile(r"(?<![\d.,])(\d{2,3})\s*m\b", re.I)
#: "13'3 x 8'6" or "13'3\" x 8'6\"" — the imperial pair, usually printed alongside.
IMPERIAL_DIM = re.compile(
    r"(\d{1,2})\s*['’]\s*(\d{1,2})?\s*[\"”]?\s*[xX×]\s*(\d{1,2})\s*['’]\s*(\d{1,2})?\s*[\"”]?")
#: "672 SQ FT / 62.4 SQ M" and friends.
AREA = re.compile(r"(\d[\d,]{0,6}(?:\.\d+)?)\s*(sq\.?\s*m(?:et(?:re|er)s?)?|sqm|m²|m2|"
                  r"sq\.?\s*f(?:ee)?t|sqft|ft²|ft2)", re.I)
SCALE_RATIO = re.compile(r"\b1\s*[:：]\s*(\d{2,4})\b")
NOT_TO_SCALE = re.compile(
    r"not\s+to\s+scale|approximate|illustrative|indicative|for\s+illustration|"
    r"guidance\s+only|no\s+responsibility|not\s+be\s+relied", re.I)

SQFT_TO_SQM = 0.09290304
FT_TO_M = 0.3048


@dataclass
class Word:
    text: str
    conf: float
    x: float
    y: float
    w: float
    h: float

    @property
    def cx(self) -> float:
        return self.x + self.w / 2

    @property
    def cy(self) -> float:
        return self.y + self.h / 2

    def to_dict(self) -> dict:
        return {"text": self.text, "conf": round(self.conf, 1),
                "box": [round(self.x, 1), round(self.y, 1), round(self.w, 1), round(self.h, 1)]}


@dataclass
class TextBlock:
    """A cluster of words that belong together — one room's caption, usually."""
    words: list[Word]
    text: str = ""
    label: str | None = None
    label_text: str | None = None
    dims_m: tuple[float, float] | None = None
    area_m2: float | None = None

    def __post_init__(self):
        if not self.text:
            self.text = " ".join(w.text for w in self.words)

    @property
    def cx(self) -> float:
        return float(np.mean([w.cx for w in self.words]))

    @property
    def cy(self) -> float:
        return float(np.mean([w.cy for w in self.words]))

    @property
    def box(self) -> tuple[float, float, float, float]:
        x0 = min(w.x for w in self.words)
        y0 = min(w.y for w in self.words)
        x1 = max(w.x + w.w for w in self.words)
        y1 = max(w.y + w.h for w in self.words)
        return x0, y0, x1, y1

    @property
    def is_room_caption(self) -> bool:
        return self.label is not None or self.dims_m is not None


@dataclass
class PlanText:
    words: list[Word] = field(default_factory=list)
    blocks: list[TextBlock] = field(default_factory=list)
    raw_text: str = ""
    room_labels: list[str] = field(default_factory=list)
    areas_sqm: list[float] = field(default_factory=list)
    dims_m: list[tuple[float, float]] = field(default_factory=list)
    total_area_m2: float | None = None
    scale_ratio: int | None = None
    not_to_scale: bool = False
    engine: str = "tesseract"


def canonical_label(text: str) -> tuple[str | None, str | None]:
    """Map free text to a canonical room label. Returns ``(label, matched_text)``."""
    for lab, rx in _LABEL_RE:
        m = rx.search(text)
        if m:
            return lab, m.group(0).strip()
    return None, None


def repair_lost_decimal(text: str) -> str:
    """Put back the decimal point tesseract dropped: "58m" -> "5.8m".

    Applied only where the raw reading is an implausible room dimension and the
    repaired one is plausible, so a genuine "12m" corridor is left alone.
    """
    def sub(m: re.Match) -> str:
        raw = m.group(1)
        v = float(raw)
        if 0.6 <= v <= 25:
            return m.group(0)
        fixed = v / 10.0
        return f"{fixed:.1f}m" if 0.6 <= fixed <= 25 else m.group(0)
    return LOST_POINT.sub(sub, text)


def _metric_candidates(text: str) -> list[tuple[float, float, int]]:
    """All metric-looking pairs, each scored: higher is more trustworthy.

    UK plans print metric and imperial one above the other, and OCR happily reads
    "8.8 ft x 11.5 ft" as a decimal pair. So a candidate is *demoted* when "ft" or
    a foot mark sits next to it and *promoted* when an "m" does.
    """
    out: list[tuple[float, float, int]] = []
    for rx, base in ((METRIC_DIM, 2), (METRIC_PAIR, 1)):
        for m in rx.finditer(text):
            try:
                a = float(m.group(1).replace(",", "."))
                b = float(m.group(2).replace(",", "."))
            except ValueError:
                continue
            if not (0.6 <= a <= 25 and 0.6 <= b <= 25):
                continue
            # A *tight* context window. Plans print metric and imperial one under
            # the other, so a wide window always sees a foot mark and would demote
            # the metric line too.
            ctx = text[max(0, m.start() - 4):m.end() + 4].lower()
            metric_marker = bool(re.search(r"\d\s*m\b|\dm\b", m.group(0).lower()))
            imperial_marker = any(t in ctx for t in ("ft", "'", "\u2019", '"', "\u201d"))
            score = base + (3 if metric_marker else 0) - (3 if imperial_marker else 0)
            out.append((max(a, b), min(a, b), score))
    return out


def _as_feet(a: float, b: float) -> tuple[float, float] | None:
    """Reinterpret a demoted pair as feet. "8.8 ft x 11.5 ft" is 2.68 x 3.51 m.

    Worth the special case: on plans that print imperial above metric, OCR often
    reads the imperial line cleanly and mangles the metric one, and the imperial
    numbers are perfectly good once you believe their units.
    """
    am, bm = a * FT_TO_M, b * FT_TO_M
    return (am, bm) if 0.6 <= bm <= 25 and 0.6 <= am <= 25 else None


def parse_dims(text: str) -> tuple[float, float] | None:
    """Room dimensions in metres, metric preferred, imperial converted.

    Guarded on plausibility: a UK room side is 0.6-25 m. Without the guard, OCR
    noise like "13'3 x 8'6" misread as "133 x 86" quietly poisons the scale solve,
    and the scale solve is what the whole G1 self-consistency check rests on.
    """
    text = repair_lost_decimal(text)
    cands = _metric_candidates(text)
    good = [c for c in cands if c[2] > 0]
    if good:
        best = max(good, key=lambda c: c[2])
        return (best[0], best[1])
    m = IMPERIAL_DIM.search(text)
    if m:
        ft1, in1, ft2, in2 = (float(g) if g else 0.0 for g in m.groups())
        a = (ft1 + in1 / 12.0) * FT_TO_M
        b = (ft2 + in2 / 12.0) * FT_TO_M
        if 0.6 <= a <= 25 and 0.6 <= b <= 25:
            return (max(a, b), min(a, b))
    if cands:  # only ft-flavoured readings survive — so read them as feet
        best = max(cands, key=lambda c: c[2])
        conv = _as_feet(best[0], best[1])
        return conv if conv else (best[0], best[1])
    return None


def parse_areas(text: str) -> list[float]:
    """Every area mentioned, in m². Imperial converted; implausible values dropped."""
    out: list[float] = []
    for value, unit in AREA.findall(text):
        try:
            v = float(value.replace(",", ""))
        except ValueError:
            continue
        u = unit.lower().replace(" ", "").replace(".", "")
        if u.startswith(("sqf", "sqft", "ft")):
            v *= SQFT_TO_SQM
        if 2.0 <= v <= 1500.0:
            out.append(round(v, 2))
    return out


def _run_tesseract(im: Image.Image, psm: int, scale: float) -> list[Word]:
    import pytesseract
    cfg = f"--psm {psm}"
    d = pytesseract.image_to_data(im, config=cfg, output_type=pytesseract.Output.DICT)
    out: list[Word] = []
    for i, t in enumerate(d["text"]):
        t = t.strip()
        if not t:
            continue
        try:
            conf = float(d["conf"][i])
        except (TypeError, ValueError):
            continue
        if conf < MIN_CONF:
            continue
        # Single stray glyphs are dimension arrowheads and hatching, not words.
        # "x" is the exception that matters: it is the separator in every printed
        # room dimension, and dropping it costs us the strongest scale constraint
        # we have.
        if len(t) < 2 and not t.isdigit() and t not in ("x", "X", "×"):
            continue
        out.append(Word(t, conf, d["left"][i] / scale, d["top"][i] / scale,
                        d["width"][i] / scale, d["height"][i] / scale))
    return out


def _dedupe(words: list[Word]) -> list[Word]:
    """Two PSM passes see the same word twice. Keep the more confident copy."""
    kept: list[Word] = []
    for w in sorted(words, key=lambda w: -w.conf):
        if any(abs(w.cx - k.cx) < max(w.w, k.w) * 0.5 and abs(w.cy - k.cy) < max(w.h, k.h) * 0.6
               and w.text.lower() == k.text.lower() for k in kept):
            continue
        kept.append(w)
    return sorted(kept, key=lambda w: (w.y, w.x))


def cluster_words(words: list[Word], gap_mult: float = 1.2) -> list[TextBlock]:
    """Union-find over word boxes: words within ~1.5 line-heights are one caption.

    "KITCHEN / BREAKFAST ROOM / 3.96 x 3.66 / 13'0 x 12'0" is four OCR lines and
    one room. Clustering is what turns it back into one seed with one label and
    one dimension pair.
    """
    n = len(words)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    own_label = [canonical_label(w.text)[0] for w in words]
    labels: dict[int, set[str]] = {i: ({own_label[i]} if own_label[i] else set())
                                   for i in range(n)}
    pairs: list[tuple[float, int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            a, b = words[i], words[j]
            hi = max(a.h, b.h)
            dx = max(0.0, max(a.x, b.x) - min(a.x + a.w, b.x + b.w))
            dy = max(0.0, max(a.y, b.y) - min(a.y + a.h, b.y + b.h))
            if dx < hi * gap_mult and dy < hi * gap_mult:
                pairs.append((dx + dy, i, j))
    # Nearest first, so the tightest captions form before the loose ones, and a
    # merge that would put two different room names in one block is refused: on a
    # dense plan the labels of adjacent rooms are closer to each other than a room
    # name is to its own dimension line.
    for _d, i, j in sorted(pairs):
        pa, pb = find(i), find(j)
        if pa == pb:
            continue
        merged = labels[pa] | labels[pb]
        # Two different room names in one block usually means two rooms whose
        # captions sit close together. But "RECEPTION / DINING ROOM" is one room
        # with a slash in its name, and there the words practically touch — so the
        # veto only applies once there is a real gap between them.
        if len(merged) > 1 and _d > 0.5 * min(words[i].h, words[j].h):
            continue
        parent[pa] = pb
        labels[pb] = merged
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    blocks: list[TextBlock] = []
    for idxs in groups.values():
        ws = [words[i] for i in sorted(idxs, key=lambda i: (words[i].y, words[i].x))]
        b = TextBlock(words=ws)
        b.label, b.label_text = canonical_label(b.text)
        b.dims_m = parse_dims(b.text)
        areas = parse_areas(b.text)
        b.area_m2 = areas[0] if areas else None
        blocks.append(b)
    return blocks


def read(rgb: np.ndarray, *, min_side: int = OCR_MIN_SIDE,
         psms: tuple[int, ...] = (3, 11)) -> PlanText:
    """OCR a plan image (already deskewed and at geometry resolution)."""
    im = Image.fromarray(rgb)
    scale = max(1.0, min_side / min(im.size))
    if scale > 1.0:
        im = im.resize((int(im.width * scale), int(im.height * scale)), Image.LANCZOS)
    words: list[Word] = []
    for psm in psms:
        try:
            words += _run_tesseract(im, psm, scale)
        except Exception as e:  # noqa: BLE001
            log.warning("tesseract psm %d failed: %s", psm, e)
    words = _dedupe(words)
    blocks = cluster_words(words)
    raw = " ".join(w.text for w in words)

    areas = parse_areas(raw)
    # The total is the biggest plausible area on the sheet — per-room areas are
    # smaller by construction. Where a plan prints both ft2 and m2 they agree, so
    # taking the max is safe.
    total = max(areas) if areas else None
    dims = [b.dims_m for b in blocks if b.dims_m]
    return PlanText(
        words=words, blocks=blocks, raw_text=raw[:4000],
        room_labels=sorted({b.label for b in blocks if b.label}),
        areas_sqm=areas, dims_m=[list(d) for d in dims],  # type: ignore[arg-type]
        total_area_m2=total,
        scale_ratio=int(SCALE_RATIO.search(raw).group(1)) if SCALE_RATIO.search(raw) else None,
        not_to_scale=bool(NOT_TO_SCALE.search(raw)),
    )
