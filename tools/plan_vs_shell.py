"""Put the plan we read next to the shell we built, one sheet per listing.

The single most useful review artifact in the project, because it makes the two
halves of AD-2 answerable by eye: the left panel is what the vectoriser believes
a room is, the right panel is the shell extruded from exactly those polygons. A
wrong colour on the left is a wrong room on the right, every time — so judging
the reconstruction reduces to judging the reading.

Three steps, each runnable alone:

``overlays``
    Shade every vectorised room onto its own plan image, labelled with the area
    stage 5 measured.

``shots``
    Screenshot the viewer's dollhouse view of every exported scene. Needs a
    built viewer (``npm run build`` in ``viewer/``) and the scenes exported
    (``python -m tools.export_scene export --all``); serves ``viewer/dist`` on a
    scratch port and drives headless Chromium.

``page``
    Compose both into a single self-contained HTML page with the per-listing
    figures and a verdict, sorted best-reading first.

    python -m tools.plan_vs_shell build --out out/review

The page embeds the plan imagery, so — like the imagery itself — it is a local
artifact and is not committed (see ``.gitignore``).
"""
from __future__ import annotations

import argparse
import base64
import colorsys
import html as H
import io
import json
import subprocess
import sys
import textwrap
import threading
import time
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core.artifacts import ArtifactStore      # noqa: E402
from pipeline.floorplan import ocr as ocr_mod          # noqa: E402
from pipeline.floorplan import preprocess, wallnet     # noqa: E402
from pipeline.floorplan.preprocess import load_rgb     # noqa: E402

REPO = Path(__file__).resolve().parents[1]
VIEWER_DIST = REPO / "viewer" / "dist"

#: Which ceiling heights a surveyor would not query. Wider than the shell stage's
#: own acceptance band because this is a reader's sanity check, not a gate.
CEILING_BELIEVABLE_M = (2.25, 3.3)
#: How much of a room's outline has to lie on a predicted wall before the reading
#: counts as good. This is the check a person makes by eye, and the one an earlier
#: version of this page did not make -- it scored room counts and area totals,
#: which agree perfectly while every outline is drawn in the wrong place.
OUTLINE_FIT_GOOD = 0.85
OUTLINE_FIT_POOR = 0.70
#: How far the measured floor area may sit from the advertised area and still
#: count as agreement. Advertised areas are themselves rounded and inconsistent
#: about whether they count wall thickness, so this cannot be tight.
AREA_AGREEMENT = (0.9, 1.1)


# ---------------------------------------------------------------- overlays


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    try:
        return ImageFont.truetype(f"/usr/share/fonts/truetype/dejavu/{name}", size)
    except OSError:
        return ImageFont.load_default()


def _hue(i: int) -> tuple[int, int, int]:
    # Golden-ratio hue stepping: adjacent rooms never land on adjacent colours,
    # however many there are.
    r, g, b = colorsys.hsv_to_rgb((i * 0.61803) % 1.0, 0.62, 0.95)
    return int(r * 255), int(g * 255), int(b * 255)


def overlay(listing_id: str, out_dir: Path) -> Path | None:
    """Shade each vectorised room onto the plan it came from."""
    plan = ArtifactStore(listing_id).read("5-plan")
    if not plan or not plan.get("source_image"):
        return None
    src = REPO / plan["source_image"]
    if not src.exists():
        return None

    # Polygons are stored in the *geometry* image's pixels -- the deskewed,
    # size-capped image stage 5 actually worked on, which for most plans is a
    # couple of percent smaller than the file on disk. Drawing them over the raw
    # file offsets every outline by up to a wall thickness at the far edge and
    # makes correct rooms look like they float clear of their walls. Rescale the
    # drawing to the space the numbers are in.
    base = Image.fromarray(load_rgb(src))
    gw, gh = plan.get("image_size_px") or base.size
    if (gw, gh) != base.size:
        base = base.resize((int(gw), int(gh)), Image.LANCZOS)
    w, h = base.size
    # Fade the drawing so the shading reads, but never so far that a wall line
    # or a printed dimension stops being legible — the reader is checking the
    # polygons *against* those.
    faded = Image.blend(base, Image.new("RGB", (w, h), (255, 255, 255)), 0.42)

    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(layer)
    rooms = plan["rooms"]
    for i, room in enumerate(rooms):
        poly = [tuple(p) for p in room["polygon_px"]]
        if len(poly) < 3:
            continue
        colour = _hue(i)
        draw.polygon(poly, fill=colour + (86,), outline=colour + (255,),
                     width=max(2, w // 420))

    out = Image.alpha_composite(faded.convert("RGBA"), layer).convert("RGB")
    label = ImageDraw.Draw(out)
    size = max(15, w // 62)
    for room in rooms:
        cx, cy = room["centroid_px"]
        name = (room.get("label_text") or room.get("label") or room["room_id"]).upper()[:18]
        area = room.get("area_m2")
        text = name + (f"\n{area:.1f} m²" if area else "")
        # Halo first, so the caption survives whatever it lands on.
        for dx, dy in ((-2, 0), (2, 0), (0, -2), (0, 2)):
            label.multiline_text((cx + dx, cy + dy), text, font=_font(size, True),
                                 fill=(255, 255, 255), anchor="mm", align="center")
        label.multiline_text((cx, cy), text, font=_font(size, True),
                             fill=(20, 20, 24), anchor="mm", align="center")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{listing_id}.png"
    out.save(path)
    return path


def build_overlays(listing_ids: list[str], out_dir: Path) -> list[str]:
    done = []
    for lid in listing_ids:
        try:
            if overlay(lid, out_dir):
                done.append(lid)
                print(f"  overlay {lid}", flush=True)
        except Exception as exc:                       # noqa: BLE001
            print(f"  overlay {lid}: {exc!r}", file=sys.stderr)
    return done


# ------------------------------------------------------------------ shots

# Strip the viewer's own chrome before the shot: everything that is not an
# ancestor of the canvas goes, which survives the HUD being restructured.
_HIDE_CHROME = """() => {
  const canvas = document.querySelector('canvas');
  const keep = new Set();
  for (let n = canvas; n; n = n.parentElement) keep.add(n);
  document.querySelectorAll('body *').forEach((el) => {
    if (!keep.has(el) && !el.contains(canvas)) el.style.display = 'none';
  });
}"""


def build_shots(out_dir: Path) -> list[str]:
    """Screenshot every exported scene's dollhouse view."""
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright is not installed — skipping the 3D shots", file=sys.stderr)
        return []

    index = VIEWER_DIST / "scenes" / "index.json"
    if not index.exists():
        print(f"no {index} — run: npm run build (in viewer/) and "
              "python -m tools.export_scene export --all", file=sys.stderr)
        return []
    scenes = json.loads(index.read_text())["scenes"]
    out_dir.mkdir(parents=True, exist_ok=True)

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, *_args):      # one line per asset is not news
            pass

    handler = partial(Quiet, directory=str(VIEWER_DIST))
    # Port 0 lets the OS pick one: two of these can run side by side, and a
    # stale server from a killed run never blocks the next one.
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    port = server.server_address[1]
    threading.Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.4)

    done = []
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(
                executable_path="/opt/pw-browsers/chromium",
                args=["--use-gl=swiftshader", "--enable-unsafe-swiftshader",
                      "--no-sandbox"])
            page = browser.new_page(viewport={"width": 1100, "height": 820})
            for entry in scenes:
                lid = entry["listing_id"]
                try:
                    page.goto(f"http://127.0.0.1:{port}/?scene={entry['url']}",
                              wait_until="networkidle", timeout=45_000)
                    page.wait_for_timeout(1600)
                    page.keyboard.press("d")            # walkthrough -> dollhouse
                    page.wait_for_timeout(1400)
                    page.evaluate(_HIDE_CHROME)
                    page.wait_for_timeout(250)
                    page.screenshot(path=str(out_dir / f"{lid}.png"))
                    done.append(lid)
                    print(f"  shot {lid}", flush=True)
                except Exception as exc:                # noqa: BLE001
                    print(f"  shot {lid}: {exc!r}", file=sys.stderr)
            browser.close()
    finally:
        server.shutdown()
    return done


# ------------------------------------------------------------------- page


def _crop(path: Path, light_ground: bool) -> Image.Image:
    """Trim the margin around the drawing so both panels fill their box."""
    im = Image.open(path).convert("RGB")
    lum = np.asarray(im).astype(int).sum(axis=2)
    ys, xs = np.nonzero(lum < 245 * 3) if light_ground else np.nonzero(lum > 42 * 3)
    if len(xs) < 50:
        return im
    w, h = im.size
    pad = 12
    return im.crop((max(0, xs.min() - pad), max(0, ys.min() - pad),
                    min(w, xs.max() + pad), min(h, ys.max() + pad)))


def _uri(im: Image.Image, max_w: int = 760, quality: int = 82) -> str:
    im = im.copy()
    im.thumbnail((max_w, max_w * 2), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def collect(listing_ids: list[str]) -> list[dict]:
    """The figures each sheet reports, straight off the artifacts."""
    rows = []
    for lid in listing_ids:
        store = ArtifactStore(lid)
        plan = store.read("5-plan")
        if not plan:
            continue
        shell = store.read("8-shell") or {}
        package = store.read("9-package") or {}
        triage = store.read("0-triage") or {}
        fit = _outline_fit(lid, plan)
        heights = [r["height_m"] for r in shell.get("rooms", []) if r.get("height_m")]
        coverage = shell.get("coverage") or {}
        area = (plan.get("totals", {}).get("area_m2")
                or sum(r.get("area_m2") or 0 for r in plan["rooms"]))
        rows.append({
            "listing_id": lid,
            "address": (package.get("address") or triage.get("address") or "")[:58],
            "n_plan": len(plan["rooms"]),
            "n_shell": len(shell.get("rooms", [])),
            "ceilings_measured": coverage.get("with_a_photograph"),
            "area": area or None,
            "stated": triage.get("area_m2") or package.get("advertised_area_m2"),
            "scale_source": plan.get("scale_source", "none"),
            "ceiling": float(np.median(heights)) if heights else None,
            "outline_fit": fit,
            "flags": plan.get("qa_flags", []),
        })
    return rows


def _outline_fit(listing_id: str, plan: dict) -> float | None:
    """Median fraction of each room outline that lies on a predicted wall.

    Scored against walls rather than against any drawn line, because a cabinet run
    is a drawn line, a door swing is a drawn line, and a bed is a drawn line -- so
    "outline sits on ink" cannot tell a room that traces its walls from one that
    traces the furniture. On the golden set the two references differ by 8 points
    of median fit, and that gap *is* the furniture error.
    """
    if not wallnet.available() or not plan.get("source_image"):
        return None
    src = REPO / plan["source_image"]
    if not src.exists():
        return None
    try:
        pi = preprocess.prepare(src)
        if list(pi.ink.shape[::-1]) != list(plan["image_size_px"]):
            return None
        barrier = wallnet.barrier(pi.rgb, pi.ink, ocr_mod.read(pi.rgb).words)
        if barrier is None or not barrier.any():
            return None
        polys = [r["polygon_px"] for r in plan["rooms"] if len(r["polygon_px"]) >= 3]
        scores = wallnet.outline_on_wall(polys, barrier)
    except Exception as exc:                                   # noqa: BLE001
        print(f"  fit {listing_id}: {exc!r}", file=sys.stderr)
        return None
    return float(np.median(scores)) if scores else None


def verdict(row: dict) -> tuple[str, str]:
    """How well the plan was *read*, which is what a person judges by eye.

    Outline fit leads, because a reading whose outlines follow the furniture is
    wrong however well its room count and floor area happen to agree. The other
    three checks can only demote a good reading, never promote a bad one.
    """
    fit = row.get("outline_fit")
    if fit is None:                       # no wall map available to judge against
        return "unknown", "Not judged"
    if fit < OUTLINE_FIT_POOR:
        return "poor", "Outlines wrong"

    ratio = ((row["area"] / row["stated"])
             if (row.get("area") and row.get("stated")) else None)
    ceiling = row.get("ceiling")
    lo, hi = CEILING_BELIEVABLE_M
    alo, ahi = AREA_AGREEMENT
    supporting = sum([
        row["n_shell"] == row["n_plan"] and row["n_plan"] >= 3,
        ratio is None or alo - 0.02 <= ratio <= ahi + 0.02,
        row["scale_source"] != "none",
        bool(ceiling and lo <= ceiling <= hi),
    ])
    if fit >= OUTLINE_FIT_GOOD and supporting == 4:
        return "clean", "Reads clean"
    return "check", "Worth a look"


def _sheet(row: dict, overlays: Path, shots: Path) -> str:
    lid = row["listing_id"]
    kind, word = verdict(row)
    ratio = ((row["area"] / row["stated"])
             if (row.get("area") and row.get("stated")) else None)
    ceiling, measured = row.get("ceiling"), row.get("ceilings_measured")
    alo, ahi = AREA_AGREEMENT
    clo, chi = CEILING_BELIEVABLE_M

    def fig(label: str, value: object, tone: str = "") -> str:
        return f'<div class="fig"><dt>{label}</dt><dd class="{tone}">{value}</dd></div>'

    fit = row.get("outline_fit")
    figures = "".join([
        fig("outline on walls", f"{fit:.0%}" if fit is not None else "&mdash;",
            "ok" if (fit or 0) >= OUTLINE_FIT_GOOD
            else "mid" if (fit or 0) >= OUTLINE_FIT_POOR else "bad"),
        fig("rooms on plan", row["n_plan"]),
        fig("rooms modelled", row["n_shell"],
            "ok" if row["n_shell"] == row["n_plan"] else "mid"),
        fig("ceilings measured",
            f'{measured}&thinsp;/&thinsp;{row["n_shell"]}' if measured is not None else "&mdash;",
            "ok" if (measured or 0) >= 0.6 * max(row["n_shell"], 1) else "mid"),
        fig("floor area", f'{row["area"]:.0f}&nbsp;m²' if row.get("area") else "&mdash;"),
        fig("advertised",
            f'{row["stated"]:.0f}&nbsp;m²' if row.get("stated") else "not stated",
            ("ok" if alo <= ratio <= ahi else "mid") if ratio else "nil"),
        fig("ceiling", f"{ceiling:.2f}&nbsp;m" if ceiling else "&mdash;",
            "ok" if (ceiling and clo <= ceiling <= chi) else "mid"),
        fig("scale from", H.escape(row["scale_source"].replace("_", " ")),
            "ok" if row["scale_source"] in ("printed_dimensions", "printed_area")
            else "nil" if row["scale_source"] == "none" else "mid"),
    ])
    flags = "".join(f"<li>{H.escape(f)}</li>" for f in sorted(row.get("flags") or []))
    plan_uri = _uri(_crop(overlays / f"{lid}.png", True))
    shell_uri = _uri(_crop(shots / f"{lid}.png", False))

    return f"""
<article class="sheet" data-verdict="{kind}">
  <header class="sheet-head">
    <h2>{lid}</h2>
    <p class="addr">{H.escape(row.get("address") or "address not recorded")}</p>
    <span class="verdict v-{kind}">{word}</span>
  </header>
  <div class="pair">
    <figure>
      <img src="{plan_uri}" loading="lazy"
           alt="Floor plan for {lid}, each detected room shaded a different colour">
      <figcaption><b>Read from the plan.</b> Each shaded outline is a room the software
        found, labelled with the area it measured.</figcaption>
    </figure>
    <figure class="plate">
      <img src="{shell_uri}" loading="lazy"
           alt="Three-dimensional shell for {lid}, seen from above with the ceilings removed">
      <figcaption><b>Built from those outlines.</b> Walls extruded to the ceiling height
        measured in the photographs.</figcaption>
    </figure>
  </div>
  <dl class="figures">{figures}</dl>
  {f'<ul class="flags">{flags}</ul>' if flags else ""}
</article>"""


_STYLE = """
:root {
  --paper:#f2f2ef; --sheet:#fbfbfa; --plate:#11151c; --ink:#171c24; --quiet:#5d6470;
  --rule:#d8d8d2; --rule-firm:#b9b9b1; --blue:#27508c; --blue-soft:#e6ebf4;
  --ok:#26714a; --mid:#8d6110; --poor:#9d3227;
  --ok-bg:#e0eee6; --mid-bg:#f5ead4; --poor-bg:#f6e0dc;
  --shadow:0 1px 2px rgba(23,28,36,.06);
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#0e1116; --sheet:#161a21; --plate:#0a0d12; --ink:#e6e6e1; --quiet:#9299a4;
    --rule:#282d36; --rule-firm:#3a414c; --blue:#8aa8dd; --blue-soft:#1b2432;
    --ok:#79c39a; --mid:#d9ac62; --poor:#e08a7e;
    --ok-bg:#152820; --mid-bg:#2c2418; --poor-bg:#2d1a18;
    --shadow:0 1px 2px rgba(0,0,0,.4);
  }
}
:root[data-theme="dark"] {
  --paper:#0e1116; --sheet:#161a21; --plate:#0a0d12; --ink:#e6e6e1; --quiet:#9299a4;
  --rule:#282d36; --rule-firm:#3a414c; --blue:#8aa8dd; --blue-soft:#1b2432;
  --ok:#79c39a; --mid:#d9ac62; --poor:#e08a7e;
  --ok-bg:#152820; --mid-bg:#2c2418; --poor-bg:#2d1a18;
  --shadow:0 1px 2px rgba(0,0,0,.4);
}

*, *::before, *::after { box-sizing:border-box; }
body {
  margin:0; background:var(--paper); color:var(--ink);
  font-family:Newsreader,Georgia,"Times New Roman",serif;
  font-size:17px; line-height:1.6; -webkit-font-smoothing:antialiased;
}
.wrap { max-width:1120px; margin:0 auto; padding:clamp(28px,5vw,60px) clamp(16px,3vw,28px) 96px; }

.eyebrow {
  font-family:Archivo,"Helvetica Neue",Arial,sans-serif; font-size:11.5px; font-weight:600;
  letter-spacing:.16em; text-transform:uppercase; color:var(--blue); margin:0 0 10px;
}
h1 {
  font-family:Archivo,"Helvetica Neue",Arial,sans-serif; font-weight:700;
  font-size:clamp(28px,4.4vw,42px); line-height:1.12; letter-spacing:-.015em;
  margin:0 0 14px; text-wrap:balance;
}
.lede { margin:0 0 32px; max-width:66ch; color:var(--quiet); font-size:18px; }
.lede b { color:var(--ink); font-weight:500; }
.rubric { font-size:15.5px; margin-bottom:0; }

.tally {
  display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(150px,1fr));
  border:1px solid var(--rule); border-radius:3px; overflow:hidden; margin-bottom:14px;
}
.tally div { background:var(--sheet); padding:14px 16px 13px; }
.tally .n {
  display:block; font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:25px;
  font-weight:600; font-variant-numeric:tabular-nums; letter-spacing:-.02em; line-height:1.1;
}
.tally .l {
  display:block; margin-top:5px; font-family:Archivo,Arial,sans-serif; font-size:11px;
  font-weight:500; letter-spacing:.08em; text-transform:uppercase; color:var(--quiet);
}
.tally .n.ok { color:var(--ok); } .tally .n.mid { color:var(--mid); }
.tally .n.poor { color:var(--poor); }

.filters { display:flex; flex-wrap:wrap; gap:8px; align-items:center; margin:26px 0 30px; }
.filters .label {
  font-family:Archivo,Arial,sans-serif; font-size:11px; font-weight:600;
  letter-spacing:.12em; text-transform:uppercase; color:var(--quiet); margin-right:4px;
}
.filters button {
  font-family:Archivo,Arial,sans-serif; font-size:13px; font-weight:500; color:var(--ink);
  background:var(--sheet); border:1px solid var(--rule-firm); border-radius:2px;
  padding:6px 13px; cursor:pointer; transition:background .12s, border-color .12s;
}
.filters button:hover { border-color:var(--blue); }
.filters button[aria-pressed="true"] {
  background:var(--blue); border-color:var(--blue); color:var(--sheet);
}
.filters button:focus-visible { outline:2px solid var(--blue); outline-offset:2px; }

.sheet {
  background:var(--sheet); border:1px solid var(--rule); border-top:2px solid var(--rule-firm);
  border-radius:2px; box-shadow:var(--shadow); padding:clamp(16px,2.4vw,24px); margin-bottom:18px;
}
.sheet[data-verdict="clean"] { border-top-color:var(--ok); }
.sheet[data-verdict="check"] { border-top-color:var(--mid); }
.sheet[data-verdict="poor"]  { border-top-color:var(--poor); }
.sheet[hidden] { display:none; }

.sheet-head {
  display:flex; align-items:baseline; gap:12px; flex-wrap:wrap;
  padding-bottom:14px; margin-bottom:16px; border-bottom:1px solid var(--rule);
}
.sheet-head h2 {
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-weight:600; font-size:17px;
  letter-spacing:-.01em; font-variant-numeric:tabular-nums; margin:0;
}
.addr { margin:0; flex:1 1 240px; color:var(--quiet); font-size:15.5px; font-style:italic; }
.verdict {
  font-family:Archivo,Arial,sans-serif; font-size:11px; font-weight:600; letter-spacing:.09em;
  text-transform:uppercase; padding:4px 10px; border-radius:2px; white-space:nowrap;
}
.v-clean { color:var(--ok); background:var(--ok-bg); }
.v-check { color:var(--mid); background:var(--mid-bg); }
.v-poor  { color:var(--poor); background:var(--poor-bg); }

.pair { display:grid; grid-template-columns:1fr 1fr; gap:16px; }
@media (max-width:720px) { .pair { grid-template-columns:1fr; } }
.pair figure { margin:0; display:flex; flex-direction:column; gap:8px; }
.pair img {
  width:100%; height:auto; display:block; background:#fff;
  border:1px solid var(--rule); border-radius:2px;
}
.pair .plate img { background:var(--plate); }
figcaption { font-size:14px; line-height:1.45; color:var(--quiet); }
figcaption b { color:var(--ink); font-weight:500; }

.figures {
  display:grid; gap:1px; background:var(--rule);
  grid-template-columns:repeat(auto-fit,minmax(126px,1fr));
  border:1px solid var(--rule); border-radius:2px; overflow:hidden; margin:18px 0 0;
}
.fig { background:var(--sheet); padding:9px 12px 10px; }
.fig dt {
  font-family:Archivo,Arial,sans-serif; font-size:10.5px; font-weight:500;
  letter-spacing:.08em; text-transform:uppercase; color:var(--quiet);
}
.fig dd {
  margin:3px 0 0; font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:15px;
  font-weight:500; font-variant-numeric:tabular-nums;
}
.fig dd.ok { color:var(--ok); } .fig dd.mid { color:var(--mid); } .fig dd.nil { color:var(--quiet); }

.flags { list-style:none; display:flex; flex-wrap:wrap; gap:6px; margin:12px 0 0; padding:0; }
.flags li {
  font-family:"IBM Plex Mono",ui-monospace,monospace; font-size:11px; color:var(--quiet);
  background:var(--blue-soft); border-radius:2px; padding:3px 8px;
}

footer {
  margin-top:44px; padding-top:20px; border-top:1px solid var(--rule);
  color:var(--quiet); font-size:15px; max-width:70ch;
}
@media (prefers-reduced-motion: reduce) { * { transition:none !important; } }
"""

_SCRIPT = """
(function () {
  var buttons = document.querySelectorAll('.filters button');
  var sheets = document.querySelectorAll('.sheet');
  function apply(f) {
    buttons.forEach(function (b) { b.setAttribute('aria-pressed', b.dataset.f === f); });
    sheets.forEach(function (s) { s.hidden = f !== 'all' && s.dataset.verdict !== f; });
    try { localStorage.setItem('pvs-filter', f); } catch (e) {}
  }
  buttons.forEach(function (b) {
    b.addEventListener('click', function () { apply(b.dataset.f); });
  });
  var saved = 'all';
  try { saved = localStorage.getItem('pvs-filter') || 'all'; } catch (e) {}
  if (saved !== 'all') apply(saved);
})();
"""


def build_page(rows: list[dict], overlays: Path, shots: Path, out: Path) -> Path:
    rows = [r for r in rows
            if (overlays / f'{r["listing_id"]}.png').exists()
            and (shots / f'{r["listing_id"]}.png').exists()]
    rank = {"clean": 0, "check": 1, "poor": 2, "unknown": 3}
    rows.sort(key=lambda r: (rank[verdict(r)[0]], -(r.get("outline_fit") or 0)))

    n = len(rows)
    kinds = [verdict(r)[0] for r in rows]
    clean, check, poor = (kinds.count("clean"), kinds.count("check"), kinds.count("poor"))
    fits = [r["outline_fit"] for r in rows if r.get("outline_fit") is not None]
    median_fit = float(np.median(fits)) if fits else 0.0
    built = sum(1 for r in rows if r["n_shell"] == r["n_plan"])
    ratios = [r["area"] / r["stated"] for r in rows if r.get("area") and r.get("stated")]
    alo, ahi = AREA_AGREEMENT
    within = sum(1 for x in ratios if alo <= x <= ahi)
    median_rooms = int(np.median([r["n_shell"] for r in rows])) if rows else 0
    clo, chi = CEILING_BELIEVABLE_M

    sheets = "".join(_sheet(r, overlays, shots) for r in rows)
    page = f"""<title>Plan Against Shell</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Archivo:wght@500;600;700\
&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400\
&family=IBM+Plex+Mono:wght@400;500;600&display=swap">
<style>{_STYLE}</style>

<div class="wrap">
  <p class="eyebrow">Phase 1 · plan-to-3D · {n} flats</p>
  <h1>Plan against shell, one flat at a time</h1>
  <p class="lede">Each sheet puts the estate agent's floor plan next to the 3D model built
    from it. <b>The colours on the left are what the software believes a room is</b> — if a
    colour is wrong there, the room on the right is wrong in exactly the same way. That is
    the whole point of showing them together: you can judge the reconstruction by judging
    the reading.</p>

  <div class="tally">
    <div><span class="n">{n}</span><span class="l">flats modelled</span></div>
    <div><span class="n ok">{clean}</span><span class="l">read clean</span></div>
    <div><span class="n mid">{check}</span><span class="l">worth a look</span></div>
    <div><span class="n poor">{poor}</span><span class="l">need work</span></div>
    <div><span class="n">{built}/{n}</span><span class="l">every room built</span></div>
    <div><span class="n">{within}/{len(ratios)}</span><span class="l">area within 10%</span></div>
    <div><span class="n">{median_fit:.0%}</span><span class="l">median outline on walls</span></div>
    <div><span class="n">{median_rooms}</span><span class="l">median rooms per flat</span></div>
  </div>
  <p class="lede rubric">The number that matters is <b>outline on walls</b>: walk each
    room's outline and ask how much of it lies on something a trained model calls a
    wall. A room that stops at the kitchen cabinets or follows a door swing cuts across
    open floor and scores low. Under {OUTLINE_FIT_POOR:.0%} the reading is wrong and the sheet says
    <b>outlines wrong</b>; at {OUTLINE_FIT_GOOD:.0%} or better, with the room count, floor area, scale
    and ceiling height all agreeing too, it <b>reads clean</b>; anything else is
    <b>worth a look</b>.</p>

  <div class="filters">
    <span class="label">Show</span>
    <button type="button" data-f="all" aria-pressed="true">All {n}</button>
    <button type="button" data-f="clean" aria-pressed="false">Read clean ({clean})</button>
    <button type="button" data-f="check" aria-pressed="false">Worth a look ({check})</button>
    <button type="button" data-f="poor" aria-pressed="false">Need work ({poor})</button>
  </div>
{sheets}
  <footer>Ordered best-reading first. Nothing here is measured against a tape: the floor
    areas come from the plan's own printed dimensions and the ceiling heights from the
    listing photographs, so every figure is a consistency check rather than a ground
    truth.</footer>
</div>
<script>{_SCRIPT}</script>
"""
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page)
    return out


# ------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["build", "overlays", "shots", "page"],
                    nargs="?", default="build")
    ap.add_argument("listing_ids", nargs="*",
                    help="default: every listing with a stage-5 artifact")
    ap.add_argument("--out", type=Path, default=Path("out/review"))
    args = ap.parse_args(argv)

    ids = args.listing_ids or sorted(ArtifactStore.list_listings())
    overlays, shots = args.out / "overlays", args.out / "shells"

    if args.step in ("build", "overlays"):
        print(f"overlays for {len(ids)} listing(s)")
        build_overlays(ids, overlays)
    if args.step in ("build", "shots"):
        print("dollhouse shots")
        build_shots(shots)
    if args.step in ("build", "page"):
        page = build_page(collect(ids), overlays, shots, args.out / "plan-vs-shell.html")
        size = page.stat().st_size / 1e6
        print(f"wrote {page} ({size:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
