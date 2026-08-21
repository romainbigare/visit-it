"""Correct the wall model's reading of our plans, so it can be fine-tuned on them.

The wall model is trained on CubiCasa5K -- Finnish plans, drawn in one house style.
Ours are UK estate-agent plans drawn in twenty, and the gap shows: door swings are
suppressed on some plans and left standing on others, which is worse than either
consistently, because you cannot tell from the output which you got. Closing that
gap means training on our plans, and training on our plans means someone has to
say where the walls are.

**You correct, you do not trace.** Every plan opens with the model's own reading
already painted on, so the work is fixing what it got wrong -- scrubbing a door
swing, painting in a wall it missed -- rather than drawing an outline from
nothing. On a typical plan that is a minute or two, against ten or fifteen to
trace it cold.

    python -m tools.annotate_walls              # http://127.0.0.1:8081
    python -m tools.annotate_walls --export     # write the training set

Masks land in ``data/golden/wall_labels/<listing_id>.png`` -- single channel, 255
where there is a wall. ``--export`` pairs each with its plan image and writes a
manifest for ``notebooks/finetune_wallnet_colab.ipynb``.

Twenty to thirty corrected plans is a useful fine-tuning set; the notebook mixes
them with CubiCasa5K so the model gains our styles without forgetting the ones it
already knows.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import sys
from datetime import datetime, timezone
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.core.artifacts import ArtifactStore          # noqa: E402
from pipeline.floorplan import ocr as ocr_mod              # noqa: E402
from pipeline.floorplan import preprocess, wallnet         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LABEL_DIR = REPO / "data" / "golden" / "wall_labels"
EXPORT_DIR = REPO / "data" / "golden" / "wall_training"

#: Annotating at the plan's full resolution is wasted effort -- the model trains
#: at 512 -- and it makes the brush sluggish in the browser. Long side of this.
WORK_SIDE = 1024


# ----------------------------------------------------------------- the plans


def plan_ids(store_root: Path | None = None) -> list[str]:
    out = []
    for lid in sorted(ArtifactStore.list_listings(store_root)):
        plan = ArtifactStore(lid, store_root).read("5-plan")
        if plan and plan.get("source_image") and (REPO / plan["source_image"]).exists():
            out.append(lid)
    return out


def _fit(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) <= WORK_SIDE:
        return img
    s = WORK_SIDE / max(h, w)
    return np.array(Image.fromarray(img).resize((int(w * s), int(h * s)), Image.LANCZOS))


def plan_image(listing_id: str, store_root: Path | None = None) -> np.ndarray:
    plan = ArtifactStore(listing_id, store_root).read("5-plan")
    return _fit(preprocess.load_rgb(REPO / plan["source_image"]))


def seed_mask(listing_id: str, rgb: np.ndarray) -> np.ndarray:
    """The model's own reading, to be corrected. Blank if it is unavailable."""
    saved = LABEL_DIR / f"{listing_id}.png"
    if saved.exists():
        m = np.array(Image.open(saved).convert("L"))
        if m.shape[:2] == rgb.shape[:2]:
            return (m > 127).astype(np.uint8) * 255
        return np.array(Image.fromarray(m).resize((rgb.shape[1], rgb.shape[0]),
                                                  Image.NEAREST))
    if not wallnet.available():
        return np.zeros(rgb.shape[:2], np.uint8)
    ink, _bg = preprocess.ink_mask(rgb)
    bar = wallnet.barrier(rgb, ink, ocr_mod.read(rgb).words)
    return np.zeros(rgb.shape[:2], np.uint8) if bar is None else (bar * 255).astype(np.uint8)


def _png_data_uri(arr: np.ndarray, mode: str = "RGB") -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).convert(mode).save(buf, "PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def done_ids() -> set[str]:
    return {p.stem for p in LABEL_DIR.glob("*.png")}


# ------------------------------------------------------------------- export


def export(store_root: Path | None = None) -> dict:
    """Pair every corrected mask with its plan and write a manifest."""
    EXPORT_DIR.mkdir(parents=True, exist_ok=True)
    (EXPORT_DIR / "images").mkdir(exist_ok=True)
    (EXPORT_DIR / "masks").mkdir(exist_ok=True)
    entries = []
    for path in sorted(LABEL_DIR.glob("*.png")):
        lid = path.stem
        try:
            rgb = plan_image(lid, store_root)
        except Exception as exc:                                  # noqa: BLE001
            print(f"  skip {lid}: {exc!r}", file=sys.stderr)
            continue
        mask = np.array(Image.open(path).convert("L"))
        if mask.shape[:2] != rgb.shape[:2]:
            mask = np.array(Image.fromarray(mask).resize(
                (rgb.shape[1], rgb.shape[0]), Image.NEAREST))
        Image.fromarray(rgb).save(EXPORT_DIR / "images" / f"{lid}.png")
        Image.fromarray((mask > 127).astype(np.uint8) * 255).save(
            EXPORT_DIR / "masks" / f"{lid}.png")
        entries.append({"listing_id": lid,
                        "image": f"images/{lid}.png",
                        "mask": f"masks/{lid}.png",
                        "size_px": [rgb.shape[1], rgb.shape[0]],
                        "wall_fraction": round(float((mask > 127).mean()), 4)})
    manifest = {
        "version": "wall_labels/v1",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "note": ("Hand-corrected wall masks for UK estate-agent plans. 255 = wall, "
                 "0 = everything else. Seeded from the CubiCasa-trained model and "
                 "corrected by hand, so they are not independent of it -- good "
                 "enough to fine-tune on, not a clean-room ground truth."),
        "count": len(entries),
        "entries": entries,
    }
    (EXPORT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


# ------------------------------------------------------------------ the page

_PAGE = """<!doctype html>
<meta charset="utf-8"><title>Wall labels · {lid}</title>
<style>
  :root {{ color-scheme: light dark; --ink:#171c24; --paper:#f4f4f1; --line:#d5d5cf;
           --accent:#27508c; --ok:#26714a; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --ink:#e8e8e3; --paper:#14171c; --line:#2b3038; --accent:#8aa8dd; --ok:#79c39a; }}
  }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink);
          font:14px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  header {{ display:flex; gap:16px; align-items:center; flex-wrap:wrap;
            padding:10px 16px; border-bottom:1px solid var(--line); position:sticky;
            top:0; background:var(--paper); z-index:5; }}
  header b {{ font-variant-numeric:tabular-nums; }}
  .grow {{ flex:1; }}
  button, select {{ font:inherit; padding:5px 12px; border:1px solid var(--line);
            border-radius:3px; background:transparent; color:inherit; cursor:pointer; }}
  button.on {{ background:var(--accent); border-color:var(--accent); color:var(--paper); }}
  button:disabled {{ opacity:.4; cursor:default; }}
  .wrap {{ position:relative; margin:16px auto; width:max-content; max-width:100%;
           line-height:0; touch-action:none; }}
  canvas {{ position:absolute; inset:0; width:100%; height:100%; }}
  img {{ display:block; max-width:100%; height:auto; }}
  #paint {{ opacity:.45; }}
  .hint {{ padding:0 16px 24px; max-width:80ch; color:color-mix(in srgb, var(--ink) 62%, transparent); }}
  kbd {{ font:12px ui-monospace,monospace; border:1px solid var(--line);
         border-radius:3px; padding:1px 5px; }}
  .saved {{ color:var(--ok); font-weight:600; }}
</style>
<header>
  <b>{lid}</b>
  <span>{index} of {total} · {done} corrected</span>
  <span class="grow"></span>
  <button id="wall" class="on" title="w">paint wall</button>
  <button id="erase" title="e">erase</button>
  <label>brush <input id="size" type="range" min="2" max="60" value="{brush}"
         style="vertical-align:middle"><span id="sizeval">{brush}</span></label>
  <button id="undo" title="ctrl+z">undo</button>
  <button id="reset" title="back to the model's own reading">reset</button>
  <button id="save">save</button>
  <button id="prev" {prev_disabled}>&larr; prev</button>
  <button id="next" {next_disabled}>next &rarr;</button>
  <span id="status"></span>
</header>

<div class="wrap" id="wrap" style="width:{w}px">
  <img id="plan" src="{plan_uri}" width="{w}" height="{h}">
  <canvas id="paint" width="{w}" height="{h}"></canvas>
</div>

<p class="hint">
  The overlay is what the model currently believes is a wall. Scrub off what is not a
  wall — door swing arcs, kitchen units, beds, baths, dimension lines — and paint in
  any wall it missed. Do not chase single pixels; the model trains at 512 and the
  brush is deliberately coarse.
  <br><br>
  <kbd>w</kbd> paint · <kbd>e</kbd> erase · <kbd>[</kbd> <kbd>]</kbd> brush size ·
  <kbd>ctrl</kbd>+<kbd>z</kbd> undo · <kbd>s</kbd> save · <kbd>n</kbd> save and next.
  Saving happens automatically when you move between plans.
</p>

<script>
const W = {w}, H = {h}, LID = "{lid}";
const cv = document.getElementById("paint"), ctx = cv.getContext("2d", {{willReadFrequently:true}});
const seed = new Image();
let mode = "wall", brush = {brush}, drawing = false, dirty = false;
const history = [];

seed.onload = () => {{ ctx.drawImage(seed, 0, 0, W, H); paintStyle(); }};
seed.src = "{mask_uri}";

function paintStyle() {{
  // The mask is stored as white-on-black; show it tinted so the plan reads through.
  const d = ctx.getImageData(0, 0, W, H);
  for (let i = 0; i < d.data.length; i += 4) {{
    const on = d.data[i] > 127;
    d.data[i] = on ? 214 : 0; d.data[i+1] = on ? 44 : 0; d.data[i+2] = on ? 60 : 0;
    d.data[i+3] = on ? 255 : 0;
  }}
  ctx.putImageData(d, 0, 0);
}}

function push() {{
  history.push(ctx.getImageData(0, 0, W, H));
  if (history.length > 40) history.shift();
}}

function at(e) {{
  const r = cv.getBoundingClientRect();
  return [ (e.clientX - r.left) * W / r.width, (e.clientY - r.top) * H / r.height ];
}}

function dab(x, y) {{
  ctx.globalCompositeOperation = mode === "wall" ? "source-over" : "destination-out";
  ctx.fillStyle = "rgba(214,44,60,1)";
  ctx.beginPath(); ctx.arc(x, y, brush / 2, 0, Math.PI * 2); ctx.fill();
  ctx.globalCompositeOperation = "source-over";
}}

let last = null;
cv.addEventListener("pointerdown", (e) => {{
  cv.setPointerCapture(e.pointerId); push(); drawing = true; dirty = true;
  last = at(e); dab(...last);
}});
cv.addEventListener("pointermove", (e) => {{
  if (!drawing) return;
  const p = at(e);
  // interpolate, or a fast drag leaves a dotted line
  const steps = Math.max(1, Math.hypot(p[0]-last[0], p[1]-last[1]) / (brush / 3));
  for (let i = 1; i <= steps; i++)
    dab(last[0] + (p[0]-last[0]) * i/steps, last[1] + (p[1]-last[1]) * i/steps);
  last = p;
}});
for (const ev of ["pointerup", "pointercancel", "pointerleave"])
  cv.addEventListener(ev, () => {{ drawing = false; }});

function setMode(m) {{
  mode = m;
  document.getElementById("wall").classList.toggle("on", m === "wall");
  document.getElementById("erase").classList.toggle("on", m === "erase");
}}
document.getElementById("wall").onclick = () => setMode("wall");
document.getElementById("erase").onclick = () => setMode("erase");
document.getElementById("size").oninput = (e) => {{
  brush = +e.target.value; document.getElementById("sizeval").textContent = brush;
}};
document.getElementById("undo").onclick = () => {{
  const s = history.pop(); if (s) ctx.putImageData(s, 0, 0);
}};
document.getElementById("reset").onclick = () => {{
  if (!confirm("Throw away your corrections and go back to the model's reading?")) return;
  push(); ctx.clearRect(0, 0, W, H); ctx.drawImage(seed, 0, 0, W, H); paintStyle();
  dirty = true;
}};

function toMaskPng() {{
  // Back to white-on-black: alpha is the signal, colour is only for display.
  const out = document.createElement("canvas");
  out.width = W; out.height = H;
  const o = out.getContext("2d");
  const src = ctx.getImageData(0, 0, W, H), dst = o.createImageData(W, H);
  for (let i = 0; i < src.data.length; i += 4) {{
    const on = src.data[i + 3] > 127 ? 255 : 0;
    dst.data[i] = dst.data[i+1] = dst.data[i+2] = on; dst.data[i+3] = 255;
  }}
  o.putImageData(dst, 0, 0);
  return out.toDataURL("image/png");
}}

async function save() {{
  const st = document.getElementById("status");
  st.textContent = "saving…"; st.className = "";
  const r = await fetch("/save?lid=" + LID, {{ method: "POST", body: toMaskPng() }});
  if (r.ok) {{ dirty = false; st.textContent = "saved"; st.className = "saved"; }}
  else st.textContent = "save failed";
}}
document.getElementById("save").onclick = save;

async function go(where) {{
  if (dirty) await save();
  location.href = where;
}}
document.getElementById("next").onclick = () => go("/?i={next_i}");
document.getElementById("prev").onclick = () => go("/?i={prev_i}");

addEventListener("keydown", (e) => {{
  if (e.key === "w") setMode("wall");
  else if (e.key === "e") setMode("erase");
  else if (e.key === "[") document.getElementById("size").value = brush = Math.max(2, brush - 3);
  else if (e.key === "]") document.getElementById("size").value = brush = Math.min(60, brush + 3);
  else if (e.key === "s") {{ e.preventDefault(); save(); }}
  else if (e.key === "n") go("/?i={next_i}");
  else if (e.key === "z" && (e.ctrlKey || e.metaKey)) document.getElementById("undo").click();
  else return;
  document.getElementById("sizeval").textContent = brush;
}});

addEventListener("beforeunload", (e) => {{ if (dirty) {{ e.preventDefault(); e.returnValue = ""; }} }});
</script>
"""


class Annotator:
    def __init__(self, store_root: Path | None, brush: int):
        self.store_root = store_root
        self.brush = brush
        self.ids = plan_ids(store_root)
        if not self.ids:
            raise SystemExit("no listings with a stage-5 plan artifact — run the pipeline first")
        self._cache: dict[str, tuple[str, str, int, int]] = {}

    def _assets(self, lid: str) -> tuple[str, str, int, int]:
        if lid not in self._cache:
            rgb = plan_image(lid, self.store_root)
            mask = seed_mask(lid, rgb)
            self._cache[lid] = (_png_data_uri(rgb), _png_data_uri(mask, "L"),
                                rgb.shape[1], rgb.shape[0])
        return self._cache[lid]

    def page(self, index: int) -> str:
        index = max(0, min(index, len(self.ids) - 1))
        lid = self.ids[index]
        plan_uri, mask_uri, w, h = self._assets(lid)
        return _PAGE.format(
            lid=lid, index=index + 1, total=len(self.ids), done=len(done_ids()),
            plan_uri=plan_uri, mask_uri=mask_uri, w=w, h=h, brush=self.brush,
            next_i=min(index + 1, len(self.ids) - 1), prev_i=max(index - 1, 0),
            next_disabled="disabled" if index >= len(self.ids) - 1 else "",
            prev_disabled="disabled" if index <= 0 else "")

    def save(self, lid: str, data_uri: bytes) -> None:
        LABEL_DIR.mkdir(parents=True, exist_ok=True)
        raw = data_uri.split(b",", 1)[1] if b"," in data_uri else data_uri
        img = Image.open(io.BytesIO(base64.b64decode(raw))).convert("L")
        arr = (np.array(img) > 127).astype(np.uint8) * 255
        Image.fromarray(arr).save(LABEL_DIR / f"{lid}.png")


def make_handler(app: Annotator):
    class Handler(BaseHTTPRequestHandler):
        def _send(self, body: bytes, status: int = 200, ctype: str = "text/html; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):                                        # noqa: N802
            url = urlparse(self.path)
            if url.path not in ("/", "/index.html"):
                self._send(b"not found", 404, "text/plain")
                return
            i = int((parse_qs(url.query).get("i") or ["0"])[0] or 0)
            self._send(app.page(i).encode())

        def do_POST(self):                                       # noqa: N802
            url = urlparse(self.path)
            if url.path != "/save":
                self._send(b"not found", 404, "text/plain")
                return
            lid = (parse_qs(url.query).get("lid") or [""])[0]
            if lid not in app.ids:
                self._send(b"unknown listing", 400, "text/plain")
                return
            body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            try:
                app.save(lid, body)
            except Exception as exc:                             # noqa: BLE001
                self._send(f"save failed: {exc!r}".encode(), 500, "text/plain")
                return
            self._send(b"ok", 200, "text/plain")

        def log_message(self, fmt, *args):
            pass

    return Handler


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--brush", type=int, default=14, help="starting brush diameter, px")
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--export", action="store_true",
                    help="write the training set and exit")
    ap.add_argument("--status", action="store_true", help="how many plans are corrected")
    args = ap.parse_args(argv)

    if args.status:
        ids, done = plan_ids(args.store), done_ids()
        print(f"{len(done)} of {len(ids)} plans corrected")
        todo = [i for i in ids if i not in done]
        if todo:
            print("next:", " ".join(todo[:10]) + (" …" if len(todo) > 10 else ""))
        return 0

    if args.export:
        m = export(args.store)
        print(f"exported {m['count']} labelled plans to {EXPORT_DIR}")
        if m["count"] < 15:
            print("  fewer than 15 — fine-tuning on this will overfit. Keep going.")
        return 0

    app = Annotator(args.store, args.brush)
    server = ThreadingHTTPServer(("127.0.0.1", args.port), make_handler(app))
    print(f"{len(app.ids)} plans, {len(done_ids())} already corrected")
    print(f"open http://127.0.0.1:{args.port}   (ctrl-c to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped. `--export` when you have twenty or so.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
