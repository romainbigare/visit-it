"""The per-listing contact sheet — the single most valuable debugging artifact.

ARCHITECTURE §3 calls it that and Phase 0 proved the point twice over: a flat grey
render scored ~12 dB and read as *merely poor* in a results table, and only opening
the PNG showed it was nothing at all. Numbers hide that; pictures do not.

So this renders every stage's output for one listing on one page: the triaged
images, the vectorised plan with its rooms drawn on it, each room's polygon, the
assembly's cost matrix, the scale constraints with their residuals, and the shell's
footprint. Static HTML with inline SVG and data-URI thumbnails, so it opens with a
double-click on any machine and survives being emailed.
"""
from __future__ import annotations

import base64
import html
import io
import json
import logging
from pathlib import Path

from pipeline.core import geom
from pipeline.core.artifacts import ArtifactStore
from pipeline.core.ledger import Ledger

log = logging.getLogger("review.contact_sheet")

THUMB = (240, 180)


def _thumb_uri(path: Path, size=THUMB) -> str | None:
    try:
        from PIL import Image
        from pipeline.floorplan.preprocess import load_rgb
        # load_rgb, not Image.convert("RGB"): several plans are greyscale-plus-alpha
        # PNGs, and converting those straight to RGB composites them onto black —
        # which is how a perfectly good floor plan shows up in the debug view as a
        # solid black rectangle.
        im = Image.fromarray(load_rgb(path))
        im.thumbnail(size, Image.LANCZOS)
        if True:
            buf = io.BytesIO()
            im.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:  # noqa: BLE001
        return None


def _svg_polys(polys: list[tuple[list, str, str]], width: int = 420,
               height: int = 340, flip_y: bool = True) -> str:
    """Draw labelled polygons to scale. ``polys`` is ``[(points, fill, label)]``."""
    pts = [p for poly, _f, _l in polys for p in poly]
    if not pts:
        return "<div class='empty'>nothing to draw</div>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    pad = 12
    s = min((width - 2 * pad) / max(maxx - minx, 1e-6),
            (height - 2 * pad) / max(maxy - miny, 1e-6))

    def tx(x, y):
        px = pad + (x - minx) * s
        py = (pad + (maxy - y) * s) if flip_y else (pad + (y - miny) * s)
        return px, py

    parts = []
    for poly, fill, label in polys:
        d = " ".join(f"{tx(x, y)[0]:.1f},{tx(x, y)[1]:.1f}" for x, y in poly)
        parts.append(f'<polygon points="{d}" fill="{fill}" fill-opacity="0.55" '
                     f'stroke="#2f3540" stroke-width="1"/>')
        if label:
            cx, cy = geom.centroid(poly)
            px, py = tx(cx, cy)
            parts.append(f'<text x="{px:.1f}" y="{py:.1f}" font-size="9" '
                         f'text-anchor="middle" fill="#1b1f27">{html.escape(label)}</text>')
    return (f'<svg viewBox="0 0 {width} {height}" width="100%" '
            f'style="max-width:{width}px">{"".join(parts)}</svg>')


PALETTE = ["#8ec3e6", "#f0b47a", "#9fd6a5", "#e6a0b4", "#c4b0e6", "#e6d98e",
           "#9ecfc9", "#d6a58e"]


def _flag_list(flags: list[str]) -> str:
    if not flags:
        return '<span class="ok">no flags</span>'
    return " ".join(f'<span class="flag">{html.escape(f)}</span>' for f in flags)


def render(listing: dict, store_root: Path | None = None,
           golden_root: Path = Path("data/golden")) -> str:
    lid = listing["listing_id"]
    store = ArtifactStore(lid, store_root)
    manifest = store.read("0-triage")
    groups = store.read("2-grouping")
    geo = store.read("3-geometry")
    layouts = store.read("4-layout")
    plan = store.read("5-plan")
    assembly = store.read("6-assembly")
    scale = store.read("7-scale")
    shell = store.read("8-shell")
    scene = store.read("9-package")
    run = Ledger(lid, store_root).latest()

    out: list[str] = [_HEAD.format(
        lid=html.escape(lid),
        address=html.escape(listing.get("display_address") or ""),
        url=html.escape(listing.get("url") or "#"),
    )]

    # --- run ledger -------------------------------------------------------
    if run:
        rows = "".join(
            f"<tr><td>{s['stage']}</td><td class='{s['status']}'>{s['status']}</td>"
            f"<td>{s.get('seconds', 0):.2f}s</td>"
            f"<td>{'over' if s.get('over_budget') else ''} {s.get('budget_s') or ''}</td>"
            f"<td>{s.get('confidence') if s.get('confidence') is not None else ''}</td>"
            f"<td>{_flag_list(s.get('qa_flags') or [])}</td>"
            f"<td class='err'>{html.escape(s.get('error') or '')}</td></tr>"
            for s in run["stages"])
        out.append(f"""<section><h2>Run {html.escape(run['run_id'])}
          <small>{run['status']} · {run['total_seconds']}s · profile {run['profile']}</small></h2>
          <table><tr><th>stage</th><th>status</th><th>time</th><th>budget</th>
          <th>conf</th><th>flags</th><th>error</th></tr>{rows}</table></section>""")

    # --- stage 0 ----------------------------------------------------------
    if manifest:
        cards = []
        for im in manifest["images"]:
            uri = _thumb_uri(golden_root / im["path"])
            badge = f"{im['type']}" + (f" · {im['room_label']}" if im.get("room_label") else "")
            cards.append(f"""<figure><img src="{uri or ''}" alt="" loading="lazy">
              <figcaption>{html.escape(im['image_id'])} · {html.escape(badge)}
              {_flag_list(im.get('quality_flags') or [])}</figcaption></figure>""")
        out.append(f"""<section><h2>0 · Triage <small>{len(manifest['images'])} images ·
          {len(manifest['plans'])} plan(s) · {_flag_list(manifest['qa_flags'])}</small></h2>
          <div class="grid">{''.join(cards)}</div></section>""")

    # --- stage 5 ----------------------------------------------------------
    if plan:
        polys = [(r["polygon_px"], PALETTE[i % len(PALETTE)],
                  f"{r['room_id']} {r.get('label') or '?'}")
                 for i, r in enumerate(plan["rooms"])]
        plan_img = (_thumb_uri(Path(plan["source_image"]), (520, 520))
                    if plan.get("source_image") else None)
        rows = "".join(
            f"<tr><td>{r['room_id']}</td><td>{r.get('label') or '—'}</td>"
            f"<td>{r.get('area_m2') if r.get('area_m2') is not None else '—'}</td>"
            f"<td>{r.get('ocr_dims_m') or '—'}</td><td>{r['confidence']}</td>"
            f"<td>{_flag_list(r.get('qa_flags') or [])}</td></tr>"
            for r in plan["rooms"])
        cands = "".join(f"<li>{c['source']} → {c['px_per_metre']} px/m "
                        f"<small>(w {c['weight']}, {html.escape(c['detail'])})</small></li>"
                        for c in plan["scale_candidates"])
        out.append(f"""<section><h2>5 · Plan channel
          <small>{plan.get('scale_source')} · {plan.get('px_per_metre')} px/m ·
          conf {plan['confidence']} · {_flag_list(plan['qa_flags'])}</small></h2>
          <div class="cols">
            <div>{f'<img class="plan" src="{plan_img}">' if plan_img else ''}</div>
            <div>{_svg_polys(polys, flip_y=False)}</div>
          </div>
          <p>Scale candidates: <ul>{cands}</ul></p>
          <table><tr><th>room</th><th>label</th><th>m²</th><th>printed dims</th>
          <th>conf</th><th>flags</th></tr>{rows}</table></section>""")

    # --- stages 3/4 -------------------------------------------------------
    if layouts:
        cards = []
        for r in layouts["rooms"]:
            svg = _svg_polys([(r["polygon_m"], "#9fd6a5", "")], 200, 160)
            cards.append(f"""<figure class="poly">{svg}<figcaption>
              <b>{html.escape(r['room_id'])}</b> {r.get('room_label') or ''}<br>
              {r['area_m2']} m² · h {r.get('room_height_m')} m · conf {r['confidence']}<br>
              {_flag_list(r['qa_flags'])}</figcaption></figure>""")
        eng = (geo or {}).get("engine", "?")
        out.append(f"""<section><h2>3-4 · Geometry and layout
          <small>engine {html.escape(str(eng))} · {layouts['summary']['n_rooms']} rooms ·
          median ceiling {layouts['summary']['median_ceiling_m']} m ·
          {_flag_list(layouts['qa_flags'])}</small></h2>
          <div class="grid">{''.join(cards)}</div></section>""")

    # --- stage 6 ----------------------------------------------------------
    if assembly:
        cm = assembly.get("cost_matrix") or {}
        head = "".join(f"<th>{html.escape(p)}</th>" for p in cm.get("plan_rooms", []))
        chosen = {(m["room_id"], m["plan_room_id"]) for m in assembly["matches"]}
        body = ""
        for i, rid in enumerate(cm.get("rooms", [])):
            cells = "".join(
                f"<td class='{'pick' if (rid, p) in chosen else ''}'>{v}</td>"
                for p, v in zip(cm["plan_rooms"], cm["cost"][i]))
            body += f"<tr><th>{html.escape(rid)}</th>{cells}</tr>"
        mrows = "".join(
            f"<tr><td>{m['room_id']}</td><td>{m['plan_room_id']}</td><td>{m['cost']}</td>"
            f"<td>{m['margin']}</td><td>{m['fit_iou']}</td><td>{m['confidence']}</td>"
            f"<td><code>{html.escape(json.dumps(m['cost_breakdown']))}</code></td></tr>"
            for m in assembly["matches"])
        out.append(f"""<section><h2>6 · Assembly
          <small>{assembly['method']} · conf {assembly['confidence']} ·
          IoU {assembly['refinement']['mean_iou_after']} ·
          {_flag_list(assembly['qa_flags'])}</small></h2>
          <table><tr><th>room ↓ / polygon →</th>{head}</tr>{body}</table>
          <table><tr><th>room</th><th>polygon</th><th>cost</th><th>margin</th>
          <th>IoU</th><th>conf</th><th>breakdown</th></tr>{mrows}</table>
          <p class="note">Unmatched rooms: {', '.join(assembly['unmatched_rooms']) or '—'} ·
          unmatched polygons: {', '.join(assembly['unmatched_plan_rooms']) or '—'}</p>
          </section>""")

    # --- stage 7 ----------------------------------------------------------
    if scale:
        rows = "".join(
            f"<tr><td>{c['kind']}</td><td>{html.escape(c.get('detail') or '')}</td>"
            f"<td>{c['target']}</td><td>{c['observed']}</td><td>{c['power']}</td>"
            f"<td>{c['weight']}</td><td>{c['residual_pct']}%</td>"
            f"<td>{'yes' if c['used'] else '<b>rejected</b>'}</td></tr>"
            for c in scale["constraints"])
        sc = scale["self_consistency"]
        out.append(f"""<section><h2>7 · Scale
          <small>×{scale['scale']} · quality {scale['quality']} ·
          rms {scale['residual_rms_pct']}% · {_flag_list(scale['qa_flags'])}</small></h2>
          <p class="note">Self-consistency: median {sc['median_abs_pct']}% over
          {sc['n_rooms_checked']} room(s), {sc['within_10pct_frac']} within ±10%.
          Cross-check: {html.escape(json.dumps(scale['cross_check']))}.
          Plausibility: {html.escape(json.dumps(scale['plausibility']))}.</p>
          <table><tr><th>kind</th><th>detail</th><th>target</th><th>observed</th>
          <th>power</th><th>weight</th><th>residual</th><th>used</th></tr>{rows}</table>
          </section>""")

    # --- stages 8/9 -------------------------------------------------------
    if shell and scene:
        polys = [(r["polygon_m"], PALETTE[i % len(PALETTE)], r.get("display_name") or "")
                 for i, r in enumerate(scene["rooms"])]
        out.append(f"""<section><h2>8-9 · Shell and scene
          <small>{shell['glb']['triangles']} triangles · {shell['glb']['bytes']} bytes ·
          {'within' if shell['glb']['within_budget'] else 'OVER'} budget ·
          tier {scene['tier']} · conf {scene['confidence']}</small></h2>
          {_svg_polys(polys, 520, 400)}
          <p class="note">{len(scene['waypoints'])} waypoints,
          {len(scene['waypoint_edges'])} edges · advertised
          {scene['advertised_area_m2']} m² · {_flag_list(scene['qa_flags'])}</p>
          </section>""")

    out.append("</main></body></html>")
    return "\n".join(out)


_HEAD = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{lid} — contact sheet</title>
<style>
:root {{ color-scheme: light; }}
body {{ margin:0; font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  color:#1b1f27; background:#f6f5f2; }}
header {{ padding:18px 22px; background:#1b1f27; color:#f2f0ec; }}
header a {{ color:#9fc6e6; }}
main {{ padding:16px 22px 60px; max-width:1180px; }}
section {{ background:#fff; border:1px solid #e0ddd6; border-radius:10px;
  padding:14px 18px; margin-bottom:18px; }}
h2 {{ font-size:16px; margin:0 0 12px; }}
h2 small {{ font-weight:400; color:#6b6f78; font-size:12px; margin-left:8px; }}
.grid {{ display:flex; flex-wrap:wrap; gap:10px; }}
figure {{ margin:0; width:240px; }}
figure img {{ width:100%; border-radius:6px; display:block; background:#e8e6e0; }}
figure.poly {{ width:210px; }}
figcaption {{ font-size:11px; color:#4b4f58; margin-top:4px; word-break:break-word; }}
.cols {{ display:flex; gap:16px; flex-wrap:wrap; align-items:flex-start; }}
.cols > div {{ flex:1 1 300px; }}
img.plan {{ max-width:100%; border-radius:6px; border:1px solid #ddd; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin-top:10px; }}
th,td {{ border:1px solid #e4e1da; padding:4px 7px; text-align:left; }}
th {{ background:#f0eee9; font-weight:600; }}
td.pick {{ background:#dff0d8; font-weight:700; }}
td.ok, .ok {{ color:#2e7d32; }}
td.failed {{ color:#c62828; font-weight:600; }}
td.skipped {{ color:#8a6d1f; }}
td.err {{ color:#c62828; font-size:11px; }}
.flag {{ display:inline-block; background:#fdf0d5; border:1px solid #e8d9a8;
  border-radius:4px; padding:0 5px; font-size:10px; margin:1px 2px; }}
.note {{ font-size:12px; color:#4b4f58; }}
.empty {{ color:#8a8d94; font-size:12px; }}
svg {{ background:#faf9f6; border:1px solid #e8e6e0; border-radius:6px; }}
code {{ font-size:10px; }}
</style></head><body>
<header><b>{lid}</b> — {address} · <a href="{url}" target="_blank" rel="noopener">listing</a></header>
<main>"""
