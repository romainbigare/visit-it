"""The review console (AD-14, ROADMAP S3/S4/S6).

    python -m services.review.server --port 8080

Three screens, and one goal behind all of them: **make a correction cost seconds,
not minutes.** The report's operational finding is that human review time rivals
all compute, so the number that matters here is M11 — median minutes per reviewed
listing — and every interaction is shaped by it.

**Queue** — every listing, sorted by how much a human is likely to be needed:
confidence, QA flags, and whether the stages completed. This is the auto-QA
routing surface; calibrating it properly is Sprint 13 work, but the ordering is
here now so the shape is right.

**Contact sheet** — every stage's output for one listing on one page.

**Fix actions** — the two Phase 1 asks:

* the *plan-vectorisation overlay editor* (S4): relabel or drop a room the
  vectoriser got wrong, or set the scale by hand, then re-run 5→9;
* the *assignment nudge* (S6): drag a room chip onto a different polygon and
  re-run 6→9, which takes about a second because the geometry pass is untouched.

Stdlib ``http.server`` on purpose. It is an internal tool, and a dependency-free
one still runs in two years when the framework we would have picked does not.
"""
from __future__ import annotations

import argparse
import html
import json
import logging
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from pipeline.core import ArtifactStore, Ledger, run_listing
from pipeline.core.overrides import clear as clear_overrides
from pipeline.core.overrides import load as load_overrides
from pipeline.core.overrides import merge as merge_overrides
from pipeline.core.overrides import rerun_stage_for

from .contact_sheet import render as render_sheet

log = logging.getLogger("review.server")

#: Flags that mean "a person should look before this ships". Sprint 13 calibrates
#: these against measured reviewer decisions; today they are the honest list of
#: what we know goes wrong.
BLOCKING_FLAGS = {
    "no_plan_scale", "no_rooms_found", "poor_polygon_fit", "empty_shell",
    "self_consistency_outside_10pct", "cross_model_scale_disagreement",
    "room_over_12m_across", "under_80pct_plausible_ceilings", "large_scale_correction",
    "plan_area_disagrees_with_stated", "multiple_plan_outlines",
}


def listing_rows(listings: dict[str, dict], store_root: Path | None) -> list[dict]:
    rows = []
    for lid, listing in listings.items():
        store = ArtifactStore(lid, store_root)
        scene = store.read("9-package")
        plan = store.read("5-plan")
        run = Ledger(lid, store_root).latest()
        flags = set(scene.get("qa_flags", []) if scene else [])
        if plan:
            flags |= set(plan.get("qa_flags", []))
        blocking = sorted(flags & BLOCKING_FLAGS)
        conf = scene.get("confidence") if scene else None
        rows.append({
            "listing_id": lid,
            "address": listing.get("display_address") or "",
            "status": (run or {}).get("status", "not run"),
            "stages_ok": sum(1 for s in (run or {}).get("stages", []) if s["status"] == "ok"),
            "confidence": conf,
            "rooms": len(scene.get("rooms", [])) if scene else 0,
            "blocking": blocking,
            "n_flags": len(flags),
            "overrides": bool(load_overrides(lid, store_root)),
            # Lowest confidence and most blocking flags first — the queue is
            # ordered by "how likely is a person needed", not by listing id.
            "priority": (len(blocking) * 2 + (1 - (conf or 0)) * 3 +
                         (5 if not scene else 0)),
        })
    return sorted(rows, key=lambda r: -r["priority"])


class Console:
    def __init__(self, golden: Path, store_root: Path | None):
        self.golden = golden
        self.store_root = store_root
        payload = json.loads((golden / "golden_set.json").read_text())
        self.listings = {l["listing_id"]: l for l in payload["listings"]}
        self.lock = threading.Lock()

    # -- pages ------------------------------------------------------------
    def queue(self) -> str:
        rows = listing_rows(self.listings, self.store_root)
        need = [r for r in rows if r["blocking"] or r["status"] != "ok"]
        body = "".join(
            f"""<tr class="{'blocked' if r['blocking'] else ''}">
              <td><a href="/listing/{r['listing_id']}">{r['listing_id']}</a>
                {'<span class="edit">edited</span>' if r['overrides'] else ''}</td>
              <td>{html.escape(r['address'][:44])}</td>
              <td class="{r['status']}">{r['status']}</td>
              <td>{r['stages_ok']}/10</td><td>{r['rooms']}</td>
              <td>{r['confidence'] if r['confidence'] is not None else '—'}</td>
              <td>{' '.join(f'<span class=flag>{html.escape(f)}</span>' for f in r['blocking']) or '—'}</td>
              <td>{r['n_flags']}</td></tr>"""
            for r in rows)
        return _page("Review queue", f"""
          <p class="note">{len(rows)} listings · <b>{len(need)}</b> want a look
          (blocking flags or an incomplete run). Ordered by how likely a person is
          to be needed, not by id — that ordering is what M11 measures.</p>
          <table><tr><th>listing</th><th>address</th><th>status</th><th>stages</th>
          <th>rooms</th><th>conf</th><th>blocking flags</th><th>all</th></tr>
          {body}</table>""")

    def listing(self, lid: str) -> str:
        if lid not in self.listings:
            return _page("Not found", f"<p>Unknown listing {html.escape(lid)}</p>")
        store = ArtifactStore(lid, self.store_root)
        plan = store.read("5-plan")
        assembly = store.read("6-assembly")
        layouts = store.read("4-layout")
        ov = load_overrides(lid, self.store_root)

        # --- assignment nudge -------------------------------------------
        nudge = "<p class='note'>Stage 6 has not run for this listing.</p>"
        if assembly and plan:
            plan_opts = "".join(
                f"<option value='{p['room_id']}'>{p['room_id']} — {p.get('label') or '?'}"
                f" ({p.get('area_m2')} m²)</option>" for p in plan["rooms"])
            rows = ""
            for m in assembly["matches"]:
                alts = ", ".join(f"{a['plan_room_id']}@{a['cost']}"
                                 for a in m.get("alternatives", [])[:3])
                rows += f"""<tr>
                  <td><b>{m['room_id']}</b></td>
                  <td><select name="pin_{m['room_id']}">
                      <option value="">{m['plan_room_id']} (as solved)</option>
                      {plan_opts}</select></td>
                  <td>{m['cost']}</td><td>{m['margin']}</td><td>{m['fit_iou']}</td>
                  <td>{m['confidence']}{' · <b>pinned</b>' if m.get('edited_by_human') else ''}</td>
                  <td class="alts">{alts}</td></tr>"""
            for rid in assembly.get("unmatched_rooms", []):
                rows += f"""<tr><td><b>{rid}</b></td>
                  <td><select name="pin_{rid}"><option value="">— unmatched —</option>
                  {plan_opts}</select></td><td colspan="4">not placed</td>
                  <td class="alts"></td></tr>"""
            nudge = f"""<form method="post" action="/apply/{lid}">
              <input type="hidden" name="section" value="assembly">
              <table><tr><th>room</th><th>put it in</th><th>cost</th><th>margin</th>
              <th>IoU</th><th>conf</th><th>runners-up</th></tr>{rows}</table>
              <button type="submit">Apply and re-run 6→9</button>
              <span class="note">≈1 s — the geometry pass is untouched.</span>
              </form>"""

        # --- plan overlay editor ----------------------------------------
        editor = "<p class='note'>Stage 5 has not run for this listing.</p>"
        if plan:
            labels = ["", "living_room", "bedroom", "kitchen", "bathroom", "dining_room",
                      "hallway", "reception", "study", "utility", "wc", "storage",
                      "balcony", "garden", "other_room"]
            rows = ""
            for r in plan["rooms"]:
                opts = "".join(
                    f"<option value='{l}'{' selected' if l == (r.get('label') or '') else ''}>"
                    f"{l or '(unchanged)'}</option>" for l in labels)
                rows += f"""<tr><td><b>{r['room_id']}</b></td>
                  <td><select name="label_{r['room_id']}">{opts}</select></td>
                  <td>{r.get('area_m2')}</td><td>{r.get('ocr_dims_m') or '—'}</td>
                  <td>{r.get('seeded_by', '?')}</td><td>{r['confidence']}</td>
                  <td><input type="checkbox" name="drop_{r['room_id']}"
                     {'checked' if False else ''}></td></tr>"""
            editor = f"""<form method="post" action="/apply/{lid}">
              <input type="hidden" name="section" value="plan">
              <p class="note">Scale is <b>{plan.get('px_per_metre')}</b> px/m from
              <b>{plan.get('scale_source')}</b>. Override it only if the printed
              dimensions were misread — it moves every room at once.</p>
              <label>px per metre <input name="px_per_metre" type="number" step="0.01"
                placeholder="{plan.get('px_per_metre') or ''}"></label>
              <table><tr><th>room</th><th>label</th><th>m²</th><th>printed</th>
              <th>seeded by</th><th>conf</th><th>drop</th></tr>{rows}</table>
              <button type="submit">Apply and re-run 5→9</button>
              <span class="note">≈5 s — the plan is re-vectorised, the photos are not.</span>
              </form>"""

        ovr_html = (f"<pre>{html.escape(json.dumps(ov, indent=2))}</pre>"
                    f"<form method='post' action='/clear/{lid}'>"
                    f"<button type='submit'>Clear all overrides and re-run</button></form>"
                    if ov else "<p class='note'>No human corrections on this listing.</p>")
        layout_note = ""
        if layouts:
            bad = [r["room_id"] for r in layouts["rooms"]
                   if "ceiling_implausible" in r["qa_flags"]]
            if bad:
                layout_note = (f"<p class='warn'>Rooms with implausible ceilings: "
                               f"{', '.join(bad)} — the geometry, not the assignment.</p>")

        return _page(f"{lid}", f"""
          <p><a href="/">← queue</a> ·
             <a href="/sheet/{lid}">full contact sheet</a> ·
             <a href="{html.escape(self.listings[lid].get('url') or '#')}"
                target="_blank" rel="noopener">listing</a></p>
          {layout_note}
          <h2>Assignment nudge</h2>{nudge}
          <h2>Plan overlay editor</h2>{editor}
          <h2>Stored corrections</h2>{ovr_html}""")

    # -- actions ----------------------------------------------------------
    def apply(self, lid: str, form: dict[str, list[str]]) -> str:
        section = (form.get("section") or ["assembly"])[0]
        patch: dict = {}
        if section == "assembly":
            pins = {k[4:]: v[0] for k, v in form.items()
                    if k.startswith("pin_") and v and v[0]}
            if pins:
                patch = {"assembly": {"pin": pins}}
        else:
            rooms: dict[str, dict] = {}
            for k, v in form.items():
                if k.startswith("label_") and v and v[0]:
                    rooms.setdefault(k[6:], {})["label"] = v[0]
                if k.startswith("drop_"):
                    rooms.setdefault(k[5:], {})["drop"] = True
            plan_patch: dict = {}
            if rooms:
                plan_patch["rooms"] = rooms
            ppm = (form.get("px_per_metre") or [""])[0]
            if ppm.strip():
                plan_patch["px_per_metre"] = float(ppm)
            if plan_patch:
                patch = {"plan": plan_patch}
        if not patch:
            return self.listing(lid)
        with self.lock:
            merge_overrides(lid, patch, self.store_root)
            stage = rerun_stage_for(list(patch))
            rec = run_listing(self.listings[lid], from_stage=stage,
                              golden_root=self.golden, store_root=self.store_root)
        return _page("Re-ran", f"""<p>Applied to <b>{html.escape(lid)}</b>, re-ran from
          <b>{stage}</b>: {rec.status} in {rec.total_seconds}s.</p>
          <p><a href="/listing/{lid}">back to the listing</a> ·
             <a href="/sheet/{lid}">contact sheet</a></p>""")

    def clear(self, lid: str) -> str:
        with self.lock:
            clear_overrides(lid, None, self.store_root)
            rec = run_listing(self.listings[lid], from_stage="5-plan",
                              golden_root=self.golden, store_root=self.store_root)
        return _page("Cleared", f"<p>Corrections removed and re-run: {rec.status}.</p>"
                                f"<p><a href='/listing/{lid}'>back</a></p>")

    def sheet(self, lid: str) -> str:
        return render_sheet(self.listings[lid], self.store_root, self.golden)


def _page(title: str, body: str) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)} — visit-it review</title><style>
body {{ margin:0; font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif;
  color:#1b1f27; background:#f6f5f2; }}
header {{ background:#1b1f27; color:#f2f0ec; padding:14px 22px; font-weight:600; }}
header a {{ color:#9fc6e6; text-decoration:none; }}
main {{ padding:16px 22px 60px; max-width:1180px; }}
h2 {{ font-size:16px; margin:22px 0 8px; }}
table {{ border-collapse:collapse; width:100%; font-size:12px; margin:8px 0 14px; }}
th,td {{ border:1px solid #e4e1da; padding:5px 8px; text-align:left; }}
th {{ background:#f0eee9; }}
tr.blocked {{ background:#fff6f5; }}
td.ok {{ color:#2e7d32; }} td.partial {{ color:#8a6d1f; }} td.failed {{ color:#c62828; }}
.flag {{ display:inline-block; background:#fdf0d5; border:1px solid #e8d9a8;
  border-radius:4px; padding:0 5px; font-size:10px; margin:1px; }}
.edit {{ background:#dff0d8; border:1px solid #b6d7a8; border-radius:4px;
  padding:0 5px; font-size:10px; }}
.note {{ color:#5a5e66; font-size:12px; }}
.warn {{ color:#8a4b1f; background:#fdf1e3; border:1px solid #e8cfa8;
  border-radius:6px; padding:8px 10px; font-size:12px; }}
.alts {{ color:#6b6f78; font-size:11px; }}
button {{ font:inherit; padding:6px 14px; border-radius:6px; cursor:pointer;
  background:#1b1f27; color:#f2f0ec; border:none; }}
select,input {{ font:inherit; font-size:12px; padding:2px 4px; }}
pre {{ background:#fff; border:1px solid #e4e1da; border-radius:6px; padding:10px;
  font-size:11px; overflow:auto; }}
a {{ color:#1a5fa8; }}
</style></head><body><header><a href="/">visit-it review console</a></header>
<main>{body}</main></body></html>"""


def make_handler(console: Console):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def _send(self, body: str, status: int = 200) -> None:
            data = body.encode()
            self.send_response(status)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            u = urlparse(self.path)
            parts = [p for p in u.path.split("/") if p]
            try:
                if not parts:
                    self._send(console.queue())
                elif parts[0] == "listing" and len(parts) > 1:
                    self._send(console.listing(parts[1]))
                elif parts[0] == "sheet" and len(parts) > 1:
                    self._send(console.sheet(parts[1]))
                elif parts[0] == "api" and parts[1:2] == ["queue"]:
                    body = json.dumps(listing_rows(console.listings, console.store_root),
                                      indent=2)
                    data = body.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    self.wfile.write(data)
                else:
                    self._send(_page("Not found", "<p>No such page.</p>"), 404)
            except Exception:  # noqa: BLE001
                log.error("GET %s failed\n%s", self.path, traceback.format_exc())
                self._send(_page("Error", f"<pre>{html.escape(traceback.format_exc())}</pre>"),
                           500)

        def do_POST(self) -> None:  # noqa: N802
            parts = [p for p in urlparse(self.path).path.split("/") if p]
            n = int(self.headers.get("Content-Length", 0))
            form = parse_qs(self.rfile.read(n).decode())
            try:
                if parts[0] == "apply":
                    self._send(console.apply(parts[1], form))
                elif parts[0] == "clear":
                    self._send(console.clear(parts[1]))
                else:
                    self._send(_page("Not found", "<p>No such action.</p>"), 404)
            except Exception:  # noqa: BLE001
                log.error("POST %s failed\n%s", self.path, traceback.format_exc())
                self._send(_page("Error", f"<pre>{html.escape(traceback.format_exc())}</pre>"),
                           500)

        def log_message(self, fmt, *args):
            log.info("%s %s", self.address_string(), fmt % args)

    return Handler


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--golden", type=Path, default=Path("data/golden"))
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8080)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    console = Console(a.golden, a.store)
    srv = ThreadingHTTPServer((a.host, a.port), make_handler(console))
    print(f"review console on http://{a.host}:{a.port}/  ({len(console.listings)} listings)")
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
