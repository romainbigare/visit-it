"""Drive the Modal GPU jobs over plain HTTPS.

The agent sandbox cannot speak gRPC, so it cannot use the Modal SDK. Once the
app is deployed once from a normal machine, this client triggers and polls the
GPU work over ordinary HTTP requests, which the sandbox proxy does allow.

    export VISITIT_GPU_URL=https://<workspace>--visit-it-gpu-api.modal.run
    export VISITIT_GPU_TOKEN=<the shared secret>

    python tools/gpu_client.py health
    python tools/gpu_client.py start mapanything --max-groups 12
    python tools/gpu_client.py wait <call_id>
    python tools/gpu_client.py results --save
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import requests

RESULTS_DIR = Path(__file__).resolve().parents[1] / "eval" / "results"
STAGES = ("moge_speed", "mapanything", "gsplat_train", "inventory")


def _cfg() -> tuple[str, str]:
    url = os.environ.get("VISITIT_GPU_URL", "").rstrip("/")
    token = os.environ.get("VISITIT_GPU_TOKEN", "")
    if not url:
        raise SystemExit("set VISITIT_GPU_URL to the deployed Modal endpoint")
    return url, token


def _headers() -> dict:
    _, token = _cfg()
    return {"Authorization": f"Bearer {token}"} if token else {}


def health() -> dict:
    url, _ = _cfg()
    r = requests.get(f"{url}/health", headers=_headers(), timeout=60)
    r.raise_for_status()
    return r.json()


def start(stage: str, **params) -> dict:
    if stage not in STAGES:
        raise SystemExit(f"unknown stage {stage!r}; one of {STAGES}")
    url, _ = _cfg()
    r = requests.post(f"{url}/start", headers=_headers(),
                      json={"stage": stage, "params": params}, timeout=120)
    r.raise_for_status()
    return r.json()


def status(call_id: str) -> dict:
    url, _ = _cfg()
    r = requests.get(f"{url}/status/{call_id}", headers=_headers(), timeout=120)
    r.raise_for_status()
    return r.json()


def wait(call_id: str, poll: int = 20, timeout_s: int = 5400) -> dict:
    """Poll until the job finishes. Long jobs are normal here - a MapAnything
    sweep plus a gsplat train is tens of minutes including image build."""
    t0 = time.time()
    last = ""
    while time.time() - t0 < timeout_s:
        s = status(call_id)
        state = s.get("status")
        if state == "done":
            return s
        if state == "error":
            raise SystemExit(f"job failed:\n{s.get('error')}")
        msg = f"  [{int(time.time()-t0):5}s] {state}"
        if msg != last:
            print(msg, flush=True)
            last = msg
        time.sleep(poll)
    raise SystemExit(f"timed out after {timeout_s}s; call_id={call_id}")


def results(save: bool = False) -> dict:
    url, _ = _cfg()
    r = requests.get(f"{url}/results", headers=_headers(), timeout=300)
    r.raise_for_status()
    payload = r.json()
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        for name, body in payload.items():
            (RESULTS_DIR / name).write_text(json.dumps(body, indent=2))
            print(f"  saved {name}")
    return payload


def fetch_images(dest: Path) -> int:
    """Download the render-vs-truth PNGs the gsplat stage writes."""
    url, _ = _cfg()
    r = requests.get(f"{url}/images", headers=_headers(), timeout=300)
    r.raise_for_status()
    names = r.json().get("images", [])
    dest.mkdir(parents=True, exist_ok=True)
    for n in names:
        b = requests.get(f"{url}/image/{n}", headers=_headers(), timeout=300)
        if b.ok:
            (dest / n).write_bytes(b.content)
    print(f"  {len(names)} images -> {dest}")
    return len(names)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("health")
    s = sub.add_parser("start"); s.add_argument("stage", choices=STAGES)
    s.add_argument("--max-groups", type=int); s.add_argument("--iters", type=int)
    s.add_argument("--n", type=int)
    st = sub.add_parser("status"); st.add_argument("call_id")
    w = sub.add_parser("wait"); w.add_argument("call_id"); w.add_argument("--poll", type=int, default=20)
    rs = sub.add_parser("results"); rs.add_argument("--save", action="store_true")
    im = sub.add_parser("images"); im.add_argument("--dest", type=Path,
                                                   default=RESULTS_DIR / "figures" / "gpu")
    a = ap.parse_args(argv)

    if a.cmd == "health":
        print(json.dumps(health(), indent=2))
    elif a.cmd == "start":
        params = {k: v for k, v in
                  (("max_groups", a.max_groups), ("iters", a.iters), ("n", a.n))
                  if v is not None}
        out = start(a.stage, **params)
        print(json.dumps(out, indent=2))
        print(f"\npoll with: python tools/gpu_client.py wait {out.get('call_id')}")
    elif a.cmd == "status":
        print(json.dumps(status(a.call_id), indent=2))
    elif a.cmd == "wait":
        r = wait(a.call_id, a.poll)
        res = r.get("result", {})
        print(json.dumps({k: v for k, v in res.items() if k != "results"}, indent=2))
    elif a.cmd == "results":
        p = results(a.save)
        for name, body in p.items():
            print(f"{name}: " + json.dumps(
                {k: v for k, v in body.items() if k != "results"})[:200])
    elif a.cmd == "images":
        fetch_images(a.dest)
    return 0


if __name__ == "__main__":
    sys.exit(main())
