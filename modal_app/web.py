"""HTTP gateway to the GPU jobs.

Why: Modal's SDK speaks gRPC, which the agent's sandbox proxy does not pass.
Plain HTTPS to *.modal.run does work (verified). So the app is deployed once
from a normal machine, and everything after that - triggering runs, polling,
pulling results and images - happens over ordinary HTTP.

    modal deploy modal_app/web.py

Security notes, because this URL is public:
  * Every route requires a bearer token held in a Modal Secret. No token, no
    service.
  * Only a fixed allowlist of stage names can be started, with typed and
    bounded parameters. There is deliberately no route that runs arbitrary
    code - a public endpoint that could would be a standing liability on the
    account it runs under.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import modal

from .common import DATA_DIR, app, data_volume, gpu_image
from .gpu_validate import gsplat_train, inventory, mapanything, moge_speed

web_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "fastapi[standard]==0.115.*", "requests"
)

# The token lives in a Modal Secret, never in the repo:
#   modal secret create visit-it-gpu-token GPU_API_TOKEN=<a long random string>
token_secret = modal.Secret.from_name("visit-it-gpu-token")

# Bounded parameters per stage. Anything not listed is rejected.
STAGE_PARAMS: dict[str, dict[str, tuple[type, int, int]]] = {
    "moge_speed": {"n": (int, 1, 200)},
    "mapanything": {"max_groups": (int, 1, 40)},
    "gsplat_train": {"max_groups": (int, 1, 20), "iters": (int, 100, 20000)},
    "inventory": {},
}
FUNCS = {"moge_speed": moge_speed, "mapanything": mapanything,
         "gsplat_train": gsplat_train, "inventory": inventory}


@app.function(image=web_image, secrets=[token_secret],
              volumes={DATA_DIR: data_volume}, timeout=900)
@modal.asgi_app(label="visit-it-gpu-api")
def api():
    from fastapi import Depends, FastAPI, HTTPException, Request
    from fastapi.responses import Response

    web = FastAPI(title="visit-it GPU validation")

    def auth(request: Request) -> None:
        expected = os.environ.get("GPU_API_TOKEN", "")
        got = (request.headers.get("authorization") or "").removeprefix("Bearer ").strip()
        if not expected or got != expected:
            raise HTTPException(status_code=401, detail="bad or missing bearer token")

    def _validate(stage: str, params: dict) -> dict:
        if stage not in STAGE_PARAMS:
            raise HTTPException(400, f"unknown stage; allowed: {sorted(STAGE_PARAMS)}")
        spec = STAGE_PARAMS[stage]
        clean = {}
        for k, v in (params or {}).items():
            if k not in spec:
                raise HTTPException(400, f"param {k!r} not allowed for {stage}")
            typ, lo, hi = spec[k]
            try:
                val = typ(v)
            except (TypeError, ValueError):
                raise HTTPException(400, f"param {k!r} must be {typ.__name__}")
            if not (lo <= val <= hi):
                raise HTTPException(400, f"param {k!r} out of range [{lo},{hi}]")
            clean[k] = val
        return clean

    @web.get("/health")
    def health(_: None = Depends(auth)):
        data_volume.reload()
        root = Path(DATA_DIR)
        media = root / "media"
        return {
            "ok": True,
            "images_in_volume": sum(1 for p in media.rglob("*") if p.is_file()) if media.exists() else 0,
            "has_groups": (root / "room_groups.json").exists(),
            "results_present": sorted(p.name for p in root.glob("results_*.json")),
            "images_present": sorted(p.name for p in root.glob("splat_*.png")),
            "stages": sorted(STAGE_PARAMS),
        }

    @web.post("/start")
    def start(body: dict, _: None = Depends(auth)):
        stage = (body or {}).get("stage", "")
        params = _validate(stage, (body or {}).get("params") or {})
        call = FUNCS[stage].spawn(**params)
        return {"call_id": call.object_id, "stage": stage, "params": params}

    @web.get("/status/{call_id}")
    def status(call_id: str, _: None = Depends(auth)):
        fc = modal.FunctionCall.from_id(call_id)
        try:
            result = fc.get(timeout=0)
        # Order matters: OutputExpiredError SUBCLASSES modal's TimeoutError, so
        # catching the latter first would swallow it and report "running"
        # forever. And modal's TimeoutError is NOT a subclass of the stdlib one,
        # so both have to be named explicitly.
        except modal.exception.OutputExpiredError:
            return {"status": "error", "call_id": call_id,
                    "error": "output expired (results are kept 7 days); re-run the stage"}
        except (modal.exception.TimeoutError, TimeoutError):
            return {"status": "running", "call_id": call_id}
        except Exception as e:  # noqa: BLE001 - surface the real failure to the caller
            return {"status": "error", "call_id": call_id,
                    "error": f"{type(e).__name__}: {e}"}
        return {"status": "done", "call_id": call_id, "result": result}

    @web.get("/results")
    def results(_: None = Depends(auth)):
        data_volume.reload()
        out = {}
        for f in sorted(Path(DATA_DIR).glob("results_*.json")):
            out[f.name] = json.loads(f.read_text())
        return out

    @web.get("/images")
    def images(_: None = Depends(auth)):
        data_volume.reload()
        return {"images": sorted(p.name for p in Path(DATA_DIR).glob("splat_*.png"))}

    @web.get("/image/{name}")
    def image(name: str, _: None = Depends(auth)):
        if "/" in name or ".." in name or not name.endswith(".png"):
            raise HTTPException(400, "bad name")
        p = Path(DATA_DIR) / name
        if not p.exists():
            raise HTTPException(404, "not found")
        return Response(content=p.read_bytes(), media_type="image/png")

    return web
