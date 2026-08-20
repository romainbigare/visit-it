# Runbook — GPU validation on Modal

**Why you have to run this, not the agent:** Modal's client speaks gRPC, and the
sandbox the agent works in does not pass gRPC through its egress proxy (plain
HTTPS to `api.modal.com` returns 200; the gRPC channel cannot be established, and
the proxy documentation lists gRPC as unsupported). Everything below is written
and structure-checked; it just needs to be fired from a normal network.

## One-time setup

```bash
pip install modal
modal token set --token-id ak-... --token-secret as-...
```

## Run everything

```bash
cd visit-it
python -m eval.models.grouping          # writes eval/results/room_groups.json (CPU, instant)
modal run modal_app/gpu_validate.py::run_all
```

That uploads the golden set, runs all three GPU stages and pulls the results
back into `eval/results/`. Expect **10–20 minutes** and **$1–3** on an L40S
(most of the first run is building the CUDA image; later runs reuse the cache).

## Run stages individually

```bash
modal run modal_app/gpu_validate.py::upload          # push images + groups to the volume
modal run modal_app/gpu_validate.py::inventory       # confirm what landed
modal run modal_app/gpu_validate.py::moge_speed      # GPU timing baseline
modal run modal_app/gpu_validate.py::mapanything     # stage 3, multi-view
modal run modal_app/gpu_validate.py::gsplat_train    # stage 8, appearance
modal run modal_app/gpu_validate.py::fetch_results   # pull JSON back down
```

`gsplat_train` depends on `mapanything` having run — it starts from the point
clouds and poses that stage writes into the volume.

Pick a different GPU with `VISITIT_GPU=A100 modal run ...` (default `L40S`).

## What each stage answers

| Stage | Question | What "good" looks like |
|---|---|---|
| `moge_speed` | How much faster is a GPU than the 14.4 s/image we measured on 4 CPU cores? | Sub-second. The `instant` profile in `docs/VARIANTS.md` needs roughly 0.2–0.5 s |
| `mapanything` | Does multi-view reconstruction work on real agent photography (AD-4)? | Cameras spread apart (a **max baseline near zero means the reconstruction collapsed** — the classic failure), scene extent of a few metres, high mean confidence |
| `gsplat_train` | Does the stage 3 → stage 8 chain hold with **no COLMAP anywhere** (AD-6)? | Held-out PSNR above ~18 dB is recognisable; above ~22 dB is good for 3–8 views |

## Reading the results

Results land as `eval/results/results_*.json`, in the same shape as the CPU
results so they merge into `VALIDATION-REPORT.md`. Render-vs-truth image pairs
for each room are written into the Modal volume as
`splat_<group_id>_render.png` / `_gt.png`.

**Look at the diagnostics before the headline numbers.** A good-looking PSNR
with a near-zero camera baseline means the model put every camera in the same
place and learned a flat billboard — it will score well and be useless.

## Cost control

Modal bills per second and scales to zero, so idle time is free. The volumes
(`visit-it-data`, `visit-it-models`) persist between runs; delete them from the
Modal dashboard when finished if you want to avoid storage charges.

## Known risks on first run

- **gsplat compiles CUDA kernels at install.** The image is `nvidia/cuda:12.4.1-devel`
  precisely so `nvcc` is present. If the build fails, try the prebuilt wheel index:
  `pip install gsplat --index-url https://docs.gsplat.studio/whl/pt24cu124`, pinning
  torch to match.
- **MapAnything does not pin torch or CUDA** by design; the image pins torch 2.5.1 + cu124.
  If it complains, that pairing is the first thing to change.
- **Intrinsics and resolution must match.** `gsplat_train` renders at MapAnything's own
  output resolution and uses its intrinsics unchanged. Do not "helpfully" resize the
  images — an earlier draft did, and the intrinsics silently stopped applying.
