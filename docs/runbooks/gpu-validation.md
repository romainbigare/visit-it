# Runbook — GPU validation

## Modal needs a payment method (verified 20 Aug 2026)

Modal's billing docs state plainly: *"you must have a payment method on file in
order to use Modal."* That is **not** specific to L40S, or even to GPUs —
switching to a T4 does not help, and neither does CPU-only. The $30/month
Starter credit still requires a card on file to claim.

The Modal path is written and structure-checked, so it is kept below, but it is
parked until someone is willing to add a card. **Use the Colab route.**

---

# Route A — Google Colab (free, no card) ← use this

Everything runs from `notebooks/gpu_validation_colab.ipynb`.

1. <https://colab.research.google.com> → **File → Upload notebook** → pick
   `notebooks/gpu_validation_colab.ipynb`.
2. **Runtime → Change runtime type → T4 GPU.**
3. Run all cells.
4. Cell 5 shows render-vs-truth pairs, cell 6 prints the numbers to paste back,
   cell 7 downloads a zip.

Roughly **15–30 minutes**, most of it installing gsplat. The repo must be public
for the clone to work unauthenticated.

Nothing needs uploading: the notebook re-fetches only the ~108 photos the room
groups reference, from the URLs already in `golden_set.json`.

### Kaggle, if Colab's quota runs out

Kaggle gives a **fixed 30 h/week** (Colab's is a variable 15–30 h) on a P100
16 GB or 2×T4, also no card, also full CUDA. Same notebook — but enable
**Settings → Internet** first, which is off by default and will otherwise break
every pip install.

**Not Hugging Face ZeroGPU**: it cannot compile CUDA extensions at runtime, so
gsplat will not build there.

### Or any GPU machine

```bash
python tools/gpu_validate_standalone.py --all --max-groups 12 --iters 1500
```

Runs anywhere with CUDA — no Modal, no volumes, no upload step.

## What each stage answers

| Stage | Question | What "good" looks like |
|---|---|---|
| MoGe timing | How much faster than the 14.4 s/image measured on 4 CPU cores? | Sub-second; the `instant` profile needs ~0.2–0.5 s |
| MapAnything | Does multi-view reconstruction work on real agent photography (AD-4)? | Cameras spread apart (**a max baseline near zero means the reconstruction collapsed** — the classic failure), scene extent of a few metres, high mean confidence |
| gsplat | Does the stage 3 → 8 chain hold with **no COLMAP anywhere** (AD-6)? | Held-out PSNR above ~18 dB is recognisable; above ~22 dB is good for 3–8 views |

**Read the diagnostics before the headline numbers.** A good PSNR with a
near-zero camera baseline means every camera landed in the same place and the
model learned a flat billboard — it scores well and is useless.

---

# Route B — Modal (parked: needs a card)

, everything else over HTTP

Modal has **no REST control plane** — `modal deploy` can only be done with the
gRPC SDK, so that step must run on a normal machine. But once the app is
deployed, its web endpoint is ordinary HTTPS, and `*.modal.run` *is* reachable
from the agent sandbox (verified: it returns proper HTTP responses). So the
division of labour is:

| Step | Who | Why |
|---|---|---|
| Deploy + upload data | **you**, once | gRPC only |
| Trigger runs, poll, fetch results and images, iterate on failures | **the agent** | plain HTTPS |

That matters because the first GPU run usually needs a fix or two, and the
agent can do those rounds itself instead of relaying every one through you.

## One-time setup (on your machine)

```bash
pip install modal
modal token set --token-id ak-... --token-secret as-...

# A shared secret protecting the public endpoint. Any long random string.
python -c "import secrets; print(secrets.token_urlsafe(32))"
modal secret create visit-it-gpu-token GPU_API_TOKEN=<that string>

cd visit-it
python -m eval.models.grouping                        # CPU, instant
modal run -m modal_app.gpu_validate::upload           # push the golden set
modal deploy -m modal_app.web                         # prints the public URL
```

Then send the agent the URL and the token. It sets:

```bash
export VISITIT_GPU_URL=https://<workspace>--visit-it-gpu-api.modal.run
export VISITIT_GPU_TOKEN=<the string>
```

and drives everything with `tools/gpu_client.py`:

```bash
python tools/gpu_client.py health
python tools/gpu_client.py start mapanything --max-groups 12
python tools/gpu_client.py wait <call_id>
python tools/gpu_client.py start gsplat_train --max-groups 4 --iters 1500
python tools/gpu_client.py results --save
python tools/gpu_client.py images
```

### Endpoint safety

The URL is public, so: every route needs the bearer token, and only a fixed
allowlist of stage names with typed, range-checked parameters can be started.
There is deliberately no route that executes arbitrary code — a public endpoint
that could run anything on your GPU account would be a standing liability.

## Run everything

```bash
cd visit-it
python -m eval.models.grouping          # writes eval/results/room_groups.json (CPU, instant)
modal run -m modal_app.gpu_validate::run_all
```

That uploads the golden set, runs all three GPU stages and pulls the results
back into `eval/results/`. Expect **10–20 minutes** and **$1–3** on an L40S
(most of the first run is building the CUDA image; later runs reuse the cache).

## Run stages individually

```bash
modal run -m modal_app.gpu_validate::upload          # push images + groups to the volume
modal run -m modal_app.gpu_validate::inventory       # confirm what landed
modal run -m modal_app.gpu_validate::moge_speed      # GPU timing baseline
modal run -m modal_app.gpu_validate::mapanything     # stage 3, multi-view
modal run -m modal_app.gpu_validate::gsplat_train    # stage 8, appearance
modal run -m modal_app.gpu_validate::fetch_results   # pull JSON back down
```

`gsplat_train` depends on `mapanything` having run — it starts from the point
clouds and poses that stage writes into the volume.

Pick a different GPU with `VISITIT_GPU=A100 modal run -m ...` (default `L40S`).

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
