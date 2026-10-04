# VoxelFlight cloud GPU backend (Modal)

Signed-in visitors upload a drone video + GPS telemetry on https://voxelflight-3d.web.app/engine, a Modal GPU runs
`vf2/run_pipeline.py` (fast mode), and the existing viewer shows the model and offers the downloads.
**Prepared, not deployed.** Nothing here has run on Modal yet.

| File | Purpose |
|---|---|
| `api.py` | FastAPI app with the same `/api/*` job contract as `public_server.py` / `mac_server.py` (public mode). Imports without Modal |
| `worker.py` | Upload checks (`inspect_inputs`, CPU side), the GPU job (`run_job`) and packaging into the viewer's result files (`package_web`) |
| `modal_app.py` | Modal images, Volumes, Dict, the GPU function, the web endpoint, `download_weights`, `selftest`, and a `smoke` entrypoint |
| `test_contract.py` | Local contract tests: fake GPU runner, test-only auth bypass (`VF_INSECURE_TEST_AUTH=1`) |
| `root_env.patch` | Optional, recommended: `VF_ROOT` env var for the hard-coded A100 `ROOT` in `common.py`, `run_pipeline.py`, `ingest_video.py`. Also keeps the inherited `LD_LIBRARY_PATH` |
| `site_enable.patch` | The viewer changes needed besides `apiOrigin` (see "Website") |

## Morning steps

Run everything from the repository root.

```bash
# 1. Modal CLI. The system python3 is 3.9, which only gets modal 1.2.x; this checkout's 3.12 gets 1.6.x (what was import-checked)
.venv-mac/bin/python3.12 -m venv .venv-modal && source .venv-modal/bin/activate
pip install modal
modal token new                      # browser login; creates the workspace and its Starter plan

# 2. (recommended) explicit root path; the image also links the A100 path, so the code runs without it
patch -p1 --dry-run < vf2/cloud/root_env.patch && patch -p1 < vf2/cloud/root_env.patch

# 3. Build the GPU image (first time 20–40 min: CUDA devel base, torch, DPVO compile) and fill the weights Volume
modal run vf2/cloud/modal_app.py::download_weights     # MapAnything 4.9 GB, DA3-GIANT, DINOv2, dpvo.pth
modal run vf2/cloud/modal_app.py::selftest             # expect OK for torch, open3d, pipeline, da3, dpvo and 3 weights

# 4. One real job without the website (60 s Zurich clip, about 4–6 min on an A100 including cold start)
modal run vf2/cloud/modal_app.py --video data/zurich-recovered-61201-63000/flight.mp4 \
  --telemetry data/zurich-recovered-61201-63000/video_telemetry.csv --calibration data/zurich-recovered-61201-63000/camera.json
#    prints "complete" and the output files. On failure: modal volume get vf-jobs /<job id>/pipeline.log .

# 5. Deploy the API + GPU function
modal deploy vf2/cloud/modal_app.py
#    prints  https://<workspace>--voxelflight-cloud-web.modal.run
curl https://<workspace>--voxelflight-cloud-web.modal.run/api/health   # {"ready": true, ...}

# 6. Website: set the API origin, apply the viewer patch, test, rebuild + deploy
#    viewer/firebase/settings.json -> "apiOrigin": "https://<workspace>--voxelflight-cloud-web.modal.run"
cd viewer && git apply ../vf2/cloud/site_enable.patch && node --test tests/*.test.mjs && npm run deploy:firebase
```

Then sign in on the site, open /engine, check that it says "Processing service · ready", and upload the Zurich clip.

Useful afterwards: `modal app logs voxelflight-cloud`, `modal app stop voxelflight-cloud` (takes processing offline;
the site falls back to "processing unavailable"), `modal volume ls vf-jobs`, `modal volume rm -r vf-jobs /<job id>`.

## Website: is `apiOrigin` enough? No

`apiOrigin` alone is not enough. Apply `site_enable.patch`, because:
1. **CSP**: `firebase.json` `connect-src` allows only the old trycloudflare tunnel, so the browser would block every
   call to `*.modal.run`. The patch adds `https://*.modal.run`. You can narrow it to the exact URL after deploying.
2. **Size cap**: `engine.js` rejects hosted uploads over 60 MiB, and `firebase/build.mjs` rewrites the "up to 1 GiB"
   copy to 60 MiB. The patch reads the cap from `/api/health` `limits.video_bytes` (1 GiB) and keeps the 1 GiB copy.
3. **Upload timeout**: POST requests abort after 180 s. The patch allows 30 min for `POST /api/jobs` only.
4. Small extras: an "FBX mesh" download button, and the GPS field is marked required on the hosted page. The GPU
   pipeline needs telemetry, so this stops an upload of up to 1 GiB from being rejected at the end.

`git apply --check` passes against the current viewer working tree. The viewer test suite passes with the patch
applied (72/72, scratch copy).

## How it works

```
browser ──Bearer Firebase ID token──► web (FastAPI, CPU, 1 container, 64 concurrent requests)
   │  POST /api/jobs: stream to /tmp, check (ffprobe/OpenCV length ≤ 30 min, calibration, telemetry via
   │  vf2/ingest_telemetry.py) -> Volume vf-jobs/<id>/ -> reconstruct.spawn()           job record: Dict "job:<id>"
   ▼
reconstruct (A100-40GB, 8 CPU, 32 GiB, timeout 30 min, ≤ 2 at once; more jobs queue in Modal)
   run_pipeline.py (DPVO ∥ MapAnything, GNSS fusion, Open3D CUDA TSDF, OBJ/PLY/GLB/FBX/LAS/GeoTIFF)
   -> package_web: Z-up GLB ≤ 32 MiB, 400 k-point PLY, LAS, DSM, poses, trajectory, 24 thumbnails, report.json
   -> vf-jobs/<id>/output, commit, then Dict "run:<id>" = complete             progress: Dict "run:<id>"
```

* **API contract** (unchanged for `engine.js`): `GET /api/health` (no auth), `GET /api/jobs`, `POST /api/jobs`
  (multipart `video`, `telemetry`, `calibration`, `max_frames`, `size`, optional `profile`), `GET /api/jobs/{id}`,
  `POST /api/jobs/{id}/cancel`, `GET /api/jobs/{id}/assets/{report.json|reconstruction_mesh.glb|pointcloud.ply|
  reconstruction.obj|reconstruction_mesh.fbx|camera_poses.npz|trajectory.csv|reconstruction_utm.las|surface_model.tif|
  metadata.json|frames/frame_NNNNN.jpg}`. States: uploading, queued, running, cancelling, complete, failed, cancelled,
  interrupted. The response headers match public mode: `no-store`, `nosniff`, `no-referrer`.
* **Auth**: the Firebase ID token is verified with google-auth against Google's public certificates for project
  `voxelflight-3d` (from `viewer/firebase/settings.json`). This is the same check as `public_security.py`: issuer,
  audience, `auth_time`, and provider Google/GitHub. No service account is used. Jobs belong to the token's `uid`, so
  other users get 404. CORS and the Origin check allow only `https://voxelflight-3d.web.app` and
  `https://voxelflight-3d.firebaseapp.com`. The test bypass env var is removed inside the Modal web function.
* **Limits**: video ≤ 1 GiB (the declared Content-Length and the bytes actually received are both enforced) and
  ≤ 30 min. Telemetry ≤ 10 MiB, calibration ≤ 1 MiB. GPU jobs: 2 at once (`MAX_GPU` in `modal_app.py`). Admitted jobs:
  6 (`VF_MAX_ACTIVE_JOBS`). Each account gets 3 runs per day (`VF_DAILY_RUNS`) and 1 active run. Rejected uploads do not
  count toward the daily limit.
* **Processing**: the fast profile is fixed at 360 keyframes and 518 px. The viewer's preview "Sampled frames" and
  "Inference size" fields are accepted but ignored. The profile comes from the telemetry:
  * `street` (vf2 defaults: 5 cm voxels, 25 m depth): chosen for a per-frame vf2 CSV whose altitude spread is < 20 m
    (Zurich), or when height above take-off is < 15 m.
  * `aerial` (15 cm, 250 m, `VF_WEIGHT=1`): chosen otherwise, for example DJI SRT `rel_alt` above 15 m.
  * API clients can override it with `profile=street|aerial`.
  * Without a calibration JSON, `run_pipeline.py` estimates intrinsics with Depth Anything 3.
* **Image**: it reproduces the A100 recipes (`analysis/a100_20260928/setup_a100.py`, `setup_dpvo_a100.py`):
  * Base: `nvidia/cuda:12.6.3-devel-ubuntu22.04` with Python 3.12.
  * One torch 2.6.0+cu126, shared by a `--system-site-packages` `.venv` and `dpvo-env`. numpy 1.26.4, open3d 0.19.0,
    pycolmap 3.10.0, LightGlue.
  * Pinned sources: MapAnything @3d10cf7, DINOv2, Depth Anything 3 @3d835ec, DPVO @0ac95b6 (+ Eigen 3.4.0, torch_scatter).
    DPVO is compiled for sm 8.0/8.6/8.9/9.0.
  * Layout: `/opt/vf`, also linked as `/workspace/voxelflight_a100_20260928`. Weights are in Volume `vf-weights`,
    mounted at `/opt/vf/cache`.
  * The vf2 code is mounted when a container starts, so each `modal deploy` ships whatever vf2 code is on disk at that
    moment. vf2 is still being edited in parallel (`ingest_video.py` and `ingest_telemetry.py` appeared tonight), so
    re-run `selftest` and the smoke job after pulling changes.

## Cost (Modal list prices, https://modal.com/pricing, read 2026-10-04)

| Resource | Price |
|---|---|
| A100 40 GB | $0.000583 / s ($2.10 / h) |
| L4 | $0.000222 / s ($0.80 / h) |
| CPU | $0.0000131 / physical core / s |
| Memory | $0.00000222 / GiB / s |
| **Starter plan** | **$30 / month free credits**, 100 containers, 10 GPU concurrency |

Each job reserves 8 cores and 32 GiB in addition to the GPU. Estimate for **one 10-minute 1080p video**:

| GPU | Billed time (assumption) | GPU | CPU + RAM | **Total** |
|---|---|---|---|---|
| A100-40GB | ~9 min (7.5–11). Measured basis: `timed10-v2` took 428 s on an A100-SXM4-40GB, plus cold start, DA3 intrinsics and packaging | $0.31 | $0.10 | **≈ $0.41** ($0.34–0.50) |
| L4 (untested) | ~25 min (20–30). Assumes about 3× slower: L4 has ~⅓ of the A100's bf16 throughput and ~⅕ of its memory bandwidth | $0.33 | $0.26 | **≈ $0.60** ($0.48–0.72) |

The L4 is cheaper per second but not per video. It runs close to the 27-min worker limit, and with 24 GB it may not fit
MapAnything's window of 48 views at 518 px (`run_pipeline.py` hard-codes `--window 48`). To try it anyway, set
`GPU = "L4"` in `modal_app.py` and redeploy. The $30 credit covers roughly 70 ten-minute A100 jobs, after a few dollars
for image builds, `download_weights` and smoke tests. The web container (0.125 core when warm) costs well under $1/day.

## Tested locally (no Modal account, no GPU)

* `VF_INSECURE_TEST_AUTH=1 python -m pytest vf2/cloud/test_contract.py` (scratch venv with fastapi, httpx,
  python-multipart, pytest, numpy, opencv): **9 passed**. Covered:
  * health, CORS, origin and auth;
  * create/list/get with per-user ownership;
  * rejected uploads (no GPS, wrong type, > 30 min, empty), declared and streamed size caps;
  * progress merge, completion, asset allow-list and traversal;
  * failure, crash, cancel (queued and running), daily and queue caps, not-ready;
  * real `inspect_inputs` on a synthetic video with a DJI-style SRT and with the vf2 CSV.
* `inspect_inputs` on the real Zurich clip: 60 s, 1800 frames, `street`.
* `package_web` on a synthetic vf2 run folder (Open3D/trimesh/laspy/rasterio from `.venv-mac`). Every file
  `engine.js` reads was present, `report.json` parsed as strict JSON, and the GLB stayed Z-up.
* `modal_app.py` imports with modal 1.6.1 and registers `reconstruct`, `download_weights`, `selftest`, `web`, `smoke`.
  This used no token and made no network calls to Modal.

## Not verified (first things to watch in the morning)

* **The image has never been built.** In particular: the DPVO CUDA build without a GPU (`TORCH_CUDA_ARCH_LIST`,
  nvcc 12.6 against torch cu126); the pip resolution of MapAnything + DA3 (xformers 0.0.29.post3) with numpy pinned to
  1.26.4; Open3D's CUDA device inside Modal's container runtime. `selftest` checks each of these.
* **Modal's 150 s HTTP request limit** ([docs](https://modal.com/docs/guide/webhook-timeouts)). It is unclear whether
  a slow upload counts against it. A 1 GiB upload needs about 60 Mbit/s to finish in 150 s. If large uploads fail,
  test with the 218 MB Zurich clip and advise judges to upload ≤ 500 MB, or add chunked uploads.
* End-to-end runtime and output quality on Modal hardware, the L4 path, and the `street`/`aerial` choice for real judge
  footage.
* Modal Dict entries expire after 7 days without reads or writes. Old runs then drop out of "Your reconstructions",
  but their files stay in `vf-jobs`.
