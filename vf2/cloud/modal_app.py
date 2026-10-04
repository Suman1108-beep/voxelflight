"""Modal deployment: VoxelFlight v2 fast pipeline on a cloud GPU behind the viewer's job API (api.py).

  modal run vf2/cloud/modal_app.py::download_weights     # once: MapAnything, DA3, DPVO weights -> Volume "vf-weights"
  modal run vf2/cloud/modal_app.py::selftest             # GPU, Open3D CUDA, DPVO extensions, DA3, weights present
  modal run vf2/cloud/modal_app.py --video V --telemetry T [--calibration C]   # one real job, no website
  modal deploy vf2/cloud/modal_app.py                    # prints https://<workspace>--voxelflight-cloud-web.modal.run

The image reproduces the A100 layout under /opt/vf (also linked as /workspace/voxelflight_a100_20260928, so vf2 code
with hard-coded ROOT runs unchanged; root_env.patch makes it explicit) and the A100 recipes in
analysis/a100_20260928/setup_a100.py + setup_dpvo_a100.py: one torch 2.6.0+cu126 shared by a --system-site-packages
.venv and dpvo-env, MapAnything / DINOv2 / DPVO / Depth Anything 3 source pinned, weights in a Volume.
vf2/*.py is mounted at container start, so `modal deploy` always ships the vf2 code currently on disk.
"""
import os
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]                                   # sih3d-gpu/ (vf2/, src/, mac_geometry.py)
VF, A100_ROOT, JOBS = "/opt/vf", "/workspace/voxelflight_a100_20260928", "/jobs"   # image root, its A100 alias, job files
CACHE = f"{VF}/cache"                                    # HF_HOME / TORCH_HOME / dpvo.pth (Volume)
GPU = "A100-40GB"                                        # measured target (RESULTS.md). "L4" is cheaper per second, untested
MAX_GPU = 2                                              # GPU containers at once; api.py reads VF_MAX_GPU_JOBS
MODEL_ID, MODEL_REV = "facebook/map-anything-apache", "00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"
DA3_MODEL = "depth-anything/DA3-GIANT-1.1"               # ingest_video.py: intrinsics when no calibration is uploaded
MAPANYTHING_COMMIT, DPVO_COMMIT = "3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9", "0ac95b656d1fda91c271d2a106460d19ad966fc7"
DA3_COMMIT = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
DPVO_WEIGHTS = ("https://www.dropbox.com/scl/fo/2g6e134ruh1acmaye5bmy/AK0DY9WLZ25elq-7N7sJcB0/models.zip"
                "?rlkey=4j336vcc4h324j4p4ji3ppl0t&dl=1")  # DPVO download_models_and_data.sh (zip holds dpvo.pth)
TORCH = "torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 numpy==1.26.4"   # repeated in each pip call so nothing upgrades them
TORCH_INDEX = "https://download.pytorch.org/whl/cu126"
DPVO_ENV = f"{VF}/thirdparty/dpvo-env"

app = modal.App("voxelflight-cloud")
weights = modal.Volume.from_name("vf-weights", create_if_missing=True)
jobs_vol = modal.Volume.from_name("vf-jobs", create_if_missing=True)
state = modal.Dict.from_name("vf-state", create_if_missing=True)   # "job:<id>" (API), "run:<id>" (worker), "meta:weights"

gpu_image = (
    modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04", add_python="3.12")
    .apt_install("git", "wget", "unzip", "build-essential", "ninja-build", "ffmpeg", "libgomp1", "libglib2.0-0",
                 "libgl1", "libglx0", "libglvnd0", "libegl1", "libx11-6", "libxext6", "libxcb1")   # Open3D (setup_open3d_runtime.py)
    .pip_install(*TORCH.split(), index_url=TORCH_INDEX)
    .pip_install("scipy==1.15.3", "rasterio==1.4.3", "open3d==0.19.0", "pycolmap==3.10.0", "laspy>=2.6", "pyproj>=3.7",
                 "huggingface_hub==0.36.2", "trimesh", "plyfile", "pillow>=11", "matplotlib>=3.10", "imageio-ffmpeg>=0.6", "kornia",
                 *TORCH.split(), extra_index_url=TORCH_INDEX)
    .pip_install("lightglue @ git+https://github.com/cvg/LightGlue.git", extra_options="--no-deps")   # keeps one OpenCV
    .run_commands(
        f"git clone https://github.com/facebookresearch/map-anything {VF}/map-anything && git -C {VF}/map-anything checkout {MAPANYTHING_COMMIT}",
        f"pip install {VF}/map-anything {TORCH} --extra-index-url {TORCH_INDEX}",
        f"git clone --depth 1 https://github.com/facebookresearch/dinov2 {VF}/dinov2",
        # Depth Anything 3 for estimated intrinsics: source on PYTHONPATH (ingest_video.py), deps in the shared env
        f"git clone https://github.com/ByteDance-Seed/Depth-Anything-3 {VF}/thirdparty/Depth-Anything-3"
        f" && git -C {VF}/thirdparty/Depth-Anything-3 checkout {DA3_COMMIT}",
        f"pip install xformers==0.0.29.post3 einops imageio omegaconf e3nn moviepy==1.0.3 typer pillow_heif safetensors evo {TORCH}"
        f" --extra-index-url {TORCH_INDEX}",
        f"python -m venv --system-site-packages {VF}/.venv",   # PY in run_pipeline.py / ingest_video.py
        f"git clone https://github.com/princeton-vl/DPVO {VF}/thirdparty/DPVO && git -C {VF}/thirdparty/DPVO checkout {DPVO_COMMIT}",
        f"cd {VF}/thirdparty/DPVO && wget -q https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.zip"
        " && unzip -q eigen-3.4.0.zip -d thirdparty && rm eigen-3.4.0.zip",
        f"python -m venv --system-site-packages {DPVO_ENV}",
        f"{DPVO_ENV}/bin/pip install setuptools wheel ninja pypose yacs numba tqdm {TORCH} --extra-index-url {TORCH_INDEX}",
        f"{DPVO_ENV}/bin/pip install --no-index --no-deps -f https://data.pyg.org/whl/torch-2.6.0+cu126.html torch_scatter",
        # No GPU at build time: compile for A100 (8.0), A10G (8.6), L4/L40S (8.9), H100 (9.0).
        f"cd {VF}/thirdparty/DPVO && TORCH_CUDA_ARCH_LIST='8.0;8.6;8.9;9.0' MAX_JOBS=8 {DPVO_ENV}/bin/pip install --no-build-isolation --no-deps .",
        f"ln -s {CACHE}/dpvo/dpvo.pth {VF}/thirdparty/DPVO/dpvo.pth",
        f"mkdir -p {VF}/runtime-packages/root/usr/lib /workspace && ln -s /usr/lib/x86_64-linux-gnu {VF}/runtime-packages/root/usr/lib/"
        f" && ln -s {VF} {A100_ROOT}",
    )
    .env({"VF_ROOT": VF, "VF_REPO": f"{VF}/repo"})
    # code last (mounted at start, no rebuild on edits); same layout as the A100: repo/vf2_mapanything_infer.py, repo/src
    .add_local_dir(REPO / "vf2", f"{VF}/repo/vf2", ignore=["experiments", "viewer", "tests", "**/__pycache__", "**/*.md", "**/*.patch"])
    .add_local_file(REPO / "vf2/mapanything_infer.py", f"{VF}/repo/vf2_mapanything_infer.py")
    .add_local_file(REPO / "mac_geometry.py", f"{VF}/repo/mac_geometry.py")
    .add_local_dir(REPO / "src/sih3d", f"{VF}/repo/src/sih3d", ignore=["**/__pycache__"])
)

web_image = (   # CPU only: FastAPI + upload checks (ffprobe, OpenCV, vf2/ingest_telemetry.py)
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("ffmpeg", "libglib2.0-0")
    .pip_install("fastapi>=0.115", "python-multipart", "google-auth[requests]==2.58.0", "CacheControl==0.14.4", "requests",
                 "numpy==1.26.4", "opencv-python-headless==4.10.0.84")
    .env({"VF_REPO": f"{VF}/repo", "VF_MAX_GPU_JOBS": str(MAX_GPU)})
    .add_local_dir(HERE, f"{VF}/repo/vf2/cloud", ignore=["**/__pycache__", "*.md", "*.patch", "test_*.py"])
    .add_local_file(REPO / "vf2/ingest_telemetry.py", f"{VF}/repo/vf2/ingest_telemetry.py")
)


@app.function(image=gpu_image, gpu=GPU, cpu=8, memory=32768, timeout=1800, max_containers=MAX_GPU,
              volumes={CACHE: weights, JOBS: jobs_vol})
def reconstruct(job_id: str, profile: str = "auto"):
    """One job: /jobs/<id>/input.* + telemetry.csv (+ calibration.json) -> /jobs/<id>/output (viewer contract)."""
    import traceback
    sys.path.insert(0, f"{VF}/repo/vf2/cloud")
    import worker
    key = f"run:{job_id}"
    put = lambda **v: state.put(key, {**(state.get(key) or {}), **v})
    put(state="running", started_at=time.time())
    jobs_vol.reload()   # a warm container must see inputs uploaded after it started
    try:
        worker.run_job(Path(JOBS) / job_id, Path("/tmp/vf"), lambda s, f, d: put(progress=dict(stage=s, progress=round(f, 3), detail=d)), profile)
        result = dict(state="complete", message="New reconstruction is ready.", progress=dict(stage="Complete", progress=1.0, detail="Model ready"))
    except worker.JobError as error:
        result = dict(state="failed", message=str(error))
    except Exception:
        traceback.print_exc()
        result = dict(state="failed", message=None)   # api.py shows its generic failure text
    jobs_vol.commit()   # outputs durable before the API can see "complete"
    put(**result, finished_at=time.time())
    return result["state"]


@app.function(image=gpu_image, volumes={CACHE: weights}, cpu=4, memory=16384, timeout=3600)
def download_weights():
    """Fill the weights Volume once (online); MapAnything then runs with HF_HUB_OFFLINE=1."""
    import subprocess
    from unittest.mock import patch
    os.environ.update(HF_HOME=f"{CACHE}/huggingface", TORCH_HOME=f"{CACHE}/torch")
    import torch
    from huggingface_hub import snapshot_download
    path = snapshot_download(MODEL_ID, revision=MODEL_REV, allow_patterns=["config.json", "model.safetensors"])
    snapshot_download(DA3_MODEL)   # used online by ingest_video.py; cached here so the first job does not wait
    # Build the model once on CPU, as mapanything_infer.py does, so any encoder checkpoint lands in TORCH_HOME.
    from mapanything.models import MapAnything
    hub_load = torch.hub.load
    local = lambda repo, entry, *a, **k: hub_load(f"{VF}/dinov2", entry, *a, source="local",
                                                  **{n: v for n, v in k.items() if n not in ("source", "force_reload")})
    with patch("torch.hub.load", local):
        MapAnything.from_pretrained(path, local_files_only=True)
    os.makedirs(f"{CACHE}/dpvo", exist_ok=True)
    subprocess.run(f"wget -q -O /tmp/models.zip '{DPVO_WEIGHTS}' && unzip -o -j /tmp/models.zip dpvo.pth -d {CACHE}/dpvo", shell=True, check=True)
    weights.commit()
    state.put("meta:weights", dict(model=MODEL_ID, revision=MODEL_REV, dpvo_bytes=os.path.getsize(f"{CACHE}/dpvo/dpvo.pth"), at=time.time()))
    print("weights ready:", path)


@app.function(image=gpu_image, gpu=GPU, volumes={CACHE: weights}, timeout=900)
def selftest():
    """Import every GPU dependency in the interpreters run_pipeline.py uses."""
    import subprocess
    checks = {
        "torch": ("python", "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name())"),
        "open3d": ("python", "import open3d.core as o3c; assert o3c.cuda.is_available(); print('Open3D CUDA', o3c.cuda.device_count())"),
        "pipeline": (f"{VF}/.venv/bin/python", "import mapanything, pycolmap, lightglue, rasterio, laspy, pyproj, trimesh, cv2; print('imports ok')"),
        "da3": ("env", f"PYTHONPATH={VF}/thirdparty/Depth-Anything-3/src", f"{VF}/.venv/bin/python", "-c",
                "from depth_anything_3.api import DepthAnything3; print('DA3 ok')"),
        "dpvo": (f"{DPVO_ENV}/bin/python", "import torch, torch_scatter, cuda_corr, cuda_ba, lietorch_backends; "
                                           "from dpvo.dpvo import DPVO; assert torch.cuda.is_available(); print('DPVO ok')"),
    }
    ok = True
    for name, cmd in checks.items():
        r = subprocess.run(cmd if len(cmd) > 2 else [cmd[0], "-c", cmd[1]], cwd=f"{VF}/thirdparty/DPVO", capture_output=True, text=True)
        ok &= r.returncode == 0
        print(f"{name:9s}", "OK  " if r.returncode == 0 else "FAIL", ((r.stdout or r.stderr).strip().splitlines() or [""])[-1])
    for f in (f"{A100_ROOT}/thirdparty/DPVO/dpvo.pth", f"{A100_ROOT}/cache/huggingface/hub/models--facebook--map-anything-apache/snapshots/{MODEL_REV}/model.safetensors",
              f"{CACHE}/huggingface/hub/models--{DA3_MODEL.replace('/', '--')}"):
        ok &= os.path.exists(f)
        print("weights  ", "OK  " if os.path.exists(f) else "MISSING", f)
    if not ok:
        raise SystemExit("selftest failed")


class ModalStore:
    """api.py store: records in the Dict, job files in the jobs Volume (written by batch_upload, read by read_file)."""

    def get(self, key):
        return state.get(key)

    def put(self, key, value):
        state.put(key, value)

    def delete(self, key):
        state.pop(key, None)

    def values(self, prefix):
        return [v for k, v in state.items() if k.startswith(prefix)]

    def save_inputs(self, job_id, folder):
        with jobs_vol.batch_upload(force=True) as batch:
            batch.put_directory(str(folder), f"/{job_id}")

    def read(self, job_id, name):
        chunks = iter(jobs_vol.read_file(f"{job_id}/output/{name}"))
        try:
            first = next(chunks)
        except (FileNotFoundError, StopIteration):
            return None
        return (b for part in ([first], chunks) for b in part)


class ModalRunner:
    """api.py runner over spawned reconstruct() calls."""
    ready_at = 0.0

    def ready(self):
        if time.time() - self.ready_at > 60:
            self.ready_at = time.time() if state.get("meta:weights") else 0.0
        return self.ready_at > 0

    def submit(self, job_id, options):
        return reconstruct.spawn(job_id, **options).object_id

    def poll(self, handle):
        """None while queued/running; True/False once it returned/failed (crash, timeout, cancel)."""
        try:
            modal.FunctionCall.from_id(handle).get(timeout=0)
            return True
        except TimeoutError:                                   # builtin: no output yet
            return None
        except modal.exception.ConnectionError:
            return None
        except (Exception, modal.exception.InputCancellation):
            return False

    def cancel(self, handle):
        modal.FunctionCall.from_id(handle).cancel(terminate_containers=True)   # also kills DPVO/MapAnything children


@app.function(image=web_image, timeout=900, max_containers=1, scaledown_window=300)
@modal.concurrent(max_inputs=64)
@modal.asgi_app()
def web():
    sys.path.insert(0, f"{VF}/repo/vf2/cloud")
    os.environ.pop("VF_INSECURE_TEST_AUTH", None)   # never bypass Firebase in a deployment
    import api
    return api.create_app(ModalStore(), ModalRunner())


@app.local_entrypoint()
def smoke(video: str, telemetry: str, calibration: str = "", profile: str = "auto"):
    """Run one real job without the website and list its outputs (inputs are inspected on the GPU worker)."""
    job_id = time.strftime("smoke-%Y%m%d-%H%M%S")
    with jobs_vol.batch_upload(force=True) as batch:
        batch.put_file(video, f"/{job_id}/input{Path(video).suffix.lower()}")
        batch.put_file(telemetry, f"/{job_id}/telemetry_source{Path(telemetry).suffix.lower()}")
        if calibration:
            batch.put_file(calibration, f"/{job_id}/calibration_source.json")
    t0 = time.time()
    print(job_id, reconstruct.remote(job_id, profile), f"{time.time() - t0:.0f}s", state.get(f"run:{job_id}"))
    for entry in jobs_vol.listdir(f"/{job_id}/output"):
        print(" ", entry.path)
    print(f"logs: modal volume get vf-jobs /{job_id}/pipeline.log .")
