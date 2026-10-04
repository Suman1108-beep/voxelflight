"""VoxelFlight v2 quality mode, one continuous timed run (video on disk -> six export formats):
  depth (MapAnything, or Depth Anything 3 on full frames / 2x2 crops)  ||  photogrammetric camera solve (GPU matching + COLMAP)
  -> lock SfM models to GNSS/barometer -> place + scale every depth view -> tiled GPU TSDF -> exports."""
import argparse, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from ingest_telemetry import ensure_pipeline_telemetry
from ingest_video import ensure_calibration, ensure_decodable

ROOT = "/workspace/voxelflight_a100_20260928"; PY = ROOT + "/.venv/bin/python"; HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--telemetry", required=True); ap.add_argument("--calibration")
ap.add_argument("--output", required=True); ap.add_argument("--keyframes", type=int, default=360); ap.add_argument("--every", type=int, default=2)
ap.add_argument("--depth-max", type=float, default=25.0)
ap.add_argument("--serial", action="store_true", help="run camera solve then depth (when GPU memory is shared)")
ap.add_argument("--depth-model", choices=["mapanything", "da3"], default="mapanything")
ap.add_argument("--tiled", action="store_true", help="DA3 on 4 overlapping crops per keyframe (maximum coverage)")
ap.add_argument("--da3-res", type=int, default=1008)
a = ap.parse_args(); os.makedirs(a.output, exist_ok=True)
env = dict(os.environ, HF_HOME=ROOT + "/cache/huggingface", TORCH_HOME=ROOT + "/cache/torch", DINO_SOURCE=ROOT + "/dinov2", HF_HUB_OFFLINE="1",
           PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", LD_LIBRARY_PATH=ROOT + "/runtime-packages/root/usr/lib/x86_64-linux-gnu", VF_WEIGHT="1")
t0 = time.time(); stages = {}
a.video = ensure_decodable(a.video, a.output)  # HEVC etc. -> H.264 when OpenCV cannot read it
a.telemetry = ensure_pipeline_telemetry(a.telemetry, a.video, a.output)  # SRT / GPX / other CSV -> per-frame CSV, passed on to the dense step
a.calibration = ensure_calibration(a.calibration, a.video, a.output)  # estimated intrinsics when none are supplied
n = int(cv2.VideoCapture(a.video).get(cv2.CAP_PROP_FRAME_COUNT))
kf = np.unique(np.linspace(0, n - 1, min(a.keyframes, n)).round().astype(int)).tolist()   # same rule as the depth stage
json.dump(kf, open(os.path.join(a.output, "keyframes.json"), "w"))
ma = [PY, "-u", ROOT + "/repo/vf2_mapanything_infer.py", "--predictions-only", "--video", a.video, "--max-frames", str(a.keyframes),
      "--size", "518", "--window", "48", "--overlap", "8", "--save-predictions", "--output", os.path.join(a.output, "inference")]
if a.calibration: ma += ["--calibration", a.calibration]
crops = os.path.join(a.output, "crops"); kfj = os.path.join(a.output, "keyframes.json")
if a.depth_model == "da3":   # one shell chain so it runs alongside the camera solve exactly like MapAnything
    da3 = (f"{PY} -u {HERE}/make_video_crops.py {a.video} {kfj} {a.calibration} {crops} && "
           f"{PY} -u {HERE}/da3_infer.py --images {crops}" if a.tiled else
           f"{PY} -u {HERE}/da3_infer.py --video {a.video} --keyframes {kfj} --calibration {a.calibration}")
    ma = ["bash", "-c", da3 + f" --output {a.output}/inference --res {a.da3_res} --window 16 --conf-drop 20"]
denv = dict(env, PYTHONPATH=f"{ROOT}/thirdparty/Depth-Anything-3/src:{ROOT}/pydeps-da3") if a.depth_model == "da3" else env
sfm = [PY, "-u", os.path.join(HERE, "fast_sfm.py"), "--video", a.video, "--keyframes-from", os.path.join(a.output, "keyframes.json"),
       "--output", os.path.join(a.output, "sfm"), "--every", str(a.every)] + (["--calibration", a.calibration] if a.calibration else [])
def run_depth_with_retry():
    """Model loading occasionally segfaults on this shared host: retry once, then with smaller windows."""
    import shutil
    for attempt, window in enumerate(("48", "48", "24") if a.depth_model == "mapanything" else ("48",)):
        cmd = [x if x != "48" else window for x in ma] if window != "48" else ma
        shutil.rmtree(os.path.join(a.output, "inference"), ignore_errors=True)
        if subprocess.run(cmd, env=denv, stdout=open(os.path.join(a.output, f"{a.depth_model}_try{attempt}.log"), "w"), stderr=subprocess.STDOUT).returncode == 0:
            stages["depth_attempts"] = attempt + 1; stages["depth_window"] = int(window); return True
    return False

if a.serial:
    if subprocess.run(sfm, env=env, stdout=open(os.path.join(a.output, "sfm.log"), "w"), stderr=subprocess.STDOUT).returncode != 0:
        raise SystemExit(f"camera_solve failed; see logs in {a.output}")
    stages["camera_solve_done_at_s"] = time.time() - t0
    if not run_depth_with_retry():
        raise SystemExit(f"depth failed after retries; see logs in {a.output}")
    stages["depth_done_at_s"] = time.time() - t0
else:
    p_ma = subprocess.Popen(ma, env=denv, stdout=open(os.path.join(a.output, a.depth_model + ".log"), "w"), stderr=subprocess.STDOUT)
    p_sfm = subprocess.Popen(sfm, env=env, stdout=open(os.path.join(a.output, "sfm.log"), "w"), stderr=subprocess.STDOUT)
    for name, p in (("camera_solve", p_sfm), ("depth", p_ma)):
        if p.wait() != 0: raise SystemExit(f"{name} failed; see logs in {a.output}")
        stages[name + "_done_at_s"] = time.time() - t0
dense = [PY, "-u", os.path.join(HERE, "dense_multi_crops.py" if a.tiled else "dense_multi_sfm.py"), "--sparse", os.path.join(a.output, "sfm", "sparse"),
         "--telemetry", a.telemetry, "--predictions", os.path.join(a.output, "inference"), "--output", a.output, "--depth-max", str(a.depth_max)]
if a.tiled: dense += ["--crops", crops]
t = time.time()
if subprocess.run(dense, env=env, stdout=open(os.path.join(a.output, "dense.log"), "w"), stderr=subprocess.STDOUT).returncode != 0:
    raise SystemExit("dense step failed")
stages["dense_fusion_exports_s"] = time.time() - t
wall = time.time() - t0
rep = json.load(open(os.path.join(a.output, "report.json")))
rep.update({"mode": "quality", "serial": a.serial, "depth_model": a.depth_model, "tiled": a.tiled, "processing_wall_seconds": wall, "within_15_min": wall < 900, "continuous_end_to_end": True,
            "quality_stage_marks": stages, "fast_sfm": json.load(open(os.path.join(a.output, "sfm", "fast_sfm.json")))})
json.dump(rep, open(os.path.join(a.output, "report.json"), "w"), indent=1)
print(json.dumps({"processing_wall_seconds": wall, "within_15_min": wall < 900, **stages, "fast_sfm": rep["fast_sfm"]["seconds"]}, indent=1))
