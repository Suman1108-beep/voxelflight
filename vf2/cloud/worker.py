"""VoxelFlight cloud worker: check an upload (API side, CPU), run vf2 fast mode (GPU) and package the run in the
viewer's job-result contract (report.json, Z-up GLB, point sample, OBJ/FBX, LAS, GeoTIFF, poses, trajectory, frames).
Imports only numpy at module level, so the API image can use inspect_inputs without the GPU stack.
run_pipeline.py itself transcodes unreadable codecs, converts telemetry and estimates missing intrinsics."""
import hashlib
import json
import math
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

REPO = Path(os.environ.get("VF_REPO", Path(__file__).resolve().parents[2]))   # sih3d-gpu/: vf2/ (+ src/ for MapAnything)
KEYFRAMES, SIZE, DEADLINE_S = 360, 518, 27 * 60      # vf2 fast profile; stop 3 min before Modal's 30-min timeout
PROFILES = {   # vf2/README.md "Profiles used for the measured runs"
    "street": ([], {}, "5 cm voxels, 25 m depth range"),
    "aerial": (["--voxel", "0.15", "--depth-max", "250", "--exchange-voxel", "0.3"], {"VF_WEIGHT": "1"}, "15 cm voxels, 250 m depth range"),
}
STAGES = {   # run_pipeline.py Timer stage -> (label, progress, detail)
    "dpvo+mapanything (parallel)": ("Camera tracking and depth", .05, "Visual odometry and metric depth on the GPU"),
    "trajectory fusion": ("Trajectory fusion", .62, "Fusing GPS with the visual trajectory"),
    "snap views": ("View placement", .66, "Placing depth views on the fused metric trajectory"),
    "gpu tsdf fusion": ("Surface fusion", .70, "GPU TSDF fusion of all views"),
    "exports": ("Exports", .84, "Writing OBJ, PLY, GLB, FBX, LAS and GeoTIFF"),
}
WEB_MESH_BYTES = 32 * 1024 ** 2


class JobError(RuntimeError):
    """A failure whose message is safe to show the user."""


def inspect_inputs(video, telemetry, calibration, folder):
    """API-side checks before a job is admitted: readable video length, calibration shape, and telemetry converted by the
    pipeline's own vf2/ingest_telemetry.py to folder/telemetry.csv (+ folder/calibration.json). Returns duration and the
    processing profile; raises ValueError with a user-facing message."""
    sys.path.insert(0, str(REPO / "vf2"))
    from ingest_telemetry import convert, probe_video
    folder = Path(folder)
    try:
        n, fps = probe_video(str(video))
    except ValueError:
        raise ValueError("The video could not be read. Upload the MP4/MOV file straight from the drone.")
    if calibration:
        try:
            c = json.loads(Path(calibration).read_text())
            K = c.get("intrinsic_matrix") or [[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]]
            size = int(c["width"]), int(c["height"])
            assert np.asarray(K, float).shape == (3, 3)
        except Exception:
            raise ValueError("Calibration JSON needs width, height and intrinsic_matrix (or fx, fy, cx, cy).")
        import cv2
        cap = cv2.VideoCapture(str(video))
        wh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        if all(wh) and wh != size:
            raise ValueError(f"Calibration is {size[0]}x{size[1]} but the video is {wh[0]}x{wh[1]}.")
        (folder / "calibration.json").write_text(json.dumps(dict(c, intrinsic_matrix=K)))
    try:
        F = convert(str(telemetry), str(folder / "telemetry.csv"), video=str(video), frames=n, fps=fps)
    except (ValueError, OSError, SyntaxError) as error:   # TelemetryError is a ValueError
        raise ValueError(f"GPS telemetry could not be used: {error}")
    if F is None:   # already the per-frame pipeline CSV (e.g. Zurich): only the altitude span is known
        alt = np.genfromtxt(folder / "telemetry.csv", delimiter=",", names=True)["altitude_m"]
        alt = alt[np.isfinite(alt)]
        if len(alt) < 2:
            raise ValueError("GPS telemetry needs at least two rows with altitude_m.")
        street = np.median(alt) - alt.min() < 20
    else:           # height above take-off (DJI rel_alt, AirData height) when the log has it
        street = F["baro"] is not None and np.nanmedian(F["baro"]) < 15
    return dict(duration_s=n / fps, frames=n, fps=fps, profile="street" if street else "aerial")


def run_job(job_dir, work, progress, profile="auto"):
    """Run run_pipeline.py on job_dir/input.* and write the viewer result to job_dir/output. progress(stage, 0..1, detail)."""
    job_dir, t0 = Path(job_dir), time.time()
    video = next(job_dir.glob("input.*"))
    if profile == "auto":   # direct submission (modal_app.py smoke): inputs not inspected by the API yet
        source = next(job_dir.glob("telemetry_source.*"))
        calib = job_dir / "calibration_source.json"
        try:
            profile = inspect_inputs(video, source, calib if calib.is_file() else None, job_dir)["profile"]
        except ValueError as error:
            raise JobError(str(error))
    args, env, _ = PROFILES[profile]
    calib = job_dir / "calibration.json"   # without it run_pipeline.py estimates intrinsics (ingest_video.py)
    out = Path(work) / job_dir.name                      # basename = DPVO run name, unique per job
    shutil.rmtree(out, ignore_errors=True)
    cmd = [sys.executable, "-u", str(REPO / "vf2/run_pipeline.py"), "--video", str(video), "--telemetry", str(job_dir / "telemetry.csv"),
           "--output", str(out), "--keyframes", str(KEYFRAMES), "--size", str(SIZE), *args]
    cmd += ["--calibration", str(calib)] if calib.is_file() else []
    progress("Starting", .02, "Checking the video, GPS and camera intrinsics")
    stage, timed_out = {"name": None}, threading.Event()
    with open(job_dir / "pipeline.log", "w") as log:
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=dict(os.environ, **env), start_new_session=True)

        def kill():   # the whole process group: run_pipeline plus its DPVO / MapAnything children
            try:
                os.killpg(p.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        def depth_progress():   # mapanything_infer.py writes inference/progress.json once per window
            while p.poll() is None:
                time.sleep(5)
                try:
                    if stage["name"] == "dpvo+mapanything (parallel)":
                        q = json.loads((out / "inference/progress.json").read_text())
                        progress("Camera tracking and depth", .05 + .55 * float(q["progress"]), q["detail"])
                except (OSError, ValueError, KeyError):
                    pass
        timer = threading.Timer(DEADLINE_S, lambda: (timed_out.set(), kill()))
        timer.start()
        threading.Thread(target=depth_progress, daemon=True).start()
        try:
            for line in p.stdout:
                log.write(line)
                m = re.search(r">> (.+?)\s*$", line)
                if m and m.group(1) in STAGES:
                    stage["name"] = m.group(1)
                    progress(*STAGES[stage["name"]])
            code = p.wait()
        finally:
            timer.cancel()
            kill()
    for name in ("dpvo.log", "mapanything.log"):
        if (out / name).is_file():
            shutil.copy(out / name, job_dir / name)
    if timed_out.is_set():
        raise JobError("Processing exceeded the 30-minute GPU limit. Try a shorter video.")
    if code != 0:
        where = STAGES.get(stage["name"], ("startup",))[0]
        raise JobError(f"The reconstruction failed during {where.lower()}. Check that the GPS telemetry belongs to this video.")
    progress("Packaging results", .93, "Preparing the web model and downloads")
    readable = out / "video_h264.mp4" if (out / "video_h264.mp4").is_file() else video   # ingest_video.py transcode (HEVC)
    report = package_web(out, readable, job_dir / "output", profile, calib.is_file(), t0)
    shutil.rmtree(out, ignore_errors=True)
    return report


def _finite(x):
    """JSON without NaN/Infinity (browsers' JSON.parse rejects them)."""
    if isinstance(x, dict):
        return {k: _finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_finite(v) for v in x]
    return None if isinstance(x, float) and not math.isfinite(x) else x


def package_web(run, video, dest, profile, calibrated, t0):
    """vf2 run folder -> viewer/engine.js result contract (mirrors make_web_demo.py, without evaluation data)."""
    import cv2, open3d as o3d, trimesh
    run, dest = Path(run), Path(dest)
    shutil.rmtree(dest, ignore_errors=True)
    (dest / "frames").mkdir(parents=True)
    rep = json.loads((run / "report.json").read_text())
    origin, epsg, model = np.array(rep["utm_origin"]), int(rep["utm_epsg"]), run / "model"

    # web mesh: local metres, Z-up (the viewer's frame; export.py's model_mesh.glb is Y-up glTF for other tools)
    full = o3d.io.read_triangle_mesh(str(model / "model_mesh.ply"))
    if not len(full.triangles):
        raise JobError("No surface could be reconstructed. Check that the GPS telemetry belongs to this video.")
    for vox in (0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 1.0):
        m = full.simplify_vertex_clustering(vox)
        V, F = np.asarray(m.vertices), np.asarray(m.triangles)
        C = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8) if m.has_vertex_colors() else np.full((len(V), 3), 180, np.uint8)
        trimesh.Trimesh(V, F, vertex_colors=np.column_stack([C, np.full(len(C), 255, np.uint8)]), process=False).export(dest / "reconstruction_mesh.glb")
        if (dest / "reconstruction_mesh.glb").stat().st_size <= WEB_MESH_BYTES:
            break

    # point sample for the viewer's cloud layer
    pc = o3d.io.read_point_cloud(str(model / "model_points.ply"))
    P, PC = np.asarray(pc.points), np.clip(np.asarray(pc.colors), 0, 1)
    idx = np.random.default_rng(0).choice(len(P), min(400_000, len(P)), replace=False)
    rec = np.zeros(len(idx), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = P[idx].T
    rgb = (PC[idx] * 255).astype(np.uint8) if len(PC) == len(P) else np.full((len(idx), 3), 180, np.uint8)
    rec["r"], rec["g"], rec["b"] = rgb.T
    with open(dest / "pointcloud.ply", "wb") as f:
        f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(rec)}\nproperty float x\nproperty float y\nproperty float z\n"
                "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
        f.write(rec.tobytes())
    for src, name in (("model_mesh.obj", "reconstruction.obj"), ("model_mesh.fbx", "reconstruction_mesh.fbx"),
                      ("model_points_utm.las", "reconstruction_utm.las"), ("dsm_utm.tif", "surface_model.tif"), ("metadata.json", "metadata.json")):
        shutil.copy(model / src, dest / name)

    # keyframe cameras (one per keyframe), trajectory and thumbnails
    kp = np.load(run / "keyframe_poses.npz")
    _, first = np.unique(kp["video_frame"], return_index=True)
    first.sort()
    c2w, vfr = kp["camera_to_world"][first], kp["video_frame"][first]
    np.savez_compressed(dest / "camera_poses.npz", camera_to_world=c2w, source_frame=vfr, utm_origin=origin, utm_epsg=epsg)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    with open(dest / "trajectory.csv", "w") as f:
        f.write("frame,timestamp_s,x,y,z,source_frame,utm_easting,utm_northing,altitude_m\n")
        for i, (T, sf) in enumerate(zip(c2w, vfr)):
            x, y, z = T[:3, 3]
            f.write(f"{i},{sf / fps:.3f},{x:.4f},{y:.4f},{z:.4f},{int(sf)},{x + origin[0]:.3f},{y + origin[1]:.3f},{z + origin[2]:.3f}\n")
    frames = []
    for t in sorted(set(np.linspace(0, len(vfr) - 1, min(24, len(vfr))).round().astype(int).tolist())):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(vfr[t]))
        ok, im = cap.read()
        if ok:
            im = cv2.resize(im, (480, max(1, round(480 * im.shape[0] / im.shape[1]))), interpolation=cv2.INTER_AREA)
            cv2.imwrite(str(dest / f"frames/frame_{t:05d}.jpg"), im, [cv2.IMWRITE_JPEG_QUALITY, 82])
            frames.append(dict(index=t, source_frame=int(vfr[t]), timestamp_s=round(float(vfr[t] / fps), 3)))
    cap.release()

    inference = run / "inference/report.json"
    windows = json.loads(inference.read_text()).get("windows") if inference.is_file() else None
    warnings = ["Absolute position accuracy follows the uploaded GPS (consumer GNSS is typically 3-5 m); it was not independently measured for this run.",
                "A single pass cannot observe backsides or surfaces above the camera; unobserved areas are left empty, never generated.",
                f"Processing profile: {profile} ({PROFILES[profile][2]}). The web preview mesh is decimated to {vox:g} m; "
                "OBJ/FBX keep the exchange mesh and LAS/GeoTIFF the full point set."]
    if not calibrated:
        warnings.append("No camera calibration was supplied; intrinsics were estimated from the video by Depth Anything 3.")
    artifacts = sorted(p.name for p in dest.iterdir() if p.is_file()) + ["report.json"]
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    report = dict(
        schema="voxelflight.v2.cloud-run", pipeline="VoxelFlight v2 fast mode", calibration_supplied=calibrated, profile=profile,
        keyframes=int(len(vfr)), windows=windows, points=int(len(P)), triangles=int(len(full.triangles)), vertices=int(len(full.vertices)),
        web_mesh_triangles=int(len(F)), web_mesh_voxel_m=vox, runtime_s=round(time.time() - t0, 1),
        processing_wall_seconds=rep.get("processing_wall_seconds"), stage_seconds=rep.get("stage_seconds"),
        trajectory_fusion=rep.get("trajectory_fusion"), dsm=rep.get("dsm"), georeferenced=True, utm_epsg=epsg, utm_origin=origin.tolist(),
        coordinate_system=f"EPSG:{epsg} local offset: x=east, y=north, z=up, metres from UTM origin {np.round(origin, 3).tolist()}",
        ground_truth_used=False, survey_or_lidar_used=False, trajectory_rmse_m=None, surface_rmse_m=None, warnings=warnings,
        frames=frames, trajectory=[dict(frame=i, position=np.round(T[:3, 3], 4).tolist()) for i, T in enumerate(c2w)],
        artifacts=artifacts, checksums={n: sha(dest / n) for n in artifacts if n != "report.json"})
    (dest / "report.json").write_text(json.dumps(_finite(report), indent=1, allow_nan=False))
    return report
