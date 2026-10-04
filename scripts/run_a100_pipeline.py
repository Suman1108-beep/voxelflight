"""Offline, one-command video/GPS to observed 3D reconstruction on a prepared A100.

Uses pretrained checkpoints. DPVO is optional and does not train on the scene.
No ground-truth files are accepted by this CLI. The final report deliberately
does not claim independent surface or absolute-position accuracy.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

import cv2

PROJECT = Path(__file__).resolve().parents[1]
WORKSPACE = PROJECT.parent
sys.path.insert(0, str(PROJECT))
from mac_reconstruct import calibration  # noqa: E402
from build_dpvo_pose_priors import build_priors  # noqa: E402


def video_frame_count(path: Path) -> int:
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError(f"Cannot decode input video: {path}")
        count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if count < 2:
            raise ValueError("A moving video needs at least two frames")
        return count
    finally:
        capture.release()


def dpvo_calibration_line(path: Path) -> str:
    result = calibration(path)
    if result is None:
        raise ValueError("DPVO requires explicit camera calibration")
    _, _, intrinsics, distortion = result
    values = [intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]]
    values.extend(distortion.tolist())
    return " ".join(str(float(value)) for value in values) + "\n"


def run_stage(name: str, command: list[str], *, cwd: Path, env: dict, root: Path) -> float:
    started = time.monotonic()
    log = root / f"{name}.log"
    with log.open("w") as output:
        completed = subprocess.run(command, cwd=cwd, env=env, stdout=output, stderr=subprocess.STDOUT)
    duration = time.monotonic() - started
    if completed.returncode:
        raise RuntimeError(f"{name} failed after {duration:.1f}s; see {log}:\n{log.read_text()[-1500:]}")
    return duration


def process(args: argparse.Namespace) -> dict:
    video = args.video.resolve()
    telemetry = args.telemetry.resolve()
    camera = args.calibration.resolve() if args.calibration else None
    for path in [video, telemetry, *([camera] if camera else [])]:
        if not path.is_file():
            raise FileNotFoundError(path)
    root = args.output.resolve()
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing job: {root}")
    root.mkdir(parents=True, exist_ok=True)

    count = video_frame_count(video)
    started = time.monotonic()
    stages: dict[str, float] = {}
    model_env = os.environ.copy()
    model_env.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    if args.hf_home:
        model_env["HF_HOME"] = str(args.hf_home.resolve())
    if args.dino_source:
        model_env["DINO_SOURCE"] = str(args.dino_source.resolve())
    dpvo_ready = (
        camera is not None
        and args.dpvo_python.is_file()
        and (args.dpvo_repo / "demo.py").is_file()
        and args.dpvo_checkpoint.is_file()
    )
    if args.dpvo == "required" and not dpvo_ready:
        raise FileNotFoundError("DPVO requested but its Python, repo, checkpoint or calibration is missing")
    use_dpvo = args.dpvo != "off" and dpvo_ready
    priors = None

    if use_dpvo:
        stride = args.dpvo_stride or max(5, math.ceil(count / 4000))
        if math.ceil(count / stride) >= 4096:
            raise ValueError("DPVO stride exceeds its 4096-frame pose buffer")
        calib_txt = root / "dpvo_calibration.txt"
        calib_txt.write_text(dpvo_calibration_line(camera))
        unique_name = f"voxelflight_{uuid.uuid4().hex}"
        dpvo_env = os.environ.copy()
        # DPVO's official demo writes to its own saved_trajectories directory.
        # A unique name isolates concurrent user jobs before copying it here.
        stages["visual_odometry"] = run_stage(
            "visual_odometry",
            [str(args.dpvo_python), "-u", "demo.py", "--imagedir", str(video),
             "--calib", str(calib_txt), "--stride", str(stride),
             "--network", str(args.dpvo_checkpoint), "--name", unique_name,
             "--save_trajectory"],
            cwd=args.dpvo_repo, env=dpvo_env, root=root,
        )
        saved = args.dpvo_repo / "saved_trajectories" / f"{unique_name}.txt"
        trajectory = root / "visual_trajectory.tum"
        shutil.copy2(saved, trajectory)
        saved.unlink()
        priors = root / "visual_pose_priors.json"
        prior_data = build_priors(
            trajectory, source_frames=count, stride=stride, max_frames=args.max_frames
        )
        priors.write_text(json.dumps(prior_data, indent=2, allow_nan=False) + "\n")

    inference = root / "inference"
    command = [str(args.model_python), "-u", str(PROJECT / "mac_reconstruct.py"),
               "--video", str(video), "--telemetry", str(telemetry),
               "--output", str(inference), "--max-frames", str(args.max_frames),
               "--size", str(args.size), "--window", str(args.window),
               "--overlap", str(args.overlap), "--save-predictions"]
    if camera:
        command += ["--calibration", str(camera)]
    if priors:
        command += ["--visual-pose-priors", str(priors)]
    stages["inference"] = run_stage(
        "inference", command, cwd=PROJECT, env=model_env, root=root
    )

    fusion_env = model_env.copy()
    if args.runtime_lib_dir.is_dir():
        fusion_env["LD_LIBRARY_PATH"] = (
            str(args.runtime_lib_dir) + ":" + fusion_env.get("LD_LIBRARY_PATH", "")
        )
    fusion = root / "fusion"
    stages["fusion"] = run_stage(
        "fusion",
        [str(args.model_python), "-u", str(PROJECT / "mac_refuse.py"),
         "--input", str(inference), "--output", str(fusion),
         "--voxel", str(args.voxel), "--min-support", "0", "--neighbor-radius", "8"],
        cwd=PROJECT, env=fusion_env, root=root,
    )
    inference_report = json.loads((inference / "report.json").read_text())
    final = fusion
    if inference_report.get("georeferenced"):
        final = root / "georeferenced"
        stages["geospatial_export"] = run_stage(
            "geospatial_export",
            [str(args.model_python), "-u",
             str(PROJECT / "scripts/georeference_refused_mesh.py"),
             "--source", str(inference), "--fusion", str(fusion),
             "--output", str(final)],
            cwd=PROJECT, env=fusion_env, root=root,
        )
    result = {
        "schema": "voxelflight.offline-pipeline.v1",
        "input_video": str(video),
        "source_frames": count,
        "selected_keyframes": inference_report["keyframes"],
        "visual_odometry_used": use_dpvo,
        "ground_truth_used": False,
        "scene_specific_training_performed": False,
        "stages_s": stages,
        "total_processing_wall_s": time.monotonic() - started,
        "final_result": str(final),
        "georeferenced": bool(inference_report.get("georeferenced")),
        "independent_surface_accuracy_verified": False,
        "absolute_position_accuracy_verified": False,
        "limitations": [
            "Observed surfaces only; occluded geometry is not measured.",
            "Onboard GPS georeferencing is not surveyed absolute accuracy.",
            "No independent surface reference is read by this pipeline.",
        ],
    }
    (root / "pipeline_report.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dpvo", choices=("auto", "off", "required"), default="auto")
    parser.add_argument("--dpvo-stride", type=int)
    parser.add_argument("--dpvo-python", type=Path, default=WORKSPACE / "thirdparty/dpvo-env/bin/python")
    parser.add_argument("--dpvo-repo", type=Path, default=WORKSPACE / "thirdparty/DPVO")
    parser.add_argument("--dpvo-checkpoint", type=Path, default=WORKSPACE / "thirdparty/DPVO/dpvo.pth")
    parser.add_argument("--model-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--hf-home", type=Path, default=WORKSPACE / "cache/huggingface")
    parser.add_argument("--dino-source", type=Path, default=WORKSPACE / "dinov2")
    parser.add_argument("--runtime-lib-dir", type=Path, default=WORKSPACE / "runtime-packages/root/usr/lib/x86_64-linux-gnu")
    parser.add_argument("--max-frames", type=int, default=360)
    parser.add_argument("--size", type=int, default=518)
    parser.add_argument("--window", type=int, default=48)
    parser.add_argument("--overlap", type=int, default=8)
    parser.add_argument("--voxel", type=float, default=0.05)
    args = parser.parse_args()
    if args.max_frames < 2 or args.window < 3 or not 2 <= args.overlap < args.window:
        parser.error("Invalid frame budget, window or overlap")
    if args.voxel <= 0 or (args.dpvo_stride is not None and args.dpvo_stride < 1):
        parser.error("Voxel and DPVO stride must be positive")
    print(json.dumps(process(args), indent=2), flush=True)


if __name__ == "__main__":
    main()
