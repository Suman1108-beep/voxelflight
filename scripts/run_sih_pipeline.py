#!/usr/bin/env python3
"""Run the complete offline single-pass video-to-3D submission pipeline."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


PROJECT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_delivery", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--video", type=Path)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--mast3r-root", type=Path, default=Path("/mnt/road/sota/mast3r-slam"))
    parser.add_argument("--mast3r-python", type=Path, default=Path("/mnt/road/sota/envs/mast3r-slam/bin/python"))
    parser.add_argument("--keyframes", type=int, default=180)
    parser.add_argument("--window-size", type=int, default=72)
    parser.add_argument("--window-overlap", type=int, default=18)
    parser.add_argument(
        "--photoreal",
        action="store_true",
        help="Train per-window 3D Gaussian splats after metric reconstruction.",
    )
    parser.add_argument("--photoreal-steps", type=int, default=3000)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def run_stage(
    name: str,
    command: list[str | Path],
    run_dir: Path,
    *,
    cwd: Path = PROJECT,
    environment: dict[str, str] | None = None,
) -> float:
    logs = run_dir / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    with (logs / f"{name}.log").open("w") as stream:
        stream.write("COMMAND: " + " ".join(map(str, command)) + "\n\n")
        stream.flush()
        subprocess.run(
            [str(value) for value in command],
            cwd=cwd,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=True,
        )
    duration = time.perf_counter() - start
    timing_path = run_dir / "stage_timings.json"
    recorded = json.loads(timing_path.read_text()) if timing_path.exists() else {}
    recorded[name] = duration
    timing_path.write_text(json.dumps(recorded, indent=2) + "\n")
    return duration


def offline_environment() -> dict[str, str]:
    environment = os.environ.copy()
    for directory in (
        PROJECT / "cache" / "huggingface",
        PROJECT / "cache" / "torch",
        PROJECT / "cache" / "tmp",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    environment["PYTHONPATH"] = str(PROJECT / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    environment["HF_HOME"] = str(PROJECT / "cache" / "huggingface")
    environment["HF_HUB_CACHE"] = str(PROJECT / "cache" / "huggingface" / "hub")
    environment["TORCH_HOME"] = str(PROJECT / "cache" / "torch")
    environment["XDG_CACHE_HOME"] = str(PROJECT / "cache")
    environment["TMPDIR"] = str(PROJECT / "cache" / "tmp")
    environment["OMP_NUM_THREADS"] = "2"
    environment["OPENBLAS_NUM_THREADS"] = "2"
    environment["MKL_NUM_THREADS"] = "2"
    environment["PYTHONUNBUFFERED"] = "1"
    environment["PATH"] = (
        str(PROJECT / ".venv" / "bin")
        + os.pathsep
        + environment.get("PATH", "")
    )
    cuda_roots = sorted(
        path
        for path in (PROJECT / ".venv" / "lib").glob(
            "python*/site-packages/nvidia/cu*"
        )
        if (path / "bin" / "nvcc").exists()
    )
    if cuda_roots:
        environment["CUDA_HOME"] = str(cuda_roots[-1])
    return environment


def discover_calibration(root: Path) -> Path | None:
    candidates = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".json", ".yaml", ".yml"}
        and any(token in path.stem.lower() for token in ("camera", "calib", "intrinsic"))
    ]
    return sorted(candidates, key=lambda path: (len(path.parts), str(path)))[0] if candidates else None


def package_viewer(viewer_dir: Path, assets_dir: Path) -> None:
    viewer_dir.mkdir(parents=True, exist_ok=True)
    for name in ("index.html", "styles.css", "app.js"):
        shutil.copy2(PROJECT / "viewer" / name, viewer_dir / name)
    shutil.copytree(
        PROJECT / "viewer" / "vendor",
        viewer_dir / "vendor",
        dirs_exist_ok=True,
    )
    packaged_assets = viewer_dir / "assets"
    packaged_assets.mkdir(parents=True, exist_ok=True)
    for name in (
        "reconstruction_mesh.glb",
        "reconstruction_clean.ply",
        "reconstruction_utm.las",
        "surface_model.tif",
        "trajectory.csv",
        "evaluation.json",
        "asset_report.json",
    ):
        destination = packaged_assets / name
        source = (assets_dir / name).resolve()
        if not source.exists():
            continue
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
    optional_assets = [
        *assets_dir.glob("photoreal_*"),
        *assets_dir.glob("reconstruction_splat_*.ply"),
    ]
    for source in sorted(optional_assets):
        destination = packaged_assets / source.name
        if destination.exists() or destination.is_symlink():
            destination.unlink()
        try:
            os.link(source.resolve(), destination)
        except OSError:
            shutil.copy2(source, destination)


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    run_dir = args.output_dir.resolve()
    if run_dir.exists() and any(run_dir.iterdir()) and not args.resume:
        raise FileExistsError(
            f"Output directory is not empty; use --resume: {run_dir}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    environment = offline_environment()
    timings: dict[str, float] = {}

    scene_dir = run_dir / "input"
    manifest_path = scene_dir / "scene_manifest.json"
    if not (args.resume and manifest_path.exists()):
        calibration_input = args.calibration or discover_calibration(
            args.input_delivery
        )
        command: list[str | Path] = [
            args.python, PROJECT / "scripts" / "prepare_input.py",
            args.input_delivery, scene_dir,
        ]
        for option, value in (
            ("--video", args.video),
            ("--telemetry", args.telemetry),
            ("--calibration", calibration_input),
        ):
            if value:
                command.extend((option, value))
        timings["prepare_input"] = run_stage(
            "01_prepare_input", command, run_dir, environment=environment
        )
    scene = json.loads(manifest_path.read_text())
    video = Path(scene["video"]["canonical_link"])

    keyframes = run_dir / "keyframes"
    if not (args.resume and (keyframes / "manifest.csv").exists()):
        timings["extract_keyframes"] = run_stage(
            "02_extract_keyframes",
            [
                args.python, PROJECT / "scripts" / "extract_keyframes.py",
                video, keyframes, "--max-frames", str(args.keyframes),
            ],
            run_dir,
            environment=environment,
        )
        timings["make_windows"] = run_stage(
            "03_make_windows",
            [
                args.python, PROJECT / "scripts" / "make_keyframe_windows.py",
                keyframes, "--window-size", str(args.window_size),
                "--window-overlap", str(args.window_overlap),
            ],
            run_dir,
            environment=environment,
        )

    telemetry_dir = run_dir / "telemetry"
    telemetry_report = telemetry_dir / "telemetry_adapter_report.json"
    if not (args.resume and telemetry_report.exists()):
        timings["adapt_telemetry"] = run_stage(
            "04_adapt_telemetry",
            [
                args.python, PROJECT / "scripts" / "adapt_canonical_telemetry.py",
                scene_dir / "frame_telemetry.csv", keyframes / "manifest.csv",
                telemetry_dir,
            ],
            run_dir,
            environment=environment,
        )
    telemetry = json.loads(telemetry_report.read_text())
    epsg = int(telemetry["utm_epsg"])

    normalized_calibration = run_dir / "camera_mast3r.yaml"
    calibration = scene.get("calibration")
    if calibration and not (args.resume and normalized_calibration.exists()):
        timings["normalize_calibration"] = run_stage(
            "05_normalize_calibration",
            [
                args.python, PROJECT / "scripts" / "normalize_camera_calibration.py",
                calibration, normalized_calibration,
            ],
            run_dir,
            environment=environment,
        )

    mast_output = run_dir / "mast3r_slam"
    mast_prediction = mast_output / "raw_prediction.npz"
    mast_save_name = "sih_" + hashlib.sha1(str(run_dir).encode()).hexdigest()[:12]
    if not (args.resume and mast_prediction.exists()):
        mast_command: list[str | Path] = [
            args.mast3r_python, args.mast3r_root / "main.py",
            "--dataset", keyframes,
            "--config", args.mast3r_root / "config" / "base.yaml",
            "--save-as", mast_save_name,
            "--no-viz",
        ]
        if normalized_calibration.exists():
            mast_command.extend(("--calib", normalized_calibration))
        timings["mast3r_slam"] = run_stage(
            "06_mast3r_slam",
            mast_command,
            run_dir,
            cwd=args.mast3r_root,
            environment=environment,
        )
        trajectory_tum = args.mast3r_root / "logs" / mast_save_name / f"{keyframes.name}.txt"
        reconstruction = args.mast3r_root / "logs" / mast_save_name / f"{keyframes.name}.ply"
        mast_output.mkdir(parents=True, exist_ok=True)
        timings["convert_mast3r"] = run_stage(
            "07_convert_mast3r",
            [
                args.python, PROJECT / "scripts" / "convert_mast3r_slam_trajectory.py",
                trajectory_tum, mast_output / "trajectory.csv",
                "--keyframe-manifest", keyframes / "manifest.csv",
                "--output-prediction", mast_prediction,
            ],
            run_dir,
            environment=environment,
        )
        shutil.copy2(reconstruction, mast_output / "points_dense.ply")

    windows = json.loads((keyframes / "windows.json").read_text())
    map_predictions: list[Path] = []
    for index, window in enumerate(windows):
        output = run_dir / "mapanything" / window["name"]
        prediction = output / "raw_prediction.npz"
        map_predictions.append(prediction)
        if args.resume and prediction.exists():
            continue
        timings[f"mapanything_{window['name']}"] = run_stage(
            f"08_mapanything_{index:02d}_{window['name']}",
            [
                args.python, PROJECT / "scripts" / "run_mapanything_window.py",
                window["path"], output,
                "--keyframe-manifest", keyframes / "manifest.csv",
                "--source-image-offset", "0",
                "--max-points", "1500000",
            ],
            run_dir,
            environment=environment,
        )

    fused = run_dir / "fused"
    if not (args.resume and (fused / "reconstruction_relative.ply").exists()):
        timings["pose_and_geometry_fusion"] = run_stage(
            "09_pose_and_geometry_fusion",
            [
                args.python, PROJECT / "scripts" / "fuse_visual_gnss_baro.py",
                fused, keyframes / "manifest.csv",
                telemetry_dir / "frame_telemetry.csv",
                telemetry_dir / "OnboardGPS.csv",
                telemetry_dir / "BarometricPressure.csv",
                *map_predictions,
                "--global-anchor-prediction", mast_prediction,
                "--source-image-offset", "0",
                "--utm-epsg", str(epsg),
                "--sensor-delta-weight", "10",
                "--visual-delta-weight", "30",
                "--max-points", "2500000",
            ],
            run_dir,
            environment=environment,
        )

    assets = run_dir / "submission"
    if not (args.resume and (assets / "reconstruction_mesh.glb").exists()):
        timings["submission_assets"] = run_stage(
            "10_submission_assets",
            [
                args.python, PROJECT / "scripts" / "build_submission_assets.py",
                fused / "reconstruction_relative.ply",
                fused / "reconstruction_utm.ply",
                assets,
                "--trajectory", fused / "trajectory.csv",
                "--epsg", str(epsg),
                "--voxel-size", "0.12",
                "--poisson-depth", "9",
                "--target-triangles", "450000",
            ],
            run_dir,
            environment=environment,
        )

    photoreal_outputs: list[dict[str, str | float | int]] = []
    if args.photoreal:
        if args.photoreal_steps < 1:
            raise ValueError("--photoreal-steps must be positive")
        final_step = args.photoreal_steps - 1
        for index, window in enumerate(windows):
            name = window["name"]
            prediction_dir = run_dir / "mapanything" / name
            photoreal_scene = run_dir / "photoreal_input" / name
            sparse = photoreal_scene / "sparse"
            result = run_dir / "photoreal" / name
            metrics = result / "stats" / f"val_step{final_step}.json"
            if not (args.resume and metrics.exists()):
                timings[f"photoreal_export_{name}"] = run_stage(
                    f"11_photoreal_export_{index:02d}_{name}",
                    [
                        args.python,
                        PROJECT / "scripts" / "export_prediction_to_colmap.py",
                        prediction_dir / "raw_prediction.npz",
                        prediction_dir / "points_dense.ply",
                        window["path"],
                        sparse,
                        "--max-points", "80000",
                        "--prediction-size", "518", "294",
                    ],
                    run_dir,
                    environment=environment,
                )
                timings[f"photoreal_prepare_{name}"] = run_stage(
                    f"11_photoreal_prepare_{index:02d}_{name}",
                    [
                        args.python,
                        PROJECT / "scripts" / "prepare_gsplat_scene.py",
                        window["path"], sparse, photoreal_scene, "--factor", "4",
                    ],
                    run_dir,
                    environment=environment,
                )
                timings[f"photoreal_train_{name}"] = run_stage(
                    f"11_photoreal_train_{index:02d}_{name}",
                    [
                        args.python,
                        PROJECT / "scripts" / "run_gsplat_compat.py",
                        "default",
                        "--data-dir", photoreal_scene,
                        "--data-factor", "4",
                        "--result-dir", result,
                        "--max-steps", str(args.photoreal_steps),
                        "--eval-steps", str(args.photoreal_steps),
                        "--save-steps", str(args.photoreal_steps),
                        "--save-ply",
                        "--ply-steps", str(args.photoreal_steps),
                        "--disable-video",
                        "--disable-viewer",
                        "--packed",
                    ],
                    run_dir,
                    environment=environment,
                )
            splat = result / "ply" / f"point_cloud_{final_step}.ply"
            renders = sorted((result / "renders").glob(f"val_step{final_step}_*.png"))
            if not (splat.exists() and metrics.exists() and renders):
                raise RuntimeError(f"Incomplete photoreal output for {name}")
            destinations = {
                "splat": assets / f"reconstruction_splat_{name}.ply",
                "metrics": assets / f"photoreal_metrics_{name}.json",
                "preview": assets / f"photoreal_preview_{name}.png",
            }
            for source, destination in (
                (splat, destinations["splat"]),
                (metrics, destinations["metrics"]),
                (renders[len(renders) // 2], destinations["preview"]),
            ):
                shutil.copy2(source, destination)
            score = json.loads(metrics.read_text())
            photoreal_outputs.append(
                {
                    "window": name,
                    "gaussians": int(score["num_GS"]),
                    "psnr": float(score["psnr"]),
                    "ssim": float(score["ssim"]),
                    "lpips": float(score["lpips"]),
                    **{key: str(value) for key, value in destinations.items()},
                }
            )
    package_viewer(run_dir / "viewer", assets)

    stage_timings_path = run_dir / "stage_timings.json"
    stage_timings = (
        json.loads(stage_timings_path.read_text())
        if stage_timings_path.exists()
        else timings
    )

    report = {
        "construction_uses_ground_truth": False,
        "input": str(args.input_delivery.resolve()),
        "output": str(run_dir),
        "coordinate_reference_system": f"EPSG:{epsg}",
        "keyframes": len(list((keyframes).glob("frame_*.jpg"))),
        "windows": len(windows),
        "photoreal_enabled": args.photoreal,
        "photoreal_outputs": photoreal_outputs,
        "timings_seconds": stage_timings,
        "pipeline_stage_runtime_seconds": float(sum(stage_timings.values())),
        "current_invocation_runtime_seconds": time.perf_counter() - start,
        "viewer": str(run_dir / "viewer" / "index.html"),
        "submission": str(assets),
    }
    (run_dir / "pipeline_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
