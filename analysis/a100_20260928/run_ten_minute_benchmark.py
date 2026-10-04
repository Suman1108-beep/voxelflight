"""Timed offline A100 reconstruction of a 10-minute drone MP4 + onboard GPS.

The official source JPEGs have been encoded as a derived 1080p H.264 MP4 before
this stopwatch. No survey/reference files enter the processing stages. Each
stage retains its own immutable outputs/log, enabling an interrupted run to be
audited without overwriting previous artifacts.
"""

import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("/workspace/voxelflight_a100_20260928")
INPUT = ROOT / "runs/zurich-10min-input"
RUN = ROOT / "runs/zurich-10min-360w48o8-v1"
PYTHON = str(ROOT / ".venv/bin/python")


def run_stage(name, command, env):
    output = RUN / name
    marker = output / "report.json"
    if marker.is_file():
        print(f"Existing complete {name} retained", flush=True)
        return 0.0, True
    if output.exists() and any(output.iterdir()):
        raise RuntimeError(f"Partial {name} requires diagnosis before retry: {output}")
    started = time.monotonic()
    with (RUN / f"{name}.log").open("w") as stream:
        subprocess.run(command, env=env, stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=3600)
    seconds = time.monotonic() - started
    if not marker.is_file():
        raise RuntimeError(f"{name} exited without its completion report")
    print(f"{name}: {seconds:.1f} s", flush=True)
    return seconds, False


def main():
    summary = json.loads((INPUT / "dataset_summary.json").read_text())
    if summary["frames"] != 18000 or not 595 <= summary["duration_s"] <= 605:
        raise ValueError("Expected the verified complete ten-minute flight")
    for name in ("flight.mp4", "video_telemetry.csv", "camera.json"):
        if not (INPUT / name).is_file():
            raise FileNotFoundError(INPUT / name)
    RUN.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HF_HOME=str(ROOT / "cache/huggingface"),
        TORCH_HOME=str(ROOT / "cache/torch"),
        DINO_SOURCE=str(ROOT / "dinov2"),
        HF_HUB_OFFLINE="1",
        OMP_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        LD_LIBRARY_PATH=str(ROOT / "runtime-packages/root/usr/lib/x86_64-linux-gnu"),
    )
    start = time.monotonic()
    stages = {}
    reused = {}
    stages["inference"], reused["inference"] = run_stage("inference", [
        PYTHON, "-u", str(ROOT / "repo/mac_reconstruct.py"),
        "--video", str(INPUT / "flight.mp4"),
        "--telemetry", str(INPUT / "video_telemetry.csv"),
        "--calibration", str(INPUT / "camera.json"),
        "--max-frames", "360", "--size", "518",
        "--window", "48", "--overlap", "8", "--save-predictions",
        "--output", str(RUN / "inference"),
    ], env)
    stages["fusion"], reused["fusion"] = run_stage("fusion", [
        PYTHON, "-u", str(ROOT / "repo/mac_refuse.py"),
        "--input", str(RUN / "inference"),
        "--output", str(RUN / "fusion"),
        "--voxel", "0.05", "--neighbor-radius", "8", "--min-support", "0",
    ], env)
    stages["georeferenced"], reused["georeferenced"] = run_stage("georeferenced", [
        PYTHON, "-u", str(ROOT / "repo/scripts/georeference_refused_mesh.py"),
        "--source", str(RUN / "inference"),
        "--fusion", str(RUN / "fusion"),
        "--output", str(RUN / "georeferenced"),
    ], env)
    wall = time.monotonic() - start
    continuous = not any(reused.values())
    inference = json.loads((RUN / "inference/report.json").read_text())
    fusion = json.loads((RUN / "fusion/report.json").read_text())
    report = dict(
        schema="voxelflight.ten-minute-benchmark.v1",
        input=str(INPUT / "flight.mp4"),
        input_provenance=summary["video_provenance"],
        input_duration_s=summary["duration_s"],
        input_source_frames=summary["frames"],
        selected_keyframes=inference["keyframes"],
        device=inference["device"],
        peak_gpu_allocation_gib=inference["peak_gpu_allocation_gib"],
        stage_seconds={name: None if reused[name] else seconds
                       for name, seconds in stages.items()},
        reused_existing_stages=reused,
        continuous_timed_run=continuous,
        processing_wall_seconds=wall if continuous else None,
        within_15_minute_processing_target=(wall < 900) if continuous else None,
        model=inference["model"],
        fused_triangles=fusion["triangles"],
        fused_components=fusion["connected_components"],
        fused_largest_component_fraction=fusion["largest_component_triangle_fraction"],
        georeferenced_output=str(RUN / "georeferenced"),
        ground_truth_used=False,
        independent_surface_accuracy_verified=False,
        note="Video preparation from official JPEGs is excluded; official moving-video input would already be an MP4/MOV.",
    )
    (RUN / "benchmark_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
