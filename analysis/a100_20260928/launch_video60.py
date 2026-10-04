"""Launch the separate Zurich 60-second video/GPS reconstruction on A100.

The MP4 is a documented CFR encoding of official contiguous drone JPEGs, not an
original camera MP4. Onboard GPS is allowed for reconstruction; reference poses
remain evaluation-only. The output directory is never overwritten.
"""

import argparse
import os
from pathlib import Path
import subprocess


ROOT = Path("/workspace/voxelflight_a100_20260928")
INPUT = ROOT / "runs/zurich-disjoint-video60-input"
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frames", type=int, default=180)
    parser.add_argument("--window", type=int, default=24)
    parser.add_argument("--overlap", type=int, default=6)
    parser.add_argument("--tag", default="v1")
    args = parser.parse_args()
    if not 2 <= args.overlap < args.window or not 2 <= args.frames <= 720:
        parser.error("Invalid frame, window or overlap setting")
    run_name = (
        f"zurich-disjoint-video60-mapanything{args.frames}-w{args.window}"
        f"-o{args.overlap}-{args.tag}"
    )
    output = ROOT / "runs" / run_name
    log = ROOT / "logs" / f"{run_name}.log"
    if not (INPUT / "dataset_summary.json").is_file():
        raise FileNotFoundError("The verified MP4/GPS input has not finished preparing")
    if output.exists():
        raise FileExistsError(f"Preserving existing run: {output}")
    log.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        HF_HOME=str(ROOT / "cache/huggingface"),
        TORCH_HOME=str(ROOT / "cache/torch"),
        DINO_SOURCE=str(ROOT / "dinov2"),
        OMP_NUM_THREADS="4",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
        HF_HUB_OFFLINE="1",
    )
    command = [
        str(ROOT / ".venv/bin/python"), "-u", str(ROOT / "repo/mac_reconstruct.py"),
        "--video", str(INPUT / "flight.mp4"),
        "--telemetry", str(INPUT / "video_telemetry.csv"),
        "--calibration", str(INPUT / "camera.json"),
        "--max-frames", str(args.frames), "--size", "518", "--window", str(args.window),
        "--overlap", str(args.overlap), "--save-predictions", "--output", str(output),
    ]
    with log.open("w") as stream:
        process = subprocess.Popen(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    print(f"pid={process.pid} log={log} output={output}", flush=True)


if __name__ == "__main__":
    main()
