"""Convert frozen DPVO video output into non-metric per-keyframe pose priors.

This reads no GPS or reference/ground-truth data. It only resamples a visual
camera trajectory to the exact video frames selected by mac_reconstruct.py.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation, Slerp


def build_priors(trajectory: Path, *, source_frames: int, stride: int, max_frames: int) -> dict:
    samples = np.loadtxt(trajectory, comments="#", ndmin=2)
    if samples.shape[1] != 8 or not np.isfinite(samples).all() or len(samples) < 3:
        raise ValueError("Expected finite TUM timestamp, XYZ, XYZW quaternion rows")
    t = samples[:, 0]
    if np.any(np.diff(t) <= 0) or not np.allclose(t, np.rint(t), atol=1e-6):
        raise ValueError("DPVO timestamps must be ordered integer video sample indices")
    if source_frames < 2 or stride < 1 or max_frames < 2:
        raise ValueError("Invalid source frame count, stride or keyframe budget")
    sampled_source = (np.rint(t).astype(int) + 1) * stride - 1
    if sampled_source.min() < 0 or sampled_source.max() >= source_frames:
        raise ValueError("DPVO trajectory extends beyond the source video")
    selected = np.unique(
        np.linspace(0, source_frames - 1, min(max_frames, source_frames)).round().astype(int)
    )
    # SciPy Slerp requires in-range times. The first skipped frames inherit
    # the first tracked orientation; no reference-derived correction is made.
    clipped = np.clip(selected, sampled_source[0], sampled_source[-1])
    orientations = Slerp(sampled_source.astype(float), Rotation.from_quat(samples[:, 4:8]))(
        clipped.astype(float)
    ).as_matrix()
    translations = np.column_stack(
        [np.interp(selected, sampled_source, samples[:, axis]) for axis in (1, 2, 3)]
    )
    poses = []
    for index, (source_frame, rotation, translation) in enumerate(
        zip(selected, orientations, translations)
    ):
        matrix = np.eye(4, dtype=float)
        matrix[:3, :3] = rotation
        matrix[:3, 3] = translation
        poses.append(
            {
                "image": f"frame_{index:05d}.jpg",
                "source_frame": int(source_frame),
                "camera_to_world": matrix.tolist(),
            }
        )
    return {
        "schema": "voxelflight.visual-pose-priors.v1",
        "method": "DPVO pretrained visual odometry, interpolated to selected video frames",
        "source_trajectory_sha256": hashlib.sha256(trajectory.read_bytes()).hexdigest(),
        "source_frames": source_frames,
        "visual_sample_stride": stride,
        "ground_truth_used": False,
        "metric_scale_established": False,
        "poses": poses,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--source-frames", required=True, type=int)
    parser.add_argument("--stride", required=True, type=int)
    parser.add_argument("--max-frames", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing prior file: {args.output}")
    result = build_priors(
        args.trajectory,
        source_frames=args.source_frames,
        stride=args.stride,
        max_frames=args.max_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(result['poses'])} non-reference camera priors to {args.output}")


if __name__ == "__main__":
    main()
