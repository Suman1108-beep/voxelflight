"""Post-hoc camera-path diagnostic for a frozen DPVO video trajectory.

The reference is deliberately loaded only by this evaluator, never by DPVO.
The Sim(3) result is an evaluation-only *trajectory* score, not an absolute
georeference or a 3D surface-accuracy measurement.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
from align_metric import apply_similarity, error_summary, load_reference, umeyama  # noqa: E402


def read_tum_positions(path: Path) -> tuple[np.ndarray, np.ndarray]:
    rows = []
    for line in path.read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        values = [float(part) for part in line.split()]
        if len(values) != 8:
            raise ValueError("Expected TUM timestamp, XYZ and XYZW quaternion")
        rows.append(values)
    trajectory = np.asarray(rows, dtype=float)
    if trajectory.ndim != 2 or len(trajectory) < 3 or not np.isfinite(trajectory).all():
        raise ValueError("DPVO trajectory must contain at least three finite poses")
    time = trajectory[:, 0]
    if np.any(np.diff(time) <= 0):
        raise ValueError("DPVO timestamps must be strictly increasing")
    return time, trajectory[:, 1:4]


def evaluate(
    trajectory_path: Path,
    reference_path: Path,
    *,
    first_image_id: int,
    stride: int,
    source_frames: int,
) -> dict:
    if stride < 1 or source_frames < 3:
        raise ValueError("Invalid video sampling configuration")
    time, cameras = read_tum_positions(trajectory_path)
    sample_index = np.rint(time).astype(int)
    if not np.allclose(time, sample_index, atol=1e-6):
        raise ValueError("Expected DPVO's contiguous integer video sample timestamps")
    source_index = (sample_index + 1) * stride - 1
    if source_index.min() < 0 or source_index.max() >= source_frames:
        raise ValueError("DPVO sample exceeds the source video")
    image_id = source_index + first_image_id

    ids, xyz, _ = load_reference(reference_path)
    order = np.argsort(ids)
    ids, xyz = ids[order], xyz[order]
    if np.any(np.diff(ids) <= 0) or image_id.min() < ids.min() or image_id.max() > ids.max():
        raise ValueError("Reference does not cover all output frames uniquely")
    truth = np.column_stack([np.interp(image_id, ids, xyz[:, axis]) for axis in range(3)])
    scale, rotation, translation = umeyama(cameras, truth)
    aligned = apply_similarity(cameras, scale, rotation, translation)
    return {
        "protocol": "Frozen DPVO video poses; reference used for post-hoc scoring only",
        "trajectory_sha256": hashlib.sha256(trajectory_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
        "frames": len(cameras),
        "source_image_ids": [int(image_id.min()), int(image_id.max())],
        "source_frame_stride": stride,
        "sim3_aligned_camera_error_m": error_summary(aligned, truth),
        "sim3_evaluation_scale": float(scale),
        "direct_absolute_camera_error_m": None,
        "surface_rmse_m": None,
        "sih_spatial_accuracy_verified": False,
        "warning": "Sim(3) fits withheld camera reference for evaluation; it cannot be used for inference or claimed as surface accuracy.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--first-image-id", required=True, type=int)
    parser.add_argument("--stride", required=True, type=int)
    parser.add_argument("--source-frames", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {args.output}")
    result = evaluate(
        args.trajectory,
        args.reference,
        first_image_id=args.first_image_id,
        stride=args.stride,
        source_frames=args.source_frames,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
