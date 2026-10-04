"""Evaluate a frozen MP4/GPS drone run against withheld Zurich camera positions.

This is a camera-trajectory diagnostic, never a 3D surface score. A Sim(3)
trajectory fit is evaluation-only; the direct absolute metric uses only the
already-frozen GPS alignment saved by inference.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

_HERE = Path(__file__).resolve()
_SCRIPT_DIRS = (
    _HERE.parents[1] / "repo/scripts",  # isolated A100 tools directory
    _HERE.parents[2] / "scripts",       # local source checkout
)
for _directory in _SCRIPT_DIRS:
    if (_directory / "align_metric.py").is_file():
        sys.path.insert(0, str(_directory))
        break
else:
    raise FileNotFoundError("Cannot locate the bundled trajectory scoring utilities")
from align_metric import apply_similarity, error_summary, load_reference, umeyama


def evaluate(run: Path, reference: Path, first_image_id: int) -> dict:
    report_path = run / "report.json"
    report = json.loads(report_path.read_text())
    if report.get("ground_truth_used") is not False:
        raise ValueError("Reconstruction provenance must explicitly exclude ground truth")
    if report.get("input", {}).get("duration_s") is None:
        raise ValueError("Expected an MP4/MOV video run, not an image sequence")
    for name, expected in report["checksums"].items():
        actual = hashlib.sha256((run / name).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"Frozen artifact checksum mismatch: {name}")

    frames = report["frames"]
    frame_numbers = np.asarray([frame["source_frame"] for frame in frames], dtype=int)
    if np.any(frame_numbers < 0) or np.any(np.diff(frame_numbers) <= 0):
        raise ValueError("Source-video frames must be nonnegative and strictly ordered")
    image_ids = frame_numbers + first_image_id
    with np.load(run / "camera_poses.npz") as data:
        predicted_local = np.asarray(data["camera_to_world"][:, :3, 3], dtype=float)
    if len(predicted_local) != len(image_ids):
        raise ValueError("Frame and saved camera count differ")

    ids, reference_points, _ = load_reference(reference)
    order = np.argsort(ids)
    ids, reference_points = ids[order], reference_points[order]
    if np.any(np.diff(ids) <= 0) or image_ids.min() < ids.min() or image_ids.max() > ids.max():
        raise ValueError("Reference does not uniquely cover the selected video frames")
    truth = np.column_stack(
        [np.interp(image_ids, ids, reference_points[:, axis]) for axis in range(3)]
    )
    scale, rotation, translation = umeyama(predicted_local, truth)
    aligned = apply_similarity(predicted_local, scale, rotation, translation)
    result = {
        "protocol": "Frozen video/GPS camera poses; reference loaded only for post-hoc scoring",
        "run_report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "reference_sha256": hashlib.sha256(reference.read_bytes()).hexdigest(),
        "video_duration_s": report["input"]["duration_s"],
        "keyframes": len(image_ids),
        "source_image_ids": [int(image_ids.min()), int(image_ids.max())],
        "sim3_aligned_camera_error_m": error_summary(aligned, truth),
        "sim3_evaluation_scale": float(scale),
        "surface_rmse_m": None,
        "sih_spatial_accuracy_verified": False,
        "warning": "Camera trajectory error is not mesh surface error. Sim(3) uses withheld reference only for evaluation.",
    }
    if report.get("georeferenced"):
        origin = np.asarray(report["utm_origin"], dtype=float)
        if origin.shape != (3,):
            raise ValueError("Expected a three-dimensional UTM origin")
        result["direct_absolute_camera_error_m"] = error_summary(
            predicted_local + origin, truth
        )
    else:
        result["direct_absolute_camera_error_m"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--first-image-id", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Preserving existing evaluation: {args.output}")
    result = evaluate(args.run, args.reference, args.first_image_id)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
