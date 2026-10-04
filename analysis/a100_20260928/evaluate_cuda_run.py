"""Score a frozen reconstruction against withheld Zurich camera reference poses.

This script is deliberately separate from inference. It never modifies the
reconstruction, and camera error is not a proxy for mesh/surface accuracy.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'repo/scripts'))
from align_metric import apply_similarity, error_summary, load_reference, segment_scale_error, umeyama


def rigid(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    src = source - source.mean(0)
    dst = target - target.mean(0)
    u, _, vt = np.linalg.svd(dst.T @ src)
    correction = np.diag([1.0, 1.0, np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    return src @ rotation.T + target.mean(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--reference', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError('Evaluation output already exists; refusing to overwrite.')
    report_path = args.run / 'report.json'
    report = json.loads(report_path.read_text())
    if report.get('ground_truth_used') is not False:
        raise ValueError('Run must explicitly exclude ground truth from reconstruction.')
    for name, digest in report['checksums'].items():
        if hashlib.sha256((args.run / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f'Frozen artifact checksum mismatch: {name}')
    poses = np.load(args.run / 'camera_poses.npz')['camera_to_world']
    ids = np.asarray([int(Path(frame['source_name']).stem) for frame in report['frames']])
    predicted = np.asarray(poses[:, :3, 3], dtype=np.float64)
    samples, points, _ = load_reference(args.reference)
    order = np.argsort(samples)
    samples, points = samples[order], points[order]
    if len(np.unique(samples)) != len(samples) or ids.min() < samples.min() or ids.max() > samples.max():
        raise ValueError('Reference does not uniquely cover every predicted camera.')
    truth = np.column_stack([np.interp(ids, samples, points[:, axis]) for axis in range(3)])
    scale, rotation, translation = umeyama(predicted, truth)
    sim3 = apply_similarity(predicted, scale, rotation, translation)
    se3 = rigid(predicted, truth)
    result = {
        'protocol': 'Frozen camera trajectory, reference used for post-hoc scoring only',
        'run_report_sha256': hashlib.sha256(report_path.read_bytes()).hexdigest(),
        'reference_sha256': hashlib.sha256(args.reference.read_bytes()).hexdigest(),
        'reference': str(args.reference),
        'keyframes': len(ids),
        'source_image_ids': [int(ids.min()), int(ids.max())],
        'sim3_aligned_camera_rmse_m': error_summary(sim3, truth),
        'se3_aligned_camera_rmse_m': error_summary(se3, truth),
        'sim3_evaluation_scale': scale,
        'se3_segment_length_error': segment_scale_error(se3, truth),
        'surface_rmse_m': None,
        'sih_spatial_accuracy_verified': False,
        'warning': 'Aligned camera scores use reference poses for evaluation; they are not absolute positioning or mesh surface scores.',
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
