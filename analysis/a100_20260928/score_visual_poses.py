"""Post-hoc trajectory scoring for frozen, non-metric visual SfM pose reports."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path('/workspace/voxelflight_a100_20260928')
sys.path.insert(0, str(ROOT / 'repo/scripts'))
sys.path.insert(0, str(ROOT / 'repo'))
from align_metric import apply_similarity, error_summary, load_reference, umeyama
from georeference_mac_run import load_gps, weighted_similarity
from mac_geometry import transform_points


def rigid(source, target):
    x, y = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(y.T @ x)
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return x @ r.T + target.mean(0)


parser = argparse.ArgumentParser()
parser.add_argument('--report', type=Path, required=True)
parser.add_argument('--reference', type=Path,
                    default=ROOT / 'datasets/zurich-61201-63000/Log Files/GroundTruthAGL.csv')
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise ValueError('Refusing to replace a score.')
report = json.loads(args.report.read_text())
assert report['ground_truth_used'] is False
poses = report['poses']
ids = np.array([int(Path(item['image']).stem) for item in poses])
source = np.array([np.array(item['camera_to_world'])[:3, 3] for item in poses])
reference_path = args.reference
samples, points, _ = load_reference(reference_path)
order = np.argsort(samples)
truth = np.column_stack([np.interp(ids, samples[order], points[order, a]) for a in range(3)])
gps, sigma, _, epsg = load_gps(ROOT / 'datasets/frame_telemetry.csv',
                               [{'source_name': item['image']} for item in poses])
origin = np.median(gps, axis=0)
gps_scale, gps_r, gps_t, quality = weighted_similarity(source, gps-origin, sigma)
georef = transform_points(source, gps_scale, gps_r, gps_t) + origin
shape_scale, shape_r, shape_t = umeyama(source, truth)
shape = apply_similarity(source, shape_scale, shape_r, shape_t)
result = {'frames': len(ids), 'source_range': [int(ids.min()), int(ids.max())],
          'ground_truth_used_in_reconstruction': False,
          'gps_scale_m_per_sfm_unit': gps_scale,
          'gps_fit_rmse_m': quality['fit_rmse_m'],
          'sim3_aligned_camera_error': error_summary(shape, truth),
          'gps_scaled_se3_camera_error': error_summary(rigid(georef, truth), truth),
          'absolute_camera_error': error_summary(georef, truth),
          'surface_rmse_m': None,
          'warning': 'Camera trajectory only; reference used post hoc for scoring.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(result, indent=2) + '\n')
print(json.dumps(result, indent=2))
