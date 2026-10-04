"""Compare frozen 90-view GPU runs after onboard-GPS-only scale/location fit."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np

root = Path('/workspace/voxelflight_a100_20260928')
sys.path.insert(0, str(root / 'repo/scripts'))
sys.path.insert(0, str(root / 'repo'))
from align_metric import error_summary, load_reference, umeyama, apply_similarity
from georeference_mac_run import load_gps, weighted_similarity
from mac_geometry import transform_points


def rigid(source, target):
    x, y = source - source.mean(0), target - target.mean(0)
    u, _, vt = np.linalg.svd(y.T @ x)
    r = u @ np.diag([1, 1, np.linalg.det(u @ vt)]) @ vt
    return x @ r.T + target.mean(0)


dataset = root / 'datasets'
reference = dataset / 'zurich-61201-63000/Log Files/GroundTruthAGL.csv'
telemetry = dataset / 'frame_telemetry.csv'
sample_ids, samples, _ = load_reference(reference)
order = np.argsort(sample_ids)
sample_ids, samples = sample_ids[order], samples[order]
results = []
for tag in ['baseline90-420-w24-o2', '90-420-w24-o6',
            '90-518-w24-o6', '90-518-w48-o8', '180-518-w24-o6']:
    run = root / 'runs' / tag
    if not (run / 'report.json').exists():
        continue
    if (run / 'gps_scoring.json').exists():
        results.append(json.loads((run / 'gps_scoring.json').read_text()))
        continue
    report = json.loads((run / 'report.json').read_text())
    assert report['ground_truth_used'] is False
    poses = np.load(run / 'camera_poses.npz')['camera_to_world']
    source = poses[:, :3, 3]
    gps, sigma, _, epsg = load_gps(telemetry, report['frames'])
    origin = np.median(gps, axis=0)
    scale, rotation, translation, gps_fit = weighted_similarity(source, gps - origin, sigma)
    georef = transform_points(source, scale, rotation, translation) + origin
    ids = np.array([int(Path(frame['source_name']).stem) for frame in report['frames']])
    truth = np.column_stack([np.interp(ids, sample_ids, samples[:, a]) for a in range(3)])
    sim_scale, sim_r, sim_t = umeyama(georef, truth)
    item = {'run': tag, 'ground_truth_used_in_fit': False,
            'output_crs': f'EPSG:{epsg}', 'gps_scale': scale,
            'gps_fit_rmse_m': gps_fit['fit_rmse_m'],
            'absolute_camera_error': error_summary(georef, truth),
            'gps_scale_se3_camera_error': error_summary(rigid(georef, truth), truth),
            'sim3_camera_error': error_summary(apply_similarity(georef, sim_scale, sim_r, sim_t), truth),
            'surface_rmse_m': None,
            'warning': 'GPS fit is not independent. Absolute camera error uses withheld reference only after GPS fit. No surface accuracy is measured.'}
    (run / 'gps_scoring.json').write_text(json.dumps(item, indent=2) + '\n')
    results.append(item)
(root / 'runs/gps_variant_comparison.json').write_text(json.dumps(results, indent=2) + '\n')
print(json.dumps(results, indent=2))
