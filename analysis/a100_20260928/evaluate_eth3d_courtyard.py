"""Evaluate a frozen image-only reconstruction against ETH3D courtyard scans.

ETH3D camera poses are used only here, after reconstruction, for one global
Sim(3) evaluation alignment. This is a deterministic nearest-neighbour proxy,
not the official ETH3D free-space-aware evaluator or an SIH drone/GPS test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from plyfile import PlyData
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def reference_centres(path: Path) -> dict[str, np.ndarray]:
    lines = [line.strip() for line in path.open() if line.strip() and not line.startswith('#')]
    if len(lines) % 2:
        raise ValueError('ETH3D images.txt must contain two lines per image.')
    centres = {}
    for line in lines[::2]:
        fields = line.split()
        if len(fields) != 10:
            raise ValueError('Unexpected ETH3D image-pose record.')
        q = np.asarray([float(fields[i]) for i in (2, 3, 4, 1)])
        translation = np.asarray([float(fields[i]) for i in (5, 6, 7)])
        rotation = Rotation.from_quat(q).as_matrix()
        centres[Path(fields[9]).name] = -rotation.T @ translation
    return centres


def similarity(source: np.ndarray, target: np.ndarray):
    source_mean, target_mean = source.mean(0), target.mean(0)
    centred_source, centred_target = source - source_mean, target - target_mean
    u, singular, vt = np.linalg.svd(centred_target.T @ centred_source / len(source))
    correction = np.diag([1., 1., np.linalg.det(u @ vt)])
    rotation = u @ correction @ vt
    variance = np.mean(np.sum(centred_source ** 2, axis=1))
    if variance < 1e-8:
        raise ValueError('Predicted camera centres are degenerate.')
    scale = float(np.sum(singular * np.diag(correction)) / variance)
    translation = target_mean - scale * rotation @ source_mean
    return scale, rotation, translation


def ply_sample(path: Path, limit: int, seed: int) -> tuple[np.ndarray, int]:
    cloud = PlyData.read(str(path), mmap=True)
    vertices = cloud['vertex'].data
    count = len(vertices)
    rng = np.random.default_rng(seed)
    indices = rng.choice(count, min(count, limit), replace=False)
    points = np.column_stack([vertices[axis][indices] for axis in ('x', 'y', 'z')]).astype(np.float64)
    return points, count


def scan_project(path: Path) -> list[tuple[Path, np.ndarray]]:
    root = ET.parse(path).getroot()
    result = []
    for mesh in root.findall('.//MLMesh'):
        values = np.fromstring(mesh.findtext('MLMatrix44') or '', sep=' ')
        if values.size != 16:
            raise ValueError('Invalid ETH3D scan alignment matrix.')
        result.append((path.parent / mesh.attrib['filename'], values.reshape(4, 4)))
    if not result:
        raise ValueError('No aligned ETH3D evaluation scans found.')
    return result


def run(args):
    if args.output.exists():
        raise ValueError('Refusing to overwrite an existing evaluation report.')
    run_report = json.loads((args.run / 'report.json').read_text())
    if run_report.get('ground_truth_used') is not False:
        raise ValueError('Reconstruction must explicitly exclude ground truth.')
    pose_run = args.poses_from or args.run
    pose_report = json.loads((pose_run / 'report.json').read_text())
    if pose_report.get('ground_truth_used') is not False:
        raise ValueError('Pose reconstruction must explicitly exclude ground truth.')
    frames = [frame['source_name'] for frame in pose_report['frames']]
    reference = reference_centres(args.dataset / 'dslr_calibration_undistorted/images.txt')
    if any(name not in reference for name in frames):
        raise ValueError('Reference camera list does not cover all predicted views.')
    predicted_poses = np.load(pose_run / 'camera_poses.npz')['camera_to_world']
    predicted_centres = predicted_poses[:, :3, 3].astype(np.float64)
    target_centres = np.stack([reference[name] for name in frames])
    scale, rotation, translation = similarity(predicted_centres, target_centres)
    aligned_centres = scale * predicted_centres @ rotation.T + translation
    camera_rmse = float(np.sqrt(np.mean(np.sum((aligned_centres - target_centres) ** 2, axis=1))))

    reconstructed, reconstructed_total = ply_sample(args.run / 'pointcloud.ply', args.max_points, 17)
    reconstructed = scale * reconstructed @ rotation.T + translation
    scan_paths = scan_project(args.dataset / 'dslr_scan_eval/scan_alignment.mlp')
    scanned_parts, scanned_total = [], 0
    for index, (path, matrix) in enumerate(scan_paths):
        points, total = ply_sample(path, args.max_points // len(scan_paths), 701 + index)
        scanned_parts.append(points @ matrix[:3, :3].T + matrix[:3, 3])
        scanned_total += total
    scanned = np.concatenate(scanned_parts)
    accuracy_distances = cKDTree(scanned).query(reconstructed, workers=args.workers)[0]
    completeness_distances = cKDTree(reconstructed).query(scanned, workers=args.workers)[0]
    thresholds = (0.05, 0.1, 0.25, 0.5, 1.0)
    threshold_scores = {}
    for threshold in thresholds:
        accuracy = float(np.mean(accuracy_distances <= threshold))
        completeness = float(np.mean(completeness_distances <= threshold))
        threshold_scores[str(threshold)] = dict(
            accuracy_fraction=accuracy, completeness_fraction=completeness,
            f1=2 * accuracy * completeness / (accuracy + completeness) if accuracy + completeness else 0.)
    result = dict(
        protocol='ETH3D courtyard sampled nearest-neighbour proxy, NOT official ETH3D evaluator',
        reconstruction='Images only; reference cameras and scans withheld until this post-hoc evaluation',
        alignment='One global Sim(3) fit to reference camera centres for evaluation only',
        not_sih_compliance=True, not_absolute_georeference=True,
        camera_sim3_rmse_m=camera_rmse, evaluation_scale=scale,
        reconstructed_points_total=reconstructed_total, reconstructed_points_sampled=len(reconstructed),
        scan_points_total=scanned_total, scan_points_sampled=len(scanned),
        accuracy_distance_quantiles_m={str(q): float(np.quantile(accuracy_distances, q)) for q in (.5, .9, .95)},
        completeness_distance_quantiles_m={str(q): float(np.quantile(completeness_distances, q)) for q in (.5, .9, .95)},
        thresholds_m=threshold_scores,
        source_sha256={'geometry_report.json': sha256(args.run / 'report.json'),
                       'geometry_pointcloud.ply': sha256(args.run / 'pointcloud.ply'),
                       'pose_report.json': sha256(pose_run / 'report.json'),
                       'pose_camera_poses.npz': sha256(pose_run / 'camera_poses.npz')},
        reference_sha256={path.name: sha256(path) for path, _ in scan_paths},
        limitations=['ETH3D DSLR courtyard is not a single-pass drone video and has no GPS.',
                     'Potential overlap with the pretrained model training corpus has not been ruled out.',
                     'Camera-reference Sim(3) alignment is post-hoc and cannot be used in deployment.',
                     'Sampling and nearest-neighbour distances omit the official evaluator free-space model.',
                     'The reference scans may cover surfaces outside the reconstructed views.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--poses-from', type=Path,
                        help='Original inference run with frozen predicted cameras, if geometry was re-fused.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-points', type=int, default=200_000)
    parser.add_argument('--workers', type=int, default=8)
    run(parser.parse_args())
