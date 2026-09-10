#!/usr/bin/env python3
"""Stitch feed-forward reconstruction windows with dense overlap geometry.

Adjacent windows contain the same video frames.  A pixel in an overlapping
frame therefore represents the same scene ray in both reconstructions.  We
back-project high-confidence pixels in each local coordinate system and fit a
robust Sim(3) from thousands of 3D correspondences.  No reference trajectory
is used to place or scale windows.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from align_metric import (
    apply_similarity,
    error_summary,
    interpolate_reference,
    load_manifest,
    load_reference,
    segment_scale_error,
    umeyama,
)
from evaluate_global_sfm import rigid_alignment, robust_metric_scale


@dataclass
class Window:
    path: Path
    names: list[str]
    extrinsics: np.ndarray
    intrinsics: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray

    @property
    def centres(self) -> np.ndarray:
        rotation = self.extrinsics[:, :3, :3]
        translation = self.extrinsics[:, :3, 3]
        return -np.einsum("nij,nj->ni", np.transpose(rotation, (0, 2, 1)), translation)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("keyframe_manifest", type=Path)
    parser.add_argument("ground_truth_csv", type=Path)
    parser.add_argument("window_npz", nargs="+", type=Path)
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--metric-window", type=Path, action="append", default=[])
    parser.add_argument("--samples-per-frame", type=int, default=1200)
    parser.add_argument("--confidence-percentile", type=float, default=75.0)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def load_window(path: Path) -> Window:
    data = np.load(path)
    return Window(
        path=path,
        names=[str(name) for name in data["image_names"]],
        extrinsics=np.asarray(data["extrinsics"], dtype=np.float64),
        intrinsics=np.asarray(data["intrinsics"], dtype=np.float64),
        depth=np.asarray(data["depth"], dtype=np.float64)[..., 0],
        confidence=np.asarray(data["confidence"], dtype=np.float64),
    )


def backproject(
    window: Window, frame_index: int, flat_indices: np.ndarray
) -> np.ndarray:
    height, width = window.depth.shape[1:]
    rows, columns = np.unravel_index(flat_indices, (height, width))
    pixels = np.column_stack([columns, rows, np.ones(len(rows))])
    rays = pixels @ np.linalg.inv(window.intrinsics[frame_index]).T
    camera_points = rays * window.depth[frame_index].ravel()[flat_indices, None]
    rotation = window.extrinsics[frame_index, :3, :3]
    translation = window.extrinsics[frame_index, :3, 3]
    return (camera_points - translation) @ rotation


def weighted_umeyama(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[float, np.ndarray, np.ndarray]:
    weights = np.maximum(weights, 0)
    weights = weights / weights.sum()
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    source_delta = source - source_mean
    target_delta = target - target_mean
    covariance = (target_delta * weights[:, None]).T @ source_delta
    left, singular_values, right_t = np.linalg.svd(covariance)
    signs = np.ones(3)
    if np.linalg.det(left @ right_t) < 0:
        signs[-1] = -1
    rotation = left @ np.diag(signs) @ right_t
    variance = np.sum(weights * np.sum(source_delta**2, axis=1))
    scale = float(np.sum(singular_values * signs) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    return scale, rotation, translation


def robust_similarity(
    source: np.ndarray,
    target: np.ndarray,
    confidence_weights: np.ndarray,
    iterations: int = 12,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    weights = confidence_weights / np.median(confidence_weights)
    weights = np.clip(weights, 0.05, 20.0)
    for _ in range(iterations):
        scale, rotation, translation = weighted_umeyama(source, target, weights)
        prediction = apply_similarity(source, scale, rotation, translation)
        residuals = np.linalg.norm(prediction - target, axis=1)
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        robust_sigma = max(1.4826 * mad, 1e-4)
        cutoff = median + 4.685 * robust_sigma
        normalized = residuals / max(cutoff, 1e-6)
        tukey = np.square(np.maximum(0.0, 1.0 - normalized**2))
        weights = confidence_weights * tukey
        if np.count_nonzero(weights) < 20:
            weights = confidence_weights / np.maximum(residuals, robust_sigma)
    inliers = residuals <= np.percentile(residuals, 80)
    return scale, rotation, translation, residuals, inliers


def overlap_correspondences(
    source: Window,
    target: Window,
    samples_per_frame: int,
    confidence_percentile: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    source_index = {name: index for index, name in enumerate(source.names)}
    target_index = {name: index for index, name in enumerate(target.names)}
    common = sorted(set(source_index) & set(target_index))
    source_points: list[np.ndarray] = []
    target_points: list[np.ndarray] = []
    weights: list[np.ndarray] = []
    for name in common:
        si, ti = source_index[name], target_index[name]
        source_depth = source.depth[si].ravel()
        target_depth = target.depth[ti].ravel()
        source_confidence = source.confidence[si].ravel()
        target_confidence = target.confidence[ti].ravel()
        joint_confidence = np.sqrt(
            np.maximum(source_confidence, 0) * np.maximum(target_confidence, 0)
        )
        valid = (
            np.isfinite(source_depth)
            & np.isfinite(target_depth)
            & (source_depth > 0.05)
            & (target_depth > 0.05)
            & (source_depth < 250.0)
            & (target_depth < 250.0)
            & np.isfinite(joint_confidence)
        )
        if not np.any(valid):
            continue
        threshold = np.percentile(joint_confidence[valid], confidence_percentile)
        candidates = np.flatnonzero(valid & (joint_confidence >= threshold))
        if len(candidates) > samples_per_frame:
            candidates = rng.choice(candidates, samples_per_frame, replace=False)
        source_points.append(backproject(source, si, candidates))
        target_points.append(backproject(target, ti, candidates))
        weights.append(joint_confidence[candidates])
    if not source_points:
        raise RuntimeError(f"No dense correspondences between {source.path} and {target.path}")
    return (
        np.concatenate(source_points),
        np.concatenate(target_points),
        np.concatenate(weights),
        common,
    )


def edge_weight(index: int, length: int) -> float:
    if length <= 1:
        return 1.0
    distance = abs((2.0 * index / (length - 1)) - 1.0)
    return float(0.2 + 0.8 * np.cos(0.5 * np.pi * distance) ** 2)


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = [load_window(path) for path in args.window_npz]
    rng = np.random.default_rng(args.seed)
    observations: dict[str, list[tuple[np.ndarray, float]]] = {}
    transforms: list[tuple[float, np.ndarray, np.ndarray]] = []
    reports: list[dict[str, object]] = []

    for index, window in enumerate(windows):
        if index == 0:
            scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
            report: dict[str, object] = {
                "window": str(window.path),
                "overlap_frames": 0,
                "correspondences": 0,
            }
        else:
            source_sets: list[np.ndarray] = []
            target_sets: list[np.ndarray] = []
            weight_sets: list[np.ndarray] = []
            common_names: set[str] = set()
            overlap_edges = 0
            for previous, previous_transform in zip(windows[:index], transforms):
                if not (set(window.names) & set(previous.names)):
                    continue
                source_edge, target_edge, weights_edge, common = overlap_correspondences(
                    window,
                    previous,
                    args.samples_per_frame,
                    args.confidence_percentile,
                    rng,
                )
                previous_scale, previous_rotation, previous_translation = previous_transform
                source_sets.append(source_edge)
                target_sets.append(
                    apply_similarity(
                        target_edge,
                        previous_scale,
                        previous_rotation,
                        previous_translation,
                    )
                )
                weight_sets.append(weights_edge)
                common_names.update(common)
                overlap_edges += 1
            if not source_sets:
                raise RuntimeError(f"{window.path} has no overlap with a placed window")
            source = np.concatenate(source_sets)
            target_global = np.concatenate(target_sets)
            weights = np.concatenate(weight_sets)
            scale, rotation, translation, residuals, inliers = robust_similarity(
                source, target_global, weights
            )
            report = {
                "window": str(window.path),
                "overlap_edges": overlap_edges,
                "unique_overlap_frames": len(common_names),
                "correspondences": len(source),
                "inlier_correspondences": int(inliers.sum()),
                "overlap_dense_rmse_all": float(np.sqrt(np.mean(residuals**2))),
                "overlap_dense_rmse_inliers": float(
                    np.sqrt(np.mean(residuals[inliers] ** 2))
                ),
            }
        transforms.append((scale, rotation, translation))
        report.update(
            {
                "scale_to_global": float(scale),
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )
        reports.append(report)
        centres = apply_similarity(window.centres, scale, rotation, translation)
        for local_index, (name, centre) in enumerate(zip(window.names, centres)):
            observations.setdefault(name, []).append(
                (centre, edge_weight(local_index, len(window.names)))
            )

    manifest = load_manifest(args.keyframe_manifest)
    names = np.asarray(sorted(observations, key=lambda name: manifest[name]))
    stitched = np.asarray(
        [
            np.average(
                np.asarray([point for point, _ in observations[name]]),
                axis=0,
                weights=np.asarray([weight for _, weight in observations[name]]),
            )
            for name in names
        ]
    )
    image_ids = np.asarray(
        [manifest[name] + args.source_image_offset for name in names]
    )
    sample_ids, gt_samples, gps_samples = load_reference(args.ground_truth_csv)
    ground_truth = interpolate_reference(image_ids, sample_ids, gt_samples)
    gps = interpolate_reference(image_ids, sample_ids, gps_samples)

    shape_scale, shape_rotation, shape_translation = umeyama(stitched, ground_truth)
    shape_aligned = apply_similarity(
        stitched, shape_scale, shape_rotation, shape_translation
    )
    metric_scale, metric_estimates = robust_metric_scale(
        names, stitched, args.metric_window
    )
    metric = stitched * metric_scale
    metric_rotation, metric_translation = rigid_alignment(metric, ground_truth)
    metric_aligned = metric @ metric_rotation.T + metric_translation
    gps_rotation, gps_translation = rigid_alignment(metric, gps)
    georeferenced = metric @ gps_rotation.T + gps_translation

    result = {
        "protocol": {
            "ground_truth_used_to_stitch_windows": False,
            "ground_truth_used_for_metric_scale": False,
            "ground_truth_use": "post-hoc benchmark scoring only",
        },
        "frames": len(names),
        "windows": reports,
        "sim3_shape_error": error_summary(shape_aligned, ground_truth),
        "metric_scale": {
            "estimated_m_per_model_unit": metric_scale,
            "independent_window_estimates": metric_estimates,
            "se3_error": error_summary(metric_aligned, ground_truth),
            "segment_length_error": segment_scale_error(metric_aligned, ground_truth),
        },
        "gps_georeferenced_absolute_error": error_summary(
            georeferenced, ground_truth
        ),
    }
    np.savez_compressed(
        args.output_dir / "trajectory.npz",
        image_names=names,
        image_ids=image_ids,
        camera_centers=stitched,
        metric_camera_centers=metric,
    )
    with (args.output_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["image", "image_id", "x", "y", "z", "shape_x", "shape_y", "shape_z", "gt_x", "gt_y", "gt_z"]
        )
        for name, image_id, point, shape, truth in zip(
            names, image_ids, stitched, shape_aligned, ground_truth
        ):
            writer.writerow([name, image_id, *point, *shape, *truth])
    (args.output_dir / "evaluation.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
