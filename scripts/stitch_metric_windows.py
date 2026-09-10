#!/usr/bin/env python3
"""Stitch overlapping metric reconstruction windows without ground-truth poses.

The window-to-window transforms are estimated only from camera centres shared by
overlapping predictions. Ground truth is optional and is used strictly after
stitching for benchmark reporting.
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
    load_reference,
    segment_scale_error,
    umeyama,
)


@dataclass(frozen=True)
class Window:
    path: Path
    names: np.ndarray
    image_ids: np.ndarray
    centres: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("window_predictions", nargs="+", type=Path)
    parser.add_argument(
        "--alignment",
        choices=("rigid", "similarity", "regularized_similarity"),
        default="regularized_similarity",
    )
    parser.add_argument(
        "--scale-strength",
        type=float,
        default=0.35,
        help="Fraction of the overlap-estimated log scale to apply (0=rigid, 1=Sim3).",
    )
    parser.add_argument("--ground-truth-csv", type=Path)
    parser.add_argument("--minimum-overlap", type=int, default=6)
    return parser.parse_args()


def load_window(path: Path) -> Window:
    raw = np.load(path)
    return Window(
        path=path,
        names=raw["image_names"].astype(str),
        image_ids=raw["image_ids"].astype(int),
        centres=raw["camera_centers"].astype(np.float64),
    )


def weighted_rigid(
    source: np.ndarray, target: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = weights / weights.sum()
    source_mean = np.sum(source * weights[:, None], axis=0)
    target_mean = np.sum(target * weights[:, None], axis=0)
    covariance = (weights[:, None] * (target - target_mean)).T @ (
        source - source_mean
    )
    left, _, right_t = np.linalg.svd(covariance)
    correction = np.eye(3)
    correction[-1, -1] = np.sign(np.linalg.det(left @ right_t))
    rotation = left @ correction @ right_t
    translation = target_mean - rotation @ source_mean
    return rotation, translation


def robust_transform(
    source: np.ndarray,
    target: np.ndarray,
    alignment: str,
    scale_strength: float,
    iterations: int = 8,
) -> tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    weights = np.ones(len(source), dtype=np.float64)
    scale = 1.0
    for _ in range(iterations):
        rotation, translation = weighted_rigid(scale * source, target, weights)
        if alignment != "rigid":
            rotated = source @ rotation.T
            source_mean = np.average(rotated, axis=0, weights=weights)
            target_mean = np.average(target, axis=0, weights=weights)
            numerator = np.sum(
                weights[:, None]
                * (rotated - source_mean)
                * (target - target_mean)
            )
            denominator = np.sum(
                weights[:, None] * (rotated - source_mean) ** 2
            )
            estimated_scale = max(float(numerator / denominator), 1e-6)
            strength = 1.0 if alignment == "similarity" else scale_strength
            scale = float(np.exp(strength * np.log(estimated_scale)))
            rotation, translation = weighted_rigid(scale * source, target, weights)
        prediction = apply_similarity(source, scale, rotation, translation)
        residuals = np.linalg.norm(prediction - target, axis=1)
        median = np.median(residuals)
        mad = np.median(np.abs(residuals - median))
        huber_delta = max(1.4826 * mad * 1.5, 0.05)
        weights = np.minimum(1.0, huber_delta / np.maximum(residuals, 1e-9))
    return scale, rotation, translation, residuals


def edge_weight(index: int, length: int) -> float:
    """Downweight predictions at the less stable edges of each local window."""
    if length <= 1:
        return 1.0
    normalized = abs((2.0 * index / (length - 1)) - 1.0)
    return float(0.25 + 0.75 * np.cos(0.5 * np.pi * normalized) ** 2)


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.scale_strength <= 1.0:
        raise ValueError("--scale-strength must be between zero and one")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    windows = sorted(
        (load_window(path) for path in args.window_predictions),
        key=lambda window: int(window.image_ids.min()),
    )
    if not windows:
        raise ValueError("At least one window prediction is required")

    observations: dict[str, list[tuple[np.ndarray, float, int]]] = {}
    image_id_by_name: dict[str, int] = {}
    transforms: list[dict[str, object]] = []
    for window_index, window in enumerate(windows):
        if window_index == 0:
            scale, rotation, translation = 1.0, np.eye(3), np.zeros(3)
            residuals = np.zeros(0)
            overlap_names: list[str] = []
        else:
            indices = [index for index, name in enumerate(window.names) if name in observations]
            if len(indices) < args.minimum_overlap:
                raise RuntimeError(
                    f"{window.path} has only {len(indices)} already-placed frames; "
                    f"need {args.minimum_overlap}"
                )
            overlap_names = [str(window.names[index]) for index in indices]
            source = window.centres[indices]
            target = np.asarray(
                [
                    np.average(
                        np.asarray([point for point, _, _ in observations[name]]),
                        axis=0,
                        weights=np.asarray([weight for _, weight, _ in observations[name]]),
                    )
                    for name in overlap_names
                ]
            )
            scale, rotation, translation, residuals = robust_transform(
                source,
                target,
                alignment=args.alignment,
                scale_strength=args.scale_strength,
            )
        transformed = apply_similarity(window.centres, scale, rotation, translation)
        for local_index, (name, image_id, point) in enumerate(
            zip(window.names, window.image_ids, transformed)
        ):
            name = str(name)
            observations.setdefault(name, []).append(
                (point, edge_weight(local_index, len(window.names)), window_index)
            )
            image_id_by_name[name] = int(image_id)
        transforms.append(
            {
                "window": str(window.path),
                "frames": int(len(window.names)),
                "overlap_frames": len(overlap_names),
                "estimated_scale": scale,
                "overlap_rmse_m": (
                    float(np.sqrt(np.mean(residuals**2))) if len(residuals) else 0.0
                ),
                "overlap_median_m": float(np.median(residuals)) if len(residuals) else 0.0,
                "rotation": rotation.tolist(),
                "translation": translation.tolist(),
            }
        )

    ordered_names = sorted(observations, key=image_id_by_name.__getitem__)
    image_ids = np.asarray([image_id_by_name[name] for name in ordered_names])
    stitched = np.asarray(
        [
            np.average(
                np.asarray([point for point, _, _ in observations[name]]),
                axis=0,
                weights=np.asarray([weight for _, weight, _ in observations[name]]),
            )
            for name in ordered_names
        ]
    )
    report: dict[str, object] = {
        "method": args.alignment,
        "scale_strength": args.scale_strength,
        "frames": len(ordered_names),
        "windows": transforms,
        "benchmark_uses_ground_truth_only_after_stitching": True,
    }

    benchmark_columns: list[np.ndarray] = []
    if args.ground_truth_csv:
        sample_ids, ground_truth_samples, gps_samples = load_reference(
            args.ground_truth_csv
        )
        ground_truth = interpolate_reference(image_ids, sample_ids, ground_truth_samples)
        gps = interpolate_reference(image_ids, sample_ids, gps_samples)

        rigid_rotation, rigid_translation = weighted_rigid(
            stitched, ground_truth, np.ones(len(stitched))
        )
        rigid_aligned = stitched @ rigid_rotation.T + rigid_translation
        sim_scale, sim_rotation, sim_translation = umeyama(stitched, ground_truth)
        sim_aligned = apply_similarity(
            stitched, sim_scale, sim_rotation, sim_translation
        )
        gps_rotation, gps_translation = weighted_rigid(
            stitched, gps, np.ones(len(stitched))
        )
        gps_aligned = stitched @ gps_rotation.T + gps_translation
        report["evaluation"] = {
            "se3_metric_error": error_summary(rigid_aligned, ground_truth),
            "sim3_shape_error": error_summary(sim_aligned, ground_truth),
            "global_scale_correction_needed": sim_scale,
            "segment_length_error": segment_scale_error(rigid_aligned, ground_truth),
            "gps_anchored_absolute_error": error_summary(gps_aligned, ground_truth),
            "raw_gps_absolute_error": error_summary(gps, ground_truth),
        }
        benchmark_columns = [ground_truth, gps, rigid_aligned, sim_aligned]

    with (args.output_dir / "camera_trajectory_stitched.csv").open(
        "w", newline=""
    ) as handle:
        writer = csv.writer(handle)
        header = ["image", "image_id", "x", "y", "z", "window_observations"]
        if benchmark_columns:
            header += [
                "gt_x",
                "gt_y",
                "gt_z",
                "gps_x",
                "gps_y",
                "gps_z",
                "se3_x",
                "se3_y",
                "se3_z",
                "sim3_x",
                "sim3_y",
                "sim3_z",
            ]
        writer.writerow(header)
        for row_index, (name, image_id, point) in enumerate(
            zip(ordered_names, image_ids, stitched)
        ):
            row: list[object] = [
                name,
                image_id,
                *point,
                len(observations[name]),
            ]
            for values in benchmark_columns:
                row.extend(values[row_index])
            writer.writerow(row)

    (args.output_dir / "stitch_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
