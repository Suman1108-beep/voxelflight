#!/usr/bin/env python3
"""Run VGGT-SLAM 2.0 headlessly and export poses plus a dense preview cloud."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import trimesh

import vggt_slam.solver as solver_module
from vggt.models.vggt import VGGT


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


class NullViewer:
    """Avoid starting an HTTP server during timed SIH reconstruction."""

    def __init__(self, *_: object, **__: object) -> None:
        pass


class CompatibleVGGT(VGGT):
    """Remove an upstream unconditional BF16 cast that breaks current PyTorch."""

    def forward(
        self,
        images: torch.Tensor,
        query_points: torch.Tensor | None = None,
        compute_similarity: bool = False,
    ) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        if images.ndim == 4:
            images = images.unsqueeze(0)
        if query_points is not None and query_points.ndim == 2:
            query_points = query_points.unsqueeze(0)
        tokens, patch_start, target_tokens, match_ratio = self.aggregator(
            images, compute_similarity
        )
        predictions: dict[str, torch.Tensor | list[torch.Tensor]] = {}
        if self.camera_head is not None:
            pose_encodings = self.camera_head(tokens)
            predictions["pose_enc"] = pose_encodings[-1]
            predictions["pose_enc_list"] = pose_encodings
        if self.depth_head is not None:
            depth, confidence = self.depth_head(
                tokens, images=images, patch_start_idx=patch_start
            )
            predictions["depth"] = depth
            predictions["depth_conf"] = confidence
        if not self.training:
            predictions["images"] = images
        predictions["target_tokens"] = target_tokens
        predictions["image_match_ratio"] = match_ratio
        return predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--submap-size", type=int, default=16)
    parser.add_argument("--overlap", type=int, default=1)
    parser.add_argument("--max-loops", type=int, default=1)
    parser.add_argument("--confidence-percentile", type=float, default=25.0)
    parser.add_argument("--loop-threshold", type=float, default=0.95)
    parser.add_argument("--max-points", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=17)
    return parser.parse_args()


def numeric_key(path: Path) -> tuple[float, str]:
    match = re.search(r"\d+(?:\.\d+)?", path.name)
    return (float(match.group()) if match else float("inf"), path.name)


def load_model(device: torch.device) -> VGGT:
    checkpoint = Path(torch.hub.get_dir()) / "checkpoints" / "model.pt"
    if not checkpoint.is_file():
        raise FileNotFoundError(
            f"Offline VGGT checkpoint missing: {checkpoint}. Run preflight before evaluation."
        )
    model = CompatibleVGGT()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.eval().to(device=device)


def deduplicated_poses(solver: solver_module.Solver) -> tuple[np.ndarray, np.ndarray]:
    observations: dict[float, list[np.ndarray]] = defaultdict(list)
    for submap in solver.map.ordered_submaps_by_key():
        if submap.get_lc_status():
            continue
        poses = submap.get_all_poses_world(solver.graph, give_camera_mat=False)
        for frame_id, pose in zip(submap.get_frame_ids(), poses):
            observations[float(frame_id)].append(pose)

    frame_ids, poses = [], []
    for frame_id in sorted(observations):
        candidates = observations[frame_id]
        # Overlap poses should already agree after graph optimization. Retain the
        # newest observation and use the median center to reject a bad seam.
        pose = candidates[-1].copy()
        pose[:3, 3] = np.median(
            np.asarray([candidate[:3, 3] for candidate in candidates]), axis=0
        )
        frame_ids.append(frame_id)
        poses.append(pose)
    return np.asarray(frame_ids), np.asarray(poses)


def export_preview_cloud(
    solver: solver_module.Solver, output: Path, maximum: int, seed: int
) -> int:
    rng = np.random.default_rng(seed)
    submaps = [
        submap for submap in solver.map.ordered_submaps_by_key() if not submap.get_lc_status()
    ]
    per_submap = max(1, maximum // max(1, len(submaps)))
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    for submap in submaps:
        points = submap.get_points_in_world_frame(solver.graph)
        colors = submap.get_points_colors()
        valid = np.isfinite(points).all(axis=1)
        points, colors = points[valid], colors[valid]
        if len(points) > per_submap:
            chosen = rng.choice(len(points), per_submap, replace=False)
            points, colors = points[chosen], colors[chosen]
        point_sets.append(points.astype(np.float32, copy=False))
        color_sets.append(colors.astype(np.uint8, copy=False))
    points = np.concatenate(point_sets) if point_sets else np.empty((0, 3), np.float32)
    colors = np.concatenate(color_sets) if color_sets else np.empty((0, 3), np.uint8)
    if len(points) > maximum:
        chosen = rng.choice(len(points), maximum, replace=False)
        points, colors = points[chosen], colors[chosen]
    trimesh.PointCloud(points, colors=colors).export(output)
    return len(points)


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.overlap != 1:
        raise ValueError("VGGT-SLAM 2.0 currently supports overlap=1")
    images = sorted(
        (
            path
            for path in args.image_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ),
        key=numeric_key,
    )
    if len(images) < 2:
        raise RuntimeError("At least two numbered images are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    solver_module.Viewer = NullViewer
    started = time.perf_counter()
    solver = solver_module.Solver(
        init_conf_threshold=args.confidence_percentile,
        lc_thres=args.loop_threshold,
    )
    model_started = time.perf_counter()
    model = load_model(device)
    model_load_seconds = time.perf_counter() - model_started

    pending: list[str] = []
    processed_windows = 0
    inference_started = time.perf_counter()
    for index, image in enumerate(images):
        pending.append(str(image))
        ready = len(pending) == args.submap_size + args.overlap
        if ready or index == len(images) - 1:
            if len(pending) >= 2:
                predictions = solver.run_predictions(
                    pending, model, args.max_loops, None, None
                )
                solver.add_points(predictions)
                solver.graph.optimize()
                processed_windows += 1
                pending = pending[-args.overlap :]
    inference_seconds = time.perf_counter() - inference_started

    frame_ids, poses = deduplicated_poses(solver)
    np.savez_compressed(
        args.output_dir / "trajectory.npz", frame_ids=frame_ids, poses=poses
    )
    with (args.output_dir / "trajectory.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "frame_id",
                "camera_x",
                "camera_y",
                "camera_z",
                *[f"pose_{row}{column}" for row in range(4) for column in range(4)],
            ]
        )
        for frame_id, pose in zip(frame_ids, poses):
            writer.writerow([frame_id, *pose[:3, 3], *pose.reshape(-1)])

    export_started = time.perf_counter()
    exported_points = export_preview_cloud(
        solver,
        args.output_dir / "points_slam.ply",
        args.max_points,
        args.seed,
    )
    export_seconds = time.perf_counter() - export_started
    total_seconds = time.perf_counter() - started
    report = {
        "method": "VGGT-SLAM 2.0",
        "input_frames": len(images),
        "unique_output_poses": len(frame_ids),
        "submaps": solver.map.get_num_submaps(),
        "loop_closures": solver.graph.get_num_loops(),
        "model_load_seconds": round(model_load_seconds, 3),
        "inference_and_optimization_seconds": round(inference_seconds, 3),
        "cloud_export_seconds": round(export_seconds, 3),
        "total_seconds": round(total_seconds, 3),
        "peak_vram_gib": round(torch.cuda.max_memory_allocated() / 1024**3, 3),
        "exported_points": exported_points,
        "submap_size": args.submap_size,
        "loop_threshold": args.loop_threshold,
    }
    (args.output_dir / "slam_report.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
