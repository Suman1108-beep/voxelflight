#!/usr/bin/env python3
"""Run a ground-truth-free MapAnything window and export metric geometry."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from mapanything.models import MapAnything
from mapanything.utils.image import load_images
from PIL import Image
from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif"}


def write_binary_ply(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    vertices = np.empty(
        len(points),
        dtype=[
            ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
            ("red", "u1"), ("green", "u1"), ("blue", "u1"),
        ],
    )
    vertices["x"], vertices["y"], vertices["z"] = points.T
    vertices["red"], vertices["green"], vertices["blue"] = colors.T
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        vertices.tofile(stream)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--keyframe-manifest",
        type=Path,
        required=True,
        help="Global keyframe manifest containing output_frame and source_frame.",
    )
    parser.add_argument("--source-image-offset", type=int, required=True)
    parser.add_argument("--model", default="facebook/map-anything-apache")
    parser.add_argument("--max-points", type=int, default=1_500_000)
    parser.add_argument("--confidence-percentile", type=float, default=20.0)
    parser.add_argument(
        "--semantic-filter-model",
        default="nvidia/segformer-b5-finetuned-ade-640-640",
    )
    parser.add_argument(
        "--no-semantic-filter",
        action="store_true",
        help="Keep sky and potentially dynamic object points.",
    )
    parser.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.max_points <= 0:
        raise ValueError("--max-points must be positive")
    start_time = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(
        path for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )
    if not paths:
        raise FileNotFoundError(f"No images found in {args.image_dir}")
    views = load_images(str(args.image_dir))
    if len(paths) != len(views):
        raise RuntimeError("Image enumeration does not match MapAnything views")

    model = MapAnything.from_pretrained(args.model).to("cuda").eval()
    with torch.inference_mode():
        outputs = model.infer(
            views,
            memory_efficient_inference=True,
            minibatch_size=1,
            use_amp=True,
            amp_dtype="bf16",
            apply_mask=True,
            mask_edges=True,
            apply_confidence_mask=True,
            confidence_percentile=args.confidence_percentile,
            use_multiview_confidence=True,
        )

    semantic_keep_masks: list[np.ndarray | None] = [None] * len(outputs)
    blocked_labels: list[str] = []
    semantic_removed = 0
    if not args.no_semantic_filter:
        processor = SegformerImageProcessor.from_pretrained(
            args.semantic_filter_model, local_files_only=True
        )
        segmenter = SegformerForSemanticSegmentation.from_pretrained(
            args.semantic_filter_model, local_files_only=True
        ).to("cuda").eval()
        blocked_names = {
            "person", "car", "truck", "bus", "van", "bicycle",
            "motorbike", "motorcycle", "airplane", "boat", "ship", "sky",
        }
        blocked_ids = {
            int(index)
            for index, label in segmenter.config.id2label.items()
            if any(
                name.strip() in blocked_names
                for name in re.split(r"[,/]", label.lower())
            )
        }
        blocked_labels = sorted(
            segmenter.config.id2label[index] for index in blocked_ids
        )
        with torch.inference_mode():
            for index, item in enumerate(outputs):
                rgb = item["img_no_norm"][0].float().cpu().numpy()
                if float(np.nanmax(rgb)) <= 1.0:
                    rgb = rgb * 255.0
                image = Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8))
                inputs = processor(images=image, return_tensors="pt")
                inputs = {name: value.to("cuda") for name, value in inputs.items()}
                logits = segmenter(**inputs).logits
                shape = tuple(
                    int(value) for value in item["mask"][0].shape[:2]
                )
                labels = torch.nn.functional.interpolate(
                    logits, size=shape, mode="bilinear", align_corners=False
                ).argmax(dim=1)[0].cpu().numpy()
                keep = ~np.isin(labels, list(blocked_ids))
                semantic_keep_masks[index] = keep
                semantic_removed += int(np.count_nonzero(~keep))
        del segmenter
        torch.cuda.empty_cache()

    camera_poses = np.asarray(
        [item["camera_poses"][0].float().cpu().numpy() for item in outputs]
    )
    intrinsics = np.asarray(
        [item["intrinsics"][0].float().cpu().numpy() for item in outputs]
    )
    image_sizes = np.asarray(
        [item["img_no_norm"][0].shape[:2] for item in outputs], dtype=np.int32
    )
    metric_scales = np.asarray(
        [float(item["metric_scaling_factor"][0].cpu()) for item in outputs]
    )

    point_parts: list[np.ndarray] = []
    color_parts: list[np.ndarray] = []
    for item_index, item in enumerate(outputs):
        points = item["pts3d"][0].float().cpu().numpy().reshape(-1, 3)
        colors = item["img_no_norm"][0].float().cpu().numpy().reshape(-1, 3)
        mask = item["mask"][0].cpu().numpy().reshape(-1).astype(bool)
        semantic_keep = semantic_keep_masks[item_index]
        if semantic_keep is not None:
            mask &= semantic_keep.reshape(-1)
        valid = mask & np.isfinite(points).all(axis=1)
        point_parts.append(points[valid])
        color_parts.append(colors[valid])
    points = np.concatenate(point_parts).astype(np.float32)
    colors = np.concatenate(color_parts)
    source_point_count = len(points)
    if source_point_count > args.max_points:
        rng = np.random.default_rng(args.seed)
        selected = rng.choice(source_point_count, args.max_points, replace=False)
        points, colors = points[selected], colors[selected]
    if colors.size and float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    colors = np.clip(colors, 0, 255).astype(np.uint8)
    write_binary_ply(args.output_dir / "points_dense.ply", points, colors)

    names = np.asarray([path.name for path in paths])
    manifest = {
        row["output_frame"]: int(row["source_frame"]) + args.source_image_offset
        for row in csv.DictReader(args.keyframe_manifest.open(newline=""))
    }
    missing_names = [name for name in names if name not in manifest]
    if missing_names:
        raise KeyError(
            f"Images are missing from {args.keyframe_manifest}: {missing_names[:10]}"
        )
    image_ids = np.asarray([manifest[name] for name in names])

    np.savez_compressed(
        args.output_dir / "raw_prediction.npz",
        image_names=names,
        image_ids=image_ids,
        camera_centers=camera_poses[:, :3, 3],
        camera_poses=camera_poses,
        intrinsics=intrinsics,
        image_sizes=image_sizes,
        metric_scaling_factors=metric_scales,
    )
    report = {
        "method": "MapAnything metric dense reconstruction",
        "construction_uses_ground_truth": False,
        "keyframe_manifest": str(args.keyframe_manifest),
        "frames": len(paths),
        "source_points": source_point_count,
        "exported_points": len(points),
        "semantic_filter": {
            "enabled": not args.no_semantic_filter,
            "model": args.semantic_filter_model,
            "blocked_labels": blocked_labels,
            "removed_pixels": semantic_removed,
        },
        "runtime_seconds": time.perf_counter() - start_time,
    }
    (args.output_dir / "construction_protocol.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
