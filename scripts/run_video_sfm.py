#!/usr/bin/env python3
"""Run a resumable, shared-intrinsics COLMAP baseline on ordered video frames."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import pycolmap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--camera-params",
        default="893.39010814,898.32648616,951.1310043,555.13350077,"
        "-0.28052513,0.115806413,-0.000984336785,0.000158479248",
        help="COLMAP OPENCV parameters: fx,fy,cx,cy,k1,k2,p1,p2",
    )
    parser.add_argument("--overlap", type=int, default=12)
    parser.add_argument("--max-features", type=int, default=8192)
    parser.add_argument("--max-image-size", type=int, default=1920)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    database = args.output_dir / "database.db"
    sparse = args.output_dir / "sparse"
    sparse.mkdir(exist_ok=True)
    device = pycolmap.Device.cpu if args.device == "cpu" else pycolmap.Device.cuda
    timings: dict[str, float] = {}

    if not database.exists():
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "OPENCV"
        reader.camera_params = args.camera_params
        sift = pycolmap.SiftExtractionOptions()
        sift.num_threads = args.threads
        sift.max_image_size = args.max_image_size
        sift.max_num_features = args.max_features
        started = time.perf_counter()
        pycolmap.extract_features(
            database_path=str(database),
            image_path=str(args.image_dir),
            camera_mode=pycolmap.CameraMode.SINGLE,
            camera_model="OPENCV",
            reader_options=reader,
            sift_options=sift,
            device=device,
        )
        timings["feature_extraction_s"] = time.perf_counter() - started

    matches_marker = args.output_dir / ".matching_complete"
    if not matches_marker.exists():
        if hasattr(pycolmap, "SequentialMatchingOptions"):
            matching = pycolmap.SequentialMatchingOptions()
            sift_matching = pycolmap.SiftMatchingOptions()
            sift_matching.num_threads = args.threads
            sift_matching.guided_matching = True
            matching_kwargs = {
                "sift_options": sift_matching,
                "matching_options": matching,
            }
        else:
            matching = pycolmap.SequentialPairingOptions()
            feature_matching = pycolmap.FeatureMatchingOptions()
            feature_matching.num_threads = args.threads
            feature_matching.guided_matching = True
            matching_kwargs = {
                "matching_options": feature_matching,
                "pairing_options": matching,
            }
        matching.overlap = args.overlap
        matching.quadratic_overlap = True
        matching.loop_detection = False
        started = time.perf_counter()
        pycolmap.match_sequential(
            database_path=str(database),
            device=device,
            **matching_kwargs,
        )
        timings["sequential_matching_s"] = time.perf_counter() - started
        matches_marker.touch()

    existing_models = list(sparse.glob("*/images.bin"))
    if not existing_models:
        options = pycolmap.IncrementalPipelineOptions()
        options.num_threads = args.threads
        options.multiple_models = False
        options.ba_refine_focal_length = False
        options.ba_refine_principal_point = False
        options.ba_refine_extra_params = False
        options.ba_local_max_num_iterations = 15
        options.ba_local_max_refinements = 1
        options.ba_global_max_refinements = 2
        if hasattr(options, "ba_global_frames_ratio"):
            options.ba_global_frames_ratio = 1.6
            options.ba_global_frames_freq = 180
        else:
            options.ba_global_images_ratio = 1.6
            options.ba_global_images_freq = 180
        options.ba_global_points_ratio = 1.6
        options.mapper.abs_pose_refine_focal_length = False
        options.mapper.abs_pose_refine_extra_params = False
        options.mapper.init_min_num_inliers = 60
        options.mapper.init_min_tri_angle = 4.0
        if hasattr(options.mapper, "local_ba_min_tri_angle"):
            options.mapper.local_ba_min_tri_angle = 2.0
        else:
            options.mapper.ba_local_min_tri_angle = 2.0
        started = time.perf_counter()
        reconstructions = pycolmap.incremental_mapping(
            database_path=str(database),
            image_path=str(args.image_dir),
            output_path=str(sparse),
            options=options,
        )
        timings["incremental_mapping_s"] = time.perf_counter() - started
    else:
        reconstructions = {
            int(path.parent.name): pycolmap.Reconstruction(path.parent)
            for path in existing_models
        }

    models = [
        {
            "model_id": int(model_id),
            "registered_images": reconstruction.num_reg_images(),
            "points3D": reconstruction.num_points3D(),
            "mean_reprojection_error_px": (
                float(reconstruction.compute_mean_reprojection_error())
                if reconstruction.num_points3D()
                else None
            ),
        }
        for model_id, reconstruction in reconstructions.items()
    ]
    report = {
        "images": len(list(args.image_dir.glob("*.jpg"))),
        "device": args.device,
        "sequential_overlap": args.overlap,
        "timings": timings,
        "models": models,
    }
    (args.output_dir / "run_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
