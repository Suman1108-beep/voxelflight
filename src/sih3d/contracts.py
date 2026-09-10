"""Stable input, output, and scoring contracts for the SIH evaluator."""

from __future__ import annotations

SIH_REQUIRED_OUTPUT_FORMATS = (
    "obj",
    "ply",
    "las",
    "geotiff",
    "glb",
    "gltf",
    "fbx",
)

SIH_SCORE_WEIGHTS = {
    "reconstruction_accuracy": 30,
    "model_completeness": 20,
    "processing_speed": 20,
    "innovation": 15,
    "scalability": 10,
    "user_interface": 5,
}

SIH_LIMITS = {
    "spatial_accuracy_m": 1.0,
    "video_duration_minutes": 10.0,
    "processing_time_minutes": 15.0,
}

# Engineering targets deliberately leave margin for a hidden evaluation set.
WINNING_TARGETS = {
    "spatial_accuracy_m": 0.5,
    "runtime_to_video_ratio": 1.0,
    "visible_scene_completeness": 0.90,
}

