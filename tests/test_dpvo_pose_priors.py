import json
from pathlib import Path
import sys

import numpy as np
import pytest

from mac_reconstruct import load_visual_pose_priors
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_dpvo_pose_priors import build_priors


def test_dpvo_video_frame_mapping_and_prior_loading(tmp_path):
    trajectory = tmp_path / "visual_only.tum"
    rows = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 1],
         [1, 1, 0, 0, 0, 0, 0, 1],
         [2, 2, 0, 0, 0, 0, 0, 1],
         [3, 3, 0, 0, 0, 0, 0, 1]],
        dtype=float,
    )
    np.savetxt(trajectory, rows)
    report = build_priors(trajectory, source_frames=8, stride=2, max_frames=4)
    assert [pose["source_frame"] for pose in report["poses"]] == [0, 2, 5, 7]
    assert [pose["image"] for pose in report["poses"]] == [
        f"frame_{i:05d}.jpg" for i in range(4)
    ]
    assert report["poses"][0]["camera_to_world"][0][3] == pytest.approx(0)
    assert report["poses"][2]["camera_to_world"][0][3] == pytest.approx(2)

    prior_path = tmp_path / "priors.json"
    prior_path.write_text(json.dumps(report))
    frames = [{"path": str(tmp_path / pose["image"]), "source_name": pose["image"]} for pose in report["poses"]]
    info = load_visual_pose_priors(prior_path, frames)
    assert info["ground_truth_used"] is False
    assert info["selected_cameras"] == 4
    assert all("input_camera_pose" in frame for frame in frames)


def test_reference_flag_is_required(tmp_path):
    prior_path = tmp_path / "priors.json"
    prior_path.write_text(json.dumps({
        "ground_truth_used": True,
        "metric_scale_established": False,
        "poses": [],
    }))
    with pytest.raises(ValueError, match="non-ground-truth"):
        load_visual_pose_priors(prior_path, [{"path": "frame_00000.jpg"}])
