import json
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_a100_pipeline import dpvo_calibration_line, run_stage, video_frame_count


def test_dpvo_calibration_and_real_video_probe(tmp_path):
    camera = tmp_path / "camera.json"
    camera.write_text(json.dumps({
        "model": "OPENCV", "width": 64, "height": 48,
        "intrinsic_matrix": [[50, 0, 32], [0, 51, 24], [0, 0, 1]],
        "distortion_coefficients": [-0.1, 0.02, 0, 0, 0],
    }))
    assert np.fromstring(dpvo_calibration_line(camera), sep=" ").tolist() == pytest.approx([
        50, 51, 32, 24, -0.1, 0.02, 0, 0, 0
    ])
    video = tmp_path / "flight.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable")
    for i in range(4):
        writer.write(np.full((48, 64, 3), i * 20, dtype=np.uint8))
    writer.release()
    assert video_frame_count(video) == 4


def test_stage_error_preserves_log(tmp_path):
    with pytest.raises(RuntimeError, match="failed"):
        run_stage(
            "failed_stage", [sys.executable, "-c", "raise RuntimeError('expected')"],
            cwd=tmp_path, env={}, root=tmp_path,
        )
    assert "expected" in (tmp_path / "failed_stage.log").read_text()
