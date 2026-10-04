import hashlib
import json

import numpy as np
import pytest

from analysis.a100_20260928.evaluate_zurich_video_run import evaluate


def make_frozen_run(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    positions = np.asarray(
        [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]
    )
    poses = np.repeat(np.eye(4)[None], len(positions), axis=0)
    poses[:, :3, 3] = positions
    pose_path = run / "camera_poses.npz"
    np.savez(pose_path, camera_to_world=poses)
    digest = hashlib.sha256(pose_path.read_bytes()).hexdigest()
    report = {
        "ground_truth_used": False,
        "georeferenced": True,
        "utm_origin": [5., 6., 7.],
        "input": {"duration_s": 2.0},
        "frames": [{"source_frame": i} for i in range(4)],
        "checksums": {"camera_poses.npz": digest},
    }
    (run / "report.json").write_text(json.dumps(report))
    reference = tmp_path / "reference.csv"
    header = "imgid,x_gt,y_gt,z_gt,x_gps,y_gps,z_gps\n"
    rows = "".join(
        f"{20001 + i},{x+5},{y+6},{z+7},0,0,0\n"
        for i, (x, y, z) in enumerate(positions)
    )
    reference.write_text(header + rows)
    return run, reference


def test_video_frame_mapping_and_absolute_score(tmp_path):
    run, reference = make_frozen_run(tmp_path)
    result = evaluate(run, reference, 20001)
    assert result["source_image_ids"] == [20001, 20004]
    assert result["direct_absolute_camera_error_m"]["rmse_m"] == pytest.approx(0)
    assert result["sim3_aligned_camera_error_m"]["rmse_m"] == pytest.approx(0)
    assert result["surface_rmse_m"] is None


def test_evaluation_rejects_modified_artifact(tmp_path):
    run, reference = make_frozen_run(tmp_path)
    with (run / "camera_poses.npz").open("ab") as stream:
        stream.write(b"modified")
    with pytest.raises(ValueError, match="checksum mismatch"):
        evaluate(run, reference, 20001)
