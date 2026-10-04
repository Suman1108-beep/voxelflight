import numpy as np
import pytest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from georeference_refused_mesh import recover_transform


def test_recover_frozen_gps_transform_from_paired_camera_poses():
    pre = np.repeat(np.eye(4)[None], 4, axis=0)
    pre[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
    rotation = np.asarray([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    translation = np.asarray([10., 20., -5.])
    post = pre.copy()
    post[:, :3, :3] = rotation @ pre[:, :3, :3]
    post[:, :3, 3] = 1.4 * (pre[:, :3, 3] @ rotation.T) + translation
    scale, recovered_rotation, recovered_translation, residual = recover_transform(pre, post)
    assert scale == pytest.approx(1.4)
    np.testing.assert_allclose(recovered_rotation, rotation)
    np.testing.assert_allclose(recovered_translation, translation)
    assert residual < 1e-10


def test_recovery_rejects_inconsistent_camera_pair():
    pre = np.repeat(np.eye(4)[None], 4, axis=0)
    pre[:, :3, 3] = [[0, 0, 0], [1, 0, 0], [0, 2, 0], [0, 0, 3]]
    post = pre.copy()
    post[-1, :3, 3] += [0., 0., 1.]
    with pytest.raises(ValueError, match="inconsistent"):
        recover_transform(pre, post)
