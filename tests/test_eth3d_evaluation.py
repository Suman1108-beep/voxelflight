from pathlib import Path

import numpy as np

from analysis.a100_20260928.evaluate_eth3d_courtyard import reference_centres, similarity


def test_similarity_recovers_known_scale_and_rotation():
    source = np.array([[0., 0., 0.], [1., 0., 0.], [0., 2., 0.], [0., 0., 3.]])
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    target = 2.5 * source @ rotation.T + np.array([4., -3., 1.])
    scale, recovered_rotation, translation = similarity(source, target)
    np.testing.assert_allclose(scale, 2.5, atol=1e-10)
    np.testing.assert_allclose(recovered_rotation, rotation, atol=1e-10)
    np.testing.assert_allclose(translation, [4., -3., 1.], atol=1e-10)


def test_reference_centres_invert_world_to_camera(tmp_path: Path):
    data = tmp_path / 'images.txt'
    data.write_text('# comment\n1 1 0 0 0 -2 3 -4 0 folder/frame.JPG\n0 0 -1\n')
    centres = reference_centres(data)
    np.testing.assert_allclose(centres['frame.JPG'], [2., -3., 4.])
