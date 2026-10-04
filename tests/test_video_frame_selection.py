import numpy as np
import pytest

from mac_reconstruct import decode_selected_video_frames


class FakeCapture:
    def __init__(self, frame_count):
        self.frame_count = frame_count
        self.position = -1
        self.grabs = 0
        self.retrievals = 0

    def grab(self):
        if self.position + 1 >= self.frame_count:
            return False
        self.position += 1
        self.grabs += 1
        return True

    def retrieve(self):
        self.retrievals += 1
        return True, np.full((2, 2, 3), self.position, np.uint8)


def test_video_is_decoded_once_and_only_selected_frames_are_retrieved():
    capture = FakeCapture(10)
    selected = list(decode_selected_video_frames(capture, [0, 3, 9]))
    assert [int(frame[0, 0, 0]) for _, frame in selected] == [0, 3, 9]
    assert capture.grabs == 10
    assert capture.retrievals == 3


def test_video_selection_rejects_duplicate_or_past_indices():
    with pytest.raises(ValueError, match="strictly increasing"):
        list(decode_selected_video_frames(FakeCapture(10), [2, 2]))


def test_video_selection_reports_truncated_input():
    with pytest.raises(ValueError, match="ended before"):
        list(decode_selected_video_frames(FakeCapture(4), [0, 5]))
