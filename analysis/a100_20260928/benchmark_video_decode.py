"""Compare exact keyframes from random H.264 seeking and one-pass decoding."""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

ROOT = Path("/workspace/voxelflight_a100_20260928")
sys.path.insert(0, str(ROOT / "repo"))
from mac_reconstruct import decode_selected_video_frames


def digest(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=180)
    args = parser.parse_args()
    first = cv2.VideoCapture(str(args.video))
    count = int(first.get(cv2.CAP_PROP_FRAME_COUNT))
    indices = np.unique(np.linspace(0, count - 1, min(args.frames, count)).round().astype(int))
    start = time.monotonic()
    seeking = []
    try:
        for number in indices:
            first.set(cv2.CAP_PROP_POS_FRAMES, int(number))
            ok, frame = first.read()
            if not ok:
                raise ValueError(f"Random seek failed at {number}")
            seeking.append(digest(frame))
    finally:
        first.release()
    seek_seconds = time.monotonic() - start
    second = cv2.VideoCapture(str(args.video))
    start = time.monotonic()
    try:
        sequential = [digest(frame) for _, frame in decode_selected_video_frames(second, indices)]
    finally:
        second.release()
    one_pass_seconds = time.monotonic() - start
    result = dict(
        video=str(args.video), source_frames=count, selected_frames=len(indices),
        random_seek_s=seek_seconds, one_pass_s=one_pass_seconds,
        matching_exact_frames=sum(a == b for a, b in zip(seeking, sequential)),
        all_frames_match=seeking == sequential,
    )
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
