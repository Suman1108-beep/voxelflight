"""Tiled inference input from a video: decode the given keyframes, undistort like the depth model's inputs, and cut each
into a 2x2 grid of overlapping crops. Writes f_<frame>_c<k>.jpg and crops.json (offsets for the dense step)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2
from features import decode_keyframes
video, keyframes_json, calib, out = sys.argv[1:5]
frac = float(sys.argv[5]) if len(sys.argv) > 5 else 0.6
os.makedirs(out, exist_ok=True)
kf = json.load(open(keyframes_json)); kf = [f["source_frame"] for f in kf["frames"]] if isinstance(kf, dict) else kf
imgs, (W, H) = decode_keyframes(video, np.array(kf), workers=16)
cal = json.load(open(calib)); K, D = np.array(cal["intrinsic_matrix"]), np.array(cal.get("distortion_coefficients", []))
meta = {}
for sf, im in zip(kf, imgs):
    if D.size:
        im = cv2.undistort(im, K, D)
    cw, ch = int(W * frac), int(H * frac)
    for k, (ox, oy) in enumerate([(0, 0), (W - cw, 0), (0, H - ch), (W - cw, H - ch)]):
        n = f"f_{sf:05d}_c{k}.jpg"
        cv2.imwrite(os.path.join(out, n), im[oy:oy + ch, ox:ox + cw], [cv2.IMWRITE_JPEG_QUALITY, 94])
        meta[n] = {"parent": f"f_{sf:05d}.jpg", "source_frame": int(sf), "ox": ox, "oy": oy, "w": cw, "h": ch, "W": W, "H": H}
json.dump(meta, open(os.path.join(out, "crops.json"), "w"))
print("crops", len(meta), "from", len(kf), "keyframes")
