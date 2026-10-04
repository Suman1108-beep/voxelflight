"""Depth Anything 3 multi-view depth for the quality pipeline. Writes the same per-frame files as the MapAnything stage
(predictions/frame_XXXXX.npz with depth, mask, color + report.json), so dense_multi_sfm.py can place and fuse them unchanged.
Frames are undistorted with the calibration and processed in consecutive windows; each view is later scaled to SfM tie points."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, torch

ap = argparse.ArgumentParser()
ap.add_argument("--video"); ap.add_argument("--keyframes", help="JSON list of source frame indices")
ap.add_argument("--images", help="folder of undistorted crops with crops.json (tiled mode) instead of --video/--keyframes")
ap.add_argument("--calibration"); ap.add_argument("--output", required=True)
ap.add_argument("--model", default="depth-anything/DA3-GIANT-1.1"); ap.add_argument("--res", type=int, default=504)
ap.add_argument("--window", type=int, default=32); ap.add_argument("--conf-drop", type=float, default=20.0,
                help="drop the lowest-confidence percent of pixels in each view")
a = ap.parse_args()
os.makedirs(os.path.join(a.output, "predictions"), exist_ok=True)
t0 = time.time()
from features import decode_keyframes
from depth_anything_3.api import DepthAnything3

if a.images:   # tiled mode: same crop position across consecutive frames per window, for overlap between views
    meta = json.load(open(os.path.join(a.images, "crops.json")))
    names = sorted(meta, key=lambda n: (n.rsplit("_c", 1)[1], meta[n]["source_frame"]))
    kf = [meta[n]["source_frame"] for n in names]
    imgs = np.stack([cv2.imread(os.path.join(a.images, n)) for n in names])
elif a.keyframes:
    names = None
    kf = json.load(open(a.keyframes)); kf = [f["source_frame"] for f in kf["frames"]] if isinstance(kf, dict) else kf
    imgs, (W, H) = decode_keyframes(a.video, np.array(kf), workers=16)
if a.calibration and not a.images:
    cal = json.load(open(a.calibration)); K, D = np.array(cal["intrinsic_matrix"]), np.array(cal.get("distortion_coefficients", []))
    if D.size:
        imgs = np.stack([cv2.undistort(im, K, D) for im in imgs])
t_decode = time.time() - t0
model = DepthAnything3.from_pretrained(a.model).to("cuda").eval()
t_load = time.time() - t0 - t_decode
frames, peak = [], 0.0
for s in range(0, len(kf), a.window):
    idx = list(range(s, min(s + a.window, len(kf))))
    with torch.inference_mode():
        pred = model.inference([cv2.cvtColor(imgs[i], cv2.COLOR_BGR2RGB) for i in idx], process_res=a.res)
    peak = max(peak, torch.cuda.max_memory_allocated() / 2**30)
    for j, i in enumerate(idx):
        d = np.asarray(pred.depth[j], np.float32); c = np.asarray(pred.conf[j], np.float32)
        mask = (d > 0) & np.isfinite(d) & (c >= np.percentile(c, a.conf_drop))
        np.savez(os.path.join(a.output, "predictions", f"frame_{i:05d}.npz"), depth=d, mask=mask, conf=c,
                 color=np.asarray(pred.processed_images[j], np.uint8), intrinsics=np.asarray(pred.intrinsics[j], np.float32),
                 pose=np.asarray(pred.extrinsics[j], np.float32))
        frames.append({"index": i, "source_frame": int(kf[i]), **({"source_name": names[i]} if names else {})})
    print(f"window {s // a.window + 1}/{-(-len(kf) // a.window)} done at {time.time() - t0:.0f}s", flush=True)
json.dump({"schema": "voxelflight.v2.da3", "model": a.model, "keyframes": len(kf), "frames": frames, "inference_resolution": list(d.shape[::-1]),
           "window_size": a.window, "conf_drop_percent": a.conf_drop, "peak_gpu_allocation_gib": peak, "ground_truth_used": False,
           "seconds": {"decode": t_decode, "model_load": t_load, "total": time.time() - t0}},
          open(os.path.join(a.output, "report.json"), "w"), indent=1)
print("done", round(time.time() - t0, 1), "s; peak GPU", round(peak, 1), "GiB")
