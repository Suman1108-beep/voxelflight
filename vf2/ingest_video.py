"""Make any drone video usable by the pipeline: transcode codecs OpenCV cannot read (e.g. DJI HEVC/H.265) to H.264 at full
resolution, and estimate camera intrinsics with Depth Anything 3 when no calibration file is supplied. Both run inside the
timed pipeline and write next to the outputs (video_h264.mp4, calibration_estimated.json)."""
import json, os, shutil, subprocess, sys
import cv2, numpy as np

ROOT = "/workspace/voxelflight_a100_20260928"


def _ffmpeg():
    exe = shutil.which("ffmpeg")
    if exe: return exe
    sys.path.insert(0, ROOT + "/pydeps-da3")
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def readable(video):
    cap = cv2.VideoCapture(video); ok = cap.isOpened() and cap.read()[0]; cap.release(); return ok


def ensure_decodable(video, out_dir):
    if readable(video): return video
    out = os.path.join(out_dir, "video_h264.mp4")
    for enc in (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"], ["-c:v", "libx264", "-preset", "ultrafast", "-crf", "16"]):
        r = subprocess.run([_ffmpeg(), "-y", "-v", "error", "-i", video, "-map", "0:v:0", *enc, "-pix_fmt", "yuv420p", "-an", out],
                           capture_output=True, text=True)
        if r.returncode == 0 and readable(out):
            print("transcoded", os.path.basename(video), "->", out, "with", enc[1], flush=True); return out
    raise SystemExit(f"cannot decode {video}: {r.stderr[-300:]}")


def ensure_calibration(calibration, video, out_dir, frames=16):
    """Pinhole intrinsics from Depth Anything 3 (median over frames spread across the video) when none are given."""
    if calibration: return calibration
    out = os.path.join(out_dir, "calibration_estimated.json")
    code = f"""
import sys, json, numpy as np, cv2, torch
from depth_anything_3.api import DepthAnything3
cap = cv2.VideoCapture({video!r}); n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); W, H = int(cap.get(3)), int(cap.get(4)); ims = []
for f in np.linspace(n * 0.1, n * 0.9, {frames}).astype(int):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(f)); ok, im = cap.read()
    if ok: ims.append(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
m = DepthAnything3.from_pretrained("depth-anything/DA3-GIANT-1.1").to("cuda").eval()
with torch.inference_mode(): p = m.inference(ims, process_res=504)
h, w = p.depth.shape[1:]; fx = float(np.median(p.intrinsics[:, 0, 0])) * W / w; fy = float(np.median(p.intrinsics[:, 1, 1])) * H / h
f = (fx + fy) / 2
json.dump({{"intrinsic_matrix": [[f, 0, W / 2], [0, f, H / 2], [0, 0, 1]], "distortion_coefficients": [], "width": W, "height": H,
           "source": "estimated by Depth Anything 3 from {frames} frames", "fx_fy_estimates": [fx, fy]}}, open({out!r}, "w"), indent=1)
print("estimated focal", round(f, 1), "px for", W, "x", H)
"""
    env = dict(os.environ, PYTHONPATH=f"{ROOT}/thirdparty/Depth-Anything-3/src:{ROOT}/pydeps-da3", HF_HOME=ROOT + "/cache/huggingface")
    env.pop("HF_HUB_OFFLINE", None)
    r = subprocess.run([ROOT + "/.venv/bin/python", "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0: raise SystemExit("intrinsics estimation failed: " + r.stderr[-500:])
    print(r.stdout.strip().splitlines()[-1], flush=True)
    return out
