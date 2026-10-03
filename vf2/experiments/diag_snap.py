import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.transform import Rotation
from trajfuse import interpolate_poses
from snap import snap_views
W = "/workspace/voxelflight_a100_20260928"
pd = W + "/runs/zurich-10min-360w48o8-v1/inference"
rep = json.load(open(pd + "/report.json")); kf = np.array([f["source_frame"] for f in rep["frames"]])
P = np.stack([np.load(f)["pose"] for f in sorted(glob.glob(pd + "/predictions/frame_*.npz"))]).astype(float)
z = np.load(W + "/vf2/runs/dev1/fused_trajectory.npz"); vf, X, R = z["video_frame"], z["position"], z["rotation"]
Cq, Rq = interpolate_poses(vf, X, R, kf)
rel = Rotation.from_matrix(Rq @ np.transpose(P[:, :3, :3], (0, 2, 1)))
ang = [np.degrees((rel[i].inv() * rel[i + 1]).magnitude()) for i in range(len(kf) - 1)]
print("consecutive change of R_fused R_pred^T (deg): median %.2f p95 %.2f max %.1f" % (np.median(ang), np.percentile(ang, 95), np.max(ang)))
# camera forward axes: angle between fused and pred viewing direction after per-window rotation
new, s, res = snap_views(P, Cq, Rq)
fwd_f = Rq[:, :, 2]; fwd_n = new[:, :3, 2]
print("view dir disagreement after snap (deg) median %.2f p95 %.2f" % tuple(np.percentile(np.degrees(np.arccos(np.clip((fwd_f * fwd_n).sum(1), -1, 1))), [50, 95])))
print("scales median %.3f p5 %.3f p95 %.3f ; centre residual median %.2f p95 %.2f" % (np.median(s), *np.percentile(s, [5, 95]), *np.percentile(res, [50, 95])))
# does DPVO camera z point forward like MapAnything? compare per-view motion direction with view direction
mv = np.gradient(Cq, axis=0); mv /= np.linalg.norm(mv, axis=1, keepdims=True) + 1e-9
mp = np.gradient(P[:, :3, 3], axis=0); mp /= np.linalg.norm(mp, axis=1, keepdims=True) + 1e-9
print("cos(motion, +z view): fused %.2f  pred %.2f" % (np.median((mv * Rq[:, :, 2]).sum(1)), np.median((mp * P[:, :3, 2]).sum(1))))
print("pred step median m", np.median(np.linalg.norm(np.diff(P[:, :3, 3], axis=0), axis=1)), " fused step", np.median(np.linalg.norm(np.diff(Cq, axis=0), axis=1)))
