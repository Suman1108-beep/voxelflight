"""Camera height above reconstructed ground (raw predictions, unscaled and scaled) vs above LiDAR ground."""
import os, sys, glob, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
W = "/workspace/voxelflight_a100_20260928"
pd = W + "/runs/zurich-10min-360w48o8-v1/inference"
files = sorted(glob.glob(pd + "/predictions/frame_*.npz"))
kp = np.load(W + "/vf2/runs/dev1/keyframe_poses.npz"); new = kp["camera_to_world"]
z = np.load(W + "/vf2/runs/lidar_ref_2018_egm96.npz"); RP = z["points"]; t2 = cKDTree(RP[:, :2])
h_pred, h_snap, h_lid, h_pts = [], [], [], []
for i in range(0, 360, 6):
    p = np.load(files[i]); d = np.where(p["mask"], p["depth"], 0); K = p["intrinsics"]; H, Wd = d.shape
    ys, xs = np.mgrid[0:H, 0:Wd]
    ok = d > 0
    cam = np.stack([(xs - K[0, 2]) / K[0, 0] * d, (ys - K[1, 2]) / K[1, 1] * d, d], -1)[ok]
    for name, c2w, out in (("pred", p["pose"].astype(float), h_pred), ("snap", new[i], h_snap)):
        sc = 1.0 if name == "pred" else None
        X = cam @ c2w[:3, :3].T + c2w[:3, 3] if name == "pred" else None
        if name == "snap":
            # depth scaled by the snap scale = ratio of camera-centre spread; recover from pose change
            pass
        if X is not None:
            up = X[:, 2]; camz = c2w[2, 3]
            out.append(camz - np.percentile(up, 3))
    lid_g = np.percentile(RP[t2.query_ball_point(new[i][:2, 3], 6.0), 2], 5) if len(t2.query_ball_point(new[i][:2, 3], 6.0)) else np.nan
    h_lid.append(new[i][2, 3] - lid_g)
print("camera height above own ground (MapAnything native units/frame): median %.2f" % np.nanmedian(h_pred))
print("camera height above LiDAR ground (snapped/GNSS-baro frame):      median %.2f" % np.nanmedian(h_lid))
rep = json.load(open(pd + "/report.json"))
print("mapanything coordinate_system:", rep.get("coordinate_system"), "| georeferenced:", rep.get("georeferenced"), "| alignment:", str(rep.get("alignment"))[:300])
print("pred pose z-axis is world up?  median |R[2,:]| of camera y axis:", np.median([abs(np.load(files[i])["pose"][2, 1]) for i in range(0, 360, 30)]))
