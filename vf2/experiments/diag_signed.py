import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, open3d as o3d
from scipy.spatial import cKDTree
W = "/workspace/voxelflight_a100_20260928"; run = W + "/vf2/runs/dev1"
z = np.load(W + "/vf2/runs/lidar_ref_2018_egm96.npz"); RP, RN = z["points"], z["normals"]
RH = RP[np.abs(RN[:, 2]) > 0.9]; t2 = cKDTree(RH[:, :2])
mesh = o3d.io.read_triangle_mesh(run + "/model/model_mesh.ply")
pc = mesh.sample_points_uniformly(300000, use_triangle_normal=True); P = np.asarray(pc.points); N = np.asarray(pc.normals)
kp = np.load(run + "/keyframe_poses.npz"); C = kp["camera_to_world"][:, :3, 3]; ct = cKDTree(C[:, :2])
h = np.abs(N[:, 2]) > 0.9; P = P[h]
dcam, jc = ct.query(P[:, :2]); below = C[jc, 2] - P[:, 2]
dd, jj = t2.query(P[:, :2], k=8, distance_upper_bound=0.5)
zz = np.vstack([RH[:, 2:3], [[np.nan]]])[np.minimum(jj, len(RH)), 0]
k = np.nanargmin(np.abs(np.nan_to_num(zz - P[:, 2:3], nan=1e9)), axis=1)
signed = zz[np.arange(len(P)), k] - P[:, 2]   # +: LiDAR above our surface
ok = np.isfinite(signed) & (np.abs(signed) < 10)
print("horizontal samples", ok.sum())
for lo, hi in [(-50, 2), (2, 5), (5, 8), (8, 12), (12, 50)]:
    m = ok & (below >= lo) & (below < hi)
    if m.sum() > 100:
        print(f"surface {lo:>3}..{hi:<3} m below camera: n={m.sum():6d}  signed(LiDAR - ours) median {np.median(signed[m]):+.2f}  |err| median {np.median(np.abs(signed[m])):.2f}")
lg = cKDTree(RP[:, :2]); camg = np.array([np.percentile(RP[lg.query_ball_point(c[:2], 6.0), 2], 5) for c in C[::10]])
print("camera height above LiDAR ground median %.2f" % np.median(C[::10, 2] - camg))
