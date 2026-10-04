"""Is the residual absolute error a constant offset (reference-datum mismatch) or varying (alignment error)?
Also: does the Zurich reference itself sit consistently on the swisstopo LiDAR (cameras in streets, not inside buildings)?"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
run, seg, first_id, lidar, tag = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5]
kp = np.load(run + f"/keyframe_poses_anchored_{tag}.npz"); C = kp["camera_to_world"][:, :3, 3]; origin = kp["utm_origin"]
img = kp["video_frame"] + first_id
gt = np.genfromtxt(seg + "/Log Files/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - origin
e = (C - GT)[m]
print("anchored - reference: mean offset E %.2f N %.2f U %.2f | std E %.2f N %.2f U %.2f" % (*e.mean(0), *e.std(0)))
print("rmse if the constant offset were removed: %.2f m" % np.sqrt(((e - e.mean(0)) ** 2).sum(1).mean()))
z = np.load(lidar); RP = z["points"]; t2 = cKDTree(RP[:, :2])
def inside_frac(X):
    n = 0; tot = 0
    for p in X[::3]:
        nb = t2.query_ball_point(p[:2], 1.0)
        if not nb: continue
        tot += 1; n += np.max(RP[nb, 2]) > p[2] + 0.5   # surface above the camera at its own xy -> camera inside structure/under cover
    return n / max(tot, 1)
print("fraction of positions with LiDAR surface above them (inside building / under tree): reference %.2f  ours-anchored %.2f" % (inside_frac(GT[m]), inside_frac(C[m])))
