import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from common import umeyama, apply_sim3
from telemetry import load_video_telemetry, GeoFrame
ROOT = "/workspace/voxelflight_a100_20260928"; INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"
C = np.load(ROOT + "/vf2/runs/ba-exp1/ba_centres.npy")
frames = np.arange(4, 18000, 15); img = frames + 40001
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv"); geo = GeoFrame(lat, lon, alt)
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :7]; gt = gt[np.argsort(gt[:, 0])]
m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
s, R, t = umeyama(C[m], GT[m]); A = apply_sim3(C[m], s, R, t); G = GT[m]
e = A - G
tan = np.gradient(G, axis=0); tan[:, 2] = 0; tan /= np.linalg.norm(tan, axis=1, keepdims=True) + 1e-9
along = (e * tan).sum(1); vert = e[:, 2]
cross = np.linalg.norm(e[:, :2] - along[:, None] * tan[:, :2], axis=1)
print("rmse along %.2f cross %.2f vert %.2f" % (np.sqrt((along**2).mean()), np.sqrt((cross**2).mean()), np.sqrt((vert**2).mean())))
for k in range(0, m.sum(), 100):
    seg = slice(k, k + 100)
    pl = np.linalg.norm(np.diff(G[seg], axis=0), axis=1).sum(); pa = np.linalg.norm(np.diff(A[seg], axis=0), axis=1).sum()
    print(f"t={k/2:5.0f}s  err {np.linalg.norm(e[seg],axis=1).mean():5.2f}  along {along[seg].mean():6.2f} cross {cross[seg].mean():5.2f} vert {vert[seg].mean():6.2f}  GTpath {pl:6.1f} predpath {pa:6.1f}")
# Does the GT path look jittery vs ours (reference noise)?
for name, P in [("gt", G), ("ba", A)]:
    d2 = np.linalg.norm(np.diff(P, 2, axis=0), axis=1)
    print(name, "median |2nd diff| at 0.5s:", np.median(d2))
