"""Method test: per-chunk 2-D (yaw, tx, ty) robust point-to-wall ICP against OSM footprints; height stays barometric.
Reference cameras are used only for the final score."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from common import ROOT, robust_sim3, apply_sim3, error_summary, sim3_error, dump
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
import features as F, ba as B
from footprints import fetch_osm_buildings, wall_segments, point_wall_distance

CH = float(os.environ.get("CHUNK_S", 60)); STEP = float(os.environ.get("CHUNK_STEP_S", 20)); PT = float(os.environ.get("PRIOR_T", 4.0))
INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"; FIRST = 40001
cam = json.load(open(INP + "/camera.json")); K = np.array(cam["intrinsic_matrix"]); dist = np.array(cam["distortion_coefficients"])
frames = np.arange(4, 18000, 15)
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv"); geo = GeoFrame(lat, lon, alt)
S = wall_segments(fetch_osm_buildings(None, ROOT + "/datasets/osm_buildings_zurich_40001.json")["rings"], geo)
feats, size = torch.load(ROOT + "/vf2/runs/cache_sp_stride15.pt", map_location="cpu")
done = pickle.load(open(ROOT + "/vf2/runs/cache_matches_stride15.pkl", "rb"))
kps_ud = F.undistort_keypoints(feats, K, dist); verified = F.verify_pairs(kps_ud, done, K)
tracks = F.build_tracks([len(k) for k in kps_ud], verified, min_len=3, max_len=120)
traj = np.loadtxt(ROOT + "/thirdparty/DPVO/saved_trajectories/zurich10m_stride5.txt"); dp = traj[:, 0].astype(int) * 5 + 4
gl = geo.to_local(lat, lon, alt); eph = np.genfromtxt(INP + "/frame_telemetry.csv", delimiter=",", skip_header=1, usecols=(7,))
bz = baro_altitude(LOGS + "/BarometricPressure.csv", LOGS + "/OnboardGPS.csv", dp + FIRST)
tgt = gl[dp].copy(); tgt[:, 2] = bz - np.median(bz - tgt[:, 2])
s0, R0, t0 = robust_sim3(traj[:, 1:4], tgt, eph[dp] / 3); sel = np.searchsorted(dp, frames)
Rwc = R0 @ Rotation.from_quat(traj[sel, 4:8]).as_matrix(); Cw = apply_sim3(traj[sel, 1:4], s0, R0, t0)
R_cw = np.transpose(Rwc, (0, 2, 1)); t_cw = -np.einsum("nij,nj->ni", R_cw, Cw)
X, keep = B.triangulate_tracks(tracks, kps_ud, K, R_cw, t_cw, max_reproj=10.0, device="cuda")
first = np.array([t[:, 0].min() for t in tracks])[keep]; X = X[keep]
dcam = np.linalg.norm(X - Cw[first], axis=1)
hcam = X[:, 2] - Cw[first, 2]
band = (dcam < 30) & (hcam > -6) & (hcam < 12)  # facade band, nearby
X, first = X[band], first[band]
tk = frames / 30.0
chunks = []
for a in np.arange(0, tk[-1] - 10, STEP):
    ka = np.where((tk >= a) & (tk <= a + CH))[0]
    pm = (first >= ka[0]) & (first <= ka[-1])
    if pm.sum() < 200: continue
    P = X[pm, :2]; c = Cw[ka, :2].mean(0)
    th, t = 0.0, np.zeros(2)
    for dmax in (3.0, 2.0, 1.5, 1.0, 0.6, 0.4):
        for _ in range(3):
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            Y = (P - c) @ Rz.T + c + t
            d, I, n = point_wall_distance(Y, S, return_normal=True)
            q = S[I, :2]
            r = np.einsum("ij,ij->i", Y - q, n)  # signed distance to wall line
            ok = np.abs(r) < dmax
            y = Y[ok] - c - t
            J = np.column_stack([n[ok, 0] * -y[:, 1] + n[ok, 1] * y[:, 0], n[ok]])
            w = np.where(np.abs(r[ok]) < 0.15, 1, 0.15 / np.abs(r[ok]))
            npt = max(1.0, ok.sum() / 100.0)
            A = J.T @ (w[:, None] * J) / npt / 0.15**2 + np.diag([1 / np.radians(3) ** 2, 1 / PT**2, 1 / PT**2])
            b = -J.T @ (w * r[ok]) / npt / 0.15**2 - np.array([th / np.radians(3) ** 2, t[0] / PT**2, t[1] / PT**2])
            dx = np.linalg.solve(A, b); th += dx[0]; t += dx[1:]
    frac = float(np.mean(np.abs(r) < 0.3))
    chunks.append((a, a + CH, c, th, t.copy(), frac))
    print(f"t={a:4.0f} pts {len(P)} yaw {np.degrees(th):5.2f} shift {t.round(2)} on-wall<0.3m {frac:.2f}")
# blend (triangular weights) and apply to camera centres
C1 = Cw.copy()
for i, tt in enumerate(tk):
    ws, sh = [], []
    for (a, b, c, th, t, frac) in chunks:
        if a <= tt <= b:
            w = (1 - abs(tt - (a + b) / 2) / ((b - a) / 2) + 1e-3) * frac
            Rz = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
            ws.append(w); sh.append((Cw[i, :2] - c) @ Rz.T + c + t)
    if ws:
        C1[i, :2] = np.average(np.array(sh), axis=0, weights=ws)
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
img = frames + FIRST; m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
hr = lambda A: float(np.sqrt(((A[m] - GT[m])[:, :2] ** 2).sum(1).mean()))
res = {"init_direct": error_summary(Cw[m], GT[m]), "init_horiz": hr(Cw), "walls_direct": error_summary(C1[m], GT[m]), "walls_horiz": hr(C1),
       "vert_rmse": float(np.sqrt(((C1[m] - GT[m])[:, 2] ** 2).mean())), "walls_sim3": sim3_error(C1[m], GT[m])["rmse_m"]}
print(json.dumps(res, indent=2)); dump(ROOT + "/vf2/runs/walls-result.json", res)
