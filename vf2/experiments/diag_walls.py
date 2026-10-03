"""DIAGNOSTIC: do street-level feature points sit on OSM walls / LiDAR ground under oracle vs GPS alignment?"""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from common import ROOT, robust_sim3, apply_sim3, umeyama
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
import features as F, ba as B
from footprints import fetch_osm_buildings, wall_segments, point_wall_distance
INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"; FIRST = 40001
cam = json.load(open(INP + "/camera.json")); K = np.array(cam["intrinsic_matrix"]); dist = np.array(cam["distortion_coefficients"])
frames = np.arange(4, 18000, 15)
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv"); geo = GeoFrame(lat, lon, alt)
rings = fetch_osm_buildings((lon.min() - 0.002, lat.min() - 0.0015, lon.max() + 0.002, lat.max() + 0.0015), ROOT + "/datasets/osm_buildings_zurich_40001.json")["rings"]
S = wall_segments(rings, geo); print("buildings", len(rings), "wall segments", len(S))
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
first = np.array([t[:, 0].min() for t in tracks]); last = np.array([t[:, 0].max() for t in tracks])
X, first, last = X[keep], first[keep], last[keep]
ok = np.linalg.norm(X - Cw[first], axis=1) < 40; X, first, last = X[ok], first[ok], last[ok]
z = np.load(ROOT + "/vf2/runs/lidar_ref_2018_egm96.npz"); RP = z["points"]; t2 = cKDTree(RP[:, :2])
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
GTc = np.column_stack([np.interp(frames + FIRST, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
tk = frames / 30.0
def ground_z(xy):
    d, j = t2.query(xy, k=30)
    return np.percentile(RP[j, 2], 10, axis=1)
for a in np.arange(0, tk[-1], 60):
    ka = np.where((tk >= a) & (tk <= a + 60))[0]
    pm = (first >= ka[0]) & (first <= ka[-1])
    if pm.sum() < 100: continue
    so, Ro, to = umeyama(Cw[ka], GTc[ka])
    for name, Y in [("gps_init", X[pm]), ("oracle", so * X[pm] @ Ro.T + to)]:
        h = Y[:, 2] - ground_z(Y[:, :2])
        g = np.abs(h) < 0.6
        dw = point_wall_distance(Y[~g, :2], S)
        print(f"t={a:4.0f} {name:8s} ground pts {g.mean():.2f} |h| med {np.median(np.abs(h[np.abs(h)<2])):.2f}  wall pts {len(dw)}: med {np.median(dw):.2f} <0.3 {np.mean(dw<0.3):.2f} <0.5 {np.mean(dw<0.5):.2f} <1 {np.mean(dw<1):.2f}")
