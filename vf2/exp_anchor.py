"""Experiment: DPVO + GPS/baro init -> feature points -> chunk-wise Sim3 ICP to public LiDAR. Reference cameras only for scoring."""
import json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from common import ROOT, robust_sim3, apply_sim3, sim3_error, error_summary, dump
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
import features as F
import ba as B
from anchor import icp_sim3, blend_chunk_transforms, apply_point

DEV = os.environ.get("DEV", "cpu")
CH = float(os.environ.get("CHUNK_S", 60)); STEP = float(os.environ.get("CHUNK_STEP_S", 30))
PT = float(os.environ.get("PRIOR_T", 3.0))
INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"; FIRST = 40001
cam = json.load(open(INP + "/camera.json")); K = np.array(cam["intrinsic_matrix"]); dist = np.array(cam["distortion_coefficients"])
frames = np.arange(4, 18000, 15)
tic = time.time()
feats, size = torch.load(ROOT + "/vf2/runs/cache_sp_stride15.pt", map_location="cpu")
done = pickle.load(open(ROOT + "/vf2/runs/cache_matches_stride15.pkl", "rb"))
kps_ud = F.undistort_keypoints(feats, K, dist)
verified = F.verify_pairs(kps_ud, done, K)
tracks = F.build_tracks([len(k) for k in kps_ud], verified, min_len=3, max_len=120)
print("tracks", len(tracks), time.time() - tic)

traj = np.loadtxt(ROOT + "/thirdparty/DPVO/saved_trajectories/zurich10m_stride5.txt")
dp_frames = traj[:, 0].astype(int) * 5 + 4
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv")
geo = GeoFrame(lat, lon, alt); gps_local = geo.to_local(lat, lon, alt)
eph = np.genfromtxt(INP + "/frame_telemetry.csv", delimiter=",", skip_header=1, usecols=(7,))
bz = baro_altitude(LOGS + "/BarometricPressure.csv", LOGS + "/OnboardGPS.csv", dp_frames + FIRST)
tgt = gps_local[dp_frames].copy(); tgt[:, 2] = bz - np.median(bz - tgt[:, 2])
s0, R0, t0 = robust_sim3(traj[:, 1:4], tgt, eph[dp_frames] / 3)
sel = np.searchsorted(dp_frames, frames)
Rwc = R0 @ Rotation.from_quat(traj[sel, 4:8]).as_matrix(); Cw = apply_sim3(traj[sel, 1:4], s0, R0, t0)
R_cw = np.transpose(Rwc, (0, 2, 1)); t_cw = -np.einsum("nij,nj->ni", R_cw, Cw)
X, keep = B.triangulate_tracks(tracks, kps_ud, K, R_cw, t_cw, max_reproj=10.0, device=DEV)
X = X[keep]; first = np.array([tr[:, 0].min() for tr, k in zip(tracks, keep) if k]); last = np.array([tr[:, 0].max() for tr, k in zip(tracks, keep) if k])
# drop far points (poorly conditioned): keep within 40 m of an observing camera
dcam = np.linalg.norm(X - Cw[first], axis=1); good = dcam < 40
X, first, last = X[good], first[good], last[good]
print("points", len(X))

z = np.load(ROOT + "/vf2/runs/lidar_ref_2018_egm96.npz"); RP, RN = z["points"], z["normals"]
tree = cKDTree(RP)
tkf = frames / 30.0

if os.environ.get("DIAG") == "1":
    # DIAGNOSTIC ONLY: oracle per-chunk Sim3 from reference camera centres -> how well do points sit on the LiDAR?
    from common import umeyama
    gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
    GTc = np.column_stack([np.interp(frames + FIRST, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
    tkf0 = frames / 30.0
    for a in np.arange(0, tkf0[-1], 60):
        ka = np.where((tkf0 >= a) & (tkf0 <= a + 60))[0]
        pm = (first >= ka[0]) & (last <= ka[-1] + 10) & (first <= ka[-1])
        if pm.sum() < 100: continue
        so, Ro, to = umeyama(Cw[ka], GTc[ka])
        for name, Y in [("gps_init", X[pm]), ("oracle", so * X[pm] @ Ro.T + to)]:
            d, _ = tree.query(Y)
            print(f"t={a:4.0f} {name:9s} median dist {np.median(d):.2f}  <0.3m {np.mean(d<0.3):.2f}  <0.5m {np.mean(d<0.5):.2f}  <1m {np.mean(d<1):.2f}")
    sys.exit(0)
chunks, infos = [], []
for a in np.arange(0, tkf[-1], STEP):
    b = a + CH
    ka = np.where((tkf >= a) & (tkf <= b))[0]
    if len(ka) < 10:
        continue
    pm = (first >= ka[0]) & (last <= ka[-1] + 10) & (first <= ka[-1])
    if pm.sum() < 100:
        continue
    c = Cw[ka].mean(0)
    s, R, t, info = icp_sim3(X[pm], tree, RP, RN, c, prior_sigma_t=PT)
    info.update({"t0": float(a), "points": int(pm.sum()), "scale": s, "shift_m": t.tolist(), "rot_deg": float(np.degrees(Rotation.from_matrix(R).magnitude()))})
    infos.append(info); chunks.append((a, b, c, s, R, t))
    print(json.dumps(info))
bl = blend_chunk_transforms(tkf, chunks)
C1 = np.stack([apply_point(Cw[i:i + 1], *bl[i])[0] for i in range(len(Cw))])

gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
img = frames + FIRST; m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
res = {"chunk_s": CH, "step_s": STEP, "prior_t": PT,
       "gps_baro_init_direct": error_summary(Cw[m], GT[m]), "gps_baro_init_horiz_rmse": float(np.sqrt(((Cw[m] - GT[m])[:, :2] ** 2).sum(1).mean())),
       "anchored_direct": error_summary(C1[m], GT[m]), "anchored_horiz_rmse": float(np.sqrt(((C1[m] - GT[m])[:, :2] ** 2).sum(1).mean())),
       "anchored_vert_rmse": float(np.sqrt(((C1[m] - GT[m])[:, 2] ** 2).mean())),
       "anchored_sim3": sim3_error(C1[m], GT[m]), "chunks": infos, "seconds": time.time() - tic}
out = ROOT + f"/vf2/runs/anchor-c{int(CH)}-s{int(STEP)}-p{PT:g}"
os.makedirs(out, exist_ok=True); np.save(out + "/centres.npy", C1); dump(out + "/result.json", res)
print(json.dumps({k: v for k, v in res.items() if k != "chunks"}, indent=2))
