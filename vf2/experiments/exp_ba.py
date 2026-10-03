"""Experiment: DPVO init -> SuperPoint/LightGlue tracks -> global BA. Reference used only for post-hoc scoring."""
import json, os, pickle, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial.transform import Rotation
from common import ROOT, Timer, robust_sim3, apply_sim3, sim3_error, error_summary, dump
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
import features as F
import ba as B

RUN = ROOT + "/vf2/runs/" + (sys.argv[1] if len(sys.argv) > 1 else "ba-exp1")
STRIDE = int(os.environ.get("KF_STRIDE", 15))
OFFSETS = [int(x) for x in os.environ.get("PAIR_OFFSETS", "1,2,3,4,6,8,10,13,16,20").split(",")]
os.makedirs(RUN, exist_ok=True)
INP = ROOT + "/runs/zurich-10min-input"
LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"
FIRST = 40001
T = Timer()

cam = json.load(open(INP + "/camera.json"))
K = np.array(cam["intrinsic_matrix"]); dist = np.array(cam["distortion_coefficients"])
frames = np.arange(4, 18000, STRIDE)
names = [f"kf_{f:05d}.jpg" for f in frames]

cache = RUN + "/features.pkl"
if os.path.exists(cache):
    feats_np, size, verified = pickle.load(open(cache, "rb"))
else:
    with T.stage("decode"):
        imgs, size = F.decode_keyframes(INP + "/flight.mp4", frames, scale=1.0, workers=16, gray=True)
        # sanity: decoded frame index must match the source JPEG (and not its neighbour)
        import cv2
        for j in (100, 700):
            ref = cv2.imread(f"{INP}/frames/{FIRST + frames[j]}.jpg", cv2.IMREAD_GRAYSCALE)
            nb = cv2.imread(f"{INP}/frames/{FIRST + frames[j] + 3}.jpg", cv2.IMREAD_GRAYSCALE)
            print("decode check MAE vs source", float(np.abs(imgs[j].astype(int) - ref).mean()),
                  "vs neighbour", float(np.abs(imgs[j].astype(int) - nb).mean()))
    with T.stage("superpoint"):
        feats, size = F.extract_superpoint(imgs)
        del imgs
    pairs = [(i, i + o) for i in range(len(frames)) for o in OFFSETS if i + o < len(frames)]
    with T.stage(f"lightglue {len(pairs)} pairs"):
        matches = F.match_pairs(feats, pairs, size)
    feats_np = [{"kp": f["keypoints"].float().cpu().numpy()} for f in feats]
    kps_ud = F.undistort_keypoints(feats, K, dist)
    with T.stage("verify"):
        verified = F.verify_pairs(kps_ud, matches, K)
    for f, u in zip(feats_np, kps_ud):
        f["ud"] = u
    pickle.dump((feats_np, size, verified), open(cache, "wb"))
kps_ud = [f["ud"] for f in feats_np]
print("verified pairs", len(verified), "of", len(frames) * len(OFFSETS), "mean inliers", np.mean([len(v) for v in verified.values()]))

with T.stage("tracks"):
    tracks = F.build_tracks([len(k) for k in kps_ud], verified, min_len=3)
print("tracks", len(tracks), "mean len", np.mean([len(t) for t in tracks]))

# ---- initial poses: DPVO (stride-5 frames 5i+4) + GPS/baro global similarity (no reference) ----
traj = np.loadtxt(ROOT + "/thirdparty/DPVO/saved_trajectories/zurich10m_stride5.txt")
dp_frames = traj[:, 0].astype(int) * 5 + 4
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv")
geo = GeoFrame(lat, lon, alt)
gps_local = geo.to_local(lat, lon, alt)  # per video frame
eph = np.genfromtxt(INP + "/frame_telemetry.csv", delimiter=",", skip_header=1, usecols=(7,))
bz = baro_altitude(LOGS + "/BarometricPressure.csv", LOGS + "/OnboardGPS.csv", dp_frames + FIRST)
tgt = gps_local[dp_frames].copy()
tgt[:, 2] = bz - np.median(bz - tgt[:, 2])
s0, R0, t0 = robust_sim3(traj[:, 1:4], tgt, eph[dp_frames] / 3)
sel = np.searchsorted(dp_frames, frames)
assert np.all(dp_frames[sel] == frames)
Rwc = R0 @ Rotation.from_quat(traj[sel, 4:8]).as_matrix()
Cw = apply_sim3(traj[sel, 1:4], s0, R0, t0)
R_cw = np.transpose(Rwc, (0, 2, 1)); t_cw = -np.einsum("nij,nj->ni", R_cw, Cw)

with T.stage("triangulate"):
    X, keep = B.triangulate_tracks(tracks, kps_ud, K, R_cw, t_cw)
print("triangulated", int(keep.sum()), "of", len(tracks))
with T.stage("write model"):
    model = RUN + "/model_init"
    B.write_colmap_text(model, K, size, names, R_cw, t_cw, tracks, X, keep, kps_ud)
with T.stage("bundle adjustment"):
    rec = B.run_bundle_adjustment(model, iterations=int(os.environ.get("BA_ITERS", 60)))
with T.stage("filter+BA2"):
    bad = B.filter_and_rerun(rec, 3.0)
    print("removed", len(bad), "points")
    import pycolmap
    os.makedirs(RUN + "/model_ba1", exist_ok=True); rec.write(RUN + "/model_ba1")
    rec = B.run_bundle_adjustment(RUN + "/model_ba1", iterations=40)
    os.makedirs(RUN + "/model_ba2", exist_ok=True); rec.write(RUN + "/model_ba2")
C = B.camera_centres(rec, len(frames))

# ---- post-hoc scoring only ----
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]
gt = gt[np.argsort(gt[:, 0])]
img = frames + FIRST
m = (img >= gt[0, 0]) & (img <= gt[-1, 0]) & np.isfinite(C).all(1)
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
s1, R1, t1 = robust_sim3(C[m], tgt[sel][m], eph[frames][m] / 3)
Cg = apply_sim3(C, s1, R1, t1)
res = {"keyframes": len(frames), "tracks": len(tracks), "points": int(keep.sum()),
       "init_dpvo_sim3": sim3_error(Cw[m], GT[m]), "init_dpvo_direct": error_summary(Cw[m], GT[m]),
       "ba_sim3": sim3_error(C[m], GT[m]), "ba_gps_direct": error_summary(Cg[m], GT[m]),
       "stage_seconds": T.stages}
# chunk scale stability
for name, P in [("dpvo", Cw), ("ba", C)]:
    sc = []
    for i in range(0, m.sum() - 120, 120):
        from common import umeyama
        s, _, _ = umeyama(P[m][i:i + 120], GT[m][i:i + 120]); sc.append(s)
    res[f"{name}_60s_scale_range"] = [float(min(sc)), float(max(sc))]
np.save(RUN + "/ba_centres.npy", C)
dump(RUN + "/result.json", res)
print(json.dumps(res, indent=2))
