"""Experiment 2: long-baseline pairs, partial-track triangulation, converged BA, optional GPS/baro pose priors."""
import json, os, pickle, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from common import ROOT, Timer, robust_sim3, apply_sim3, sim3_error, error_summary, dump, umeyama
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
import features as F
import ba as B
import pycolmap

TAG = sys.argv[1]
RUN = ROOT + "/vf2/runs/" + TAG
os.makedirs(RUN, exist_ok=True)
STRIDE = 15
OFFSETS = [int(x) for x in os.environ.get("PAIR_OFFSETS", "1,2,3,4,6,8,10,13,16,20,30,40,60").split(",")]
PRIOR = os.environ.get("GPS_PRIOR", "0") == "1"
SIG_H = float(os.environ.get("SIG_H", 8.0)); SIG_V = float(os.environ.get("SIG_V", 0.5))
ITERS = int(os.environ.get("BA_ITERS", 300))
INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"; FIRST = 40001
T = Timer()
cam = json.load(open(INP + "/camera.json"))
K = np.array(cam["intrinsic_matrix"]); dist = np.array(cam["distortion_coefficients"])
frames = np.arange(4, 18000, STRIDE); names = [f"kf_{f:05d}.jpg" for f in frames]

fcache = ROOT + "/vf2/runs/cache_sp_stride15.pt"
if os.path.exists(fcache):
    feats, size = torch.load(fcache)
else:
    with T.stage("decode"):
        imgs, size = F.decode_keyframes(INP + "/flight.mp4", frames, workers=16, gray=True)
    with T.stage("superpoint"):
        feats, size = F.extract_superpoint(imgs)
    torch.save((feats, size), fcache)
pairs = [(i, i + o) for i in range(len(frames)) for o in OFFSETS if i + o < len(frames)]
mcache = ROOT + "/vf2/runs/cache_matches_stride15.pkl"
done = pickle.load(open(mcache, "rb")) if os.path.exists(mcache) else {}
todo = [p for p in pairs if p not in done]
with T.stage(f"lightglue {len(todo)} new pairs"):
    if todo:
        done.update(F.match_pairs_batched(feats, todo, size))
        pickle.dump(done, open(mcache, "wb"))
matches = {p: done[p] for p in pairs if p in done}
kps_ud = F.undistort_keypoints(feats, K, dist)
with T.stage("verify"):
    verified = F.verify_pairs(kps_ud, matches, K)
with T.stage("tracks"):
    tracks = F.build_tracks([len(k) for k in kps_ud], verified, min_len=3, max_len=120)
print("pairs", len(verified), "tracks", len(tracks), "mean len", np.mean([len(t) for t in tracks]))

# initial poses: DPVO + robust GPS/baro similarity (no reference data)
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
prior_xyz = tgt[sel]

# partial-track triangulation: split long tracks into windows of <=40 obs, accept if >=2 consistent obs
with T.stage("triangulate"):
    sub = []
    for tr in tracks:
        for s in range(0, len(tr), 40):
            if len(tr[s:s + 40]) >= 2:
                sub.append(tr[s:s + 40] if len(tr) - s >= 3 or s == 0 else tr[s - 1:s + 40])
    X, keep = B.triangulate_tracks(sub, kps_ud, K, R_cw, t_cw, max_reproj=10.0)
print("triangulated", int(keep.sum()), "of", len(sub))
model = RUN + "/model_init"
B.write_colmap_text(model, K, size, names, R_cw, t_cw, sub, X, keep, kps_ud)
rec = pycolmap.Reconstruction(model)

def make_opts(iters):
    o = pycolmap.BundleAdjustmentOptions()
    o.refine_focal_length = False; o.refine_principal_point = False; o.refine_extra_params = False
    o.ceres.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    o.ceres.loss_function_scale = 1.0
    o.ceres.solver_options.max_num_iterations = iters
    o.ceres.solver_options.num_threads = 96
    o.ceres.solver_options.function_tolerance = 1e-8
    o.print_summary = True
    return o

def solve(rec, iters):
    o = make_opts(iters)
    if not PRIOR:
        pycolmap.bundle_adjustment(rec, o); return
    cfg = pycolmap.BundleAdjustmentConfig()
    for iid in rec.reg_image_ids():
        cfg.add_image(iid)
    priors = []
    cov = np.diag([SIG_H ** 2, SIG_H ** 2, SIG_V ** 2])
    for iid in rec.reg_image_ids():
        im = rec.image(iid)
        pp = pycolmap.PosePrior()
        pp.position = prior_xyz[iid - 1]
        pp.position_covariance = cov
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        pp.corr_data_id = im.data_id
        priors.append(pp)
    po = pycolmap.PosePriorBundleAdjustmentOptions()
    po.alignment_ransac.max_error = 3 * SIG_H if hasattr(po, "alignment_ransac") else None
    adj = pycolmap.create_pose_prior_bundle_adjuster(o, po, cfg, priors, rec)
    summ = adj.solve()
    print(summ.BriefReport() if hasattr(summ, "BriefReport") else summ)

with T.stage("BA1"):
    solve(rec, ITERS)
with T.stage("filter+BA2"):
    rec.update_point_3d_errors()
    bad = [pid for pid, p in rec.points3D.items() if p.error > 2.5]
    for pid in bad: rec.delete_point3D(pid)
    print("removed", len(bad))
    solve(rec, ITERS)
os.makedirs(RUN + "/model", exist_ok=True); rec.write(RUN + "/model")
C = B.camera_centres(rec, len(frames))

# post-hoc scoring only
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
img = frames + FIRST
m = (img >= gt[0, 0]) & (img <= gt[-1, 0]) & np.isfinite(C).all(1)
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
if PRIOR:
    Cg = C  # already in the GPS/baro frame
else:
    s1, R1, t1 = robust_sim3(C[m], prior_xyz[m], eph[frames][m] / 3); Cg = apply_sim3(C, s1, R1, t1)
res = {"tag": TAG, "offsets": OFFSETS, "prior": PRIOR, "sig_h": SIG_H, "points": len(rec.points3D),
       "mean_track_len": float(rec.compute_mean_track_length()), "mean_reproj_px": float(rec.compute_mean_reprojection_error()),
       "init_dpvo_sim3": sim3_error(Cw[m], GT[m])["rmse_m"], "ba_sim3": sim3_error(C[m], GT[m]),
       "ba_direct": error_summary(Cg[m], GT[m]), "stage_seconds": T.stages}
np.save(RUN + "/centres.npy", C)
dump(RUN + "/result.json", res)
print(json.dumps(res, indent=2))
