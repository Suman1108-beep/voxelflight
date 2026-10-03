"""Post-hoc evaluation of a frozen VoxelFlight run (never used by the pipeline).
1) keyframe camera centres vs Zurich reference poses (direct + Sim3)
2) surface vs swissSURFACE3D airborne LiDAR (independent survey), on horizontal surfaces (ground/roofs) where
   airborne LiDAR is dense, plus all-surface nearest distances
3) visible-ground completeness proxy: fraction of LiDAR ground within 20 m of the flight path that is reconstructed."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from common import error_summary, sim3_error, dump
ROOT = "/workspace/voxelflight_a100_20260928"
run = sys.argv[1]; seg = sys.argv[2] if len(sys.argv) > 2 else ROOT + "/datasets/zurich-40001-58000"; first_id = int(sys.argv[3]) if len(sys.argv) > 3 else 40001
lidar_npz = sys.argv[4] if len(sys.argv) > 4 else ROOT + "/vf2/runs/lidar_ref_2018_egm96.npz"
import open3d as o3d
kp = np.load(run + "/keyframe_poses.npz"); origin = kp["utm_origin"]
C = kp["camera_to_world"][:, :3, 3]; img = kp["video_frame"] + first_id
gt = np.genfromtxt(seg + "/Log Files/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - origin
res = {"run": run, "camera_direct": error_summary(C[m], GT[m]),
       "camera_direct_horizontal_rmse_m": float(np.sqrt(((C[m] - GT[m])[:, :2] ** 2).sum(1).mean())),
       "camera_direct_vertical_rmse_m": float(np.sqrt(((C[m] - GT[m])[:, 2] ** 2).mean())),
       "camera_sim3": sim3_error(C[m], GT[m])}
# surfaces
z = np.load(lidar_npz); RP, RN = z["points"], z["normals"]
lid_tree = cKDTree(RP)
mesh = o3d.io.read_triangle_mesh(run + "/model/model_mesh.ply")
pc = mesh.sample_points_uniformly(400000, use_triangle_normal=True)
P = np.asarray(pc.points); N = np.asarray(pc.normals)
d_all, j = lid_tree.query(P, distance_upper_bound=5.0)
horiz = np.abs(N[:, 2]) > 0.9
lid_h = np.abs(RN[:, 2]) > 0.9
# vertical distance from horizontal predicted surfaces to nearest horizontal LiDAR surface (xy-local)
RH = RP[lid_h]
t2 = cKDTree(RH[:, :2])
dd, jj = t2.query(P[horiz, :2], k=8, distance_upper_bound=0.5)
zz = np.vstack([RH[:, 2:3], [[np.inf]]])[np.minimum(jj, len(RH)), 0]  # missing neighbours -> inf
dz = np.min(np.abs(zz - P[horiz, 2:3]), axis=1)
dz = dz[np.isfinite(dz)]
fin = np.isfinite(d_all)
def pct(x, t): return float(np.mean(x < t))
res["surface_all"] = {"samples": int(len(P)), "within_5m_frac": float(fin.mean()), "median_m": float(np.median(d_all[fin])),
                      "lt_0.5m": pct(d_all, 0.5), "lt_1m": pct(d_all, 1.0), "lt_2m": pct(d_all, 2.0),
                      "note": "airborne LiDAR barely samples facades, so facade points inflate this distance"}
res["surface_horizontal_vertical_error"] = {"samples": int(len(dz)), "median_m": float(np.median(dz)), "rmse_m": float(np.sqrt(np.mean(np.minimum(dz, 5) ** 2))),
                                            "lt_0.3m": pct(dz, 0.3), "lt_0.5m": pct(dz, 0.5), "lt_1m": pct(dz, 1.0)}

# shape-only surface accuracy: one evaluation-only rigid ICP of the frozen model onto the LiDAR (like Sim3 for cameras)
src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); src.normals = o3d.utility.Vector3dVector(N)
tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(RP)); tgt.normals = o3d.utility.Vector3dVector(RN)
Tm = np.eye(4)
for dmax in (4.0, 2.0, 1.0, 0.5):
    Tm = o3d.pipelines.registration.registration_icp(src, tgt, dmax, Tm, o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=40)).transformation
Pa = P @ Tm[:3, :3].T + Tm[:3, 3]
da, _ = lid_tree.query(Pa, distance_upper_bound=5.0)
res["surface_all_after_eval_rigid_icp"] = {"median_m": float(np.median(da[np.isfinite(da)])), "lt_0.5m": pct(da, 0.5), "lt_1m": pct(da, 1.0), "lt_2m": pct(da, 2.0),
    "icp_translation_m": float(np.linalg.norm(Tm[:3, 3])), "icp_rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(Tm[:3, :3]) - 1) / 2, -1, 1)))),
    "note": "evaluation-only alignment; isolates model shape from GNSS placement error"}

# visibility-aware completeness: LiDAR points seen by >=1 keyframe camera (FOV, range, z-buffer occlusion)
from visible_completeness import visible_mask
import json as _json
cam = _json.load(open(sys.argv[5])) if len(sys.argv) > 5 else None
if cam is not None:
    Kf = np.array(cam["intrinsic_matrix"]); Wf, Hf = int(cam["width"]), int(cam["height"])
    c2w_eval = kp["camera_to_world"].copy()
    for i in range(len(c2w_eval)):
        c2w_eval[i] = Tm @ c2w_eval[i]          # same evaluation-only rigid alignment as the shape metric
    vis = visible_mask(RP, c2w_eval, Kf, Wf, Hf, normals=RN)
    Tinv = np.linalg.inv(Tm)
    RPm = RP[vis] @ Tinv[:3, :3].T + Tinv[:3, 3]  # visible reference points expressed in the model frame
    dv, _ = cKDTree(P).query(RPm, distance_upper_bound=3.0)
    res["visible_completeness"] = {"visible_reference_points": int(vis.sum()), "of_reference_points": int(len(RP)),
        "recall_0.5m": pct(dv, 0.5), "recall_1m": pct(dv, 1.0), "recall_2m": pct(dv, 2.0),
        "note": "reference = swisstopo LiDAR points visible from >=1 camera (FOV, <=35 m, z-buffer occlusion); after eval-only rigid alignment"}
    prec1 = res["surface_all_after_eval_rigid_icp"]["lt_1m"]; rec1 = res["visible_completeness"]["recall_1m"]
    res["f_score_1m"] = float(2 * prec1 * rec1 / max(prec1 + rec1, 1e-9))
# visible-ground completeness proxy (legacy, not visibility-aware)
traj_tree = cKDTree(C[:, :2])
near, _ = traj_tree.query(RP[:, :2])
ground_idx = np.where(lid_h & (near < 20) & (RP[:, 2] < np.interp(0, [0], [0]) + C[:, 2].max()))[0]
ground_idx = ground_idx[RP[ground_idx, 2] < np.median(C[:, 2]) - 3]  # below camera height -> street level
pt_tree = cKDTree(P)
dg, _ = pt_tree.query(RP[ground_idx], distance_upper_bound=2.0)
res["ground_completeness_proxy"] = {"lidar_ground_pts_within_20m_of_path": int(len(ground_idx)),
                                    "reconstructed_within_0.5m": pct(dg, 0.5), "within_1m": pct(dg, 1.0)}
res["protocol"] = "Run frozen before evaluation; reference poses and LiDAR are never inputs to the pipeline."
dump(run + "/evaluation.json", res)
print(json.dumps(res, indent=2))
