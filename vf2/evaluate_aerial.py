"""Post-hoc evaluation of an aerial run against independent USGS 3DEP LiDAR (never a pipeline input).
Reports, per surface class (roof / ground-road / other): precision (model->LiDAR), visible recall (LiDAR->model),
both after one evaluation-only rigid ICP; and the GNSS placement offset (that ICP's translation)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, open3d as o3d, pyproj
from scipy.spatial import cKDTree
from common import dump
from visible_completeness import visible_mask
run, raw = sys.argv[1], sys.argv[2]
rep = json.load(open(run + "/report.json")); origin = np.array(rep["utm_origin"]); epsg = rep["utm_epsg"]
z = np.load(raw, allow_pickle=True)
src = pyproj.CRS.from_wkt(str(z["src_crs_wkt"]))
t = pyproj.Transformer.from_crs(src, epsg, always_xy=True)
x, y = t.transform(z["points_src"][:, 0], z["points_src"][:, 1])
R = np.column_stack([x, y, z["points_src"][:, 2]]) - origin
pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(R)).voxel_down_sample(0.3)
pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=1.5, max_nn=30))
RP, RN = np.asarray(pc.points), np.asarray(pc.normals)
# classes: local ground = 5th pct height on a 20 m grid
g = np.floor(RP[:, :2] / 20).astype(int); key = g[:, 0] * 100000 + g[:, 1]
gz = {k: np.percentile(RP[key == k, 2], 5) for k in np.unique(key)}
hag = RP[:, 2] - np.array([gz[k] for k in key])
horiz = np.abs(RN[:, 2]) > 0.85
cls = np.where(horiz & (hag < 0.5), "ground_road", np.where(horiz & (hag > 2.5), "roof", "other"))
mesh = o3d.io.read_triangle_mesh(run + "/model/model_mesh.ply")
s = mesh.sample_points_uniformly(800000, use_triangle_normal=True); P, N = np.asarray(s.points), np.asarray(s.normals)
reg = o3d.pipelines.registration
srcpc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); srcpc.normals = o3d.utility.Vector3dVector(N)
tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(RP)); tgt.normals = o3d.utility.Vector3dVector(RN)
# coarse vertical pre-alignment (EXIF altitude may be relative to take-off): match median heights
T = np.eye(4); T[2, 3] = np.median(RP[cls == "ground_road", 2]) - np.percentile(P[:, 2], 10)
for d in (8.0, 4.0, 2.0, 1.0, 0.5):
    T = reg.registration_icp(srcpc, tgt, d, T, reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d)), reg.ICPConvergenceCriteria(max_iteration=50)).transformation
Pa = P @ T[:3, :3].T + T[:3, 3]
lt = cKDTree(RP); dp, jp = lt.query(Pa, distance_upper_bound=5.0)
pct = lambda a, th: float(np.mean(a < th))
_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True).transform(origin[0], origin[1])
_N = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4326+5773", always_xy=True).transform(_ll[0], _ll[1], 0.0)[2] * -1.0  # geoid undulation N
res = {"run": run, "reference": str(z["resource"]), "reference_points_0.3m_voxel": int(len(RP)),
       "gnss_placement_offset_m": {"east": float(T[0, 3]), "north": float(T[1, 3]), "horizontal": float(np.hypot(T[0, 3], T[1, 3])),
                                   "rotation_deg": float(np.degrees(np.arccos(np.clip((np.trace(T[:3, :3]) - 1) / 2, -1, 1)))),
                                   "vertical_incl_datum": float(T[2, 3]),
                                   "geoid_egm96_N_m": float(_N)},
       "precision_all": {"median_m": float(np.median(dp[np.isfinite(dp)])), "lt_0.5m": pct(dp, 0.5), "lt_1m": pct(dp, 1.0), "lt_2m": pct(dp, 2.0)}}
# per-class precision: model samples whose nearest reference point has that class
for c in ("roof", "ground_road", "other"):
    m = np.isfinite(dp) & (cls[np.minimum(jp, len(RP) - 1)] == c)
    res[f"precision_{c}"] = {"samples": int(m.sum()), "median_m": float(np.median(dp[m])) if m.any() else None, "lt_0.5m": pct(dp[m], 0.5), "lt_1m": pct(dp[m], 1.0)}
# visibility-aware recall per class
kp = np.load(run + "/keyframe_poses.npz"); c2w = kp["camera_to_world"].copy()
for i in range(len(c2w)): c2w[i] = T @ c2w[i]
pf = sorted(glob.glob(os.path.join(rep.get("inference_dir") or run + "/inference", "predictions", "frame_*.npz")))
p0 = np.load(pf[0]); Hh, Ww = p0["depth"].shape; Kp = p0["intrinsics"]
vis = visible_mask(RP, c2w, Kp, Ww, Hh, depth_max=250.0, scale=0.5, normals=RN)
Tinv = np.linalg.inv(T); RPm = RP @ Tinv[:3, :3].T + Tinv[:3, 3]
dr, _ = cKDTree(P).query(RPm, distance_upper_bound=5.0)
for c in ("roof", "ground_road", "other", "all"):
    m = vis & ((cls == c) if c != "all" else True)
    res[f"visible_recall_{c}"] = {"visible_points": int(m.sum()), "lt_0.5m": pct(dr[m], 0.5), "lt_1m": pct(dr[m], 1.0), "lt_2m": pct(dr[m], 2.0)}
pr, rc = res["precision_all"]["lt_1m"], res["visible_recall_all"]["lt_1m"]
res["f_score_1m"] = 2 * pr * rc / max(pr + rc, 1e-9)
res["protocol"] = "Frozen run; USGS 3DEP LiDAR used only here. One evaluation-only rigid ICP separates shape from GNSS placement."
dump(run + "/evaluation_aerial.json", res); print(json.dumps(res, indent=1))
