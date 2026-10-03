"""Test dense-model anchoring to public LiDAR; score cameras vs the independent Zurich reference poses."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, open3d as o3d
from common import error_summary, sim3_error, dump
from anchor_dense import anchor, correct_poses
run, seg, first_id, lidar = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
tile = float(sys.argv[5]) if len(sys.argv) > 5 else 60.0
t0 = time.time()
mesh = o3d.io.read_triangle_mesh(run + "/model/model_mesh.ply")
pc = mesh.sample_points_uniformly(600000, use_triangle_normal=True)
P, N = np.asarray(pc.points), np.asarray(pc.normals)
z = np.load(lidar); RP, RN = z["points"], z["normals"]
Tg, gi, tiles = anchor(P, N, RP, RN, tile=tile)
kp = np.load(run + "/keyframe_poses.npz"); c2w = kp["camera_to_world"]; origin = kp["utm_origin"]
new = correct_poses(c2w, Tg, tiles)
newg = correct_poses(c2w, Tg, [])
img = kp["video_frame"] + first_id
gt = np.genfromtxt(seg + "/Log Files/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
m = (img >= gt[0, 0]) & (img <= gt[-1, 0])
GT = np.column_stack([np.interp(img, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - origin
H = lambda C: float(np.sqrt(((C[m] - GT[m])[:, :2] ** 2).sum(1).mean()))
V = lambda C: float(np.sqrt(((C[m] - GT[m])[:, 2] ** 2).mean()))
res = {"run": run, "tile_m": tile, "seconds": time.time() - t0, "global_icp": gi, "global_shift_m": Tg[:3, 3].tolist(),
       "tiles_used": len(tiles), "tile_fitness": [round(t["fitness"], 3) for t in tiles],
       "before": {**error_summary(c2w[m, :3, 3], GT[m]), "horizontal_rmse_m": H(c2w[:, :3, 3]), "vertical_rmse_m": V(c2w[:, :3, 3])},
       "after_global": {**error_summary(newg[m, :3, 3], GT[m]), "horizontal_rmse_m": H(newg[:, :3, 3]), "vertical_rmse_m": V(newg[:, :3, 3])},
       "after_tiles": {**error_summary(new[m, :3, 3], GT[m]), "horizontal_rmse_m": H(new[:, :3, 3]), "vertical_rmse_m": V(new[:, :3, 3])},
       "after_tiles_sim3": sim3_error(new[m, :3, 3], GT[m])["rmse_m"]}
np.savez(run + f"/keyframe_poses_anchored_t{int(tile)}.npz", camera_to_world=new, video_frame=kp["video_frame"], utm_origin=origin, T_global=Tg)
dump(run + f"/anchor_dense_t{int(tile)}.json", res)
print(json.dumps({k: v for k, v in res.items() if k != "tile_fitness"}, indent=1))
