import json, struct, numpy as np, subprocess, os
R = "/workspace/voxelflight_a100_20260928/vf2"; run = R + "/runs/q720-da3g-tiled"; out = R + "/runs/web-extra"; os.makedirs(out, exist_ok=True)
# 1) cameras.json: one exact pose per keyframe, same order as trajectory.csv (first pose per video frame)
kp = np.load(run + "/keyframe_poses.npz"); c2w, vfr = kp["camera_to_world"], kp["video_frame"]
_, first = np.unique(vfr, return_index=True); first = np.sort(first); c2w, vfr = c2w[first], vfr[first]
cal = json.load(open("/workspace/voxelflight_a100_20260928/runs/zurich-10min-input/camera.json")); K = np.array(cal["intrinsic_matrix"])
json.dump({"convention": "camera_to_world, OpenCV camera axes (x right, y down, z forward), local metres (x=east, y=north, z=up)",
           "width": cal["width"], "height": cal["height"], "fx": float(K[0, 0]), "fy": float(K[1, 1]), "cx": float(K[0, 2]), "cy": float(K[1, 2]),
           "source_frames": vfr.tolist(), "c2w": [np.round(m[:3, :4], 4).ravel().tolist() for m in c2w]}, open(out + "/cameras.json", "w"))
print("cameras", len(vfr), "W,H", cal["width"], cal["height"])
# 2) errors.bin: per-vertex distance of the web textured mesh to survey LiDAR after an evaluation-only rigid ICP (uint8 decimetres; 255 = none within 5 m)
import open3d as o3d
from scipy.spatial import cKDTree
b = open(run + "/textured-web3/textured_mesh_web.glb", "rb").read(); n = struct.unpack("<I", b[12:16])[0]; g = json.loads(b[20:20 + n])
binstart = 20 + n + 8; acc = g["accessors"][0]; bv = g["bufferViews"][acc["bufferView"]]
P = np.frombuffer(b, np.float32, acc["count"] * 3, binstart + bv["byteOffset"]).reshape(-1, 3).astype(np.float64)
z = np.load(R + "/runs/lidar_ref_2018_egm96.npz"); RP, RN = z["points"], z["normals"]
src = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)); src.estimate_normals()
tgt = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(RP)); tgt.normals = o3d.utility.Vector3dVector(RN)
T = np.eye(4); reg = o3d.pipelines.registration
for d in (4.0, 2.0, 1.0, 0.5):
    T = reg.registration_icp(src, tgt, d, T, reg.TransformationEstimationPointToPlane(), reg.ICPConvergenceCriteria(max_iteration=40)).transformation
dist, _ = cKDTree(RP).query(P @ T[:3, :3].T + T[:3, 3], distance_upper_bound=5.0)
e = np.where(np.isfinite(dist), np.clip(np.round(dist * 10), 0, 254), 255).astype(np.uint8)
open(out + "/errors.bin", "wb").write(e.tobytes())
fin = np.isfinite(dist)
print("vertices", len(P), "with LiDAR", round(fin.mean(), 3), "median m", round(float(np.median(dist[fin])), 3), "<0.5 m", round(float((dist[fin] < .5).mean()), 3), "<1 m", round(float((dist[fin] < 1).mean()), 3), "icp shift m", round(float(np.linalg.norm(T[:3, 3])), 2))
