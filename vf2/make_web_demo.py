"""Package a VoxelFlight v2 run as the website's saved demo (viewer/assets/latest/ contract).
Geometry stays in the run's local metric frame (x=E, y=N, z=up, metres from the UTM origin): the viewer is Z-up."""
import hashlib, json, os, shutil, subprocess, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, open3d as o3d, trimesh
from export import write_las
from fbx import write_fbx

run, video, evaluation, out = sys.argv[1:5]
SPEED = 4.0                       # preview time-lapse factor
FFMPEG = sys.argv[5] if len(sys.argv) > 5 else "ffmpeg"
MAX_FILE = 24 * 1024 * 1024
os.makedirs(out + "/frames", exist_ok=True)
rep = json.load(open(run + "/report.json")); ev = json.load(open(evaluation))
origin = np.array(rep["utm_origin"]); epsg = int(rep["utm_epsg"])

# --- web mesh (Z-up, vertex colours), decimated until it fits the 24 MiB budget ---
full = o3d.io.read_triangle_mesh(run + "/model/model_mesh.ply")
full_tris = len(full.triangles)
for vox in (0.15, 0.2, 0.25, 0.3, 0.4):
    m = full.simplify_vertex_clustering(vox)
    V, F = np.asarray(m.vertices), np.asarray(m.triangles); C = (np.clip(np.asarray(m.vertex_colors), 0, 1) * 255).astype(np.uint8)
    tm = trimesh.Trimesh(V, F, vertex_colors=np.column_stack([C, np.full(len(C), 255, np.uint8)]), process=False)
    tm.export(out + "/reconstruction_mesh.glb")
    if os.path.getsize(out + "/reconstruction_mesh.glb") <= MAX_FILE:
        break
print("web mesh voxel", vox, "triangles", len(F), "bytes", os.path.getsize(out + "/reconstruction_mesh.glb"))
write_fbx(out + "/reconstruction_mesh.fbx", V, F, C)

# --- point samples ---
pc = o3d.io.read_point_cloud(run + "/model/model_points.ply")
P, PC = np.asarray(pc.points), np.clip(np.asarray(pc.colors), 0, 1)
rng = np.random.default_rng(0); idx = rng.choice(len(P), min(400000, len(P)), replace=False)
Ps, Cs = P[idx], PC[idx]
with open(out + "/pointcloud.ply", "wb") as f:
    f.write(f"ply\nformat binary_little_endian 1.0\nelement vertex {len(Ps)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n".encode())
    rec = np.zeros(len(Ps), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    rec["x"], rec["y"], rec["z"] = Ps[:, 0], Ps[:, 1], Ps[:, 2]
    rec["r"], rec["g"], rec["b"] = (Cs * 255).astype(np.uint8).T
    f.write(rec.tobytes())
write_las(out + "/reconstruction_utm.las", Ps, Cs, origin, epsg)
shutil.copy(run + "/model/dsm_utm.tif", out + "/surface_model.tif")

# --- trajectory (keyframes) in the same local frame ---
kp = np.load(run + "/keyframe_poses.npz"); c2w = kp["camera_to_world"]; vfr = kp["video_frame"]
with open(out + "/trajectory.csv", "w") as f:
    f.write("frame_id,camera_x,camera_y,camera_z,source_frame,utm_easting,utm_northing,altitude_m\n")
    for i, (T, sf) in enumerate(zip(c2w, vfr)):
        x, y, z = T[:3, 3]
        f.write(f"{i},{x:.4f},{y:.4f},{z:.4f},{int(sf)},{x + origin[0]:.3f},{y + origin[1]:.3f},{z + origin[2]:.3f}\n")

# --- preview video (H.264, time-lapse) + poster + thumbnails ---
cap = cv2.VideoCapture(video); fps = cap.get(cv2.CAP_PROP_FPS) or 30.0; nfr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
for crf, scale in ((28, "1280:720"), (31, "1280:720"), (31, "960:540"), (34, "960:540")):
    subprocess.run([FFMPEG, "-y", "-loglevel", "error", "-i", video, "-vf", f"setpts=PTS/{SPEED},scale={scale}", "-r", "30",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", str(crf), "-pix_fmt", "yuv420p", "-an", "-movflags", "+faststart",
                    out + "/flight-preview.mp4"], check=True)
    if os.path.getsize(out + "/flight-preview.mp4") <= MAX_FILE:
        break
print("preview", crf, scale, os.path.getsize(out + "/flight-preview.mp4"))
K = len(vfr); thumbs = sorted(set(np.linspace(0, K - 1, 24).round().astype(int).tolist()))
def grab(frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame)); ok, im = cap.read(); return im if ok else None
for t in thumbs:
    im = grab(vfr[t]); cv2.imwrite(out + f"/frames/frame_{t:05d}.jpg", cv2.resize(im, (480, 270), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 82])
cv2.imwrite(out + "/poster.jpg", cv2.resize(grab(vfr[K // 3]), (1280, 720), interpolation=cv2.INTER_AREA), [cv2.IMWRITE_JPEG_QUALITY, 85])

# --- coverage map (north-up local XY, 2 m cells) ---
res = 2.0; lo = np.minimum(Ps[:, :2].min(0), c2w[:, :2, 3].min(0)); hi = np.maximum(Ps[:, :2].max(0), c2w[:, :2, 3].max(0))
ij = np.floor((Ps[:, :2] - lo) / res).astype(int); key, cnt = np.unique(ij, axis=0, return_counts=True)
json.dump({"bounds": [lo.tolist(), hi.tolist()], "resolution_m": res, "cells": [[int(a), int(b), int(c)] for (a, b), c in zip(key, cnt)],
           "frame": None, "note": "Local metric frame aligned to UTM grid north (x = east, y = north)."}, open(out + "/coverage.json", "w"))

# --- scene.json ---
stages = {k: float(v) for k, v in rep["stage_seconds"].items() if not k.startswith(" ")}
json.dump({"keyframes": int(K), "duration_s": round(nfr / fps, 1), "source_frames": nfr, "run_date": "2026-10-03", "run_id": os.path.basename(run.rstrip("/")),
           "thumbnail_frames": thumbs, "frame_times": [float(sf / fps / SPEED) for sf in vfr], "preview_speed": SPEED,
           "point_count": int(len(P)), "web_point_count": int(len(Ps)), "mesh_triangles": int(full_tris), "web_mesh_triangles": int(len(F)),
           "web_mesh_voxel_m": vox, "timings": stages, "processing_wall_seconds": float(rep["processing_wall_seconds"]),
           "parallel_stage_seconds": {k.strip(): float(v) for k, v in rep["stage_seconds"].items() if k.startswith(" ")},
           "runtime_scope": "Continuous end-to-end run on one A100: video on disk -> all six export formats.",
           "geometry_coordinate_frame": f"Local metres from UTM origin {origin.round(3).tolist()} (EPSG:{epsg}); x=east, y=north, z=up",
           "utm_epsg": epsg, "utm_origin": origin.tolist(),
           "provenance": "Zurich Urban MAV images 40001-58000 encoded as a 10-minute 1080p H.264 video; onboard GPS + barometer. "
                         "Web preview is a 4x time-lapse, not an original camera file."}, open(out + "/scene.json", "w"), indent=1)

# --- evaluation.json (this run) ---
json.dump({"run_id": os.path.basename(run.rstrip("/")), "keyframes": int(K),
           "protocol": {"ground_truth_used_in_construction": False, "survey_or_lidar_used_in_construction": False,
                        "camera_reference": "Zurich Urban MAV GroundTruthAGL (evaluation only)",
                        "surface_reference": "swisstopo swissSURFACE3D airborne LiDAR 2018 (evaluation only)"},
           "absolute_camera_error": ev["camera_direct"], "absolute_camera_error_horizontal_rmse_m": ev["camera_direct_horizontal_rmse_m"],
           "absolute_camera_error_vertical_rmse_m": ev["camera_direct_vertical_rmse_m"], "sim3_camera_error": ev["camera_sim3"],
           "surface_shape_vs_lidar": ev["surface_all_after_eval_rigid_icp"], "surface_absolute_vs_lidar": ev["surface_all"],
           "visible_completeness": ev.get("visible_completeness"), "f_score_1m": ev.get("f_score_1m")}, open(out + "/evaluation.json", "w"), indent=1)

# --- manifest (every file except thumbnails) ---
man = {}
for n in sorted(os.listdir(out)):
    p = os.path.join(out, n)
    if os.path.isfile(p) and n != "manifest.json":
        man[n] = {"bytes": os.path.getsize(p), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest()}
json.dump(man, open(out + "/manifest.json", "w"), indent=2)
print(json.dumps({k: v["bytes"] for k, v in man.items()}, indent=1))
