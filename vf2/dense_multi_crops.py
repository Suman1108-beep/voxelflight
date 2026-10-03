"""Tiled variant of dense_multi_sfm.py: every depth view is a crop of a keyframe; crops use the parent frame's camera
(solved or interpolated) with intrinsics shifted to the crop.
Original description: dense step for consumer-GNSS flights solved by visual SfM that may split into several models.
Each SfM model is locked to GNSS/barometer independently with a robust similarity; every neural depth view is then
placed with its solved camera and scaled to the tie points it observes; tiled GPU TSDF; exports."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pycolmap
from common import Timer, dump, robust_sim3
from run_pipeline import load_telemetry
from telemetry import GeoFrame

ap = argparse.ArgumentParser()
ap.add_argument("--sparse", required=True, help="folder containing SfM models 0/, 1/, ...")
ap.add_argument("--telemetry", required=True); ap.add_argument("--predictions", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--voxel", type=float, default=0.05); ap.add_argument("--depth-max", type=float, default=25.0)
ap.add_argument("--crops", required=True); ap.add_argument("--exchange-voxel", type=float, default=0.1); ap.add_argument("--min-model-images", type=int, default=20)
a = ap.parse_args(); os.makedirs(a.output, exist_ok=True); T = Timer()
tel = load_telemetry(a.telemetry); geo = GeoFrame(tel["lat"], tel["lon"], tel["alt"]); gl = geo.to_local(tel["lat"], tel["lon"], tel["alt"])
if tel.get("baro") is not None and np.isfinite(tel["baro"]).all():
    gl[:, 2] = tel["baro"] - np.median(tel["baro"] - tel["alt"]) - geo.origin[2]
fidx = {int(f): i for i, f in enumerate(tel["frame"])}
sig = np.maximum(tel["eph"], 0.5) / 3.0

with T.stage("lock SfM models to GNSS"):
    placed = {}   # image name -> (cam_from_world 4x4, list of world tie points)
    locks = []
    for d in sorted(glob.glob(os.path.join(a.sparse, "*"))):
        rec = pycolmap.Reconstruction(d)
        if rec.num_reg_images() < a.min_model_images:
            continue
        names, C, G, S = [], [], [], []
        for im in rec.images.values():
            if im.has_pose:
                sf = int(im.name.split("_")[1].split(".")[0]); i = fidx[sf]
                names.append(im.name); C.append(im.projection_center()); G.append(gl[i]); S.append(sig[i])
        s, R, t = robust_sim3(np.array(C), np.array(G), np.array(S))
        rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
        resid = np.linalg.norm(np.array([rec.images[iid].projection_center() for iid in rec.images if rec.images[iid].has_pose]) - np.array(G), axis=1)
        locks.append({"model": os.path.basename(d), "images": len(names), "scale": s, "gnss_residual_median_m": float(np.median(resid))})
        for im in rec.images.values():
            if im.has_pose:
                cw = np.eye(4); cw[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix()
                pts = np.array([rec.points3D[q.point3D_id].xyz for q in im.points2D if q.has_point3D()])
                placed[im.name] = (cw, pts, rec.cameras[im.camera_id])
    print("locks", locks)

def interpolated(sf, max_gap=150):
    """Camera for an unsolved frame: interpolate between the nearest solved frames of the same SfM model
    (linear centre, slerp rotation). Returns None when no close pair exists — never extrapolates far."""
    from scipy.spatial.transform import Rotation, Slerp
    best = None
    for model_frames in by_model.values():
        lo = [f for f in model_frames if f < sf]; hi = [f for f in model_frames if f > sf]
        if lo and hi and hi[0] - lo[-1] <= max_gap:
            best = (lo[-1], hi[0]); break
    if best is None:
        return None
    f0, f1 = best; (c0, p0, cam), (c1, p1, _) = placed[f"f_{f0:05d}.jpg"], placed[f"f_{f1:05d}.jpg"]
    w = (sf - f0) / (f1 - f0)
    C0, C1 = -c0[:3, :3].T @ c0[:3, 3], -c1[:3, :3].T @ c1[:3, 3]
    Rwc = Slerp([0, 1], Rotation.from_matrix([c0[:3, :3].T, c1[:3, :3].T]))([w]).as_matrix()[0]
    C = (1 - w) * C0 + w * C1
    cw = np.eye(4); cw[:3, :3] = Rwc.T; cw[:3, 3] = -Rwc.T @ C
    return cw, np.concatenate([p0, p1]) if len(p0) and len(p1) else (p0 if len(p0) else p1), cam

by_model = {}
for d in sorted(glob.glob(os.path.join(a.sparse, "*"))):
    if not os.path.isdir(d): continue
    r = pycolmap.Reconstruction(d)
    if r.num_reg_images() >= a.min_model_images:
        by_model[d] = sorted(int(im.name.split("_")[1].split(".")[0]) for im in r.images.values() if im.has_pose)

with T.stage("place neural views"):
    rep = json.load(open(os.path.join(a.predictions, "report.json")))
    files = sorted(glob.glob(os.path.join(a.predictions, "predictions", "frame_*.npz")))
    views, c2ws, vframes, scales = [], [], [], []; n_interp = 0
    crops = json.load(open(os.path.join(a.crops, "crops.json")))
    for fpath, fr in zip(files, rep["frames"]):
        m = crops.get(fr.get("source_name"))
        if m is None:
            continue
        sf = m["source_frame"]; item = placed.get(m["parent"])
        if item is None:
            item = interpolated(sf)
            if item is None:
                continue
            n_interp += 1
        cw, pts, cam = item
        p = np.load(fpath); d0 = np.where(p["mask"], p["depth"], 0).astype(np.float32); h, w = d0.shape
        K = cam.calibration_matrix().copy(); K[0, :] *= m["W"] / cam.width; K[1, :] *= m["H"] / cam.height
        K[0, 2] -= m["ox"]; K[1, 2] -= m["oy"]; K[0, :] *= w / m["w"]; K[1, :] *= h / m["h"]
        if len(pts) < 20:
            continue
        X = pts @ cw[:3, :3].T + cw[:3, 3]; z = X[:, 2]
        u = (K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int); v = (K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dn = d0[v[ok], u[ok]]; good = dn > 0
        if good.sum() < 15:
            continue
        sc = float(np.median(z[ok][good] / dn[good])); scales.append(sc)
        c2w = np.linalg.inv(cw); c2ws.append(c2w); vframes.append(sf)
        views.append({"depth": d0 * sc, "color": p["color"], "K": K, "c2w": c2w})
    print("interpolated cameras", n_interp)
    print("views", len(views), "scale median %.3f p5 %.3f p95 %.3f" % (np.median(scales), *np.percentile(scales, [5, 95])))

with T.stage("tiled GPU TSDF"):
    from fusion import gpu_tsdf_tiled, mesh_stats
    mesh, pcd = gpu_tsdf_tiled(views, tile=60.0, voxel=a.voxel, depth_max=a.depth_max); mesh.compute_vertex_normals()
with T.stage("exports"):
    from export import export_all
    P, Cc = np.asarray(pcd.points), np.asarray(pcd.colors)
    meta = {"ground_truth_used": False, "survey_or_lidar_used": False, "pipeline": "VoxelFlight v2 visual SfM + GNSS lock"}
    export_all(a.output + "/model", mesh, P, Cc, geo.origin, geo.epsg, extra_meta=meta, formats=("ply", "las", "tif"))
    light = mesh.simplify_vertex_clustering(a.exchange_voxel); light.compute_vertex_normals()
    export_all(a.output + "/model", light, P[::4], Cc[::4], geo.origin, geo.epsg, extra_meta=meta, formats=("obj", "glb", "fbx"))
np.savez(a.output + "/keyframe_poses.npz", camera_to_world=np.array(c2ws), video_frame=np.array(vframes), utm_origin=geo.origin, epsg=geo.epsg)
dump(a.output + "/report.json", {"schema": "voxelflight.v2.sfm-gnss-tiled", "stage_seconds": T.stages, "models": locks, "views": len(views), "interpolated_views": n_interp,
     "mesh": mesh_stats(mesh), "utm_epsg": geo.epsg, "utm_origin": geo.origin.tolist(), "inference_dir": a.predictions,
     "ground_truth_used": False, "survey_or_lidar_used": False})
print("done", T.total())
