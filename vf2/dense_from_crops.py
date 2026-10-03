"""Dense step for tiled inference: each crop's neural depth is placed with its parent photo's solved (RTK-locked)
camera, using intrinsics shifted to the crop, and scaled by the photogrammetric tie points it sees."""
import argparse, glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pycolmap
from common import Timer, dump, robust_sim3
from run_pipeline import load_telemetry
from telemetry import GeoFrame

ap = argparse.ArgumentParser()
ap.add_argument("--solved", required=True, help="rtk_photogrammetry run dir (sparse/0, frames/)")
ap.add_argument("--telemetry", required=True); ap.add_argument("--crops", required=True)
ap.add_argument("--predictions", required=True); ap.add_argument("--output", required=True)
ap.add_argument("--voxel", type=float, default=0.08); ap.add_argument("--depth-max", type=float, default=120.0)
ap.add_argument("--exchange-voxel", type=float, default=0.2)
ap.add_argument("--depth-field", action="store_true")
a = ap.parse_args(); os.makedirs(a.output, exist_ok=True); T = Timer()
tel = load_telemetry(a.telemetry); geo = GeoFrame(tel["lat"], tel["lon"], tel["alt"]); gl = geo.to_local(tel["lat"], tel["lon"], tel["alt"])
rec = pycolmap.Reconstruction(os.path.join(a.solved, "sparse", "0"))
C, G = [], []
for im in rec.images.values():
    if im.has_pose:
        i = int(im.name.split("_")[1].split(".")[0]); C.append(im.projection_center()); G.append(gl[i])
s, R, t = robust_sim3(np.array(C), np.array(G), np.full(len(C), 0.02))
rec.transform(pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t))
cam = rec.cameras[next(iter(rec.cameras))]
byname = {im.name: im for im in rec.images.values() if im.has_pose}
crops = json.load(open(os.path.join(a.crops, "crops.json")))
rep = json.load(open(os.path.join(a.predictions, "report.json")))
files = sorted(glob.glob(os.path.join(a.predictions, "predictions", "frame_*.npz")))
views, scales, c2ws = [], [], []
with T.stage("place crops"):
    for fpath, fr in zip(files, rep["frames"]):
        cn = fr.get("source_name"); m = crops.get(cn)
        im = byname.get(m["parent"]) if m else None
        if im is None: continue
        p = np.load(fpath); d0 = np.where(p["mask"], p["depth"], 0).astype(np.float32); h, w = d0.shape
        K = cam.calibration_matrix().copy(); K[0, :] *= m["W"] / cam.width; K[1, :] *= m["H"] / cam.height
        K[0, 2] -= m["ox"]; K[1, 2] -= m["oy"]; K[0, :] *= w / m["w"]; K[1, :] *= h / m["h"]
        cw = np.eye(4); cw[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix()
        pts = [rec.points3D[q.point3D_id].xyz for q in im.points2D if q.has_point3D()]
        if len(pts) < 20: continue
        X = np.array(pts) @ cw[:3, :3].T + cw[:3, 3]; z = X[:, 2]
        u = (K[0, 0] * X[:, 0] / z + K[0, 2]).astype(int); v = (K[1, 1] * X[:, 1] / z + K[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dn = d0[v[ok], u[ok]]; good = dn > 0
        if good.sum() < 15: continue
        sc = float(np.median(z[ok][good] / dn[good])); scales.append(sc)
        c2w = np.linalg.inv(cw); c2ws.append(c2w)
        if a.depth_field:
            from depthfix import correct_view
            dcorr, _ = correct_view(d0, K, cw, np.array(pts))
            views.append({"depth": dcorr, "color": p["color"], "K": K, "c2w": c2w})
        else:
            views.append({"depth": d0 * sc, "color": p["color"], "K": K, "c2w": c2w})
print("crop views", len(views), "scale median %.3f p5 %.3f p95 %.3f" % (np.median(scales), *np.percentile(scales, [5, 95])))
with T.stage("tiled GPU TSDF"):
    from fusion import gpu_tsdf_tiled, mesh_stats
    mesh, pcd = gpu_tsdf_tiled(views, tile=60.0, voxel=a.voxel, depth_max=a.depth_max); mesh.compute_vertex_normals()
with T.stage("exports"):
    from export import export_all
    P, Cc = np.asarray(pcd.points), np.asarray(pcd.colors)
    meta = {"ground_truth_used": False, "survey_or_lidar_used": False, "pipeline": "VoxelFlight v2 RTK + tiled inference"}
    export_all(a.output + "/model", mesh, P, Cc, geo.origin, geo.epsg, extra_meta=meta, formats=("ply", "las", "tif"))
    light = mesh.simplify_vertex_clustering(a.exchange_voxel); light.compute_vertex_normals()
    export_all(a.output + "/model", light, P[::4], Cc[::4], geo.origin, geo.epsg, extra_meta=meta, formats=("obj", "glb", "fbx"))
# evaluation helpers expect one pose per prediction file in inference/ order -> store parent poses per crop
np.savez(a.output + "/keyframe_poses.npz", camera_to_world=np.array(c2ws), utm_origin=geo.origin, epsg=geo.epsg)
dump(a.output + "/report.json", {"schema": "voxelflight.v2.rtk-tiled", "stage_seconds": T.stages, "crop_views": len(views),
     "mesh": mesh_stats(mesh), "utm_epsg": geo.epsg, "utm_origin": geo.origin.tolist(), "inference_dir": a.predictions,
     "ground_truth_used": False, "survey_or_lidar_used": False})
print("done", T.total())
