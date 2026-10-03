"""RTK/PPK mode: photogrammetric camera solution constrained by centimetre GNSS positions, then dense neural depth
rescaled per view to the solved geometry and fused on the GPU.

  frames from the video -> COLMAP SIFT features -> spatial matching (RTK neighbours)
  -> incremental SfM with position priors (RTK std as prior) -> similarity lock to RTK positions
  -> MapAnything depth per view, scaled by the median ratio to the SfM points it sees -> tiled GPU TSDF -> exports
Survey/reference data are never read."""
import argparse, glob, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import cv2
import pycolmap
from common import Timer, dump, umeyama, robust_sim3
from run_pipeline import load_telemetry
from telemetry import GeoFrame

ap = argparse.ArgumentParser()
ap.add_argument("--video", required=True); ap.add_argument("--telemetry", required=True)
ap.add_argument("--predictions", required=True, help="MapAnything prediction dir (inference/) for the same frames")
ap.add_argument("--output", required=True); ap.add_argument("--voxel", type=float, default=0.08)
ap.add_argument("--depth-max", type=float, default=120.0); ap.add_argument("--exchange-voxel", type=float, default=0.2)
ap.add_argument("--depth-field", action="store_true", help="per-view quadratic depth correction to tie points")
ap.add_argument("--keyframes-only", action="store_true", help="decode only the prediction keyframes")
ap.add_argument("--calibration", help="camera.json: undistort frames like the depth model inputs")
a = ap.parse_args()
os.makedirs(a.output, exist_ok=True)
T = Timer()
tel = load_telemetry(a.telemetry)
geo = GeoFrame(tel["lat"], tel["lon"], tel["alt"])
gl = geo.to_local(tel["lat"], tel["lon"], tel["alt"])
if tel.get("baro") is not None and np.isfinite(tel["baro"]).all():  # barometer shape, GNSS datum (as in run_pipeline)
    gl[:, 2] = tel["baro"] - np.median(tel["baro"] - tel["alt"]) - geo.origin[2]

SPARSE0 = os.path.join(a.output, "sparse", "0")
REUSE = os.path.exists(os.path.join(SPARSE0, "images.bin"))
if REUSE:
    names = sorted(os.listdir(os.path.join(a.output, "frames")))
img_dir = os.path.join(a.output, "frames")
with T.stage("frames") if not REUSE else T.stage("reuse solved model"):
    os.makedirs(img_dir, exist_ok=True)
    if not REUSE:
        und = None
        if a.calibration:  # undistort exactly like the depth model's inputs, keeping the same K
            cal = json.load(open(a.calibration)); und = (np.array(cal["intrinsic_matrix"]), np.array(cal.get("distortion_coefficients", [])))
        if a.keyframes_only:  # only the frames the depth predictions used (long videos)
            from features import decode_keyframes
            kf_list = [f["source_frame"] for f in json.load(open(os.path.join(a.predictions, "report.json")))["frames"]]
            imgs, _ = decode_keyframes(a.video, np.array(kf_list), workers=16)
            items = zip(kf_list, imgs)
        else:
            def _all():
                cap = cv2.VideoCapture(a.video); k = 0
                while True:
                    ok, frame = cap.read()
                    if not ok: return
                    yield k, frame; k += 1
            items = _all()
        names = []
        for k, frame in items:
            if und is not None and und[1].size:
                frame = cv2.undistort(frame, und[0], und[1])
            n = f"f_{k:05d}.jpg"; cv2.imwrite(os.path.join(img_dir, n), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]); names.append(n)
db = os.path.join(a.output, "colmap.db")
if os.path.exists(db) and not REUSE: os.remove(db)
if not REUSE:
 with T.stage("sift features"):
    eo = pycolmap.FeatureExtractionOptions() if hasattr(pycolmap, "FeatureExtractionOptions") else None
    ro = pycolmap.ImageReaderOptions(); ro.camera_model = "SIMPLE_RADIAL"
    kwargs = dict(camera_mode=pycolmap.CameraMode.SINGLE, reader_options=ro)
    if eo is not None:
        eo.num_threads = 32
        try: eo.sift.max_num_features = 8000
        except Exception: pass
        kwargs["extraction_options"] = eo
    pycolmap.extract_features(db, img_dir, **kwargs)
if not REUSE:
 with T.stage("pose priors"):
    d = pycolmap.Database.open(db) if hasattr(pycolmap.Database, "open") else pycolmap.Database(db)
    cov = None
    for im in d.read_all_images():
        i = int(im.name.split("_")[1].split(".")[0])
        pp = pycolmap.PosePrior()
        pp.position = gl[i]
        pp.coordinate_system = pycolmap.PosePriorCoordinateSystem.CARTESIAN
        s = max(float(tel["eph"][i]), 0.02)
        pp.position_covariance = np.diag([s * s, s * s, (2 * s) ** 2])
        if hasattr(pp, "corr_data_id"):
            pp.corr_data_id = im.data_id
        try:
            d.write_pose_prior(pp)
        except TypeError:
            d.write_pose_prior(im.image_id, pp)
    d.close()
if not REUSE:
 with T.stage("spatial matching"):
    so = pycolmap.SpatialPairingOptions() if hasattr(pycolmap, "SpatialPairingOptions") else None
    if so is not None:
        so.max_num_neighbors = 30; so.max_distance = 60.0
        try: so.ignore_z = False
        except Exception: pass
        pycolmap.match_spatial(db, pairing_options=so)
    else:
        pycolmap.match_spatial(db)
with T.stage("incremental SfM with RTK priors"):
    opts = pycolmap.IncrementalPipelineOptions()
    opts.use_prior_position = True
    opts.num_threads = 64
    opts.multiple_models = False
    if REUSE:
        rec = pycolmap.Reconstruction(SPARSE0)
    else:
        recs = pycolmap.incremental_mapping(db, img_dir, os.path.join(a.output, "sparse"), opts)
        rec = max(recs.values(), key=lambda r: r.num_reg_images())
print("registered", rec.num_reg_images(), "of", len(names), "points", rec.num_points3D(), "reproj", rec.compute_mean_reprojection_error())

# lock to RTK positions with a (robust) similarity — removes any residual gauge/scale freedom
ids, C, G = [], [], []
for iid, im in rec.images.items():
    if im.has_pose:
        i = int(im.name.split("_")[1].split(".")[0]); ids.append(i); C.append(im.projection_center()); G.append(gl[i])
C, G = np.array(C), np.array(G)
s, R, t = robust_sim3(C, G, np.full(len(C), 0.02))
res_rtk = np.linalg.norm(s * C @ R.T + t - G, axis=1)
print("SfM->RTK similarity scale %.4f, camera residual median %.3f m p95 %.3f m" % (s, np.median(res_rtk), np.percentile(res_rtk, 95)))
Sim = pycolmap.Sim3d(s, pycolmap.Rotation3d(R), t)
rec.transform(Sim)

with T.stage("dense: scaled neural depth + tiled GPU TSDF"):
    from fusion import gpu_tsdf_tiled, mesh_stats
    rep = json.load(open(os.path.join(a.predictions, "report.json")))
    files = sorted(glob.glob(os.path.join(a.predictions, "predictions", "frame_*.npz")))
    kf = [f["source_frame"] for f in rep["frames"]]
    cam = rec.cameras[next(iter(rec.cameras))]
    byname = {im.name: im for im in rec.images.values() if im.has_pose}
    views, scales, c2ws, fits, vframes = [], [], [], [], []
    for fpath, sf in zip(files, kf):
        im = byname.get(f"f_{sf:05d}.jpg")
        if im is None: continue
        p = np.load(fpath); d0 = np.where(p["mask"], p["depth"], 0).astype(np.float32); h, w = d0.shape
        Ksfm = cam.calibration_matrix().copy(); Ksfm[0] *= w / cam.width; Ksfm[1] *= h / cam.height
        cw = np.eye(4); cw[:3, :4] = (im.cam_from_world() if callable(im.cam_from_world) else im.cam_from_world).matrix()
        Rcw, tcw = cw[:3, :3], cw[:3, 3]
        # SfM points seen in this image -> their depth vs neural depth at the same pixel
        pts = [rec.points3D[p2.point3D_id].xyz for p2 in im.points2D if p2.has_point3D()]
        if len(pts) < 30: continue
        X = np.array(pts) @ Rcw.T + tcw; z = X[:, 2]
        u = (Ksfm[0, 0] * X[:, 0] / z + Ksfm[0, 2]).astype(int); v = (Ksfm[1, 1] * X[:, 1] / z + Ksfm[1, 2]).astype(int)
        ok = (z > 0) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
        dn = d0[v[ok], u[ok]]; good = dn > 0
        if good.sum() < 20: continue
        sc = float(np.median(z[ok][good] / dn[good])); scales.append(sc)
        c2w = np.linalg.inv(cw); c2ws.append(c2w); vframes.append(sf)
        if a.depth_field:
            from depthfix import correct_view
            dcorr, finfo = correct_view(d0, Ksfm, cw, np.array(pts)); fits.append(finfo)
            views.append({"depth": dcorr, "color": p["color"], "K": Ksfm, "c2w": c2w})
        else:
            views.append({"depth": d0 * sc, "color": p["color"], "K": Ksfm, "c2w": c2w})
    if fits:
        mb = [f["mad_before"] for f in fits if "mad_before" in f]; ma = [f["mad_after"] for f in fits if "mad_after" in f]
        print("depth field: views fitted %d, log-residual MAD median before %.3f after %.3f" % (len(ma), np.median(mb), np.median(ma)))
    print("views", len(views), "depth scale median %.3f p5 %.3f p95 %.3f" % (np.median(scales), *np.percentile(scales, [5, 95])))
    mesh, pcd = gpu_tsdf_tiled(views, tile=60.0, voxel=a.voxel, depth_max=a.depth_max)
    mesh.compute_vertex_normals()
with T.stage("exports"):
    from export import export_all
    P, Cc = np.asarray(pcd.points), np.asarray(pcd.colors)
    meta = {"ground_truth_used": False, "survey_or_lidar_used": False, "pipeline": "VoxelFlight v2 RTK photogrammetry"}
    full = export_all(a.output + "/model", mesh, P, Cc, geo.origin, geo.epsg, extra_meta=meta, formats=("ply", "las", "tif"))
    light = mesh.simplify_vertex_clustering(a.exchange_voxel); light.compute_vertex_normals()
    exch = export_all(a.output + "/model", light, P[::4], Cc[::4], geo.origin, geo.epsg, extra_meta=meta, formats=("obj", "glb", "fbx"))
np.savez(a.output + "/keyframe_poses.npz", camera_to_world=np.array(c2ws), video_frame=np.array(vframes), utm_origin=geo.origin, epsg=geo.epsg)
os.makedirs(a.output + "/inference", exist_ok=True)
dump(a.output + "/report.json", {"schema": "voxelflight.v2.rtk", "processing_wall_seconds": T.total(), "stage_seconds": T.stages,
     "registered_images": rec.num_reg_images(), "frames": len(names), "sfm_points": rec.num_points3D(),
     "sfm_to_rtk_scale": s, "camera_vs_rtk_residual_median_m": float(np.median(res_rtk)), "camera_vs_rtk_residual_p95_m": float(np.percentile(res_rtk, 95)),
     "mesh": mesh_stats(mesh), "utm_epsg": geo.epsg, "utm_origin": geo.origin.tolist(), "inference_dir": a.predictions,
     "ground_truth_used": False, "survey_or_lidar_used": False})
print("done", T.total())
