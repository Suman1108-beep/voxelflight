"""VoxelFlight v2: single-pass drone video + GNSS (+ optional barometer) -> georeferenced 3-D model.

Input contract
  --video        MP4/MOV from one UAV pass
  --telemetry    CSV, one row per video frame: source_frame,timestamp_s,latitude,longitude,altitude_m
                 optional columns: gps_eph_m (horizontal accuracy), baro_alt_m (barometric altitude)
                 (a DJI .SRT, GPX, flight-log CSV or JSON is converted first by ingest_telemetry.py -> <output>/telemetry_v2.csv)
  --calibration  JSON {intrinsic_matrix, distortion_coefficients, width, height} (optional camera intrinsics)
No survey, reference-pose or LiDAR data are read. Stage wall times are recorded from video-on-disk to final exports.
"""
import argparse, glob, json, os, subprocess, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from common import Timer, dump
from ingest_telemetry import ensure_pipeline_telemetry
from ingest_video import ensure_calibration, ensure_decodable
from telemetry import GeoFrame

ROOT = "/workspace/voxelflight_a100_20260928"
PY = ROOT + "/.venv/bin/python"
DPVO_PY = ROOT + "/thirdparty/dpvo-env/bin/python"
DPVO_DIR = ROOT + "/thirdparty/DPVO"
LDP = ROOT + "/runtime-packages/root/usr/lib/x86_64-linux-gnu"


def load_telemetry(path):
    import csv
    rows = list(csv.DictReader(open(path)))
    col = lambda k, d=None: np.array([float(r[k]) if r.get(k) not in (None, "") else np.nan for r in rows]) if k in rows[0] else d
    out = {"frame": col("source_frame").astype(int), "lat": col("latitude"), "lon": col("longitude"), "alt": col("altitude_m")}
    out["eph"] = col("gps_eph_m", np.full(len(rows), 5.0))
    out["baro"] = col("baro_alt_m")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True); ap.add_argument("--telemetry", required=True); ap.add_argument("--calibration")
    ap.add_argument("--output", required=True)
    ap.add_argument("--keyframes", type=int, default=360); ap.add_argument("--size", type=int, default=518)
    ap.add_argument("--dpvo-stride", type=int, default=5); ap.add_argument("--voxel", type=float, default=0.05)
    ap.add_argument("--depth-max", type=float, default=25.0); ap.add_argument("--exchange-voxel", type=float, default=0.10); ap.add_argument("--tile", type=float, default=60.0); ap.add_argument("--confidence-percentile", type=float, default=20.0)
    ap.add_argument("--no-ghost-gate", action="store_true", help="disable the below-ground view consistency check")
    ap.add_argument("--vo", choices=["dpvo", "mapanything"], default="dpvo", help="visual odometry source (mapanything for sparse stills / low frame rate)")
    ap.add_argument("--reuse-inference", help="DEV ONLY: existing prediction dir (timing then not end-to-end)")
    ap.add_argument("--reuse-dpvo", help="DEV ONLY: existing DPVO trajectory txt")
    a = ap.parse_args()
    os.makedirs(a.output, exist_ok=True)
    T = Timer()
    a.video = ensure_decodable(a.video, a.output)  # HEVC etc. -> H.264 when OpenCV cannot read it
    a.telemetry = ensure_pipeline_telemetry(a.telemetry, a.video, a.output)  # SRT / GPX / other CSV -> per-frame CSV
    a.calibration = ensure_calibration(a.calibration, a.video, a.output)  # estimated intrinsics when none are supplied
    env = dict(os.environ, HF_HOME=ROOT + "/cache/huggingface", TORCH_HOME=ROOT + "/cache/torch", DINO_SOURCE=ROOT + "/dinov2",
               HF_HUB_OFFLINE="1", OMP_NUM_THREADS="8", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True", LD_LIBRARY_PATH=LDP)
    cam = json.load(open(a.calibration)) if a.calibration else None
    run_name = os.path.basename(os.path.normpath(a.output))

    # ---- Stage 1: visual odometry and neural depth, concurrently on the GPU ----
    with T.stage("dpvo+mapanything (parallel)"):
        procs = {}
        traj_path = a.reuse_dpvo or os.path.join(DPVO_DIR, "saved_trajectories", f"{run_name}.txt")
        if not a.reuse_dpvo and a.vo == "dpvo":
            calib = os.path.join(a.output, "dpvo_calib.txt")
            K = np.array(cam["intrinsic_matrix"]); dco = cam.get("distortion_coefficients", [])
            np.savetxt(calib, [[K[0, 0], K[1, 1], K[0, 2], K[1, 2], *dco]], fmt="%.10f")
            procs["dpvo"] = (subprocess.Popen([DPVO_PY, "demo.py", "--imagedir", a.video, "--calib", calib, "--stride", str(a.dpvo_stride),
                                               "--name", run_name, "--save_trajectory"], cwd=DPVO_DIR, env=env,
                                              stdout=open(a.output + "/dpvo.log", "w"), stderr=subprocess.STDOUT), time.time())
        pred_dir = a.reuse_inference or os.path.join(a.output, "inference")
        if not a.reuse_inference:
            cmd = [PY, "-u", ROOT + "/repo/vf2_mapanything_infer.py", "--predictions-only", "--video", a.video,
                   "--max-frames", str(a.keyframes), "--size", str(a.size), "--window", "48", "--overlap", "8",
                   "--save-predictions", "--confidence-percentile", str(a.confidence_percentile), "--output", pred_dir]
            if a.calibration:
                cmd += ["--calibration", a.calibration]
            procs["mapanything"] = (subprocess.Popen(cmd, env=env, stdout=open(a.output + "/mapanything.log", "w"), stderr=subprocess.STDOUT), time.time())
        for k, (p, t0) in procs.items():
            if p.wait() != 0:
                raise RuntimeError(f"{k} failed, see {a.output}/{k}.log")
            T.stages[f"  {k}"] = time.time() - t0

    # ---- Stage 2: GNSS + barometer + VO trajectory fusion (metric, georeferenced) ----
    with T.stage("trajectory fusion"):
        from trajfuse import fuse, interpolate_poses
        tel = load_telemetry(a.telemetry)
        geo = GeoFrame(tel["lat"], tel["lon"], tel["alt"])
        g_local = geo.to_local(tel["lat"], tel["lon"], tel["alt"])
        if a.vo == "dpvo":
            tr = np.loadtxt(traj_path)
            vf = tr[:, 0].astype(int) * a.dpvo_stride + (a.dpvo_stride - 1)
        else:  # neural-network camera poses as the relative visual trajectory
            from scipy.spatial.transform import Rotation as _R
            _rep = json.load(open(os.path.join(pred_dir, "report.json")))
            _P = np.stack([np.load(f)["pose"] for f in sorted(glob.glob(os.path.join(pred_dir, "predictions", "frame_*.npz")))]).astype(float)
            vf = np.array([f["source_frame"] for f in _rep["frames"]])
            tr = np.column_stack([np.arange(len(vf)), _P[:, :3, 3], _R.from_matrix(_P[:, :3, :3]).as_quat()])
        idx = np.searchsorted(tel["frame"], vf).clip(0, len(tel["frame"]) - 1)
        baro = tel["baro"][idx] if tel["baro"] is not None and np.isfinite(tel["baro"]).all() else None
        Xf, Rf, fdiag = fuse(tr[:, 1:4], tr[:, 4:8], g_local[idx], np.maximum(tel["eph"][idx], 0.03) / 3.0, baro)  # eph floor 3 cm so RTK/PPK accuracy is used, not discarded
        np.savez(a.output + "/fused_trajectory.npz", video_frame=vf, position=Xf, rotation=Rf, utm_origin=geo.origin, epsg=geo.epsg)

    # ---- Stage 3: snap neural views onto the fused trajectory ----
    with T.stage("snap views"):
        from snap import snap_views
        rep = json.load(open(os.path.join(pred_dir, "report.json")))
        files = sorted(glob.glob(os.path.join(pred_dir, "predictions", "frame_*.npz")))
        kf = np.array([f["source_frame"] for f in rep["frames"]])
        preds = [np.load(f) for f in files]
        Pc2w = np.stack([p["pose"] for p in preds]).astype(np.float64)
        Cq, Rq = interpolate_poses(vf, Xf, Rf, kf)
        pin = bool(np.nanmedian(tel["eph"]) < 0.2)  # RTK/PPK-grade positions: pin camera centres to them
        new_c2w, scales, resid = snap_views(Pc2w, Cq, Rq, pin_positions=pin)
        np.savez(a.output + "/keyframe_poses.npz", video_frame=kf, camera_to_world=new_c2w, utm_origin=geo.origin, epsg=geo.epsg)

    # ---- Stage 4: GPU TSDF fusion ----
    with T.stage("gpu tsdf fusion"):
        from fusion import gpu_tsdf_tiled, clean_mesh, mesh_stats
        from snap import ghost_gate
        if a.no_ghost_gate:
            keep, floor, low = np.ones(len(preds), bool), np.inf, np.zeros(len(preds))
        else:
            keep, floor, low = ghost_gate([np.where(p["mask"], p["depth"] * s, 0) for p, s in zip(preds, scales)],
                                          [p["intrinsics"] for p in preds], new_c2w)
        print(f"ghost gate: floor {floor:.1f} m below camera, dropped {int((~keep).sum())}/{len(keep)} views", flush=True)
        def views():
            for p, c2w, s, k in zip(preds, new_c2w, scales, keep):
                if not k:
                    continue
                d = np.where(p["mask"], p["depth"] * s, 0).astype(np.float32)
                if np.isfinite(floor):  # clip stray points deeper than the plausible ground below this camera
                    K = p["intrinsics"]; H, W = d.shape
                    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
                    zc = c2w[2, 0] * (xs - K[0, 2]) / K[0, 0] * d + c2w[2, 1] * (ys - K[1, 2]) / K[1, 1] * d + c2w[2, 2] * d
                    d = np.where(-zc > floor, 0, d).astype(np.float32)
                yield {"depth": d, "color": p["color"], "K": p["intrinsics"], "c2w": c2w}
        mesh, pcd = gpu_tsdf_tiled(views(), tile=a.tile, voxel=a.voxel, depth_max=a.depth_max)
        raw_stats = None
        mesh.compute_vertex_normals()
        stats = {"triangles": int(len(mesh.triangles)), "vertices": int(len(mesh.vertices))}

    # ---- Stage 5: exports ----
    with T.stage("exports"):
        from export import export_all
        import open3d as o3d
        P = np.asarray(pcd.points); C = np.asarray(pcd.colors)
        meta_common = {"ground_truth_used": False, "survey_or_lidar_used": False, "pipeline": "VoxelFlight v2"}
        full = export_all(a.output + "/model", mesh, P, C, geo.origin, geo.epsg, extra_meta=meta_common, formats=("ply", "las", "tif"))
        light = mesh.simplify_vertex_clustering(a.exchange_voxel)
        light.compute_vertex_normals()
        exch = export_all(a.output + "/model", light, P[::4], C[::4], geo.origin, geo.epsg, extra_meta=meta_common, formats=("obj", "glb", "fbx"))

    total = T.total()
    report = {"schema": "voxelflight.v2.run", "video": a.video, "telemetry": a.telemetry, "calibration": a.calibration,
              "reused_dev_stages": {"inference": bool(a.reuse_inference), "dpvo": bool(a.reuse_dpvo)},
              "continuous_end_to_end": not (a.reuse_inference or a.reuse_dpvo),
              "processing_wall_seconds": total, "within_15_min": total < 900, "stage_seconds": T.stages,
              "keyframes": len(kf), "dpvo_samples": len(vf), "trajectory_fusion": fdiag,
              "snap_scale_median": float(np.median(scales)), "snap_scale_p5_p95": [float(np.percentile(scales, 5)), float(np.percentile(scales, 95))],
              "snap_centre_residual_median_m": float(np.median(resid)), "positions_pinned_to_gnss": pin, "ghost_gate": {"floor_m_below_camera": float(floor), "dropped_views": int((~keep).sum()), "views": int(len(keep))},
              "mesh_raw": raw_stats, "mesh": stats, "exchange_mesh_triangles": int(len(light.triangles)),
              "utm_epsg": geo.epsg, "utm_origin": geo.origin.tolist(), "dsm": full.get("dsm"),
              "artifacts": {**full["artifacts"], **exch["artifacts"]},
              "ground_truth_used": False, "survey_or_lidar_used": False}
    dump(a.output + "/report.json", report)
    print(json.dumps({k: v for k, v in report.items() if k != "artifacts"}, indent=2))


if __name__ == "__main__":
    main()
