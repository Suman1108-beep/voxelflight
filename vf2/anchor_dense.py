"""Optional absolute anchoring: register the dense fused model to a public reference surface (e.g. national
LiDAR/DSM) — virtual ground control without field GCPs. Global robust point-to-plane ICP, then per-tile
refinement to absorb slow drift; camera poses are corrected by blending nearby tile transforms."""
import numpy as np
from scipy.spatial.transform import Rotation


def _icp(src, tgt, T0, schedule, iters=40):
    import open3d as o3d
    reg = o3d.pipelines.registration
    T = T0.copy(); info = None
    for d in schedule:
        est = reg.TransformationEstimationPointToPlane(reg.TukeyLoss(k=d))
        r = reg.registration_icp(src, tgt, d, T, est, reg.ICPConvergenceCriteria(max_iteration=iters))
        T, info = r.transformation, {"fitness": r.fitness, "inlier_rmse": r.inlier_rmse, "dmax": d}
    return T, info


def anchor(mesh_points, mesh_normals, ref_points, ref_normals, tile=60.0, min_pts=20000, max_tile_dev_m=3.0):
    import open3d as o3d
    pc = lambda P, N: (lambda c: (c.__setattr__("normals", o3d.utility.Vector3dVector(N)), c)[1])(o3d.geometry.PointCloud(o3d.utility.Vector3dVector(P)))
    tgt = pc(ref_points, ref_normals)
    src = pc(mesh_points, mesh_normals)
    Tg, gi = _icp(src, tgt, np.eye(4), (6.0, 4.0, 2.0, 1.0, 0.5))
    tiles = []
    if tile and tile > 0:
        lo = mesh_points[:, :2].min(0)
        ij = np.floor((mesh_points[:, :2] - lo) / tile).astype(int)
        for key in np.unique(ij, axis=0):
            m = np.all(ij == key, axis=1)
            if m.sum() < min_pts:
                continue
            Tt, ti = _icp(pc(mesh_points[m], mesh_normals[m]), tgt, Tg, (1.5, 1.0, 0.5))
            dev = np.linalg.norm(Tt[:3, 3] - Tg[:3, 3] + (Tt[:3, :3] - Tg[:3, :3]) @ mesh_points[m].mean(0))
            if dev < max_tile_dev_m and ti["fitness"] > 0.2:
                tiles.append({"centre": mesh_points[m].mean(0), "T": Tt, "n": int(m.sum()), **ti, "dev_from_global_m": float(dev)})
    return Tg, gi, tiles


def correct_poses(c2w, Tg, tiles, sigma=None, look_ahead=10.0):
    """Blend tile transforms (Gaussian in distance from each camera's look-at point); fall back to global."""
    out = np.array(c2w, float)
    if not tiles:
        for i in range(len(out)):
            out[i] = Tg @ out[i]
        return out
    C = np.stack([t["centre"] for t in tiles]); sigma = sigma or 40.0
    rots = Rotation.from_matrix(np.stack([t["T"][:3, :3] for t in tiles]))
    for i in range(len(out)):
        look = out[i, :3, 3] + look_ahead * out[i, :3, 2]
        d = np.linalg.norm(C[:, :2] - look[:2], axis=1)
        w = np.exp(-0.5 * (d / sigma) ** 2) * np.array([t["n"] for t in tiles])
        if w.sum() < 1e-6:
            out[i] = Tg @ out[i]; continue
        w = w / w.sum()
        R = rots.mean(w).as_matrix()
        # apply each tile transform to the camera centre and blend the results
        p = sum(wk * (t["T"][:3, :3] @ out[i, :3, 3] + t["T"][:3, 3]) for wk, t in zip(w, tiles))
        out[i, :3, :3] = R @ out[i, :3, :3]; out[i, :3, 3] = p
    return out
