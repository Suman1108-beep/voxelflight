"""Track triangulation from initial poses and global COLMAP bundle adjustment."""
import os

import numpy as np
import torch
from scipy.spatial.transform import Rotation


def triangulate_tracks(tracks, kps_ud, K, R_cw, t_cw, max_reproj=6.0, min_angle_deg=1.0, device="cuda"):
    """Batched multi-view DLT on GPU. R_cw/t_cw: (N,3,3)/(N,3) world->camera."""
    P = torch.from_numpy(np.einsum("ij,njk->nik", K, np.concatenate([R_cw, t_cw[:, :, None]], 2))).to(device)
    L = max(len(t) for t in tracks)
    T = len(tracks)
    im = np.zeros((T, L), np.int64); uv = np.zeros((T, L, 2)); valid = np.zeros((T, L), bool)
    for k, tr in enumerate(tracks):
        n = len(tr)
        im[k, :n] = tr[:, 0]; valid[k, :n] = True
        uv[k, :n] = kps_ud_lookup(kps_ud, tr)
    im_t = torch.from_numpy(im).to(device); uv_t = torch.from_numpy(uv).to(device); v_t = torch.from_numpy(valid).to(device)
    Pk = P[im_t]  # T,L,3,4
    A1 = uv_t[..., 0:1] * Pk[..., 2, :] - Pk[..., 0, :]
    A2 = uv_t[..., 1:2] * Pk[..., 2, :] - Pk[..., 1, :]
    A = torch.stack([A1, A2], 2).reshape(T, 2 * L, 4)
    A = A / (A.norm(dim=-1, keepdim=True) + 1e-12)
    A = A * v_t.repeat_interleave(2, 1)[..., None]
    out = []
    for s in range(0, T, 200000):
        _, _, Vh = torch.linalg.svd(A[s:s + 200000])
        out.append(Vh[:, -1])
    X = torch.cat(out)
    X = X[:, :3] / X[:, 3:4]
    # reprojection + cheirality
    Xh = torch.cat([X, torch.ones_like(X[:, :1])], 1)
    proj = torch.einsum("tlij,tj->tli", Pk, Xh)
    z = proj[..., 2]
    pix = proj[..., :2] / z[..., None].clamp(min=1e-9)
    err = (pix - uv_t).norm(dim=-1)
    ok_obs = v_t & (z > 0.1) & (err < max_reproj)
    # triangulation angle: max angle between rays from camera centres
    C = torch.from_numpy(-np.einsum("nji,nj->ni", R_cw, t_cw)).to(device)[im_t]
    rays = X[:, None, :] - C
    rays = rays / rays.norm(dim=-1, keepdim=True)
    rays = rays * ok_obs[..., None]
    first = rays[:, 0:1]
    cosang = (rays * first).sum(-1).masked_fill(~ok_obs, 1.0).min(1).values
    ang = torch.rad2deg(torch.arccos(cosang.clamp(-1, 1)))
    keep = (ok_obs.sum(1) >= 2) & (ok_obs.sum(1) == v_t.sum(1)) & (ang > min_angle_deg) & torch.isfinite(X).all(1)
    keep = keep.cpu().numpy()
    return X.cpu().numpy(), keep


def kps_ud_lookup(kps_ud, tr):
    return np.stack([kps_ud[i][k] for i, k in tr])


def write_colmap_text(out_dir, K, size, names, R_cw, t_cw, tracks, X, keep, kps_ud):
    os.makedirs(out_dir, exist_ok=True)
    W, H = size
    with open(os.path.join(out_dir, "cameras.txt"), "w") as f:
        f.write(f"1 PINHOLE {W} {H} {K[0,0]!r} {K[1,1]!r} {K[0,2]!r} {K[1,2]!r}\n")
    obs = [[] for _ in names]  # per image: list of (x, y, pid)
    pts = []
    pid = 0
    for k in np.where(keep)[0]:
        pid += 1
        tr = tracks[k]
        el = []
        for i, kp_idx in tr:
            xy = kps_ud[i][kp_idx]
            el.append((i + 1, len(obs[i])))
            obs[i].append((xy[0], xy[1], pid))
        pts.append((pid, X[k], el))
    with open(os.path.join(out_dir, "images.txt"), "w") as f:
        for i, n in enumerate(names):
            q = Rotation.from_matrix(R_cw[i]).as_quat()  # x y z w
            t = t_cw[i]
            f.write(f"{i+1} {q[3]!r} {q[0]!r} {q[1]!r} {q[2]!r} {t[0]!r} {t[1]!r} {t[2]!r} 1 {n}\n")
            f.write(" ".join(f"{x:.3f} {y:.3f} {p}" for x, y, p in obs[i]) + "\n")
    with open(os.path.join(out_dir, "points3D.txt"), "w") as f:
        for p, x, el in pts:
            f.write(f"{p} {x[0]!r} {x[1]!r} {x[2]!r} 128 128 128 0 " + " ".join(f"{a} {b}" for a, b in el) + "\n")
    return pid


def run_bundle_adjustment(rec_dir, iterations=60, loss_scale=1.0, threads=64, use_gpu=True):
    import pycolmap
    rec = pycolmap.Reconstruction(rec_dir)
    opts = pycolmap.BundleAdjustmentOptions()
    opts.refine_focal_length = False
    opts.refine_principal_point = False
    opts.refine_extra_params = False
    c = opts.ceres
    try:
        c.loss_function_type = pycolmap.LossFunctionType.CAUCHY
    except Exception as e:  # pragma: no cover - version differences
        print("loss type not set:", e)
    c.loss_function_scale = loss_scale
    c.solver_options.max_num_iterations = iterations
    c.solver_options.num_threads = threads
    c.use_gpu = use_gpu
    opts.print_summary = True
    pycolmap.bundle_adjustment(rec, opts)
    return rec


def filter_and_rerun(rec, max_err=3.0, **kw):
    import pycolmap
    rec.update_point_3d_errors()
    bad = [pid for pid, p in rec.points3D.items() if p.error > max_err]
    for pid in bad:
        rec.delete_point3D(pid)
    return bad


def camera_centres(rec, n):
    C = np.full((n, 3), np.nan)
    for iid, im in rec.images.items():
        if im.has_pose:
            C[iid - 1] = im.projection_center()
    return C
