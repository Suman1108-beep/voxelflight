"""Place per-view neural depth predictions onto the fused (metric, georeferenced) trajectory.
Each view gets a local similarity estimated from its +-K neighbours: rotation by averaging
R_fused R_pred^T, then scale/translation from camera centres given that rotation."""
import numpy as np
from scipy.spatial.transform import Rotation


def ghost_gate(depths, Ks, c2ws, stride=4, rel=0.4, margin=3.0, max_low_frac=0.02):
    """Reject mis-scaled/mis-placed views that create 'ghost' copies below the ground.
    For each view, how far below its camera does its lowest geometry reach (3rd percentile)? The flight's typical
    value d_g (median over views) is the ground clearance; anything deeper than d_g*(1+rel)+margin is implausible.
    Returns (keep mask, floor depth below camera, per-view low fraction)."""
    below = np.full(len(depths), np.nan); pts_z = []
    for i, (d, K, T) in enumerate(zip(depths, Ks, c2ws)):
        dd = d[::stride, ::stride]; ys, xs = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        ok = dd > 0
        if ok.sum() < 50:
            pts_z.append(None); continue
        X = np.stack([(xs[ok] - K[0, 2]) / K[0, 0] * dd[ok], (ys[ok] - K[1, 2]) / K[1, 1] * dd[ok], dd[ok]], 1)
        z = X @ T[:3, :3].T[:, 2] + T[2, 3]
        rel_z = T[2, 3] - z  # metres below the camera
        below[i] = np.percentile(rel_z, 97); pts_z.append(rel_z)
    d_g = float(np.nanmedian(below)); floor = d_g * (1 + rel) + margin
    low = np.array([np.mean(r > floor) if r is not None else 0.0 for r in pts_z])
    keep = low <= max_low_frac
    return keep, floor, low


def snap_views(pred_c2w, fused_C, fused_R, half_window=20, pin_positions=False):
    """pin_positions=True (centimetre GNSS such as RTK/PPK): camera centres are set exactly to the fused
    positions; only orientation and depth scale come from the neural prediction."""
    n = len(pred_c2w)
    Rp = pred_c2w[:, :3, :3]; Cp = pred_c2w[:, :3, 3]
    out = np.zeros_like(pred_c2w); scales = np.zeros(n); resid = np.zeros(n)
    rel = Rotation.from_matrix(fused_R @ np.transpose(Rp, (0, 2, 1)))
    # global scale first: local scales are only allowed to deviate moderately from it
    Rg = rel.mean().as_matrix(); a = Cp @ Rg.T
    s_global = float(np.sum((fused_C - fused_C.mean(0)) * (a - a.mean(0))) / max(np.sum((a - a.mean(0)) ** 2), 1e-9))
    for i in range(n):
        j = np.arange(max(0, i - half_window), min(n, i + half_window + 1))
        Rw = rel[j].mean().as_matrix()
        a = Cp[j] @ Rw.T; b = fused_C[j]
        am, bm = a.mean(0), b.mean(0)
        s = float(np.sum((b - bm) * (a - am)) / max(np.sum((a - am) ** 2), 1e-9))
        s = float(np.clip(s, 0.7 * s_global, 1.4 * s_global))
        t = bm - s * am
        out[i] = np.eye(4)
        out[i, :3, :3] = Rw @ Rp[i]
        out[i, :3, 3] = fused_C[i] if pin_positions else s * (Rw @ Cp[i]) + t
        scales[i] = s
        resid[i] = np.linalg.norm(out[i, :3, 3] - fused_C[i])
    return out, scales, resid
