"""Per-view depth correction of neural depth maps against photogrammetric tie points.
A single scale per view cannot remove the smooth bending of monocular/feed-forward depth on oblique views.
Fit log(z_true / z_pred) as a robust quadratic in normalised image coordinates and apply it as a scale field."""
import numpy as np


def fit_scale_field(u, v, z_true, z_pred, w, h, min_pts=40, max_dev=0.35, iters=6, delta=0.04):
    """Return (coeffs, info). coeffs map (x, y) in [-1, 1] -> log-scale; falls back to a constant when sparse."""
    ok = (z_true > 0) & (z_pred > 0)
    u, v, r = u[ok], v[ok], np.log(z_true[ok] / z_pred[ok])
    if len(r) < 10:
        return None, {"n": int(len(r)), "mode": "none"}
    x, y = 2 * u / w - 1, 2 * v / h - 1
    if len(r) < min_pts:
        c = np.zeros(6); c[0] = np.median(r)
        return c, {"n": int(len(r)), "mode": "constant", "mad_before": float(np.median(np.abs(r - np.median(r))))}
    A = np.column_stack([np.ones_like(x), x, y, x * x, x * y, y * y])
    wts = np.ones(len(r)); c = np.zeros(6); c[0] = np.median(r)
    for _ in range(iters):
        sw = np.sqrt(wts)
        c = np.linalg.lstsq(A * sw[:, None], r * sw, rcond=None)[0]
        res = r - A @ c
        wts = np.where(np.abs(res) < delta, 1.0, delta / np.abs(res))
    res = r - A @ c
    info = {"n": int(len(r)), "mode": "quadratic",
            "mad_before": float(np.median(np.abs(r - np.median(r)))), "mad_after": float(np.median(np.abs(res)))}
    c[1:] = np.clip(c[1:], -max_dev, max_dev)
    return c, info


def apply_scale_field(depth, coeffs, max_dev=0.35):
    if coeffs is None:
        return depth
    h, w = depth.shape
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    x, y = 2 * (xs + 0.5) / w - 1, 2 * (ys + 0.5) / h - 1
    field = coeffs[1] * x + coeffs[2] * y + coeffs[3] * x * x + coeffs[4] * x * y + coeffs[5] * y * y
    field = np.clip(field, -max_dev, max_dev) + coeffs[0]
    return (depth * np.exp(field)).astype(np.float32)


def correct_view(d0, K, cw, points_world, max_dev=0.35):
    """d0: neural depth (HxW, unscaled); K: intrinsics at d0 resolution; cw: 4x4 world->camera;
    points_world: (N,3) tie points observed in this view. Returns corrected depth and fit info."""
    h, w = d0.shape
    X = points_world @ cw[:3, :3].T + cw[:3, 3]; z = X[:, 2]
    front = z > 0
    u = K[0, 0] * X[front, 0] / z[front] + K[0, 2]; v = K[1, 1] * X[front, 1] / z[front] + K[1, 2]
    zt = z[front]
    inb = (u >= 0) & (u < w - 1) & (v >= 0) & (v < h - 1)
    u, v, zt = u[inb], v[inb], zt[inb]
    zp = d0[v.astype(int), u.astype(int)]
    c, info = fit_scale_field(u, v, zt, zp, w, h, max_dev=max_dev)
    return apply_scale_field(d0, c, max_dev), info
