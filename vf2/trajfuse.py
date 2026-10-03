"""Fuse visual odometry (relative, arbitrary scale) with GNSS position and barometric altitude.
No reference/survey data: global robust Sim(3) to GNSS, then a sparse least-squares smoother that lets
the GNSS correct slow visual drift while keeping visual step-to-step geometry."""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.spatial.transform import Rotation, Slerp

from common import robust_sim3, apply_sim3


def fuse(vo_xyz, vo_quat, gnss_xyz, gnss_sigma_h, baro_z=None, drift_per_m=0.05, sigma_baro=0.5, sigma_gnss_z=6.0, iters=3):
    """vo_* at N samples; gnss_xyz (N,3) local metres; baro_z (N,) absolute altitude or None.
    Returns fused positions (N,3), camera-to-world rotations (N,3,3), and diagnostics."""
    tgt = gnss_xyz.copy()
    if baro_z is not None:
        tgt[:, 2] = baro_z - np.median(baro_z - gnss_xyz[:, 2])  # baro shape, GNSS datum
        sz = np.full(len(tgt), sigma_baro)
    else:
        sz = np.full(len(tgt), sigma_gnss_z)
    s0, R0, t0 = robust_sim3(vo_xyz, tgt, gnss_sigma_h)
    X0 = apply_sim3(vo_xyz, s0, R0, t0)
    n = len(X0)
    d = np.diff(X0, axis=0)
    sv = drift_per_m * (np.linalg.norm(d, axis=1) + 0.01) + 0.005
    I = sp.identity(n, format="csr")
    D = sp.diags([-np.ones(n - 1), np.ones(n - 1)], [0, 1], shape=(n - 1, n))
    X = X0.copy(); w = np.ones(n)
    for _ in range(iters):
        for k in range(3):
            sg = gnss_sigma_h if k < 2 else sz
            Wg = sp.diags(np.sqrt(w) / sg); Wv = sp.diags(1.0 / sv)
            A = sp.vstack([Wg @ I, Wv @ D]).tocsr()
            b = np.concatenate([Wg @ tgt[:, k], Wv @ d[:, k]])
            X[:, k] = spla.lsqr(A, b, atol=1e-12, btol=1e-12, iter_lim=20000, x0=X[:, k])[0]
        r = np.linalg.norm((X - tgt)[:, :2], axis=1) / gnss_sigma_h
        w = np.where(r < 2.5, 1.0, 2.5 / r)
    Rwc = R0 @ Rotation.from_quat(vo_quat).as_matrix()
    diag = {"vo_to_gnss_scale": s0, "gnss_residual_rms_m": float(np.sqrt(np.mean(np.sum((X - tgt)[:, :2] ** 2, 1)))),
            "used_barometer": baro_z is not None, "drift_per_m": drift_per_m}
    return X, Rwc, diag


def interpolate_poses(sample_frames, X, Rwc, query_frames):
    """Linear position / slerp rotation interpolation at query video frames (clamped to the ends)."""
    q = np.clip(np.asarray(query_frames, float), sample_frames[0], sample_frames[-1])
    P = np.column_stack([np.interp(q, sample_frames, X[:, k]) for k in range(3)])
    R = Slerp(sample_frames.astype(float), Rotation.from_matrix(Rwc))(q).as_matrix()
    return P, R
