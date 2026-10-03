"""Chunk-wise Sim(3) anchoring of a visual reconstruction to a public reference surface (point-to-plane ICP)."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def _skew(v):
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def icp_sim3(src, ref_tree, ref_pts, ref_nrm, centre, prior_sigma_t=3.0, prior_sigma_s=0.05,
             schedule=(4.0, 3.0, 2.0, 1.5, 1.0, 0.7, 0.5), iters_per=4, huber=0.15):
    """Estimate x' = s R (x - c) + c + t aligning src to the reference surface.
    Priors keep t ~ 0 (sigma prior_sigma_t) and log s ~ 0, so unconstrained directions stay at the GPS solution."""
    s, R, t = 1.0, np.eye(3), np.zeros(3)
    info = {}
    for dmax in schedule:
        for _ in range(iters_per):
            x = s * (src - centre) @ R.T + centre + t
            d, j = ref_tree.query(x, distance_upper_bound=dmax)
            ok = np.isfinite(d)
            if ok.sum() < 30:
                break
            q, n, xo = ref_pts[j[ok]], ref_nrm[j[ok]], x[ok]
            r = np.einsum("ij,ij->i", xo - q, n)
            # Jacobian wrt (dtheta, dt, dlogs) of n·(x)
            y = xo - centre - t
            J = np.column_stack([np.cross(y, n), n, np.einsum("ij,ij->i", y, n)])
            w = np.where(np.abs(r) < huber, 1.0, huber / np.abs(r))
            A = J.T @ (w[:, None] * J); b = -J.T @ (w * r)
            # priors (in the same units: residual metres^2 vs prior)
            npt = max(1.0, ok.sum() / 200.0)  # treat correlated points as ~200x fewer independent samples
            A /= npt; b /= npt
            P = np.diag([1e-6, 1e-6, 1e-6, 1 / prior_sigma_t**2, 1 / prior_sigma_t**2, 1 / prior_sigma_t**2, 1 / prior_sigma_s**2])
            A = A / (0.1 ** 2) + P
            b = b / (0.1 ** 2) - P @ np.concatenate([np.zeros(3), t, [np.log(s)]])
            dx = np.linalg.solve(A, b)
            dR = Rotation.from_rotvec(dx[:3]).as_matrix()
            R = dR @ R; t = t + dx[3:6]  # perturbation acts about the current chunk centre (c + t)
            s = s * np.exp(dx[6])
            info = {"inliers": int(ok.sum()), "rms_plane_m": float(np.sqrt(np.mean(r[np.abs(r) < dmax] ** 2))), "dmax": dmax}
    return s, R, t, info


def blend_chunk_transforms(times, chunks):
    """chunks: list of (t_start, t_end, centre, s, R, t). Triangular-weighted blend per time -> per-time (s, R, t, centre)."""
    out = []
    for tt in times:
        ws, items = [], []
        for (a, b, c, s, R, t) in chunks:
            if a <= tt <= b:
                w = 1.0 - abs(tt - 0.5 * (a + b)) / (0.5 * (b - a) + 1e-9) + 1e-3
                ws.append(w); items.append((c, s, R, t))
        if not items:
            # nearest chunk
            k = int(np.argmin([min(abs(tt - a), abs(tt - b)) for (a, b, *_ ) in chunks]))
            a, b, c, s, R, t = chunks[k]; ws, items = [1.0], [(c, s, R, t)]
        ws = np.array(ws) / np.sum(ws)
        rot = Rotation.from_matrix(np.stack([it[2] for it in items])).mean(ws).as_matrix()
        out.append((sum(w * it[0] for w, it in zip(ws, items)), float(np.exp(sum(w * np.log(it[1]) for w, it in zip(ws, items)))),
                    rot, sum(w * it[3] for w, it in zip(ws, items))))
    return out


def apply_point(x, c, s, R, t):
    return s * (x - c) @ R.T + c + t
