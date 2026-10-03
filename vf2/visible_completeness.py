"""Visibility-aware completeness ("entire visible scene"): which reference LiDAR points are actually seen by
the flight's cameras (in FOV, within range, not occluded — z-buffer splatting), and what fraction of those the
model reconstructs. Optionally after an evaluation-only rigid alignment (to separate shape from GNSS placement)."""
import numpy as np
import torch


def visible_mask(ref, c2w, K, W, H, depth_max=35.0, scale=0.125, tol_rel=0.03, tol_abs=0.4, device="cuda", normals=None):
    """ref (N,3) world; c2w (M,4,4); K for full-res (W,H). Returns bool (N,) visible in >=1 camera."""
    P = torch.tensor(ref, dtype=torch.float32, device=device)
    Nn = torch.tensor(normals, dtype=torch.float32, device=device) if normals is not None else None
    vis = torch.zeros(len(P), dtype=torch.bool, device=device)
    w, h = int(W * scale), int(H * scale)
    Ks = torch.tensor(K, dtype=torch.float32, device=device) * torch.tensor([[scale], [scale], [1.0]], device=device)
    for T in c2w:
        Tw = torch.tensor(np.linalg.inv(T), dtype=torch.float32, device=device)
        X = P @ Tw[:3, :3].T + Tw[:3, 3]
        z = X[:, 2]
        inf = (z > 0.5) & (z < depth_max)
        u = (Ks[0, 0] * X[:, 0] / z + Ks[0, 2]).round().long(); v = (Ks[1, 1] * X[:, 1] / z + Ks[1, 2]).round().long()
        inf &= (u >= 0) & (u < w) & (v >= 0) & (v < h)
        idx = torch.nonzero(inf).squeeze(1)
        if len(idx) == 0:
            continue
        pix = v[idx] * w + u[idx]
        zb = torch.full((w * h,), float("inf"), device=device)
        zb.scatter_reduce_(0, pix, z[idx], reduce="amin")          # every in-frustum point occludes
        zb2 = -torch.nn.functional.max_pool2d(-zb.view(1, 1, h, w), 3, 1, 1).view(-1)
        seen = z[idx] <= zb2[pix] * (1 + tol_rel) + tol_abs
        if Nn is not None:  # only surfaces facing the camera count as visible (horizontal ones only from above)
            cv = torch.tensor(T[:3, 3], dtype=torch.float32, device=device) - P[idx]
            n = Nn[idx]
            cosang = (n * cv).sum(1) / cv.norm(dim=1).clamp(min=1e-6)
            seen &= torch.where(n[:, 2].abs() > 0.8, cv[:, 2] > 0.3, cosang.abs() > 0.1)
        vis[idx[seen]] = True
    return vis.cpu().numpy()
