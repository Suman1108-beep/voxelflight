"""Shared helpers for the VoxelFlight v2 A100 pipeline."""
import json
import os
import time
from contextlib import contextmanager

import numpy as np

ROOT = "/workspace/voxelflight_a100_20260928"


def umeyama(src, dst, w=None):
    """Weighted similarity: dst ~= s * R @ src + t."""
    w = np.ones(len(src)) if w is None else np.asarray(w, float)
    ms = (w[:, None] * src).sum(0) / w.sum()
    md = (w[:, None] * dst).sum(0) / w.sum()
    a, b = src - ms, dst - md
    U, S, Vt = np.linalg.svd((w[:, None] * b).T @ a / w.sum())
    d = np.ones(3)
    if np.linalg.det(U @ Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    s = (S * d).sum() / ((w * (a ** 2).sum(1)).sum() / w.sum())
    return float(s), R, md - s * R @ ms


def robust_sim3(src, dst, sigma, iterations=10):
    """IRLS similarity fit with per-point sigma and a Huber-like reweighting."""
    w = 1.0 / sigma ** 2
    for _ in range(iterations):
        s, R, t = umeyama(src, dst, w)
        r = np.linalg.norm(s * src @ R.T + t - dst, axis=1)
        k = 2.0 * np.median(r) + 1e-9
        w = (1.0 / sigma ** 2) * np.where(r < k, 1.0, k / r)
    return s, R, t


def apply_sim3(x, s, R, t):
    return s * x @ R.T + t


def rmse(a, b):
    return float(np.sqrt(((a - b) ** 2).sum(1).mean()))


def error_summary(a, b):
    e = np.linalg.norm(a - b, axis=1)
    return {"rmse_m": float(np.sqrt((e ** 2).mean())), "median_m": float(np.median(e)),
            "p95_m": float(np.percentile(e, 95)), "max_m": float(e.max())}


def sim3_error(x, gt):
    s, R, t = umeyama(x, gt)
    return error_summary(apply_sim3(x, s, R, t), gt)


class Timer:
    def __init__(self):
        self.stages = {}
        self.t0 = time.time()

    @contextmanager
    def stage(self, name):
        tic = time.time()
        print(f"[{time.time() - self.t0:7.1f}s] >> {name}", flush=True)
        yield
        self.stages[name] = time.time() - tic
        print(f"[{time.time() - self.t0:7.1f}s] << {name} ({self.stages[name]:.1f}s)", flush=True)

    def total(self):
        return time.time() - self.t0


def dump(path, obj):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
