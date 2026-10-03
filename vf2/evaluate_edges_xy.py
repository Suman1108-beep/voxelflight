"""Horizontal absolute check by building-edge registration: our model's height raster vs the official DSM on the same
0.5 m grid; the 2-D shift maximising normalised cross-correlation of height-gradient magnitude (edges) is the
horizontal georeferencing error. Vertical datum and vegetation changes barely affect edge positions."""
import json, sys
import numpy as np, pyproj, rasterio
from scipy import ndimage
run, raw = sys.argv[1], sys.argv[2]
res_m = 0.5
with rasterio.open(run + "/model/dsm_utm.tif") as r:
    crs = r.crs; b = r.bounds
    a0 = r.read(1).astype(np.float32); f = int(round(res_m / r.res[0]))
    h, w = a0.shape[0] // f * f, a0.shape[1] // f * f
    blk = np.where(np.isfinite(a0[:h, :w]), a0[:h, :w], -np.inf).reshape(h // f, f, w // f, f).max(axis=(1, 3))
    a = np.where(np.isfinite(blk), blk, np.nan).astype(np.float32)
x0, y1 = b.left, b.top
z = np.load(raw, allow_pickle=True); D = z["points_src"]
t = pyproj.Transformer.from_crs(pyproj.CRS.from_wkt(str(z["src_crs_wkt"])), crs, always_xy=True)
X, Y = t.transform(D[:, 0], D[:, 1])
ix = ((X - x0) / res_m).astype(int); iy = ((y1 - Y) / res_m).astype(int)
ok = (ix >= 0) & (ix < a.shape[1]) & (iy >= 0) & (iy < a.shape[0])
ref = np.full(a.shape, np.nan, np.float32); ref[iy[ok], ix[ok]] = D[ok, 2]
_m = ndimage.maximum_filter(np.where(np.isfinite(ref), ref, -np.inf), size=3); ref = np.where(np.isfinite(_m), _m, np.nan)  # 1 m DSM onto 0.5 m grid
def grad(h):
    hh = np.where(np.isfinite(h), h, np.nanmedian(h))
    g = np.hypot(ndimage.sobel(hh, 0), ndimage.sobel(hh, 1))
    g[~np.isfinite(h)] = 0
    return np.clip(g, 0, np.percentile(g[g > 0], 99))
gm, gr = grad(a), grad(ref)
valid = np.isfinite(a) & np.isfinite(ref)
best = (-1, 0, 0)
R = int(6 / res_m)
for dy in range(-R, R + 1):
    for dx in range(-R, R + 1):
        s = np.roll(np.roll(gm, dy, 0), dx, 1); v = np.roll(np.roll(valid, dy, 0), dx, 1) & valid
        if v.sum() < 1000: continue
        p, q = s[v] - s[v].mean(), gr[v] - gr[v].mean()
        c = float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum() + 1e-12))
        if c > best[0]: best = (c, dx, dy)
c, dx, dy = best
# shifting our raster by (+dx cols, +dy rows) aligns it: our features are at (-dx, +dy) cells relative to reference
res = {"grid_m": res_m, "overlap_cells": int(valid.sum()), "ncc_best": c,
       "model_minus_reference_east_m": float(-dx * res_m), "model_minus_reference_north_m": float(dy * res_m),
       "horizontal_error_m": float(np.hypot(dx, dy) * res_m),
       "ncc_at_zero_shift": None}
p, q = gm[valid] - gm[valid].mean(), gr[valid] - gr[valid].mean()
res["ncc_at_zero_shift"] = float((p * q).sum() / np.sqrt((p * p).sum() * (q * q).sum() + 1e-12))
print(json.dumps(res, indent=1)); json.dump(res, open(run + "/edges_xy.json", "w"), indent=1)
