"""Gap-free surface for the flown area: 2.5D fill of holes in a fused model, kept separate from measured geometry.
Default (--surface ground): measured ground cells (lowest point within 0.6 m of the local minimum over 7 m) are found, and
empty cells within --range of the camera path and within --max-gap of measured ground get the nearby ground height, so roads
and pavements close without vertical sheets. --surface top reproduces the earlier fill from the highest measured surface. Filled cells become a tinted mesh layer and are reported
separately, so interpolated area is never presented as measured. Output is a new run folder that the evaluator can score."""
import argparse, json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--cell", type=float, default=0.5); ap.add_argument("--max-gap", type=float, default=5.0)
ap.add_argument("--range", type=float, default=35.0); ap.add_argument("--tint", default="0.30,0.36,0.48")
ap.add_argument("--surface", choices=["ground", "top"], default="ground")
a = ap.parse_args()
os.makedirs(a.out + "/model", exist_ok=True)
for f in ("keyframe_poses.npz", "report.json", "keyframes.json"):
    if os.path.exists(os.path.join(a.run, f)): shutil.copy(os.path.join(a.run, f), a.out)
mesh = o3d.io.read_triangle_mesh(a.run + "/model/model_mesh.ply")
V = np.asarray(mesh.vertices); col = np.asarray(mesh.vertex_colors)
C = np.load(a.run + "/keyframe_poses.npz")["camera_to_world"][:, :3, 3]

lo = V[:, :2].min(0) - 1; ij = ((V[:, :2] - lo) / a.cell).astype(np.int64); nx, ny = ij.max(0) + 2
flat = ij[:, 1] * nx + ij[:, 0]
top = np.full(nx * ny, -np.inf); np.maximum.at(top, flat, V[:, 2])                 # highest measured surface per cell
low = np.full(nx * ny, np.inf); np.minimum.at(low, flat, V[:, 2]); low = low.reshape(ny, nx)   # lowest per cell
rgb = np.zeros((nx * ny, 3)); cnt = np.zeros(nx * ny); np.add.at(rgb, flat, col); np.add.at(cnt, flat, 1)
occ = np.isfinite(top).reshape(ny, nx); top = top.reshape(ny, nx); rgb = (rgb / np.maximum(cnt, 1)[:, None]).reshape(ny, nx, 3)

gx, gy = np.meshgrid(lo[0] + (np.arange(nx) + 0.5) * a.cell, lo[1] + (np.arange(ny) + 0.5) * a.cell)
near, _ = cKDTree(C[:, :2]).query(np.column_stack([gx.ravel(), gy.ravel()]), distance_upper_bound=a.range)
domain = np.isfinite(near).reshape(ny, nx)
k = max(3, int(round(2.0 / a.cell)) | 1)                                            # ~2 m median window
if a.surface == "ground":
    w = int(round(7.0 / a.cell)) | 1
    ground = occ & (low <= ndimage.minimum_filter(np.where(occ, low, np.inf), size=w, mode="nearest") + 0.6)
    dist, (iy, ix) = ndimage.distance_transform_edt(~ground, return_indices=True)
    fill = domain & ~occ & (dist * a.cell <= a.max_gap)
    h = np.where(ground, low, low[iy, ix])                                          # nearest measured ground height
else:
    dist, (iy, ix) = ndimage.distance_transform_edt(~occ, return_indices=True)
    fill = domain & ~occ & (dist * a.cell <= a.max_gap)
    h = np.where(occ, top, top[iy, ix])                                             # nearest measured top height
h = np.where(fill, ndimage.median_filter(h, size=k, mode="nearest"), h)

# vertex grid at cell corners: corner height = mean of adjacent filled cells; triangles only for filled cells
corner = np.zeros((ny + 1, nx + 1)); cw = np.zeros((ny + 1, nx + 1)); hf = np.where(fill, h, 0); wf = fill.astype(float)
for dy in (0, 1):
    for dx in (0, 1):
        corner[dy:dy + ny, dx:dx + nx] += hf; cw[dy:dy + ny, dx:dx + nx] += wf
corner = corner / np.maximum(cw, 1)
fy, fx = np.nonzero(fill)
vid = -np.ones((ny + 1, nx + 1), np.int64); used = np.zeros((ny + 1, nx + 1), bool)
for dy in (0, 1):
    for dx in (0, 1):
        used[fy + dy, fx + dx] = True
uy, ux = np.nonzero(used); vid[uy, ux] = np.arange(len(uy))
FV = np.column_stack([lo[0] + ux * a.cell, lo[1] + uy * a.cell, corner[uy, ux]])
q = [vid[fy, fx], vid[fy, fx + 1], vid[fy + 1, fx + 1], vid[fy + 1, fx]]
FT = np.vstack([np.column_stack([q[0], q[1], q[2]]), np.column_stack([q[0], q[2], q[3]])])
tint = np.array([float(x) for x in a.tint.split(",")])
neigh = rgb[iy, ix]; vcol = 0.4 * tint + 0.6 * neigh[np.clip(uy, 0, ny - 1), np.clip(ux, 0, nx - 1)]   # nearest ground colour, tinted
fm = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(FV), o3d.utility.Vector3iVector(FT))
fm.vertex_colors = o3d.utility.Vector3dVector(np.clip(vcol, 0, 1)); fm.compute_vertex_normals()
o3d.io.write_triangle_mesh(a.out + "/model/fill_mesh.ply", fm)
o3d.io.write_triangle_mesh(a.out + "/model/model_mesh.ply", mesh + fm)
stats = {"source_run": a.run, "surface": a.surface, "cell_m": a.cell, "max_gap_m": a.max_gap, "range_m": a.range,
         "measured_cells": int(occ.sum()), "domain_cells": int(domain.sum()), "filled_cells": int(fill.sum()),
         "filled_area_m2": float(fill.sum() * a.cell ** 2), "measured_area_m2": float((occ & domain).sum() * a.cell ** 2),
         "domain_covered_before": float((occ & domain).sum() / max(domain.sum(), 1)),
         "domain_covered_after": float(((occ | fill) & domain).sum() / max(domain.sum(), 1)),
         "fill_triangles": int(len(FT)), "measured_triangles": int(len(mesh.triangles)),
         "note": f"fill = interpolated {a.surface} surface (tinted), not measured; reported separately from measured geometry"}
json.dump(stats, open(a.out + "/fill_report.json", "w"), indent=1)
print(json.dumps(stats, indent=1))
