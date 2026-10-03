"""Absolute accuracy of the RTK-locked photogrammetric points vs an official DSM (no alignment of any kind).
Vertical: DSM height at each point's XY vs the point height (after geoid conversion); horizontal: best 2-D shift
that minimises vertical differences on sloped / edge cells (reported, never applied to the product)."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pyproj, pycolmap
from scipy.interpolate import RegularGridInterpolator
pyproj.network.set_network_enabled(True)
run, raw = sys.argv[1], sys.argv[2]
rep = json.load(open(run + "/report.json")); origin = np.array(rep["utm_origin"]); epsg = rep["utm_epsg"]
rec = pycolmap.Reconstruction(run + "/sparse/0")
P = np.array([p.xyz for p in rec.points3D.values()])
err = np.array([p.error for p in rec.points3D.values()]); trk = np.array([p.track.length() for p in rec.points3D.values()])
keep = (err < 1.0) & (trk >= 3); P = P[keep]
# The sparse model on disk is in the SfM frame; apply the same RTK lock as the run used
gl = None
import csv
# recompute the RTK similarity (same as rtk_photogrammetry.py)
from common import robust_sim3
from telemetry import GeoFrame
tel = list(csv.DictReader(open(sys.argv[3])))
lat = np.array([float(r["latitude"]) for r in tel]); lon = np.array([float(r["longitude"]) for r in tel]); alt = np.array([float(r["altitude_m"]) for r in tel])
geo = GeoFrame(lat, lon, alt); gl = geo.to_local(lat, lon, alt)
C, G = [], []
for im in rec.images.values():
    if im.has_pose:
        i = int(im.name.split("_")[1].split(".")[0]); C.append(im.projection_center()); G.append(gl[i])
s, R, t = robust_sim3(np.array(C), np.array(G), np.full(len(C), 0.02))
W = (s * P @ R.T + t) + origin  # UTM, ellipsoidal heights (RTK)
to_ll = pyproj.Transformer.from_crs(epsg, 4326, always_xy=True)
lon_p, lat_p = to_ll.transform(W[:, 0], W[:, 1])
# ellipsoidal -> EGM2008 and EGM96 orthometric (Austrian heights are close to these; residual datum difference reported)
h08 = pyproj.Transformer.from_crs("EPSG:4979", "EPSG:4326+3855", always_xy=True).transform(lon_p, lat_p, W[:, 2])[2]
N08 = W[:, 2] - h08
# DSM
z = np.load(raw, allow_pickle=True); D = z["points_src"]; src = pyproj.CRS.from_wkt(str(z["src_crs_wkt"]))
xs, ys = np.unique(D[:, 0]), np.unique(D[:, 1])
grid = np.full((len(ys), len(xs)), np.nan); grid[np.searchsorted(ys, D[:, 1]), np.searchsorted(xs, D[:, 0])] = D[:, 2]
f = RegularGridInterpolator((ys, xs), grid, bounds_error=False, fill_value=np.nan)
to_src = pyproj.Transformer.from_crs(4326, src, always_xy=True)
ex, ey = to_src.transform(lon_p, lat_p)
def dz(dx=0.0, dy=0.0):
    return f(np.column_stack([ey + dy, ex + dx])) - h08
d0 = dz(); ok = np.isfinite(d0)
# horizontal: grid search of a 2-D shift minimising robust |dz| (diagnostic only)
best = (np.inf, 0, 0)
for dx in np.arange(-3, 3.01, 0.25):
    for dy in np.arange(-3, 3.01, 0.25):
        d = dz(dx, dy); m = np.isfinite(d)
        c = np.median(np.abs(d[m] - np.median(d[m])))
        if c < best[0]: best = (c, dx, dy)
res = {"sparse_points": int(len(P)), "geoid_EGM2008_N_median_m": float(np.median(N08)),
       "vertical_DSM_minus_points_m": {"median": float(np.median(d0[ok])), "mad": float(np.median(np.abs(d0[ok] - np.median(d0[ok])))),
                                        "abs_lt_0.5m": float(np.mean(np.abs(d0[ok]) < 0.5)), "abs_lt_1m": float(np.mean(np.abs(d0[ok]) < 1.0))},
       "best_horizontal_shift_m": {"east": float(best[1]), "north": float(best[2]), "norm": float(np.hypot(best[1], best[2])),
                                   "mad_at_best": float(best[0]), "mad_at_zero": float(np.median(np.abs(d0[ok] - np.median(d0[ok]))))},
       "note": "Points = RTK-locked photogrammetric tie points (reprojection < 1 px, >= 3 views). DSM = BEV ALS DSM 1 m (Austrian heights). "
               "Positive vertical median = DSM above points (points on ground under vegetation also read positive)."}
print(json.dumps(res, indent=1))
json.dump(res, open(run + "/sparse_vs_dsm.json", "w"), indent=1)
