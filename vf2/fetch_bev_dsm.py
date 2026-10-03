"""Independent reference for Austria: BEV ALS DSM 1 m (airborne LiDAR surface model, CC-BY-4.0), windowed
HTTP read of the cloud-optimised GeoTIFF tile around a lat/lon. Saved as points for evaluate_aerial.py."""
import sys
import numpy as np
import pyproj
import rasterio
from rasterio.windows import from_bounds

lat0, lon0, half, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
t = pyproj.Transformer.from_crs(4326, 3035, always_xy=True)
e, n = t.transform(lon0, lat0)
E, N = int(e // 50000 * 50000), int(n // 50000 * 50000)
src = None
for date in ("20230915", "20220915", "20240915", "20210401"):
    url = f"https://data.bev.gv.at/download/ALS/DSM/{date}/ALS_DSM_CRS3035RES50000mN{N}E{E}.tif"
    try:
        src = rasterio.open("/vsicurl/" + url); print("using", url, src.crs, src.res); break
    except Exception as ex:
        print("not found", url, str(ex)[:60])
if src is None:
    raise SystemExit("no BEV DSM tile found")
win = from_bounds(e - half, n - half, e + half, n + half, src.transform)
a = src.read(1, window=win).astype(np.float64)
tr = src.window_transform(win)
rows, cols = np.mgrid[0:a.shape[0], 0:a.shape[1]]
xs, ys = rasterio.transform.xy(tr, rows.ravel(), cols.ravel())
ok = np.isfinite(a.ravel()) & (a.ravel() > -1000) & (a.ravel() != (src.nodata if src.nodata is not None else -9999))
P = np.column_stack([np.asarray(xs)[ok], np.asarray(ys)[ok], a.ravel()[ok]])
np.savez_compressed(out, points_src=P, src_crs_wkt=src.crs.to_wkt(), resource=url)
print("saved", len(P), "DSM cells; height range", P[:, 2].min(), P[:, 2].max())
