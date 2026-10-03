"""Independent reference: USGS 3DEP airborne LiDAR (public, AWS Entwine Point Tiles) cropped to a lat/lon box,
converted to the flight's UTM frame (heights: NAVD88 orthometric via the resource's vertical CRS -> EGM96 where possible)."""
import io, json, os, sys, urllib.request
sys.path.insert(0, "/workspace/voxelflight_a100_20260928/extra-packages")
import numpy as np

lat0, lon0, half_m, out = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), sys.argv[4]
BASE = "https://s3-us-west-2.amazonaws.com/usgs-lidar-public"
res = json.load(urllib.request.urlopen("https://raw.githubusercontent.com/hobuinc/usgs-lidar/master/boundaries/resources.geojson", timeout=60))
from shapely.geometry import shape, Point
cands = [f["properties"]["name"] for f in res["features"] if shape(f["geometry"]).contains(Point(lon0, lat0))]
print("covering EPT resources:", cands)
import laspy, pyproj
best = None
for name in sorted(cands, reverse=True):
    try:
        ept = json.load(urllib.request.urlopen(f"{BASE}/{name}/ept.json", timeout=30)); best = name; break
    except Exception as e:
        print("skip", name, e)
srs = ept["srs"]; wkt = srs.get("wkt"); epsg = srs.get("horizontal")
crs = pyproj.CRS.from_wkt(wkt) if wkt else pyproj.CRS.from_epsg(int(epsg))
print("resource", best, "crs", crs.to_string()[:80], "points", ept["points"])
to_src = pyproj.Transformer.from_crs(4326, crs.to_2d() if hasattr(crs, "to_2d") else crs, always_xy=True)
cx, cy = to_src.transform(lon0, lat0)
deg = half_m / 111000.0
xs, ys = to_src.transform([lon0 - deg * 1.4, lon0 + deg * 1.4], [lat0 - deg, lat0 + deg])
bx = (min(xs), min(ys), max(xs), max(ys))
b = ept["bounds"]; span = b[3] - b[0]
def node_bounds(k):
    d, x, y, z = map(int, k.split("-")); s = span / 2 ** d
    return b[0] + x * s, b[1] + y * s, b[0] + (x + 1) * s, b[1] + (y + 1) * s
def hits(k):
    nb = node_bounds(k); return nb[0] < bx[2] and nb[2] > bx[0] and nb[1] < bx[3] and nb[3] > bx[1]
hier = {}
def hierarchy(key):
    h = json.load(urllib.request.urlopen(f"{BASE}/{best}/ept-hierarchy/{key}.json", timeout=30))
    for k, v in h.items():
        if not hits(k):
            continue                      # prune subtrees outside the flight area
        if v == -1: hierarchy(k)
        else: hier[k] = v
hierarchy("0-0-0-0")
sel = list(hier)
print("nodes intersecting:", len(sel))
pts = []
for k in sel:
    data = urllib.request.urlopen(f"{BASE}/{best}/ept-data/{k}.laz", timeout=60).read()
    las = laspy.read(io.BytesIO(data))
    X, Y, Z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
    m = (X >= bx[0]) & (X <= bx[2]) & (Y >= bx[1]) & (Y <= bx[3])
    if "classification" in las.point_format.dimension_names:
        m &= np.asarray(las.classification) != 7  # drop low noise
    pts.append(np.column_stack([X[m], Y[m], Z[m]]))
P = np.concatenate(pts)
np.savez_compressed(out, points_src=P, src_crs_wkt=crs.to_wkt(), resource=best)
print("saved", len(P), "points from", best)
