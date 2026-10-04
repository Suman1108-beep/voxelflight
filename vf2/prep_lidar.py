import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from telemetry import load_video_telemetry, GeoFrame
from lidar import lv95_to_utm_ellipsoidal, load_reference_cloud
ROOT = "/workspace/voxelflight_a100_20260928"; INP = sys.argv[1] if len(sys.argv) > 1 else ROOT + "/runs/zurich-10min-input"
OUT = sys.argv[2] if len(sys.argv) > 2 else ROOT + "/vf2/runs/lidar_ref_2018_egm96.npz"
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv")
geo = GeoFrame(lat, lon, alt)
# datum sanity: geoid separation at flight centre (expect ~ +47..50 m in Zurich)
E, N = 2683539.6, 1248885.2
u = lv95_to_utm_ellipsoidal(np.array([E]), np.array([N]), np.array([400.0]), geo.epsg)
print("LHN95 400 m -> EGM96", u[0, 2], "difference", u[0, 2] - 400.0)
g = geo.to_local(lat, lon, alt)
pad = 60.0
bbox = (g[:, 0].min() - pad, g[:, 1].min() - pad, g[:, 0].max() + pad, g[:, 1].max() + pad)
print("bbox local", bbox)
t = time.time()
P, Nm = load_reference_cloud(ROOT + "/datasets/swisstopo", geo, bbox, voxel=0.15, cache=OUT)
print("reference points", len(P), "z range", P[:, 2].min(), P[:, 2].max(), "s", time.time() - t)
print("GPS-frame camera z median", np.median(g[:, 2]), "LiDAR ground (5th pct z)", np.percentile(P[:, 2], 5))
