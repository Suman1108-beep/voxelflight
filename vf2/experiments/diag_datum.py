import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.spatial import cKDTree
from telemetry import load_video_telemetry, GeoFrame, baro_altitude
ROOT = "/workspace/voxelflight_a100_20260928"; INP = ROOT + "/runs/zurich-10min-input"; LOGS = ROOT + "/datasets/zurich-40001-58000/Log Files"
fr, ts, lat, lon, alt, _ = load_video_telemetry(INP + "/video_telemetry.csv"); geo = GeoFrame(lat, lon, alt)
print("origin", geo.origin)
z = np.load(ROOT + "/vf2/runs/lidar_ref_2018_egm96.npz"); P = z["points"]
t2 = cKDTree(P[:, :2])
idx = np.arange(0, 18000, 300)
g = geo.to_local(lat, lon, alt)[idx]
bz = baro_altitude(LOGS + "/BarometricPressure.csv", LOGS + "/OnboardGPS.csv", idx + 40001) - geo.origin[2]
gt = np.genfromtxt(LOGS + "/GroundTruthAGL.csv", delimiter=",", skip_header=1)[:, :4]; gt = gt[np.argsort(gt[:, 0])]
GT = np.column_stack([np.interp(idx + 40001, gt[:, 0], gt[:, k]) for k in (1, 2, 3)]) - geo.origin
def ground(xy):
    out = []
    for p in xy:
        nb = t2.query_ball_point(p, 4.0)
        out.append(np.percentile(P[nb, 2], 5) if nb else np.nan)
    return np.array(out)
gg = ground(g[:, :2]); gG = ground(GT[:, :2])
print("median height above LiDAR ground: GPS alt %.1f  baro %.1f  GT(diag) %.1f" % (np.nanmedian(g[:, 2] - gg), np.nanmedian(bz - gg), np.nanmedian(GT[:, 2] - gG)))
print("GT z - GPS z median %.2f ; GT z - baro median %.2f" % (np.median(GT[:, 2] - g[:, 2]), np.median(GT[:, 2] - bz)))
