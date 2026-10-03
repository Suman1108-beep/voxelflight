"""Input preparation (not timed): Zurich MAV onboard logs -> VoxelFlight per-frame telemetry CSV.
Uses only onboard GPS (position, eph) and barometer. Reference/ground-truth files are not read."""
import csv, sys
import numpy as np
seg = sys.argv[1]           # e.g. /workspace/.../datasets/zurich-40001-58000
first_id = int(sys.argv[2])  # first image id of the segment, e.g. 40001
n_frames = int(sys.argv[3])  # number of video frames
out = sys.argv[4]
L = seg + "/Log Files"
g = np.genfromtxt(L + "/OnboardGPS.csv", delimiter=",", skip_header=1, usecols=range(14))
g = g[np.argsort(g[:, 1])]
ids = first_id + np.arange(n_frames)
I = lambda c: np.interp(ids, g[:, 1], c)
ts_us = I(g[:, 0])
baro = np.genfromtxt(L + "/BarometricPressure.csv", delimiter=",", skip_header=1, usecols=(0, 2))
bz = np.interp(ts_us, baro[:, 0], baro[:, 1])
with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["source_frame", "timestamp_s", "latitude", "longitude", "altitude_m", "gps_eph_m", "baro_alt_m"])
    for k in range(n_frames):
        w.writerow([k, f"{(ts_us[k] - ts_us[0]) / 1e6:.6f}", f"{I(g[:, 2])[k]:.8f}", f"{I(g[:, 3])[k]:.8f}", f"{I(g[:, 4])[k]:.3f}",
                    f"{I(g[:, 8])[k]:.3f}", f"{bz[k]:.3f}"])
print("wrote", out, n_frames)
