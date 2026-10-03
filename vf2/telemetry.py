"""Load per-frame GPS (+ optional barometer) and express it in UTM / local ENU."""
import csv

import numpy as np
from pyproj import CRS, Transformer


def utm_epsg(lat, lon):
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def load_video_telemetry(path):
    """CSV with source_frame,timestamp_s,latitude,longitude,altitude_m (one row per video frame)."""
    rows = list(csv.DictReader(open(path)))
    frame = np.array([int(r["source_frame"]) for r in rows])
    ts = np.array([float(r["timestamp_s"]) for r in rows])
    lat = np.array([float(r["latitude"]) for r in rows])
    lon = np.array([float(r["longitude"]) for r in rows])
    alt = np.array([float(r["altitude_m"]) for r in rows])
    eph = np.array([float(r.get("gps_eph_m") or 5.0) for r in rows]) if "gps_eph_m" in rows[0] else np.full(len(rows), 5.0)
    return frame, ts, lat, lon, alt, eph


class GeoFrame:
    """Local ENU-like frame = UTM easting/northing/altitude minus an origin (metres)."""

    def __init__(self, lat, lon, alt):
        self.epsg = utm_epsg(float(np.median(lat)), float(np.median(lon)))
        self.fwd = Transformer.from_crs(4326, self.epsg, always_xy=True)
        e, n = self.fwd.transform(lon, lat)
        self.origin = np.array([np.median(e), np.median(n), np.median(alt)])

    def to_local(self, lat, lon, alt):
        e, n = self.fwd.transform(lon, lat)
        return np.column_stack([e, n, alt]) - self.origin

    def crs_wkt(self):
        return CRS.from_epsg(self.epsg).to_wkt()


def baro_altitude(baro_csv, gps_csv, frame_image_ids):
    """Zurich-style logs: map barometric altitude onto image ids via shared onboard timestamps."""
    gps = np.genfromtxt(gps_csv, delimiter=",", skip_header=1, usecols=(0, 1))
    gps = gps[np.argsort(gps[:, 1])]
    t = np.interp(frame_image_ids, gps[:, 1], gps[:, 0])
    baro = np.genfromtxt(baro_csv, delimiter=",", skip_header=1, usecols=(0, 2))
    return np.interp(t, baro[:, 0], baro[:, 1])
