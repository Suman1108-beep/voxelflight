"""Input preparation (not timed): one flight of geotagged drone stills -> 1 fps H.264-like MP4 + per-frame telemetry CSV.
Selects the flight (camera model) with the most images; frames keep capture order."""
import csv, json, os, sys
from collections import Counter
from datetime import datetime
import cv2

src, out = sys.argv[1], sys.argv[2]
os.makedirs(out, exist_ok=True)
ex = json.load(open(src + "/exif.json"))["images"]
model = Counter(r["model"] for r in ex).most_common(1)[0][0]
rows = sorted([r for r in ex if r["model"] == model], key=lambda r: (r["time"], r["image"]))
W = 1920; H = int(round(W * rows[0]["height"] / rows[0]["width"] / 2) * 2)
vw = cv2.VideoWriter(out + "/flight.mp4", cv2.VideoWriter_fourcc(*"mp4v"), 1.0, (W, H))
t0 = datetime.strptime(rows[0]["time"], "%Y:%m:%d %H:%M:%S")
with open(out + "/telemetry_v2.csv", "w", newline="") as f:
    w = csv.writer(f); w.writerow(["source_frame", "timestamp_s", "latitude", "longitude", "altitude_m", "gps_eph_m"])
    for k, r in enumerate(rows):
        img = cv2.imread(os.path.join(src, "images", r["image"]))
        vw.write(cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA))
        ts = (datetime.strptime(r["time"], "%Y:%m:%d %H:%M:%S") - t0).total_seconds()
        rtk = r.get("xmp_RtkFlag") == 50.0   # RTK fixed: use the corrected position and its reported accuracy
        lat = r.get("xmp_GpsLatitude", r["lat"]) if rtk else r["lat"]
        lon = r.get("xmp_GpsLongtitude", r.get("xmp_GpsLongitude", r["lon"])) if rtk else r["lon"]
        alt = r.get("xmp_AbsoluteAltitude", r["alt"]) if rtk else r["alt"]
        eph = max(r.get("xmp_RtkStdLat", 0.02), r.get("xmp_RtkStdLon", 0.02), 0.02) if rtk else 3.0
        w.writerow([k, f"{ts:.3f}", f"{lat:.9f}", f"{lon:.9f}", f"{alt:.3f}", f"{eph:.3f}"])
vw.release()
json.dump({"source": src, "flight_camera_model": model.strip("\x00"), "frames": len(rows), "width": W, "height": H,
           "note": "1 fps video assembled from one flight's geotagged stills (capture order); GPS from EXIF"},
          open(out + "/dataset_summary.json", "w"), indent=2)
print("frames", len(rows), "model", model.strip("\x00"), "span_s", ts)
