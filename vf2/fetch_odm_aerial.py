"""Download a public OpenDroneMap aerial dataset and extract per-image GPS/time/camera EXIF."""
import json, os, sys, urllib.request
from concurrent.futures import ThreadPoolExecutor
from PIL import Image, ExifTags

name = sys.argv[1]  # e.g. odm_data_toledo
out = sys.argv[2]
os.makedirs(out + "/images", exist_ok=True)
items = json.load(urllib.request.urlopen(f"https://api.github.com/repos/OpenDroneMap/{name}/contents/images", timeout=30))
def get(it):
    p = os.path.join(out, "images", it["name"])
    if not os.path.exists(p) or os.path.getsize(p) != it["size"]:
        urllib.request.urlretrieve(it["download_url"], p)
    return p
with ThreadPoolExecutor(8) as ex:
    paths = list(ex.map(get, items))
TAGS = {v: k for k, v in ExifTags.TAGS.items()}
def dms(v, ref):
    d = float(v[0]) + float(v[1]) / 60 + float(v[2]) / 3600
    return -d if ref in ("S", "W") else d
import re
def _xmp(path):
    """DJI XMP: RTK status/std-devs and RTK-corrected position/altitude when present."""
    head = open(path, "rb").read(200000)
    a, b = head.find(b"<x:xmpmeta"), head.find(b"</x:xmpmeta>")
    if a < 0: return {}
    x = head[a:b].decode("latin1"); out = {}
    for k in ("RtkFlag", "RtkStdLon", "RtkStdLat", "RtkStdHgt", "GpsLatitude", "GpsLongitude", "GpsLongtitude",
              "AbsoluteAltitude", "RelativeAltitude", "GimbalPitchDegree", "FlightYawDegree"):
        m = re.search(k + r'="?([-+0-9.eE]+)', x)
        if m: out["xmp_" + k] = float(m.group(1))
    return out
rows = []
for p in sorted(paths):
    im = Image.open(p); ex = im._getexif() or {}
    g = ex.get(TAGS["GPSInfo"], {})
    rows.append({"image": os.path.basename(p), "width": im.width, "height": im.height,
                 "time": ex.get(TAGS["DateTimeOriginal"]), "lat": dms(g[2], g[1]), "lon": dms(g[4], g[3]),
                 "alt": float(g[6]) if 6 in g else None, "focal_mm": float(ex.get(TAGS["FocalLength"], 0) or 0),
                 "focal_35mm": ex.get(TAGS["FocalLengthIn35mmFilm"]), "model": ex.get(TAGS["Model"]), **_xmp(p)})
json.dump({"source": f"https://github.com/OpenDroneMap/{name}", "license": "see repository", "images": rows}, open(out + "/exif.json", "w"), indent=1)
print(len(rows), rows[0], rows[-1], sep="\n")
