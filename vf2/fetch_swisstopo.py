"""Download open swisstopo reference data (swissSURFACE3D LiDAR + 0.5 m DSM raster) covering a lat/lon box.
Reference data are used for (a) optional public-geodata registration and (b) independent surface evaluation."""
import hashlib, json, os, sys, urllib.request, zipfile

OUT = sys.argv[1] if len(sys.argv) > 1 else "/workspace/voxelflight_a100_20260928/datasets/swisstopo"
bbox = [8.5420, 47.3830, 8.5490, 47.3885]  # lon_min, lat_min, lon_max, lat_max (flight area + margin)
YEAR = os.environ.get("YEAR", "2018")
os.makedirs(OUT, exist_ok=True)
manifest = []
for coll, want in [("ch.swisstopo.swisssurface3d", ".las.zip"), ("ch.swisstopo.swisssurface3d-raster", "_0.5_2056_5728.tif")]:
    url = f"https://data.geo.admin.ch/api/stac/v0.9/collections/{coll}/items?bbox={','.join(map(str, bbox))}&limit=50"
    items = json.load(urllib.request.urlopen(url, timeout=30))["features"]
    for it in items:
        if f"_{YEAR}_" not in it["id"]:
            continue
        for k, a in it["assets"].items():
            if not k.endswith(want):
                continue
            dst = os.path.join(OUT, k)
            if not os.path.exists(dst):
                urllib.request.urlretrieve(a["href"], dst)
            if dst.endswith(".zip"):
                with zipfile.ZipFile(dst) as z:
                    z.extractall(OUT)
            sha = hashlib.sha256(open(dst, "rb").read()).hexdigest()
            manifest.append({"collection": coll, "item": it["id"], "asset": k, "url": a["href"], "bytes": os.path.getsize(dst), "sha256": sha})
            print("got", k, os.path.getsize(dst))
json.dump({"source": "swisstopo open government data (data.geo.admin.ch), free use with attribution",
           "bbox_wgs84": bbox, "year": YEAR, "files": manifest}, open(os.path.join(OUT, f"manifest_{YEAR}.json"), "w"), indent=2)
print(os.listdir(OUT))
