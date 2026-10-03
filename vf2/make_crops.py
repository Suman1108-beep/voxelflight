"""Tiled inference input: split each frame into a 2x2 grid of overlapping crops (frac of width/height each).
Crops keep their pixel offsets so the dense step can use the parent camera with shifted intrinsics."""
import json, os, sys
import cv2
src, dst = sys.argv[1], sys.argv[2]
frac = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6
os.makedirs(dst, exist_ok=True)
meta = {}
for n in sorted(os.listdir(src)):
    if not n.endswith(".jpg"): continue
    im = cv2.imread(os.path.join(src, n)); H, W = im.shape[:2]
    cw, ch = int(W * frac), int(H * frac)
    for k, (ox, oy) in enumerate([(0, 0), (W - cw, 0), (0, H - ch), (W - cw, H - ch)]):
        cn = n.replace(".jpg", f"_c{k}.jpg")
        cv2.imwrite(os.path.join(dst, cn), im[oy:oy + ch, ox:ox + cw], [cv2.IMWRITE_JPEG_QUALITY, 95])
        meta[cn] = {"parent": n, "ox": ox, "oy": oy, "w": cw, "h": ch, "W": W, "H": H}
json.dump(meta, open(os.path.join(dst, "crops.json"), "w"))
print("crops", len(meta))
