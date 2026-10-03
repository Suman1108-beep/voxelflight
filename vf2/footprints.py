"""Building footprints (OpenStreetMap, ODbL) as vertical-wall anchors; 2-D point-to-wall distances."""
import json
import os
import urllib.parse
import urllib.request

import numpy as np


def fetch_osm_buildings(bbox_wgs84, cache):
    """bbox = (lon_min, lat_min, lon_max, lat_max). Returns list of rings [(lon, lat), ...]."""
    if os.path.exists(cache):
        return json.load(open(cache))
    lo0, la0, lo1, la1 = bbox_wgs84
    q = f'[out:json][timeout:60];(way["building"]({la0},{lo0},{la1},{lo1});relation["building"]({la0},{lo0},{la1},{lo1}););out geom;'
    data = None
    for url in ("https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter"):
        try:
            req = urllib.request.Request(url, data=urllib.parse.urlencode({"data": q}).encode(), headers={"User-Agent": "VoxelFlight-SIH/1.0"})
            data = json.load(urllib.request.urlopen(req, timeout=90))
            break
        except Exception as e:  # try mirror
            print("overpass failed", url, e)
    rings = []
    for el in data["elements"]:
        if el["type"] == "way" and "geometry" in el:
            rings.append([(g["lon"], g["lat"]) for g in el["geometry"]])
        elif el["type"] == "relation":
            for mem in el.get("members", []):
                if mem.get("role") == "outer" and "geometry" in mem:
                    rings.append([(g["lon"], g["lat"]) for g in mem["geometry"]])
    json.dump({"source": "OpenStreetMap contributors (ODbL)", "bbox": bbox_wgs84, "rings": rings}, open(cache, "w"))
    return json.load(open(cache))


def wall_segments(rings, geo):
    segs = []
    for r in rings:
        lon = np.array([p[0] for p in r]); lat = np.array([p[1] for p in r])
        xy = geo.to_local(lat, lon, np.zeros(len(lat)))[:, :2]
        segs.append(np.column_stack([xy[:-1], xy[1:]]))
    S = np.concatenate(segs)
    keep = np.linalg.norm(S[:, 2:] - S[:, :2], axis=1) > 0.3
    return S[keep]


def point_wall_distance(P2, S, chunk=4096, return_normal=False):
    """Unsigned 2-D distance from points (N,2) to nearest segment; optional unit normal of that segment."""
    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    A = torch.tensor(S[:, :2], device=dev); B = torch.tensor(S[:, 2:], device=dev)
    AB = B - A; L2 = (AB ** 2).sum(1).clamp(min=1e-9)
    D = np.empty(len(P2)); I = np.empty(len(P2), int)
    for s in range(0, len(P2), chunk):
        p = torch.tensor(P2[s:s + chunk], device=dev)
        t = (((p[:, None] - A[None]) * AB[None]).sum(-1) / L2[None]).clamp(0, 1)
        proj = A[None] + t[..., None] * AB[None]
        d = (p[:, None] - proj).norm(dim=-1)
        dm, im = d.min(1)
        D[s:s + chunk] = dm.cpu().numpy(); I[s:s + chunk] = im.cpu().numpy()
    if not return_normal:
        return D
    ab = S[I, 2:] - S[I, :2]
    n = np.column_stack([-ab[:, 1], ab[:, 0]]) / np.linalg.norm(ab, axis=1, keepdims=True)
    return D, I, n
