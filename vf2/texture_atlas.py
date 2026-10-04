"""Photo-textured mesh from a fused run: decimate, UV-unwrap (xatlas), and bake an atlas from the source video frames.
Each triangle takes its texels from the keyframe that sees it most frontally and closest, with an occlusion test by ray
casting; triangles no camera sees keep their fused vertex colour. Writes a textured GLB (JPEG atlas) and OBJ/MTL/JPG."""
import argparse, json, os, struct, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, cv2, open3d as o3d

ap = argparse.ArgumentParser()
ap.add_argument("--run", required=True); ap.add_argument("--video", required=True); ap.add_argument("--calibration", required=True)
ap.add_argument("--out", required=True); ap.add_argument("--triangles", type=int, default=400_000)
ap.add_argument("--atlas", type=int, default=8192); ap.add_argument("--depth-max", type=float, default=35.0)
ap.add_argument("--jpeg-quality", type=int, default=85)
ap.add_argument("--cluster", type=float, default=0.1, help="vertex-clustering voxel (m) before decimation")
ap.add_argument("--min-component", type=int, default=150, help="drop floating fragments with fewer triangles than this")
ap.add_argument("--packing", choices=["xatlas", "pairs"], default="xatlas",
                help="pairs: two triangles per square atlas cell (seconds instead of xatlas chart packing; no mip-maps)")
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); t0 = time.time(); log = lambda *m: print(f"[{time.time() - t0:6.0f}s]", *m, flush=True)
if a.packing == "xatlas": import xatlas
from features import decode_keyframes

mesh = o3d.io.read_triangle_mesh(a.run + "/model/model_mesh.ply")
mesh = mesh.simplify_vertex_clustering(a.cluster); mesh.remove_unreferenced_vertices(); log("clustered", len(mesh.triangles), "triangles")
if len(mesh.triangles) > a.triangles:   # Open3D's quadric decimation stalls on this mesh; fast-simplification does not
    import fast_simplification
    from scipy.spatial import cKDTree
    V0, C0 = np.asarray(mesh.vertices), np.asarray(mesh.vertex_colors)
    V1, T1 = fast_simplification.simplify(V0.astype(np.float32), np.asarray(mesh.triangles).astype(np.int32), 1 - a.triangles / len(mesh.triangles))
    mesh = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(V1.astype(np.float64)), o3d.utility.Vector3iVector(T1))
    mesh.vertex_colors = o3d.utility.Vector3dVector(C0[cKDTree(V0).query(V1)[1]])
tri_c, n_c, _ = mesh.cluster_connected_triangles(); small = np.asarray(n_c)[np.asarray(tri_c)] < a.min_component
mesh.remove_triangles_by_mask(small); log("removed", int(small.sum()), "triangles in fragments <", a.min_component)
mesh.remove_unreferenced_vertices(); mesh.compute_triangle_normals()
V, T, VC = np.asarray(mesh.vertices), np.asarray(mesh.triangles), np.asarray(mesh.vertex_colors)
log("mesh", len(V), "vertices", len(T), "triangles")
if a.packing == "xatlas":
    atl = xatlas.Atlas(); atl.add_mesh(V.astype(np.float32), T.astype(np.uint32))
    po = xatlas.PackOptions(); po.resolution = a.atlas; po.padding = 2
    atl.generate(pack_options=po); vmap, F, UV = atl[0]
else:   # cell k holds triangles 2k (lower-left) and 2k+1 (upper-right), one texel apart along the diagonal
    n = len(T); cells = -(-n // 2); G = int(np.ceil(np.sqrt(cells))); c = a.atlas // G
    k = np.arange(n) // 2; ox, oy = (k % G) * c, (k // G) * c; x0, y0, x1, y1 = ox + 1, oy + 1, ox + c - 2, oy + c - 2
    up = np.arange(n) % 2 == 1
    tri = np.where(up[:, None, None], np.stack([np.stack([x1, y0 + 1], 1), np.stack([x1, y1], 1), np.stack([x0 + 1, y1], 1)], 1),
                   np.stack([np.stack([x0, y0], 1), np.stack([x1 - 1, y0], 1), np.stack([x0, y1 - 1], 1)], 1))
    UV = ((tri.reshape(-1, 2) + 0.5) / a.atlas).astype(np.float32); vmap = T.reshape(-1); F = np.arange(3 * n).reshape(n, 3)
log("atlas", len(vmap), "uv vertices")

kp = np.load(a.run + "/keyframe_poses.npz"); c2w, frames = kp["camera_to_world"], kp["video_frame"]
_, first = np.unique(frames, return_index=True); first = np.sort(first); c2w, frames = c2w[first], frames[first]   # tiled runs: one pose per crop
cal = json.load(open(a.calibration)); K, D = np.array(cal["intrinsic_matrix"]), np.array(cal.get("distortion_coefficients", []))
imgs, (W, H) = decode_keyframes(a.video, frames, workers=16)
if D.size: imgs = np.stack([cv2.undistort(im, K, D) for im in imgs])
log("decoded", len(imgs), "frames", W, "x", H)

# best camera per triangle: frontal, close, inside the frame and not occluded
scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
Pc = V[T].mean(1); Nt = np.asarray(mesh.triangle_normals); best = np.full(len(T), -1); score = np.zeros(len(T))
for c in range(len(c2w)):
    R, O = c2w[c, :3, :3], c2w[c, :3, 3]; X = (Pc - O) @ R; z = X[:, 2]
    ok = (z > 0.5) & (z < a.depth_max)
    u = K[0, 0] * X[:, 0] / np.where(ok, z, 1) + K[0, 2]; v = K[1, 1] * X[:, 1] / np.where(ok, z, 1) + K[1, 2]
    ok &= (u > 8) & (u < W - 8) & (v > 8) & (v < H - 8)
    d = O - Pc; dist = np.linalg.norm(d, axis=1); cos = (Nt * d).sum(1) / np.maximum(dist, 1e-6)
    s = np.abs(cos) / np.maximum(dist, 1.0); ok &= (np.abs(cos) > 0.15) & (s > score)
    idx = np.nonzero(ok)[0]
    if not len(idx): continue
    rays = np.hstack([np.repeat(O[None], len(idx), 0), -d[idx] / dist[idx, None]]).astype(np.float32)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy()
    vis = hit > dist[idx] - np.maximum(0.15, 0.02 * dist[idx])   # fused surfaces are noisy: tolerance grows with range
    best[idx[vis]] = c; score[idx[vis]] = s[idx[vis]]
log("textured triangles", round(float((best >= 0).mean()), 3))

def sample(im, u, v, w=4096):
    """Bilinear lookup of many pixels (cv2.remap maps must stay below 32767 per side)."""
    n = len(u); rows = -(-n // w); mu = np.zeros(rows * w, np.float32); mv = np.zeros(rows * w, np.float32); mu[:n], mv[:n] = u, v
    out = cv2.remap(im, mu.reshape(rows, w), mv.reshape(rows, w), cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    return out.reshape(-1, 3)[:n, ::-1] / 255.0

# bake: rasterise every triangle in UV space, map texels to 3D, sample the chosen frame (bilinear)
S = a.atlas; tex = np.zeros((S, S, 3), np.float32); have = np.zeros((S, S), bool)
uvp = UV * S - 0.5; Vn = V[vmap]; Cn = VC[vmap] if len(VC) else np.full((len(vmap), 3), 0.6)
for s0 in range(0, len(F), 40000):
    f = F[s0:s0 + 40000]; tri = uvp[f]; lo = np.floor(tri.min(1)).astype(int); hi = np.ceil(tri.max(1)).astype(int)
    wh = hi - lo + 1; n = wh[:, 0] * wh[:, 1]; ti = np.repeat(np.arange(len(f)), n)
    k = np.arange(n.sum()) - np.repeat(np.cumsum(n) - n, n)
    px = lo[ti, 0] + k % wh[ti, 0]; py = lo[ti, 1] + k // wh[ti, 0]
    a0, a1, a2 = tri[ti, 0], tri[ti, 1], tri[ti, 2]; e1, e2, p = a1 - a0, a2 - a0, np.column_stack([px, py]) - a0
    den = e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]; den = np.where(np.abs(den) < 1e-12, 1e-12, den)
    b1 = (p[:, 0] * e2[:, 1] - p[:, 1] * e2[:, 0]) / den; b2 = (e1[:, 0] * p[:, 1] - e1[:, 1] * p[:, 0]) / den; b0 = 1 - b1 - b2
    inside = (b0 > -0.02) & (b1 > -0.02) & (b2 > -0.02) & (px >= 0) & (px < S) & (py >= 0) & (py < S)
    ti, px, py, b = ti[inside], px[inside], py[inside], np.column_stack([b0, b1, b2])[inside]
    P3 = (Vn[f[ti]] * b[:, :, None]).sum(1); cam = best[s0 + ti]; col = (Cn[f[ti]] * b[:, :, None]).sum(1)
    for c in np.unique(cam[cam >= 0]):
        m = cam == c; R, O = c2w[c, :3, :3], c2w[c, :3, 3]; X = (P3[m] - O) @ R
        u = K[0, 0] * X[:, 0] / X[:, 2] + K[0, 2]; v = K[1, 1] * X[:, 1] / X[:, 2] + K[1, 2]
        col[m] = sample(imgs[c], u, v)
    tex[py, px] = col; have[py, px] = True
log("baked texels", int(have.sum()))
img = (np.clip(tex, 0, 1) * 255).astype(np.uint8)
for _ in range(6):   # bleed colours outward so mip-maps and chart borders do not show black seams
    grown = cv2.dilate(img, np.ones((3, 3), np.uint8)); img[~have] = grown[~have]; have = cv2.dilate(have.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0
img = img[::-1]   # glTF/OBJ UV origin: v = 0 at the bottom of the image -> flip rows once here
jpg = cv2.imencode(".jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, a.jpeg_quality])[1].tobytes()
open(a.out + "/textured_atlas.jpg", "wb").write(jpg)

uv_gl = UV.astype(np.float32)
with open(a.out + "/textured_mesh.obj", "w") as fo:
    fo.write("mtllib textured_mesh.mtl\nusemtl atlas\n")
    fo.write("".join(f"v {x:.3f} {y:.3f} {z:.3f}\n" for x, y, z in Vn)); fo.write("".join(f"vt {s:.6f} {t:.6f}\n" for s, t in uv_gl))
    fo.write("".join(f"f {i + 1}/{i + 1} {j + 1}/{j + 1} {k + 1}/{k + 1}\n" for i, j, k in F))
open(a.out + "/textured_mesh.mtl", "w").write("newmtl atlas\nKa 1 1 1\nKd 1 1 1\nmap_Kd textured_atlas.jpg\n")

def glb(path, P, UVs, I, image, y_up=True):
    """Minimal textured glTF 2.0 binary. y_up: same axes as export.write_glb (ENU (x, y, z) -> (x, z, -y)) for exchange;
    y_up=False keeps the website's Z-up local frame (make_web_demo.py convention)."""
    P = (np.column_stack([P[:, 0], P[:, 2], -P[:, 1]]) if y_up else P).astype(np.float32); UVs = np.column_stack([UVs[:, 0], 1 - UVs[:, 1]])
    pos, uvb, idx = P.tobytes(), UVs.astype(np.float32).tobytes(), I.astype(np.uint32).tobytes()
    pad = lambda b: b + b"\0" * (-len(b) % 4); views, blob = [], b""
    for data, target in ((pos, 34962), (uvb, 34962), (idx, 34963), (image, None)):
        v = {"buffer": 0, "byteOffset": len(blob), "byteLength": len(data)}
        if target: v["target"] = target
        views.append(v); blob += pad(data)
    g = {"asset": {"version": "2.0", "generator": "VoxelFlight texture_atlas"}, "scene": 0, "scenes": [{"nodes": [0]}],
         "nodes": [{"mesh": 0}],
         "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
         "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 1}}],
         "textures": [{"source": 0, "sampler": 0}], "samplers": [{"magFilter": 9729, "minFilter": 9987 if a.packing == "xatlas" else 9729}],
         "images": [{"bufferView": 3, "mimeType": "image/jpeg"}], "buffers": [{"byteLength": len(blob)}], "bufferViews": views,
         "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(P), "type": "VEC3", "min": P.min(0).tolist(), "max": P.max(0).tolist()},
                       {"bufferView": 1, "componentType": 5126, "count": len(UVs), "type": "VEC2"},
                       {"bufferView": 2, "componentType": 5125, "count": int(I.size), "type": "SCALAR"}]}
    js = json.dumps(g).encode(); js += b" " * (-len(js) % 4)
    with open(path, "wb") as fo:
        fo.write(struct.pack("<III", 0x46546C67, 2, 12 + 8 + len(js) + 8 + len(blob)))
        fo.write(struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob)
glb(a.out + "/textured_mesh.glb", Vn, uv_gl, F, jpg)
glb(a.out + "/textured_mesh_web.glb", Vn, uv_gl, F, jpg, y_up=False)
rep = {"triangles": int(len(F)), "uv_vertices": int(len(vmap)), "atlas_px": S, "textured_triangle_fraction": float((best >= 0).mean()),
       "frames_used": int(len(np.unique(best[best >= 0]))), "glb_bytes": os.path.getsize(a.out + "/textured_mesh.glb"),
       "atlas_jpeg_bytes": len(jpg), "seconds": time.time() - t0, "source_run": a.run}
json.dump(rep, open(a.out + "/texture_report.json", "w"), indent=1); log(rep)
