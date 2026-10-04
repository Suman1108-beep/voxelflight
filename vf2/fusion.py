"""GPU TSDF fusion (Open3D tensor VoxelBlockGrid on CUDA) + mesh cleanup."""
import numpy as np


def gpu_tsdf(views, voxel=0.05, trunc_mult=6.0, depth_max=60.0, block_res=8, block_count=int(__import__("os").environ.get("VF_BLOCKS", 420_000)),
             weight_threshold=float(__import__("os").environ.get("VF_WEIGHT", 2.0)), device="CUDA:0", log=print):
    """views: iterable of dict(depth HxW float32 metres (0=invalid), color HxWx3 uint8, K 3x3, c2w 4x4)."""
    import open3d as o3d
    import open3d.core as o3c
    dev = o3c.Device(device)
    vbg = o3d.t.geometry.VoxelBlockGrid(
        attr_names=("tsdf", "weight", "color"),
        attr_dtypes=(o3c.float32, o3c.float32, o3c.float32),
        attr_channels=((1), (1), (3)),
        voxel_size=voxel, block_resolution=block_res, block_count=block_count, device=dev)
    n = 0
    for v in views:
        depth = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(v["depth"], np.float32))).to(dev)
        color = o3d.t.geometry.Image(o3c.Tensor(np.ascontiguousarray(v["color"], np.float32) / 255.0)).to(dev)
        K = o3c.Tensor(np.asarray(v["K"], np.float64))
        ext = o3c.Tensor(np.linalg.inv(np.asarray(v["c2w"], np.float64)))
        if not np.any((v["depth"] > 0) & (v["depth"] < depth_max)):
            continue  # nothing within range for this view (e.g. tile-masked to far pixels only)
        try:
            blocks = vbg.compute_unique_block_coordinates(depth, K, ext, 1.0, depth_max, trunc_mult)
        except RuntimeError as e:
            if "No block is touched" in str(e):
                continue
            raise
        vbg.integrate(blocks, depth, color, K, K, ext, 1.0, depth_max, trunc_mult)
        n += 1
    log(f"integrated {n} views; active blocks {vbg.hashmap().size()}")
    try:
        mesh = vbg.extract_triangle_mesh(weight_threshold=weight_threshold, estimated_vertex_number=-1)
        pcd = vbg.extract_point_cloud(weight_threshold=weight_threshold)
    except RuntimeError as e:  # marching-cubes scratch memory exhausted on GPU -> extract on CPU
        log(f"GPU extraction failed ({str(e)[:80]}...), extracting on CPU")
        vbg = vbg.cpu()
        mesh = vbg.extract_triangle_mesh(weight_threshold=weight_threshold, estimated_vertex_number=-1)
        pcd = vbg.extract_point_cloud(weight_threshold=weight_threshold)
    mesh, pcd = mesh.to_legacy(), pcd.to_legacy()
    del vbg
    if "CUDA" in str(dev):
        o3c.cuda.release_cache()   # free this tile's grid before the next tile allocates its own
    return mesh, pcd


def _world_xy(v, step=4):
    """World XY of (subsampled) valid depth pixels, for tile assignment."""
    d = v["depth"]; H, W = d.shape
    ys, xs = np.mgrid[0:H:step, 0:W:step]
    z = d[ys, xs]; ok = z > 0
    K = np.asarray(v["K"]); c2w = np.asarray(v["c2w"])
    x = (xs[ok] - K[0, 2]) / K[0, 0] * z[ok]; y = (ys[ok] - K[1, 2]) / K[1, 1] * z[ok]
    P = np.column_stack([x, y, z[ok]]) @ c2w[:3, :3].T + c2w[:3, 3]
    return P[:, :2]


def _mask_to_tile(v, lo, hi):
    d = v["depth"]; H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W]
    K = np.asarray(v["K"]); c2w = np.asarray(v["c2w"])
    x = (xs - K[0, 2]) / K[0, 0] * d; y = (ys - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1) @ c2w[:3, :3].T + c2w[:3, 3]
    inside = (P[..., 0] >= lo[0]) & (P[..., 0] < hi[0]) & (P[..., 1] >= lo[1]) & (P[..., 1] < hi[1])
    return {**v, "depth": np.where(inside & (d > 0), d, 0).astype(np.float32)}


def _pixel_xy(v):
    """World XY of every pixel (float32 HxWx2); invalid depth -> NaN."""
    d = v["depth"]; H, W = d.shape
    ys, xs = np.mgrid[0:H, 0:W].astype(np.float32)
    K = np.asarray(v["K"], np.float32); c2w = np.asarray(v["c2w"], np.float32)
    x = (xs - K[0, 2]) / K[0, 0] * d; y = (ys - K[1, 2]) / K[1, 1] * d
    P = np.stack([x, y, d], -1) @ c2w[:2, :3].T + c2w[:2, 3]
    P[d <= 0] = np.nan
    return P


def gpu_tsdf_tiled(views, tile=60.0, margin=1.0, clean_min_triangles=300, log=print, **kw):
    """Spatially tiled TSDF: bounded GPU memory for arbitrarily large scenes. Each tile is cropped exactly to its
    bounds (no duplicate surfaces) and fragment-cleaned in a worker thread while the GPU fuses the next tile."""
    import open3d as o3d
    from concurrent.futures import ThreadPoolExecutor
    views = list(views)
    pxy = [_pixel_xy(v) for v in views]
    allxy = np.concatenate([p[::4, ::4].reshape(-1, 2) for p in pxy]); allxy = allxy[np.isfinite(allxy[:, 0])]
    lo = np.percentile(allxy, 0.1, axis=0) - 1.0; hi = np.percentile(allxy, 99.9, axis=0) + 1.0
    nx, ny = max(1, int(np.ceil((hi[0] - lo[0]) / tile))), max(1, int(np.ceil((hi[1] - lo[1]) / tile)))

    def finish(mesh, pcd, a, b, name):
        V = np.asarray(mesh.vertices); F = np.asarray(mesh.triangles)
        if len(F):
            cen = V[F].mean(1)
            out = ~((cen[:, 0] >= a[0]) & (cen[:, 0] < b[0]) & (cen[:, 1] >= a[1]) & (cen[:, 1] < b[1]))
            mesh.remove_triangles_by_mask(out); mesh.remove_unreferenced_vertices()
            mesh = clean_mesh(mesh, clean_min_triangles)
        P = np.asarray(pcd.points)
        keep = (P[:, 0] >= a[0]) & (P[:, 0] < b[0]) & (P[:, 1] >= a[1]) & (P[:, 1] < b[1])
        log(f"tile {name}: {len(mesh.triangles)} triangles after crop+clean")
        return mesh, pcd.select_by_index(np.where(keep)[0])

    futures = []
    with ThreadPoolExecutor(4) as pool:
        for ix in range(nx):
            for iy in range(ny):
                a = lo + np.array([ix, iy]) * tile; b = a + tile
                al, bl = a - margin, b + margin
                tv = []
                for v, p in zip(views, pxy):
                    with np.errstate(invalid="ignore"):
                        inside = (p[..., 0] >= al[0]) & (p[..., 0] < bl[0]) & (p[..., 1] >= al[1]) & (p[..., 1] < bl[1])
                    if (inside & (v["depth"] < kw.get("depth_max", 60.0))).sum() > 500:
                        tv.append({**v, "depth": np.where(inside, v["depth"], 0).astype(np.float32)})
                if not tv:
                    continue
                # Open3D sizes its per-grid block scratch table from the first view: integrate the fullest view first
                tv.sort(key=lambda v: -int(np.count_nonzero(v["depth"])))
                mesh, pcd = gpu_tsdf(tv, log=lambda *_: None, **kw)
                futures.append(pool.submit(finish, mesh, pcd, a, b, f"{ix},{iy} ({len(tv)} views)"))
        mesh_all, pcd_all = o3d.geometry.TriangleMesh(), o3d.geometry.PointCloud()
        for f in futures:
            m, p = f.result(); mesh_all += m; pcd_all += p
    log(f"tiles {nx}x{ny}, merged {len(mesh_all.triangles)} triangles")
    return mesh_all, pcd_all


def clean_mesh(mesh, min_component_triangles=200, keep_fraction_of_largest=0.0):
    """Remove tiny floating components (common TSDF noise)."""
    import open3d as o3d
    tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
    tri_cluster = np.asarray(tri_cluster); cluster_n = np.asarray(cluster_n)
    thresh = max(min_component_triangles, keep_fraction_of_largest * cluster_n.max())
    remove = cluster_n[tri_cluster] < thresh
    mesh.remove_triangles_by_mask(remove)
    mesh.remove_unreferenced_vertices()
    return mesh


def mesh_stats(mesh):
    tri_cluster, cluster_n, _ = mesh.cluster_connected_triangles()
    cluster_n = np.asarray(cluster_n)
    return {"triangles": int(len(mesh.triangles)), "vertices": int(len(mesh.vertices)),
            "components": int(len(cluster_n)),
            "largest_component_fraction": float(cluster_n.max() / max(1, cluster_n.sum())) if len(cluster_n) else 0.0}
