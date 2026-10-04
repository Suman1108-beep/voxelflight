"""Surface from a trained PGSR (planar Gaussian splatting) model: render unbiased plane depth at every training camera,
drop grazing-angle pixels, fuse with the pipeline's tiled GPU TSDF and export, in the GNSS-locked frame used by prep_gs.py.
Run with PGSR on PYTHONPATH:  python gs_fuse.py -m <pgsr model dir>   (settings via GS_OUT, GS_RES, GS_DEPTH_MAX, GS_SRC_RUN)."""
import json, os, shutil, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from scene import Scene
from scene.gaussian_model import GaussianModel
from gaussian_renderer import render

out_dir, src = os.environ["GS_OUT"], os.environ["GS_SRC_RUN"]
res, depth_max = int(os.environ.get("GS_RES", "1")), float(os.environ.get("GS_DEPTH_MAX", "35"))
parser = ArgumentParser(); mp = ModelParams(parser, sentinel=True); pp = PipelineParams(parser)
parser.add_argument("--iteration", default=-1, type=int); args = get_combined_args(parser)
ds = mp.extract(args); ds.data_device = "cpu"; ds.resolution = res
gaussians = GaussianModel(ds.sh_degree); scene = Scene(ds, gaussians, load_iteration=args.iteration, shuffle=False)
bg = torch.zeros(3, device="cuda"); views = []
with torch.no_grad():
    for cam in scene.getTrainCameras():
        out = render(cam, gaussians, pp.extract(args), bg)
        depth = out["plane_depth"].squeeze()
        ray = torch.nn.functional.normalize(cam.get_rays(), dim=-1)
        dn = torch.nn.functional.normalize(out["depth_normal"].permute(1, 2, 0), dim=-1)
        depth[(ray * dn).sum(-1).abs() < np.cos(np.radians(80))] = 0           # grazing views of a surface are unreliable
        depth[(depth > depth_max) | ~torch.isfinite(depth)] = 0
        w2c = np.eye(4); w2c[:3, :3] = cam.R.T; w2c[:3, 3] = cam.T
        K = np.array([[cam.Fx, 0, cam.Cx], [0, cam.Fy, cam.Cy], [0, 0, 1]])
        views.append({"depth": depth.cpu().numpy().astype(np.float32), "K": K, "c2w": np.linalg.inv(w2c),
                      "color": (out["render"].clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)})
print("rendered", len(views), "views at", views[0]["depth"].shape, flush=True)
del gaussians, scene; torch.cuda.empty_cache()

from fusion import gpu_tsdf_tiled, mesh_stats
from export import export_all
prep = json.load(open(os.path.join(ds.source_path, "prep.json")))
mesh, pcd = gpu_tsdf_tiled(views, tile=60.0, voxel=0.05, depth_max=depth_max); mesh.compute_vertex_normals()
P, C = np.asarray(pcd.points), np.asarray(pcd.colors)
meta = {"ground_truth_used": False, "survey_or_lidar_used": False, "pipeline": "VoxelFlight v2 + PGSR planar Gaussian splatting surface"}
os.makedirs(out_dir, exist_ok=True)
export_all(out_dir + "/model", mesh, P, C, np.array(prep["utm_origin"]), prep["epsg"], extra_meta=meta, formats=("ply", "las", "tif"))
shutil.copy(os.path.join(src, "keyframe_poses.npz"), out_dir)
json.dump({"schema": "voxelflight.v2.pgsr", "views": len(views), "render_resolution": list(views[0]["depth"].shape[::-1]),
           "mesh": mesh_stats(mesh), "pgsr_model": args.model_path, "ground_truth_used": False}, open(out_dir + "/report.json", "w"), indent=1)
print("done", mesh_stats(mesh))
