#!/usr/bin/env python3
"""Real pretrained MapAnything inference on Apple MPS or NVIDIA CUDA.

Bounded windows trade global consistency and resolution for 16 GB compatibility.
All run metrics are newly measured; accuracy is explicitly unmeasured without a reference.
"""
from __future__ import annotations
import argparse
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parent
os.environ.setdefault("HF_HOME",str(ROOT/"cache-mac/huggingface"))
os.environ.setdefault("TORCH_HOME",str(ROOT/"cache-mac/torch"))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK","1")
os.environ.setdefault("OMP_NUM_THREADS","4")
sys.path.insert(0,str(ROOT/"src"))
MODEL_ID="facebook/map-anything-apache"
MODEL_REVISION="00f9c245bbcb60522d1ed7f9e9d88462c6e3f38a"


def calibration(path):
    """Explicit pinhole intrinsics; never assume another flight's calibration."""
    import numpy as np
    if path is None:return None
    value=json.loads(path.read_text())
    if str(value.get("model","pinhole")).lower() not in {"pinhole","opencv"}:
        raise ValueError("This Mac mode supports OpenCV pinhole calibration, not fisheye models.")
    width,height=int(value["width"]),int(value["height"])
    if "intrinsic_matrix" in value:k=np.array(value["intrinsic_matrix"],dtype=np.float32)
    else:
        vals=value.get("intrinsics",value.get("calibration"))
        fx,fy,cx,cy=(vals[:4] if vals is not None else [value[n] for n in ("fx","fy","cx","cy")])
        k=np.array([[fx,0,cx],[0,fy,cy],[0,0,1]],np.float32)
    legacy=value.get("calibration",[])
    d=np.array(value.get("distortion_coefficients",legacy[4:]),dtype=np.float32)
    if width<=0 or height<=0 or k.shape!=(3,3) or not np.isfinite(k).all() or k[0,0]<=0 or k[1,1]<=0 or not np.allclose(k[2],[0,0,1]):
        raise ValueError("Invalid calibration dimensions or intrinsic matrix.")
    if d.size not in {0,4,5,8,12,14} or not np.isfinite(d).all():raise ValueError("Invalid OpenCV distortion coefficients.")
    return width,height,k,d


def write_json(path,value):
    temporary=path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False))
    temporary.replace(path)


def load_visual_pose_priors(path,frames):
    """Accept auditable, non-metric SfM cameras; never consume reference poses."""
    import numpy as np
    value=json.loads(path.read_text())
    if value.get("ground_truth_used") is not False or value.get("metric_scale_established") is not False:
        raise ValueError("Pose priors must explicitly be non-ground-truth, non-metric visual estimates.")
    by_name={}
    for item in value["poses"]:
        name=item["image"];matrix=np.asarray(item["camera_to_world"],dtype=np.float32)
        if name in by_name:raise ValueError("Duplicate image in visual pose priors.")
        if matrix.shape!=(4,4) or not np.isfinite(matrix).all() or not np.allclose(matrix[3],[0,0,0,1],atol=1e-5):
            raise ValueError("Invalid camera-to-world matrix.")
        rotation=matrix[:3,:3]
        if not np.allclose(rotation.T@rotation,np.eye(3),atol=1e-4) or not np.isclose(np.linalg.det(rotation),1,atol=1e-4):
            raise ValueError("Pose rotation must be a proper orthonormal rotation.")
        by_name[name]=matrix
    for frame in frames:
        name=frame.get("source_name")
        if name not in by_name:raise ValueError(f"Visual pose prior missing for source image: {name}")
        frame["input_camera_pose"]=by_name[name].tolist()
    return dict(source=str(path.resolve()),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        method=value.get("method"),input_scale="arbitrary SfM scale",ground_truth_used=False,
        metric_scale_established=False,selected_cameras=len(frames))


def progress(out,stage,fraction,detail):
    write_json(out/"progress.json",dict(stage=stage,progress=fraction,detail=detail,updated_at=time.time()))
    print(f"[{fraction:.0%}] {stage}: {detail}",flush=True)


def decode_selected_video_frames(cap,indices):
    """Decode a long video once, retrieving RGB only at selected frame indices.

    Repeated random H.264 seeks can re-decode the same GOP hundreds of times
    over a ten-minute flight. `grab` advances the decoder without transferring
    every unselected frame into a Python image array.
    """
    cursor=0
    for number in indices:
        number=int(number)
        if number<cursor:
            raise ValueError("Video sample indices must be strictly increasing.")
        while cursor<=number:
            if not cap.grab():
                raise ValueError(f"Video decoding ended before source frame {number}.")
            cursor+=1
        ok,img=cap.retrieve()
        if not ok:
            raise ValueError(f"Video frame {number} could not be retrieved.")
        yield number,img


def extract(args,out):
    import cv2
    import numpy as np
    from sih3d.ingest import probe_video
    folder=out/"frames";folder.mkdir()
    frames=[]
    camera=calibration(args.calibration)
    if args.video:
        video=probe_video(args.video)
        if camera and (video.width,video.height)!=camera[:2]:
            raise ValueError("Calibration width and height must match the decoded video exactly.")
        if video.duration_s>1800:
            raise ValueError("This Mac preview accepts videos up to 30 minutes. Split longer footage.")
        indices=np.unique(np.linspace(0,video.frame_count-1,min(args.max_frames,video.frame_count)).round().astype(int))
        cap=cv2.VideoCapture(str(args.video))
        try:
            for i,(number,img) in enumerate(decode_selected_video_frames(cap,indices)):
                if camera and camera[3].size:img=cv2.undistort(img,camera[2],camera[3])
                scale=min(1,1280/max(img.shape[:2]));img=cv2.resize(img,None,fx=scale,fy=scale) if scale<1 else img
                path=folder/f"frame_{i:05d}.jpg";cv2.imwrite(str(path),img,[cv2.IMWRITE_JPEG_QUALITY,94])
                frames.append(dict(index=i,source_frame=int(number),timestamp_s=float(number/video.fps),path=str(path)))
                if camera:
                    k=camera[2].copy();k[0]*=img.shape[1]/video.width;k[1]*=img.shape[0]/video.height;frames[-1]["input_intrinsics"]=k.tolist()
        finally:
            cap.release()
        info=dict(name=args.video.name,duration_s=video.duration_s,width=video.width,height=video.height,source_frames=video.frame_count)
    else:
        from PIL import Image,ImageOps
        images=sorted(p for p in args.images.iterdir() if p.suffix.lower() in {".jpg",".jpeg",".png"})
        if len(images)<2:
            raise ValueError("At least two overlapping images are required.")
        source_count=len(images)
        images=[images[i] for i in np.unique(np.linspace(0,len(images)-1,min(args.max_frames,len(images))).round().astype(int))]
        for i,path in enumerate(images):
            with Image.open(path) as raw:
                im=ImageOps.exif_transpose(raw).convert("RGB")
                if camera and im.size!=camera[:2]:raise ValueError("Calibration dimensions must match every source image.")
                if camera and camera[3].size:im=Image.fromarray(cv2.undistort(np.asarray(im),camera[2],camera[3]))
                im.thumbnail((1280,1280))
                saved=folder/f"frame_{i:05d}.jpg";im.save(saved,quality=94)
                resized_size=im.size
            frames.append(dict(index=i,source_frame=None,timestamp_s=None,path=str(saved),source_name=path.name))
            if camera:
                k=camera[2].copy();k[0]*=resized_size[0]/camera[0];k[1]*=resized_size[1]/camera[1];frames[-1]["input_intrinsics"]=k.tolist()
        video=None;info=dict(name=args.images.name,duration_s=None,source_frames=source_count,input_type="image_sequence")
    if len(frames)<2:
        raise ValueError("At least two decodable overlapping views are required, not a single image.")
    return frames,video,info


def georeference(args,video,frames,poses,warnings):
    """Optional noisy-GPS alignment, not a spatial-accuracy test or surveyed datum."""
    import numpy as np
    from mac_geometry import similarity
    if not args.telemetry:
        return None
    if video is None:
        raise ValueError("Telemetry requires video timestamps in this processing mode.")
    from sih3d.ingest import read_telemetry,canonicalise_telemetry
    from pyproj import Transformer
    rows,meta=canonicalise_telemetry(read_telemetry(args.telemetry),video)
    if meta.get("timestamp_inferred"):
        warnings.append("GPS timestamps were inferred. Georeferencing withheld until synchronization is supplied.")
        return None
    if any(row.get("altitude_m") is None for row in rows):
        warnings.append("GPS altitude is missing. No 3D georeferencing was applied.")
        return None
    rows=sorted(rows,key=lambda r:r["timestamp_s"])
    ts=np.array([r["timestamp_s"] for r in rows]);times=np.array([f["timestamp_s"] for f in frames])
    if len(np.unique(ts))!=len(ts) or times.min()<ts.min()-.1 or times.max()>ts.max()+.1:
        warnings.append("GPS timestamps repeat or do not cover the sampled video. No georeferencing was applied.")
        return None
    lat=float(np.median([r["latitude"] for r in rows]));lon=float(np.median([r["longitude"] for r in rows]))
    epsg=(32600 if lat>=0 else 32700)+min(60,max(1,int((lon+180)//6)+1))
    project=Transformer.from_crs(4326,epsg,always_xy=True)
    east,north=project.transform([r["longitude"] for r in rows],[r["latitude"] for r in rows])
    gps=np.stack([np.interp(times,ts,east),np.interp(times,ts,north),np.interp(times,ts,[r["altitude_m"] for r in rows])],1)
    origin=np.median(gps,axis=0)
    try:
        s,r,t,quality=similarity(poses[:,:3,3],gps-origin)
        singular=np.linalg.svd(gps-gps.mean(0),compute_uv=False)
        if singular[1]/max(singular[0],1e-9)<.015:
            raise ValueError("GPS path is too straight to reliably constrain 3D orientation.")
        if not .1<s<10:
            raise ValueError("GPS requires an implausibly large metric-scale correction.")
    except ValueError as error:
        warnings.append(str(error)+" Georeferencing was withheld.")
        return None
    warnings.append("GPS alignment is provisional: timestamp origin and altitude datum need verification. Its fit residual is NOT independent accuracy.")
    return dict(scale=s,rotation=r,translation=t,origin=origin,epsg=epsg,gps_fit=quality)


def export_gis(out,points,point_colors,geo):
    import numpy as np
    artifacts=[]
    import laspy
    from pyproj import CRS
    header=laspy.LasHeader(point_format=3,version="1.2");header.offsets=geo["origin"];header.scales=np.array([.001]*3);header.add_crs(CRS.from_epsg(geo["epsg"]))
    las=laspy.LasData(header);absolute=points+geo["origin"]
    las.x,las.y,las.z=absolute.T;las.red,las.green,las.blue=(point_colors.astype(np.uint16)*257).T
    las.write(out/"reconstruction_utm.las");artifacts.append("reconstruction_utm.las")
    # DSM contains only occupied observed cells. No hole filling or invented terrain.
    import rasterio
    from rasterio.transform import from_origin
    west,south=absolute[:,:2].min(0);east,north=absolute[:,:2].max(0)
    resolution=max(.25,float(np.sqrt((east-west)*(north-south)/4_000_000)),(east-west)/8191,(north-south)/8191)
    width=int(np.ceil((east-west)/resolution))+1;height=int(np.ceil((north-south)/resolution))+1
    surface=np.full((height,width),-9999,np.float32)
    ix=np.clip(((absolute[:,0]-west)/resolution).astype(int),0,width-1);iy=np.clip(((north-absolute[:,1])/resolution).astype(int),0,height-1)
    np.maximum.at(surface,(iy,ix),absolute[:,2])
    with rasterio.open(out/"surface_model.tif","w",driver="GTiff",width=width,height=height,count=1,dtype="float32",crs=f"EPSG:{geo['epsg']}",transform=from_origin(west,north,resolution,resolution),nodata=-9999,compress="deflate") as dataset:dataset.write(surface,1)
    artifacts.append("surface_model.tif")
    return artifacts


def run(args):
    import numpy as np
    import torch
    from mac_geometry import similarity,transform_points,depth_mesh,voxel_fuse,camera_similarity
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Output folder must be empty; existing results will never be overwritten.")
    out.mkdir(parents=True,exist_ok=True)
    args._owns_output=True
    begin=time.monotonic();warnings=["No independent surface or trajectory accuracy measured in this inference run.","Sparse keyframe sampling may miss visible surfaces. Unobserved surfaces are left empty.","No semantic segmentation, inertial fusion, bundle adjustment or Gaussian-splat training in this mode."]
    torch.set_num_threads(4)
    device="cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else None
    if device is None:
        raise RuntimeError("A CUDA or Apple MPS GPU is required for this reconstruction mode.")
    if device=="mps":torch.mps.set_per_process_memory_fraction(.8)
    else:torch.cuda.reset_peak_memory_stats()
    progress(out,"Preparing inputs",.02,"Decoding actual input views")
    frames,video,info=extract(args,out)
    prior_path=getattr(args,"visual_pose_priors",None)
    pose_prior_info=None
    if prior_path:
        if video is not None or not args.calibration:
            raise ValueError("Visual pose conditioning requires calibrated images.")
        pose_prior_info=load_visual_pose_priors(prior_path,frames)
        warnings=[warning.replace("No semantic segmentation, inertial fusion, bundle adjustment", "No semantic segmentation or inertial fusion") for warning in warnings]
        warnings.append("Camera priors come from separate visual bundle adjustment, not ground truth. SfM input scale is arbitrary; output metric scale remains model-estimated and unverified.")
    progress(out,"Loading model",.06,f"Loading cached pretrained MapAnything weights onto {device}")
    from huggingface_hub import snapshot_download
    from mapanything.models import MapAnything
    from mapanything.utils.image import load_images,preprocess_inputs
    model_path=snapshot_download(MODEL_ID,revision=MODEL_REVISION,local_files_only=True,allow_patterns=["config.json","model.safetensors"])
    # Force the encoder's code dependency to local disk too; HF offline mode alone
    # does not control torch.hub's GitHub calls.
    from unittest.mock import patch
    hub_load=torch.hub.load
    encoder_path=Path(os.environ.get("DINO_SOURCE",str(ROOT/"cache-mac/torch/hub/facebookresearch_dinov2_main")))
    if not (encoder_path/"hubconf.py").is_file():raise RuntimeError("DINOv2 source cache missing. Run setup_mac.py while online first.")
    def local_encoder(repo,entry,*positional,**kwargs):
        if repo!="facebookresearch/dinov2":raise RuntimeError("Unexpected model code dependency; refusing an online fetch.")
        kwargs.pop("force_reload",None);kwargs.pop("source",None)
        return hub_load(str(encoder_path),entry,*positional,source="local",**kwargs)
    with patch("torch.hub.load",local_encoder):
        model=MapAnything.from_pretrained(model_path,local_files_only=True).to(device).eval()
    if device=="mps":torch.mps.empty_cache()
    else:torch.cuda.empty_cache()
    load_seconds=time.monotonic()-begin
    kept={};joins=[];window_size=args.window;step=window_size-args.overlap
    visual_anchors=None;anchor_scale=None
    if pose_prior_info:
        # One shared SfM frame for all windows; no independent point-cloud joins.
        priors=np.asarray([frame["input_camera_pose"] for frame in frames],np.float64)
        axis=np.eye(4);axis[:3,:3]=[[1,0,0],[0,0,1],[0,-1,0]]
        visual_anchors=axis@np.linalg.inv(priors[0])@priors
    starts=list(range(0,max(1,len(frames)-2),step))
    for wi,start in enumerate(starts):
        ids=list(range(start,min(start+window_size,len(frames))))
        progress(out,"GPU inference",.1+.65*wi/len(starts),f"Window {wi+1}/{len(starts)} · {len(ids)} views · {args.size}px")
        if args.calibration:
            from PIL import Image
            inputs=[]
            for i in ids:
                with Image.open(frames[i]["path"]) as im:
                    inputs.append(dict(img=np.array(im.convert("RGB")),intrinsics=np.array(frames[i]["input_intrinsics"],np.float32)))
                if pose_prior_info:
                    inputs[-1].update(camera_poses=np.array(frames[i]["input_camera_pose"],np.float32),is_metric_scale=False)
            views=preprocess_inputs(inputs,resize_mode="longest_side",size=args.size,verbose=False)
        else:views=load_images([frames[i]["path"] for i in ids],resize_mode="longest_side",size=args.size,verbose=False)
        with torch.inference_mode():
            predictions=model.infer(views,memory_efficient_inference=True,minibatch_size=1,use_amp=True,amp_dtype="bf16" if device=="cuda" else "fp16",apply_mask=True,mask_edges=True,apply_confidence_mask=True,confidence_percentile=20)
        local={}
        for i,pred in zip(ids,predictions):
            cpu=lambda key:pred[key][0].detach().float().cpu().numpy()
            p=cpu("pts3d");depth=cpu("depth_z").squeeze();mask=cpu("mask").squeeze()>0
            mask &= np.isfinite(p).all(-1)&np.isfinite(depth)&(depth>0)
            if mask.sum()<100:
                raise RuntimeError("Too few valid depth predictions. Try clearer, more overlapping frames.")
            local[i]=dict(points=p,depth=depth,mask=mask,color=np.clip(cpu("img_no_norm")*255,0,255).astype(np.uint8),pose=cpu("camera_poses"),intrinsics=cpu("intrinsics"))
        del predictions,views
        if visual_anchors is not None:
            current=np.stack([local[i]["pose"] for i in ids])
            if anchor_scale is None:
                to_sfm,_,_,_=camera_similarity(current,visual_anchors[ids])
                anchor_scale=1/to_sfm
                visual_anchors[:,:3,3]*=anchor_scale
                pose_prior_info["learned_m_per_sfm_unit"]=anchor_scale
            s,r,t,quality=camera_similarity(current,visual_anchors[ids])
            if not .05<s<20:raise RuntimeError('Visual camera windows disagree on scale by more than 20×.')
            joins.append(dict(window=wi,scale=s,method="Shared visual SfM cameras; no ground truth",**quality))
        elif wi:
            sources=[];targets=[]
            for i in ids:
                if i not in kept:continue
                mask=local[i]["mask"]&kept[i]["mask"]
                sources.append(local[i]["points"][mask][::4]);targets.append(kept[i]["points"][mask][::4])
            s,r,t,quality=similarity(np.concatenate(sources),np.concatenate(targets))
            if not .2<s<5:
                raise RuntimeError("Adjacent windows disagree on scale. Use a shorter clip or more keyframes.")
            joins.append(dict(window=wi,scale=s,**quality))
        else:
            # OpenCV camera coordinates -> viewer's Z-up local frame. Not geographic north.
            s=1.;r=np.array([[1,0,0],[0,0,1],[0,-1,0.]]);t=np.zeros(3)
        for i,pred in local.items():
            if i in kept:continue
            pred["points"]=transform_points(pred["points"],s,r,t).astype(np.float32)
            pred["pose"][:3,3]=transform_points(pred["pose"][:3,3],s,r,t)
            pred["pose"][:3,:3]=r@pred["pose"][:3,:3]
            pred["depth"]*=s
            kept[i]=pred
        del local
        gc.collect()
        if device=="mps":torch.mps.empty_cache()
        else:torch.cuda.empty_cache()
    peak_gpu=float((torch.cuda.max_memory_allocated() if device=="cuda" else torch.mps.driver_allocated_memory())/1024**3)
    del model;gc.collect()
    if device=="mps":torch.mps.empty_cache()
    else:torch.cuda.empty_cache()
    progress(out,"Fusing geometry",.79,"Combining observed surfaces and retaining camera poses")
    poses=np.stack([kept[i]["pose"] for i in range(len(frames))])
    geo=georeference(args,video,frames,poses,warnings)
    # Optional reproducible geometry cache. This is never served by the public API.
    # Save camera-frame depth before georeferencing so refusion cannot accidentally
    # apply the GPS transform twice. All cached poses share the visual world frame.
    if getattr(args,"save_predictions",False):
        cache=out/"predictions";cache.mkdir()
        for i,pred in kept.items():
            np.savez_compressed(cache/f"frame_{i:05d}.npz",depth=pred["depth"],
                mask=pred["mask"],color=pred["color"],pose=pred["pose"],
                intrinsics=pred["intrinsics"],points=pred["points"])
    from PIL import Image
    point_parts=[];color_parts=[];vertices=[];colors=[];faces=[];uvs=[];offset=0
    cols=int(np.ceil(np.sqrt(len(kept))));rows=int(np.ceil(len(kept)/cols))
    tile_w=max(p["color"].shape[1] for p in kept.values());tile_h=max(p["color"].shape[0] for p in kept.values())
    atlas=Image.new("RGB",(cols*tile_w,rows*tile_h))
    for i,pred in kept.items():
        if geo:
            pred["points"]=transform_points(pred["points"],geo["scale"],geo["rotation"],geo["translation"]).astype(np.float32)
            poses[i,:3,3]=transform_points(poses[i,:3,3],geo["scale"],geo["rotation"],geo["translation"])
            poses[i,:3,:3]=geo["rotation"]@poses[i,:3,:3]
        p,c,f,uv=depth_mesh(pred["points"],pred["color"],pred["depth"],pred["mask"],stride=2,return_uv=True)
        atlas.paste(Image.fromarray(pred["color"]),(i%cols*tile_w,i//cols*tile_h))
        image_h,image_w=pred["color"].shape[:2]
        uv[:,0]=(uv[:,0]*image_w+i%cols*tile_w)/atlas.width
        uv[:,1]=1-((1-uv[:,1])*image_h+i//cols*tile_h)/atlas.height
        uvs.append(uv)
        vertices.append(p);colors.append(c);faces.append(f+offset);offset+=len(p)
        point_parts.append(pred["points"][pred["mask"]]);color_parts.append(pred["color"][pred["mask"]])
    points,point_colors=voxel_fuse(np.concatenate(point_parts),np.concatenate(color_parts))
    import trimesh
    from plyfile import PlyData,PlyElement
    mesh=trimesh.Trimesh(vertices=np.concatenate(vertices),faces=np.concatenate(faces),vertex_colors=np.concatenate(colors),process=False)
    if not len(mesh.faces):raise RuntimeError("No supported surface triangles were reconstructed.")
    mesh.export(out/"reconstruction.obj")  # Self-contained colored OBJ, no missing MTL dependencies.
    mesh.visual=trimesh.visual.texture.TextureVisuals(uv=np.concatenate(uvs),image=atlas)
    mesh.export(out/"reconstruction_mesh.glb")  # Source-image atlas is embedded in the GLB.
    cloud=np.empty(len(points),dtype=[("x","<f4"),("y","<f4"),("z","<f4"),("red","u1"),("green","u1"),("blue","u1")])
    for j,k in enumerate(("x","y","z")):cloud[k]=points[:,j]
    for j,k in enumerate(("red","green","blue")):cloud[k]=point_colors[:,j]
    PlyData([PlyElement.describe(cloud,"vertex")],text=False).write(str(out/"pointcloud.ply"))
    np.savez_compressed(out/"camera_poses.npz",camera_to_world=poses,intrinsics=np.stack([kept[i]["intrinsics"] for i in range(len(frames))]))
    with (out/"trajectory.csv").open("w",newline="") as stream:
        writer=csv.writer(stream);writer.writerow(["frame","timestamp_s","x","y","z"])
        for i,pose in enumerate(poses):writer.writerow([i,frames[i]["timestamp_s"],*pose[:3,3]])
    artifacts=["reconstruction_mesh.glb","pointcloud.ply","reconstruction.obj","camera_poses.npz","trajectory.csv"]
    if geo:artifacts.extend(export_gis(out,points,point_colors,geo))
    progress(out,"Packaging results",.96,"Writing new-run provenance and honest validation status")
    duration=time.monotonic()-begin
    report=dict(schema="sih3d.gpu-run.v2",input=info,device=device,model=MODEL_ID,model_revision=MODEL_REVISION,model_code_revision="3d10cf7a3016fc0f9bb13a071ee66c47b10be0d9",torch_version=torch.__version__,inference_resolution=args.size,window_size=window_size,overlap=args.overlap,keyframes=len(frames),windows=len(starts),points=len(points),vertices=len(mesh.vertices),triangles=len(mesh.faces),runtime_s=duration,load_and_input_s=load_seconds,peak_gpu_allocation_gib=peak_gpu,ground_truth_used=False,training_performed=False,coordinate_system=f"EPSG:{geo['epsg']} local offset" if geo else "Local Z-up / learned metric scale; north unknown",georeferenced=bool(geo),utm_origin=geo["origin"].tolist() if geo else None,gps_fit_residual=geo["gps_fit"] if geo else None,trajectory_rmse_m=None,surface_rmse_m=None,sih_requirements_verified=False,alignment=joins,warnings=warnings,artifacts=artifacts+["report.json"],frames=[{k:v for k,v in frame.items() if k!="path"} for frame in frames],trajectory=[dict(frame=i,position=pose[:3,3].tolist()) for i,pose in enumerate(poses)])
    report["checksums"]={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in artifacts}
    report["calibration_used"]=bool(args.calibration)
    report["visual_pose_priors"]=pose_prior_info
    report["prediction_cache_saved"]=bool(getattr(args,"save_predictions",False))
    report["appearance"]="Embedded atlas sampled from actual input pixels; no generated textures or GANs"
    if not args.calibration:warnings.append("Camera intrinsics were estimated by the model; no measured calibration was supplied.")
    write_json(out/"report.json",report)
    progress(out,"Complete",1,f"{len(points):,} points · {len(mesh.faces):,} triangles · {duration:.0f}s")
    return report


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    source=p.add_mutually_exclusive_group(required=True);source.add_argument("--video",type=Path);source.add_argument("--images",type=Path)
    p.add_argument("--telemetry",type=Path);p.add_argument("--calibration",type=Path);p.add_argument("--output",type=Path,required=True)
    p.add_argument("--max-frames",type=int,default=24,choices=range(2,721));p.add_argument("--size",type=int,default=350,choices=[224,252,350,420,518]);p.add_argument("--window",type=int,default=4,choices=[3,4,6,12,24,32,48],help="Views inferred jointly; larger windows require more GPU memory.")
    p.add_argument("--overlap",type=int,default=2,help="Shared views between adjacent windows; must be between 2 and window-1.")
    p.add_argument("--save-predictions",action="store_true",help="Keep model depths, poses and masks for auditable offline geometry refusion.")
    p.add_argument("--visual-pose-priors",type=Path,help="Experimental calibrated SfM report: non-metric, non-reference camera priors shared across inference windows.")
    args=p.parse_args()
    if not 2<=args.overlap<args.window:p.error("--overlap must be at least 2 and smaller than --window")
    try:run(args)
    except Exception as error:
        if getattr(args,"_owns_output",False):progress(args.output,"Failed",0,str(error))
        raise
