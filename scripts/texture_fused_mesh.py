"""Texture a real fused surface from calibrated source cameras, without synthesis.

Select one visible input camera per triangle to avoid cross-view color blur.
Visibility uses the cached predicted depth. Unmatched triangles retain their
original measured vertex colors. This changes appearance, not the geometry.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import trimesh
from PIL import Image


def run(a):
    if a.output.exists() and any(a.output.iterdir()):raise ValueError('Output must be new or empty.')
    source=json.loads((a.predictions/'report.json').read_text())
    fusion=json.loads((a.mesh/'report.json').read_text())
    if source.get('ground_truth_used') is not False or fusion.get('ground_truth_used') is not False:
        raise ValueError('Only non-GT reconstructed geometry is accepted.')
    if Path(fusion['input']).resolve()!=a.predictions.resolve():raise ValueError('Fusion and camera source frames differ.')
    if not source.get('calibration_used'):raise ValueError('Source-image reprojection requires measured calibration.')
    a.output.mkdir(parents=True,exist_ok=True);started=time.monotonic()
    mesh=trimesh.load(a.mesh/'reconstruction_mesh.ply',process=False)
    vertices=np.asarray(mesh.vertices,float);faces=np.asarray(mesh.faces)
    centers=vertices[faces].mean(1);normals=np.asarray(mesh.face_normals)
    best=np.full(len(faces),-np.inf);selected=np.full(len(faces),-1,np.int32)
    cameras=[]
    for i,frame in enumerate(source['frames']):
        with np.load(a.predictions/'predictions'/f'frame_{i:05d}.npz') as raw:
            pose=raw['pose'].astype(float);k=raw['intrinsics'].astype(float);depth=raw['depth'];mask=raw['mask']
        cameras.append((pose,np.asarray(frame['input_intrinsics']),a.predictions/'frames'/f'frame_{i:05d}.jpg'))
        xyz=(centers-pose[:3,3])@pose[:3,:3];z=xyz[:,2];good=z>.05
        indices=np.flatnonzero(good);q=xyz[indices]@k.T;uv=np.rint(q[:,:2]/q[:,2:]).astype(int)
        h,w=depth.shape;inside=(uv[:,0]>=1)&(uv[:,0]<w-1)&(uv[:,1]>=1)&(uv[:,1]<h-1)
        indices,uv=indices[inside],uv[inside]
        measured=depth[uv[:,1],uv[:,0]]
        ok=mask[uv[:,1],uv[:,0]]&(np.abs(measured-z[indices])<(.15+.03*measured))
        indices=indices[ok]
        ray=centers[indices]-pose[:3,3];ray/=np.maximum(np.linalg.norm(ray,axis=1,keepdims=True),1e-8)
        cosine=np.abs(np.sum(normals[indices]*ray,axis=1))
        # Projected area per world area; grazing angles are less useful.
        score=k[0,0]*k[1,1]*cosine/(z[indices]**2)
        choose=(score>best[indices])&(cosine>.15);ids=indices[choose];score=score[choose]
        if len(ids):
            # The whole triangle must lie inside the original image, not just its centroid.
            original_k=cameras[-1][1]
            with Image.open(cameras[-1][2]) as im:iw,ih=im.size
            triangle=(vertices[faces[ids]]-pose[:3,3])@pose[:3,:3]
            proj=triangle@original_k.T;pix=proj[...,:2]/np.maximum(proj[...,2:],1e-8)
            valid=(triangle[...,2]>.05).all(1)&(pix[...,0]>=0).all(1)&(pix[...,0]<iw-1).all(1)&(pix[...,1]>=0).all(1)&(pix[...,1]<ih-1).all(1)
            ids,score=ids[valid],score[valid];best[ids]=score;selected[ids]=i
        if i%15==0:print(f'Checking source-camera visibility {i+1}/{len(source["frames"])}',flush=True)
    valid=selected>=0
    if not valid.any():raise ValueError('No source camera could texture a real triangle.')
    used=np.unique(selected[valid]);cols=int(np.ceil(np.sqrt(len(used))))
    tile=a.tile_width
    with Image.open(cameras[int(used[0])][2]) as im:tile_h=round(tile*im.height/im.width)
    rows=int(np.ceil(len(used)/cols));atlas=Image.new('RGB',(cols*tile,rows*tile_h))
    # Share vertices only within the same camera chart; preserve every original triangle.
    keys=faces[valid]*len(cameras)+selected[valid,None]
    unique,inverse=np.unique(keys,return_inverse=True)
    original_vertex=unique//len(cameras);view_index=unique%len(cameras)
    uv=np.empty((len(unique),2),np.float32)
    for chart,camera_id in enumerate(used):
        pose,k,path=cameras[int(camera_id)];subset=view_index==camera_id
        xyz=(vertices[original_vertex[subset]]-pose[:3,3])@pose[:3,:3];projected=xyz@k.T
        pixel=projected[:,:2]/projected[:,2:]
        with Image.open(path) as im:
            width,height=im.size;tile_image=im.resize((tile,tile_h),Image.Resampling.LANCZOS)
            atlas.paste(tile_image,(chart%cols*tile,chart//cols*tile_h))
        uv[subset,0]=(pixel[:,0]/width*tile+chart%cols*tile)/atlas.width
        uv[subset,1]=1-(pixel[:,1]/height*tile_h+chart//cols*tile_h)/atlas.height
    texture=trimesh.Trimesh(vertices=vertices[original_vertex],faces=inverse.reshape(-1,3),process=False)
    texture.visual=trimesh.visual.texture.TextureVisuals(uv=uv,image=atlas)
    scene=trimesh.Scene();scene.add_geometry(texture,node_name='source_image_textured_surface')
    if (~valid).any():
        fallback=trimesh.Trimesh(vertices=vertices,faces=faces[~valid],vertex_colors=mesh.visual.vertex_colors,process=False)
        fallback.remove_unreferenced_vertices();scene.add_geometry(fallback,node_name='original_vertex_colors_no_visible_source')
    scene.export(a.output/'reconstruction_mesh.glb')
    # Every triangle is preserved. Validate geometry, not just a successful export.
    restored=trimesh.load(a.output/'reconstruction_mesh.glb',force='scene',process=False)
    count=sum(len(g.faces) for g in restored.geometry.values())
    if count!=len(faces):raise RuntimeError('Texturing changed the triangle count.')
    result=dict(schema='voxelflight.source-texturing.v1',input=str(a.mesh.resolve()),
        prediction_source=str(a.predictions.resolve()),method='Depth-tested camera selection and source-image atlas projection',
        ground_truth_used=False,generated_textures=False,geometry_modified=False,
        triangles=count,source_textured_triangles=int(valid.sum()),vertex_color_fallback_triangles=int((~valid).sum()),
        source_cameras_used=len(used),atlas_size=list(atlas.size),runtime_s=time.monotonic()-started,
        source_mesh_sha256=hashlib.sha256((a.mesh/'reconstruction_mesh.ply').read_bytes()).hexdigest(),
        sha256=hashlib.sha256((a.output/'reconstruction_mesh.glb').read_bytes()).hexdigest(),
        warnings=['View selection can produce texture seams; this is not independent geometry validation.',
            'No holes, roofs or unseen surfaces were invented. Mesh accuracy remains unverified.'])
    (a.output/'report.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mesh',type=Path,required=True)
    p.add_argument('--predictions',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--tile-width',type=int,default=768,choices=[420,768,1024]);run(p.parse_args())
