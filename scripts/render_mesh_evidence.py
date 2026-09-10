"""Deterministic CPU triangle rasterizer for real geometry QA (no generated imagery).

Mesh triangles are z-buffered with perspective-correct interpolated source
vertex colors. Shared reference bounds keep before/after overview cameras fixed.
"""
import argparse
from pathlib import Path
import sys
import time
import numpy as np
import trimesh
from PIL import Image, ImageDraw, ImageFont


def load(path):
    scene=trimesh.load(path,process=False)
    if isinstance(scene,trimesh.Trimesh):scene=trimesh.Scene(scene)
    vertices=[];faces=[];colors=[];offset=0
    for node in scene.graph.nodes_geometry:
        transform,name=scene.graph[node];mesh=scene.geometry[name]
        vertices.append(trimesh.transform_points(mesh.vertices,transform))
        faces.append(np.asarray(mesh.faces)+offset);offset+=len(mesh.vertices)
        visual=mesh.visual.to_color() if mesh.visual.kind=='texture' else mesh.visual
        colors.append(np.asarray(visual.vertex_colors)[:,:3])
    return np.concatenate(vertices),np.concatenate(faces),np.concatenate(colors)


def render(vertices,faces,colors,eye,target,width=1000,height=700):
    forward=target-eye;forward/=np.linalg.norm(forward)
    right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    camera=(vertices-eye)@np.stack([right,down,forward],axis=1)
    focal=.94*width
    xy=camera[:,:2]/np.maximum(camera[:,2:],.01)*focal+[width/2,height/2]
    image=np.full((height,width,3),[13,20,31],np.uint8)
    zbuffer=np.full((height,width),np.inf)
    mask=(camera[faces,2]>.05).all(1)
    for tri in faces[mask]:
        q=xy[tri];lo=np.maximum(np.floor(q.min(0)).astype(int),0);hi=np.minimum(np.ceil(q.max(0)).astype(int),[width-1,height-1])
        if np.any(lo>hi):continue
        a,b,c=q
        denominator=(b[1]-c[1])*(a[0]-c[0])+(c[0]-b[0])*(a[1]-c[1])
        if abs(denominator)<1e-8:continue
        yy,xx=np.mgrid[lo[1]:hi[1]+1,lo[0]:hi[0]+1];xx=xx+.5;yy=yy+.5
        w0=((b[1]-c[1])*(xx-c[0])+(c[0]-b[0])*(yy-c[1]))/denominator
        w1=((c[1]-a[1])*(xx-c[0])+(a[0]-c[0])*(yy-c[1]))/denominator
        w2=1-w0-w1
        inside=(w0>=-1e-5)&(w1>=-1e-5)&(w2>=-1e-5)
        if not inside.any():continue
        bary=np.stack([w0,w1,w2],-1)/camera[tri,2]
        invz=bary.sum(-1);depth=1/np.maximum(invz,1e-12)
        region=zbuffer[lo[1]:hi[1]+1,lo[0]:hi[0]+1]
        update=inside&(depth<region)
        if not update.any():continue
        rgb=(bary@colors[tri])*depth[...,None]
        image[lo[1]:hi[1]+1,lo[0]:hi[0]+1][update]=np.clip(rgb[update],0,255).astype(np.uint8)
        region[update]=depth[update]
    return image,float(np.isfinite(zbuffer).mean())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('mesh',type=Path);p.add_argument('output',type=Path)
    p.add_argument('--reference',type=Path)
    p.add_argument('--title',default='Actual reconstructed mesh')
    p.add_argument('--view',choices=['overview','facade'],default='overview')
    a=p.parse_args();v,f,c=load(a.mesh)
    ref=load(a.reference)[0] if a.reference else v
    if not np.isfinite(ref).all():raise ValueError('Reference mesh contains non-finite vertices.')
    low,high=ref.min(0),ref.max(0);target=(low+high)/2
    direction=np.array([.5,-1.,.55]) if a.view=='overview' else np.array([0.,-1.,.08])
    direction/=np.linalg.norm(direction)
    forward=-direction;right=np.cross(forward,[0.,0.,1.]);right/=np.linalg.norm(right)
    down=np.cross(forward,right)
    relative=(ref-target)@np.stack([right,down,forward],axis=1)
    # Fit every reference vertex inside the frustum, with a 6% image margin.
    # Do not hide poorly reconstructed regions using percentile cropping.
    distance=max(np.max(np.abs(relative[:,0])*.94*1000/440-relative[:,2]),
                 np.max(np.abs(relative[:,1])*.94*1000/308-relative[:,2]),
                 .1-np.min(relative[:,2]))
    eye=target+direction*distance
    rgb,coverage=render(v,f,c,eye,target)
    canvas=Image.new('RGB',(1000,800),(13,20,31));canvas.paste(Image.fromarray(rgb),(0,65))
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',24);small=ImageFont.truetype('/System/Library/Fonts/Supplemental/Arial.ttf',16)
    except OSError:font=small=ImageFont.load_default()
    draw.text((24,20),a.title,font=font,fill=(182,243,216))
    camera_label='shared reference camera' if a.reference else 'full mesh bounds'
    draw.text((24,766),f'Real mesh rasterization • {camera_label} • no generated building imagery',font=small,fill=(173,191,209))
    a.output.parent.mkdir(parents=True,exist_ok=True);canvas.save(a.output)
    print(a.output,'triangles',len(f),'viewport_occupancy',coverage,flush=True)


if __name__=='__main__':main()
