"""CPU geometry operations for the bounded-memory Apple-GPU reconstruction path.

No reference poses, benchmark answers or generated replacement geometry are used.
"""
from __future__ import annotations

import numpy as np


def transform_points(points, scale, rotation, translation):
    return np.asarray(points) @ np.asarray(rotation).T * scale + translation


def camera_similarity(source_poses, target_poses):
    """Fit a similarity using camera orientations and centres, without references.

    Orientation correspondences constrain roll even along a straight camera
    path. They must be visual estimates, never evaluation ground truth.
    """
    source=np.asarray(source_poses,float);target=np.asarray(target_poses,float)
    if source.shape!=target.shape or source.ndim!=3 or source.shape[1:]!=(4,4) or len(source)<2:
        raise ValueError('Need matching arrays of at least two camera poses.')
    if not np.isfinite(source).all() or not np.isfinite(target).all():raise ValueError('Non-finite camera pose.')
    for poses in (source,target):
        rotations=poses[:,:3,:3]
        if not np.allclose(np.swapaxes(rotations,1,2)@rotations,np.eye(3),atol=1e-3) or not np.allclose(np.linalg.det(rotations),1,atol=1e-3):
            raise ValueError('Camera orientations must be proper rotations.')
    relative=target[:,:3,:3]@np.swapaxes(source[:,:3,:3],1,2)
    u,_,vt=np.linalg.svd(relative.mean(0));sign=np.ones(3);sign[-1]=np.linalg.det(u@vt)
    rotation=(u*sign)@vt
    x,y=source[:,:3,3],target[:,:3,3];xc=x-x.mean(0);yc=y-y.mean(0)
    denom=np.sum(xc*xc)
    if denom<1e-9:raise ValueError('Stationary cameras cannot constrain scale.')
    scale=float(np.sum((xc@rotation.T)*yc)/denom)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Invalid camera scale.')
    translation=y.mean(0)-scale*rotation@x.mean(0)
    residual=np.linalg.norm(transform_points(x,scale,rotation,translation)-y,axis=1)
    angles=np.rad2deg(np.arccos(np.clip((np.trace(relative@rotation.T,axis1=1,axis2=2)-1)/2,-1,1)))
    return scale,rotation,translation,dict(camera_fit_rmse=float(np.sqrt(np.mean(residual**2))),
        mean_orientation_residual_deg=float(angles.mean()),correspondences=len(source))


def similarity(source, target, robust=True):
    """Trimmed Umeyama fit. Refuse collinear/constant data, including straight GPS paths."""
    source, target = np.asarray(source, float), np.asarray(target, float)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError("Correspondences must be matching N×3 arrays.")
    keep = np.isfinite(source).all(1) & np.isfinite(target).all(1)
    if keep.sum() < 6:
        raise ValueError("At least six finite correspondences are required.")
    valid = keep.copy()
    for _ in range(5 if robust else 1):
        x, y = source[keep], target[keep]
        xm, ym = x.mean(0), y.mean(0)
        xc, yc = x-xm, y-ym
        if min(np.linalg.svd(xc, compute_uv=False)[1], np.linalg.svd(yc, compute_uv=False)[1]) < 1e-5:
            raise ValueError("Collinear or stationary geometry cannot constrain this alignment.")
        u, d, vt = np.linalg.svd(yc.T @ xc / len(x))
        sign = np.ones(3)
        sign[-1] = np.linalg.det(u @ vt)
        rotation = (u * sign) @ vt
        scale = float(d @ sign / np.mean(np.sum(xc*xc, axis=1)))
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("Invalid reconstruction scale.")
        translation = ym-scale*rotation@xm
        errors = np.linalg.norm(transform_points(source, scale, rotation, translation)-target, axis=1)
        if robust:
            threshold = max(float(np.quantile(errors[valid], .8)), 1e-6)
            keep = valid & (errors <= threshold)
            if keep.sum() < 6:
                break
    return scale, rotation, translation, {"fit_rmse": float(np.sqrt(np.mean(errors[valid]**2))), "inliers": int(keep.sum()), "correspondences": int(valid.sum())}


def depth_mesh(points, colors, depth, mask, stride=2, return_uv=False):
    """Triangulate only supported neighboring depth pixels; never close missing surfaces."""
    p = np.asarray(points)[::stride, ::stride].astype(np.float32)
    c = np.asarray(colors)[::stride, ::stride].astype(np.uint8)
    d = np.asarray(depth)[::stride, ::stride].squeeze()
    m = np.asarray(mask)[::stride, ::stride].squeeze() & np.isfinite(p).all(-1) & np.isfinite(d) & (d>0)
    h,w = m.shape
    ids = np.arange(h*w).reshape(h,w)
    a,b,cid,e = ids[:-1,:-1],ids[:-1,1:],ids[1:,:-1],ids[1:,1:]
    faces = np.concatenate([np.stack([a,cid,b],-1).reshape(-1,3),np.stack([b,cid,e],-1).reshape(-1,3)])
    ds = d.reshape(-1)[faces]
    ok = m.reshape(-1)[faces].all(1) & ((ds.max(1)-ds.min(1)) <= .08*np.maximum(ds.mean(1),.01))
    faces=faces[ok]
    if not len(faces):
        empty=(np.empty((0,3),np.float32),np.empty((0,3),np.uint8),np.empty((0,3),np.int32))
        return (*empty,np.empty((0,2),np.float32)) if return_uv else empty
    used, inverse = np.unique(faces,return_inverse=True)
    result=(p.reshape(-1,3)[used],c.reshape(-1,3)[used],inverse.reshape(-1,3).astype(np.int32))
    if not return_uv:return result
    source_h,source_w=np.asarray(points).shape[:2]
    uv=np.stack([((used%w)*stride+.5)/source_w,1-((used//w)*stride+.5)/source_h],1).astype(np.float32)
    return (*result,uv)


def voxel_fuse(points, colors, voxel=.035):
    """Average actual observations in occupied voxels; no extrapolation."""
    points,colors=np.asarray(points),np.asarray(colors)
    valid=np.isfinite(points).all(1)
    points,colors=points[valid],colors[valid]
    if not len(points):
        raise ValueError("No finite reconstructed points survived filtering.")
    keys=np.floor(points/voxel).astype(np.int64)
    _,inverse=np.unique(keys,axis=0,return_inverse=True)
    count=np.bincount(inverse)
    p=np.stack([np.bincount(inverse,weights=points[:,i])/count for i in range(3)],1)
    c=np.stack([np.bincount(inverse,weights=colors[:,i])/count for i in range(3)],1)
    return p.astype(np.float32),np.clip(c,0,255).astype(np.uint8)
