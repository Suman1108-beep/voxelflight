"""Parallel keyframe decoding, SuperPoint extraction and LightGlue matching on GPU."""
import os
import sys
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import cv2
import numpy as np
import torch

sys.path.insert(0, "/workspace/voxelflight_a100_20260928/extra-packages")


def _decode_chunk(args):
    video, frames, shm_path, shape, slots, scale, gray = args
    cv2.setNumThreads(1)
    out = np.memmap(shm_path, dtype=np.uint8, mode="r+", shape=shape)
    cap = cv2.VideoCapture(video)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frames[0]))
    want = dict(zip(frames.tolist(), slots.tolist()))
    pos = int(frames[0])
    while pos <= frames[-1]:
        if pos in want:
            ok, img = cap.read()
            if not ok:
                break
            if gray:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            if scale != 1.0:
                img = cv2.resize(img, (shape[2], shape[1]), interpolation=cv2.INTER_AREA)
            out[want[pos]] = img
        elif not cap.grab():
            break
        pos += 1
    out.flush()
    return pos - 1


def decode_keyframes(video, frames, scale=1.0, workers=12, tmp_dir="/tmp", gray=False):
    """Decode selected frame indices into uint8 (N,H,W[,3]) using parallel seeks (BGR or gray)."""
    cap = cv2.VideoCapture(video)
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    w, h = int(round(W * scale)), int(round(H * scale))
    shape = (len(frames), h, w) if gray else (len(frames), h, w, 3)
    path = os.path.join(tmp_dir, f"vf2_frames_{os.getpid()}.u8")
    np.memmap(path, dtype=np.uint8, mode="w+", shape=shape).flush()
    try:
        slots = np.arange(len(frames))
        parts = [p for p in np.array_split(slots, workers) if len(p)]
        jobs = [(video, np.asarray(frames)[p], path, shape, p, scale, gray) for p in parts]
        with ProcessPoolExecutor(len(jobs)) as ex:
            list(ex.map(_decode_chunk, jobs))
        arr = np.array(np.memmap(path, dtype=np.uint8, mode="r", shape=shape))
    finally:
        os.remove(path)
    return arr, (W, H)


@torch.inference_mode()
def extract_superpoint(images_bgr, max_kp=2048, long_side=1280, device="cuda"):
    from lightglue import SuperPoint
    ext = SuperPoint(max_num_keypoints=max_kp, detection_threshold=0.0025).eval().to(device)
    feats = []
    H, W = images_bgr.shape[1:3]
    s = long_side / max(H, W)
    for img in images_bgr:
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if s != 1:
            g = cv2.resize(g, (int(round(W * s)), int(round(H * s))), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(g).to(device).float()[None, None] / 255.0
        f = ext.extract(t, resize=None)
        kp = f["keypoints"][0] / s  # back to full-resolution pixels
        feats.append({"keypoints": kp.half(), "descriptors": f["descriptors"][0].half(),
                      "scores": f["keypoint_scores"][0]})
    return feats, (W, H)


@torch.inference_mode()
def match_pairs(feats, pairs, image_size, device="cuda", min_matches=15):
    from lightglue import LightGlue
    m = LightGlue(features="superpoint", depth_confidence=0.95, width_confidence=0.99,
                  filter_threshold=0.15).eval().to(device)
    size = torch.tensor([image_size], device=device, dtype=torch.float32)
    out = {}
    for i, j in pairs:
        a, b = feats[i], feats[j]
        r = m({"image0": {"keypoints": a["keypoints"][None].float(), "descriptors": a["descriptors"][None].float(), "image_size": size},
               "image1": {"keypoints": b["keypoints"][None].float(), "descriptors": b["descriptors"][None].float(), "image_size": size}})
        mm = r["matches"][0]
        if len(mm) >= min_matches:
            out[(i, j)] = mm.cpu().numpy().astype(np.int32)
    return out


@torch.inference_mode()
def match_pairs_batched(feats, pairs, image_size, device="cuda", batch=48, min_matches=15, max_kp=2048):
    """Batched fp16 LightGlue: pad every image to max_kp keypoints, drop matches touching padding."""
    from lightglue import LightGlue
    m = LightGlue(features="superpoint", depth_confidence=-1, width_confidence=-1,
                  filter_threshold=0.15, flash=True).eval().to(device)
    n = len(feats)
    KP = torch.zeros(n, max_kp, 2, device=device)
    DS = torch.zeros(n, max_kp, 256, device=device, dtype=torch.float16)
    cnt = torch.zeros(n, dtype=torch.long, device=device)
    W, H = image_size
    for i, f in enumerate(feats):
        k = len(f["keypoints"])
        KP[i, :k] = f["keypoints"].float(); DS[i, :k] = f["descriptors"].half(); cnt[i] = k
        if k < max_kp:  # padding: points far outside the image with zero descriptors
            KP[i, k:] = torch.tensor([-10.0 * W, -10.0 * H], device=device)
    size = torch.tensor([image_size], device=device, dtype=torch.float32)
    out = {}
    pairs = list(pairs)
    for s in range(0, len(pairs), batch):
        pb = pairs[s:s + batch]
        a = torch.tensor([p[0] for p in pb], device=device); b = torch.tensor([p[1] for p in pb], device=device)
        with torch.autocast("cuda", dtype=torch.float16):
            r = m({"image0": {"keypoints": KP[a], "descriptors": DS[a].float(), "image_size": size.expand(len(pb), 2)},
                   "image1": {"keypoints": KP[b], "descriptors": DS[b].float(), "image_size": size.expand(len(pb), 2)}})
        m0 = r["matches0"]  # B, N  (-1 = unmatched)
        for k, (i, j) in enumerate(pb):
            idx0 = torch.nonzero(m0[k] >= 0).squeeze(1)
            idx1 = m0[k][idx0]
            ok = (idx0 < cnt[i]) & (idx1 < cnt[j])
            mm = torch.stack([idx0[ok], idx1[ok]], 1)
            if len(mm) >= min_matches:
                out[(i, j)] = mm.cpu().numpy().astype(np.int32)
    return out


def undistort_keypoints(feats, K, dist):
    res = []
    for f in feats:
        p = f["keypoints"].float().cpu().numpy().reshape(-1, 1, 2)
        if dist is not None and np.any(np.abs(dist) > 0):
            p = cv2.undistortPoints(p, K, np.asarray(dist, float), P=K)
        res.append(p.reshape(-1, 2).astype(np.float64))
    return res


def verify_pairs(kps_ud, matches, K, threshold_px=1.5, workers=48):
    """Essential-matrix RANSAC on undistorted points; returns inlier matches per pair."""
    f = 0.5 * (K[0, 0] + K[1, 1])

    def one(item):
        (i, j), mm = item
        p0, p1 = kps_ud[i][mm[:, 0]], kps_ud[j][mm[:, 1]]
        if len(mm) < 15:
            return (i, j), None
        E, mask = cv2.findEssentialMat(p0, p1, K, method=cv2.RANSAC, prob=0.999, threshold=threshold_px)
        if mask is None:
            return (i, j), None
        keep = mask.ravel().astype(bool)
        return (i, j), (mm[keep] if keep.sum() >= 15 else None)

    with ThreadPoolExecutor(workers) as ex:
        res = dict(ex.map(one, matches.items()))
    return {k: v for k, v in res.items() if v is not None}


def build_tracks(num_kps, verified, min_len=3, max_len=60):
    """Connected components of the match graph; drop components seeing an image twice."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    offs = np.concatenate([[0], np.cumsum(num_kps)])
    rows, cols = [], []
    for (i, j), mm in verified.items():
        rows.append(offs[i] + mm[:, 0]); cols.append(offs[j] + mm[:, 1])
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    N = offs[-1]
    g = coo_matrix((np.ones(len(rows), np.int8), (rows, cols)), shape=(N, N))
    _, lab = connected_components(g, directed=False)
    img_of = np.repeat(np.arange(len(num_kps)), num_kps)
    kp_of = np.arange(N) - offs[img_of]
    sizes = np.bincount(lab)
    good_lab = np.where((sizes >= min_len) & (sizes <= max_len))[0]
    sel = np.isin(lab, good_lab)
    nodes = np.where(sel)[0]
    order = np.argsort(lab[nodes], kind="stable")
    nodes = nodes[order]
    l = lab[nodes]
    splits = np.where(np.diff(l))[0] + 1
    tracks = []
    for grp in np.split(nodes, splits):
        im = img_of[grp]
        if len(np.unique(im)) != len(im):
            continue
        tracks.append(np.column_stack([im, kp_of[grp]]))
    return tracks
