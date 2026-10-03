"""Benchmark batched vs per-pair LightGlue on 200 keyframes."""
import os, sys, time, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, torch
import features as F
INP = "/workspace/voxelflight_a100_20260928/runs/zurich-10min-input"
frames = np.arange(4, 18000, 15)[:200]
imgs, size = F.decode_keyframes(INP + "/flight.mp4", frames, workers=8, gray=True)
feats, size = F.extract_superpoint(imgs)
print("kp counts min/median", min(len(f["keypoints"]) for f in feats), np.median([len(f["keypoints"]) for f in feats]))
pairs = [(i, i + o) for i in range(200) for o in [1, 2, 3, 4, 6, 8, 10, 13, 16, 20] if i + o < 200]
torch.cuda.synchronize(); t = time.time(); a = F.match_pairs(feats, pairs[:300], size); torch.cuda.synchronize(); ta = (time.time() - t) / 300
for B in (32, 64):
    torch.cuda.synchronize(); t = time.time(); b = F.match_pairs_batched(feats, pairs, size, batch=B); torch.cuda.synchronize(); tb = (time.time() - t) / len(pairs)
    print(f"batch {B}: {tb*1000:.2f} ms/pair")
print(f"single: {ta*1000:.2f} ms/pair")
common = [k for k in a if k in b]
print("mean matches single", np.mean([len(a[k]) for k in common]), "batched", np.mean([len(b[k]) for k in common]))
