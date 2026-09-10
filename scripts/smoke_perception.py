#!/usr/bin/env python3
"""Run semantic segmentation and learned matching on real aerial frames."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoModelForKeypointMatching,
    SegformerForSemanticSegmentation,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_a", type=Path)
    parser.add_argument("image_b", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = torch.device("cuda")
    image_a = Image.open(args.image_a).convert("RGB")
    image_b = Image.open(args.image_b).convert("RGB")
    report: dict[str, object] = {}

    matcher_name = "ETH-CVG/lightglue_superpoint"
    processor = AutoImageProcessor.from_pretrained(matcher_name)
    matcher = AutoModelForKeypointMatching.from_pretrained(matcher_name).to(device).eval()
    inputs = processor([image_a, image_b], return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode():
        matches = matcher(**inputs)
    torch.cuda.synchronize()
    processed = processor.post_process_keypoint_matching(
        matches,
        [[(image_a.height, image_a.width), (image_b.height, image_b.width)]],
        threshold=0.2,
    )[0]
    report["lightglue"] = {
        "model": matcher_name,
        "matches": len(processed["matching_scores"]),
        "median_score": float(processed["matching_scores"].median().cpu())
        if len(processed["matching_scores"])
        else 0.0,
        "inference_seconds": round(time.perf_counter() - started, 4),
    }
    del matcher, inputs, matches
    torch.cuda.empty_cache()

    segmenter_name = "nvidia/segformer-b5-finetuned-ade-640-640"
    seg_processor = AutoImageProcessor.from_pretrained(segmenter_name)
    segmenter = SegformerForSemanticSegmentation.from_pretrained(
        segmenter_name
    ).to(device).eval()
    seg_inputs = seg_processor(images=image_a, return_tensors="pt").to(device)
    started = time.perf_counter()
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        logits = segmenter(**seg_inputs).logits
    resized = torch.nn.functional.interpolate(
        logits,
        size=(image_a.height, image_a.width),
        mode="bilinear",
        align_corners=False,
    )
    labels = resized.argmax(1)[0].cpu().numpy().astype(np.uint8)
    torch.cuda.synchronize()
    counts = Counter(labels.reshape(-1).tolist())
    total = labels.size
    top = [
        {
            "label": segmenter.config.id2label[index],
            "fraction": round(count / total, 5),
        }
        for index, count in counts.most_common(12)
    ]
    report["segformer"] = {
        "model": segmenter_name,
        "inference_seconds": round(time.perf_counter() - started, 4),
        "top_labels": top,
    }

    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        palette = np.random.default_rng(5).integers(0, 255, (256, 3), dtype=np.uint8)
        overlay = np.asarray(image_a, dtype=np.float32) * 0.55 + palette[labels] * 0.45
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(
            args.output / "semantic_overlay.jpg",
            quality=92,
        )
        (args.output / "perception_report.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

