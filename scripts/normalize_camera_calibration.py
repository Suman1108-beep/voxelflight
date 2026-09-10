#!/usr/bin/env python3
"""Normalize common camera calibration JSON/YAML into MASt3R-SLAM YAML."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_calibration", type=Path)
    parser.add_argument("output_yaml", type=Path)
    args = parser.parse_args()
    text = args.input_calibration.read_text()
    payload = (
        json.loads(text)
        if args.input_calibration.suffix.lower() == ".json"
        else yaml.safe_load(text)
    )
    if not isinstance(payload, dict):
        raise ValueError("Calibration must be a JSON/YAML object")
    if "calibration" in payload:
        calibration = list(payload["calibration"])
    elif "intrinsic_matrix" in payload:
        matrix = payload["intrinsic_matrix"]
        calibration = [matrix[0][0], matrix[1][1], matrix[0][2], matrix[1][2]]
        calibration.extend(payload.get("distortion_coefficients", []))
    elif "intrinsics" in payload:
        calibration = list(payload["intrinsics"])
        calibration.extend(payload.get("distortion_coefficients", []))
    else:
        names = ("fx", "fy", "cx", "cy")
        if not all(name in payload for name in names):
            raise KeyError("Calibration needs calibration, intrinsic_matrix, intrinsics, or fx/fy/cx/cy")
        calibration = [payload[name] for name in names]
        calibration.extend(payload.get("distortion_coefficients", []))
    result = {
        "width": int(payload["width"]),
        "height": int(payload["height"]),
        "calibration": [float(value) for value in calibration],
    }
    args.output_yaml.parent.mkdir(parents=True, exist_ok=True)
    args.output_yaml.write_text(yaml.safe_dump(result, sort_keys=False))
    print(yaml.safe_dump(result, sort_keys=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
