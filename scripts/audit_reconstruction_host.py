"""Read-only host audit for planning isolated VoxelFlight GPU experiments.

Does not read credentials, install packages, stop processes or change GPU state.
The optional CUDA smoke test performs one small matrix multiplication.
"""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys


def command(args: list[str]) -> dict:
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=15)
        return {"exit_code": result.returncode, "stdout": result.stdout[-12000:],
                "stderr": result.stderr[-1500:]}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {"error": str(error)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/workspace"))
    parser.add_argument("--cuda-smoke", action="store_true")
    args = parser.parse_args()
    root = args.root.resolve()
    report = {
        "schema": "voxelflight.host-audit.v1",
        "python": sys.executable,
        "python_version": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "cpu_affinity": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "gpu": command(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu,driver_version", "--format=csv"]),
        "gpu_processes": command(["nvidia-smi", "--query-compute-apps=pid,process_name,used_memory", "--format=csv"]),
        "disk": command(["df", "-h", str(root)]),
        "executables": {name: shutil.which(name) for name in
                        ["python3", "git", "ffmpeg", "colmap", "nvcc"]},
        "reconstruction_project_entries": sorted(p.name for p in root.iterdir()
            if any(token in p.name.lower() for token in ["voxel", "sih", "map-anything", "pi3"])),
        "cgroup_limits": {name: (Path("/sys/fs/cgroup") / name).read_text().strip()
                          for name in ["cpu.max", "memory.max", "memory.current", "cpuset.cpus.effective"]
                          if (Path("/sys/fs/cgroup") / name).is_file()},
        "packages": {},
    }
    memory = Path("/proc/meminfo")
    if memory.is_file():
        report["memory"] = [line for line in memory.read_text().splitlines()
                            if line.startswith(("MemTotal:", "MemAvailable:"))]
    for package in ["torch", "torchvision", "numpy", "scipy", "open3d", "pycolmap",
                    "mapanything", "gsplat", "transformers", "huggingface_hub", "opencv-python"]:
        try:
            report["packages"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["packages"][package] = None
    instruction_file = root / "AGENTS.md"
    if instruction_file.is_file():
        report["root_instructions"] = instruction_file.read_text()
    if args.cuda_smoke:
        try:
            import torch
            report["torch_cuda_build"] = torch.version.cuda
            report["cuda_available"] = torch.cuda.is_available()
            if report["cuda_available"]:
                with torch.inference_mode():
                    x = torch.ones((512, 512), device="cuda", dtype=torch.float32)
                    y = x @ x
                    torch.cuda.synchronize()
                    report["cuda_smoke"] = {
                        "device": torch.cuda.get_device_name(0),
                        "correct": bool(torch.all(y == 512).item()),
                        "allocated_bytes": torch.cuda.memory_allocated(),
                    }
        except Exception as error:
            report["cuda_smoke_error"] = f"{type(error).__name__}: {error}"
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
