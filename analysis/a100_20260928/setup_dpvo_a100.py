"""Isolated, reproducible attempt to add official DPVO as a pose baseline.

This does not modify the MapAnything runtime. DPVO is experimental until its
pretrained weights, CUDA extensions, image/calibration parsing and evaluation
are independently checked on a flight withheld from any parameter tuning.
"""

import os
from pathlib import Path
import subprocess


ROOT = Path("/workspace/voxelflight_a100_20260928")
SOURCE = ROOT / "thirdparty/DPVO"
ENV = ROOT / "thirdparty/dpvo-env"


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def main():
    if not (SOURCE / "dpvo.pth").is_file():
        raise FileNotFoundError("Official pretrained DPVO checkpoint missing")
    if not (SOURCE / "thirdparty/eigen-3.4.0/Eigen/Core").is_file():
        raise FileNotFoundError("Official Eigen 3.4.0 headers missing")
    if not (ENV / "bin/python").is_file():
        run(["/usr/bin/python3", "-m", "venv", "--system-site-packages", ENV])
    python = ENV / "bin/python"
    pip = [python, "-m", "pip"]
    run([*pip, "install", "--no-deps", "numpy==1.26.4", "evo", "pypose", "kornia", "yacs"])
    run([*pip, "install", "--no-index", "--no-deps", "--find-links",
         "https://data.pyg.org/whl/torch-2.6.0+cu126.html", "torch_scatter"])
    environment = os.environ.copy()
    environment.update(
        CUDA_HOME="/usr/local/cuda", TORCH_CUDA_ARCH_LIST="8.0", MAX_JOBS="4",
    )
    run([*pip, "install", "--no-build-isolation", "--no-deps", "-e", SOURCE],
        env=environment)
    run([python, "-c", "import torch,torch_scatter,cuda_corr,cuda_ba,lietorch_backends,dpvo;"
         "print(torch.__version__,torch.version.cuda,'DPVO_IMPORT_OK')"])


if __name__ == "__main__":
    main()
