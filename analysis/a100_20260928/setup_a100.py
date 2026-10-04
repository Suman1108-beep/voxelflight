"""Set up an isolated A100 inference environment; never modify shared Python."""

import os
from pathlib import Path
import subprocess
import sys

ROOT = Path('/workspace/voxelflight_a100_20260928')
ENV = ROOT / '.venv'
os.environ['HF_HOME'] = str(ROOT / 'cache/huggingface')
os.environ['TORCH_HOME'] = str(ROOT / 'cache/torch')


def run(*args: str) -> None:
    print('+', ' '.join(args), flush=True)
    subprocess.run(args, check=True)


if not (ENV / 'bin/python').exists():
    run(sys.executable, '-m', 'venv', '--system-site-packages', str(ENV))

pip = str(ENV / 'bin/python')
run(pip, '-m', 'pip', 'install', '--disable-pip-version-check', '-e', str(ROOT / 'map-anything'))
run(pip, '-m', 'pip', 'install', '--disable-pip-version-check',
    'numpy==1.26.4', 'scipy==1.15.3', 'rasterio==1.4.3',
    'open3d==0.19.0', 'pycolmap==3.10.0', 'laspy>=2.6',
    'pyproj>=3.7', 'huggingface_hub==0.36.2', 'plotly==6.9.0',
    'imageio-ffmpeg>=0.6', 'pillow>=11', 'matplotlib>=3.10')

dino = ROOT / 'dinov2'
if not (dino / 'hubconf.py').exists():
    run('git', 'clone', '--depth', '1', 'https://github.com/facebookresearch/dinov2.git', str(dino))

run(pip, '-c', 'import numpy,torch,open3d,pycolmap,mapanything,huggingface_hub;'
    'assert torch.from_numpy(numpy.zeros(1, dtype=numpy.float32)).numel()==1;'
    'print("CUDA",torch.cuda.is_available(),"torch",torch.__version__,'
    '"numpy",numpy.__version__,"open3d",open3d.__version__,"pycolmap",pycolmap.__version__)')
