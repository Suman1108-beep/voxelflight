"""Extract missing shared libraries privately for Open3D, without OS install."""

from pathlib import Path
import subprocess

root = Path('/workspace/voxelflight_a100_20260928')
packages = root / 'runtime-packages'
packages.mkdir(parents=True, exist_ok=True)
subprocess.run(['apt-get', 'update', '-qq'], check=True, timeout=180)
subprocess.run(['apt-get', 'download', 'libx11-6', 'libgl1', 'libglx0',
                'libglvnd0', 'libxext6', 'libxcb1'],
               cwd=packages, check=True, timeout=180)
for archive in packages.glob('*.deb'):
    subprocess.run(['dpkg-deb', '-x', str(archive), str(packages / 'root')], check=True)
print('Extracted libraries under', packages / 'root', flush=True)
