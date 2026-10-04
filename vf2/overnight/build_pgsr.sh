set -x
cd /workspace/voxelflight_a100_20260928/thirdparty/PGSR
grep -q "def quaternion_to_matrix" scene/gaussian_model.py || python3 - <<'PY'
p="scene/gaussian_model.py"; s=open(p).read()
s=s.replace("from pytorch3d.transforms import quaternion_to_matrix", '''def quaternion_to_matrix(q):
    r, i, j, k = torch.unbind(q, -1); two_s = 2.0 / (q * q).sum(-1)
    o = torch.stack((1 - two_s * (j * j + k * k), two_s * (i * j - k * r), two_s * (i * k + j * r),
                     two_s * (i * j + k * r), 1 - two_s * (i * i + k * k), two_s * (j * k - i * r),
                     two_s * (i * k - j * r), two_s * (j * k + i * r), 1 - two_s * (i * i + j * j)), -1)
    return o.reshape(q.shape[:-1] + (3, 3))''')
open(p,"w").write(s); print("patched pytorch3d import")
PY
grep -rn "pytorch3d" --include=*.py . | head
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=32 CUDA_HOME=/usr/local/cuda
/workspace/voxelflight_a100_20260928/.venv/bin/pip install --no-deps --no-build-isolation --target /workspace/voxelflight_a100_20260928/pydeps-pgsr submodules/diff-plane-rasterization submodules/simple-knn 2>&1 | tail -5
/workspace/voxelflight_a100_20260928/.venv/bin/pip install -q --no-deps --target /workspace/voxelflight_a100_20260928/pydeps-pgsr lpips 2>&1 | tail -2
PYTHONPATH=/workspace/voxelflight_a100_20260928/pydeps-pgsr:/workspace/voxelflight_a100_20260928/thirdparty/PGSR /workspace/voxelflight_a100_20260928/.venv/bin/python -c "import torch, diff_plane_rasterization, simple_knn._C; from scene.gaussian_model import GaussianModel; print('PGSR_IMPORT_OK')"
