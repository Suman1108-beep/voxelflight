cd /workspace/voxelflight_a100_20260928/thirdparty/PGSR/submodules
grep -q "<cfloat>" simple-knn/simple_knn.cu || sed -i '1i #include <cfloat>' simple-knn/simple_knn.cu
for f in diff-plane-rasterization/cuda_rasterizer/rasterizer_impl.h diff-plane-rasterization/cuda_rasterizer/rasterizer.h; do [ -f $f ] && (grep -q "<cstdint>" $f || sed -i '1i #include <cstdint>' $f); done
export TORCH_CUDA_ARCH_LIST=8.0 MAX_JOBS=32 CUDA_HOME=/usr/local/cuda
for d in simple-knn diff-plane-rasterization; do
  (cd $d && rm -rf build && /workspace/voxelflight_a100_20260928/.venv/bin/python setup.py build_ext --inplace > build.log 2>&1; echo "$d exit $?"; grep -E "error|Error" build.log | head -8)
done
ls simple-knn/simple_knn/ diff-plane-rasterization/diff_plane_rasterization/ | head
PYTHONPATH=/workspace/voxelflight_a100_20260928/thirdparty/PGSR/submodules/simple-knn:/workspace/voxelflight_a100_20260928/thirdparty/PGSR/submodules/diff-plane-rasterization:/workspace/voxelflight_a100_20260928/pydeps-pgsr:/workspace/voxelflight_a100_20260928/thirdparty/PGSR /workspace/voxelflight_a100_20260928/.venv/bin/python -c "import torch, diff_plane_rasterization, simple_knn._C; from scene.gaussian_model import GaussianModel; print('PGSR_IMPORT_OK')" 2>&1 | tail -3
