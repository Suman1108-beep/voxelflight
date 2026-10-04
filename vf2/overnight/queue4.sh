# Coverage experiments on the same cameras (quality720-timed SfM): confidence, ensembles; each scored
export LD_LIBRARY_PATH=/workspace/voxelflight_a100_20260928/runtime-packages/root/usr/lib/x86_64-linux-gnu VF_WEIGHT=1 OMP_NUM_THREADS=8 VF_BLOCKS=900000
cd /workspace/voxelflight_a100_20260928/vf2; SP=runs/quality720-timed/sfm/sparse; TEL=/workspace/voxelflight_a100_20260928/runs/zurich-10min-input/telemetry_v2.csv; CR=runs/quality720-tiled/crops
EV="/workspace/voxelflight_a100_20260928/datasets/zurich-40001-58000 40001 /workspace/voxelflight_a100_20260928/vf2/runs/lidar_ref_2018_egm96.npz /workspace/voxelflight_a100_20260928/runs/zurich-10min-input/camera.json"
run() { name=$1; shift; mkdir -p runs/$name; /workspace/voxelflight_a100_20260928/.venv/bin/python -u dense_multi_crops.py --sparse $SP --telemetry $TEL --output runs/$name --depth-max 35 "$@" > runs/$name/dense.log 2>&1 && /workspace/voxelflight_a100_20260928/.venv/bin/python -u evaluate_run.py runs/$name $EV > runs/$name/eval.txt 2>&1; echo "$name $? $(date +%T)"; }
run q720-da3t-c5 --predictions runs/q720-da3g-tiled/inference --crops $CR --conf-drop 5
run q720-ens2 --predictions runs/q720-da3g-tiled/inference runs/quality720-tiled/inference --crops $CR $CR
run q720-ens3 --predictions runs/q720-da3g-tiled/inference runs/quality720-tiled/inference runs/q720-da3g-hr/inference --crops $CR $CR none
echo QUEUE4_DONE
