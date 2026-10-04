# After queue2: Austria 4K (clean folder, solo GPU) + aerial evaluation of both 4K clips
until [ -f /workspace/voxelflight_a100_20260928/vf2/logs/queue2.exit ]; do sleep 30; done
date; cd /workspace/voxelflight_a100_20260928/vf2 && export LD_LIBRARY_PATH=/workspace/voxelflight_a100_20260928/runtime-packages/root/usr/lib/x86_64-linux-gnu VF_WEIGHT=1
[ -d runs/real4k-bambi-fast ] && mv runs/real4k-bambi-fast runs/real4k-bambi-fast-stale-$(date +%s)
/workspace/voxelflight_a100_20260928/.venv/bin/python -u run_pipeline.py --video /workspace/voxelflight_a100_20260928/datasets/real4k/bambi102/DJI_20230426181821_0001_V.MP4 --telemetry /workspace/voxelflight_a100_20260928/datasets/real4k/bambi102/DJI_20230426181821_0001_V.SRT --calibration /workspace/voxelflight_a100_20260928/datasets/real4k/bambi102/camera.json --output /workspace/voxelflight_a100_20260928/vf2/runs/real4k-bambi-fast --voxel 0.15 --depth-max 150 --exchange-voxel 0.3 > /workspace/voxelflight_a100_20260928/vf2/logs/bambi4k.log 2>&1; echo BAMBI $?
/workspace/voxelflight_a100_20260928/.venv/bin/python -u evaluate_aerial.py runs/real4k-bambi-fast runs/bev_bambi102_raw.npz > runs/real4k-bambi-fast/eval.txt 2>&1; echo BAMBI_EVAL $?
/workspace/voxelflight_a100_20260928/.venv/bin/python -u evaluate_aerial.py runs/real4k-colorado-fast runs/usgs_colorado_raw.npz > runs/real4k-colorado-fast/eval.txt 2>&1; echo COLORADO_EVAL $?
date
