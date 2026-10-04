#!/bin/zsh
set -e
task_root="${0:A:h}"
cd "$task_root"
if [[ ! -x "$task_root/.venv-mac/bin/python" ]]; then
  echo "The Mac environment is missing. See MAC_ENGINE.md for setup."
  exit 1
fi
echo "SIH3D Mac GPU engine — http://127.0.0.1:8133/engine.html"
echo "Keep this window open. Press Control-C to stop. Files stay on this Mac."
exec "$task_root/.venv-mac/bin/python" -m uvicorn mac_server:app --host 127.0.0.1 --port 8133
