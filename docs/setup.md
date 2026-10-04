# Setup and run

## 1. Public review, no installation

Open [VoxelFlight](https://voxelflight-3d.web.app/). The entrance offers Google sign-in and **Explore as a guest**. The guest workspace includes the original camera video, the saved 3D surface and its evidence. No account is needed for public review.

New reconstruction is a separate feature. It requires sign-in and an available processing worker. A saved demo can remain available while that worker is offline. Live uploads are offline during judging.

## 1b. Run the reconstruction pipeline (v2, GPU)

The pipeline that produced every current result is in [`vf2/`](../vf2/README.md): one command per flight, for example
`python vf2/run_pipeline.py --video flight.mp4 --telemetry flight.SRT --output runs/my-flight`. It needs an NVIDIA GPU
(measured on an A100-40GB); package versions are in [`vf2/requirements-a100.txt`](../vf2/requirements-a100.txt), and
[`vf2/cloud/modal_app.py`](../vf2/cloud/README.md) builds the same environment as a container image.

## 2. Run the website from this repository

Use Node.js **22.15 or newer** (Node 24 recommended) and pnpm 10. The Node geometry tests use `registerHooks`.

```sh
git clone https://github.com/Suman1108-beep/voxelflight.git
cd voxelflight/viewer
pnpm install --frozen-lockfile
node scripts/fetch-demo-assets.mjs
pnpm test
pnpm run build:firebase
pnpm dlx firebase-tools emulators:start --only hosting --project voxelflight-3d
```

Open `http://127.0.0.1:8140/`. The guest demo works locally. Production Google sign-in requires a properly configured Firebase project and authorized domain; running the Hosting emulator alone does not emulate Authentication or provide a production session.

Demo media and browser modules are restored from the public site and checked against `viewer/demo-assets.lock.json`. This download is approximately 122 MB. No model weights or private uploads are fetched. A changed or unavailable file fails the checksum check instead of silently replacing the release evidence. The geometry payload is not needed on the login page.

## 3. Reconstruction environment

The tested reconstruction machine uses Apple Silicon and Python 3.12. Its full package snapshot is `requirements-mac.lock.txt`; `requirements-refusion.txt` adds Open3D and the newer PyCOLMAP used for refusion. This snapshot is platform-specific, not a universal Linux/CUDA lockfile.

```sh
cd voxelflight
python3.12 -m venv .venv-mac
.venv-mac/bin/pip install -r requirements.txt
.venv-mac/bin/python setup_mac.py
.venv-mac/bin/python mac_reconstruct.py --help
.venv-mac/bin/python -m pytest tests/test_mac_engine.py tests/test_public_engine.py tests/test_gps_georeference.py tests/test_refusion.py
```

The one-time setup needs internet access and disk space for pretrained weights. `requirements-mac.lock.txt` pins the MapAnything source revision and `setup_mac.py` pins the model checkpoint revision. The DINOv2 architecture cache currently uses the upstream `main` code; fully pinning that remaining dependency is a reproducibility improvement, not a completed guarantee.

Use the inference script's `--help` to choose an MP4/MOV or ordered image input, output directory, frame budget and calibration. Do not use interpolated images to imply new camera measurements. A single photograph does not satisfy the single-pass video task.

## 4. Start the processing service

```sh
.venv-mac/bin/python -m uvicorn public_server:app --host 127.0.0.1 --port 8141
```

The public-mode API verifies Firebase bearer tokens. The default project is `voxelflight-3d`. For your own deployment, change the project in `public_server.py`, use the same project in `viewer/firebase/settings.json` and `viewer/.firebaserc`, and update the origin allowlist/CSP in `viewer/firebase.json`. Enable Google Authentication and register the correct redirect/authorized domains in your Firebase console.

Expose **only the authenticated public-mode API** through a managed HTTPS ingress. Do not expose the default local development API. Update `apiOrigin` in `viewer/firebase/settings.json` and the CSP together when the ingress hostname changes. Keep the processing machine awake. API health is not evidence of a completed inference run.

Uploads are currently limited to 60 MiB in public mode, with bounded queue admission and a per-account daily allowance. This is not a production scaling or ten-minute-video capacity claim.

## 5. Deploy the frontend

Authenticate the Firebase CLI locally as an authorized project member. Then:

```sh
cd viewer
pnpm run build:firebase
pnpm dlx firebase-tools deploy --only hosting --project YOUR_FIREBASE_PROJECT
```

The repository CI tests the frontend but does not receive deployment credentials. Publishing code on GitHub does not by itself create cloud GPU capacity or enable GitHub OAuth. Google is the currently enabled website provider.

## 6. Earlier GPU experiment

`scripts/run_sih_pipeline.py` contains the earlier multi-model pipeline and accepts explicit upstream paths via command-line options. It needs additional CUDA-capable model environments and checkpoints. The `/mnt/...` defaults in those research scripts describe the original machine, not directories automatically shipped by this repository. Use `--help` and override them. Do not install the CUDA-heavy `pyproject.toml` dependency set as the default Mac setup.
