"""VoxelFlight cloud API: the public_server.py / mac_server.py public job contract over pluggable storage + GPU runner.

Endpoints (unchanged for viewer/engine.js): GET /api/health, GET /api/jobs, POST /api/jobs (multipart video, telemetry,
calibration, max_frames, size), GET /api/jobs/{id}, POST /api/jobs/{id}/cancel, GET /api/jobs/{id}/assets/{name}.
Every route except health needs a Firebase ID token (Authorization: Bearer). Jobs are private to their owner.

This module imports without Modal. A store keeps JSON records ("job:<id>" written here, "run:<id>" written by the GPU
worker) and job files; a runner starts/polls/cancels GPU jobs. modal_app.py provides the Modal ones, tests a fake.
"""
import asyncio
import json
import mimetypes
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.middleware.cors import CORSMiddleware

PROJECT = "voxelflight-3d"                      # viewer/firebase/settings.json projectId
ORIGINS = [f"https://{PROJECT}.web.app", f"https://{PROJECT}.firebaseapp.com"]
MAX_VIDEO, MAX_TELEMETRY, MAX_CALIBRATION = 1024 ** 3, 10 * 1024 ** 2, 1024 ** 2
MAX_BODY = MAX_VIDEO + MAX_TELEMETRY + 2 * 1024 ** 2
MAX_SECONDS = 1800                              # reject videos longer than 30 min
MAX_GPU = int(os.environ.get("VF_MAX_GPU_JOBS", 2))        # GPU containers at once (modal max_containers)
MAX_ACTIVE = int(os.environ.get("VF_MAX_ACTIVE_JOBS", 6))  # admitted jobs, running + queued
DAILY_RUNS = int(os.environ.get("VF_DAILY_RUNS", 3))
KEYFRAMES, SIZE = 360, 518                      # vf2 fast profile (README "One command")
ACTIVE = {"uploading", "queued", "running", "cancelling"}
ASSETS = {"report.json", "reconstruction_mesh.glb", "pointcloud.ply", "reconstruction.obj", "camera_poses.npz", "trajectory.csv",
          "reconstruction_utm.las", "surface_model.tif", "reconstruction_mesh.fbx", "metadata.json"}
FRAME = re.compile(r"frames/frame_\d{5}\.jpg")
GENERIC_FAILURE = "Processing did not complete. Try a shorter, steady video with per-frame GPS telemetry."
mimetypes.add_type("model/gltf-binary", ".glb")


class FirebaseVerifier:
    """Firebase ID-token check against Google's public certificates (public_security.py); no service account."""

    def __init__(self, project):
        import cachecontrol, requests
        from google.auth.transport.requests import Request as GoogleRequest
        self.project, self.lock = project, threading.Lock()
        self.request = GoogleRequest(session=cachecontrol.CacheControl(requests.Session()))

    def __call__(self, token):
        from google.oauth2.id_token import verify_firebase_token
        if not token or len(token) > 8192:
            raise ValueError("Invalid token")
        with self.lock:
            claims = verify_firebase_token(token, self.request, audience=self.project)
        uid = claims.get("sub")
        if claims.get("iss") != f"https://securetoken.google.com/{self.project}":
            raise ValueError("Invalid issuer")
        if not isinstance(uid, str) or not 1 <= len(uid) <= 128:
            raise ValueError("Invalid subject")
        if claims.get("auth_time", time.time() + 1) > time.time():
            raise ValueError("Invalid authentication time")
        if claims.get("firebase", {}).get("sign_in_provider") not in {"google.com", "github.com"}:
            raise ValueError("Unsupported provider")
        return uid


def _test_verifier(token):
    """VF_INSECURE_TEST_AUTH=1 only (test_contract.py): the bearer string itself is the user id."""
    if not token.startswith("test-user-"):
        raise ValueError("Invalid token")
    return token


class LocalStore:
    """Records as JSON files and job folders on one disk (tests, or a single-host deployment)."""

    def __init__(self, root):
        self.root = Path(root)
        (self.root / "kv").mkdir(parents=True, exist_ok=True)

    def _path(self, key):
        return self.root / "kv" / (key.replace(":", "_") + ".json")

    def get(self, key):
        p = self._path(key)
        return json.loads(p.read_text()) if p.is_file() else None

    def put(self, key, value):
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value))
        tmp.replace(self._path(key))

    def delete(self, key):
        self._path(key).unlink(missing_ok=True)

    def values(self, prefix):
        return [json.loads(p.read_text()) for p in (self.root / "kv").glob(prefix.replace(":", "_") + "*.json")]

    def save_inputs(self, job_id, folder):
        shutil.copytree(folder, self.root / "jobs" / job_id, dirs_exist_ok=True)

    def read(self, job_id, name):
        p = self.root / "jobs" / job_id / "output" / name
        if not p.is_file():
            return None
        def chunks():
            with p.open("rb") as f:
                while b := f.read(1 << 20):
                    yield b
        return chunks()


class BodyLimit:
    """Count the bytes actually received, so a missing or false Content-Length cannot fill the disk."""

    def __init__(self, app, limit):
        self.app, self.limit = app, limit

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] in ("GET", "HEAD"):
            return await self.app(scope, receive, send)
        seen = 0

        async def bounded():
            nonlocal seen
            message = await receive()
            seen += len(message.get("body", b""))
            if seen > self.limit:
                raise HTTPException(413, "Upload too large. Choose a video under 1 GiB.")
            return message
        await self.app(scope, bounded, send)


def create_app(store, runner, verify_token=None, inspect=None, project=PROJECT):
    """runner: ready() -> bool, submit(job_id, options) -> handle, poll(handle) -> None|bool, cancel(handle).
    inspect(video, telemetry, calibration, folder) -> {duration_s, ...}; raises ValueError with a user message."""
    if verify_token is None:
        verify_token = _test_verifier if os.environ.get("VF_INSECURE_TEST_AUTH") == "1" else FirebaseVerifier(project)
    if inspect is None:
        from worker import inspect_inputs as inspect
    origins = {f"https://{project}.web.app", f"https://{project}.firebaseapp.com"}
    admission = asyncio.Lock()
    app = FastAPI(title="VoxelFlight cloud", docs_url=None, redoc_url=None, openapi_url=None)
    app.add_middleware(BodyLimit, limit=MAX_BODY)   # innermost, so its 413 is raised inside the endpoint

    @app.middleware("http")
    async def guard(request, call_next):
        path = request.url.path
        if not path.startswith("/api/"):
            return JSONResponse({"detail": "Not found"}, status_code=404)
        if request.headers.get("origin") and request.headers["origin"] not in origins:
            return JSONResponse({"detail": "Origin not allowed"}, status_code=403)
        if path != "/api/health":
            authorization = request.headers.get("authorization", "")
            if not authorization.startswith("Bearer "):
                return JSONResponse({"detail": "Sign in to continue."}, status_code=401)
            try:
                request.state.uid = await asyncio.to_thread(verify_token, authorization[7:])
            except Exception:
                return JSONResponse({"detail": "Your session could not be verified. Sign in again."}, status_code=401)
        if request.method not in {"GET", "HEAD"}:
            try:
                length = int(request.headers.get("content-length", "0"))
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY:
                return JSONResponse({"detail": "Upload too large. Choose a video under 1 GiB."}, status_code=413)
        response = await call_next(request)
        response.headers.update({"Cache-Control": "no-store", "X-Content-Type-Options": "nosniff", "Referrer-Policy": "no-referrer"})
        return response

    app.add_middleware(CORSMiddleware, allow_origins=sorted(origins), allow_methods=["GET", "HEAD", "POST"],
                       allow_headers=["Authorization", "Content-Type"], expose_headers=["Content-Disposition"], max_age=3600)

    # ---- job state: "job:<id>" is ours; "run:<id>" (state/progress/message) belongs to the GPU worker ----
    def current(job):
        """Merge worker progress; settle jobs whose worker finished, crashed, timed out or was cancelled."""
        if job["state"] not in ACTIVE:
            return job
        run = store.get(f"run:{job['id']}") or {}
        done = None
        if run.get("state") == "complete":
            done = dict(state="complete", message=run.get("message", "New reconstruction is ready."))
        elif run.get("state") == "failed":
            done = dict(state="cancelled" if job["state"] == "cancelling" else "failed", message=run.get("message") or GENERIC_FAILURE)
        elif job["state"] == "uploading":
            if job["created_at"] < time.time() - 3600:
                done = dict(state="interrupted", message="The processing service restarted before this upload finished.")
        elif job.get("call") and runner.poll(job["call"]) is not None:   # worker exited without reporting
            done = dict(state="cancelled", message="Reconstruction cancelled.") if job["state"] == "cancelling" else \
                dict(state="failed", message="The GPU worker stopped (time limit or crash). " + GENERIC_FAILURE)
        if done:
            job = dict(job, **done, progress=run.get("progress"), finished_at=run.get("finished_at", time.time()))
            store.put(f"job:{job['id']}", job)
            return job
        state = "running" if job["state"] == "queued" and run.get("state") == "running" else job["state"]
        return dict(job, state=state, progress=run.get("progress"))

    def all_jobs():
        return [current(j) for j in store.values("job:")]

    def visible(job, jobs=None):
        v = {k: x for k, x in job.items() if k not in ("owner", "call", "options")}
        v["queue_position"] = 0
        if v["state"] == "queued":
            queued = sorted((j for j in (jobs or all_jobs()) if j["state"] == "queued"), key=lambda j: j["created_at"])
            v["queue_position"] = next((i + 1 for i, j in enumerate(queued) if j["id"] == v["id"]), 1)
        return v

    def owned(job_id, request):
        try:
            if str(uuid.UUID(job_id)) != job_id:
                raise ValueError
        except ValueError:
            raise HTTPException(404, "Job not found")
        job = store.get(f"job:{job_id}")
        if not job or job.get("owner") != request.state.uid:
            raise HTTPException(404, "Job not found")
        return current(job)

    @app.get("/api/health")
    def health():
        active = [j for j in all_jobs() if j["state"] in ACTIVE]
        running = sum(j["state"] in ("running", "cancelling") for j in active)
        return dict(ready=bool(runner.ready()), busy=running >= MAX_GPU, queue_available=len(active) < MAX_ACTIVE, queued=len(active),
                    limits=dict(video_bytes=MAX_VIDEO, video_seconds=MAX_SECONDS, max_keyframes=KEYFRAMES, max_concurrent_jobs=MAX_GPU),
                    accuracy="Not independently evaluated")

    @app.get("/api/jobs")
    def jobs(request: Request):
        rows = all_jobs()
        mine = sorted((j for j in rows if j.get("owner") == request.state.uid), key=lambda j: j["created_at"], reverse=True)[:50]
        return [visible(j, rows) for j in mine]

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        return visible(owned(job_id, request))

    def admit(uid):
        rows = all_jobs()
        if sum(j["state"] in ACTIVE for j in rows) >= MAX_ACTIVE:
            raise HTTPException(429, "The processing queue is full. Try again shortly.")
        recent = [j for j in rows if j.get("owner") == uid and j["created_at"] > time.time() - 86400]
        if len(recent) >= DAILY_RUNS:
            raise HTTPException(429, f"This account has reached the demo limit of {DAILY_RUNS} runs per day.")
        if any(j["state"] in ACTIVE for j in recent):
            raise HTTPException(409, "Your previous reconstruction is still in progress.")
        if not runner.ready():
            raise HTTPException(503, "The processing service is not ready yet")
        job = dict(id=str(uuid.uuid4()), name="Uploading video", state="uploading", created_at=time.time(), owner=uid,
                   max_frames=KEYFRAMES, resolution=SIZE)
        store.put(f"job:{job['id']}", job)
        return job

    async def upload(file, destination, limit):
        count = 0
        with destination.open("xb") as stream:
            while chunk := await file.read(1024 ** 2):
                count += len(chunk)
                if count > limit:
                    raise HTTPException(413, "File exceeds the upload limit")
                stream.write(chunk)
        if not count:
            raise HTTPException(400, "The selected file is empty")

    @app.post("/api/jobs")
    async def create_job(request: Request):
        async with admission:   # check + reserve atomically per container; the upload itself runs outside the lock
            job = await asyncio.to_thread(admit, request.state.uid)
        folder = Path(tempfile.mkdtemp(prefix=f"vf-{job['id']}-"))
        try:
            async with request.form(max_files=3, max_fields=4, max_part_size=MAX_TELEMETRY) as form:
                video, telemetry, camera = form.get("video"), form.get("telemetry"), form.get("calibration")
                if not hasattr(video, "filename"):
                    raise HTTPException(400, "Choose an MP4 or MOV video")
                suffix = Path(video.filename or "").suffix.lower()
                if suffix not in {".mp4", ".mov"}:
                    raise HTTPException(400, "Choose an MP4 or MOV video")
                try:   # the viewer's preview settings are accepted for compatibility; the GPU profile is fixed
                    frames, size = int(form.get("max_frames", 24)), int(form.get("size", 350))
                except (TypeError, ValueError):
                    raise HTTPException(400, "Invalid processing settings")
                profile = form.get("profile", "auto")
                if not 4 <= frames <= 96 or size not in {224, 252, 350, 420, 518} or profile not in {"auto", "street", "aerial"}:
                    raise HTTPException(400, "Unsupported processing settings")
                if not (hasattr(telemetry, "filename") and telemetry.filename):
                    raise HTTPException(400, "GPS telemetry is required: a per-frame CSV, JSON or DJI SRT file.")
                ext = Path(telemetry.filename).suffix.lower()
                if ext not in {".csv", ".json", ".srt"}:
                    raise HTTPException(400, "GPS must be CSV, JSON or SRT")
                await upload(video, folder / f"input{suffix}", MAX_VIDEO)
                await upload(telemetry, folder / f"telemetry_source{ext}", MAX_TELEMETRY)
                calibration = None
                if hasattr(camera, "filename") and camera.filename:
                    if Path(camera.filename).suffix.lower() != ".json":
                        raise HTTPException(400, "Calibration must be JSON")
                    calibration = folder / "calibration_source.json"
                    await upload(camera, calibration, MAX_CALIBRATION)
                name = Path(video.filename).name[:180]
            try:
                info = await asyncio.to_thread(inspect, folder / f"input{suffix}", folder / f"telemetry_source{ext}", calibration, folder)
            except ValueError as error:
                raise HTTPException(400, str(error))
            if info["duration_s"] > MAX_SECONDS:
                raise HTTPException(400, "Videos up to 30 minutes are supported. Split longer footage.")
            options = dict(profile=info.get("profile", "aerial") if profile == "auto" else profile)
            await asyncio.to_thread(store.save_inputs, job["id"], folder)
            call = await asyncio.to_thread(runner.submit, job["id"], options)
            job.update(name=name, state="queued", call=call, options=options, duration_s=round(info["duration_s"], 1))
            await asyncio.to_thread(store.put, f"job:{job['id']}", job)
            return JSONResponse(visible(job, await asyncio.to_thread(all_jobs)), status_code=202)
        except BaseException:
            if job["state"] == "uploading":   # rejected uploads do not count against the daily limit
                await asyncio.to_thread(store.delete, f"job:{job['id']}")
            raise
        finally:
            shutil.rmtree(folder, ignore_errors=True)

    @app.post("/api/jobs/{job_id}/cancel")
    def cancel(job_id: str, request: Request):
        job = owned(job_id, request)
        if job["state"] == "queued":
            runner.cancel(job["call"])
            store.put(f"job:{job_id}", dict(job, state="cancelled", message="Reconstruction cancelled.", finished_at=time.time()))
            return {"state": "cancelled"}
        if job["state"] != "running":
            raise HTTPException(409, "This job is not running")
        runner.cancel(job["call"])
        store.put(f"job:{job_id}", dict(job, state="cancelling"))
        return {"state": "cancelling"}

    @app.get("/api/jobs/{job_id}/assets/{filename:path}")
    def artifact(job_id: str, filename: str, request: Request):
        job = owned(job_id, request)
        if job["state"] != "complete" or (filename not in ASSETS and not FRAME.fullmatch(filename)):
            raise HTTPException(404, "Artifact not found")
        data = store.read(job_id, filename)
        if data is None:
            raise HTTPException(404, "Artifact not found")
        if filename == "report.json":
            report = json.loads(b"".join(data))
            report.pop("device", None)
            return JSONResponse(report)
        return StreamingResponse(data, media_type=mimetypes.guess_type(filename)[0] or "application/octet-stream")

    return app
