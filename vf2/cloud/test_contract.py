"""Contract tests for api.py without Modal or a GPU: LocalStore, a fake runner, and the test-only auth bypass.
Run: VF_INSECURE_TEST_AUTH=1 python -m pytest -q vf2/cloud/test_contract.py (the flag is also set below)."""
import json
import os
import sys
import time
from pathlib import Path

import pytest

os.environ["VF_INSECURE_TEST_AUTH"] = "1"          # test only: "Bearer test-user-x" authenticates as test-user-x
sys.path.insert(0, str(Path(__file__).resolve().parent))
import api                                           # noqa: E402
from fastapi.testclient import TestClient            # noqa: E402

SITE = "https://voxelflight-3d.web.app"
A = {"Authorization": "Bearer test-user-a", "Origin": SITE}
B = {"Authorization": "Bearer test-user-b", "Origin": SITE}
CSV = b"source_frame,timestamp_s,latitude,longitude,altitude_m\n0,0,47.38,8.54,463\n1,0.03,47.38,8.54,463\n"


class FakeRunner:
    def __init__(self):
        self.is_ready, self.calls, self.cancelled, self.done = True, {}, [], {}

    def ready(self):
        return self.is_ready

    def submit(self, job_id, options):
        self.calls[f"call-{job_id}"] = (job_id, options)
        return f"call-{job_id}"

    def poll(self, handle):
        return self.done.get(handle)

    def cancel(self, handle):
        self.cancelled.append(handle)


def fake_inspect(video, telemetry, calibration, folder):
    (Path(folder) / "telemetry.csv").write_bytes(Path(telemetry).read_bytes())
    return dict(duration_s=1801.0 if Path(video).read_bytes().startswith(b"LONG") else 60.0, profile="street")


@pytest.fixture
def env(tmp_path):
    store, runner = api.LocalStore(tmp_path), FakeRunner()
    return store, runner, TestClient(api.create_app(store, runner, inspect=fake_inspect))


def submit(client, headers=A, video=b"\x00" * 1024, name="flight.mp4", telemetry=(("gps.csv", CSV),), **data):
    files = [("video", (name, video, "video/mp4"))] + [("telemetry", t) for t in telemetry]
    return client.post("/api/jobs", headers=headers, files=files, data={"max_frames": "24", "size": "350", **data})


def finish(store, runner, job_id, ok=True):
    """What reconstruct() in modal_app.py leaves behind."""
    out = store.root / "jobs" / job_id / "output"
    (out / "frames").mkdir(parents=True)
    (out / "report.json").write_text(json.dumps({"points": 10, "triangles": 5, "device": "A100", "warnings": [], "frames": [{"index": 0}]}))
    (out / "reconstruction_mesh.glb").write_bytes(b"glTF" + b"\x00" * 60)
    (out / "frames/frame_00000.jpg").write_bytes(b"\xff\xd8jpg")
    (out / "pipeline.log").write_text("private")
    store.put(f"run:{job_id}", dict(state="complete" if ok else "failed", message=None if ok else "The reconstruction failed during exports.",
                                    finished_at=time.time()))


def test_health_is_public_with_limits_and_cors(env):
    _, runner, c = env
    r = c.get("/api/health", headers={"Origin": SITE})
    assert r.status_code == 200 and r.headers["access-control-allow-origin"] == SITE and r.headers["cache-control"] == "no-store"
    h = r.json()
    assert h["ready"] and h["queue_available"] and not h["busy"]
    assert h["limits"]["video_bytes"] == 1024 ** 3 and h["limits"]["video_seconds"] == 1800
    runner.is_ready = False
    assert c.get("/api/health").json()["ready"] is False
    assert c.get("/api/health", headers={"Origin": "https://evil.example"}).status_code == 403
    pre = c.options("/api/jobs", headers={"Origin": "https://voxelflight-3d.firebaseapp.com", "Access-Control-Request-Method": "POST",
                                          "Access-Control-Request-Headers": "authorization"})
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == "https://voxelflight-3d.firebaseapp.com"
    assert c.options("/api/jobs", headers={"Origin": "https://evil.example", "Access-Control-Request-Method": "POST"}).status_code == 400
    assert c.get("/docs").status_code == 404 and c.get("/").status_code == 404


def test_auth_required(env):
    _, _, c = env
    assert c.get("/api/jobs").status_code == 401
    assert c.get("/api/jobs", headers={"Authorization": "Bearer forged"}).status_code == 401
    assert c.get("/api/jobs", headers=A).json() == []
    with pytest.raises(ValueError):
        api._test_verifier("eyJhbGciOi.real.firebase")


def test_create_job_contract_and_ownership(env):
    store, runner, c = env
    r = submit(c)
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["state"] == "queued" and job["name"] == "flight.mp4" and job["queue_position"] == 1
    assert job["max_frames"] == 360 and job["resolution"] == 518 and "owner" not in job and "call" not in job
    assert runner.calls[f"call-{job['id']}"] == (job["id"], {"profile": "street"})
    folder = store.root / "jobs" / job["id"]
    assert (folder / "input.mp4").stat().st_size == 1024 and (folder / "telemetry.csv").read_bytes() == CSV
    assert [j["id"] for j in c.get("/api/jobs", headers=A).json()] == [job["id"]]
    assert c.get(f"/api/jobs/{job['id']}", headers=A).json()["state"] == "queued"
    assert c.get(f"/api/jobs/{job['id']}", headers=B).status_code == 404       # private to its owner
    assert c.get("/api/jobs", headers=B).json() == []
    assert c.get("/api/jobs/not-a-uuid", headers=A).status_code == 404
    assert submit(c).status_code == 409                                         # one active run per account


def test_rejected_uploads_do_not_count(env):
    store, _, c = env
    assert submit(c, telemetry=()).status_code == 400                           # GPS required by the GPU pipeline
    assert submit(c, name="flight.avi").status_code == 400
    assert submit(c, telemetry=(("gps.kml", b"<kml/>"),)).status_code == 400
    assert submit(c, max_frames="5000").status_code == 400
    r = submit(c, video=b"LONG" + b"\x00" * 100)
    assert r.status_code == 400 and "30 minutes" in r.json()["detail"]
    assert submit(c, video=b"").status_code == 400
    assert store.values("job:") == []
    assert submit(c).status_code == 202


def test_size_caps(env, monkeypatch):
    store, _, c = env
    r = c.post("/api/jobs", headers={**A, "Content-Length": str(api.MAX_BODY + 1)}, content=b"x")
    assert r.status_code == 413                                                 # declared length over 1 GiB + telemetry
    monkeypatch.setattr(api, "MAX_VIDEO", 512)
    assert submit(c, video=b"\x00" * 600).status_code == 413                    # actual video bytes over the cap
    monkeypatch.setattr(api, "MAX_BODY", 1000)
    chunked = TestClient(api.create_app(store, FakeRunner(), inspect=fake_inspect))
    r = chunked.post("/api/jobs", headers={**A, "Content-Type": "multipart/form-data; boundary=x"},
                     content=(b"x" * 500 for _ in range(10)))                   # no Content-Length: counted while received
    assert r.status_code == 413 and store.values("job:") == []


def test_progress_completion_and_assets(env):
    store, runner, c = env
    job = submit(c).json()
    store.put(f"run:{job['id']}", dict(state="running", progress=dict(stage="Surface fusion", progress=0.7, detail="GPU TSDF fusion")))
    r = c.get(f"/api/jobs/{job['id']}", headers=A).json()
    assert r["state"] == "running" and r["progress"]["stage"] == "Surface fusion"
    assert c.get("/api/health").json()["queued"] == 1
    assert c.get(f"/api/jobs/{job['id']}/assets/report.json", headers=A).status_code == 404   # not complete yet
    finish(store, runner, job["id"])
    assert c.get(f"/api/jobs/{job['id']}", headers=A).json()["state"] == "complete"
    base = f"/api/jobs/{job['id']}/assets/"
    rep = c.get(base + "report.json", headers=A)
    assert rep.status_code == 200 and rep.json()["points"] == 10 and "device" not in rep.json()
    glb = c.get(base + "reconstruction_mesh.glb", headers=A)
    assert glb.status_code == 200 and glb.content.startswith(b"glTF") and glb.headers["content-type"] == "model/gltf-binary"
    assert glb.headers["access-control-allow-origin"] == SITE
    assert c.get(base + "frames/frame_00000.jpg", headers=A).content == b"\xff\xd8jpg"
    for bad in ("pipeline.log", "..%2F..%2Fkv%2Fjob.json", "frames%2F..%2Freport.json", "frames/frame_1.jpg", "pointcloud.ply"):
        assert c.get(base + bad, headers=A).status_code == 404, bad
    assert c.get(base + "report.json", headers=B).status_code == 404
    assert submit(c).status_code == 202                                         # previous run finished


def test_failure_crash_and_cancel(env):
    store, runner, c = env
    job = submit(c).json()
    finish(store, runner, job["id"], ok=False)
    r = c.get(f"/api/jobs/{job['id']}", headers=A).json()
    assert r["state"] == "failed" and "exports" in r["message"]
    job = submit(c).json()                                                      # crash: call ended without a report
    runner.done[f"call-{job['id']}"] = False
    r = c.get(f"/api/jobs/{job['id']}", headers=A).json()
    assert r["state"] == "failed" and "GPU worker stopped" in r["message"]
    job = submit(c).json()                                                      # third and last run today
    assert c.post(f"/api/jobs/{job['id']}/cancel", headers=B).status_code == 404
    assert c.post(f"/api/jobs/{job['id']}/cancel", headers=A).json() == {"state": "cancelled"}
    assert runner.cancelled == [f"call-{job['id']}"]
    assert c.post(f"/api/jobs/{job['id']}/cancel", headers=A).status_code == 409
    r = submit(c)
    assert r.status_code == 429 and "3 runs per day" in r.json()["detail"]


def test_cancel_running_and_queue_cap(env, monkeypatch):
    store, runner, c = env
    job = submit(c).json()
    store.put(f"run:{job['id']}", dict(state="running"))
    assert c.post(f"/api/jobs/{job['id']}/cancel", headers=A).json() == {"state": "cancelling"}
    assert c.get(f"/api/jobs/{job['id']}", headers=A).json()["state"] == "cancelling"
    runner.done[f"call-{job['id']}"] = False                                     # Modal terminated the container
    assert c.get(f"/api/jobs/{job['id']}", headers=A).json()["state"] == "cancelled"
    monkeypatch.setattr(api, "MAX_ACTIVE", 1)
    assert submit(c).status_code == 202
    r = submit(c, headers=B)
    assert r.status_code == 429 and "queue is full" in r.json()["detail"]
    assert c.get("/api/health").json()["queue_available"] is False
    runner.is_ready = False
    monkeypatch.setattr(api, "MAX_ACTIVE", 6)
    assert submit(c, headers=B).status_code == 503


def test_real_inspect_converts_srt(tmp_path):
    """worker.inspect_inputs on a synthetic 2 s video + DJI-style SRT (CPU only)."""
    cv2, np = pytest.importorskip("cv2"), pytest.importorskip("numpy")
    import worker
    video = tmp_path / "clip.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 10, (64, 48))
    for i in range(20):
        w.write(np.full((48, 64, 3), i * 10, np.uint8))
    w.release()
    srt = "\n\n".join(f"{i + 1}\n00:00:0{i // 10},{i % 10}00 --> 00:00:0{(i + 1) // 10},{(i + 1) % 10}00\n"
                      f"[latitude: {28.6 + i * 1e-5:.6f}] [longitude: 77.2] [rel_alt: 80.0 abs_alt: 300.0]" for i in range(20))
    (tmp_path / "gps.srt").write_text(srt)
    info = worker.inspect_inputs(video, tmp_path / "gps.srt", None, tmp_path)
    assert info["frames"] == 20 and abs(info["duration_s"] - 2) < 0.2 and info["profile"] == "aerial"
    rows = (tmp_path / "telemetry.csv").read_text().splitlines()
    assert rows[0] == "source_frame,timestamp_s,latitude,longitude,altitude_m,baro_alt_m" and len(rows) == 21
    csv = "source_frame,timestamp_s,latitude,longitude,altitude_m\n" + "".join(f"{i},{i / 10},47.38,8.54,{463 + i % 3}\n" for i in range(20))
    (tmp_path / "zurich.csv").write_text(csv)                                    # vf2 per-frame contract: passed through
    assert worker.inspect_inputs(video, tmp_path / "zurich.csv", None, tmp_path)["profile"] == "street"
    assert (tmp_path / "telemetry.csv").read_text() == csv
    with pytest.raises(ValueError, match="GPS telemetry"):
        (tmp_path / "bad.csv").write_text("time,speed\n0,1\n1,2\n")
        worker.inspect_inputs(video, tmp_path / "bad.csv", None, tmp_path)
    with pytest.raises(ValueError, match="Calibration"):
        (tmp_path / "cam.json").write_text(json.dumps({"width": 1920, "height": 1080, "fx": 1, "fy": 1, "cx": 1, "cy": 1}))
        worker.inspect_inputs(video, tmp_path / "gps.srt", tmp_path / "cam.json", tmp_path)
    store, runner = api.LocalStore(tmp_path / "store"), FakeRunner()
    c = TestClient(api.create_app(store, runner))                               # default inspect = worker.inspect_inputs
    r = c.post("/api/jobs", headers=A, files=[("video", ("clip.mp4", video.read_bytes(), "video/mp4")),
                                              ("telemetry", ("gps.srt", srt.encode(), "application/x-subrip"))])
    assert r.status_code == 202, r.text
    assert runner.calls[f"call-{r.json()['id']}"][1] == {"profile": "aerial"}
    assert (store.root / "jobs" / r.json()["id"] / "telemetry.csv").is_file()
