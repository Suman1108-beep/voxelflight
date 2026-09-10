"""Reconstruction API with separate local and authenticated-public modes.

The default app is loopback-only. public_server explicitly opts into Firebase
authentication, per-user artifacts, bounded admissions and one GPU worker.
"""
from __future__ import annotations
import asyncio
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import secrets
import shutil
import sys
import time
import uuid

from fastapi import FastAPI,HTTPException,Request,UploadFile
from fastapi.responses import FileResponse,JSONResponse
from starlette.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware
from mac_reconstruct import MODEL_REVISION,ROOT,write_json

MAX_VIDEO=1024**3
MAX_TELEMETRY=10*1024**2


def processing_readiness():
    """Probe real hardware and cached weights, independently of API unit tests."""
    import torch
    weights=ROOT/f"cache-mac/huggingface/hub/models--facebook--map-anything-apache/snapshots/{MODEL_REVISION}/model.safetensors"
    encoder=ROOT/"cache-mac/torch/hub/facebookresearch_dinov2_main/hubconf.py"
    available=torch.backends.mps.is_available()
    cached=weights.is_file() and weights.stat().st_size==4_914_062_480
    encoder_cached=encoder.is_file()
    return dict(available=available,cached=cached,encoder_cached=encoder_cached,
                ready=available and cached and encoder_cached)


def create_app(data_root=None,worker_command=None,public_project=None,verify_token=None):
    data=Path(data_root or ROOT/"runs-mac/jobs");data.mkdir(parents=True,exist_ok=True)
    session=secrets.token_urlsafe(32)
    processes={};tasks=set();lock=asyncio.Lock();admission=asyncio.Lock()
    public=bool(public_project);video_limit=60*1024**2 if public else MAX_VIDEO
    allowed_origins={f'https://{public_project}.web.app',f'https://{public_project}.firebaseapp.com'} if public else set()
    if public and verify_token is None:
        from public_security import FirebaseVerifier
        verify_token=FirebaseVerifier(public_project)

    @asynccontextmanager
    async def lifespan(app):
        # A killed/restarted server must not leave a permanent fictional 'running' state.
        for meta in data.glob("*/job.json"):
            try:
                value=json.loads(meta.read_text())
                if value.get("state") in {"uploading","queued","running","cancelling"}:
                    value.update(state="interrupted",message="The local engine stopped before this job finished.")
                    write_json(meta,value)
            except (ValueError,OSError):pass
        yield
        for process in list(processes.values()):
            if process.returncode is None:process.terminate()
        for task in tasks:task.cancel()
        if tasks:await asyncio.gather(*tasks,return_exceptions=True)

    app=FastAPI(title="SIH3D Mac Engine",docs_url=None,redoc_url=None,openapi_url=None,lifespan=lifespan)

    @app.middleware("http")
    async def local_guard(request,call_next):
        if public:
            path=request.url.path
            if not path.startswith('/api/'):
                return JSONResponse({'detail':'Not found'},status_code=404)
            if request.headers.get('origin') and request.headers['origin'] not in allowed_origins:
                return JSONResponse({'detail':'Origin not allowed'},status_code=403)
            if path!='/api/health':
                authorization=request.headers.get('authorization','')
                if not authorization.startswith('Bearer '):return JSONResponse({'detail':'Sign in to continue.'},status_code=401)
                try:request.state.uid=await asyncio.to_thread(verify_token,authorization[7:])
                except Exception:return JSONResponse({'detail':'Your session could not be verified. Sign in again.'},status_code=401)
            if request.method not in {'GET','HEAD'}:
                try:length=int(request.headers.get('content-length','-1'))
                except ValueError:length=-1
                if length<0 or length>video_limit+MAX_TELEMETRY+2*1024**2:
                    return JSONResponse({'detail':'Upload too large. Choose a video under 60 MiB.'},status_code=413)
                # Enforce the actual received length too, before multipart parsing.
                received=0;receive=request._receive
                async def bounded_receive():
                    nonlocal received
                    message=await receive();received+=len(message.get('body',b''))
                    if received>video_limit+MAX_TELEMETRY+2*1024**2:raise HTTPException(413,'Upload too large')
                    return message
                request._receive=bounded_receive
            response=await call_next(request)
            response.headers['Cache-Control']='no-store'
            response.headers['X-Content-Type-Options']='nosniff'
            response.headers['Referrer-Policy']='no-referrer'
            return response
        host=request.headers.get("host","").split(":")[0]
        if host not in {"127.0.0.1","localhost"}:
            return JSONResponse({"detail":"This engine is available only on localhost."},status_code=403)
        origin=request.headers.get("origin")
        if origin and origin != f"{request.url.scheme}://{request.headers.get('host')}":
            return JSONResponse({"detail":"Cross-origin access is disabled."},status_code=403)
        if request.url.path.startswith("/api/") and request.headers.get("sec-fetch-site")=="cross-site":
            return JSONResponse({"detail":"Open the local engine directly."},status_code=403)
        if request.method not in {"GET","HEAD"}:
            if not secrets.compare_digest(request.headers.get("x-sih3d-session",""),session):
                return JSONResponse({"detail":"Reload the local engine to authorize this request."},status_code=403)
            if request.url.path=="/api/jobs" and request.method=="POST":
                try:length=int(request.headers.get("content-length","-1"))
                except ValueError:length=-1
                if length<0 or length>MAX_VIDEO+MAX_TELEMETRY+1024**2:
                    return JSONResponse({"detail":"An upload length is required; maximum video size is 1 GiB."},status_code=413)
        response=await call_next(request)
        if request.url.path.startswith("/api/"):
            response.headers["Cache-Control"]="no-store"
            response.headers["X-Content-Type-Options"]="nosniff"
        response.headers["Referrer-Policy"]="same-origin"
        return response

    if public:
        app.add_middleware(CORSMiddleware,allow_origins=list(allowed_origins),allow_methods=['GET','HEAD','POST'],allow_headers=['Authorization','Content-Type'],expose_headers=['Content-Disposition'],max_age=3600)

    def job_path(job_id,request=None):
        try:
            if str(uuid.UUID(job_id))!=job_id:raise ValueError()
        except ValueError:raise HTTPException(404,"Job not found")
        folder=data/job_id
        if not (folder/"job.json").is_file():raise HTTPException(404,"Job not found")
        if public and (request is None or read_job(folder).get('owner')!=request.state.uid):raise HTTPException(404,'Job not found')
        return folder

    def read_job(folder):
        value=json.loads((folder/"job.json").read_text())
        progress=folder/"output/progress.json"
        if progress.is_file():value["progress"]=json.loads(progress.read_text())
        return value

    def visible_job(value):
        value=dict(value);value.pop('owner',None)
        if public:
            if value.get('state')=='failed':value['message']='Processing did not complete. Try a shorter, steady video.'
            elif value.get('state')=='cancelled':value['message']='Reconstruction cancelled.'
            elif value.get('state')=='interrupted':value['message']='The processing service restarted before this run finished.'
            progress=value.get('progress')
            if progress:
                value['progress']={key:progress[key] for key in ['stage','progress'] if key in progress}
                value['progress']['detail']='Reconstruction is processing your sampled video frames.'
            value['queue_position']=0
            if value.get('state')=='queued':
                waiting=sorted((read_job(p.parent) for p in data.glob('*/job.json')),key=lambda j:j['created_at'])
                value['queue_position']=next((i+1 for i,j in enumerate(j for j in waiting if j['state']=='queued') if j['id']==value['id']),1)
        return value

    @app.get("/api/health")
    async def health():
        state=processing_readiness()
        available=state['available'];cached=state['cached'];ready=state['ready']
        if public:
            active=sum(read_job(p.parent).get('state') in {'uploading','queued','running','cancelling'} for p in data.glob('*/job.json'))
            return dict(ready=ready,busy=lock.locked(),queue_available=active<3,queued=active,limits=dict(video_bytes=video_limit,max_keyframes=48,max_concurrent_jobs=1),accuracy='Not independently evaluated')
        return dict(engine="sih3d-mac-v1",device="Apple MPS" if available else "Unavailable",gpu_available=available,weights_cached=cached,encoder_cached=state['encoder_cached'],ready=ready,busy=lock.locked(),session=session,network_required=False,limits=dict(video_bytes=MAX_VIDEO,max_keyframes=96,max_concurrent_jobs=1),accuracy="Not independently evaluated")

    @app.get("/api/jobs")
    async def jobs(request:Request):
        rows=[read_job(p.parent) for p in data.glob('*/job.json')]
        if public:rows=[row for row in rows if row.get('owner')==request.state.uid]
        return [visible_job(row) for row in sorted(rows,key=lambda j:j['created_at'],reverse=True)[:50]]

    @app.get("/api/jobs/{job_id}")
    async def job(job_id:str,request:Request):return visible_job(read_job(job_path(job_id,request)))

    async def upload(file,destination,limit):
        count=0
        with destination.open("xb") as stream:
            while chunk:=await file.read(1024**2):
                count+=len(chunk)
                if count>limit:raise HTTPException(413,"File exceeds the upload limit")
                stream.write(chunk)
        if not count:raise HTTPException(400,"The selected file is empty")

    async def watch(job_id,folder,command):
        if public:await lock.acquire()
        value=read_job(folder)
        try:
            if value.get('state')=='cancelled':return
            if public:value['state']='running';write_json(folder/'job.json',value)
            env=os.environ.copy();env.update(HF_HUB_OFFLINE="1",HF_DATASETS_OFFLINE="1",PYTORCH_ENABLE_MPS_FALLBACK="1")
            with (folder/"worker.log").open("wb") as log:
                process=await asyncio.create_subprocess_exec(*command,cwd=ROOT,env=env,stdout=log,stderr=asyncio.subprocess.STDOUT)
                processes[job_id]=process
                try:code=await asyncio.wait_for(process.wait(),1200 if public else None)
                except asyncio.TimeoutError:
                    process.kill();await process.wait();code=-1
            latest=read_job(folder)
            if latest.get("state")=="cancelling":
                value.update(state="cancelled",message="Processing cancelled. Uploaded files remain on this Mac.")
            elif code==0 and (folder/"output/report.json").is_file():
                value.update(state="complete",message="New reconstruction is ready.")
            else:
                detail=latest.get("progress",{}).get("detail","The worker stopped. Inspect the local worker log.")
                value.update(state="failed",message=detail)
        except Exception as error:
            value.update(state="failed",message=f"Worker could not start: {error}")
        finally:
            processes.pop(job_id,None);value["finished_at"]=time.time();write_json(folder/"job.json",value)
            if lock.locked():lock.release()

    @app.post("/api/jobs")
    async def create_job(request:Request):
        if public:
            await admission.acquire()
        else:
            if lock.locked():raise HTTPException(409,"The Mac is processing another run. Wait or cancel that run.")
            await lock.acquire()
        folder=None
        try:
            if public:
                rows=[read_job(p.parent) for p in data.glob('*/job.json')]
                if sum(row['state'] in {'uploading','queued','running','cancelling'} for row in rows)>=3:raise HTTPException(429,'The processing queue is full. Try again shortly.')
                recent=[row for row in rows if row.get('owner')==request.state.uid and row['created_at']>time.time()-86400]
                if len(recent)>=3:raise HTTPException(429,'This account has reached the demo limit of three runs per day.')
                if any(row['state'] in {'uploading','queued','running','cancelling'} for row in recent):raise HTTPException(409,'Your previous reconstruction is still in progress.')
                if sum(p.stat().st_size for p in data.rglob('*') if p.is_file())>5*1024**3 or shutil.disk_usage(data).free<8*1024**3:raise HTTPException(503,'Processing storage is full. Please try later.')
            async with request.form(max_files=3,max_fields=3,max_part_size=MAX_TELEMETRY) as form:
                video=form.get("video");telemetry=form.get("telemetry")
                # Starlette supplies its base UploadFile type, so use the documented interface.
                if not hasattr(video,"filename"):raise HTTPException(400,"Choose an MP4 or MOV video")
                suffix=Path(video.filename or "").suffix.lower()
                if suffix not in {".mp4",".mov"}:raise HTTPException(400,"Choose an MP4 or MOV video")
                try:frames=int(form.get("max_frames",24));size=int(form.get("size",350))
                except (ValueError,TypeError):raise HTTPException(400,"Invalid processing settings")
                if not 4<=frames<=(48 if public else 96) or size not in {224,252,350,420}:raise HTTPException(400,"Unsupported processing settings")
                if not (await health())["ready"]:raise HTTPException(503,"The processing service is not ready yet" if public else "Model cache or Apple GPU is not ready yet")
                job_id=str(uuid.uuid4());folder=data/job_id;folder.mkdir()
                value=dict(id=job_id,name=Path(video.filename).name[:180],state="uploading",created_at=time.time(),max_frames=frames,resolution=size)
                if public:value['owner']=request.state.uid
                write_json(folder/"job.json",value)
                video_path=folder/f"input{suffix}";await upload(video,video_path,video_limit)
                command=list(worker_command or [sys.executable,"-u",str(ROOT/"mac_reconstruct.py")])
                command += ["--video",str(video_path),"--output",str(folder/"output"),"--max-frames",str(frames),"--size",str(size)]
                if hasattr(telemetry,"filename") and telemetry.filename:
                    ext=Path(telemetry.filename).suffix.lower()
                    if ext not in {".csv",".json",".srt"}:raise HTTPException(400,"GPS must be CSV, JSON or SRT")
                    telemetry_path=folder/f"telemetry{ext}";await upload(telemetry,telemetry_path,MAX_TELEMETRY)
                    command += ["--telemetry",str(telemetry_path)]
                camera=form.get("calibration")
                if hasattr(camera,"filename") and camera.filename:
                    if Path(camera.filename).suffix.lower()!=".json":raise HTTPException(400,"Calibration must be JSON")
                    camera_path=folder/"calibration.json";await upload(camera,camera_path,1024**2)
                    command += ["--calibration",str(camera_path)]
                value["state"]="queued" if public else "running";write_json(folder/"job.json",value)
            task=asyncio.create_task(watch(job_id,folder,command));tasks.add(task);task.add_done_callback(tasks.discard)
            return JSONResponse(visible_job(value),status_code=202)
        except BaseException:
            if folder and (folder/"job.json").exists():
                value["state"]="failed";value["message"]="Input upload or validation failed.";write_json(folder/"job.json",value)
            if not public:lock.release()
            raise
        finally:
            if public:admission.release()

    @app.post("/api/jobs/{job_id}/cancel")
    async def cancel(job_id:str,request:Request):
        folder=job_path(job_id,request);value=read_job(folder);process=processes.get(job_id)
        if public and value['state']=='queued':
            value['state']='cancelled';write_json(folder/'job.json',value);return {'state':'cancelled'}
        if not process or process.returncode is not None:raise HTTPException(409,"This job is not running")
        value["state"]="cancelling";write_json(folder/"job.json",value);process.terminate()
        return {"state":"cancelling"}

    @app.get("/api/jobs/{job_id}/assets/{filename:path}")
    async def artifact(job_id:str,filename:str,request:Request):
        folder=job_path(job_id,request)/"output"
        permitted={"report.json","reconstruction_mesh.glb","pointcloud.ply","reconstruction.obj","camera_poses.npz","trajectory.csv","reconstruction_utm.las","surface_model.tif"}
        is_frame=filename.startswith("frames/frame_") and Path(filename).suffix==".jpg" and len(Path(filename).parts)==2
        target=(folder/filename).resolve()
        if (filename not in permitted and not is_frame) or not target.is_relative_to(folder.resolve()) or not target.is_file():
            raise HTTPException(404,"Artifact not found")
        if public and filename=='report.json':
            report=json.loads(target.read_text());report.pop('device',None);report.pop('gpu_allocation_at_end_gib',None)
            report['warnings']=[w.replace('Experimental Mac inference','Experimental reconstruction').replace('this Mac mode','this processing mode') for w in report.get('warnings',[])]
            return JSONResponse(report)
        return FileResponse(target)

    @app.get("/workspace")
    async def workspace_page():
        page=ROOT/"viewer/dist/client/workspace.html"
        if not page.is_file():
            raise HTTPException(503,"The local viewer is not built. Use the hosted website or build the local viewer first.")
        return FileResponse(page)

    # Only the built public site is served, never source code, weights, private inputs or logs.
    # Importing the API must also work in a fresh source checkout with no UI build.
    # Public-mode requests are already restricted to /api/ by the guard above.
    site_directory=ROOT/"viewer/dist/client"
    if not public and site_directory.is_dir():
        app.mount("/",StaticFiles(directory=site_directory,html=True),name="viewer")
    return app


app=create_app()
