// Studio: check a drone video and its GPS log in the browser, submit it to the processing service, follow your runs,
// and open your own GLB/PLY models. Uses the same /api job contract as the processing backend (vf2/cloud).
import {validateVideoFile} from './video-input.js';
import {sessionHeaders} from './session-bridge.js';

const $=s=>document.querySelector(s);
const esc=s=>String(s).replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const mmss=s=>`${Math.floor(s/60)}:${String(Math.round(s%60)).padStart(2,'0')}`;
const hosted=document.querySelector('meta[name="hosting-platform"]')?.content==='firebase';
let apiOrigin='',health=null,signedIn=false,videoOk=false,videoInfo=null,activeJob=null,viewer=null,submitting=false;
const terminal=new Set(['complete','failed','cancelled','interrupted']);

async function api(path,options={}){
  const headers=path==='health'?{}:await sessionHeaders();
  const r=await fetch(`${apiOrigin}/api/${path}`,{...options,credentials:'omit',headers:{...options.headers,...headers},signal:AbortSignal.timeout(options.method?600000:15000)});
  if(!r.ok){let m=`Request failed (${r.status})`;try{m=(await r.json()).detail??m;}catch{}throw new Error(m);}
  return r.json();
}
function service(state,title,detail){$('#service').dataset.state=state;$('#service-title').textContent=title;$('#service-detail').textContent=detail;}
async function connect(){
  try{
    const settings=hosted?await fetch('/auth-settings.json',{cache:'no-store'}).then(r=>r.json()):{};
    if(!settings.apiOrigin)throw new Error('not configured');
    apiOrigin=new URL(settings.apiOrigin).origin;health=await api('health');
    service(health.ready?'ready':'offline',health.ready?'Processing service available':'Processing service busy',health.ready?'Jobs start as soon as the upload completes.':'Jobs are queued and start when a GPU is free.');
    if(signedIn)await runs();
  }catch{
    health=null;service('offline','Processing service offline','Uploads are disabled during evaluation. File checks run locally and nothing is uploaded. The Zurich sample shows a finished result.');
  }
  refresh();
}
function refresh(){
  const b=$('#submit');
  b.disabled=!videoOk||!health?.ready||!signedIn||submitting;
  b.textContent=submitting?'Uploading…':!videoOk?'Select a video':!signedIn?'Sign in to submit':!health?'Processing service offline':'Start reconstruction';
}

function checkList(items){return `<ul>${items.map(([cls,text])=>`<li class="${cls}">${esc(text)}</li>`).join('')}</ul>`;}
$('#video').addEventListener('change',()=>{
  const file=$('#video').files[0],box=$('#video-check');videoOk=false;videoInfo=null;box.hidden=!file;
  if(!file){refresh();return;}
  const problem=validateVideoFile(file);
  $('#video-drop').classList.toggle('filled',!problem);
  if(problem){box.className='check text';box.innerHTML=checkList([['bad',problem]]);refresh();return;}
  videoOk=true;box.className='check';box.innerHTML=`<canvas width="320" height="180"></canvas>${checkList([['',`${file.name} · ${(file.size/1024**2).toFixed(0)} MB`],['warn','Reading video']])}`;refresh();
  const v=document.createElement('video'),url=URL.createObjectURL(file);v.muted=true;v.preload='metadata';v.src=url;
  v.addEventListener('loadedmetadata',()=>{videoInfo={duration:v.duration,width:v.videoWidth,height:v.videoHeight};v.currentTime=Math.min(2,v.duration/2);});
  v.addEventListener('seeked',()=>{
    box.querySelector('canvas').getContext('2d').drawImage(v,0,0,320,180);URL.revokeObjectURL(url);
    const {duration,width,height}=videoInfo,items=[['',`${file.name} · ${(file.size/1024**2).toFixed(0)} MB`],['',`${width}×${height} · ${mmss(duration)} long`]];
    if(duration>1800){items.push(['bad','Longer than 30 minutes; split the recording']);videoOk=false;}
    else if(width<1280)items.push(['warn','Below 720p; expect reduced detail']);
    else items.push(['',width>=3840?'4K, processed at full resolution':'Resolution sufficient']);
    items.push(['',`Estimated processing time (fast mode): ${mmss(Math.max(60,duration*.72))}`]);
    box.querySelector('ul').outerHTML=checkList(items);gpsCheck();refresh();
  },{once:true});
  v.addEventListener('error',()=>{URL.revokeObjectURL(url);box.className='check text';box.innerHTML=checkList([['',`${file.name} · ${(file.size/1024**2).toFixed(0)} MB`],['warn','Preview not supported for this codec (often HEVC/H.265); the server transcodes it']]);refresh();});
});

async function gpsCheck(){
  const file=$('#telemetry').files[0],box=$('#gps-check');box.hidden=!file;if(!file)return;
  $('#gps-drop').classList.add('filled');box.className='check text';
  const text=await file.slice(0,8*1024**2).text(),name=file.name.toLowerCase(),items=[];let first=null,count=0,span=null;
  if(name.endsWith('.srt')){
    count=(text.match(/-->/g)||[]).length;
    const m=text.match(/latitude\s*:\s*([-\d.]+)\]?\s*\[?longt?itude\s*:\s*([-\d.]+)/i),g=text.match(/GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)/);
    if(m)first=[+m[1],+m[2]];else if(g)first=Math.abs(+g[1])>90?[+g[2],+g[1]]:[+g[2],+g[1]];
    const times=[...text.matchAll(/(\d+):(\d\d):(\d\d)[,.](\d+)\s*-->/g)];if(times.length){const t=times.at(-1);span=+t[1]*3600+ +t[2]*60+ +t[3];}
    items.push(['',`DJI SRT log, ${count.toLocaleString()} samples`]);
  }else if(name.endsWith('.gpx')){
    count=(text.match(/<trkpt/g)||[]).length;const m=text.match(/<trkpt[^>]*lat="([-\d.]+)"[^>]*lon="([-\d.]+)"/);if(m)first=[+m[1],+m[2]];
    items.push([count?'':'bad',`GPX track, ${count.toLocaleString()} points`]);
  }else if(name.endsWith('.csv')){
    const lines=text.split(/\r?\n/).filter(Boolean),head=lines[0].toLowerCase();count=lines.length-1;
    const hasLat=/lat/.test(head),hasLon=/lon|lng/.test(head);
    items.push([hasLat&&hasLon?'':'bad',hasLat&&hasLon?`CSV flight log, ${count.toLocaleString()} rows${/isvideo|camera\./.test(head)?' (DJI flight record)':''}`:'No latitude/longitude columns found']);
  }else{
    try{const j=JSON.parse(text);count=Array.isArray(j)?j.length:0;items.push([count?'':'bad',`JSON, ${count} samples`]);}catch{items.push(['bad','Not valid JSON']);}
  }
  if(first&&Math.abs(first[0])>.01)items.push(['',`First fix ${first[0].toFixed(5)}, ${first[1].toFixed(5)}`]);
  if(span!=null&&videoInfo)items.push([Math.abs(span-videoInfo.duration)<5?'':'warn',`Covers ${mmss(span)} of ${mmss(videoInfo.duration)} video`]);
  box.innerHTML=checkList(items);
}
$('#telemetry').addEventListener('change',gpsCheck);
for(const id of ['video-drop','gps-drop']){const el=$('#'+id);
  el.addEventListener('dragover',e=>{e.preventDefault();el.classList.add('over');});el.addEventListener('dragleave',()=>el.classList.remove('over'));el.addEventListener('drop',()=>el.classList.remove('over'));}

$('#run-form').addEventListener('submit',async e=>{
  e.preventDefault();if($('#submit').disabled)return;
  const data=new FormData(e.currentTarget);for(const n of ['telemetry','calibration'])if(!data.get(n)?.size)data.delete(n);
  submitting=true;refresh();$('#form-note').className='note';$('#form-note').textContent='Uploading';
  try{const job=await api('jobs',{method:'POST',body:data});activeJob=job.id;progress(job);$('#form-note').className='note good';$('#form-note').textContent='Job submitted. You can leave this page; progress is saved.';await runs();}
  catch(err){$('#form-note').className='note bad';$('#form-note').textContent=err.message;}
  finally{submitting=false;refresh();}
});
function progress(job){
  $('#progress').hidden=false;$('#progress-stage').textContent=job.state==='queued'?`Queued · position ${job.queue_position||1}`:job.progress?.stage??job.state;
  if(Number.isFinite(job.progress?.progress))$('#progress-bar').value=job.progress.progress;else $('#progress-bar').removeAttribute('value');
  $('#progress-detail').textContent=job.progress?.detail??job.message??'';$('#cancel').hidden=terminal.has(job.state);
}
$('#cancel').addEventListener('click',async()=>{if(activeJob)try{await api(`jobs/${activeJob}/cancel`,{method:'POST'});}catch(err){$('#form-note').textContent=err.message;}});
async function runs(){
  const list=await api('jobs');$('#runs').querySelectorAll('.run.user').forEach(n=>n.remove());
  $('#runs').insertAdjacentHTML('afterbegin',list.map(j=>`<button class="run user" data-job="${esc(j.id)}" type="button"><img src="assets/latest/poster.jpg" alt=""><span><b>${esc(j.name)}</b><small>${esc(j.state)}${j.created?` · ${esc(new Date(j.created*1000).toLocaleString())}`:''}</small></span><em>${j.state==='complete'?'Open':esc(j.state)}</em></button>`).join(''));
  $('#runs').querySelectorAll('[data-job]').forEach(b=>b.addEventListener('click',()=>openJob(b.dataset.job)));
  const running=list.find(j=>!terminal.has(j.state));if(running){activeJob=running.id;progress(running);}
}
async function openJob(id){
  const job=await api(`jobs/${id}`);if(job.state!=='complete'){activeJob=id;progress(job);return;}
  await ensureViewer();$('#viewer-title').textContent=job.name;$('#viewer-dialog').showModal();
  await viewer.loadReconstruction(`${apiOrigin}/api/jobs/${id}/assets/`,{getHeaders:sessionHeaders});viewer.resize();viewer.fit('overview');
}
async function poll(){if(activeJob&&health)try{const job=await api(`jobs/${activeJob}`);progress(job);if(terminal.has(job.state)){activeJob=null;await runs();}}catch{}setTimeout(poll,activeJob?3000:20000);}
$('#refresh').addEventListener('click',connect);

async function ensureViewer(){
  if(viewer)return;const {SceneViewer}=await import('./scene.js');
  viewer=new SceneViewer($('#viewer-canvas'),{onPick:r=>{if(r?.distance)$('#viewer-note').textContent=`Distance ${r.distance.toFixed(2)} model units`;}});viewer.controls.enabled=true;
}
$('#model-file').addEventListener('change',async()=>{
  const file=$('#model-file').files[0];if(!file)return;await ensureViewer();
  $('#viewer-title').textContent=file.name;$('#viewer-dialog').showModal();viewer.resize();
  try{const s=await viewer.importFile(file);viewer.styleMesh();viewer.resize();viewer.fit('overview');$('#viewer-stats').textContent=`${s.vertices.toLocaleString()} vertices${s.triangles?` · ${s.triangles.toLocaleString()} triangles`:''} · extent ${s.extent.map(x=>x.toFixed(1)).join(' × ')}`;}
  catch(err){$('#viewer-stats').textContent=err.message;}
});
$('#viewer-close').addEventListener('click',()=>$('#viewer-dialog').close());
document.querySelectorAll('[data-fit]').forEach(b=>b.addEventListener('click',()=>viewer?.fit(b.dataset.fit)));
$('#viewer-measure').addEventListener('click',()=>{viewer?.setMeasuring(true);$('#viewer-note').textContent='Click two points on the model';});

function account(profile){
  signedIn=!!profile;$('#signin-banner').hidden=signedIn||!hosted;
  $('#avatar').textContent=profile?.initial??'?';$('#account-name').textContent=profile?.name??'Not signed in';
  $('#account-mail').textContent=profile?`${profile.email??''}${profile.provider?` · ${profile.provider}`:''}`:'Sign in to submit jobs';
  $('#account-signin').hidden=signedIn;$('#signout').hidden=!signedIn;   // #signout is wired to Firebase sign-out by the auth module
  $('#greeting-kicker').textContent=profile?`Studio · ${profile.name}`:'Studio';
  refresh();if(signedIn&&health)runs().catch(()=>{});
}
addEventListener('voxelflight:authchange',e=>account(e.detail.profile??null));
if(!hosted){$('#signin-banner').hidden=true;}
fetch('assets/latest/evaluation.json').then(r=>r.json()).then(e=>{$('#demo-meta').textContent=`10-min video · coverage ${Math.round(e.visible_completeness.recall_1m*100)}% · shape error ${e.surface_shape_vs_lidar.median_m.toFixed(2)} m`;}).catch(()=>{});
connect();poll();
