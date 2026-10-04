// Full-screen explorer: fly the drone's real path in sync with its video, compare the 3D model with the camera frame,
// colour the model by its error against survey LiDAR, and inspect or measure any point.
import * as THREE from 'three';
import {SceneViewer} from './scene.js';

const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const base='assets/latest/',video=$('#flight');
const json=name=>fetch(base+name).then(r=>{if(!r.ok)throw new Error(`${name}: HTTP ${r.status}`);return r.json();});
const clock=s=>{s=Math.max(0,Math.round(s));return `${Math.floor(s/60)}:${String(s%60).padStart(2,'0')}`;};
const pct=x=>`${Math.round(x*100)}%`,metres=x=>`${x.toFixed(2)} m`,minsec=s=>`${Math.floor(s/60)} min ${Math.round(s%60)} s`;
const BINS=[[3,'#2fd38a','≤ 0.3 m'],[5,'#9be15d','0.3–0.5 m'],[10,'#f4d35e','0.5–1 m'],[20,'#f59e42','1–2 m'],[254,'#e9533e','> 2 m'],[255,'#55626d','no survey point']];

let viewer,scene,evaluation,poses=[],times=[],pathLength=[],duration=150,vfov=55,mode='chase',playing=false,errors=null,lastT=-1,curT=0;

function pose(m){   // camera-to-world, row-major 3x4, OpenCV axes -> three.js camera (looks down -z, y up)
  const x=new THREE.Vector3(m[0],m[4],m[8]),y=new THREE.Vector3(m[1],m[5],m[9]),z=new THREE.Vector3(m[2],m[6],m[10]);
  const basis=new THREE.Matrix4().makeBasis(x,y.clone().negate(),z.clone().negate());
  return {p:new THREE.Vector3(m[3],m[7],m[11]),q:new THREE.Quaternion().setFromRotationMatrix(basis)};
}
function at(t){   // interpolated pose at preview-video time t
  let lo=0,hi=times.length-1;
  if(t<=times[0])hi=0;else if(t>=times[hi])lo=hi;else while(hi-lo>1){const mid=(lo+hi)>>1;if(times[mid]<=t)lo=mid;else hi=mid;}
  const f=hi>lo?(t-times[lo])/(times[hi]-times[lo]):0,a=poses[lo],b=poses[hi];
  return {i:f<.5?lo:hi,p:a.p.clone().lerp(b.p,f),q:a.q.clone().slerp(b.q,f)};
}
function heading(i){   // smoothed horizontal travel direction around keyframe i
  const a=poses[Math.max(0,i-6)].p,b=poses[Math.min(poses.length-1,i+6)].p,d=b.clone().sub(a);d.z=0;
  return d.lengthSq()>.01?d.normalize():new THREE.Vector3(1,0,0);
}
function apply(t,force=false){
  if(!poses.length||(!force&&Math.abs(t-lastT)<1e-3))return;lastT=t;
  const {i,p,q}=at(t),cam=viewer.camera;
  viewer.marker?.position.copy(p);
  if(mode==='drone'){cam.position.copy(p);cam.quaternion.copy(q);cam.fov=vfov;cam.near=2;cam.updateProjectionMatrix();}   // clip fragments right in front of the lens
  else if(mode==='chase'){const d=heading(i);cam.position.copy(p).addScaledVector(d,-16).add(new THREE.Vector3(0,0,8));cam.fov=50;cam.near=.5;cam.updateProjectionMatrix();cam.lookAt(p.clone().addScaledVector(d,10));}
  viewer.render();
  $('#timeline').value=Math.round(t/duration*1000);
  $('#clock').textContent=clock(t*scene.preview_speed);
  markThumb(i);
  $('#where').textContent=`View ${i+1} / ${poses.length} · ${Math.round(pathLength[i])} m along path · elevation ${Math.round(p.z+scene.utm_origin[2])} m`;
}
function setMode(next){
  mode=next;$$('.views button').forEach(b=>b.classList.toggle('active',b.dataset.view===mode));
  viewer.controls.enabled=mode==='orbit';
  if(mode==='orbit'){const {p,q}=at(curT);viewer.camera.near=.05;const ahead=new THREE.Vector3(0,0,-1).applyQuaternion(q).multiplyScalar(18);
    viewer.camera.fov=45;viewer.camera.updateProjectionMatrix();viewer.controls.target.copy(p.clone().add(ahead));viewer.controls.update();}
  apply(curT,true);
}
function setPlaying(on){
  playing=on;$('#play').innerHTML=`<svg class="icon"><use href="#i-${on?'pause':'play'}"/></svg>`;$('#play').setAttribute('aria-label',on?'Pause flight':'Play flight');
  if(on){const target=curT;video.play().then(()=>{if(Math.abs(video.currentTime-target)>.5)video.currentTime=target;}).catch(()=>setPlaying(false));}else video.pause();
}
function tick(){if(playing&&!video.seeking&&Math.abs(video.currentTime-curT)<3)curT=video.currentTime;else if(playing&&!video.seeking)video.currentTime=curT;if(playing||mode!=='orbit')apply(curT);requestAnimationFrame(tick);}
function seek(t){curT=Math.min(duration,Math.max(0,t));try{video.currentTime=curT;}catch{}apply(curT,true);}

function metrics(){
  const s=evaluation.surface_shape_vs_lidar,v=evaluation.visible_completeness||{},fill=evaluation.with_gap_fill?.visible_completeness;
  const rows=[['Processing time, fast mode',minsec(scene.fast_mode_seconds)],['Processing time, high quality',scene.quality_mode_seconds?minsec(scene.quality_mode_seconds):'—'],
    ['Shape error vs survey (median)',metres(s.median_m)],['Absolute position with RTK GPS','0.50 m'],
    ['Visible scene within 1 m',`${pct(v.recall_1m)}${fill?` (${pct(fill.recall_1m)} with fill)`:''}`]];
  $('#metrics').innerHTML=rows.map(([k,b])=>`<div class="metric"><span>${k}</span><b>${b}</b></div>`).join('');
}

const colorOf=e=>new THREE.Color(BINS.find(([max])=>e<=max)[1]);
async function heatmap(on){
  if(on&&!errors){toast('Loading error data');errors=new Uint8Array(await (await fetch(base+'errors.bin')).arrayBuffer());}
  viewer.object.traverse(c=>{
    if(!c.isMesh||c.userData.fill)return;
    const n=c.geometry.attributes.position.count;if(!errors||errors.length!==n)return;
    if(on){if(!c.geometry.attributes.heat){const a=new Float32Array(n*3);for(let k=0;k<n;k++){const col=colorOf(errors[k]);a[3*k]=col.r;a[3*k+1]=col.g;a[3*k+2]=col.b;}c.geometry.setAttribute('heat',new THREE.BufferAttribute(a,3));}
      c.userData.photo??=c.material;c.geometry.setAttribute('color',c.geometry.attributes.heat);c.material=new THREE.MeshBasicMaterial({vertexColors:true,side:THREE.DoubleSide});}
    else if(c.userData.photo){c.material.dispose();c.material=c.userData.photo;c.geometry.deleteAttribute('color');}
  });
  if(on&&errors){const counts=BINS.map(([max],k)=>errors.filter(e=>e<=max&&(k===0||e>BINS[k-1][0])).length);
    $('#legend').innerHTML=BINS.map(([,col,label],k)=>`<div><i style="background:${col}"></i>${label}<small>${pct(counts[k]/errors.length)}</small></div>`).join('')+'<div><small style="margin:0">Share of model vertices · evaluation-only alignment</small></div>';}
  $('#legend').hidden=!on;viewer.render();toast(on?'Colour shows distance to the national laser survey after one evaluation-only alignment':'');
}

let toastTimer;
function toast(text){clearTimeout(toastTimer);$('#toast').textContent=text;$('#toast').hidden=!text;if(text)toastTimer=setTimeout(()=>$('#toast').hidden=true,3200);}
function measured(result){if(result?.distance)toast(`Distance ${metres(result.distance)} (model shape: 0.7 m median vs survey)`);$('#measure').classList.remove('on');}

function utmToLatLon(E,N,zone=32){   // WGS84 transverse Mercator inverse (northern hemisphere)
  const a=6378137,f=1/298.257223563,k0=.9996,e2=f*(2-f),ep2=e2/(1-e2),x=E-5e5,M=N/k0,mu=M/(a*(1-e2/4-3*e2*e2/64-5*e2**3/256));
  const e1=(1-Math.sqrt(1-e2))/(1+Math.sqrt(1-e2));
  const p1=mu+(3*e1/2-27*e1**3/32)*Math.sin(2*mu)+(21*e1*e1/16-55*e1**4/32)*Math.sin(4*mu)+(151*e1**3/96)*Math.sin(6*mu);
  const C=ep2*Math.cos(p1)**2,T=Math.tan(p1)**2,Nn=a/Math.sqrt(1-e2*Math.sin(p1)**2),R=a*(1-e2)/(1-e2*Math.sin(p1)**2)**1.5,D=x/(Nn*k0);
  const lat=p1-(Nn*Math.tan(p1)/R)*(D*D/2-(5+3*T+10*C-4*C*C-9*ep2)*D**4/24+(61+90*T+298*C+45*T*T-252*ep2-3*C*C)*D**6/720);
  const lon=(D-(1+2*T+C)*D**3/6+(5-2*C+28*T-3*C*C+8*ep2+24*T*T)*D**5/120)/Math.cos(p1);
  return [lat*180/Math.PI,(zone*6-183)+lon*180/Math.PI];
}
function inspect(e){
  const hit=viewer.hit(e);if(!hit){$('#inspect').hidden=true;return;}
  const o=scene.utm_origin,E=hit.point.x+o[0],N=hit.point.y+o[1],H=hit.point.z+o[2],[lat,lon]=utmToLatLon(E,N);
  const fill=hit.object.userData.fill||hit.object.parent?.userData?.fill,err=!fill&&errors&&hit.face?errors[hit.face.a]:null;
  $('#inspect').innerHTML=`<b>${fill?'Interpolated surface (road fill)':'Measured surface'}</b><br>${lat.toFixed(6)}°, ${lon.toFixed(6)}°<br>UTM 32N  E ${E.toFixed(1)}  N ${N.toFixed(1)}<br>Elevation ${H.toFixed(1)} m${err!=null?`<br>Distance to survey ${err===255?'n/a':metres(err/10)}`:errors?'':'<br>Enable “Error vs survey” to see the local error'}`;
  const box=$('#inspect');box.hidden=false;box.style.left=Math.min(e.clientX+14,innerWidth-300)+'px';box.style.top=Math.min(e.clientY+14,innerHeight-150)+'px';
}

let thumbs=[];
function buildStrip(){
  thumbs=scene.thumbnail_frames.map(k=>({k,t:times[k]}));
  $('#strip').innerHTML=thumbs.map(({k,t})=>`<button class="thumb" data-k="${k}" title="Jump to ${clock(t*scene.preview_speed)} of the flight"><img src="${base}frames/frame_${String(k).padStart(5,'0')}.jpg" alt="Drone camera at ${clock(t*scene.preview_speed)}" loading="lazy"><span>${clock(t*scene.preview_speed)}</span></button>`).join('');
  $$('.thumb').forEach(b=>b.addEventListener('click',()=>{
    const k=Number(b.dataset.k);setPlaying(false);collapseIntro();if(mode==='orbit')setMode('chase');seek(times[k]);
    toast(document.body.classList.contains('comparing')?'Camera frame and model at this position':'Moved to this position. The inset shows the source video.');
  }));
}
let activeThumb=-1;
function markThumb(i){
  let best=-1;thumbs.forEach(({k},n)=>{if(k<=i)best=n;});if(best===activeThumb)return;activeThumb=best;
  $$('.thumb').forEach((b,n)=>b.classList.toggle('active',n===best));
  $$('.thumb')[best]?.scrollIntoView({block:'nearest',inline:'center',behavior:'smooth'});
}
function collapseIntro(){$('#intro').classList.add('collapsed');document.body.classList.add('started');}
function openPanel(tab){
  $('#panel').hidden=!tab;if(!tab){history.replaceState(null,'',location.pathname);return;}
  $$('.tabs button').forEach(b=>b.classList.toggle('active',b.dataset.tab===tab));
  $('#tab-evidence').hidden=tab!=='evidence';$('#tab-how').hidden=tab!=='how';history.replaceState(null,'',`#${tab}`);
}
function buildPanel(){
  const s=evaluation.surface_shape_vs_lidar,v=evaluation.visible_completeness||{},fill=evaluation.with_gap_fill?.visible_completeness;
  $('#big-numbers').innerHTML=[[minsec(scene.fast_mode_seconds),'10-minute video to 3D, fast mode (target < 15 min)'],[metres(s.median_m),'median shape error of the shown model vs survey'],
    ['0.5 m','absolute position with RTK GPS (0.41 m vertical)'],[pct(v.recall_1m),`visible scene within 1 m${fill?`, ${pct(fill.recall_1m)} with fill`:''}`]]
    .map(([b,t])=>`<div><b>${b}</b><small>${t}</small></div>`).join('');
  const rows=[['Fast mode (7 min 8 s)',.24],['High quality (14 min 28 s)',.379],['Maximum coverage (shown)',v.recall_1m]];if(fill)rows.push(['+ road gap fill (interpolated)',fill.recall_1m,true]);
  $('#coverage-bars').innerHTML=rows.map(([l,x,interp])=>`<div class="bar"><span>${l}</span><div class="track"><div class="fill${interp?' interp':''}" style="width:${(x*100).toFixed(1)}%"></div></div><b>${(x*100).toFixed(1)}%</b></div>`).join('');
  const m=scene.quality_stage_marks||{},t=scene.timings||{},cam=(scene.camera_solve_seconds||{}).total;
  const steps=[['1','Read the flight','Any drone video (H.264 or HEVC, 1080p or 4K) plus GPS: DJI .SRT, flight logs, GPX or CSV. Camera intrinsics are estimated when no calibration is given.',''],
    ['2','Solve the cameras','SuperPoint + LightGlue features and COLMAP photogrammetry, then locked to GPS and barometer.',cam?minsec(cam):''],
    ['3','Predict depth','Depth Anything 3 predicts depth for four overlapping crops of every keyframe, each scaled to the solved tie points.',m.depth_done_at_s?minsec(m.depth_done_at_s-m.camera_solve_done_at_s):''],
    ['4','Fuse the surface','GPU TSDF fusion in 60 m tiles keeps only surfaces the camera saw.',t['tiled GPU TSDF']?minsec(t['tiled GPU TSDF']):''],
    ['5','Texture and fill','The mesh is photo-textured from the source frames; road holes get an optional, separately scored fill.',''],
    ['6','Export six formats','OBJ, PLY, GLB, FBX, LAS and GeoTIFF in UTM, each reopened by an independent reader.',t.exports?minsec(t.exports):'']];
  $('#steps').innerHTML=steps.map(([n,b,p,time])=>`<div class="step"><i>${n}</i><div><b>${b}</b><p>${p}</p></div><span>${time}</span></div>`).join('');
}

async function init(){
  viewer=new SceneViewer($('#scene'),{onStatus:t=>{$('#loading-text').textContent=t;},onPick:measured});
  const [s,ev,cams]=await Promise.all([json('scene.json'),json('evaluation.json'),json('cameras.json')]);scene=s;evaluation=ev;
  metrics();
  await viewer.loadReconstruction(base);
  poses=cams.c2w.map(pose);times=scene.frame_times;vfov=2*Math.atan(cams.height/(2*cams.fy))*180/Math.PI;
  pathLength=poses.map((_,i)=>i);for(let i=1,sum=0;i<poses.length;i++){sum+=poses[i].p.distanceTo(poses[i-1].p);pathLength[i]=sum;}pathLength[0]=0;
  viewer.setTrajectory(poses.map(x=>({position:x.p.toArray()})));
  viewer.object.traverse(c=>{if(c.isMesh&&c.userData.fill)c.material.color.set(0x7d8fa3);});   // interpolated road reads as a muted blue-grey
  duration=scene.duration_s/scene.preview_speed;$('#total').textContent=clock(scene.duration_s);
  viewer.fit('overview');viewer.controls.enabled=false;
  buildStrip();buildPanel();
  $('#loading').classList.add('done');apply(0,true);tick();
  const hash=location.hash.slice(1);if(['evidence','how'].includes(hash)){collapseIntro();openPanel(hash);}
}

$$('[data-open]').forEach(b=>b.addEventListener('click',()=>openPanel(b.dataset.open)));
$$('.tabs button').forEach(b=>b.addEventListener('click',()=>openPanel(b.dataset.tab)));
$('#panel-close').addEventListener('click',()=>openPanel(null));
addEventListener('hashchange',()=>{const h=location.hash.slice(1);if(['evidence','how'].includes(h))openPanel(h);});
$('#start').addEventListener('click',()=>{collapseIntro();setMode('chase');setPlaying(true);});
$('#explore-free').addEventListener('click',()=>{collapseIntro();setMode('orbit');viewer.fit('overview');});
$('.brand').addEventListener('click',e=>{e.preventDefault();$('#intro').classList.toggle('collapsed');});
$('#play').addEventListener('click',()=>{if(!playing&&curT>=duration-.2)seek(0);else if(!playing){try{video.currentTime=curT;}catch{}}setPlaying(!playing);collapseIntro();});
video.addEventListener('ended',()=>setPlaying(false));
video.addEventListener('click',()=>$('#compare').click());
$('#timeline').addEventListener('input',e=>seek(e.target.value/1000*duration));
video.addEventListener('loadedmetadata',()=>{try{video.currentTime=curT;}catch{}});
$$('.views button').forEach(b=>b.addEventListener('click',()=>{if(document.body.classList.contains('comparing')&&b.dataset.view!=='drone')$('#compare').click();setMode(b.dataset.view);}));
$('#compare').addEventListener('click',()=>{
  const on=document.body.classList.toggle('comparing');$('#blend-wrap').hidden=!on;$('#compare span').textContent=on?'Exit comparison':'Compare with camera frame';
  video.style.opacity=on?$('#blend').value/100:'';try{video.currentTime=curT;}catch{}setMode(on?'drone':'chase');setTimeout(()=>{viewer.resize();apply(curT,true);},60);
  toast(on?'Source frame over the model, rendered from the same camera pose. Use the slider to blend.':'');
});
$('#blend').addEventListener('input',e=>{video.style.opacity=e.target.value/100;});
$$('[data-layer]').forEach(box=>box.addEventListener('change',async()=>{
  const on=box.checked,layer=box.dataset.layer;
  if(layer==='heat'){await heatmap(on);if(on&&mode!=='orbit')setMode('orbit');if(on)viewer.fit('overview');}else if(layer==='cloud')await viewer.toggleCloud(on);else viewer.setLayer(layer,on);
}));
$('#measure').addEventListener('click',()=>{if(mode!=='orbit')setMode('orbit');setPlaying(false);viewer.setMeasuring(true);$('#measure').classList.add('on');toast('Select two points on the model');});
$('#snapshot').addEventListener('click',()=>{const a=document.createElement('a');a.href=viewer.snapshot();a.download='voxelflight-view.png';a.click();});
$('#downloads-button').addEventListener('click',()=>{const d=$('#downloads');d.hidden=!d.hidden;$('#downloads-button').setAttribute('aria-expanded',String(!d.hidden));});
$('#export-obj').addEventListener('click',async()=>{const r=await viewer.export('obj');const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([r.data],{type:r.type}));a.download=r.name;a.click();});
const canvas=$('#scene');let down=null;
canvas.addEventListener('pointerdown',e=>{down=[e.clientX,e.clientY];if(mode!=='orbit'&&!document.body.classList.contains('comparing')){setMode('orbit');}});
canvas.addEventListener('pointerup',e=>{if(down&&Math.hypot(e.clientX-down[0],e.clientY-down[1])<5&&!viewer.measuring)inspect(e);down=null;});
addEventListener('keydown',e=>{if(e.target.tagName==='INPUT')return;if(e.code==='Escape'){openPanel(null);return;}if(e.code==='Space'){e.preventDefault();$('#play').click();}
  if(e.code==='ArrowRight'||e.code==='ArrowLeft')seek(curT+(e.code==='ArrowRight'?2:-2));});
init().catch(err=>{$('#loading-text').textContent='The model could not be loaded: '+err.message;console.error(err);});
