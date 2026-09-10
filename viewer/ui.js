
import {escapeHTML,formatTime} from './core.js';
export const $=s=>document.querySelector(s);
const paths={
 cube:'m12 3 9 5v9l-9 5-9-5V8l9-5Z M3 8l9 5 9-5 M12 13v9',
 mesh:'M4 4h16v16H4z M4 4l16 16 M20 4 4 20 M12 4v16 M4 12h16',
 points:'M5 5h.01 M12 5h.01 M19 5h.01 M5 12h.01 M12 12h.01 M19 12h.01 M5 19h.01 M12 19h.01 M19 19h.01',
 path:'M4 18c0-9 16 0 16-12 M2 18h4 M18 6h4 M4 16v4 M20 4v4',
 grid:'M4 4h16v16H4z M4 12h16 M12 4v16',
 wire:'m12 3 9 17H3L12 3Z M3 20l9-6 9 6 M12 3v11',
 image:'M4 4h16v16H4z M4 17l5-6 4 4 3-3 4 5 M16 8h.01',
 upload:'M12 16V3 M7 8l5-5 5 5 M4 15v6h16v-6',
 download:'M12 3v13 M7 11l5 5 5-5 M4 16v5h16v-5',
 ruler:'m3 16 13-13 5 5L8 21l-5-5Z M7 12l3 3 M10 9l3 3 M13 6l3 3',
 monitor:'M3 4h18v13H3z M8 21h8 M12 17v4',
 plus:'M12 4v16 M4 12h16',
 reset:'M4 10a8 8 0 1 1 1 8 M4 4v6h6',
 camera:'M3 7h4l2-3h6l2 3h4v13H3V7Z M16 13a4 4 0 1 1-8 0 4 4 0 0 1 8 0',
 expand:'M4 9V4h5 M15 4h5v5 M20 15v5h-5 M9 20H4v-5',
 orbit:'M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0 M3 8c-5 8 13 14 18 8 M4 4l-1 4 4 1',
 film:'M3 3h18v18H3z M7 3v18 M17 3v18 M3 8h4 M3 16h4 M17 8h4 M17 16h4',
 play:'m9 5 11 7-11 7V5Z',
 pause:'M8 5v14 M16 5v14',
 previous:'M5 5v14 M19 5 8 12l11 7V5Z',
 next:'M19 5v14 M5 5l11 7-11 7V5Z',
 external:'M13 4h7v7 M20 4 10 14 M10 4H4v16h16v-6',
 sliders:'M4 7h16 M4 17h16 M9 4v6 M16 14v6',
 chart:'M4 4v16h16 M8 15l4-5 4 2 4-7',
 info:'M12 16v-4 M12 8h.01 M22 12a10 10 0 1 1-20 0 10 10 0 0 1 20 0',
 close:'M6 6l12 12 M18 6 6 18'
};
export function icon(name){return `<svg class="icon" viewBox="0 0 24 24" aria-hidden="true"><path d="${paths[name]??paths.cube}"/></svg>`;}
export function hydrateIcons(root=document){root.querySelectorAll('[data-icon]').forEach(el=>{el.innerHTML=icon(el.dataset.icon);});}
let timer;
export function toast(message){const el=$('#toast');el.textContent=message;el.hidden=false;clearTimeout(timer);timer=setTimeout(()=>el.hidden=true,4500);}
export function status(message){$('#status-text').textContent=message;}
export function openDialog(id){const d=$(id);if(d&&!d.open)d.showModal();}
export function setupDialogs(){document.querySelectorAll('.close-dialog').forEach(b=>b.addEventListener('click',()=>b.closest('dialog').close()));document.querySelectorAll('dialog').forEach(d=>d.addEventListener('click',e=>{if(e.target===d){const r=d.getBoundingClientRect();if(e.clientX<r.left||e.clientX>r.right||e.clientY<r.top||e.clientY>r.bottom)d.close();}}));}
export function drawCoverage(coverage,trajectory,frame=0){
 const canvas=$('#coverage-map'),ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,pad=22;
 ctx.clearRect(0,0,w,h);ctx.fillStyle='#0c161e';ctx.fillRect(0,0,w,h);
 ctx.strokeStyle='#20303c';ctx.lineWidth=.8;
 for(let x=0;x<w;x+=40){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}
 for(let y=0;y<h;y+=40){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
 const [min,max]=coverage.bounds;
 const xs=trajectory.map(t=>t.position[0]),ys=trajectory.map(t=>t.position[1]);
 const bounds=[Math.min(min[0],...xs),Math.min(min[1],...ys),Math.max(max[0],...xs),Math.max(max[1],...ys)];
 const scale=Math.min((w-pad*2)/Math.max(bounds[2]-bounds[0],1),(h-pad*2)/Math.max(bounds[3]-bounds[1],1));
 const ox=(w-(bounds[2]-bounds[0])*scale)/2,oy=(h-(bounds[3]-bounds[1])*scale)/2;
 const project=(x,y)=>[ox+(x-bounds[0])*scale,h-oy-(y-bounds[1])*scale];
 for(const [x,y,count] of coverage.cells){const p=project(min[0]+x*coverage.resolution_m,min[1]+y*coverage.resolution_m);ctx.fillStyle=`rgba(139,211,181,${Math.min(.7,.12+Math.log1p(count)/10)})`;ctx.fillRect(p[0],p[1],Math.max(1,coverage.resolution_m*scale),Math.max(1,coverage.resolution_m*scale));}
 ctx.strokeStyle='#edca8b';ctx.lineWidth=1.5;ctx.beginPath();trajectory.forEach((t,i)=>{const p=project(...t.position);if(i===0)ctx.moveTo(...p);else ctx.lineTo(...p);});ctx.stroke();
 const p=project(...trajectory[Math.min(frame,trajectory.length-1)].position);ctx.fillStyle='#e6fff0';ctx.beginPath();ctx.arc(...p,3.8,0,Math.PI*2);ctx.fill();
 ctx.fillStyle='#829fab';ctx.font='12px sans-serif';ctx.fillText(coverage.frame?'Local XY':'N ↑',w-65,20);
}
export function populateReports({evaluation,assets,timings,photos}){
 const shape=evaluation.trajectory_shape_error_after_global_sim3;
 const total=Object.values(timings).reduce((a,b)=>a+b,0);
 $('#rmse').textContent=shape.rmse_m.toFixed(3);
 $('#point-count').textContent=assets.clean_point_count.toLocaleString();
 $('#triangle-count').textContent=assets.mesh_triangles.toLocaleString();
 $('#runtime').textContent=formatTime(total);
 const metrics=[['Trajectory shape',shape.rmse_m.toFixed(3)+' m','Global Sim(3) alignment',''],['Absolute positioning',evaluation.absolute_georeferencing_error.rmse_m.toFixed(3)+' m','Compared in global coordinates','warn'],['Total recorded runtime',formatTime(total),'For a 60-second input',''],['Reconstructed points',assets.clean_point_count.toLocaleString(),'After fusion and filtering','']];
 $('#validation-metrics').innerHTML=metrics.map(([name,value,caption,style])=>`<div class="report-metric ${style}"><span>${name}</span><strong>${value}</strong><small>${caption}</small></div>`).join('');
 $('#error-details').innerHTML=[['Median camera error',shape.median_m.toFixed(3)+' m'],['95th-percentile camera error',shape.p95_m.toFixed(3)+' m'],['Maximum camera error',shape.max_m.toFixed(3)+' m'],['Ground truth in construction','No'],['Surface accuracy','Not measured']].map(([a,b])=>`<div><dt>${a}</dt><dd>${b}</dd></div>`).join('');
 $('#appearance-report').innerHTML=photos.map(p=>`<div class="appearance-result"><img src="assets/photoreal_preview_${p.id}.png" alt="Observed and reconstructed views for ${p.name}"><div><h3>${p.name}</h3><p>PSNR ${p.psnr.toFixed(2)} dB<br>SSIM ${p.ssim.toFixed(3)} · LPIPS ${p.lpips.toFixed(3)}</p></div></div>`).join('');
 const sumPrefix=prefix=>Object.entries(timings).filter(([k])=>prefix.some(p=>k.startsWith(p))).reduce((s,[,v])=>s+v,0);
 const steps=[['Prepare the flight','Discover video, normalize telemetry and camera calibration.',['01','04','05']],['Select keyframes','Extract a quality-aware sequence and create overlapping windows.',['02','03']],['Estimate camera motion','Use calibrated MASt3R-SLAM for global visual anchors.',['06','07']],['Reconstruct local geometry','MapAnything predicts metric depth and camera poses in each window.',['08']],['Fuse & export','Fuse visual and GPS information, filter geometry and export mesh / GIS assets.',['09','10']],['Refine appearance','Optimize optional Gaussian-splat tiles from predicted poses and observed images.',['11']]];
 $('#pipeline-steps').innerHTML=steps.map(([title,detail,prefix],i)=>`<article class="pipeline-step"><span class="step-no">0${i+1}</span><h2>${title}</h2><p>${detail}</p><span>${sumPrefix(prefix).toFixed(1)} s</span></article>`).join('');
}
