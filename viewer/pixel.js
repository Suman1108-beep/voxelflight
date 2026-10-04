const reduced=matchMedia('(prefers-reduced-motion: reduce)');
let chosen=false;try{chosen=localStorage.getItem('sih3d-pause-motion')==='true';}catch{}
let motionButton;
export const motionDisabled=()=>chosen||reduced.matches;
function updateMotion(){const paused=motionDisabled();document.documentElement.classList.toggle('motion-off',paused);document.body.classList.toggle('motion-paused',paused||document.hidden);document.body.classList.toggle('pixel-transition-enabled',!paused);if(motionButton){motionButton.textContent=paused?'Motion: off':'Motion: on';motionButton.setAttribute('aria-pressed',String(!paused));motionButton.title=reduced.matches?'Reduced motion is enabled in your system settings':'Toggle decorative motion';}document.dispatchEvent(new CustomEvent('sih3d:motionchange',{detail:{paused}}));}
const footer=document.querySelector('.entrance-footer,.statusbar');
if(footer){motionButton=document.createElement('button');motionButton.className='pixel-motion-control';motionButton.type='button';motionButton.addEventListener('click',()=>{chosen=!chosen;try{localStorage.setItem('sih3d-pause-motion',String(chosen));}catch{}updateMotion();});footer.append(motionButton);}
document.addEventListener('visibilitychange',updateMotion);reduced.addEventListener('change',updateMotion);updateMotion();
// The private Mac worker serves geometry, not the hosted authentication backend.
if(['127.0.0.1','localhost'].includes(location.hostname)&&location.port==='8133')document.querySelectorAll('a[href]').forEach(link=>{const target=new URL(link.href,location.href);if(['/', '/account','/signed-out','/signin-with-chatgpt','/signout-with-chatgpt'].includes(target.pathname))link.href='https://sih3d-flight-lab.diva-bhattacharya-ug.chatgpt.site'+target.pathname+target.search;});
const existing=document.querySelector('#transition-screen');
const overlay=existing??document.createElement('div');
if(!existing){overlay.className='route-transition';overlay.hidden=true;overlay.setAttribute('role','status');overlay.setAttribute('aria-live','polite');const runner=document.createElement('div');runner.className='runner';runner.setAttribute('aria-hidden','true');const title=document.createElement('h2');title.id='transition-title';const detail=document.createElement('p');detail.id='transition-detail';const dismiss=document.createElement('button');dismiss.type='button';dismiss.id='cancel-transition';dismiss.textContent='Hide loading screen';overlay.append(runner,title,detail,dismiss);document.body.append(overlay);}
// In-page views of the explorer; each also has its own address (/#guide, /#evidence, /#how).
export const views=Object.freeze({
 guide:{label:'Guide',title:'Opening the operator guide',detail:'How to fly, upload, review and export a flight.'},
 evidence:{label:'Evidence',title:'Opening the evidence',detail:'Accuracy, coverage and timing measured against the national LiDAR survey.'},
 how:{label:'How it works',title:'Opening how it works',detail:'The six processing stages and their recorded timings.'}
});
const legacyViews={validation:'evidence',pipeline:'how'};
// Copy for a link to another page: [context, title, detail].
function routeScreen(url){
 const path=url.pathname.replace(/\.html$/,'').replace(/\/+$/,'')||'/';
 if(/^\/(login|signin)/.test(path))return ['SIGN IN','Opening sign-in','Google or GitHub checks your password; VoxelFlight never sees it.'];
 if(/^\/(logout|signout|signed-out)/.test(path))return ['SIGN OUT','Signing out','Ending this sign-in session…'];
 if(path.startsWith('/account'))return ['ACCOUNT','Opening your account','Checking your sign-in session…'];
 if(['/studio','/workspace','/engine'].includes(path))return ['STUDIO','Opening your studio','Upload a flight, check it and follow your runs.'];
 const key=url.hash.slice(1),legacy=path.slice(1),view=Object.hasOwn(views,key)?views[key]:Object.hasOwn(legacyViews,legacy)?views[legacyViews[legacy]]:null;
 if(path==='/'||path==='/index'||path==='/explore'||view)return view?['EXPLORER',view.title,view.detail]:['EXPLORER','Opening the 3D explorer','The reconstructed city loads in the next view.'];
 return ['VOXELFLIGHT','Opening the page','One moment…'];
}
const card=document.createElement('div');card.className='transition-card';
const topLine=document.createElement('div');topLine.className='transition-topline';
const brand=document.createElement('span');brand.textContent='VoxelFlight';const label=document.createElement('span');label.className='transition-context';label.textContent='ON THE WAY';topLine.append(brand,label);
const lane=document.createElement('div');lane.className='scout-lane';lane.setAttribute('aria-hidden','true');lane.append(overlay.querySelector('.runner'));
const destinations=document.createElement('div');destinations.className='transition-destinations';destinations.setAttribute('aria-hidden','true');for(const [id,{label:name}] of Object.entries(views)){const item=document.createElement('span');item.dataset.destination=id;item.textContent=name;destinations.append(item);}
card.append(topLine,lane,document.querySelector('#transition-title'),document.querySelector('#transition-detail'),destinations,document.querySelector('#cancel-transition'));overlay.append(card);
export function hideTransition(){overlay.hidden=true;overlay.classList.remove('in-app-transition');}
export function showTransition(title,detail,{destination='',inApp=false,context=''}={}){
 document.querySelector('#transition-title').textContent=title;document.querySelector('#transition-detail').textContent=detail;
 overlay.dataset.destination=destination;overlay.classList.toggle('in-app-transition',inApp);label.textContent=context||(inApp?'VIEW TRANSITION':'ON THE WAY');
 destinations.hidden=!inApp;for(const item of destinations.children)item.classList.toggle('active',item.dataset.destination===destination);
 overlay.hidden=false;document.querySelector('#cancel-transition').focus({preventScroll:true});
}
function dismiss(){hideTransition();document.dispatchEvent(new Event('sih3d:transition-dismissed'));}
document.querySelector('#cancel-transition').addEventListener('click',dismiss);
window.addEventListener('pageshow',hideTransition);document.addEventListener('keydown',event=>{if(event.key==='Escape')dismiss();});
document.addEventListener('click',event=>{
 const link=event.target.closest('a');if(!link||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||link.hasAttribute('download')||link.target==='_blank')return;
 const url=new URL(link.href,location.href);if(url.origin!==location.origin||url.pathname.startsWith('/api/')||/\.(glb|ply|json|csv|las|tif|png|jpg|zip|md)$/.test(url.pathname))return;
 if(url.pathname===location.pathname&&url.search===location.search)return;
 const [context,title,detail]=routeScreen(url);
 showTransition(title,detail,{context});
});
const stage=document.querySelector('#world-stage');
stage?.addEventListener('pointermove',event=>{if(motionDisabled()||event.pointerType==='touch')return;const rect=stage.getBoundingClientRect();stage.style.setProperty('--tilt-x',`${(0.5-(event.clientY-rect.top)/rect.height)*2}deg`);stage.style.setProperty('--tilt-y',`${((event.clientX-rect.left)/rect.width-0.5)*3}deg`);});
stage?.addEventListener('pointerleave',()=>{stage.style.setProperty('--tilt-x','0deg');stage.style.setProperty('--tilt-y','0deg');});
