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
const card=document.createElement('div');card.className='transition-card';
const topLine=document.createElement('div');topLine.className='transition-topline';
const brand=document.createElement('span');brand.textContent='VoxelFlight';const label=document.createElement('span');label.className='transition-context';label.textContent='STUDIO';topLine.append(brand,label);
const lane=document.createElement('div');lane.className='scout-lane';lane.setAttribute('aria-hidden','true');lane.append(overlay.querySelector('.runner'));
const destinations=document.createElement('div');destinations.className='transition-destinations';destinations.setAttribute('aria-hidden','true');for(const name of ['Workspace','Validation','Pipeline']){const item=document.createElement('span');item.dataset.destination=name.toLowerCase();item.textContent=name;destinations.append(item);}
card.append(topLine,lane,document.querySelector('#transition-title'),document.querySelector('#transition-detail'),destinations,document.querySelector('#cancel-transition'));overlay.append(card);
export function hideTransition(){overlay.hidden=true;overlay.classList.remove('in-app-transition');}
export function showTransition(title,detail,{destination='',inApp=false}={}){
 document.querySelector('#transition-title').textContent=title;document.querySelector('#transition-detail').textContent=detail;
 overlay.dataset.destination=destination;overlay.classList.toggle('in-app-transition',inApp);label.textContent=inApp?'VIEW TRANSITION':'STUDIO';
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
 const titles=url.pathname.includes('signin-with-chatgpt')?['Connecting to ChatGPT','Your provider will securely handle sign-in.']:url.pathname.includes('signout-with-chatgpt')?['Signing out','Ending this sign-in session…']:url.pathname.includes('engine')?['Opening the engine','Your local processing controls are one step away.']:url.pathname.includes('account')?['Opening your account','Checking your sign-in session…']:url.pathname.includes('index')||url.pathname==='/workspace'?['Entering the workspace','The real reconstruction loads in the next view.']:['Back to the studio','Opening the entrance…'];
 showTransition(...titles);
});
const stage=document.querySelector('#world-stage');
stage?.addEventListener('pointermove',event=>{if(motionDisabled()||event.pointerType==='touch')return;const rect=stage.getBoundingClientRect();stage.style.setProperty('--tilt-x',`${(0.5-(event.clientY-rect.top)/rect.height)*2}deg`);stage.style.setProperty('--tilt-y',`${((event.clientX-rect.left)/rect.width-0.5)*3}deg`);});
stage?.addEventListener('pointerleave',()=>{stage.style.setProperty('--tilt-x','0deg');stage.style.setProperty('--tilt-y','0deg');});
