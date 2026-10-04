import {initializeApp} from 'firebase/app';
import {getAuth,onAuthStateChanged,setPersistence,browserLocalPersistence,browserSessionPersistence,GoogleAuthProvider,GithubAuthProvider,signInWithPopup,signInWithRedirect,getRedirectResult,signOut} from 'firebase/auth';
import {showTransition,hideTransition} from '/pixel.js';
import {displayProfile,safeReturnTo,authMessage} from './auth-policy.js';
import {registerSessionProvider} from '/session-bridge.js';

const $=selector=>document.querySelector(selector);
const page=document.body.dataset.page;
let auth,settings,ready=false,busy=false,pendingProvider=null;
let resolveInitial;
const initialUser=new Promise(resolve=>{resolveInitial=resolve;});
registerSessionProvider(async()=>{await initialUser;if(!auth?.currentUser)throw new Error('Sign in to start a reconstruction.');return {Authorization:`Bearer ${await auth.currentUser.getIdToken()}`};});
const returnTo=safeReturnTo(new URLSearchParams(location.search).get('return_to')||'/workspace',location.origin);
const output=$('#auth-message')||$('#account-error')||$('#logout-error');
function message(text){if(output)output.textContent=text;}
function setBusy(value){busy=value;document.querySelectorAll('[data-firebase-provider]').forEach(button=>{button.disabled=value||!ready||!settings?.providers?.includes(button.dataset.firebaseProvider);});const fallback=$('#redirect-signin');if(fallback)fallback.disabled=value||!ready;}
function renderUser(user){
  resolveInitial(user);const profile=displayProfile(user);
  window.dispatchEvent(new CustomEvent('voxelflight:authchange',{detail:{signedIn:!!user,profile}}));
  document.querySelectorAll('[data-account-link]').forEach(link=>{
    link.textContent=profile?`Hi, ${profile.name.split(/\s+/)[0]}`:'Sign in';
    link.title=profile?`Welcome, ${profile.name}`:'Sign in to VoxelFlight';
    link.href=profile?'/account':`/login?return_to=${encodeURIComponent(safeReturnTo(location.pathname+location.hash,location.origin))}`;
  });
  if(page==='signin'){
    const welcome=$('#returning-account');welcome.hidden=!profile;
    if(profile){$('#returning-name').textContent=`Welcome back, ${profile.name}.`;$('#returning-link').href=returnTo;}
    $('#signin-methods').hidden=!!profile;$('#redirect-signin').hidden=!!profile;
    return;
  }
  if(page==='account'){
    $('#account-loading').hidden=true;$('#account-content').hidden=!profile;
    if(!profile){if(!busy)location.replace('/login?return_to=/account');return;}
    $('#account-name').textContent=`Welcome, ${profile.name}.`;
    $('#account-initial').textContent=profile.initial;
    $('#account-email').textContent=profile.email;
    $('#account-provider').textContent=profile.provider;
  }
  if(page==='signed-out'){
    $('#logout-loading').hidden=true;$('#finish-signout').hidden=!profile;
    $('#logout-state').textContent=profile?'SESSION STILL ACTIVE':'SESSION CLOSED';
    $('#logout-title').textContent=profile?'One more step.':'See you next flight.';
    $('#logout-message').textContent=profile?'Your session is still active. Use the button below to sign out.':'You’re signed out of VoxelFlight. Saved public reconstructions are still available.';
  }
}
function providerFor(id){
  if(id==='google'){const provider=new GoogleAuthProvider();provider.setCustomParameters({prompt:'select_account'});return provider;}
  if(id==='github')return new GithubAuthProvider();
  throw Object.assign(new Error(),{code:'studio/not-configured'});
}
async function signin(id,redirect=false){
  if(busy)return;
  if(!ready||!settings.providers.includes(id)){message(authMessage({code:'studio/not-ready'}));return;}
  pendingProvider=id;message('');setBusy(true);
  showTransition(`Connecting to ${id==='google'?'Google':'GitHub'}`,'Your provider handles sign-in. We never see your password.');
  try{
    // No contacts, repository, or social-post permissions are requested.
    if(redirect){await signInWithRedirect(auth,providerFor(id));return;}
    await signInWithPopup(auth,providerFor(id));
    location.assign(returnTo);
  }catch(error){hideTransition();message(authMessage(error));console.warn('Sign-in status:',error.code||'unknown');$('#redirect-signin').hidden=false;setBusy(false);}
}
async function logout(){
  if(!auth||busy)return;
  setBusy(true);message('');showTransition('Until the next flight','Signing you out of VoxelFlight…');
  try{await signOut(auth);location.replace('/logout');}
  catch(error){hideTransition();message(authMessage(error));setBusy(false);}
}
document.querySelectorAll('[data-firebase-provider]').forEach(button=>button.addEventListener('click',()=>signin(button.dataset.firebaseProvider)));
$('#redirect-signin')?.addEventListener('click',()=>signin(pendingProvider||'google',true));
$('#signout')?.addEventListener('click',logout);
$('#finish-signout')?.addEventListener('click',logout);
$('#switch-account')?.addEventListener('click',logout);
// Browser history restoration must not show a previous visitor's account details.
window.addEventListener('pageshow',event=>{if(event.persisted)location.reload();});
async function start(){
  try{
    const response=await fetch('/auth-settings.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});
    if(!response.ok)throw Object.assign(new Error(),{code:'studio/not-configured'});
    settings=await response.json();
    if(!settings.projectId||!settings.providers?.length)throw Object.assign(new Error(),{code:'studio/not-configured'});
    // Reserved Hosting URL supplies public web config for the deployed project.
    // OAuth client secrets never belong in this file or in the browser bundle.
    const configResponse=await fetch('/__/firebase/init.json',{cache:'no-store',signal:AbortSignal.timeout(10000)});
    if(!configResponse.ok)throw Object.assign(new Error(),{code:'studio/not-configured'});
    const config=await configResponse.json();
    if(config.projectId!==settings.projectId||!config.apiKey||!config.appId)throw Object.assign(new Error(),{code:'studio/not-configured'});
    // Firebase's recommended same-origin auth helper avoids third-party-storage
    // dependence on Safari/mobile. Both Hosting callback URLs are registered.
    if([`${settings.projectId}.web.app`,`${settings.projectId}.firebaseapp.com`].includes(location.hostname))config.authDomain=location.hostname;
    auth=getAuth(initializeApp(config));auth.useDeviceLanguage();
    try{await setPersistence(auth,browserLocalPersistence);}catch{await setPersistence(auth,browserSessionPersistence);}
    onAuthStateChanged(auth,renderUser,error=>message(authMessage(error)));
    const result=await getRedirectResult(auth);
    if(result?.user){location.replace(returnTo);return;}
    ready=true;setBusy(false);
    document.querySelectorAll('[data-firebase-provider]').forEach(button=>{button.querySelector('.provider-state').textContent=settings.providers.includes(button.dataset.firebaseProvider)?'↗':'Not connected';});
  }catch(error){
    resolveInitial(null);
    ready=false;setBusy(false);hideTransition();message(authMessage(error));
    $('#account-loading')?.setAttribute('hidden','');$('#logout-loading')?.setAttribute('hidden','');
    document.querySelectorAll('.provider-state').forEach(label=>label.textContent='Not connected');
    if(page==='signed-out')$('#logout-state').textContent='STATUS UNAVAILABLE';
  }
}
start();
