import {showTransition,hideTransition} from './pixel.js';
const $=selector=>document.querySelector(selector);
async function json(url,options={}){const response=await fetch(url,{...options,credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(12000)});if(!response.ok)throw new Error('The sign-in service is temporarily unavailable. You can still open the public demo.');return response.json();}
async function submitAuth(action,provider){
 const {csrfToken}=await json('/api/auth/csrf');if(!csrfToken)throw new Error('Could not verify this request. Reload and try again.');
 const form=document.createElement('form');form.method='POST';form.target='_top';form.action='/api/auth/'+action+(provider?'/'+provider:'');
 const callbackUrl=action==='signout'?'/signout-with-chatgpt?return_to=/signed-out':'/account';
 for(const [name,value] of Object.entries({csrfToken,callbackUrl})){const input=document.createElement('input');input.type='hidden';input.name=name;input.value=value;form.append(input);}document.body.append(form);form.submit();
}
function nativeSignout(){const link=document.createElement('a');link.href='/signout-with-chatgpt?return_to=/signed-out';link.target='_top';document.body.append(link);link.click();link.remove();}
async function signout(user){
 showTransition('Until the next flight','Ending your sign-in session…');
 try{const options=await json('/api/signin-options');if(options.externalAuth)await submitAuth('signout');else if(user.provider==='chatgpt')nativeSignout();else throw new Error('Sign-out is unavailable. Please reload and try again.');}
 catch(error){hideTransition();const output=$('#account-error')??$('#logout-error');output.textContent=error.message;}
}
async function signin(){
 const message=$('#auth-message');if(new URLSearchParams(location.search).has('error'))message.textContent='Sign-in was not completed. Please try again or use another provider.';
 try{const configuration=await json('/api/signin-options');document.querySelectorAll('[data-provider]').forEach(button=>{const enabled=configuration.providers.includes(button.dataset.provider);button.dataset.ready=String(enabled);button.disabled=!enabled;button.querySelector('.provider-state').textContent=enabled?'↗':'Not connected';});}
 catch(error){message.textContent=error.message;document.querySelectorAll('.provider-state').forEach(label=>label.textContent='Unavailable');}
 document.querySelectorAll('[data-provider]').forEach(button=>button.addEventListener('click',async()=>{button.disabled=true;message.textContent='';showTransition('Connecting to '+(button.dataset.provider==='google'?'Google':'Facebook'),'Your provider will securely handle sign-in.');try{await submitAuth('signin',button.dataset.provider);}catch(error){hideTransition();message.textContent=error.message;button.disabled=false;}}));
 window.addEventListener('pageshow',()=>document.querySelectorAll('[data-provider]').forEach(button=>button.disabled=button.dataset.ready!=='true'));
}
async function getAccount(){return fetch('/api/account',{credentials:'same-origin',cache:'no-store',signal:AbortSignal.timeout(12000)});}
async function account(){
 try{const response=await getAccount();if(response.status===401){location.replace('/?return_to=/account');return;}if(!response.ok)throw new Error('Account details could not be loaded. Please refresh.');
 const {user}=await response.json();$('#account-name').textContent=user.name||'Your account';$('#account-initial').textContent=(user.name||user.email||'S').slice(0,1).toUpperCase();$('#account-email').textContent=user.email||'Not shared by provider';$('#account-provider').textContent=user.provider==='chatgpt'?'ChatGPT':user.provider==='google'?'Google':'Facebook';$('#account-loading').hidden=true;$('#account-content').hidden=false;$('#signout').addEventListener('click',()=>signout(user));
 }catch(error){$('#account-loading').hidden=true;$('#account-error').textContent=error.message;}
}
async function signedOut(){
 try{const response=await getAccount();$('#logout-loading').hidden=true;
 if(response.status===401){$('#logout-title').textContent='See you next flight.';$('#logout-message').textContent='You’re signed out. Your local reconstructions stay right where you left them.';$('#logout-state').textContent='SESSION CLOSED';return;}
 if(!response.ok)throw new Error('Could not confirm sign-out. Please try again.');
 const {user}=await response.json();$('#logout-title').textContent='One more step.';$('#logout-message').textContent='Your session is still active. Finish signing out below.';$('#logout-state').textContent='SESSION STILL ACTIVE';$('#finish-signout').hidden=false;$('#finish-signout').addEventListener('click',()=>signout(user));
 }catch(error){$('#logout-loading').hidden=true;$('#logout-state').textContent='STATUS UNAVAILABLE';$('#logout-error').textContent=error.message;}
}
if(document.body.dataset.page==='signin')signin();
if(document.body.dataset.page==='account')account();
if(document.body.dataset.page==='signed-out')signedOut();
