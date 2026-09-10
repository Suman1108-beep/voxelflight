import {Auth} from '@auth/core';
import Google from '@auth/core/providers/google';
import Facebook from '@auth/core/providers/facebook';

export const PUBLIC_ORIGIN='https://sih3d-flight-lab.diva-bhattacharya-ug.chatgpt.site';
const SESSION_SECONDS=60*60*24;
const noStore={'Cache-Control':'private, no-store','Vary':'Cookie','X-Content-Type-Options':'nosniff'};
const json=(body,status=200)=>Response.json(body,{status,headers:noStore});

export function safeRedirect(value,origin){
  if(typeof value!=='string'||value.includes('\\')||/[\u0000-\u001f]/.test(value))return origin+'/account';
  try{const url=new URL(value,origin);return url.origin===origin&&!value.startsWith('//')?url.href:origin+'/account';}catch{return origin+'/account';}
}
export function providerIds(env){
  if(!env.AUTH_SECRET)return [];
  return [...(env.GOOGLE_CLIENT_ID&&env.GOOGLE_CLIENT_SECRET?['google']:[]),...(env.FACEBOOK_APP_ID&&env.FACEBOOK_APP_SECRET?['facebook']:[])];
}
export function config(env,origin){
  const ids=providerIds(env);
  return {
    secret:env.AUTH_SECRET,basePath:'/api/auth',trustHost:true,useSecureCookies:origin.startsWith('https:'),
    providers:[
      ...(ids.includes('google')?[Google({clientId:env.GOOGLE_CLIENT_ID,clientSecret:env.GOOGLE_CLIENT_SECRET,checks:['pkce','state'],authorization:{params:{scope:'openid email profile'}}})]:[]),
      ...(ids.includes('facebook')?[Facebook({clientId:env.FACEBOOK_APP_ID,clientSecret:env.FACEBOOK_APP_SECRET,checks:['state'],authorization:{url:'https://www.facebook.com/dialog/oauth',params:{scope:'email'}}})]:[])
    ],
    session:{strategy:'jwt',maxAge:SESSION_SECONDS},
    pages:{signIn:'/',error:'/'},
    callbacks:{
      redirect:({url})=>safeRedirect(url,origin),
      jwt:({token,account})=>{
        if(account){token.provider=account.provider;token.sub=`${account.provider}:${account.providerAccountId}`;}
        // Provider access/refresh tokens are not kept in our session.
        return {sub:token.sub,name:token.name,email:token.email,provider:token.provider};
      },
      session:({session,token})=>({expires:session.expires,user:{id:token.sub,name:token.name??null,email:token.email??null,provider:token.provider}})
    },
    // Never log OAuth callback contents, provider payloads, tokens or personal data.
    logger:{error:()=>console.error('Authentication request failed'),warn:()=>{},debug:()=>{}}
  };
}
function trustedRequest(request,origin,path){
  const incoming=new URL(request.url),url=new URL(path??incoming.pathname+incoming.search,origin);
  const headers=new Headers(request.headers);
  headers.delete('forwarded');headers.delete('x-forwarded-host');headers.delete('x-forwarded-proto');
  headers.set('host',url.host);
  return new Request(url,{method:path?'GET':request.method,headers,body:path||['GET','HEAD'].includes(request.method)?undefined:request.body,duplex:'half'});
}
export function platformUser(request){
  // Sites dispatch supplies these identity headers; local dev strips client-supplied values.
  const id=request.headers.get('oai-authenticated-user-id'),email=request.headers.get('oai-authenticated-user-email');
  if(!id||!email)return null;
  let name=request.headers.get('oai-authenticated-user-full-name')||null;
  if(name&&request.headers.get('oai-authenticated-user-full-name-encoding')==='percent-encoded-utf-8'){try{name=decodeURIComponent(name);}catch{name=null;}}
  return {id:`chatgpt:${id}`,email,name,provider:'chatgpt'};
}
async function identity(request,env,origin){
  // Prefer the platform identity so a single sign-out path has unambiguous meaning.
  const native=platformUser(request);if(native)return {user:native,cookies:[]};
  if(!env.AUTH_SECRET)return {user:null,cookies:[]};
  const response=await Auth(trustedRequest(request,origin,'/api/auth/session'),config(env,origin));
  const session=await response.json();
  return {user:session?.user?.id?session.user:null,cookies:response.headers.getSetCookie()};
}
async function asset(request,env,path){
  if(!env.ASSETS?.fetch)return new Response('Static assets unavailable',{status:503});
  const url=new URL(request.url);if(path){url.pathname=path;url.search='';}
  return env.ASSETS.fetch(new Request(url,{method:request.method==='HEAD'?'HEAD':'GET',headers:request.headers}));
}
function secure(response,privatePage=false){
  const headers=new Headers(response.headers);
  headers.set('X-Content-Type-Options','nosniff');headers.set('Referrer-Policy','strict-origin-when-cross-origin');
  headers.set('Permissions-Policy','camera=(), microphone=(), geolocation=()');
  // Inline import maps are required by the existing offline Three.js viewer. No inline event handlers are used.
  headers.set('Content-Security-Policy',"default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; font-src 'self'; connect-src 'self' blob:; worker-src 'self' blob:; object-src 'none'; base-uri 'none'; form-action 'self' https://accounts.google.com https://www.facebook.com; frame-ancestors 'self' https://chatgpt.com https://*.chatgpt.com");
  if(privatePage)for(const [name,value] of Object.entries(noStore))headers.set(name,value);
  return new Response(response.body,{status:response.status,statusText:response.statusText,headers});
}
export default {
  async fetch(request,env){
    const url=new URL(request.url);
    const local=env.LOCAL_DEV===true&&url.protocol==='http:'&&['127.0.0.1','localhost'].includes(url.hostname);
    if(!local&&url.origin!==PUBLIC_ORIGIN)return json({error:'Unrecognized site origin'},400);
    const origin=local?url.origin:PUBLIC_ORIGIN;
    try{
      if(url.pathname==='/api/signin-options')return json({providers:providerIds(env),chatgpt:true,externalAuth:!!env.AUTH_SECRET});
      if(url.pathname.startsWith('/api/auth/')){
        if(!env.AUTH_SECRET)return json({error:'External sign-in is not configured'},503);
        if(!['GET','POST'].includes(request.method))return json({error:'Method not allowed'},405);
        const requestOrigin=request.headers.get('origin');
        if(request.method==='POST'&&requestOrigin!==origin)return json({error:'Cross-site request rejected'},403);
        if(Number(request.headers.get('content-length')??0)>16384)return json({error:'Request too large'},413);
        if(request.method==='POST'){
          const reader=request.body?.getReader();const chunks=[];let bytes=0;
          if(reader)while(true){const {done,value}=await reader.read();if(done)break;bytes+=value.byteLength;if(bytes>16384){await reader.cancel();return json({error:'Request too large'},413);}chunks.push(value);}
          const body=new Uint8Array(bytes);let offset=0;for(const chunk of chunks){body.set(chunk,offset);offset+=chunk.byteLength;}
          request=new Request(request.url,{method:'POST',headers:request.headers,body});
        }
        return secure(await Auth(trustedRequest(request,origin),config(env,origin)),true);
      }
      if(url.pathname==='/api/account'||url.pathname==='/account'||url.pathname==='/account.html'){
        if(!['GET','HEAD'].includes(request.method))return json({error:'Method not allowed'},405);
        const {user,cookies}=await identity(request,env,origin);
        if(!user)return url.pathname==='/api/account'?json({error:'Sign in required'},401):new Response(null,{status:303,headers:{...noStore,Location:'/?return_to=/account'}});
        const response=url.pathname==='/api/account'?json({user}):secure(await asset(request,env,'/_pages/account.html'),true);
        for(const cookie of cookies)response.headers.append('Set-Cookie',cookie);
        return response;
      }
      if(url.pathname.startsWith('/api/'))return json({error:'This endpoint is not available on the hosted viewer. Reconstruction runs on the local Mac engine.'},404);
      if(!['GET','HEAD'].includes(request.method))return json({error:'Method not allowed'},405);
      // Only bundled public assets are reachable. Source, credentials, and models on the Mac are never served here.
      if(url.pathname.split('/').some(part=>part.startsWith('.'))||url.pathname.startsWith('/server/'))return new Response('Not found',{status:404});
      return secure(await asset(request,env,url.pathname==='/'?'/login.html':url.pathname==='/workspace'?'/workspace.html':url.pathname==='/signed-out'?'/logout.html':undefined),['/','/login.html','/signed-out','/logout.html'].includes(url.pathname));
    }catch{
      console.error('Site request failed');return json({error:'This request could not be completed. Please try again.'},500);
    }
  }
};
