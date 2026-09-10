import {defineConfig,loadEnv} from 'vite';
import {sites} from '@openai/sites-vite-plugin';
import {readFile} from 'node:fs/promises';
import {resolve} from 'node:path';
import {Readable} from 'node:stream';

const root=import.meta.dirname;
function workerDev(){return {name:'flight-lab-worker-dev',configureServer(server){
  // sites() runs before this adapter: it owns local SIWC and sanitizes identity headers.
  server.middlewares.use(async(req,res,next)=>{
    const pathname=new URL(req.url,'http://127.0.0.1').pathname;
    if(!['/','/account','/account.html','/workspace','/signed-out'].includes(pathname)&&!pathname.startsWith('/api/'))return next();
    try{
      const worker=await server.ssrLoadModule('/server/worker.js');
      const headers=new Headers();for(const [key,value] of Object.entries(req.headers)){if(value!==undefined)headers.set(key,Array.isArray(value)?value.join(', '):value);}
      const request=new Request(`http://${req.headers.host}${req.url}`,{method:req.method,headers,body:['GET','HEAD'].includes(req.method)?undefined:Readable.toWeb(req),duplex:'half'});
      const env={...loadEnv('development',root,''),LOCAL_DEV:true,ASSETS:{async fetch(input){const path=new URL(input.url).pathname;const name={'/login.html':'login.html','/_pages/account.html':'account.html','/workspace.html':'index.html','/logout.html':'logout.html'}[path];if(!name)return new Response('Not found',{status:404});return new Response(await server.transformIndexHtml(path,await readFile(resolve(root,name),'utf8')),{headers:{'Content-Type':'text/html; charset=utf-8'}});}}};
      const response=await worker.default.fetch(request,env);
      res.statusCode=response.status;response.headers.forEach((value,key)=>{if(key!=='set-cookie')res.setHeader(key,value);});
      const cookies=response.headers.getSetCookie();if(cookies.length)res.setHeader('set-cookie',cookies);
      if(!response.body||req.method==='HEAD')res.end();else Readable.fromWeb(response.body).pipe(res);
    }catch{res.statusCode=500;res.end('Local page could not be loaded');}
  });
}};}
export default defineConfig({plugins:[sites(),workerDev()],resolve:{alias:[{find:/^three$/,replacement:resolve(root,'vendor/three.module.js')},{find:'three/addons',replacement:resolve(root,'vendor/addons')}]},optimizeDeps:{noDiscovery:true},server:{host:'127.0.0.1',port:8132,strictPort:true},ssr:{noExternal:true},build:{ssr:'server/worker.js',outDir:'dist/server',emptyOutDir:true,minify:true,rollupOptions:{output:{entryFileNames:'index.js'}}}});
