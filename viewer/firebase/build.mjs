import {readFile,writeFile,cp,rm,readdir} from 'node:fs/promises';
import {resolve} from 'node:path';
import {build} from 'vite';

export async function prepareFirebase(root,dist){
  const settings=JSON.parse(await readFile(resolve(root,'firebase/settings.json'),'utf8'));
  if(!Array.isArray(settings.providers)||settings.providers.some(id=>!['google','github'].includes(id)))throw new Error('Unsupported Firebase sign-in provider');
  await writeFile(resolve(dist,'auth-settings.json'),JSON.stringify(settings,null,2));
  await build({configFile:false,root,logLevel:'warn',build:{outDir:resolve(dist,'firebase-auth'),emptyOutDir:true,minify:true,sourcemap:false,rollupOptions:{input:resolve(root,'firebase/auth.js'),external:['/pixel.js','/session-bridge.js'],output:{entryFileNames:'session-[hash].js',chunkFileNames:'shared-[hash].js'}}}});
  const entry=(await readdir(resolve(dist,'firebase-auth'))).find(name=>/^session-.*\.js$/.test(name));
  if(!entry)throw new Error('Firebase authentication bundle was not emitted');
  const scripts=`<meta name="hosting-platform" content="firebase"><link rel="stylesheet" href="/firebase-account.css"><script type="module" src="/firebase-auth/${entry}"></script>`;
  await cp(resolve(root,'firebase/account.css'),resolve(dist,'firebase-account.css'));
  const methods=`<div id="returning-account" class="returning-account" hidden><p id="returning-name"></p><a class="provider primary" id="returning-link" href="/workspace">Continue to your studio <span aria-hidden="true">→</span></a><button class="signout" id="switch-account">Sign out / switch account</button></div>
      <div id="signin-methods"><div class="signin-divider"><span>Sign in to your studio</span></div>
      <div class="provider-buttons">
        <button class="provider primary" data-firebase-provider="google" disabled><span class="provider-icon google" aria-hidden="true">G</span><span>Continue with Google</span><span class="provider-state">Connecting</span></button>${settings.providers.includes('github')?`
        <button class="provider" data-firebase-provider="github" disabled><span class="provider-icon" aria-hidden="true"><svg viewBox="0 0 24 24" width="24" height="24" fill="currentColor"><path d="M12 .9a11.1 11.1 0 0 0-3.5 21.63c.56.1.76-.24.76-.54v-2.1c-3.1.68-3.76-1.31-3.76-1.31-.5-1.29-1.24-1.63-1.24-1.63-1.01-.7.08-.68.08-.68 1.12.08 1.7 1.15 1.7 1.15.99 1.7 2.6 1.21 3.24.92.1-.72.38-1.21.7-1.49-2.47-.28-5.07-1.24-5.07-5.49 0-1.21.43-2.2 1.15-2.98-.12-.28-.5-1.41.1-2.94 0 0 .94-.3 3.05 1.14A10.6 10.6 0 0 1 12 6.2c.94 0 1.88.13 2.76.37 2.12-1.44 3.05-1.14 3.05-1.14.61 1.53.23 2.66.11 2.94.72.78 1.15 1.77 1.15 2.98 0 4.27-2.6 5.2-5.08 5.48.4.35.75 1.03.75 2.07v3.08c0 .3.2.65.77.54A11.1 11.1 0 0 0 12 .9Z"/></svg></span><span>Continue with GitHub</span><span class="provider-state">Connecting</span></button>`:''}
      </div></div>`;
  for(const [destination,source] of [['index.html','login.html'],['login.html','login.html'],['account.html','account.html'],['logout.html','logout.html'],['workspace.html','index.html'],['engine.html','engine.html']]){
    let html=await readFile(resolve(root,source),'utf8');
    html=html.replace(/<script type="module" src="\/entrance\.js"><\/script>/g,'');
    html=html.replace('</head>',scripts+'</head>');
    html=html.replace(/href="(?:\/)?engine\.html"/g,'href="/engine"');
    html=html.replace(/<a href="\/account"([^>]*)>Account<\/a>/g,'<a href="/account"$1 data-account-link>Account</a>');
    html=html.replaceAll('Processing workstation required','Authenticated processing queue')
      .replaceAll('on the connected processing workstation','through the processing service')
      .replaceAll('Worker connection required to reconstruct','Sign-in required for processing')
      .replaceAll('The local preview workflow','The current preview workflow')
      .replaceAll('Local processing workflow available; independent accuracy testing remains','Processing workflow available; independent accuracy testing remains')
      .replaceAll('Local preview','Preview');
    if(source==='login.html'){
      html=html.replace(/<div class="signin-divider">[\s\S]*?(?=\s*<p class="auth-message")/,methods);
      html=html.replace('<p class="auth-message"','<button class="signout" id="redirect-signin" disabled>Continue in this tab</button><p class="auth-message"');
      html=html.replace('Session cookies keep you signed in.','Firebase Authentication stores your sign-in session on this browser. Signing out clears this site’s session.');
      html=html.replace('Reconstruction files stay on the processing workstation; signing in does not upload videos or start a reconstruction.','Reconstruction files are stored by the operator-managed processing service and restricted to your account. Signing in does not upload videos or start a reconstruction.');
    }
    if(source==='account.html'){
      html=html.replace('WELCOME TO YOUR STUDIO','YOUR VOXELFLIGHT ACCOUNT');
      html=html.replace('<h1 id="account-name">Your account</h1>','<h1 id="account-name">Welcome.</h1>');
      html=html.replace('New video processing runs on the configured workstation. Local runs are not synced to this account.','Your submitted reconstructions are private to your account. Open the processing workspace to create a run or revisit your results.');
      html=html.replace('id="account-error" role="alert"></p>','id="account-error" role="alert"></p><a class="account-recovery" href="/login">Back to sign-in</a>');
    }
    if(source==='engine.html'){
      html=html.replace(/<details><summary>Already have the engine installed\?[\s\S]*?<\/details>/,'');
      html=html.replace('Explore now. Process when connected.','The processing service is unavailable.');
      html=html.replace('Saved models work for everyone. New videos cannot be processed on this public site yet; a cloud engine is not connected.','Saved models remain available. New submissions resume when the processing service is online.');
      html=html.replace('up to 1 GiB','up to 60 MiB').replace('<option value="96">96 · longer run</option>','');
      html=html.replace('Nothing is uploaded until a connected engine accepts your run.','Uploads start only when you submit. Videos and results are stored on the operator-managed processing service, visible only to your account. Demo limit: 3 runs per account per day.');
      html=html.replace('Recent runs <span id="history-count">This device</span>','Your reconstructions <span id="history-count">Your account</span>');
      html=html.replace('Local runs appear here when the engine is connected.','Sign in to see your submitted reconstructions.');
      html=html.replace('Local processing · no scene-specific training','Processing service · no scene-specific training');
      html=html.replace('<p id="engine-feedback"','<a class="button quiet wide" id="engine-signin" href="/login?return_to=/engine">Sign in to reconstruct →</a><p id="engine-feedback"');
    }
    if(/signin-with-chatgpt|data-provider="facebook"|src="\/entrance\.js"/.test(html))throw new Error('Legacy authentication link remains in '+destination);
    await writeFile(resolve(dist,destination),html);
  }
  // Runtime assets are hardware-neutral; provenance and accuracy warnings stay intact.
  const samplePath=resolve(dist,'assets/mac-proof/report.json');
  const sample=JSON.parse(await readFile(samplePath,'utf8'));
  delete sample.device;delete sample.gpu_allocation_at_end_gib;
  sample.warnings=sample.warnings.map(w=>w.replace('Experimental Mac inference','Experimental reconstruction').replace('this Mac mode','this processing mode'));
  await writeFile(samplePath,JSON.stringify(sample,null,2));
  // The Firebase account shell contains no user data. The SDK only renders the
  // current user's authenticated profile. There is no public account-data API.
  await rm(resolve(dist,'_pages'),{recursive:true,force:true});
  await rm(resolve(dist,'entrance.js'),{force:true});
  await writeFile(resolve(dist,'404.html'),'<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>VoxelFlight — Page not found</title><link rel="stylesheet" href="/entrance.css"></head><body class="entrance"><main class="signin-panel"><span class="eyebrow">VOXELFLIGHT</span><h1>This route is off the flight path.</h1><p>The page was not found. Your workspace is still here.</p><a class="provider primary" href="/workspace">Open the workspace →</a><a class="guest-button" href="/">Return to the entrance</a></main></body></html>');
}
