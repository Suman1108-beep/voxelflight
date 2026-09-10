// Restore the public demo assets separately from the source repository.
// The release manifest pins every byte; no credentials or private run data are fetched.
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,dirname} from 'node:path';
import {createHash} from 'node:crypto';
const root=resolve(import.meta.dirname,'..');
const release=JSON.parse(await readFile(resolve(root,'demo-assets.lock.json'),'utf8'));
const files=release.files.filter(f=>/^(assets|vendor)\//.test(f.path));
let cursor=0,completed=0;
async function worker(){while(cursor<files.length){const file=files[cursor++];
 if(file.path.includes('..')||file.path.includes('\\'))throw Error('Unsafe asset path');
 const dest=resolve(root,file.path);if(!dest.startsWith(root+'/'))throw Error('Asset escaped root');
 try{const cached=await readFile(dest);if(createHash('sha256').update(cached).digest('hex')===file.sha256){completed++;continue;}}catch{}
 const r=await fetch(new URL(file.path,release.origin),{signal:AbortSignal.timeout(120000)});
 if(!r.ok)throw Error(`Asset unavailable: ${file.path} (${r.status})`);
 const bytes=Buffer.from(await r.arrayBuffer());
 if(bytes.length!==file.bytes||createHash('sha256').update(bytes).digest('hex')!==file.sha256)throw Error(`Asset changed since this release: ${file.path}`);
 await mkdir(dirname(dest),{recursive:true});await writeFile(dest,bytes);completed++;
}}
await Promise.all(Array.from({length:4},worker));console.log(`Restored ${completed} verified public assets.`);
