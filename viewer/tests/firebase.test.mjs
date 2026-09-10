import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {displayProfile,safeReturnTo,authMessage} from '../firebase/auth-policy.js';

test('welcome uses the signed-in visitor, never the site owner',()=>{
  assert.equal(displayProfile(null),null);
  assert.equal(displayProfile({displayName:'Unverified name'}),null);
  const person=displayProfile({uid:'visitor',displayName:'Ada Lovelace',email:'ada@example.test',providerData:[{providerId:'github.com'}]});
  assert.deepEqual(person,{name:'Ada Lovelace',initial:'A',email:'ada@example.test',provider:'GitHub'});
  assert.equal(displayProfile({uid:'visitor',email:'grace@example.test'}).name,'grace');
});
test('sign-in return paths are restricted to studio routes',()=>{
  for(const value of ['https://evil.test','//evil.test','/\\evil.test','javascript:alert(1)','/logout','/api/admin','/%2f%2fevil.test'])assert.equal(safeReturnTo(value,'https://app.web.app'),'/account');
  assert.equal(safeReturnTo('/workspace#validation','https://app.web.app'),'/workspace#validation');
});
test('provider errors do not reveal tokens or raw sensitive messages',()=>{
  assert.ok(!authMessage({message:'private-token'}).includes('private-token'));
  assert.match(authMessage({code:'auth/popup-blocked'}),/Continue in this tab/);
  assert.match(authMessage({code:'auth/account-exists-with-different-credential'}),/used before/);
});
test('Firebase hosts only the dedicated public output, with no catch-all API rewrite',async()=>{
  const {hosting}=JSON.parse(await readFile(new URL('../firebase.json',import.meta.url),'utf8'));
  assert.equal(hosting.public,'dist-firebase');
  assert.equal(hosting.rewrites,undefined);
  assert.equal(hosting.cleanUrls,true);
  assert.ok(hosting.headers.some(rule=>rule.headers.some(h=>h.key==='Cross-Origin-Opener-Policy'&&h.value==='same-origin-allow-popups')));
});
