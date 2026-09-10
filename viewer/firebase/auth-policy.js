export const providerNames=Object.freeze({'google.com':'Google','github.com':'GitHub'});
export function displayProfile(user){
  if(!user?.uid)return null;
  const provider=user.providerData?.find(item=>providerNames[item.providerId]);
  const name=(user.displayName||provider?.displayName||user.email?.split('@')[0]||'Explorer').trim().slice(0,120)||'Explorer';
  return {name,initial:Array.from(name)[0].toUpperCase(),email:user.email||'Not shared by your provider',provider:providerNames[provider?.providerId]||'Firebase'};
}
export function safeReturnTo(value,origin){
  if(typeof value!=='string'||!value.startsWith('/')||value.startsWith('//')||/[\\\u0000-\u001f]/.test(value))return '/account';
  try{const url=new URL(value,origin);if(url.origin!==origin||!['/account','/workspace','/engine','/engine.html'].includes(url.pathname))return '/account';return url.pathname+url.hash;}catch{return '/account';}
}
export function authMessage(error){
  const messages={
    'auth/popup-closed-by-user':'Sign-in was cancelled. You can try again or continue as a guest.',
    'auth/cancelled-popup-request':'Another sign-in window is already open.',
    'auth/popup-blocked':'Your browser blocked the sign-in window. Allow pop-ups for this site, or use “Continue in this tab”.',
    'auth/account-exists-with-different-credential':'This email already has an account with another provider. Sign in with the provider you used before.',
    'auth/operation-not-allowed':'This sign-in provider has not been enabled yet. The public demo is still available.',
    'auth/unauthorized-domain':'Sign-in is not configured for this domain yet. The public demo is still available.',
    'auth/network-request-failed':'Sign-in could not reach Firebase. Check your connection and try again.',
    'auth/web-storage-unsupported':'Your browser is blocking session storage. Allow site storage to sign in.',
    'auth/too-many-requests':'Too many sign-in attempts. Please wait a little and try again.',
    'auth/configuration-not-found':'Account setup is not complete yet. The public demo is still available.',
    'studio/not-configured':'Sign-in is awaiting Firebase configuration. You can explore the saved flight without an account.',
    'studio/not-ready':'Sign-in is still connecting. Please try again in a moment.'
  };
  return messages[error?.code]||'Sign-in could not be completed. Please try again; the public demo remains available.';
}
