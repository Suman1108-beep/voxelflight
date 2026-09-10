let provider=null;
export function registerSessionProvider(value){provider=value;}
export async function sessionHeaders(){
  if(!provider)throw new Error('Sign in to start a reconstruction.');
  return provider();
}
