"""Firebase bearer verification. No service-account key or admin token is needed."""
import threading
import time
import cachecontrol
import requests
from google.auth.transport.requests import Request
from google.oauth2.id_token import verify_firebase_token


class FirebaseVerifier:
    def __init__(self, project):
        self.project=project
        self.request=Request(session=cachecontrol.CacheControl(requests.Session()))
        self.lock=threading.Lock()

    def __call__(self, token):
        if not token or len(token)>8192:raise ValueError('Invalid token')
        # Reuse Google's cached signing certificates, never an unverified token payload.
        with self.lock:
            claims=verify_firebase_token(token,self.request,audience=self.project)
        uid=claims.get('sub')
        if claims.get('iss')!=f'https://securetoken.google.com/{self.project}':raise ValueError('Invalid issuer')
        if not isinstance(uid,str) or not 1<=len(uid)<=128:raise ValueError('Invalid subject')
        if claims.get('auth_time',time.time()+1)>time.time():raise ValueError('Invalid authentication time')
        if claims.get('firebase',{}).get('sign_in_provider') not in {'google.com','github.com'}:raise ValueError('Unsupported provider')
        return uid
