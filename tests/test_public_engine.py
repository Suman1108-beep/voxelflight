import sys
import tempfile
import time
import unittest
from unittest.mock import patch
from fastapi.testclient import TestClient
from mac_server import create_app

ORIGIN='https://voxelflight-3d.web.app'

def fixture_verifier(token):
    if token not in {'alice-fixture','bob-fixture'}:raise ValueError('invalid')
    return token.split('-')[0]

class PublicApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        readiness=patch('mac_server.processing_readiness',return_value=dict(available=True,cached=True,encoder_cached=True,ready=True))
        readiness.start();self.addCleanup(readiness.stop)
        script="import sys,pathlib,json,time; time.sleep(.3); p=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); p.mkdir(); (p/'report.json').write_text(json.dumps({'unit_test_fixture':True,'device':'private hardware','warnings':['Experimental Mac inference: accuracy not measured.']}))"
        app=create_app(self.temp.name,[sys.executable,'-c',script],public_project='voxelflight-3d',verify_token=fixture_verifier)
        self.client=TestClient(app);self.client.__enter__();self.addCleanup(self.client.__exit__,None,None,None)
        self.alice={'Authorization':'Bearer alice-fixture','Origin':ORIGIN};self.bob={'Authorization':'Bearer bob-fixture','Origin':ORIGIN}
    def submit(self,headers):return self.client.post('/api/jobs',headers=headers,files={'video':('fixture.mp4',b'unit-test-only')},data={'max_frames':'4','size':'252'})
    def test_no_source_or_anonymous_private_data(self):
        for path in ['/engine.html','/mac_server.py','/.env']:
            self.assertEqual(self.client.get(path).status_code,404)
        self.assertEqual(self.client.get('/api/jobs').status_code,401)
        self.assertEqual(self.client.get('/api/jobs',headers={'Authorization':'Bearer forged'}).status_code,401)
        health=self.client.get('/api/health').json();self.assertNotIn('session',health);self.assertNotIn('device',health)
    def test_cors_and_upload_bounds(self):
        self.assertEqual(self.client.get('/api/health',headers={'Origin':'https://evil.test'}).status_code,403)
        pre=self.client.options('/api/jobs',headers={'Origin':ORIGIN,'Access-Control-Request-Method':'POST','Access-Control-Request-Headers':'authorization,content-type'})
        self.assertEqual(pre.status_code,200);self.assertEqual(pre.headers['access-control-allow-origin'],ORIGIN)
        response=self.client.post('/api/jobs',headers={**self.alice,'Content-Length':str(100*1024**2)})
        self.assertEqual(response.status_code,413)
    def test_ownership_queue_and_private_results(self):
        first=self.submit(self.alice);self.assertEqual(first.status_code,202,first.text);a=first.json()['id'];self.assertNotIn('owner',first.json())
        self.assertEqual(self.submit(self.alice).status_code,409)
        second=self.submit(self.bob);self.assertEqual(second.status_code,202,second.text);b=second.json()['id']
        self.assertEqual(self.client.get(f'/api/jobs/{a}',headers=self.bob).status_code,404)
        self.assertEqual(self.client.post(f'/api/jobs/{a}/cancel',headers=self.bob).status_code,404)
        self.assertEqual(self.client.get(f'/api/jobs/{a}/assets/report.json',headers=self.bob).status_code,404)
        self.assertEqual([j['id'] for j in self.client.get('/api/jobs',headers=self.bob).json()],[b])
        self.assertEqual(self.client.post(f'/api/jobs/{b}/cancel',headers=self.bob).status_code,200)
        for _ in range(50):
            if self.client.get(f'/api/jobs/{a}',headers=self.alice).json()['state']=='complete':break
            time.sleep(.05)
        report=self.client.get(f'/api/jobs/{a}/assets/report.json',headers=self.alice)
        self.assertEqual(report.status_code,200);self.assertNotIn('device',report.json())
        self.assertTrue(report.json()['unit_test_fixture'])
        self.assertEqual(self.client.get(f'/api/jobs/{b}',headers=self.bob).json()['state'],'cancelled')

if __name__=='__main__':unittest.main()
