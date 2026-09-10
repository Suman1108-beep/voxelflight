"""Small deterministic geometry and loopback API tests; no GPU inference in fixtures."""
from pathlib import Path
import json
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from mac_geometry import depth_mesh,similarity,transform_points,voxel_fuse
from mac_server import create_app
from mac_reconstruct import calibration,export_gis


class GeometryTests(unittest.TestCase):
    def test_camera_similarity_constrains_straight_path_with_orientations(self):
        from mac_geometry import camera_similarity
        source=np.tile(np.eye(4),(8,1,1));source[:,0,3]=np.arange(8)
        rotation=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        target=source.copy();target[:,:3,:3]=rotation@source[:,:3,:3]
        target[:,:3,3]=transform_points(source[:,:3,3],2.5,rotation,[2,-4,1])
        scale,r,t,quality=camera_similarity(source,target)
        self.assertAlmostEqual(scale,2.5);np.testing.assert_allclose(r,rotation,atol=1e-7)
        np.testing.assert_allclose(t,[2,-4,1],atol=1e-7)
        self.assertLess(quality['camera_fit_rmse'],1e-8)
        source[:,:3,3]=0
        with self.assertRaises(ValueError):camera_similarity(source,target)

    def test_similarity_recovers_known_transform_with_outliers(self):
        rng=np.random.default_rng(0);source=rng.normal(size=(500,3))
        r=np.array([[0,-1,0],[1,0,0],[0,0,1]])
        target=transform_points(source,2.3,r,[7,-2,1]);target[:40]+=30
        s,rotation,t,metrics=similarity(source,target)
        self.assertAlmostEqual(s,2.3,places=6);np.testing.assert_allclose(rotation,r,atol=1e-6);np.testing.assert_allclose(t,[7,-2,1],atol=1e-6)
        self.assertGreater(metrics['fit_rmse'],1)  # Not falsely reporting the trimmed fit as all-point accuracy.

    def test_straight_path_rejected(self):
        points=np.array([[i,0,0] for i in range(10)])
        with self.assertRaises(ValueError):similarity(points,points)

    def test_mesh_does_not_bridge_depth_edges_or_missing_pixels(self):
        x,y=np.meshgrid(np.arange(4),np.arange(4));d=np.ones((4,4));d[:,2:]=10
        p=np.stack([x,y,d],-1);m=np.ones((4,4),bool);m[0,0]=False
        vertices,_,faces=depth_mesh(p,np.ones((4,4,3))*128,d,m,stride=1)
        self.assertGreater(len(faces),0)
        self.assertTrue(np.all(np.ptp(vertices[faces,2],axis=1)==0))
        self.assertFalse(np.any(np.all(vertices==[0,0,1],axis=1)))

    def test_voxels_average_only_observed_points(self):
        p,c=voxel_fuse([[0,0,0],[.01,0,0],[1,1,1]],[[100,0,0],[200,0,0],[0,255,0]],.1)
        self.assertEqual(len(p),2);self.assertEqual(c[0,0],150);self.assertAlmostEqual(p[0,0],.005)

    def test_calibration_rejects_fisheye_and_preserves_distortion(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'camera.json'
            path.write_text(json.dumps(dict(width=640,height=480,calibration=[500,500,320,240,.1,0,0,0])))
            w,h,k,d=calibration(path);self.assertEqual((w,h),(640,480));self.assertEqual(len(d),4)
            path.write_text(json.dumps(dict(model='fisheye')))
            with self.assertRaises(ValueError):calibration(path)

    def test_calibrated_image_sequence_scales_intrinsics(self):
        from types import SimpleNamespace
        from PIL import Image
        from mac_reconstruct import extract
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);images=root/'input';images.mkdir();out=root/'output';out.mkdir()
            for i in range(2):Image.new('RGB',(1600,900),(i*100,20,30)).save(images/f'{i}.jpg')
            camera=root/'camera.json';camera.write_text(json.dumps(dict(model='OPENCV',width=1600,height=900,fx=1000,fy=1000,cx=800,cy=450)))
            frames,video,info=extract(SimpleNamespace(video=None,images=images,calibration=camera,max_frames=2),out)
            self.assertIsNone(video);self.assertEqual(frames[0]['source_name'],'0.jpg')
            np.testing.assert_allclose(frames[0]['input_intrinsics'],[[800,0,640],[0,800,360],[0,0,1]])

    def test_visual_pose_priors_reject_reference_and_bad_rotations(self):
        from mac_reconstruct import load_visual_pose_priors
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'sfm.json'
            value=dict(ground_truth_used=False,metric_scale_established=False,
                poses=[dict(image='1.jpg',camera_to_world=np.eye(4).tolist())])
            path.write_text(json.dumps(value));frames=[dict(source_name='1.jpg')]
            info=load_visual_pose_priors(path,frames)
            self.assertFalse(info['metric_scale_established']);self.assertIn('input_camera_pose',frames[0])
            with self.assertRaises(ValueError):load_visual_pose_priors(path,[dict(source_name='missing.jpg')])
            value['ground_truth_used']=True;path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):load_visual_pose_priors(path,frames)
            value['ground_truth_used']=False;value['poses'][0]['camera_to_world'][0][0]=2
            path.write_text(json.dumps(value))
            with self.assertRaises(ValueError):load_visual_pose_priors(path,frames)

    def test_geospatial_exports_have_crs_and_unobserved_nodata(self):
        import laspy
        import rasterio
        with tempfile.TemporaryDirectory() as directory:
            points=np.array([[0,0,1],[.5,.5,2],[1,1,3]],np.float32)
            colors=np.array([[255,0,0],[0,255,0],[0,0,255]],np.uint8)
            names=export_gis(Path(directory),points,colors,dict(origin=np.array([465580,5248057,462]),epsg=32632))
            self.assertEqual(len(names),2)
            cloud=laspy.read(Path(directory)/'reconstruction_utm.las')
            self.assertEqual(cloud.header.parse_crs().to_epsg(),32632)
            np.testing.assert_allclose(cloud.x,[465580,465580.5,465581],atol=.001)
            with rasterio.open(Path(directory)/'surface_model.tif') as raster:
                self.assertEqual(raster.crs.to_epsg(),32632)
                self.assertTrue(np.any(raster.read(1)==raster.nodata))


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.addCleanup(self.temp.cleanup)
        # The API contract must be testable without downloading 5 GB of weights.
        self.readiness=patch('mac_server.processing_readiness',return_value=dict(available=True,cached=True,encoder_cached=True,ready=True))
        self.readiness.start();self.addCleanup(self.readiness.stop)
        root=Path(self.temp.name)
        static=root/'viewer/dist/client';static.mkdir(parents=True)
        (static/'engine.html').write_text('<!doctype html><title>API test fixture</title>')
        root_patch=patch('mac_server.ROOT',root)
        root_patch.start();self.addCleanup(root_patch.stop)
        # Deliberately a unit-test worker, confined to this temporary directory.
        script="import sys,pathlib,json,time; time.sleep(.35); p=pathlib.Path(sys.argv[sys.argv.index('--output')+1]); p.mkdir(); (p/'report.json').write_text(json.dumps({'unit_test_fixture':True}))"
        self.client=TestClient(create_app(self.temp.name,[sys.executable,'-c',script]),base_url='http://127.0.0.1')
        self.client.__enter__();self.addCleanup(self.client.__exit__,None,None,None)
        self.token=self.client.get('/api/health').json()['session']
        self.headers={'X-SIH3D-Session':self.token}

    def test_health_and_local_page(self):
        self.assertEqual(self.client.get('/engine.html').status_code,200)
        self.assertTrue(self.client.get('/api/health').json()['weights_cached'])

    def test_unavailable_model_rejects_processing(self):
        with patch('mac_server.processing_readiness',return_value=dict(available=False,cached=False,encoder_cached=False,ready=False)):
            self.assertFalse(self.client.get('/api/health').json()['ready'])
            response=self.client.post('/api/jobs',headers=self.headers,files={'video':('fixture.mp4',b'fixture')})
            self.assertEqual(response.status_code,503)

    def test_host_origin_and_csrf_guards(self):
        self.assertEqual(self.client.get('/engine.html',headers={'Sec-Fetch-Site':'cross-site'}).status_code,200)
        self.assertEqual(self.client.get('/api/health',headers={'Sec-Fetch-Site':'cross-site'}).status_code,403)
        self.assertEqual(self.client.get('/api/health',headers={'Host':'evil.test'}).status_code,403)
        self.assertEqual(self.client.get('/api/health',headers={'Origin':'https://evil.test'}).status_code,403)
        self.assertEqual(self.client.post('/api/jobs').status_code,403)
        self.assertEqual(self.client.post('/api/jobs',headers={**self.headers,'Origin':'https://evil.test'}).status_code,403)

    def test_bad_inputs_rejected_before_worker(self):
        for name,payload,settings in [('bad.exe',b'abc',{}),('empty.mp4',b'',{}),('a.mp4',b'123',{'max_frames':'99999'})]:
            response=self.client.post('/api/jobs',headers=self.headers,files={'video':(name,payload)},data=settings)
            self.assertEqual(response.status_code,400,response.text)

    def test_lifecycle_conflict_and_artifact_isolation(self):
        response=self.client.post('/api/jobs',headers=self.headers,files={'video':('../../sample.mp4',b'video-unit-test')},data={'max_frames':'4'})
        self.assertEqual(response.status_code,202,response.text);job=response.json();jid=job['id']
        self.assertEqual(job['name'],'sample.mp4')
        self.assertEqual(self.client.post('/api/jobs',headers=self.headers,files={'video':('other.mp4',b'abc')}).status_code,409)
        for _ in range(30):
            state=self.client.get(f'/api/jobs/{jid}').json()['state']
            if state=='complete':break
            time.sleep(.05)
        self.assertEqual(state,'complete')
        self.assertEqual(self.client.get(f'/api/jobs/{jid}/assets/report.json').json(),{'unit_test_fixture':True})
        self.assertEqual(self.client.get(f'/api/jobs/{jid}/assets/../input.mp4').status_code,404)
        self.assertEqual(self.client.get(f'/api/jobs/{jid}/assets/worker.log').status_code,404)
        self.assertEqual(self.client.get('/mac_server.py').status_code,404)

    def test_cancel_real_subprocess(self):
        job=self.client.post('/api/jobs',headers=self.headers,files={'video':('cancel.mp4',b'fixture')}).json()
        time.sleep(.03)
        self.assertEqual(self.client.post(f"/api/jobs/{job['id']}/cancel",headers=self.headers).status_code,200)
        for _ in range(30):
            state=self.client.get(f"/api/jobs/{job['id']}").json()['state']
            if state=='cancelled':break
            time.sleep(.05)
        self.assertEqual(state,'cancelled')


if __name__=='__main__':unittest.main()
