import csv
from pathlib import Path
import tempfile
import unittest
import sys
import numpy as np
from mac_geometry import transform_points
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from georeference_mac_run import weighted_similarity,load_gps


class GpsTests(unittest.TestCase):
    def test_weighted_transform_recovers_known_georeference(self):
        rng=np.random.default_rng(23);source=rng.normal(size=(40,3))
        r=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        target=transform_points(source,2.7,r,[10,-4,2])
        scale,rotation,t,quality=weighted_similarity(source,target,np.ones(40)*3)
        self.assertAlmostEqual(scale,2.7);np.testing.assert_allclose(rotation,r,atol=1e-7)
        np.testing.assert_allclose(t,[10,-4,2],atol=1e-7)
        self.assertFalse(quality['independent_accuracy'])

    def test_invalid_uncertainty_and_stationary_path_rejected(self):
        xyz=np.zeros((8,3))
        with self.assertRaises(ValueError):weighted_similarity(xyz,xyz,np.zeros(8))
        with self.assertRaises(ValueError):weighted_similarity(xyz,xyz,np.ones(8))

    def test_telemetry_requires_exact_images_and_monotonic_time(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'telemetry.csv'
            columns=['image','timestamp_us','latitude','longitude','altitude_m','gps_fix_type','gps_eph_m']
            with path.open('w',newline='') as stream:
                writer=csv.writer(stream);writer.writerow(columns)
                writer.writerows([['1.jpg',1000,47.38,8.54,460,3,9],['2.jpg',2000,47.38001,8.54001,461,3,9]])
            frames=[{'source_name':'1.jpg'},{'source_name':'2.jpg'}]
            gps,_,times,epsg=load_gps(path,frames)
            self.assertEqual(epsg,32632);self.assertEqual(gps.shape,(2,3));self.assertEqual(times.tolist(),[1000,2000])
            with self.assertRaises(ValueError):load_gps(path,[{'source_name':'missing.jpg'}])
            with self.assertRaises(ValueError):load_gps(path,frames[::-1])
            with self.assertRaises(ValueError):load_gps(Path(directory)/'GroundTruthAGL.csv',frames)


if __name__=='__main__':unittest.main()
