import unittest
import numpy as np
from pathlib import Path
import argparse
import json
import tempfile
from unittest.mock import patch
from mac_refuse import rasterize_prediction, unproject, multiview_support, run


class RefusionTests(unittest.TestCase):
    def view(self, depth=5., shift=0.):
        pose=np.eye(4);pose[0,3]=shift
        return dict(depth=np.full((10,10),depth,np.float32), color=np.full((10,10,3),100,np.uint8),
                    intrinsics=np.array([[10.,0,4.5],[0,10.,4.5],[0,0,1.]]), pose=pose)

    def test_rasterization_preserves_supported_depth_and_holes(self):
        view=self.view();mask=np.ones((10,10),bool);mask[5,5]=False
        pred={**view,'points':unproject(view),'mask':mask}
        result=rasterize_prediction(pred)
        np.testing.assert_allclose(result['depth'][mask],5)
        self.assertEqual(result['depth'][5,5],0)

    def test_source_cannot_vote_for_itself(self):
        counts,_,_=multiview_support([self.view()])
        self.assertFalse(counts[0].any())

    def test_two_view_plane_is_supported(self):
        counts,_,_=multiview_support([self.view(),self.view(shift=.5)])
        self.assertGreater(np.count_nonzero(counts[0]),70)

    def test_inconsistent_depth_is_rejected(self):
        counts,_,_=multiview_support([self.view(),self.view(depth=8.)])
        self.assertFalse(counts[0].any())
        self.assertFalse(counts[1].any())

    def test_behind_camera_has_no_support(self):
        opposite=self.view();opposite['pose'][:3,:3]=np.diag([-1,1,-1])
        counts,_,_=multiview_support([self.view(),opposite])
        self.assertFalse(counts[0].any())

    def test_temporal_neighbor_radius_excludes_distant_view(self):
        views=[self.view(5.),self.view(8.),self.view(8.),self.view(8.),self.view(5.)]
        all_counts,_,_=multiview_support(views)
        local_counts,_,_=multiview_support(views,neighbor_radius=1)
        self.assertGreater(np.count_nonzero(all_counts[0]),0)
        self.assertFalse(local_counts[0].any())

    def test_invalid_neighbor_radius_is_rejected(self):
        with self.assertRaises(ValueError):
            multiview_support([self.view()],neighbor_radius=0)

    def test_zero_support_skips_expensive_crossview_check(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);source=root/'source';(source/'predictions').mkdir(parents=True)
            (source/'report.json').write_text(json.dumps(dict(
                ground_truth_used=False,model='test',device='cpu',runtime_s=0.,
                input={'input_type':'test'},window_size=2)))
            for i,shift in enumerate((0.,.1)):
                view=self.view(depth=5.,shift=shift)
                view['depth']=np.full((30,30),5.,np.float32)
                view['color']=np.full((30,30,3),100,np.uint8)
                view['intrinsics']=np.array([[30.,0,14.5],[0,30.,14.5],[0,0,1.]])
                np.savez(source/'predictions'/f'frame_{i:05d}.npz',
                         **view,mask=np.ones((30,30),bool),points=unproject(view))
            args=argparse.Namespace(input=source,output=root/'output',voxel=.1,
                min_support=0,relative_tolerance=.015,absolute_tolerance=.08,
                max_depth=80.,min_component_triangles=1,view_stride=1,
                neighbor_radius=1)
            with patch('mac_refuse.multiview_support',side_effect=AssertionError('must not be called')):
                report=run(args)
            self.assertTrue(report['support_validation_skipped'])
            self.assertIsNone(report['supported_fraction'])
            self.assertEqual(report['retained_fraction'],1.)


if __name__=='__main__':unittest.main()
