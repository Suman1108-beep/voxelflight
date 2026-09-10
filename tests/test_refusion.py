import unittest
import numpy as np
from mac_refuse import rasterize_prediction, unproject, multiview_support


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


if __name__=='__main__':unittest.main()
