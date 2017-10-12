import inspect
import os
import tempfile
import unittest

import numpy as np

from spectroscopy.dataset import Dataset
from spectroscopy.util import vec2bearing

class FlySpecPluginTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        fin = os.path.join(self.data_dir, 'TOFP04', 'wind', '2017_06_14.txt')
        gf = d.read(fin,ftype='flyspecwind')
        vx = gf.vx[0]
        vy = gf.vy[0]
        dt = gf.datetime[0]
        v = np.sqrt(vx*vx + vy*vy)
        self.assertAlmostEqual(v, 10.88, 2)
        self.assertAlmostEqual(vec2bearing(vx,vy), 255, 6)
        self.assertEqual(dt, '2017-06-14T06:00:00')

def suite():
    return unittest.makeSuite(FlySpecPluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

