import inspect
import os
import tempfile
import unittest

import numpy as np
from scipy.interpolate import interp1d

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.flyspecref import FlySpecRefPluginException


class FlySpecRefTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """
    
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        x = [521,637,692,818]
        y = [305.,315.,319.5,330.]
        f = interp1d(x, y, fill_value='extrapolate')
        xnew = range(0,2048)
        wavelengths = f(xnew)
        
        with self.assertRaises(FlySpecRefPluginException):    
            e = d.read(os.path.join(self.data_dir, 'TOFP04', 'Cal_20170602_0956_dark.bin'),
                        ftype='FLYSPECREF', wavelengths=wavelengths)
        
        e = d.read(os.path.join(self.data_dir, 'TOFP04', 'Cal_20170602_0956_dark.bin'),
                        ftype='FLYSPECREF', type='dark', wavelengths=wavelengths)
        self.assertEqual(e['RawDataBuffer'].d_var.shape, (10,2048))


def suite():
    return unittest.makeSuite(FlySpecRefTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
