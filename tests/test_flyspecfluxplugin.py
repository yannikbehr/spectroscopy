import inspect
import os
import tempfile
import unittest

import numpy as np
from scipy.interpolate import interp1d

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.flyspecref import FlySpecRefPluginException


class FlySpecFluxTestCase(unittest.TestCase):
    """
    Test plugin to read flux estimates from GNS FlySpec UI.
    """
    
    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        fin = os.path.join(self.data_dir, 'TOFP04', 'TOFP04_2017_06_14.txt') 
        e = d.read(fin, ftype='flyspecflux')
        nlines = None
        with open(fin) as fd:
            nlines = len(fd.readlines())
        self.assertEqual(e['FluxBuffer'].value.shape, (nlines-1,))


def suite():
    return unittest.makeSuite(FlySpecFluxTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
