import inspect
import math
import os
import tempfile
import unittest

import numpy as np

from spectroscopy.dataset import Dataset
from spectroscopy.util import vec2bearing, get_wind_speed 


class NZMetservicePluginTestCase(unittest.TestCase):
    """
    Test plugin to read wind data provided by NZ MetService.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        d.read(os.path.join(self.data_dir, 'gns_wind_model_data_ecmwf_20160921_0630.txt'),
              ftype='NZMETSERVICE')
        gf = d.elements['GasFlow'][0]
        lon, lat, hght, time, vx, vx_error, vy, vy_error, vz, vz_error = \
            get_wind_speed(gf, 174.735, -36.890, 1000, '2016-09-21T06:00:00+12:00')
        self.assertEqual(lon, 174.735)
        self.assertEqual(lat, -36.890)
        self.assertEqual(hght, 1000)
        v = math.sqrt(vx * vx + vy * vy)
        self.assertAlmostEqual(v / 0.514444, 17, 6)
        self.assertAlmostEqual(70., vec2bearing(vx, vy), 6)
        m = gf.methods[0]
        self.assertEqual(m.name[:][0],'gfs')
        d.read(os.path.join(self.data_dir, 'gns_wind_model_data_ecmwf_20160921_0630.txt'),
               ftype='NZMETSERVICE', preferred_model='ecmwf')
        gf1 = d.elements['GasFlow'][1]
        lon, lat, hght, time, vx, vx_error, vy, vy_error, vz, vz_error = \
            get_wind_speed(gf1, 174.755, -36.990, 1000, '2016-09-21T06:00:00+12:00')
        v = math.sqrt(vx * vx + vy * vy)
        self.assertEqual(lon, 174.735)
        self.assertEqual(lat, -36.890)
        self.assertEqual(hght, 1000)
        v = math.sqrt(vx * vx + vy * vy)
        self.assertAlmostEqual(v / 0.514444, 19, 6)
        self.assertAlmostEqual(65., vec2bearing(vx, vy), 6)
        self.assertEqual(gf1.methods[0].name[:][0],'ecmwf')
        self.assertEqual(gf1.unit[:][0], 'm/s')

def suite():
    return unittest.makeSuite(NZMetservicePluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
