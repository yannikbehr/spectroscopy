import inspect
import math
import os
import unittest

import numpy as np

from spectroscopy.dataset import Dataset
from spectroscopy.util import vec2bearing


class NZMetservicePluginTestCase(unittest.TestCase):
    """
    Test plugin to read wind data provided by NZ MetService.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_open(self):
        w = Dataset.open(os.path.join(
            self.data_dir, 'gns_wind_model_data_ecmwf_20160921_0630.txt'),
            format='NZMETSERVICE')
        lon, lat, hght, time, vx, vx_error, vy, vy_error, vz, vz_error =\
            w.get_velocity(174.735, -36.890, 1000, '2016-09-21T06:00:00+12:00')
        self.assertEqual(lon, 174.735)
        self.assertEqual(lat, -36.890)
        self.assertEqual(hght, 1000)
        v = math.sqrt(vx * vx + vy * vy)
        self.assertAlmostEqual(v / 0.514444, 17, 6)
        self.assertAlmostEqual(70., vec2bearing(vx, vy), 6)


def suite():
    return unittest.makeSuite(NZMetservicePluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
