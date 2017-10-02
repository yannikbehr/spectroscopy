import inspect
import os
import tempfile
import unittest

import numpy as np
from scipy.stats import binned_statistic

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.flyspec import FlySpecPlugin
from spectroscopy.plugins.flyspec import FlySpecPluginException


class FlySpecPluginTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_add(self):
        d1 = Dataset(tempfile.mktemp(), 'w')
        d1.read(os.path.join(self.data_dir, '2016_06_11_0830_TOFP04.txt'),
                          ftype='FLYSPEC', timeshift=12.0)
        d2 = Dataset(tempfile.mktemp(), 'w')
        d2.read(os.path.join(self.data_dir, '2016_06_11_0900_TOFP04.txt'),
                          ftype='FLYSPEC', timeshift=12.0)
        d1 += d2
        self.assertEqual(len(d1.elements['Concentration']), 2)
        self.assertEqual(len(d1.elements['RawData']), 2)


    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        #d.read(os.path.join(self.data_dir,'2016_06_11_0830_TOFP04.txt'),ftype='FLYSPEC')
        d.read(os.path.join(self.data_dir,'2012_02_29_1340_CHILE.txt'),
               ftype='FLYSPEC')
        r = d.elements['RawData'][0]
        self.assertEqual(sum([x.size for x in r.datetime]), 4600)
        self.assertEqual(r.inc_angle[0][0], 135.140)
        c = d.elements['Concentration'][0]
        r1 = c.rawdata
        angle = r1.inc_angle[c.rawdata_index[3]]
        id_max = np.argmax(c.value[3])
        np.testing.assert_almost_equal(angle[id_max], 168.04, 2)
        self.assertEqual(len(c.value[:]), 36)
        np.testing.assert_array_almost_equal(r1.position[0][0],
                                             [-67.8047, -23.3565, 3927.], 2)

        # dicretize all retrievals onto a grid to show a daily plot
        bins = np.arange(0, 180, 1.0)
        nretrieval = len(c.value[:])
        m = np.zeros((nretrieval, bins.size - 1))
        for i, _so2 in enumerate(c.value[:]):
            _angle = r1.inc_angle[c.rawdata_index[i]]
            _so2_binned = binned_statistic(_angle, _so2, 'mean', bins)
            m[i, :] = _so2_binned.statistic
        ids = np.argmax(np.ma.masked_invalid(m), axis=1)
        maxima = np.array([166., 167., 167., 167., 168., 167., 168., 167.,
                           167., 167., 167., 167., 168., 167., 167., 167.,
                           167., 166., 167., 166., 166., 167., 165., 165.,
                           165., 164., 165., 163., 163., 164., 163., 165.,
                           164., 164., 164., 161.])
        np.testing.assert_array_almost_equal(maxima, bins[ids], 2)

        d1 = Dataset(tempfile.mktemp(), 'w')
        d1.read(os.path.join(self.data_dir, '2016_06_11_0830_TOFP04.txt'),
                          ftype='FLYSPEC', timeshift=12.0)
        c = d1.elements['Concentration'][0]
        nretrieval = len(c.value[:])
        r = c.rawdata
        m = np.zeros((nretrieval, bins.size - 1))
        for i, _so2 in enumerate(c.value[:]):
            _angle = r.inc_angle[c.rawdata_index[i]]
            _so2_binned = binned_statistic(_angle, _so2, 'mean', bins)
            m[i, :] = _so2_binned.statistic
        ids = np.argmax(np.ma.masked_invalid(m), axis=1)
        maxima = np.array([147., 25., 27., 86., 29., 31., 27., 27., 28., 137.,
                           34., 34.])
        np.testing.assert_array_almost_equal(maxima, bins[ids], 2)

    def test_not_enough_data(self):
        with self.assertRaises(FlySpecPluginException):
            d1 = Dataset(tempfile.mktemp(), 'w')
            d1 = d1.read(os.path.join(self.data_dir, '2015_05_03_1630_TOFP04.txt'),
                              ftype='FLYSPEC', timeshift=12.0)

    def test_split_by_scan(self):
        f = FlySpecPlugin()
        angles = np.array([30, 35, 40, 35, 30, 35, 40])
        result = [np.array([30, 35, 40]), np.array([30, 35]),
                  np.array([35, 40])]
        for i, a in enumerate(f._split_by_scan(angles)):
            np.testing.assert_array_equal(a[0], result[i])

        result1 = [np.array([1, 2, 3]), np.array([5, 4]), np.array([6, 7])]
        for i, a in enumerate(f._split_by_scan(angles, np.array([1, 2, 3, 4, 5, 6, 7]))):
            np.testing.assert_array_equal(a[1], result1[i])

        angles1 = np.array([30, 30, 35, 40, 35, 30, 35, 40, 40])
        result2 = [np.array([30, 30, 35, 40]), np.array([30, 35]),
                   np.array([35, 40, 40])]
        for i, a in enumerate(f._split_by_scan(angles1)):
            np.testing.assert_array_equal(a[0], result2[i])

        angles2 = np.array([30, 35, 40, 45, 30, 35, 40, 45])
        result3 = [np.array([30, 35, 40, 45]),
                   np.array([30, 35, 40, 45])]
        for i, a in enumerate(f._split_by_scan(angles2)):
            np.testing.assert_array_equal(a[0], result3[i])

        angles3 = np.array([30., 35., 40., 40., 45., 30., 35., 40., 45.])
        result4 = [np.array([30, 35, 40, 40, 45]),
                   np.array([30, 35, 40, 45])]
        for i, a in enumerate(f._split_by_scan(angles3)):
            np.testing.assert_array_equal(a[0], result4[i])

        angles4 = np.array([30, 35, 40, 40, 40, 45, 30, 35, 40, 45])
        with self.assertRaises(ValueError):
            [a for a in f._split_by_scan(angles4)]

        angles5 = np.array([174.750, 174.750, 174.420, 174.090, 173.750,
                            173.420, 173.080, 172.750, 172.420, 172.080,
                            171.750, 171.750, 171.410, 171.080, 170.740])
        result5 = [angles5[::-1]]
        for i, a in enumerate(f._split_by_scan(angles5)):
            np.testing.assert_array_equal(a[0], result5[i])

    def test_array_multi_sort(self):
        f = FlySpecPlugin()
        x1 = np.array([4., 5., 1., 2.])
        x2 = np.array([10., 11., 12., 13.])
        result = (np.array([1.,  2.,  4.,  5.]),
                  np.array([12.,  13.,  10.,  11.]))
        out = f._array_multi_sort(*tuple([x1, x2]))
        np.testing.assert_array_equal(out[0], result[0])
        np.testing.assert_array_equal(out[1], result[1])

    def test_plot(self):

        import matplotlib.image

        d = Dataset(tempfile.mktemp(), 'w')
        d.read(os.path.join(self.data_dir, '2012_02_29_1340_CHILE.txt'),
               ftype='FLYSPEC', timeshift=12.0)
        with tempfile.TemporaryFile() as fd:
            d.plot(savefig=fd, timeshift=12.0)
            expected_image = matplotlib.image.imread(
                os.path.join(self.data_dir, 'chile_retrievals_overview.png'),
                format='png')
            fd.seek(0)
            actual_image = matplotlib.image.imread(fd, format='png')

            # Set the "color" of fully transparent pixels to white. This avoids
            # the issue of different "colors" for transparent pixels.
            expected_image[expected_image[..., 3] <= 0.0035] = \
                [1.0, 1.0, 1.0, 0.0]
            actual_image[actual_image[..., 3] <= 0.0035] = \
                [1.0, 1.0, 1.0, 0.0]

            # This deviates a bit from the matplotlib version and just
            # calculates the root mean square error of all pixel values without
            # any other fancy considerations. It also uses the alpha channel of
            # the images. Scaled by 255.
            rms = np.sqrt(
                np.sum((255.0 * (expected_image - actual_image)) ** 2) /
                float(expected_image.size))
            self.assertTrue(rms <= 0.001)


def suite():
    return unittest.makeSuite(FlySpecPluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
