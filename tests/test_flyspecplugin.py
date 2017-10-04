import inspect
import os
import tempfile
import unittest

import matplotlib.image
import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.flyspec import FlySpecPlugin
from spectroscopy.plugins.flyspec import FlySpecPluginException
from spectroscopy.util import split_by_scan, _array_multi_sort
from spectroscopy.visualize import plot

class FlySpecPluginTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_add(self):
        d1 = Dataset(tempfile.mktemp(), 'w')
        e = d1.read(os.path.join(self.data_dir, '2016_06_11_0830_TOFP04.txt'),
                         ftype='FLYSPEC', timeshift=12.0)
        r = d1.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = r
        d1.new(cb)

        d2 = Dataset(tempfile.mktemp(), 'w')
        e = d2.read(os.path.join(self.data_dir, '2016_06_11_0900_TOFP04.txt'),
                         ftype='FLYSPEC', timeshift=12.0)
        r = d2.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = r
        d2.new(cb)
        d1 += d2
        self.assertEqual(len(d1.elements['Concentration']), 2)
        self.assertEqual(len(d1.elements['RawData']), 2)


    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        #d.read(os.path.join(self.data_dir,'2016_06_11_0830_TOFP04.txt'),ftype='FLYSPEC')
        e = d.read(os.path.join(self.data_dir,'2012_02_29_1340_CHILE.txt'),
                        ftype='FLYSPEC')
        r = d.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = r
        c = d.new(cb)
        r = d.elements['RawData'][0]
        self.assertEqual(sum([x.size for x in r.datetime]), 4600)
        self.assertEqual(r.inc_angle[0], 174.750)
        c = d.elements['Concentration'][0]
        r1 = c.rawdata
        self.assertEqual(len(c.value[:]), 4600)
        np.testing.assert_array_almost_equal(r1.position[0],
                                             [-67.8047, -23.3565, 3927.], 2)

        # dicretize all retrievals onto a grid to show a daily plot
        bins = np.arange(0, 180, 1.0)
        m = [] 
        for _angle, _so2 in split_by_scan(r1.inc_angle[:], c.value[:]):
            _so2_binned = binned_statistic(_angle, _so2, 'mean', bins)
            m.append(_so2_binned.statistic)
        m = np.array(m)
        ids = np.argmax(np.ma.masked_invalid(m), axis=1)
        maxima = np.array([166., 167., 167., 167., 168., 167., 168., 167.,
                           167., 167., 167., 167., 168., 167., 167., 167.,
                           167., 166., 167., 166., 166., 167., 165., 165.,
                           165., 164., 165., 163., 163., 164., 163., 165.,
                           164., 164., 164., 161.])
        np.testing.assert_array_almost_equal(maxima, bins[ids], 2)

        d1 = Dataset(tempfile.mktemp(), 'w')
        e = d1.read(os.path.join(self.data_dir, '2016_06_11_0830_TOFP04.txt'),
                         ftype='FLYSPEC', timeshift=12.0)
        r = d1.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = r
        d1.new(cb)
        c = d1.elements['Concentration'][0]
        r = c.rawdata
        m = [] 
        for _angle, _so2 in split_by_scan(r.inc_angle[:], c.value[:]):
            _so2_binned = binned_statistic(_angle, _so2, 'mean', bins)
            m.append(_so2_binned.statistic)
        m = np.array(m)
        ids = np.argmax(np.ma.masked_invalid(m), axis=1)
        maxima = np.array([147., 25., 27., 86., 29., 31., 27., 27., 28., 137.,
                           34., 34.])
        np.testing.assert_array_almost_equal(maxima, bins[ids], 2)

    def test_not_enough_data(self):
        with self.assertRaises(FlySpecPluginException):
            d1 = Dataset(tempfile.mktemp(), 'w')
            e = d1.read(os.path.join(self.data_dir, '2015_05_03_1630_TOFP04.txt'),
                             ftype='FLYSPEC', timeshift=12.0)

    def test_plot(self):
        d = Dataset(tempfile.mktemp(), 'w')
        e = d.read(os.path.join(self.data_dir, '2012_02_29_1340_CHILE.txt'),
                        ftype='FLYSPEC', timeshift=12.0)
        r = d.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = r
        c = d.new(cb)
        with tempfile.TemporaryFile() as fd:
            plot(c, savefig=fd, timeshift=12.0)
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

    def test_spectra(self):
        """
        Test reading binary file containing the raw spectra together with 
        the text file.
        """
        d = Dataset(tempfile.mktemp(), 'w')
        fin_txt = os.path.join(self.data_dir,'TOFP04', '2017_06_14_0930.txt')
        fin_bin = os.path.join(self.data_dir,'TOFP04', '2017_06_14_0930.bin')
        x = [521,637,692,818]
        y = [305.,315.,319.5,330.]
        f = interp1d(x, y, fill_value='extrapolate')
        xnew = range(0,2048)
        wavelengths = f(xnew)
        e = d.read(fin_txt, spectra=fin_bin, wavelengths=wavelengths,
                   ftype='flyspec', timeshift=12.0)
        self.assertEqual(e['RawDataBuffer'].d_var.shape, (1321,2048))
        with tempfile.TemporaryFile() as fd:
            plot(e['RawDataBuffer'], savefig=fd)
            expected_image = matplotlib.image.imread(
                os.path.join(self.data_dir, 'raw_data_plot.png'),
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
