from __future__ import division
from __future__ import print_function
from builtins import zip
from builtins import range
from past.utils import old_div
import datetime
import glob
import inspect
import os
import tempfile
import unittest

import matplotlib
matplotlib.use('Agg')
import matplotlib.image
import numpy as np
from scipy.stats import binned_statistic
from scipy.interpolate import interp1d

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.flyspec import FlySpecPlugin
from spectroscopy.plugins.flyspec import FlySpecPluginException
from spectroscopy.util import split_by_scan, _array_multi_sort, vec2bearing
from spectroscopy.visualize import plot
from spectroscopy.datamodel import (InstrumentBuffer,
                                    TargetBuffer,
                                    PreferredFluxBuffer)


class FlySpecPluginTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def compare_images(self, fh, image_fn):
        fh.seek(0)
        actual_image = matplotlib.image.imread(fh, format='png')
        expected_image = matplotlib.image.imread(image_fn, format='png')

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
            old_div(np.sum((255.0 * (expected_image - actual_image)) ** 2),
            float(expected_image.size)))
        return rms

    def test_add(self):
        d1 = Dataset(tempfile.mktemp(), 'w')
        e = d1.read(os.path.join(self.data_dir, '2016_06_11_0830_TOFP04.txt'),
                    ftype='FLYSPEC', timeshift=12.0)
        r = d1.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = [r]
        d1.new(cb)

        d2 = Dataset(tempfile.mktemp(), 'w')
        e = d2.read(os.path.join(self.data_dir, '2016_06_11_0900_TOFP04.txt'),
                    ftype='FLYSPEC', timeshift=12.0)
        r = d2.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = [r]
        d2.new(cb)
        d1 += d2
        self.assertEqual(len(d1.elements['Concentration']), 2)
        self.assertEqual(len(d1.elements['RawData']), 2)

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        e = d.read(os.path.join(self.data_dir,
                                '2012_02_29_1340_CHILE.txt'),
                   ftype='FLYSPEC')
        r = d.new(e['RawDataBuffer'])
        cb = e['ConcentrationBuffer']
        cb.rawdata = [r]
        c = d.new(cb)
        r = d.elements['RawData'][0]
        self.assertEqual(sum([x.size for x in r.datetime]), 4600)
        self.assertEqual(r.inc_angle[0], 174.750)
        c = d.elements['Concentration'][0]
        r1 = c.rawdata[0]
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
        cb.rawdata = [r]
        d1.new(cb)
        c = d1.elements['Concentration'][0]
        r = c.rawdata[0]
        m = []
        for _angle, _so2 in split_by_scan(r.inc_angle[:], c.value[:]):
            _so2_binned = binned_statistic(_angle, _so2, 'mean', bins)
            m.append(_so2_binned.statistic)
        m = np.array(m)
        ids = np.argmax(np.ma.masked_invalid(m), axis=1)
        maxima = np.array([147., 25., 27., 86., 29., 31., 27., 27., 28., 137.,
                           34., 34.])
        np.testing.assert_array_almost_equal(maxima, bins[ids], 2)

    def test_read_flux(self):
        d = Dataset(tempfile.mktemp(), 'w')
        fin = os.path.join(self.data_dir, 'TOFP04', 'TOFP04_2017_06_14.txt')
        e = d.read(fin, ftype='flyspecflux', timeshift=13.0)
        nlines = None
        with open(fin) as fd:
            nlines = len(fd.readlines())
        self.assertEqual(e['FluxBuffer'].value.shape, (nlines-1,))
        fb = e['FluxBuffer']
        self.assertEqual(fb.datetime[-1], 
                         np.datetime64('2017-06-14T03:29:38.033000'))

    def test_read_refspec(self):
        d = Dataset(tempfile.mktemp(), 'w')
        x = [521, 637, 692, 818]
        y = [305., 315., 319.5, 330.]
        f = interp1d(x, y, fill_value='extrapolate')
        xnew = list(range(0, 2048))
        wavelengths = f(xnew)

        with self.assertRaises(FlySpecPluginException):
            e = d.read(os.path.join(self.data_dir, 'TOFP04',
                                    'Cal_20170602_0956_dark.bin'),
                       ftype='FLYSPECREF', wavelengths=wavelengths)

        e = d.read(os.path.join(self.data_dir, 'TOFP04',
                                'Cal_20170602_0956_dark.bin'),
                   ftype='FLYSPECREF', type='dark', wavelengths=wavelengths)
        self.assertEqual(e['RawDataBuffer'].d_var.shape, (10, 2048))

    def test_read_wind(self):
        d = Dataset(tempfile.mktemp(), 'w')
        fin = os.path.join(self.data_dir, 'TOFP04', 'wind', '2017_06_14.txt')
        gf = d.read(fin, ftype='flyspecwind', timeshift=13)
        vx = gf.vx[0]
        vy = gf.vy[0]
        dt = gf.datetime[0]
        v = np.sqrt(vx*vx + vy*vy)
        self.assertAlmostEqual(v, 10.88, 2)
        self.assertAlmostEqual(vec2bearing(vx, vy), 255, 6)
        self.assertEqual(dt, np.datetime64('2017-06-13T17:00:00'))

    @unittest.skip("Skipping")
    def test_plot(self):
        d = Dataset(tempfile.mktemp(), 'w')
        e = d.read(os.path.join(self.data_dir, '2012_02_29_1340_CHILE.txt'),
                   ftype='FLYSPEC', timeshift=12.0)
        rdt = d.new(e['RawDataTypeBuffer'])
        rb = e['RawDataBuffer']
        rb.type = rdt
        r = d.new(rb)
        cb = e['ConcentrationBuffer']
        cb.rawdata = [r]
        cb.rawdata_indices = np.arange(cb.value.shape[0])
        c = d.new(cb)
        if False:
            with tempfile.TemporaryFile() as fd:
                plot(c, savefig=fd, timeshift=12.0)
                expected_image = os.path.join(self.data_dir,
                                              'chile_retrievals_overview.png')
                rms = self.compare_images(fd, expected_image)
                self.assertTrue(rms <= 0.001)

    def test_spectra(self):
        """
        Test reading binary file containing the raw spectra together with
        the text file.
        """
        d = Dataset(tempfile.mktemp(), 'w')
        fin_txt = os.path.join(self.data_dir, 'TOFP04', '2017_06_14_0930.txt')
        fin_bin = os.path.join(self.data_dir, 'TOFP04', '2017_06_14_0930.bin')
        fin_high = os.path.join(self.data_dir, 'TOFP04',
                                'Cal_20170602_0956_high.bin')
        fin_low = os.path.join(self.data_dir, 'TOFP04',
                               'Cal_20170602_0956_low.bin')
        fin_dark = os.path.join(self.data_dir, 'TOFP04',
                                'Cal_20170602_0956_dark.bin')
        fin_ref = os.path.join(self.data_dir, 'TOFP04',
                               'Cal_20170602_0956_ref.bin')

        x = [521, 637, 692, 818]
        y = [305., 315., 319.5, 330.]
        f = interp1d(x, y, fill_value='extrapolate')
        xnew = list(range(0, 2048))
        wavelengths = f(xnew)
        e = d.read(fin_txt, spectra=fin_bin, wavelengths=wavelengths,
                   ftype='flyspec', timeshift=12.0)
        self.assertEqual(e['RawDataBuffer'].d_var.shape, (1321, 2048))
        rdtb = e['RawDataTypeBuffer']
        rdt = d.new(rdtb)
        rb = e['RawDataBuffer']
        rb.type = rdt
        r = d.new(rb)
        cb = e['ConcentrationBuffer']
        rdlist = [r]
        for _f in [fin_high, fin_low, fin_dark, fin_ref]:
            e = d.read(_f, ftype='flyspecref', wavelengths=wavelengths,
                       type=_f.replace('fin_', ''))
            rdtb = e['RawDataTypeBuffer']
            rdt = d.new(rdtb)
            rb = e['RawDataBuffer']
            rb.type = rdt
            r = d.new(rb)
            rdlist.append(r)
        cb.rawdata = rdlist
        c = d.new(cb)
        for _r in c.rawdata[:]:
            if _r.type.name[0] == 'measurement':
                break
        if False:
            with tempfile.TemporaryFile() as fd:
                plot(_r, savefig=fd)
                expected_image = os.path.join(self.data_dir,
                                              'raw_data_plot.png')
                rms = self.compare_images(fd, expected_image)
                self.assertTrue(rms <= 0.001)

    def test_readabunch(self):
        """
        Read in a whole day's worth of data including the reference spectra,
        the flux results, and the wind data.
        """
        def keyfunc(fn):
            date = os.path.basename(fn).split('.')[0]
            year, month, day, hourmin = date.split('_')
            return datetime.datetime(int(year), int(month), int(day),
                                     int(hourmin[0:2]), int(hourmin[2:]))

        # Reference spectra
        fin_high = os.path.join(self.data_dir, 'TOFP04',
                                'Cal_20170602_0956_high.bin')
        fin_low = os.path.join(self.data_dir, 'TOFP04',
                               'Cal_20170602_0956_low.bin')
        fin_dark = os.path.join(self.data_dir, 'TOFP04',
                                'Cal_20170602_0956_dark.bin')
        fin_ref = os.path.join(self.data_dir, 'TOFP04',
                               'Cal_20170602_0956_ref.bin')

        bearing = 285.
        x = [521, 637, 692, 818]
        y = [305., 315., 319.5, 330.]
        f = interp1d(x, y, fill_value='extrapolate')
        xnew = list(range(0, 2048))
        wavelengths = f(xnew)

        d = Dataset(tempfile.mktemp(), 'w')
        ib = InstrumentBuffer(location='Te Maari crater',
                              type='FlySpec',
                              name='TOFP04')
        inst = d.new(ib)
        tb = TargetBuffer(name='Upper Te Maari crater',
                          position=[175.671854359, -39.107850505, 1505.])
        t = d.new(tb)

        rdlist = []
        for _k, _f in zip(['high', 'low', 'dark', 'ref'],
                          [fin_high, fin_low, fin_dark, fin_ref]):
            e = d.read(_f, ftype='flyspecref', wavelengths=wavelengths,
                       type=_k)
            rdtb = e['RawDataTypeBuffer']
            rdt = d.new(rdtb)
            rb = e['RawDataBuffer']
            rb.type = rdt
            rb.instrument = inst
            r = d.new(rb)
            rdlist.append(r)

        files = glob.glob(os.path.join(self.data_dir, 'TOFP04', '2017*.txt'))
        files = sorted(files, key=keyfunc)
        r = None
        c = None
        nlines = 0
        last_index = 0
        for _f in files:
            try:
                fin_bin = _f.replace('.txt', '.bin')
                with open(_f) as fd:
                    nlines += len(fd.readlines())
                e = d.read(_f, ftype='FLYSPEC', spectra=fin_bin,
                           wavelengths=wavelengths, bearing=bearing,
                           timeshift=12)
                if r is None and c is None:
                    rdt = d.new(e['RawDataTypeBuffer'])
                    rb = e['RawDataBuffer']
                    rb.type = rdt
                    rb.instrument = inst
                    rb.target = t
                    r = d.new(rb)
                    cb = e['ConcentrationBuffer']
                    rdlist.append(r)
                    cb.rawdata = rdlist
                    cb.rawdata_indices = np.arange(cb.value.shape[0])
                    last_index = cb.value.shape[0] - 1
                    c = d.new(cb)
                else:
                    r.append(e['RawDataBuffer'])
                    cb = e['ConcentrationBuffer']
                    cb.rawdata_indices = (last_index + 1 +
                                          np.arange(cb.value.shape[0]))
                    last_index = last_index + cb.value.shape[0]
                    c.append(cb)
            except Exception as ex:
                print((ex, _f, fin_bin))
                continue
        # Check all data has been read
        self.assertEqual(c.rawdata[4].d_var.shape, (nlines, 2048))
        self.assertEqual(c.rawdata[4].inc_angle.shape, (nlines,))
        self.assertEqual(c.value[0], 119.93)
        self.assertEqual(c.value[-1], 23.30)
        self.assertEqual(c.rawdata[4].datetime[-1],
                         np.datetime64('2017-06-14T04:30:00.535'))
        self.assertEqual(c.rawdata[4].datetime[0],
                         np.datetime64('2017-06-13T20:30:49.512'))
        if False:
            with tempfile.TemporaryFile() as fd:
                plot(c, savefig=fd)
                expected_image = os.path.join(self.data_dir, 'TOFP04',
                                              'concentration_plot.png')
                rms = self.compare_images(fd, expected_image)
                self.assertTrue(rms <= 0.001)
            with tempfile.TemporaryFile() as fd:
                plot(c.rawdata[0], savefig=fd)
                expected_image = os.path.join(self.data_dir, 'TOFP04',
                                              'ref_spectrum.png')
                rms = self.compare_images(fd, expected_image)
                self.assertTrue(rms <= 0.001)

        fe = d.read(os.path.join(self.data_dir, 'TOFP04',
                                 'TOFP04_2017_06_14.txt'),
                    ftype='flyspecflux', timeshift=12)
        gf = d.read(os.path.join(self.data_dir, 'TOFP04', 'wind',
                                 '2017_06_14.txt'),
                    ftype='flyspecwind', timeshift=12)
        fb = fe['FluxBuffer']
        draw = r.datetime[:].astype('datetime64[us]')
        inds = []
        for i in range(fb.value.shape[0]):
            d0 = fb.datetime[i].astype('datetime64[us]')
            idx0 = np.argmin(abs(draw-d0))
            if i < fb.value.shape[0]-1:
                d1 = fb.datetime[i+1].astype('datetime64[us]')
                idx1 = np.argmin(abs(draw-d1))
                # There is a small bug in Nial's program that gets
                # the start of the final scan in a file wrong
                if r.inc_angle[idx1-1] < r.inc_angle[idx1]:
                    idx1 -= 1
                    fb.datetime[i+1] = r.datetime[idx1]
            else:
                idx1 = r.datetime.shape[0]
            inds.append([idx0, idx1-1])

        fb.concentration_indices = inds
        fb.concentration = c
        mb = fe['MethodBuffer']
        m = d.new(mb)
        fb.method = m
        fb.gasflow = gf
        f = d.new(fb)
        nos = 18
        i0, i1 = f.concentration_indices[nos]
        cn = f.concentration
        rn = cn.rawdata[4]
        self.assertAlmostEqual(f.value[nos], 0.62, 2)
        self.assertEqual(rn.inc_angle[i0], 25.)
        self.assertEqual(rn.inc_angle[i1], 150.)
        self.assertEqual(f.datetime[nos],
                         np.datetime64('2017-06-13T21:20:17.196000'))

        pfb = PreferredFluxBuffer(fluxes=[f],
                                  flux_indices=[[nos]],
                                  value=[f.value[nos]],
                                  datetime=[f.datetime[nos]])
        d.new(pfb)


def suite():
    return unittest.makeSuite(FlySpecPluginTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
