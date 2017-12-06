import inspect
import os
import tempfile
import unittest

import numpy as np

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.minidoas import MiniDoasException
from spectroscopy.visualize import plot
from spectroscopy.datamodel import (PreferredFluxBuffer,
                                    InstrumentBuffer,
                                    TargetBuffer,
                                    MethodBuffer)
from spectroscopy.util import vec2bearing, bearing2vec


class MiniDoasPluginTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def setUp(self):
        self.data_dir = os.path.join(os.path.dirname(os.path.abspath(
            inspect.getfile(inspect.currentframe()))), "data")

    def test_read(self):
        d = Dataset(tempfile.mktemp(), 'w')
        e = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_20161101.csv'),
                        ftype='minidoas-raw')
        rb = e['RawDataBuffer']
        self.assertEqual(rb.d_var.shape, (7615, 482)) 
        self.assertEqual(rb.d_var[0,0], 78)
        self.assertEqual(rb.datetime[0], '2016-11-01T09:00:00.070000')
        self.assertEqual(rb.datetime[-1], '2016-11-01T16:29:54.850000')
        
        with self.assertRaises(MiniDoasException):
            e1 = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Spectra.csv'),
                        ftype='minidoas-spectra')
        e1 = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Spectra.csv'),
                    date='2016-11-01', ftype='minidoas-spectra', timeshift=13)
        cb = e1['ConcentrationBuffer']
        self.assertEqual(cb.datetime[-1], '2016-11-01T03:28:07.410000')

        fn_wd = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WD_00.txt')
        fn_ws = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WS_00.txt')

        e2 = d.read({'direction': fn_wd, 'speed': fn_ws}, timeshift=13, ftype='minidoas-wind')
        gfb = e2['GasFlowBuffer']
        self.assertEqual(int(vec2bearing(gfb.vx[0], gfb.vy[0])), 240)
        self.assertAlmostEqual(np.sqrt(gfb.vx[0]**2 + gfb.vy[0]**2), 3.2, 1)
        self.assertEqual(gfb.datetime[0], "2016-10-31T19:10:00")
        e3 = d.read(os.path.join(self.data_dir, 'minidoas', 'XX_2016_11_01_Combined.csv'),
                    date='2016-11-01', ftype='minidoas-scan', station='NE', timeshift=13)
        fb = e3['FluxBuffer']
        np.testing.assert_array_almost_equal(fb.value[:], np.array([328.2, 103.8]), 1)
        self.assertEqual(fb.datetime[0], '2016-10-31T23:15:04')

    def test_readall(self):
        """
        Produce a complete HDF5 file for 1 day of MiniDOAS analysis at one station.
        """
        #d = Dataset(tempfile.mktemp(), 'w')
        d = Dataset('/tmp/minidoas_test.h5', 'w')
        # Read the raw data
        e0 = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_20161101.csv'),
                     ftype='minidoas-raw', timeshift=13)
        ib = InstrumentBuffer(name='WI301',
                              location='White Island North-East Point',
                              no_bits=16,
                              type='MiniDOAS')
        i = d.new(ib)
        # ToDo: get correct plume coordinates
        tb = TargetBuffer(name='White Island main plume',
                          target_id='WI001',
                          position=[177.18375770, -37.52170799, 321.0])
        t = d.new(tb)
        rdt = d.new(e0['RawDataTypeBuffer'])
        rb = e0['RawDataBuffer']
        rb.type = rdt
        rb.instrument = i
        rb.target = t
        lat = np.ones(rb.d_var.shape[0])*-37.516690
        lon = np.ones(rb.d_var.shape[0])*177.1929793
        elev = np.ones(rb.d_var.shape[0])*30.0
        rb.position = np.array([lon, lat, elev]).T
        rb.inc_angle_error = np.ones(rb.d_var.shape[0])*0.013127537*180./np.pi
        rr = d.new(rb)

        # Read the concentration
        e1 = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Spectra.csv'),
                     date='2016-11-01', ftype='minidoas-spectra', timeshift=13)
        cb = e1['ConcentrationBuffer']
        wpoptions = "{'Pixel316nm':479, 'TrimLower':30, 'LPFilterCount':3, 'MinWindSpeed':3,"
        wpoptions += "'BrightEnough':500, 'BlueStep':5, 'MinR2:0.8, 'MaxFitCoeffError':50.0,"
        wpoptions += "'InPlumeThresh':0.05, 'MinPlumeAngle':0.1, 'MaxPlumeAngle':3.0,"
        wpoptions += "'MinPlumeSect':0.4, 'MaxPlumeSect':2.0, 'MeanPlumeCtrHeight':310,"
        wpoptions += "'SEMeanPlumeCtrHeight':0.442, 'MaxRangeToPlume':5000, 'MaxPlumeWidth':2600"
        wpoptions += "'MaxPlumeCentreAltitude':2000, 'MaxRangeSeperation':5000,"
        wpoptions += "'MaxAltSeperation':1000, 'MaxTimeDiff':30,"
        wpoptions += "'MinTriLensAngle':0.1745, 'MaxTriLensAngle':2.9671,"
        wpoptions += "'SEWindSpeed':0.20, 'WindMultiplier':1.24, 'SEWindDir':0.174}"
        mb1 = MethodBuffer(name='WidPro v1.2',
                           description='Jscript wrapper for DOASIS',
                           settings=wpoptions)
        m1 = d.new(mb1)
        idxs = np.zeros(cb.value.shape)
        for i in range(cb.value.shape[0]):
            idx = np.argmin(np.abs(rr.datetime[:].astype('datetime64[ms]') - cb.datetime[i].astype('datetime64[ms]')))
            idxs[i] = idx
        cb.rawdata = [rr]
        cb.rawdata_indices = idxs
        cb.method = m1
        cc = d.new(cb)
        # Read in the raw wind data
        fn_wd = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WD_00.txt')
        fn_ws = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WS_00.txt')
        e2 = d.read({'direction': fn_wd, 'speed': fn_ws}, ftype='minidoas-wind', timeshift=13)
        gfb = e2['GasFlowBuffer']
        # Read in the flux estimates
        e3 = d.read(os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Scans.csv'),
                    date='2016-11-01', ftype='minidoas-scan', timeshift=13)
        fb = e3['FluxBuffer']
        dt = fb.datetime[:].astype('datetime64[s]')
        indices = []
        for _dt in dt:
            idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]') - _dt))
            idx0 = idx
            while True:
                angle = rr.inc_angle[cc.rawdata_indices[idx]+1]
                if angle > 180.:
                    break
                idx += 1
            idx1 = idx
            indices.append([idx0, idx1])
        fb.concentration = cc
        fb.concentration_indices = indices
        
        # Now combine the wind speed with the plume direction
        mb2 = e3['MethodBuffer']
        gfb1 = e3['GasFlowBuffer']
        dt = gfb1.datetime[:].astype('datetime64[s]')
        vx = []
        vy = []
        for i, _dt in enumerate(dt):
            idx = np.argmin(np.abs(gfb.datetime[:].astype('datetime64[s]') - _dt))
            vx1, vy1 = gfb.vx[idx], gfb.vy[idx]
            ws = np.sqrt(vx1*vx1 + vy1*vy1)
            vx2, vy2 = gfb1.vx[i], gfb1.vy[i]
            wd = vec2bearing(vx2, vy2)
            vx2, vy2 = bearing2vec(wd, ws)
            vx.append(vx2)
            vy.append(vy2)
        gfb1.vx = vx
        gfb1.vy = vy
        gf = d.new(gfb)
        m2 = d.new(mb2)
        gfb1.methods = [m2]
        gf1 = d.new(gfb1) 
        fb.gasflow = gf1 
        f = d.new(fb)

        # Do the same thing with combined scan
        e4 = d.read(os.path.join(self.data_dir, 'minidoas', 'XX_2016_11_01_Combined.csv'),
                    date='2016-11-01', ftype='minidoas-scan', station='NE', timeshift=13)
        fb1 = e4['FluxBuffer']
        dt = fb1.datetime[:].astype('datetime64[s]')
        indices = []
        for _dt in dt:
            idx = np.argmin(np.abs(cc.datetime[:].astype('datetime64[us]') - _dt))
            idx0 = idx
            while True:
                angle = rr.inc_angle[cc.rawdata_indices[idx]+1]
                if angle > 180.:
                    break
                idx += 1
            idx1 = idx
            indices.append([idx0, idx1])
        fb1.concentration = cc
        fb1.concentration_indices = indices
        
        # Now combine the wind speed with the plume direction
        mb3 = e4['MethodBuffer']
        new_description = mb3.description[0] + '; plume geometry inferred from triangulation'
        mb3.description = new_description
        mb3.name = 'WS2PVT'
        m3 = d.new(mb3)
        gfb2 = e4['GasFlowBuffer']
        dt = gfb2.datetime[:].astype('datetime64[s]')
        vx = []
        vy = []
        for i, _dt in enumerate(dt):
            idx = np.argmin(np.abs(gfb.datetime[:].astype('datetime64[s]') - _dt))
            vx1, vy1 = gfb.vx[idx], gfb.vy[idx]
            ws = np.sqrt(vx1*vx1 + vy1*vy1)
            vx2, vy2 = gfb2.vx[i], gfb2.vy[i]
            wd = vec2bearing(vx2, vy2)
            vx2, vy2 = bearing2vec(wd, ws)
            vx.append(vx2)
            vy.append(vy2)
        gfb2.vx = vx
        gfb2.vy = vy
        gfb2.methods = [m3]
        gf2 = d.new(gfb2)
        fb1.gasflow = gf2
        f1 = d.new(fb1)

        # Now read in preferred flux values for assumed height downloaded from FITS
        data = np.loadtxt(os.path.join(self.data_dir, 'minidoas', 'FITS_NE_20161101_ah.csv'),
                          dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                          skiprows=1, delimiter=',')
        dates = data['date'].astype('datetime64[s]')
        indices = []
        for i, dt in enumerate(dates):
            idx = np.argmin(np.abs(f.datetime[:].astype('datetime64[s]') - dt))
            indices.append(idx)
        pfb = PreferredFluxBuffer(fluxes=[f],
                                  flux_indices=[indices],
                                  value=data['val']*86.4,
                                  value_error=data['err'],
                                  datetime=dates.astype(str))
        d.new(pfb)

        # Now read in preferred flux values for calculated height downloaded from FITS
        data = np.loadtxt(os.path.join(self.data_dir, 'minidoas', 'FITS_NE_20161101_ch.csv'),
                          dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                          skiprows=1, delimiter=',')
        dates = data['date'].astype('datetime64[s]')
        indices = []
        for i, dt in enumerate(dates):
            idx = np.argmin(np.abs(f1.datetime[:].astype('datetime64[s]') - dt))
            indices.append(idx)
        pfb1 = PreferredFluxBuffer(fluxes=[f1],
                                  flux_indices=[indices],
                                  value=data['val']*86.4,
                                  value_error=data['err'],
                                  datetime=dates.astype(str))
        d.new(pfb1)
        d.close()


def suite():
    return unittest.makeSuite(MiniDoasPluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

