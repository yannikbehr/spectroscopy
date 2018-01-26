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
        np.testing.assert_array_almost_equal(fb.value[:], np.array([3.8, 1.2]), 1)
        self.assertEqual(fb.datetime[0], '2016-10-31T23:15:04')

    def read_single_station(self, d, station_info):
        """
        Read all the data for a single MiniDoas station for one day.
        """
        # Read the raw data
        e0 = d.read(station_info['files']['raw'], 
                    ftype='minidoas-raw', timeshift=13)
        ib = InstrumentBuffer(name=station_info['stationID'],
                              location=station_info['stationLoc'],
                              no_bits=16,
                              type='MiniDOAS')
        i = d.new(ib)
        try:
            rdt = d.elements['RawDataType'][0]
        except:
            rdt = d.new(e0['RawDataTypeBuffer'])
        rb = e0['RawDataBuffer']
        rb.type = rdt
        rb.instrument = i
        rb.target = station_info['target']
        lat = np.ones(rb.d_var.shape[0])*station_info['lat']
        lon = np.ones(rb.d_var.shape[0])*station_info['lon']
        elev = np.ones(rb.d_var.shape[0])*station_info['elev']
        bearing = np.ones(rb.d_var.shape[0])*np.rad2deg(station_info['bearing'])
        rb.position = np.array([lon, lat, elev]).T
        rb.bearing = bearing
        rb.inc_angle_error = np.ones(rb.d_var.shape[0])*0.013127537*180./np.pi
        rr = d.new(rb)

        # Read the concentration
        e1 = d.read(station_info['files']['spectra'],
                    date='2016-11-01', ftype='minidoas-spectra', timeshift=13)
        cb = e1['ConcentrationBuffer']
        idxs = np.zeros(cb.value.shape)
        for i in range(cb.value.shape[0]):
            idx = np.argmin(np.abs(rr.datetime[:].astype('datetime64[ms]') - cb.datetime[i].astype('datetime64[ms]')))
            idxs[i] = idx
        cb.rawdata = [rr]
        cb.rawdata_indices = idxs
        cb.method = station_info['widpro_method'] 
        cc = d.new(cb)

       
        # Read in the flux estimates for assumed height
        e3 = d.read(station_info['files']['flux_ah'],
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
            indices.append([idx0, idx1+1])
        fb.concentration = cc
        fb.concentration_indices = indices
        
        gfb1 = e3['GasFlowBuffer']

        m2 = None
        for _m in d.elements['Method']:
            if _m.name[:] == 'WS2PV':
                m2 = _m
        if m2 is None:
            mb2 = e3['MethodBuffer']
            m2 = d.new(mb2)

        gfb1.methods = [m2]
        gf1 = d.new(gfb1) 
        fb.gasflow = gf1 
        f = d.new(fb)

        # Read in the flux estimates for calculated height
        e4 = d.read(station_info['files']['flux_ch'],
                    date='2016-11-01', ftype='minidoas-scan',
                    station=station_info['wp_station_id'], timeshift=13)
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
        
        m3 = None
        for _m in d.elements['Method']:
            if _m.name[:] == 'WS2PVT':
                m3 = _m
        if m3 is None:
            mb3 = e4['MethodBuffer']
            new_description = mb3.description[0] + '; plume geometry inferred from triangulation'
            mb3.description = new_description
            mb3.name = 'WS2PVT'
            m3 = d.new(mb3)
            
        gfb2 = e4['GasFlowBuffer']
        gfb2.methods = [m3]
        gf2 = d.new(gfb2)
        fb1.gasflow = gf2
        f1 = d.new(fb1)

        # Now read in preferred flux values for assumed height downloaded from FITS
        data_ah = np.loadtxt(station_info['files']['fits_flux_ah'],
                          dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                          skiprows=1, delimiter=',')
        dates = data_ah['date'].astype('datetime64[s]')
        indices = []
        for i, dt in enumerate(dates):
            idx = np.argmin(np.abs(f.datetime[:].astype('datetime64[s]') - dt))
            indices.append(idx)
        pfb = PreferredFluxBuffer(fluxes=[f],
                                  flux_indices=[indices],
                                  value=data_ah['val'],
                                  value_error=data_ah['err'],
                                  datetime=dates.astype(str))
        d.new(pfb)

        # Now read in preferred flux values for calculated height downloaded from FITS
        data_ch = np.loadtxt(station_info['files']['fits_flux_ch'],
                          dtype=np.dtype([('date','S19'),('val',np.float),('err',np.float)]),
                          skiprows=1, delimiter=',')
        dates = data_ch['date'].astype('datetime64[s]')
        indices = []
        for i, dt in enumerate(dates):
            idx = np.argmin(np.abs(f1.datetime[:].astype('datetime64[s]') - dt))
            indices.append(idx)
        pfb1 = PreferredFluxBuffer(fluxes=[f1],
                                  flux_indices=[indices],
                                  value=data_ch['val'],
                                  value_error=data_ch['err'],
                                  datetime=dates.astype(str))
        d.new(pfb1)

    def test_readall(self):
        """
        Produce a complete HDF5 file for 1 day of MiniDOAS analysis at one station.
        """
        d = Dataset(tempfile.mktemp(), 'w')

        # ToDo: get correct plume coordinates
        tb = TargetBuffer(name='White Island main plume',
                          target_id='WI001',
                          position=[177.18375770, -37.52170799, 321.0])
        t = d.new(tb)
        
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

        # Read in the raw wind data; this is currently not needed to reproduce
        # flux estimates so it's just stored for reference
        fn_wd = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WD_00.txt')
        fn_ws = os.path.join(self.data_dir, 'minidoas', 'wind', '20161101_WS_00.txt')
        e2 = d.read({'direction': fn_wd, 'speed': fn_ws}, ftype='minidoas-wind', timeshift=13)
        gfb = e2['GasFlowBuffer']
        gf = d.new(gfb)

        station_info = {}
        station_info['WI301'] = {'files':{'raw':os.path.join(self.data_dir, 'minidoas', 'NE_20161101.csv'),
                                          'spectra':os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Spectra.csv'),
                                          'flux_ah':os.path.join(self.data_dir, 'minidoas', 'NE_2016_11_01_Scans.csv'),
                                          'flux_ch':os.path.join(self.data_dir, 'minidoas', 'XX_2016_11_01_Combined.csv'),
                                          'fits_flux_ah':os.path.join(self.data_dir, 'minidoas', 'FITS_NE_20161101_ah.csv'),
                                          'fits_flux_ch':os.path.join(self.data_dir, 'minidoas', 'FITS_NE_20161101_ch.csv')},
                                 'stationID': 'WI301',
                                 'stationLoc':'White Island North-East Point', 
                                 'target':t,
                                 'bearing':6.0214,
                                 'lon':177.192979384, 'lat':-37.5166903535, 'elev': 49.0,
                                 'widpro_method':m1,
                                 'wp_station_id':'NE'}

        station_info['WI302'] = {'files':{'raw':os.path.join(self.data_dir, 'minidoas', 'SR_20161101.csv'),
                                          'spectra':os.path.join(self.data_dir, 'minidoas', 'SR_2016_11_01_Spectra.csv'),
                                          'flux_ah':os.path.join(self.data_dir, 'minidoas', 'SR_2016_11_01_Scans.csv'),
                                          'flux_ch':os.path.join(self.data_dir, 'minidoas', 'XX_2016_11_01_Combined.csv'),
                                          'fits_flux_ah':os.path.join(self.data_dir, 'minidoas', 'FITS_SR_20161101_ah.csv'),
                                          'fits_flux_ch':os.path.join(self.data_dir, 'minidoas', 'FITS_SR_20161101_ch.csv')},
                                 'stationID': 'WI302',
                                 'stationLoc':'White Island South Rim', 
                                 'target':t,
                                 'bearing':3.8223,
                                 'lon':177.189013316, 'lat':-37.5265334424, 'elev':96.0,
                                 'widpro_method':m1,
                                 'wp_station_id':'SR'}
                                                                  
        self.read_single_station(d, station_info['WI301'])
        self.read_single_station(d, station_info['WI302'])
        d.close()


def suite():
    return unittest.makeSuite(MiniDoasPluginTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')

