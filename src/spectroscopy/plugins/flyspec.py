"""
Plugin to read FlySpec data.
"""
import datetime
import os
import struct

import numpy as np

from spectroscopy.datamodel import (RawDataBuffer,
                                    ConcentrationBuffer,
                                    RawDataTypeBuffer,
                                    MethodBuffer,
                                    FluxBuffer,
                                    GasFlowBuffer)
from spectroscopy.plugins import DatasetPluginBase
from spectroscopy.util import bearing2vec


class FlySpecPluginException(Exception):
    pass


class FlySpecPlugin(DatasetPluginBase):

    def _read_spectra(self, fin):
        """
        Read spectra from binary file.
        """
        with open(fin, "rb") as ifp:
            raw_data = ifp.read()
        i = 0
        counts = []
        while i < len(raw_data):
            counts.append(struct.unpack("2048f", raw_data[i:i+(2048 * 4)]))
            i += (2048 * 4)
        return counts

    def read(self, dataset, filename, timeshift=0, **kargs):
        """
        Load data from FlySpec instruments.

        :param timeshift: float
        :type timeshift: FlySpecs record data in local time so a timeshift in
            hours of local time with respect to UTC can be given. For example
            `timeshift=12.00` will subtract 12 hours from the recorded time.

        """
        # load data and convert southern hemisphere to negative
        # latitudes and western hemisphere to negative longitudes
        def todd(x):
            """
            Convert degrees and decimal minutes to decimal degrees.
            """
            idx = x.find('.')
            minutes = float(x[idx - 2:]) / 60.
            deg = float(x[:idx - 2])
            return deg + minutes

        data = np.loadtxt(filename, usecols=range(0, 21),
                          converters={
                              8: todd,
                              9: lambda x: -1.0 if x.lower() == 's' else 1.0,
                              10: todd,
                              11: lambda x: -1.0 if x.lower() == 'w' else 1.0})
        data = np.atleast_2d(data)
        specfile = kargs.get('spectra', None)
        if specfile is not None:
            wavelengths = kargs.get('wavelengths', None)
            if wavelengths is not None:
                spectra = np.array(self._read_spectra(specfile))
                if spectra.shape[0] != data.shape[0]:
                    raise FlySpecPluginException(
                        "Spectra and concentration don't have the same shape.")
                if spectra.shape[1] != wavelengths.size:
                    raise FlySpecPluginException(
                        "Spectra and wavelengths don't have the same size.")

        if len(data.shape) < 2:
            raise FlySpecPluginException(
                'File %s contains only one data point.'
                % (os.path.basename(filename)))

        bearing = None
        try:
            bearing = kargs['bearing']
        except KeyError:
            pass
        else:
            bearing = np.ones(data.shape[0])*bearing
        ts = -1. * timeshift * 60. * 60.
        int_times = np.zeros(data[:, :7].shape, dtype='int')
        int_times[:, :6] = data[:, 1:7]
        # convert decimal seconds to microseconds
        int_times[:, 6] = (data[:, 6] - int_times[:, 5]) * 1e6
        # ToDo: handle timezones properly
        times = [datetime.datetime(*int_times[i, :]) +
                 datetime.timedelta(seconds=ts)
                 for i in range(int_times.shape[0])]
        unix_times = [i.isoformat() for i in times]
        latitude = data[:, 8] * data[:, 9]
        longitude = data[:, 10] * data[:, 11]
        elevation = data[:, 12]
        so2 = data[:, 16]
        angles = data[:, 17]
        if specfile is not None:
            rb = RawDataBuffer(inc_angle=angles,
                               bearing=bearing,
                               position=np.array([longitude,
                                                  latitude,
                                                  elevation]).T,
                               datetime=unix_times,
                               ind_var=wavelengths,
                               d_var=spectra)
        else:
            rb = RawDataBuffer(inc_angle=angles,
                               position=np.array([longitude,
                                                  latitude,
                                                  elevation]).T,
                               datetime=unix_times)
        rdtb = RawDataTypeBuffer(d_var_unit='ppm m',
                                 ind_var_unit='nm', name='measurement')
        cb = ConcentrationBuffer(gas_species='SO2', value=so2)
        return {str(rb): rb, str(rdtb): rdtb, str(cb): cb}

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpec backend')

    @staticmethod
    def get_format():
        return 'flyspec'


class FlySpecFluxPlugin(DatasetPluginBase):

    def read(self, dataset, filename, timeshift=0, **kargs):
        """
        Read flux estimates.
        """
        data = np.fromregex(filename, r'(\S+ \S+)\s+(-?\d+\.\d+)',
                            dtype={'names': ('datetime', 'flux'),
                                   'formats': ('S26', np.float)})
        dt = data['datetime'].astype('datetime64[us]')
        # convert milliseconds to microseconds to fix a bug in Nial's code
        us = dt - dt.astype('datetime64[s]')
        dtn = dt.astype('datetime64[s]') + us*1000
        ts = np.timedelta64(int(timeshift), 'h')
        # convert to UTC
        dtn -= ts
        f = data['flux']

        mb = MethodBuffer(name='GNS FlySpec UI')
        fb = FluxBuffer(value=f, datetime=dtn.astype(str))
        return {str(fb): fb, str(mb): mb}

    @staticmethod
    def get_format():
        return 'flyspecflux'


class FlySpecRefPlugin(DatasetPluginBase):

    def _read_spectra(self, fin):
        """
        Read spectra from binary file.
        """
        with open(fin, "rb") as ifp:
            raw_data = ifp.read()
        i = 0
        counts = []
        while i < len(raw_data):
            counts.append(struct.unpack("2048f",raw_data[i:i+(2048 * 4)]))
            i += (2048 * 4)
        return counts

    def read(self, dataset, filename,  **kargs):
        """
        Read reference spectra for FlySpec.
        """
        try:
            wavelengths = kargs['wavelengths']
            mtype = kargs['type']
        except KeyError:
            raise FlySpecPluginException('Please provide wavelengths and measurement type.')
        
        spectra = np.array(self._read_spectra(filename))
        if spectra.shape[1] != wavelengths.size:
            raise FlySpecPluginException("Spectra and wavelengths don't have the same size.")
        rb = RawDataBuffer(ind_var=wavelengths, d_var=spectra)
        rdtb = RawDataTypeBuffer(d_var_unit='ppm m', ind_var_unit='nm', name=mtype)
        return {str(rb):rb, str(rdtb):rdtb}

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpecRef backend')

    @staticmethod
    def get_format():
        return 'flyspecref'


class FlySpecWindPlugin(DatasetPluginBase):
    
    def read(self, dataset, filename, timeshift=0, **kargs):
        """
        Read the wind data for the Flyspecs on Tongariro.
        """
        data = np.loadtxt(filename, dtype={'names':('date', 'wd', 'ws'), 
                                           'formats':('S19', np.float, np.float)})

        npts = data.shape[0]
        position = np.tile(np.array([175.673, -39.108, 0.0]), (npts, 1)) 
        vx = np.zeros(npts)
        vy = np.zeros(npts)
        vz = np.zeros(npts)
        time = np.empty(npts,dtype='S19')
        for i in range(npts):
            ws = data['ws'][i]
            wd = data['wd'][i]
            date = data['date'][i]
            # if windspeed is 0 give it a tiny value
            # so that the bearing can be reconstructed
            if ws == 0.:
                ws = 0.0001
            _vx, _vy = bearing2vec(wd, ws)
            vx[i] = _vx
            vy[i] = _vy
            vz[i] = np.nan
            time[i] = date 

        dt = time.astype('datetime64[us]')
        ts = np.timedelta64(int(timeshift), 'h')
        # convert to UTC
        dt -= ts
        description = 'Wind measurements and forecasts by NZ metservice \
        for Te Maari.'
        mb = MethodBuffer(name='some model')
        m = dataset.new(mb)
        gfb = GasFlowBuffer(methods=[m], vx=vx, vy=vy, vz=vz,
                            position=position, datetime=dt.astype(str), 
                            user_notes=description, unit='m/s')
        gf = dataset.new(gfb)
        return gf

    @staticmethod
    def get_format():
        return 'flyspecwind'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
