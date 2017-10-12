"""
Plugin to read FlySpec data.
"""
import calendar
import datetime
import os
import struct

import numpy as np

from spectroscopy.datamodel import (RawDataBuffer, 
                                    ConcentrationBuffer,
                                    RawDataTypeBuffer)
from spectroscopy.plugins import DatasetPluginBase

class FlySpecPluginException(Exception):
    pass


class FlySpecPlugin(DatasetPluginBase):

    def _read_spectra(self, fin):
        """
        Read spectra from binary file.
        """
        with open(fin,"rb") as ifp:
            raw_data = ifp.read()
        i = 0
        counts = []
        while i < len(raw_data):
            counts.append(struct.unpack("2048f",raw_data[i:i+(2048 * 4)]))
            i += (2048 * 4)
        return counts

    def read(self, dataset, filename, timeshift=0.0, **kargs):
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

        bearing=None
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
                               position=np.array([longitude, latitude, elevation]).T,
                               datetime=unix_times,
                               ind_var = wavelengths,
                               d_var = spectra)
        else:
            rb = RawDataBuffer(inc_angle=angles,
                               position=np.array([longitude, latitude, elevation]).T,
                               datetime=unix_times)
        rdtb = RawDataTypeBuffer(d_var_unit='ppm m', ind_var_unit='nm', name='measurement')
        cb = ConcentrationBuffer(gas_species='SO2', value=so2)
        return {str(rb):rb, str(rdtb):rdtb, str(cb):cb}

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpec backend')

    @staticmethod
    def get_format():
        return 'flyspec'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
