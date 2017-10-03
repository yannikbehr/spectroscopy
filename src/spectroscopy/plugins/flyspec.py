"""
Plugin to read and write FlySpec data.
"""
import calendar
import datetime
import os

import numpy as np

from spectroscopy.datamodel import RawDataBuffer, ConcentrationBuffer
from spectroscopy.plugins import DatasetPluginBase

class FlySpecPluginException(Exception):
    pass


class FlySpecPlugin(DatasetPluginBase):

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
        if len(data.shape) < 2:
            raise FlySpecPluginException(
                'File %s contains only one data point.'
                % (os.path.basename(filename)))
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
        rb = RawDataBuffer(inc_angle=angles,
                           position=np.array([longitude, latitude, elevation]).T,
                           datetime=unix_times)
        cb = ConcentrationBuffer(gas_species='SO2', value=so2)
        return (rb, cb)

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpec backend')

    @staticmethod
    def get_format():
        return 'flyspec'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
