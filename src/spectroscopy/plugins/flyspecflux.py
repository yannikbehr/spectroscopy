"""
Plugin to read flux estimates from Nial Peter's code.
"""
import calendar
import datetime
import os
import struct

import numpy as np

from spectroscopy.datamodel import FluxBuffer, MethodBuffer 
from spectroscopy.plugins import DatasetPluginBase

class FlySpecFluxPluginException(Exception):
    pass


class FlySpecFluxPlugin(DatasetPluginBase):

    def read(self, dataset, filename,  **kargs):
        """
        Read flux estimates.
        """
        data = np.fromregex(filename, r'(\S+ \S+)\s+(-?\d+\.\d+)',
                            dtype={'names': ('datetime','flux'),
                                   'formats':('S26',np.float)})
        dt = data['datetime'].astype('datetime64[us]')
        # convert milliseconds to microseconds to fix a bug in Nial's code
        us = dt - dt.astype('datetime64[s]')
        dtn = dt.astype('datetime64[s]') + us*1000
        f = data['flux']
       
        mb = MethodBuffer(name='GNS FlySpec UI')
        fb = FluxBuffer(value=f, datetime=dtn.astype(str))
        return {str(fb):fb, str(mb):mb}

    @staticmethod
    def get_format():
        return 'flyspecflux'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
