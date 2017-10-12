"""
Plugin to read FlySpec reference spectra.
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

class FlySpecRefPluginException(Exception):
    pass


class FlySpecRefPlugin(DatasetPluginBase):

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

    def read(self, dataset, filename,  **kargs):
        """
        Read reference spectra for FlySpec.
        """
        try:
            wavelengths = kargs['wavelengths']
            mtype = kargs['type']
        except KeyError:
            raise FlySpecRefPluginException('Please provide wavelengths and measurement type.')
        
        spectra = np.array(self._read_spectra(filename))
        if spectra.shape[1] != wavelengths.size:
            raise FlySpecRefPluginException("Spectra and wavelengths don't have the same size.")
        rb = RawDataBuffer(ind_var=wavelengths, d_var=spectra)
        rdtb = RawDataTypeBuffer(d_var_unit='ppm m', ind_var_unit='nm', name=mtype)
        return {str(rb):rb, str(rdtb):rdtb}

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpecRef backend')

    @staticmethod
    def get_format():
        return 'flyspecref'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
