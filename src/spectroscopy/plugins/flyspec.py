"""
Plugin to read and write FlySpec data.
"""
import calendar
import datetime

import numpy as np

from spectroscopy.dataset import Dataset, Spectra
from spectroscopy.plugins import DatasetPluginBase


class FlySpecPlugin(DatasetPluginBase):
    
    def __init__(self):
        self.__root = self
        super(FlySpecPlugin,self).__init__()
        
    def open(self, filename, format=None):
        # load data and convert southern hemisphere to negative 
        # latitudes and western hemisphere to negative longitudes
        data = np.loadtxt(filename,usecols=range(0,21),
                          converters={9:lambda x: -1.0 if x.lower() == 's' else 1.0,
                                      11:lambda x: -1.0 if x.lower() == 'w' else 1.0})
        int_times = np.zeros(data[:, 1:7].shape, dtype='int')
        int_times[:, :6] = data[:, 1:7]
        int_times[:, 5] = (data[:, 6] - int_times[:, 5]) * 1000  # convert decimal seconds to milliseconds
        times = [datetime.datetime(*int_times[i, :]) for i in range(int_times.shape[0])]
        unix_times = [calendar.timegm(i.utctimetuple()) for i in times]
        data[:,8] *= data[:,9]
        data[:,10] *= data[:,11]
        s = Spectra(self,time=np.array(unix_times),angle=data[:,17])
        self.d = Dataset(self, spectra=[s.resource_id.id]) 
        return self.d
                
    def close(self, filename):
        raise Exception('Close is undefined for the RAM backend')
    
    def get_item(self,path):
        pass
    
    def set_item(self,path,value):
        pass
    
    def create_item(self,path,value):
        pass

    @staticmethod
    def get_format():
        return 'flyspec'