import numpy as np

from spectroscopy.datamodel import GasFlowBuffer, MethodBuffer
from spectroscopy.plugins import DatasetPluginBase, DatasetPluginBaseException
from spectroscopy.util import bearing2vec


class FlySpecWindPlugin(DatasetPluginBase):
    
    def read(self, dataset, filename, **kargs):
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
        description = 'Wind measurements and forecasts by NZ metservice \
        for Te Maari.'
        mb = MethodBuffer(name='some model')
        m = dataset.new(mb)
        gfb = GasFlowBuffer(methods=[m], vx=vx, vy=vy, vz=vz,
                            position=position, datetime=time, 
                            user_notes=description, unit='m/s')
        gf = dataset.new(gfb)
        return gf

    @staticmethod
    def get_format():
        return 'flyspecwind'

