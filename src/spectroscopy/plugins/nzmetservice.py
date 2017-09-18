"""
Plugin to read and write FlySpec data.
"""
import calendar
from collections import defaultdict
from datetime import datetime
import os
import re

import numpy as np
import pyproj
from pytz import timezone

from spectroscopy.datamodel import GasFlowBuffer, MethodBuffer
from spectroscopy.plugins import DatasetPluginBase, DatasetPluginBaseException
from spectroscopy.util import bearing2vec


class NZMetservicePluginException(Exception):
    pass


class NZMetservicePlugin(DatasetPluginBase):


    def __init__(self):
        # Geographic coordinates of volcanoes
        self.volc_dict_keys = ['Auckland', 'Haroharo', 'Mayor Island', 'Ngauruhoe',
                               'Ruapehu', 'Taranaki', 'Tarawera', 'Taupo',
                               'Tongariro', 'White Island']
        self.volc_dict_values = [(174.735, -36.890), (176.466, -38.147),
                                 (176.256, -37.287), (175.632, -39.157),
                                 (175.564, -39.281), (174.064, -39.297),
                                 (176.506, -38.227), (175.978, -38.809),
                                 (175.673, -39.108), (177.183, -37.521)]
        self.volc_dict = {}
        for _k, _v in zip(self.volc_dict_keys, self.volc_dict_values):
            self.volc_dict[_k] = _v

        self.met_models = ['ecmwf', 'gfs', 'ukmo']


    def _parse_model(self, md, ct, lines):
        """
        Parse the forecasts for each volcano.
        """
        # get the times
        times = []
        vals = []
        _a = lines[2].split()
        for _t in _a:
            _d = timezone('Pacific/Auckland').localize(
                datetime.strptime('%d%d%s' % (ct.year, ct.month, _t),
                                  '%Y%m%d%H%M'))
            times.append(_d.astimezone(timezone('UTC')))

        for _l in lines[3:-1]:
            _a = _l.split()
            _h = float(_a[0])
            for _i, _e in enumerate(_a[1:]):
                if _e == '-':
                    continue
                d, s = map(float, _e.split('/'))
                vals.append((times[_i], self.volc_dict[md][0], self.volc_dict[md][1], _h, d, s * 0.514444))
        return vals

    def _readfile(self, filename):
        """
        Read a single forecast file.
        """
        # parse the first 6 lines of any of the files to see which is the
        # preferred model for the day _fns.values()[0]
        with open(filename) as fd:
            _l = fd.readline()
            match = re.search(
                r'(?P<time>\d{2}\:\d{2}\S{2}) (?P<date>\d{2}-\d{2}-\d{4})', _l)
            try:
                ct = datetime.strptime(' '.join((match.group('date'),
                                                 match.group('time'))),
                                       '%d-%m-%Y %I:%M%p')
            except:
                raise NZMetservicePluginException('Unexpected file format on line %s.' % _l)
            # ignore the next two lines
            fd.readline()
            fd.readline()
            # get the model of the day
            _l = fd.readline()
            try:
                _mod = re.match(r'Model of the day is (\S+)', _l).group(1)
                _mod = _mod.lower()
            except:
                raise NZMetservicePluginException('Unexpected file format on line %s.' % _l)
            fd.readline()
            # which model is this
            _l = fd.readline()
            try:
                _mdl = re.match(r'Data for model (\S+)', _l).group(1).lower()
            except:
                raise NZMetservicePluginException(
                    'Unexpected file format on line %s.' % _l)
            # check whether model file is empty in which 
            # case it'll be ignored
            try:
                _mdl = re.match(r'Data for model (\S+) is unavailable.', _l).group(1).lower()
            except:
                # parse the rest of the file
                for _v in self.volc_dict_keys:
                    lines = []
                    for _i in xrange(12):
                        lines.append(fd.readline())
                    try:
                        re.match(r'(^%s\s+)' % _v, lines[0]).group(1)
                    except:
                        raise NZMetservicePluginException(
                            'Expected data for %s but got %s.' %
                            (_v, lines[0].rstrip()))
                    return (_mod, self._parse_model(_v, ct, lines))
            else:
                return (_mod, None)
 

    def read(self, dataset, filename, **kargs):
        
        if not os.path.isfile(filename):
            raise NZMetservicePluginException('File %s does not exist.' %
                                             filename)
        # Construct the filenames for all three model files
        mdl = os.path.basename(filename).split('_')[4]
        _data_dir = os.path.dirname(filename)
        _mdls = defaultdict(list)
        for _mdl in self.met_models:
            _fn = filename.replace(mdl, _mdl)
            if os.path.isfile(_fn):
               _mod, vals = self._readfile(_fn)
               if vals is not None:
                   _mdls[_mdl] = vals
        _mod = kargs.get('preferred_model', _mod)
        if _mod not in _mdls:
            # if data for model of the day is unavailable raise an exception
            raise NZMetservicePluginException(
                'Data for model %s is unavailable.' % _mod)

        npts = len(_mdls[_mod])
        g = pyproj.Geod(ellps='WGS84')
        vx = np.zeros(npts)
        vy = np.zeros(npts)
        vz = np.zeros(npts)
        position = np.zeros((npts, 3))
        time = np.empty(npts,dtype='S26')
        for _i, _e in enumerate(_mdls[_mod]):
            t, lon, lat, h, d, s = _e
            _vx, _vy = bearing2vec(d, s)
            vx[_i] = _vx
            vy[_i] = _vy
            vz[_i] = np.nan
            position[_i, :] = lon, lat, h
            time[_i] = t.isoformat()
        description = 'Wind measurements and forecasts by NZ metservice \
        for selected sites.'
        mb = MethodBuffer(name=_mod)
        m = dataset.new(mb)
        gfb = GasFlowBuffer(methods=[m], vx=vx, vy=vy, vz=vz,
                            position=position, datetime=time, 
                            user_notes=description, unit='m/s')
        gf = dataset.new(gfb)

    def close(self, filename):
        raise Exception(
            'Close is undefined for the NZMetservicePlugin backend')

    @staticmethod
    def get_format():
        return 'nzmetservice'
