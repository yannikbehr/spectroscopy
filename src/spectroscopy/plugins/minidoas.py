"""
Plugin to read MiniDOAS data.
"""
import codecs
import datetime
import io

import numpy as np
import pyproj

from spectroscopy.datamodel import (RawDataBuffer,
                                    ConcentrationBuffer,
                                    RawDataTypeBuffer,
                                    FluxBuffer,
                                    MethodBuffer,
                                    GasFlowBuffer)
from spectroscopy.plugins import DatasetPluginBase, DatasetPluginBaseException
from spectroscopy.util import bearing2vec


def find_bad_lines(filename, **loadtxt_kargs):
    """
    Find lines that are not properly formatted and
    return a clean buffer.
    """
    master = io.StringIO()
    with open(filename) as fh:
        lines = fh.readlines()
        for _l in lines[1:]:
            fs = io.StringIO()
            fs.write(str(_l))
            fs.seek(0)
            try:
                np.loadtxt(fs, **loadtxt_kargs)
            except ValueError:
                pass
            else:
                master.write(str(_l))
    master.seek(0)
    return master


class MiniDoasException(DatasetPluginBaseException):
    pass


class MiniDoasRaw(DatasetPluginBase):

    def read(self, dataset, filename, timeshift=0, **kargs):

        fh = codecs.open(filename, encoding='utf-8-sig', errors='ignore')
        for line in fh.readlines():
            a = line.encode('utf-8')
            if a == b'\x00'*len(a):
                fh.close()
                msg = "File {} contains line of binary 0's"
                raise MiniDoasException(msg.format(filename))
        fh.seek(0)
        dt = np.dtype([('station', 'S2'), ('date', 'S10'), ('time', np.float),
                       ('stept', np.int), ('angle', np.float),
                       ('intt', np.int), ('nspec', np.int),
                       ('specin', np.float), ('counts', np.int, (482,))])

        def date_converter(x):
            y = x.decode('ascii')
            return '%s-%s-%s' % (y[0:4], y[4:6], y[6:8])

        data = np.loadtxt(fh, converters={1: date_converter},
                          dtype=dt, delimiter=',', ndmin=1)
        fh.close()
        # Construct datetimes
        date = data['date'].astype('datetime64')
        hours = (data['time']/3600.).astype(int)
        minutes = ((data['time'] - hours*3600.)/60.).astype(int)
        fseconds = data['time'] - hours*3600. - minutes*60.
        iseconds = (fseconds).astype(int)
        mseconds = np.round((fseconds - iseconds), 3)*1e3
        datetime = date + hours.astype('timedelta64[h]') \
            + minutes.astype('timedelta64[m]') \
            + iseconds.astype('timedelta64[s]') \
            + mseconds.astype('timedelta64[ms]')
        datetime -= np.timedelta64(int(timeshift), 'h')

        # Convert radians to decimal degrees
        angles = data['angle']*360./(2.*np.pi)
        rdtb = RawDataTypeBuffer(d_var_unit='ppm-m',
                                 ind_var_unit='nm',
                                 name='measurement',
                                 acquisition='stationary')
        wavelengths = np.arange(30, 512)
        rb = RawDataBuffer(inc_angle=angles,
                           datetime=datetime.astype(str),
                           ind_var=wavelengths,
                           d_var=data['counts'],
                           integration_time=data['intt'])

        return {str(rb): rb, str(rdtb): rdtb}

    @staticmethod
    def get_format():
        return 'minidoas-raw'


class MiniDoasSpectra(DatasetPluginBase):

    def read(self, dataset, filename, timeshift=0, **kargs):
        try:
            date = kargs['date']
        except KeyError:
            raise MiniDoasException("'date' unspecified.")

        dt = np.dtype([('datetime', 'S26'), ('angle', np.float),
                       ('value', np.float), ('model_value', np.float),
                       ('fitcoeff', np.float), ('fitcoeff_err', np.float),
                       ('fitshift', np.float), ('fitshift_err', np.float),
                       ('fitsqueeze', np.float), ('fitsqueezeerr', np.float)])
        data = np.loadtxt(filename, delimiter=',', skiprows=1,
                          converters={0: lambda x: date+'T'+x.decode('ascii')},
                          dtype=dt)
        dtm = data['datetime'].astype('datetime64[ms]')
        dtm -= np.timedelta64(int(timeshift), 'h')
        if kargs.get('model', False):
            c = data['model_value']
        else:
            c = data['value']
        cb = ConcentrationBuffer(value=c,
                                 datetime=dtm.astype(str),
                                 gas_species='SO2',
                                 unit='ppm-m')
        return {str(cb): cb}

    @staticmethod
    def get_format():
        return 'minidoas-spectra'


class MiniDoasScan(DatasetPluginBase):
    def __init__(self):
        # setup projection from NZGD49 (NZMG) to WGS84
        self.destp = pyproj.Proj(init="epsg:4326")
        self.srcp = pyproj.Proj(init="epsg:27200")

    def _plumegeometry2gasflow(self, ws, pheight, pwidth, peasting, pnorthing,
                               ptrack, datetime):
        position = []
        time = []
        vx = []
        vy = []
        vz = []
        for _ws, h, w, e, n, t, dt in zip(ws, pheight, pwidth, peasting,
                                          pnorthing, ptrack, datetime):
            lon, lat = pyproj.transform(self.srcp, self.destp, e, n)
            position.append([lon, lat, h-w/2.])
            position.append([lon, lat, h])
            position.append([lon, lat, h+w/2.])
            time.extend([dt]*3)
            x, y = bearing2vec(t, _ws)
            vx.extend([x]*3)
            vy.extend([y]*3)
            vz.extend([np.nan]*3)
        description = 'Plume velocity inferred from plume geometry'
        description += 'and wind speed'
        mb = MethodBuffer(name='WS2PV', description=description)
        gfb = GasFlowBuffer(vx=vx, vy=vy, vz=vz,
                            position=position,
                            datetime=np.array(time).astype(str),
                            unit='m/s')
        return (mb, gfb)

    def read(self, dataset, filename, timeshift=0, **kargs):
        try:
            date = kargs['date']
        except KeyError:
            raise MiniDoasException("'date' unspecified.")

        station = kargs.get('station', None)

        def dateconverter(x):
            return date+'T'+x.decode('ascii')

        dt = np.dtype([('time', 'U19'), ('ws', np.float), ('wd', np.float),
                       ('R2', np.float), ('SO2Start', np.float),
                       ('SO2Max', np.float), ('SO2End', np.float),
                       ('PlumeRange', np.float), ('PlumeWidth', np.float),
                       ('PlumeHeight', np.float), ('Easting', np.float),
                       ('Northing', np.float), ('Track', np.float),
                       ('Emission', np.float), ('Station', 'U2'),
                       ('EmissionSE', np.float)])
        try:
            data = np.loadtxt(filename, delimiter=',', skiprows=1, dtype=dt,
                              converters={0: dateconverter},
                              ndmin=1)
        except ValueError:
            buf = find_bad_lines(filename, delimiter=',', dtype=dt,
                                 converters={0: dateconverter},
                                 ndmin=1)
            data = np.loadtxt(buf, delimiter=',', dtype=dt,
                              converters={0: dateconverter},
                              ndmin=1)

        if station is not None:
            idx = np.where(data['Station'] == station)
        else:
            idx = np.arange(data.shape[0])
        dtm = data['time'][idx].astype('datetime64[s]')
        dtm -= np.timedelta64(int(timeshift), 'h')
        fb = FluxBuffer(value=data['Emission'][idx]/86.4,
                        value_error=data['EmissionSE'][idx]/86.4,
                        datetime=dtm.astype(str))
        mb, gfb = self._plumegeometry2gasflow(data['ws'][idx],
                                              data['PlumeHeight'][idx],
                                              data['PlumeWidth'][idx],
                                              data['Easting'][idx],
                                              data['Northing'][idx],
                                              data['Track'][idx],
                                              dtm)
        return {str(fb): fb, str(mb): mb, str(gfb): gfb}

    @staticmethod
    def get_format():
        return 'minidoas-scan'


class MiniDoasWind(DatasetPluginBase):

    def read(self, dataset, filename, timeshift=0, **kargs):
        try:
            fn_wd = filename['direction']
            fn_ws = filename['speed']
        except KeyError:
            msg = "Plugin expects as filename a dictionary of the form:"
            msg += "{'direction': /path/to/wind-direction-file, "
            msg += "'speed': /path/to/wind-speed-file}"
            raise MiniDoasException(msg)

        def dateconverter(x):
            return (datetime.datetime.strptime(x.decode('ascii'),
                                               '%d/%m/%Y %H:%M:%S')
                    .strftime('%Y-%m-%dT%H:%M:%S'))

        try:
            dtp1 = np.dtype([('datetime', 'S19'), ('direction', np.float)])
            data1 = np.loadtxt(fn_wd, skiprows=1, dtype=dtp1, delimiter='\t',
                               converters={0: dateconverter}, ndmin=1)
        except ValueError:
            buf = find_bad_lines(fn_wd, dtype=dtp1, delimiter='\t',
                                 converters={0: dateconverter}, ndmin=1)
            data1 = np.loadtxt(buf, dtype=dtp1, delimiter='\t',
                               converters={0: dateconverter}, ndmin=1)

        try:
            dtp2 = np.dtype([('datetime', 'S19'), ('speed', np.float)])
            data2 = np.loadtxt(fn_ws, skiprows=1, dtype=dtp2, delimiter='\t',
                               converters={0: dateconverter}, ndmin=1)
        except ValueError:
            buf = find_bad_lines(fn_ws, dtype=dtp2, delimiter='\t',
                                 converters={0: dateconverter}, ndmin=1)
            data2 = np.loadtxt(buf, dtype=dtp2, delimiter='\t',
                               converters={0: dateconverter}, ndmin=1)

        # Some of the wind data files have been created
        # by hand because the weather station didn't
        # send any data. They then may have different
        # number of entries, hence we have to find the
        # matching times
        dates = []
        vx = []
        vy = []
        vz = []
        for i, _d in enumerate(data1['datetime'].astype("datetime64[s]")):
            tdiff = np.abs(_d - data2['datetime'].astype("datetime64[s]"))
            idx = np.argmin(tdiff)
            if tdiff.astype('int').min() > 1:
                continue
            wd = data1['direction'][i]
            ws = data2['speed'][idx]
            # if windspeed is 0 give it a tiny value
            # so that the bearing can be reconstructed
            if ws == 0.:
                ws = 0.0001
            _vx, _vy = bearing2vec(wd, ws)
            vx.append(_vx)
            vy.append(_vy)
            vz.append(np.nan)
            dates.append(_d)
        dtm = np.array(dates, dtype="datetime64[s]")
        dtm -= np.timedelta64(int(timeshift), 'h')
        description = 'Autonomous weather station operated by NZ metservice'
        mb = MethodBuffer(name='AWS', description=description)
        m = dataset.new(mb)
        gfb = GasFlowBuffer(methods=[m], vx=vx, vy=vy, vz=vz,
                            datetime=dtm.astype(str), unit='m/s')
        return {str(gfb): gfb}

    @staticmethod
    def get_format():
        return 'minidoas-wind'
