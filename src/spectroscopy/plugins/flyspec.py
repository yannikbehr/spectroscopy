"""
Plugin to read and write FlySpec data.
"""
import calendar
import datetime
import os

import numpy as np

from spectroscopy.dataset import Dataset, Spectra, ResourceIdentifier, Retrievals
from spectroscopy.plugins import DatasetPluginBase


class FlySpecPluginException(Exception):
    pass


class FlySpecPlugin(DatasetPluginBase):

    def open(self, filename, format=None, timeshift=0.0, **kargs):
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
        # convert decimal seconds to milliseconds
        int_times[:, 6] = (data[:, 6] - int_times[:, 5]) * 1000
        times = [datetime.datetime(*int_times[i, :]) +
                 datetime.timedelta(seconds=ts)
                 for i in range(int_times.shape[0])]
        unix_times = [calendar.timegm(i.utctimetuple()) for i in times]
        latitude = data[:, 8] * data[:, 9]
        longitude = data[:, 10] * data[:, 11]
        elevation = data[:, 12]
        so2 = data[:, 16]
        angles = data[:, 17]
        s = Spectra(self, angle=np.zeros(angles.shape),
                    position=np.zeros((latitude.size, 3)),
                    time=np.zeros(angles.shape))
        slice_start = 0
        slice_end = slice_start
        self.d = Dataset(self, spectra=[s])
        for a in self._split_by_scan(angles, unix_times, longitude,
                                     latitude, elevation, so2):
            slice_end = slice_start + a[0].size
            s.angle[slice_start:slice_end] = a[0]
            s.time[slice_start:slice_end] = a[1]
            position = np.vstack((a[2], a[3], a[4])).T
            s.position[slice_start:slice_end, :] = position
            r = Retrievals(self,
                           spectra_id=ResourceIdentifier(s.resource_id.id),
                           type='FlySpec', gas_species='SO2',
                           slice=slice(slice_start, slice_end), sca=a[5])
            self.d.retrievals.append(r)
            slice_start = slice_end
        # Consistency check to make sure no data was dropped during slicing
        assert s.angle.std() == angles.std()

        return self.d

    def _array_multi_sort(self, *arrays):
        """
        Sorts multiple numpy arrays based on the contents of the first array.

        >>> x1 = np.array([4.,5.,1.,2.])
        >>> x2 = np.array([10.,11.,12.,13.])
        >>> f = FlySpecPlugin()
        >>> f._array_multi_sort(*tuple([x1,x2]))
        (array([ 1.,  2.,  4.,  5.]), array([ 12.,  13.,  10.,  11.]))
        """
        c = np.rec.fromarrays(
            arrays, names=[str(i) for i in range(len(arrays))])
        c.sort()  # sort based on values in first array
        return tuple([c[str(i)] for i in range(len(arrays))])

    def _split_by_scan(self, angles, *vars_):
        """
        Returns an iterator that will split lists/arrays of data by scan (i.e. 
        between start and end angle) an arbitrary number of lists of data can 
        be passed in - the iterator will return a list of arrays of length 
        len(vars_) + 1 with the split angles array at index one, and the 
        remaining data lists in order afterwards. The lists will be sorted 
        into ascending angle order.

        >>> angles = np.array([30, 35, 40, 35, 30, 35, 40])
        >>> f = FlySpecPlugin()
        >>> [a[0] for a in f._split_by_scan(angles)]
        [array([30, 35, 40]), array([30, 35]), array([35, 40])]
        >>> [a[1] for a in f._split_by_scan(angles, np.array([1,2,3,4,5,6,7]))]
        [array([1, 2, 3]), array([5, 4]), array([6, 7])]
        """
        # everything breaks if there are more than two equal angles in a row.
        if np.any(np.logical_and((angles[1:] == angles[:-1])[:-1],
                                 angles[2:] == angles[:-2])):
            idx = np.argmax(np.logical_and(
                (angles[1:] == angles[:-1])[:-1], angles[2:] == angles[:-2]))
            raise ValueError, "Data at line " + str(idx + 2) + \
                " contains three or more repeated angle entries (in a row). \
                Don't know how to split this into scans."

        anglegradient = np.zeros(angles.shape)
        anglegradient[1:] = np.diff(angles)
        # if there are repeated start or end angles, then you end up with zeros
        # in the gradients. Possible zeros at the start need to be dealt with
        # separately, otherwise you end up with the first point being put in a
        # scan of its own.
        if anglegradient[1] == 0:
            anglegradient[1] = anglegradient[2]

        if anglegradient[-1] == 0:
            anglegradient[-1] = anglegradient[-2]

        anglegradient[0] = anglegradient[1]

        # replace zero gradients within the array with the value of its left
        # neighbour
        b = np.roll(anglegradient, 1)
        b[0] = anglegradient[0]
        anglegradient = np.where(np.abs(anglegradient) > 0, anglegradient, b)

        firstarray = anglegradient > 0
        secondarray = np.copy(firstarray)
        secondarray[1:] = secondarray[0:-1]
        secondarray[0] = not secondarray[0]
        inflectionpoints = np.where(firstarray != secondarray)[0]

        if len(inflectionpoints) < 2:
            yield self._array_multi_sort(angles, *vars_)
        else:
            d = [angles[:inflectionpoints[1]]]
            for l in vars_:
                d.append(l[0:inflectionpoints[1]:])
            yield self._array_multi_sort(*tuple(d))

            i = 1
            while i < len(inflectionpoints) - 1:
                if inflectionpoints[i + 1] - inflectionpoints[i] < 2:
                    inflectionpoints[i + 1] = inflectionpoints[i]
                    i += 1
                    continue
                d = [angles[inflectionpoints[i]: inflectionpoints[i + 1]]]
                for l in vars_:
                    d.append(l[inflectionpoints[i]: inflectionpoints[i + 1]])
                i += 1
                yield self._array_multi_sort(*tuple(d))

            # the final point is not an inflection point so now we need to
            # return the final scan
            d = [angles[inflectionpoints[i]:]]
            for l in vars_:
                d.append(l[inflectionpoints[i]:])
            yield self._array_multi_sort(*tuple(d))

    def close(self, filename):
        raise Exception('Close is undefined for the FlySpec backend')

    def get_item(self, path):
        _e = path.split('/')
        id = _e[1]
        name = _e[2]
        ref_o = ResourceIdentifier(id).get_referred_object()
        return ref_o.__dict__[name]

    def set_item(self, path, value):
        _e = path.split('/')
        id = _e[1]
        name = _e[2]
        ref_o = ResourceIdentifier(id).get_referred_object()
        ref_o.__dict__[name] = value

    def create_item(self, path, value):
        pass

    def new(self, filename=None):
        self._root = FlySpecPlugin()

    @staticmethod
    def get_format():
        return 'flyspec'

if __name__ == '__main__':
    import doctest
    doctest.testmod()
