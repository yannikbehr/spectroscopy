# Copyright (C) Nial Peters 2015
#
# This file is part of gns_flyspec.
#
# gns_flyspec is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# gns_flyspec is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with gns_flyspec.  If not, see <http://www.gnu.org/licenses/>.
import numpy
import datetime
import math
import json
import calendar
import Queue
import threading

import dir_iter
import bkgd_subtract

def date2secs(d):
    return calendar.timegm(d.timetuple()) + d.microsecond / 1e6

def bearing2vec(bearing):
    """
    Returns an [x,y] array representing a unit vector along the bearing given.
    Bearing in this sense refers to angle clockwise from the direction [0, 1].
    So, for example: bearing2vec(90) -> [1, 0]
    """
    if bearing >= 180:
        x_sign = -1
        if bearing < 270:
            y_sign = -1
            bearing -= 180
        else:
            y_sign = 1
            bearing = 360 - bearing
    elif bearing > 90:
        y_sign = -1
        x_sign = 1
        bearing = bearing - 90
    else:
        y_sign = 1
        x_sign = 1

    assert bearing <= 90

    y = y_sign * math.cos(math.radians(bearing))
    x = x_sign * math.sin(math.radians(bearing))

    x /= math.sqrt(x ** 2 + y ** 2)
    y /= math.sqrt(x ** 2 + y ** 2)

    return numpy.array([x, y])


def intercept(pt1, bearing1, pt2, bearing2):
    """
    Computes the intercept point of two lines described by a point on the line
    and a bearing along the line. Returns the [x, y] coordinate of the intercept
    or None if the lines do not cross.
    """

    # convert to numpy arrays
    pt1 = numpy.array(pt1)
    pt2 = numpy.array(pt2)

    # first we convert the bearings to unit vectors along the lines
    vec1 = bearing2vec(bearing1)
    vec2 = bearing2vec(bearing2)

    denom = numpy.cross(vec2, vec1)

    if denom == 0:
        return None, None, None

    # b = distance along vec2 of the intercept
    # see http://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    b = numpy.cross(pt1 - pt2, vec1) / denom

    a = (b * (vec2 / vec1) - ((pt1 - pt2) / vec1))[0]
    intercept_pt = pt2 + b * vec2

    return intercept_pt, a, b


def get_plume_dist_and_angle(wind_dir, scanner_config, config):

        # convert wind dir to bearing along a wind vector (i.e. direction wind
        # blows from source.)
        if wind_dir > 180:
            wind_dir -= 180
        else:
            wind_dir += 180

        # the x,y position on the ground where the plume crosses the scan line
        # print config["vent_location"]
        point_on_ground, a, b = intercept(config["vent_location"],
                                               wind_dir,
                                               scanner_config["scanner_location"],
                                               scanner_config["scan_plane_bearing"]
                                               )

        if a < 0:
            # intercept is upwind of vent - not possible
            return None, None

        # print "point_on_ground = ",point_on_ground
        if point_on_ground is None:
            return None, None

        # find the straight line distance (3d) to the plume
        dist_to_intercept = math.sqrt(numpy.sum((point_on_ground - scanner_config["scanner_location"]) ** 2))
        # print "dist_to_intercept = ",dist_to_intercept
        dist_to_plume = math.sqrt(dist_to_intercept ** 2 + scanner_config["default_plume_height"] ** 2)

        angle_above_horiz = math.degrees(math.atan(scanner_config["default_plume_height"] / dist_to_intercept))

        if b > 0:
            scan_angle = 180 - angle_above_horiz
        else:
            scan_angle = angle_above_horiz

        return dist_to_plume, scan_angle


class Scan:
    def __init__(self, angles, times, so2, saturated, wind_dir, wind_speed, scanner_config, config):
        """
        Container class to hold the data from a single scan.

        Note that scanner_config should be just the scanner config dict, not
        the entire contents of the config file i.e. config["scanner1"]

        """
        self._is_processed = False
        self.__angles = numpy.array(angles)
        self.__times = numpy.array(times)
        self.__so2 = numpy.array(so2)
        self.is_saturated = saturated
        self.config = config
        self.scanner_config = scanner_config

        self._dist_to_plume, self._plume_pos_guess = get_plume_dist_and_angle(wind_dir, scanner_config, config)

        assert (self._dist_to_plume > 0) or (self._dist_to_plume is None)
        assert (self._plume_pos_guess > 0) or (self._plume_pos_guess is None)

        t_angle = abs(wind_dir - scanner_config["scan_plane_bearing"])
        if t_angle > 180:
            t_angle -= 180
        if t_angle > 90:
            t_angle -= 90

        self._transect_angle = t_angle  # angle that the scan plane transects the plume

        self._out_of_scan_range = False

        if self._plume_pos_guess > scanner_config["scan_angles"][1] or self._plume_pos_guess < scanner_config["scan_angles"][0]:
            self._out_of_scan_range = True

        self.wind_speed = wind_speed

        self._ica = None  # integrated column amount - not the same as flux!

        self.g_fit_params = None  # Gaussian fit parameters


    def toJSON(self):
        """
        Returns a JSON string representation of the scan
        """
        if self.g_fit_params is None:
            g_fit = None
        else:
            g_fit = list(self.g_fit_params)

        dict_ = {
                "config":self.config,
                "scanner_config":self.scanner_config,
                "angles":list(self.__angles),
                "times":[date2secs(t) for t in self.__times],
                "so2":list(self.__so2),
                "saturated":int(self.is_saturated),
                "ica":self._ica,
                "g_fit_params":g_fit,
                "is_processed":int(self._is_processed),
                "transect_angle":self._transect_angle,
                "dist_to_plume":self._dist_to_plume,
                "wind_speed":self.wind_speed
                }
        return json.dumps(dict_)

    @staticmethod
    def fromJSON(dict_):
        # dict_ is the decoded dict object, not the json string

        times = [datetime.datetime.fromtimestamp(t) for t in dict_["times"]]

        scan = Scan(dict_["angles"], times, dict_["so2"], bool(dict_["saturated"]), 0, 0, dict_["scanner_config"], dict_["config"])

        scan._ica = dict_["ica"]
        if dict_["g_fit_params"] is None:
            scan.g_fit_params = None
        else:
            scan.g_fit_params = bkgd_subtract.GaussianParameters(*dict_["g_fit_params"])
        scan._is_processed = bool(dict_["is_processed"])
        scan._transect_angle = dict_["transect_angle"]
        scan._dist_to_plume = dict_["dist_to_plume"]
        scan.wind_speed = dict_["wind_speed"]

        return scan

    @property
    def transect_angle(self):
        return self._transect_angle

    @transect_angle.setter
    def transect_angle(self, value):
        self._is_processed = False
        self._transect_angle = value

    @property
    def angles(self):
        return self.__angles

    @angles.setter
    def angles(self, value):
        self._is_processed = False
        self.__angles = numpy.array(value)

    @property
    def times(self):
        return self.__times

    @times.setter
    def times(self, value):
        self._is_processed = False
        self.__times = numpy.array(value)

    @property
    def col_amounts(self):
        return self.__so2

    @col_amounts.setter
    def col_amounts(self, value):
        self._is_processed = False
        self.__so2 = numpy.array(value)


    def plot_bkgd_fit(self, mpl_subplot, style='b.'):
        if not self._is_processed:
            self.get_ica()

        mpl_subplot.plot(self.__angles, self.__so2, style)
        mpl_subplot.set_xlabel("Scan angle (degrees)")
        mpl_subplot.set_ylabel("SO$_2$ (ppmm)")
        mpl_subplot.set_title(self.__times[0].strftime("%H:%M:%S"))
        if self.g_fit_params is not None:
            g_fit_func = bkgd_subtract.gaussian_func(self.g_fit_params)
            mpl_subplot.plot(self.__angles, g_fit_func(self.__angles), 'm-', linewidth=2)


    def _scan_looks_good(self, g_fit_params):
        """
        This method runs a selection of tests on the supplied Gaussian fit parameters
        to see if the scan looks like it actually captured the plume or not.
        """

        # see if the size of the peak is greater than the noise
        if g_fit_params.amplitude < 1.5 * numpy.std(self.__so2):
            return False

        # see if the peak is exceptionally large - possibly due to saturation
        if g_fit_params.amplitude > 1000.0:
            return False


        # see if the width of the peak is exceptionally large
        angle_range = self.__angles[-1] - self.__angles[0]
        if abs(g_fit_params.sigma) > angle_range / 4.0:
            return False

        # see if the peak of the gaussian is within the scan range
        if (g_fit_params.mean < (self.__angles[0] + abs(g_fit_params.sigma) * 2) or
            g_fit_params.mean > (self.__angles[-1] - abs(g_fit_params.sigma) * 2)):
            return False

        return True


    def get_ica(self):
        """
        Computes and returns the integrated column amount of SO2. The computed
        value is cached so that future calls do not require recalculation.
        """
        if self.is_saturated:
            self._ica = 0.0
            self._is_processed = True
            return self._ica

        if self._out_of_scan_range:
            self._ica = 0.0
            self._is_processed = True
            return self._ica

        if not self._is_processed:

            # calculate the background level
            try:
                g_fit_params = bkgd_subtract.fit_gaussian(self.__angles, self.__so2, mean_guess=self._plume_pos_guess)
                self.g_fit_params = g_fit_params
            except bkgd_subtract.FittingError:
                self._ica = 0.0
                self._is_processed = True
                return self._ica

            self.__g_fit_func = bkgd_subtract.gaussian_func(g_fit_params)

            bkgd = numpy.ones_like(self.__so2) * g_fit_params.y_offset

            # subtract the background from the points
            bkgd_subtracted_so2 = self.__so2 - bkgd

            # correct for a non-perpendicular transect through the plume
            bkgd_subtracted_so2 *= math.cos(math.radians(self._transect_angle))

            # calculate distance between measurements assuming dx = r * theta
            d_theta = numpy.radians(numpy.abs(self.__angles[1:] - self.__angles[:-1]))
            dx = self._dist_to_plume * d_theta

            col_amt = dx * ((bkgd_subtracted_so2[:-1] + bkgd_subtracted_so2[1:]) / 2.0)

            self._ica = numpy.sum(col_amt)

            self._is_processed = True

        if self.g_fit_params is None or not self._scan_looks_good(self.g_fit_params):
            self._ica = 0.0
            self._is_processed = True

        return self._ica


    def get_flux(self):
        """
        returns flux in kg/s (assuming wind speeds are in m/s)
        """
        # conversion factor from ppmm to kg/s
        ppmm_to_kgs = 2.660e-06
        # conversion factor from kg/s to t/day (added by Agnes Mazot:04/09/2015)
        # kgs_to_tday = 86.4

        return self.get_ica() * self.wind_speed * ppmm_to_kgs


class ScanIter:
    def __init__(self, wind_data, scanner_config, config, *args, **kwargs):
        self._stay_alive = True
        self._obj_q = Queue.Queue()
        self.wind_data = wind_data
        self.scanner_config = scanner_config
        self.config = config
        if isinstance(args[0], basestring):
            self.file_iter = dir_iter.ListDirIter(*args, **kwargs)
        else:
            assert kwargs == {}, "No keyword args supported if passing in list of files"
            self.file_iter = sorted(args[0])

        self._worker_thread = threading.Thread(target=self.__load_scans)
        self._worker_thread.start()


    def close(self):
        self._stay_alive = False
        try:
            self.file_iter.close()
        except AttributeError:
            pass
        self._obj_q.put(None)

        self._worker_thread.join()


    def __iter__(self):
        """
        Method required by iterator protocol. Allows iterator to be used in
        for loops.
        """
        return self

    def __next__(self):
        # needed for Py3k compatibility
        return self.next()


    def next(self):

        s = self._obj_q.get(block=True)

        if s is None or not self._stay_alive:

            raise StopIteration

        return s


    def __load_scans(self):
        # call superclass next() method to retrieve a filename (raises StopIteration
        # if there are no more files left - ending the iteration)
        part_scan = None
        for filename in self.file_iter:
            print "Loading data file: %s" % filename
            scans = [i for i in split_into_scans(self.wind_data, self.scanner_config, self.config, filename, part=part_scan)]
            if not scans:
                continue
            part_scan = scans.pop()
            for scan in scans:
                if len(scan.angles) > 5:  # requirement for fitting
                    self._obj_q.put(scan)
        if part_scan is not None and len(part_scan.angles) > 5:
            self._obj_q.put(part_scan)

        # FIXME - does this line break realtime processing?
        self._obj_q.put(None)




def split_into_scans(wind_data, scanner_config, config, filename, part=None):

    # load the file using numpy, specifying which columns to use (can't just
    # load all of them since they are mixed data types)
    try:
        data = numpy.loadtxt(filename, "float", usecols=(1, 2, 3, 4, 5, 6, 13, 16, 17))
    except:
        print "Failed to load data from %s" % filename
        raise StopIteration

    if len(data.shape) < 2:
        raise StopIteration
    # convert times to datetime objects
    # need integers for datetimes
    int_times = numpy.zeros(data[:, :7].shape, dtype='int')
    int_times[:, :6] = data[:, :6]
    int_times[:, 6] = (data[:, 5] - int_times[:, 5]) * 1000  # convert decimal seconds to milliseconds
    try:
        times = [datetime.datetime(*int_times[i, :]) for i in range(int_times.shape[0])]
    except:
        print data[i, :]
        print int_times[i, :]

    saturated_pix = data[:, 6]
    so2 = data[:, 7]
    angles = data[:, 8]

    if part is not None:
        angles = numpy.concatenate((part.angles, angles))
        so2 = numpy.concatenate((part.col_amounts, so2))
        times = numpy.concatenate((part.times, times))
        saturated_pix = numpy.concatenate((numpy.zeros_like(part.times) + part.is_saturated, saturated_pix))

    # split the data into individual scans
    split_data = split_by_scan(angles, times, so2, saturated_pix)

    # encapsulate the data into Scan objects and return them one by one
    for d in split_data:
        saturated = numpy.any(d[3] > 0)
        mean_time = d[1][0] + ((d[1][-1] - d[1][0]) / 2)
        wind_dir, wind_speed = wind_data.get_direction_and_speed(mean_time)
        yield Scan(d[0], d[1], d[2], saturated, wind_dir, wind_speed, scanner_config, config)



# function taken from std_ops.iter_ module
def array_multi_sort(*arrays):
    """
    Sorts multiple numpy arrays based on the contents of the first array.
    """
    c = numpy.rec.fromarrays(arrays, names=[str(i) for i in range(len(arrays))])
    c.sort()  # sort based on values in first array
    return tuple([c[str(i)] for i in range(len(arrays))])



# function taken from AvoScan software
def split_by_scan(angles, *vars_):
    """
    returns an iterator that will split lists/arrays of data by scan (i.e. between start and end angle)
    an arbitrary number of lists of data can be passed in - the iterator will return a list of arrays
    of length len(vars_) + 1 with the split angles array at index one, and the remaining data lists
    in order afterwards. The lists will be sorted into ascending angle order.


    >>> angles = numpy.array([30, 35, 40, 35, 30, 35, 40])
    >>> for a in split_by_scan(angles):
            print a[0]
    [30, 35]
    [30, 35, 40]
    [35, 40]
    >>> for a in split_by_scan(angles, numpy.array([1,2,3,4,5,6,7])):
            print a[1]
    [1, 2]
    [5, 4, 3]
    [6, 7]


    """

    # everything breaks if there are more than two equal angles in a row.
    if numpy.any(numpy.logical_and((angles[1:] == angles[:-1])[:-1], angles[2:] == angles[:-2])):
        idx = numpy.argmax(numpy.logical_and((angles[1:] == angles[:-1])[:-1], angles[2:] == angles[:-2]))
        raise ValueError, "Data at line " + str(idx + 2) + " contains three or more repeated angle entries (in a row). Don't know how to split this into scans."

    anglegradient = numpy.gradient(angles)

    # if there are repeated start or end angles, then you end up with zeros in the gradients.
    # possible zeros at the start need to be dealt with separately, otherwise you end up with
    # the first point being put in a scan of its own.
    if anglegradient[0] == 0:
        anglegradient[0] = anglegradient[1]

    if anglegradient[-1] == 0:
        anglegradient[-1] = anglegradient[-2]

    firstarray = anglegradient > 0
    secondarray = numpy.copy(firstarray)
    secondarray[1 :] = secondarray[0 :-1]
    secondarray[0] = not secondarray[0]
    inflectionpoints = numpy.where(firstarray != secondarray)[0]

    if len(inflectionpoints) < 2:
        yield array_multi_sort(angles, *vars_)

    else:

        d = [angles[:inflectionpoints [1]]]
        for l in vars_:
            d.append(l[0:inflectionpoints [1]:])
        yield array_multi_sort(*tuple(d))

        for i in range (1, len(inflectionpoints) - 1):
            if inflectionpoints [i + 1] == inflectionpoints[i] + 2:
                continue
            d = [angles[inflectionpoints[i] - 1 : inflectionpoints [i + 1] + 1]]
            for l in vars_:
                d.append(l[inflectionpoints[i] - 1 : inflectionpoints [i + 1] + 1])
            yield array_multi_sort(*tuple(d))

        # the final point is not an inflection point so now we need to return the final scan
        d = [angles[inflectionpoints [i + 1]:]]
        for l in vars_:
            d.append(l[inflectionpoints [i + 1]:])
        yield array_multi_sort(*tuple(d))

