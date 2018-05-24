import calendar
import datetime
import math

import numpy as np


def bearing2vec(bearing, norm=1.0):
    """
    Returns an [x,y] array representing by default a unit vector along the
    bearing given unless norm is not 1.0. Bearing in this sense refers to angle
    clockwise from the direction [0, 1].
    So, for example: bearing2vec(90) -> [1, 0]

    >>> bearing2vec(90)
    array([  1.00000000e+00,   6.12323400e-17])
    >>> bearing2vec(45)
    array([ 0.70710678,  0.70710678])
    >>> bearing2vec(30,3.0)
    array([ 1.5       ,  2.59807621])
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

    y = y_sign * math.cos(math.radians(bearing)) * norm
    x = x_sign * math.sin(math.radians(bearing)) * norm

    return np.array([x, y])


def vec2bearing(vx, vy):
    """
    Compute the angle clockwise from the direction [0, 1] from the given x and
    y components of a vector.

    >>> vec2bearing(1,1) # doctest: +ELLIPSIS
    45.0...
    >>> vec2bearing(-1,-1) # doctest: +ELLIPSIS
    225.0...
    >>> vec2bearing(-2,3) # doctest: +ELLIPSIS
    326.30...
    """

    x = math.sqrt(vx * vx + vy * vy)
    phi = math.degrees(math.acos(abs(vy) / x))
    bearing = phi
    if vy < 0 and vx > 0:
        bearing = 90.0 + phi
    elif vy < 0 and vx <= 0:
        bearing = 180.0 + phi
    elif vy >= 0 and vx < 0:
        bearing = 360.0 - phi
    return bearing


def parse_iso_8601(value):
    """
    Parses an ISO8601:2004 date time string and returns a datetime object in
    UTC.

    >>> d = parse_iso_8601('2016-09-26T23:45:43+12:00')
    >>> d.isoformat()
    '2016-09-26T11:45:43'
    >>> d = parse_iso_8601('2016-09-26T23:45:43.001Z')
    >>> d.isoformat()
    '2016-09-26T23:45:43.001000'
    """
    # remove trailing 'Z'
    value = value.replace('Z', '')
    # split between date and time
    try:
        (date, time) = value.split("T")
    except:
        date = value
        time = ""
    # remove all hyphens in date
    date = date.replace('-', '')
    # remove colons in time
    time = time.replace(':', '')
    # guess date pattern
    length_date = len(date)
    if date.count('W') == 1 and length_date == 8:
        # we got a week date: YYYYWwwD
        # remove week indicator 'W'
        date = date.replace('W', '')
        date_pattern = "%Y%W%w"
        year = int(date[0:4])
        # [Www] is the week number prefixed by the letter 'W', from W01
        # through W53.
        # strpftime %W == Week number of the year (Monday as the first day
        # of the week) as a decimal number [00,53]. All days in a new year
        # preceding the first Monday are considered to be in week 0.
        week = int(date[4:6]) - 1
        # [D] is the weekday number, from 1 through 7, beginning with
        # Monday and ending with Sunday.
        # strpftime %w == Weekday as a decimal number [0(Sunday),6]
        day = int(date[6])
        if day == 7:
            day = 0
        date = "%04d%02d%1d" % (year, week, day)
    elif length_date == 7 and date.isdigit() and value.count('-') != 2:
        # we got a ordinal date: YYYYDDD
        date_pattern = "%Y%j"
    elif length_date == 8 and date.isdigit():
        # we got a calendar date: YYYYMMDD
        date_pattern = "%Y%m%d"
    else:
        raise ValueError("Wrong or incomplete ISO8601:2004 date format")
    # check for time zone information
    # note that the zone designator is the actual offset from UTC and
    # does not include any information on daylight saving time
    if time.count('+') == 1 and '+' in time[-6:]:
        (time, tz) = time.rsplit('+')
        delta = -1
    elif time.count('-') == 1 and '-' in time[-6:]:
        (time, tz) = time.rsplit('-')
        delta = 1
    else:
        delta = 0
    if delta:
        while len(tz) < 3:
            tz += '0'
        delta = delta * (int(tz[0:2]) * 60 * 60 + int(tz[2:]) * 60)
    # split microseconds
    ms = 0
    if '.' in time:
        (time, ms) = time.split(".")
        ms = float('0.' + ms.strip())
    # guess time pattern
    length_time = len(time)
    if length_time == 6 and time.isdigit():
        time_pattern = "%H%M%S"
    elif length_time == 4 and time.isdigit():
        time_pattern = "%H%M"
    elif length_time == 2 and time.isdigit():
        time_pattern = "%H"
    elif length_time == 0:
        time_pattern = ""
    else:
        raise ValueError("Wrong or incomplete ISO8601:2004 time format")
    # parse patterns
    dt = datetime.datetime.strptime(date + 'T' + time,
                                    date_pattern + 'T' + time_pattern)
    # add microseconds and eventually correct time zone
    return dt + datetime.timedelta(seconds=float(delta) + ms)


def get_wind_speed(gf, lon, lat, elev, date):
    """
    Given a GasFlow object return the wind speed vector
    closest to the requested location and time.

    :type gf: `spectroscopy.datamodel.GasFlow`
    :param gf: GasFlow object
    :type lon: float
    :param lon: Longitude of the point of interest.
    :type lat: float
    :param lat: Latitude of the point of interest.
    :type elev: float
    :param elev: Altitude in meters of the point of interest.
    :type date: str
    :param date: Date of interest formatted according to the ISO8601
        standard.
    """
    from scipy.spatial import KDTree
    from spectroscopy.util import parse_iso_8601
    if not isinstance(date, datetime.datetime):
        _d = parse_iso_8601(date)
    else:
        _d = date
    _ts = calendar.timegm((_d.utctimetuple()))
    # convert to ms
    _ts *= 1e3
    _dt = gf.datetime[:].astype(np.datetime64)
    # find nearest point
    _t = np.atleast_2d(_dt).T
    # TODO: this needs to be changed to account for regular and irregular
    # grids
    a = np.append(gf.position[:], _t.astype('float'), axis=1)
    tree = KDTree(a, leafsize=a.shape[0] + 1)
    point = [lon, lat, elev, _ts]
    distances, ndx = tree.query([point], k=1)
    vx = gf.vx[:][ndx[0]]
    vy = gf.vy[:][ndx[0]]
    vz = gf.vz[:][ndx[0]]
    try:
        vx_error = gf.vx_error[:][ndx[0]]
        vy_error = gf.vy_error[:][ndx[0]]
        vz_error = gf.vz_error[:][ndx[0]]
    except TypeError:
        vx_error = None
        vy_error = None
        vz_error = None
    time = gf.datetime[:][ndx[0]]
    lon, lat, hght = gf.position[:][ndx[0], :]
    return (lon, lat, hght, time, vx, vx_error,
            vy, vy_error, vz, vz_error, distances[0])


def _array_multi_sort(*arrays):
    """
    Sorts multiple numpy arrays based on the contents of the first array.

    >>> x1 = np.array([4.,5.,1.,2.])
    >>> x2 = np.array([10.,11.,12.,13.])
    >>> _array_multi_sort(*tuple([x1,x2]))
    (array([ 1.,  2.,  4.,  5.]), array([ 12.,  13.,  10.,  11.]))
    """
    c = np.rec.fromarrays(
        arrays, names=[str(i) for i in range(len(arrays))])
    c.sort()  # sort based on values in first array
    return tuple([c[str(i)] for i in range(len(arrays))])


def split_by_scan(angles, *vars_):
    """
    Returns an iterator that will split lists/arrays of data by scan (i.e.
    between start and end angle) an arbitrary number of lists of data can
    be passed in - the iterator will return a list of arrays of length
    len(vars_) + 1 with the split angles array at index one, and the
    remaining data lists in order afterwards. The lists will be sorted
    into ascending angle order.

    >>> angles = np.array([30, 35, 40, 35, 30, 35, 40])
    >>> [a[0] for a in split_by_scan(angles)]
    [array([30, 35, 40]), array([30, 35]), array([35, 40])]
    >>> [a[1] for a in split_by_scan(angles, np.array([1,2,3,4,5,6,7]))]
    [array([1, 2, 3]), array([5, 4]), array([6, 7])]
    """
    # everything breaks if there are more than two equal angles in a row.
    if np.any(np.logical_and((angles[1:] == angles[:-1])[:-1],
                             angles[2:] == angles[:-2])):
        idx = np.argmax(np.logical_and(
            (angles[1:] == angles[:-1])[:-1], angles[2:] == angles[:-2]))
        msg = "Data at line {} ".format(str(idx + 2))
        msg += "contains three or more repeated angle entries (in a row)."
        msg += "Don't know how to split this into scans."
        raise ValueError(msg)

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
        yield _array_multi_sort(angles, *vars_)
    else:
        d = [angles[:inflectionpoints[1]]]
        for l in vars_:
            d.append(l[0:inflectionpoints[1]:])
        yield _array_multi_sort(*tuple(d))

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
            yield _array_multi_sort(*tuple(d))

        # the final point is not an inflection point so now we need to
        # return the final scan
        d = [angles[inflectionpoints[i]:]]
        for l in vars_:
            d.append(l[inflectionpoints[i]:])
        yield _array_multi_sort(*tuple(d))


if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)
