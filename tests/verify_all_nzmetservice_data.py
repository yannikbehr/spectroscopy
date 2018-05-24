#!/usr/bin/env python

import datetime
import filecmp
import math
import os
import re
import tempfile

import numpy as np
import pytz
from pytz import timezone

from spectroscopy.dataset import Dataset
from spectroscopy.plugins.nzmetservice import NZMetservicePluginException
from spectroscopy.util import vec2bearing, get_wind_speed, parse_iso_8601


volc_dict_keys = ['Auckland', 'Haroharo', 'Mayor Island', 'Ngauruhoe',
                  'Ruapehu', 'Taranaki (Egmont)', 'Tarawera', 'Taupo',
                  'Tongariro', 'White Island']
volc_dict_values = [(174.735, -36.890), (176.466, -38.147),
                    (176.256, -37.287), (175.632, -39.157),
                    (175.564, -39.281), (174.064, -39.297),
                    (176.506, -38.227), (175.978, -38.809),
                    (175.673, -39.108), (177.183, -37.521)]
volc_dict = {}

nztz = timezone('Pacific/Auckland')

for _k, _v in zip(volc_dict_keys, volc_dict_values):
    volc_dict[_k] = _v


def ord(n):
    """
    Add one of 'st', 'nd', 'rd', 'th' to the day number.
    """
    return str(n)+("th" if 4 <= n % 100 <= 20
                   else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th"))


def ampm(n):
    """
    Change 24-h format to 12-h format.
    """
    return ('{:d}am'.format(n) if n < 12
            else {24: "12am", 12: "12pm"}.get(n, '{:d}pm'.format(n % 12)))


def write_testfile(d, date, time, fin):
    """
    This will write a file that is identical to the input file
    but taking all the information from the GasFlow element.
    """
    gf = d.elements['GasFlow'][0]
    mod = gf.methods[0].name
    year, month, day = int(date[0:4]), int(date[4:6]), int(date[6:8])
    hour, minute = int(time[0:2]), int(time[2:4])
    lines = []
    # Write the header
    _h, _s = re.match('(\d+)(\S+)', ampm(hour)).groups()
    line = 'Forecast issued by MetService at '
    line += '{:02d}:{:02d}{:s} '.format(int(_h), minute, _s)
    line += '{:02d}-{:02d}-{:4d}\n\n'.format(day, month, year)
    line += 'For GNS Wairakei Research Centre - Volcano Watch All times NZDT '
    line += 'e.g. {:02d}{:02d}00 '.format(day, hour)
    line += 'is {:s} on {:s}. '.format(ampm(hour), ord(day))
    line += 'Winds in degrees/knots, heights in metres.\n'
    line += 'Model of the day is {:s}\n\n'.format(mod.upper())
    line += 'Data for model {:s}\n'.format(mod.upper())
    lines.append(line)
    # Cycle through all points
    for point in volc_dict_keys:
        # Write header for each point
        lines.append('{:s}\n'.format(point))
        times = []
        ntimes = len(np.unique(gf.datetime[:]))
        line = 'Height  '
        for i in range(ntimes):
            line += 'Valid at      '
        line += '\n'
        lines.append(line)
        line = '{:8}'.format('')
        # Write datetimes
        dt = (nztz.localize(datetime.datetime(year, month, day, hour)).
              astimezone(timezone('UTC')))
        otimes = np.unique(gf.datetime[:])
        otimes.sort()
        for t in otimes.astype(np.str_):
            dt = timezone('UTC').localize(parse_iso_8601(t))
            times.append(dt)
            line += '{:<14s}'.format(dt.astimezone(nztz).strftime('%d%H%M'))
        line += '\n'
        lines.append(line)
        ln, lt = volc_dict[point]
        # Write actual wind data
        for height in [1000, 2000, 3000, 4000, 6000, 8000, 10000, 12000]:
            line = ''
            line += '{:<8d}'.format(height)
            for i, dt in enumerate(times):
                res = get_wind_speed(gf, ln, lt, height,
                                     dt.astimezone(pytz.utc))
                (lon, lat, hght, time, vx, vx_error,
                 vy, vy_error, vz, vz_error, dist) = res
                if dist > 0.0001:
                    line += '{0:<14s}'.format('-')
                else:
                    try:
                        v = int(round(math.sqrt(vx * vx + vy * vy)/0.514444))
                        vd = int(round(vec2bearing(vx, vy)))
                        if vd == 0:
                            vd = 360
                    except Exception as e:
                        print(point, hght, time)
                        raise e
                    line += '{0:03d}/{1:02d}{2:8}'.format(vd, v, '')
            line += '\n'
            lines.append(line)
        lines.append('\n')
    fin = fin.replace('ecmwf', mod)
    fout = os.path.basename(fin)
    fout = os.path.join('/tmp', fout.replace('.txt', '_test.txt'))
    fh = open(fout, 'w')
    fh.writelines(lines)
    fh.close()
    # Now compare with the original
    if not filecmp.cmp(fout, fin, shallow=False):
        raise Exception('Files %s and %s are not identical!' % (fin, fout))
    else:
        os.unlink(fout)


def main(rootdir):
    for root, dirs, files in os.walk(rootdir):
        for f in files:
            match = re.match('gns_wind_model_data_ecmwf_(\d+)_(\d+).txt', f)
            if match:
                msg = "reading " + os.path.join(root, f)
                msg += " for day " + match.group(1)
                print(msg)
                d = Dataset(tempfile.mktemp(), 'w')
                try:
                    d.read(os.path.join(root, f), ftype='NZMETSERVICE')
                    write_testfile(d, match.group(1), match.group(2),
                                   os.path.join(root, f))
                except NZMetservicePluginException as e:
                    print(e)
                del d


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('directory',
                        help="root directory to start search")
    args = parser.parse_args()
    main(args.directory)
