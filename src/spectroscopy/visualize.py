"""
Overview plots for different elements in a dataset.
"""

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.pyplot import cm
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic

from spectroscopy.util import split_by_scan


def plot_concentration(c, savefig=None, **kargs):
    """
    Provide overview plots for data contained in a dataset.

    :type toplot: str
    :param toplot: Choose the datatype to plot.

    Parameters specific to `retrievals` contour plots:

    :type log: bool
    :param log: Turn on logarithmic colour scales.
    :type cmap_name: str
    :param cmap_name: The name of the matplotlib colour scale to use.
    :type angle_bins: :class:`numpy.ndarray`
    :param angle_bins: Define the bins onto which the angles of the
        retrievals are discretized to.
    :type ncontours: int
    :param ncontours: Number of contours used in the contour plot.

    """

    cmap_name = kargs.get('cmap_name', 'RdBu_r')
    log = kargs.get('log', False)
    angle_bins = kargs.get('angle_bins', np.arange(0, 180, 1.0))
    ncontours = kargs.get('ncontours', 100)
    ts = kargs.get('timeshift', 0.0) * 60. * 60.
    cmap = cm.get_cmap(cmap_name)
    # dicretize all retrievals onto a grid to show a daily plot
    r = c.rawdata
    m = [] 
    times = []
    ymin = angle_bins[-1]
    ymax = angle_bins[0]
    nretrieval = 0
    for _angle, _so2, _t in split_by_scan(r.inc_angle[:], c.value[:], r.datetime[:]):
        ymin = min(_angle.min(), ymin)
        ymax = max(_angle.max(), ymax)
        times.append(_t)
        _so2_binned = binned_statistic(
            _angle, _so2, 'mean', angle_bins)
        m.append(_so2_binned.statistic)
        nretrieval += 1
    m = np.array(m)

    fig = plt.figure()
    if log:
        z = np.where(m > 0.0, m, 0.1)
        plt.contourf(range(nretrieval), angle_bins[1:], z.T, ncontours,
                     norm=LogNorm(z.min(), z.max()), cmap=cmap)
    else:
        z = np.ma.masked_invalid(m)
        plt.contourf(range(nretrieval), angle_bins[1:], m.T, ncontours,
                     norm=Normalize(z.min(), z.max()), cmap=cmap)
    new_labels = []
    new_ticks = []
    for _xt in plt.xticks()[0]:
        try:
            dt = times[int(_xt)].astype('datetime64[us]').min()
            dt += np.timedelta64(int(ts),'s')                    
            new_labels.append(pd.to_datetime(str(dt)).strftime("%Y-%m-%d %H:%M"))
            new_ticks.append(_xt)
        except IndexError:
            continue
    plt.xticks(new_ticks, new_labels, rotation=30,
               horizontalalignment='right')
    cb = plt.colorbar()
    cb.set_label('Slant column amount SO2 [ppm m]')
    plt.ylim(ymin, ymax)
    plt.ylabel(r'Angle [$\circ$]')
    if savefig is not None:
        plt.savefig(
            savefig, bbox_inches='tight', dpi=300, format='png')
    return fig


def plot_rawdata(r, savefig=None, **kargs):
    matplotlib.style.use('ggplot')
    counts = r.d_var[:]
    w = r.ind_var[:]
    nc = counts.shape[0]
    cmap = cm.ScalarMappable(norm=Normalize(vmin=0, vmax=nc-1), cmap='RdBu')
    fig = plt.figure(figsize=(12,6))
    for i in range(nc):
        c = cmap.to_rgba(i)
        plt.plot(w,counts[i],color=c, alpha=0.2)
    
    plt.xlabel('Wavelength [nm]')
    plt.ylabel('Intensity')
    cax, kw = matplotlib.colorbar.make_axes(plt.gca())
    norm = Normalize(vmin = 0, vmax = nc, clip = False)
    c = matplotlib.colorbar.ColorbarBase(cax, cmap='RdBu', norm=norm)
    ticks = np.array([0, int(nc/2.), nc-1])
    times = r.datetime[:]
    labels = np.array([times[0],times[int(nc/2.)],times[nc-1]])
    c.set_ticks(ticks)
    c.set_ticklabels(labels)
    if savefig is not None:
        plt.savefig(
            savefig, bbox_inches='tight', dpi=300, format='png')
    return fig



def plot(element, **kargs):
    name = str(element).replace('Buffer','')
    name = 'plot_'+name.lower()
    globals()[name](element, **kargs)


