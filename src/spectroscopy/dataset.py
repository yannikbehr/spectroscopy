import calendar
import collections
from copy import copy, deepcopy
import datetime
import inspect
import string
from uuid import uuid4
import warnings
import weakref

from dateutil import tz
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, LogNorm
from matplotlib.pyplot import cm
import numpy as np
import tables
from scipy.stats import binned_statistic

from spectroscopy.class_factory import ResourceIdentifier
from spectroscopy.plugins import get_registered_plugins, DatasetPluginBase
import spectroscopy.util
from spectroscopy.datamodel import (_Target, _RawData)


class Dataset(object):
    """
    This class is a container for all data describing a spectroscopy analysis
    from the raw measurements, over instruments and information on gas plumes
    to the final gas flux results.

    :type preferredFluxIDs: list
    :param preferredFluxIDs: IDs of the best/final flux estimate. As a dataset
        can contain analyses from different targets, there can be more than one
        preferred flux estimate.
    :type spectra: list
    :param spectra: List of all spectra that are part of the dataset.
    :type instruments: list
    :param instruments: List of all instruments that are part of the dataset.
    :type retrievals: list
    :param retrievals: List of all retrievals that are part of the dataset.
    :type plumevelocities: list
    :param plumevelocities: List of all plume velocities that are part of the
        dataset.
    :type targets: list
    :param targets: List of all target plumes that are part of the dataset.
    :type flux: list
    :param flux: List of all flux estimates that are part of the dataset.
    """

    def __init__(self, filename, mode):
        self.preferred_fluxes = []
        self.fluxes = []
        self.methods = []
        self.gas_flows = []
        self.concentrations = []
        self.raw_data = []
        self.instruments = []
        self.targets = []
        self.raw_data_types = []
        self.data_quality_types = []
        self._rids = {}
        self._f = tables.open_file(filename, mode)
        self.func_table = {'TargetBuffer': self._new_target,
                           'RawDataBuffer': self._new_raw_data
                          }

    def __del__(self):
        self._f.close()

    def _new_raw_data(self, rb):
        rid = ResourceIdentifier()
        try:
            self._f.create_group('/','RawData')
        except tables.NodeError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f.create_group('/RawData',str(rid))

        return _RawData(getattr(self._f.root.RawData,str(rid)),rb)

    def _new_target(self,tb):
        """
        Create a new target object entry in the HDF5 file.
        """
        rid = ResourceIdentifier()
        try:
            self._f.create_group('/','Target')
        except tables.NodeError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self._f.create_group('/Target',str(rid))

        return _Target(getattr(self._f.root.Target,str(rid)),tb)
        
    def new(self, data_buffer):
        """
        Create a new entry in the HDF5 file from the given data buffer.
        """
        return self.func_table[type(data_buffer).__name__](data_buffer)

    def plot(self, toplot='retrievals', savefig=None, **kargs):
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

        if toplot.lower() == 'concentrations':
            cmap_name = kargs.get('cmap_name', 'RdBu_r')
            log = kargs.get('log', False)
            angle_bins = kargs.get('angle_bins', np.arange(0, 180, 1.0))
            ncontours = kargs.get('ncontours', 100)
            ts = kargs.get('timeshift', 0.0) * 60. * 60.
            cmap = cm.get_cmap(cmap_name)
            # dicretize all retrievals onto a grid to show a daily plot
            rts = self.concentrations
            nretrieval = len(rts)
            m = np.zeros((nretrieval, angle_bins.size - 1))

            # first sort retrievals based on start time
            def mycmp(r1, r2):
                s1 = r1.rawdata_id.get_referred_object()
                t1 = s1.time[r1.rawdata_indices].min()
                s2 = r2.rawdata_id.get_referred_object()
                t2 = s2.time[r2.rawdata_indices].min()
                if t1 < t2:
                    return -1
                if t1 == t2:
                    return 0
                if t1 > t2:
                    return 1
            rts.sort(cmp=mycmp)

            for i, _r in enumerate(rts):
                _s = _r.rawdata_id.get_referred_object()
                _angle = _s.angle[_r.rawdata_indices]
                _so2 = _r.sca
                _so2_binned = binned_statistic(
                    _angle, _so2, 'mean', angle_bins)
                m[i, :] = _so2_binned.statistic

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
            ymin = angle_bins[-1]
            ymax = angle_bins[0]
            for _xt in plt.xticks()[0]:
                try:
                    _r = rts[int(_xt)]
                    _s = _r.spectra_id.get_referred_object()
                    _a = _s.angle[_r.rawdata_indices]
                    ymin = min(_a.min(), ymin)
                    ymax = max(_a.max(), ymax)
                    dt = datetime.datetime.fromtimestamp(
                        _s.time[_r.rawdata_indices].min(), tz=tz.gettz('UTC'))
                    dt += datetime.timedelta(seconds=ts)
                    new_labels.append(dt.strftime("%Y-%m-%d %H:%M"))
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

        else:
            print 'Plotting %s has not been implemented yet' % toplot

            
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

