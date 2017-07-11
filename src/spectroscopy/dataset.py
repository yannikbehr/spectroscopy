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
from tables.group import Group
from tables.exceptions import NoSuchNodeError, NodeError
from scipy.stats import binned_statistic

from spectroscopy.class_factory import ResourceIdentifier
from spectroscopy.plugins import get_registered_plugins, DatasetPluginBase
import spectroscopy.util
from spectroscopy.datamodel import all_classes


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
        self.elements = {}
        self.base_elements = {}
        for c in all_classes:
            name = c.__name__.strip('_') 
            self.elements[name] = []
            self.base_elements[name+'Buffer'] = c
        self._rids = {}
        self._f = tables.open_file(filename, mode)

    def __del__(self):
        self._f.close()
    
    def __add__(self, other):
        msg = "__add__ is undefined as the return value would "
        msg += "be a new hdf5 file with unknown filename."
        raise AttributeError(msg)                             

    def __iadd__(self, other):
        if self._f == other._f:
            raise ValueError("You can't add a dataset to itself.")
        update_refs = []
        postfix = self._gen_sc3_id(datetime.datetime.now())
        for e in other.elements.keys():
            for k in other.elements[e]:
                ne = self._copy_children(k, postfix)
                self.elements[e].append(ne)
                update_refs.append(ne)

        for ne in update_refs:
            for k in ne._reference_keys:
                table = getattr(ne._root,'data',None)
                if table is not None:
                    ref = getattr(ne._root.data.cols,k,None)
                    if ref is not None:
                        ref[0] += postfix
        return self

    def _gen_sc3_id(self, dt, numenc=6, sym="abcdefghijklmnopqrstuvwxyz"):
        """
        Generate an event ID following the SeisComP3 convention. By default it
        divides a year into 26^6 intervals assigning each a unique combination of
        characters.

        >>> import datetime 
        >>> import tempfile
        >>> d = Dataset(tempfile.mktemp(), 'w')
        >>> print d.gen_sc3_id(datetime.datetime(2015, 8, 18, 10, 55, 51, 367580))
        2015qffasl
        """
        numsym = len(sym)
        julday = int(dt.strftime('%j'))
        x = (((((julday - 1) * 24) + dt.hour) * 60 + dt.minute) *
             60 + dt.second) * 1000 + dt.microsecond / 1000
        dx = (((370 * 24) * 60) * 60) * 1000
        rng = numsym ** numenc
        w = int(dx / rng)
        if w == 0:
            w = 1

        if dx >= rng:
            x = int(x / w)
        else:
            x = x * int(rng / dx)
        enc = ''
        for _ in range(numenc):
            r = x % numsym
            enc += sym[r]
            x = int(x / numsym)
        return '%d%s' % (dt.year, enc[::-1])

    def _newdst_group(self, dstgroup, title='', filters=None):
        """
        Create the destination group in a new HDF5 file.
        """
        group = self._f.root
        # Now, create the new group. This works even if dstgroup == '/'
        for nodename in dstgroup.split('/'):
            if nodename == '':
                continue
            # First try if possible intermediate groups already exist.
            try:
                group2 = self._f.get_node(group, nodename)
            except NoSuchNodeError:
                # The group does not exist. Create it.
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    group2 = self._f.create_group(group, nodename,
                                                title=title,
                                                filters=filters)
            group = group2
        return group

    def _copy_children(self, src, postfix, title='', recursive=True,
                       filters=None, copyuserattrs=False,
                       overwrtnodes=False):
        """
        Copy the children from source group to destination group
        """
        srcgroup = src._root
        dstgroup = srcgroup._v_pathname + postfix
        created_dstgroup = False
        # Create the new group
        dstgroup = self._newdst_group(dstgroup, title, filters)

        # Copy the attributes to dstgroup, if needed
        if copyuserattrs:
            srcgroup._v_attrs._f_copy(dstgroup)

        # Finally, copy srcgroup children to dstgroup
        try:
            srcgroup._f_copy_children(
                dstgroup, recursive=recursive, filters=filters,
                copyuserattrs=copyuserattrs, overwrite=overwrtnodes)
        except:
            msg = "Problems doing the copy of '{:s}'.".format(dstgroup)
            msg += "Please check that the node names are not "
            msg += "duplicated in destination, and if so, enable "
            msg += "overwriting nodes if desired."
            raise RuntimeError(msg)
        return type(src)(dstgroup)

    def new(self, data_buffer):
        """
        Create a new entry in the HDF5 file from the given data buffer.
        """
        _C = self.base_elements[type(data_buffer).__name__]
        group_name = _C.__name__.strip('_')
        rid = ResourceIdentifier()
        try:
            self._f.create_group('/',group_name)
        except tables.NodeError:
            pass
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            group = self._f.create_group('/'+group_name,str(rid))
        e = _C(group,data_buffer)
        self.elements[group_name].append(e)
        return e         

    def register_tags(self, tags):
        """
        Register one or more tag names.
        """
        try:
            self._f.create_group('/','tags')
        except NodeError:
            pass
        for tag in tags:
            try:
                self._f.create_earray('/tags', tag, tables.StringAtom(itemsize=60), (0,))
            except NodeError:
                raise ValueError("Tag '{:s}' has already been registered".format(tag))

    def remove_tags(self, tags):
        """
        Remove one or more tag names. This will also remove the tag from every
        element that had been tagged.
        """
        for tag in tags:
            try:
                ea = self._f.root.tags._v_children[tag]
                for rid in ea[:]:
                    e = ResourceIdentifier(rid).get_referred_object()
                    e.tags.remove(tag)
            except (KeyError, NoSuchNodeError):
                warnings.warn("Can't remove tag {} as it doesn't exist.".format(tag)) 
                
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

