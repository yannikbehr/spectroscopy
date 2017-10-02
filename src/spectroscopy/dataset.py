import calendar
import collections
from copy import copy, deepcopy
import datetime
import hashlib
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
import pandas as pd
import tables
from tables.group import Group
from tables.exceptions import NoSuchNodeError, NodeError
from scipy.stats import binned_statistic

from spectroscopy.class_factory import ResourceIdentifier
from spectroscopy.plugins import get_registered_plugins
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
        # Create an array of sha224 hash values; when
        # opening an existing file this will throw an
        # exception
        try:
            self._f.create_earray('/','hash',tables.StringAtom(itemsize=28),(0,))
        except NodeError:
            pass

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
        rid_dict = {}
        for e in other.elements.keys():
            for k in other.elements[e]:
                ne = self._copy_children(k, postfix)
                self.elements[e].append(ne)
                update_refs.append(ne)
                rid_dict[str(k._resource_id)] = str(ne._resource_id)

        for ne in update_refs:
            for k in ne._reference_keys:
                table = getattr(ne._root,'data',None)
                if table is not None:
                    ref = getattr(ne._root.data.cols,k,None)
                    if ref is not None:
                        if type(ref[0]) == np.ndarray:
                            newentry = []
                            for iref in ref[0]:
                                newentry.append(rid_dict[iref])
                            ref[0] = np.array(newentry)
                        else:
                            ref[0] = rid_dict[ref[0]]
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
        # assign a new resource ID so that both objects can 
        # be referred to within the same session
        dstgroup = srcgroup._v_parent._v_pathname+'/'+ str(ResourceIdentifier())
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

    def new(self, data_buffer, pedantic=False):
        """
        Create a new entry in the HDF5 file from the given data buffer.
        """
        if pedantic:
            s = hashlib.sha224()
            # If data buffer is empty raise an exception
            empty = True
            for k,v in data_buffer.__dict__.iteritems():
                if k == 'tags':
                    continue
                if v is not None:
                    if k in data_buffer._property_dict.keys():
                        s.update('{}'.format(v))
                    empty = False
            if empty:
                raise ValueError("You can't add empty buffers if 'pedantic=True'.")

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
        e = _C(group,data_buffer, pedantic=pedantic)
        self.elements[group_name].append(e)
        return e         

    def read(self, filename, ftype, **kwargs):
        """
        Read in a datafile.
        """
        plugins = get_registered_plugins()
        pg = plugins[ftype.lower()]()
        pg.read(self,filename,**kwargs)
    
    @staticmethod
    def open(filename):
        """
        Open an existing HDF5 file.
        """
        dnew = Dataset(filename,'r+')         
        for group in dnew._f.walk_groups('/'):
            if group._v_name is '/' or group._v_name+'Buffer' not in dnew.base_elements:
                continue
            for sgroup in group._v_groups.keys():
                _C = dnew.base_elements[group._v_name+'Buffer']
                e = _C(group._v_groups[sgroup])
                dnew.elements[group._v_name].append(e)
        return dnew

    def close(self):
        """
        Close the HDF5 file and clear the ResourceIdentifiers.
        """
        for g in self.elements:
            for e in self.elements[g]:
                del e._resource_id
        self._f.close()

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
                
    def plot(self, toplot='concentrations', savefig=None, **kargs):
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
            rts = self.elements['Concentration'][0]
            nretrieval = len(rts.value[:])
            m = np.zeros((nretrieval, angle_bins.size - 1))

            r = rts.rawdata
            for i, _so2 in enumerate(rts.value[:]):
                _angle = r.inc_angle[i]
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
                    _a = r.inc_angle[int(_xt)]
                    ymin = min(_a.min(), ymin)
                    ymax = max(_a.max(), ymax)
                    dt = r.datetime[int(_xt)].astype('datetime64[us]').min()
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

        else:
            print 'Plotting %s has not been implemented yet' % toplot

            
if __name__ == '__main__':
    import doctest
    doctest.testmod(exclude_empty=True)

