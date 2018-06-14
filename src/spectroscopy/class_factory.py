"""
Generate classes defined in the datamodel.
"""
import collections
from copy import deepcopy
import datetime
import hashlib
import inspect
from uuid import uuid4
import warnings
import weakref

import numpy as np
import tables

import spectroscopy.util


class ResourceIdentifier(object):
    """
    Unique identifier of any resource so it can be referred to.

    All elements of a Dataset instance have a unique id that other elements
    use to refer to it. This is called a ResourceIdentifier.

    In this class it can be any hashable object, e.g. most immutable objects
    like numbers and strings.

    :type id: str, optional
    :param id: A unique identifier of the element it refers to. It is
        not verified, that it actually is unique. The user has to take care of
        that. If no resource_id is given, uuid.uuid4() will be used to
        create one which assures uniqueness within one Python run.
        If no fixed id is provided, the ID will be built from prefix
        and a random uuid hash. The random hash part can be regenerated by the
        referred object automatically if it gets changed.
    :type prefix: str, optional
    :param prefix: An optional identifier that will be put in front of any
        automatically created resource id. The prefix will only have an effect
        if `id` is not specified (for a fixed ID string). Makes automatically
        generated resource ids more reasonable.
    :type referred_object: Python object, optional
    :param referred_object: The object this instance refers to. All instances
        created with the same resource_id will be able to access the object as
        long as at least one instance actual has a reference to it.

    .. rubric:: General Usage

    >>> ResourceIdentifier('2012-04-11--385392')
    ResourceIdentifier(id="2012-04-11--385392")
    >>> # If 'id' is not specified it will be generated automatically.
    >>> ResourceIdentifier()  # doctest: +ELLIPSIS
    ResourceIdentifier(id="...")
    >>> # Supplying a prefix will simply prefix the automatically generated ID
    >>> ResourceIdentifier(prefix='peru09')  # doctest: +ELLIPSIS
    ResourceIdentifier(id="peru09_...")

    ResourceIdentifiers can, and oftentimes should, carry a reference to the
    object they refer to. This is a weak reference which means that if the
    object gets deleted or runs out of scope, e.g. gets garbage collected, the
    reference will cease to exist.

    >>> class A(object): pass
    >>> a = A()
    >>> import sys
    >>> ref_count = sys.getrefcount(a)
    >>> res_id = ResourceIdentifier(referred_object=a)
    >>> # The reference does not change the reference count of the object.
    >>> print(ref_count == sys.getrefcount(a))
    True
    >>> # It actually is the same object.
    >>> print(a is res_id.get_referred_object())
    True
    >>> # Deleting it, or letting the garbage collector handle the object will
    >>> # invalidate the reference.
    >>> del a
    >>> print(res_id.get_referred_object())
    None

    The most powerful ability (and reason why one would want to use a resource
    identifier class in the first place) is that once a ResourceIdentifier with
    an attached referred object has been created, any other ResourceIdentifier
    instances with the same ID can retrieve that object. This works
    across all ResourceIdentifiers that have been instantiated within one
    Python run.
    This enables, e.g. the resource references between the different elements
    to work in a rather natural way.

    >>> a = A()
    >>> obj_id = id(a)
    >>> res_id = "someid"
    >>> ref_a = ResourceIdentifier(res_id)
    >>> # The object is refers to cannot be found yet. Because no instance that
    >>> # an attached object has been created so far.
    >>> print(ref_a.get_referred_object())
    None
    >>> # This instance has an attached object.
    >>> ref_b = ResourceIdentifier(res_id, referred_object=a)
    >>> ref_c = ResourceIdentifier(res_id)
    >>> # All ResourceIdentifiers will refer to the same object.
    >>> assert(id(ref_a.get_referred_object()) == obj_id)
    >>> assert(id(ref_b.get_referred_object()) == obj_id)
    >>> assert(id(ref_c.get_referred_object()) == obj_id)


    ResourceIdentifiers are considered identical if the IDs are
    the same.

    >>> # Create two different resource identifiers.
    >>> res_id_1 = ResourceIdentifier()
    >>> res_id_2 = ResourceIdentifier()
    >>> assert(res_id_1 != res_id_2)
    >>> # Equalize the IDs. NEVER do this. This is just an example.
    >>> res_id_2.id = res_id_1.id = "smi:local/abcde"
    >>> assert(res_id_1 == res_id_2)

    ResourceIdentifier instances can be used as dictionary keys.

    >>> dictionary = {}
    >>> res_id = ResourceIdentifier(id="foo")
    >>> dictionary[res_id] = "bar1"
    >>> # The same ID can still be used as a key.
    >>> dictionary["foo"] = "bar2"
    >>> items = sorted(dictionary.items(), key=lambda kv: kv[1])
    >>> for k, v in items:  # doctest: +ELLIPSIS
    ...     print repr(k), v
    ResourceIdentifier(id="foo") bar1
    ...'foo' bar2
    """
    # Class (not instance) attribute that keeps track of all resource
    # identifier throughout one Python run. Will only store weak references and
    # therefore does not interfere with the garbage collection.
    # DO NOT CHANGE THIS FROM OUTSIDE THE CLASS.
    __resource_id_weak_dict = weakref.WeakValueDictionary()
    # Use an additional dictionary to track all resource ids.
    __resource_id_tracker = collections.defaultdict(int)

    def __init__(self, oid=None, prefix=None,
                 referred_object=None):
        # Create a resource id if None is given and possibly use a prefix.
        if oid is None:
            self.fixed = False
            self._prefix = prefix
            self._uuid = str(uuid4())
        else:
            self.fixed = True
            self.id = oid
        # Append the referred object in case one is given to the class level
        # reference dictionary.
        if referred_object is not None:
            self.set_referred_object(referred_object)

        # Increment the counter for the current resource id.
        ResourceIdentifier.__resource_id_tracker[self.id] += 1

    def __del__(self):
        if self.id not in ResourceIdentifier.__resource_id_tracker:
            return
        # Decrement the resource id counter.
        ResourceIdentifier.__resource_id_tracker[self.id] -= 1
        # If below or equal to zero, delete it and also delete it from the weak
        # value dictionary.
        if ResourceIdentifier.__resource_id_tracker[self.id] <= 0:
            del ResourceIdentifier.__resource_id_tracker[self.id]
            try:
                del ResourceIdentifier.__resource_id_weak_dict[self.id]
            except KeyError:
                pass

    def get_referred_object(self):
        """
        Returns the object associated with the resource identifier.

        This works as long as at least one ResourceIdentifier with the same
        ID as this instance has an associate object.

        Will return None if no object could be found.
        """
        try:
            return ResourceIdentifier.__resource_id_weak_dict[self.id]
        except KeyError:
            return None

    def set_referred_object(self, referred_object):
        """
        Sets the object the ResourceIdentifier refers to.

        If it already a weak reference it will be used, otherwise one will be
        created. If the object is None, None will be set.

        Will also append self again to the global class level reference list so
        everything stays consistent.
        """
        # If it does not yet exists simply set it.
        if self.id not in ResourceIdentifier.__resource_id_weak_dict:
            ResourceIdentifier.__resource_id_weak_dict[self.id] = \
                referred_object
            return
        # Otherwise check if the existing element the same as the new one. If
        # it is do nothing, otherwise raise a warning and set the new object as
        # the referred object.
        if ResourceIdentifier.__resource_id_weak_dict[self.id] == \
                referred_object:
            return
        msg = "The resource identifier '%s' already exists and points to " + \
              "another object: '%s'." + \
              "It will now point to the object referred to by the new " + \
              "resource identifier."
        msg = msg % (
            self.id,
            repr(ResourceIdentifier.__resource_id_weak_dict[self.id]))
        # Always raise the warning!
        warnings.warn_explicit(msg, UserWarning, __file__,
                               inspect.currentframe().f_back.f_lineno)
        ResourceIdentifier.__resource_id_weak_dict[self.id] = \
            referred_object

    def copy(self):
        """
        Returns a copy of the ResourceIdentifier.

        >>> res_id = ResourceIdentifier()
        >>> res_id_2 = res_id.copy()
        >>> print(res_id is res_id_2)
        False
        >>> print(res_id == res_id_2)
        True
        """
        return deepcopy(self)

    @property
    def id(self):
        """
        Unique identifier of the current instance.
        """
        if self.fixed:
            return self.__dict__.get("id")
        else:
            oid = self.prefix
            if oid is not None and not oid.endswith("_"):
                oid += "_"
                oid += self.uuid
                return oid
            return self.uuid

    @id.deleter
    def id(self):
        msg = "The resource id cannot be deleted."
        raise Exception(msg)

    @id.setter
    def id(self, value):
        self.fixed = True
        # XXX: no idea why I had to add bytes for PY2 here
        if not isinstance(value, (str, bytes)):
            msg = "attribute id needs to be a string."
            raise TypeError(msg)
        self.__dict__["id"] = value

    @property
    def prefix(self):
        return self._prefix

    @prefix.deleter
    def prefix(self):
        self._prefix = ""

    @prefix.setter
    def prefix(self, value):
        if not isinstance(value, str):
            msg = "prefix id needs to be a string."
            raise TypeError(msg)
        self._prefix = value

    @property
    def uuid(self):
        return self._uuid

    @uuid.deleter
    def uuid(self):
        """
        Deleting is uuid hash is forbidden and will not work.
        """
        msg = "The uuid cannot be deleted."
        raise Exception(msg)

    @uuid.setter
    def uuid(self, value):  # @UnusedVariable
        """
        Setting is uuid hash is forbidden and will not work.
        """
        msg = "The uuid cannot be set manually."
        raise Exception(msg)

    @property
    def resource_id(self):
        return self.id

    @resource_id.deleter
    def resource_id(self):
        del self.id

    @resource_id.setter
    def resource_id(self, value):
        self.id = value

    def __str__(self):
        return self.id

    def _repr_pretty_(self, p, cycle):
        p.text(str(self))

    def __repr__(self):
        return 'ResourceIdentifier(id="%s")' % self.id

    def __eq__(self, other):
        if self.id == other:
            return True
        if not isinstance(other, ResourceIdentifier):
            return False
        if self.id == other.id:
            return True
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __hash__(self):
        """
        Uses the same hash as the resource id. This means that class instances
        can be used in dictionaries and other hashed types.

        Both the object and it's id can still be independently used as
        dictionary keys.
        """
        # "Salt" the hash with a string so the hash of the object and a
        # string identical to the id can both be used as individual
        # dictionary keys.
        return hash("RESOURCE_ID") + self.id.__hash__()

    def regenerate_uuid(self):
        """
        Regenerates the uuid part of the ID. Does nothing for resource
        identifiers with a user-set, fixed id.
        """
        self._uuid = str(uuid4())


class RetVal(object):
    """
    Wrapper to make tables.array.Array read only.
    """

    def __init__(self, wrapped_object):
        self.__dict__['_wrapped_object'] = wrapped_object
        attributes = dir(wrapped_object)
        for attr in attributes:
            if hasattr(self, attr):
                continue
            self.__dict__[attr] = attr

    def __setitem__(self, key, value):
        raise AttributeError('Data type is read only.')

    def __setslice__(self, i, j, value):
        raise AttributeError('Data type is read only.')

    def __setattr__(self, key, value):
        raise AttributeError('Data type is read only.')

    def __getattribute__(self, key):
        if key in ['_wrapped_object', '__dict__', '__class__']:
            return object.__getattribute__(self, key)
        return getattr(self._wrapped_object, key)

    def __getitem__(self, key):
        return self._wrapped_object.__getitem__(key)

    def __str__(self):
        return self._wrapped_object.__str__()


class H5Set(set):
    """
    An hdf5 set class for tags.
    """

    def __init__(self, h5node):
        self.h5node = h5node
        # check for already existing tags e.g. when
        # reading in a file
        f = self.h5node._v_file
        try:
            for _t in f.root.tags._v_children:
                ea = f.root.tags._v_children[_t]
                entries = ea[np.where(ea[:] == self.h5node._v_name.encode())]
                if len(entries) > 0:
                    super(H5Set, self).add(_t)
        except (KeyError, tables.NoSuchNodeError):
            pass

    def add(self, val):
        """
        Add an element to list of given tag.
        """
        f = self.h5node._v_file
        if val in self:
            return
        try:
            super(H5Set, self).add(val)
        except Exception as e:
            print(val)
            raise e
        try:
            ea = f.root.tags._v_children[val]
        except (KeyError, tables.NoSuchNodeError):
            msg = "Tag {:s} has not been registered yet. "
            msg += "Use the 'Dataset.register_tags' function first."
            raise ValueError(msg.format(val))
        found = False
        for i in range(ea.nrows):
            if ea[i] == '':
                ea[i] = np.array(
                    self.h5node._v_name, dtype='S60')
                found = True
                break
        if not found:
            ea.append(
                np.array([self.h5node._v_name], dtype='S60'))

    def append(self, val):
        """
        Append an element to the list of given tag.
        """
        self.add(val)

    def remove(self, val):
        """
        Remove element from list of given tag.
        """
        f = self.h5node._v_file
        super(H5Set, self).remove(val)
        ea = f.root.tags._v_children[val]
        ea[np.where(ea[:] == self.h5node._v_name.encode())] = np.array(
            [''], dtype='S60')
        if np.all(ea[:] == np.array('', dtype='S60')):
            f.remove_node('/tags/' + val)

    def pop(self):
        val = set.pop(self)
        self.remove(val)
        return val

    def discard(self, val):
        try:
            self.remove(val)
        except KeyError:
            pass

    def clear(self):
        while True:
            try:
                self.pop()
            except:
                break

    def update(self, vals):
        for v in vals:
            self.add(v)

    def difference_update(self, vals):
        for v in vals:
            self.discard(v)


def _buffer_property_factory(name, datatype, reference=False):
    """
    Generate properties for a buffer class based on the datatype.
    """
    # the private class attribute name
    attr_name = '_'+name

    def setter(self, value):
        self.__dict__[attr_name] = value
    fset = setter

    def getter(self):
        return self.__dict__[attr_name]
    fget = getter

    if datatype[0] == np.ndarray:
        if reference:
            def set_reference_array(self, value):
                if not isinstance(value, np.ndarray):
                    value = np.array(value, ndmin=1)
                _t = []
                for n in value:
                    if not isinstance(n, datatype[1]):
                        msg = "{:s} has to be of type: {}"
                        msg = msg.format(name, datatype[1])
                        raise ValueError(msg)
                    _t.append(str(getattr(n, '_resource_id')).encode('ascii'))
                self.__dict__[attr_name] = np.array(_t)
            fset = set_reference_array

        elif datatype[1] == datetime.datetime:
            # if the array contains datetime we need to convert
            # it into ascii byte strings as pytables can't handle datetime
            # objects
            def set_datetime_array(self, value):
                if not isinstance(value, np.ndarray):
                    value = np.array(value, ndmin=1).astype(np.str_)
                _vals = []
                for v in value:
                    _vals.append((spectroscopy.util
                                  .parse_iso_8601(v)
                                  .isoformat().encode('ascii')))
                    value = np.array(_vals)

                self.__dict__[attr_name] = np.array(_vals)
            fset = set_datetime_array

            def get_datetime_array(self):
                if self.__dict__[attr_name] is None:
                    return None
                dts = self.__dict__[attr_name]
                _vals = []
                for _dt in dts:
                    _vals.append(_dt.decode('ascii'))
                return np.array(_vals, dtype='datetime64[ms]')

            fget = get_datetime_array

        elif datatype[1] == np.str_:
            # strings are encoded into ascii byte strings
            # as this is how pytables stores strings
            # internally
            def set_string_array(self, value):
                if not isinstance(value, np.ndarray):
                    value = np.array(value, ndmin=1).astype(np.str_)
                _vals = []
                for v in value:
                    _vals.append(v.encode('ascii'))
                self.__dict__[attr_name] = np.array(_vals)
            fset = set_string_array

            def get_string_array(self):
                if self.__dict__[attr_name] is None:
                    return None
                value = self.__dict__[attr_name]
                _vals = []
                for v in value:
                    _vals.append(v.decode('ascii'))
                return np.array(_vals)
            fget = get_string_array

        else:
            def set_array(self, value):
                self.__dict__[attr_name] = (np.array(value, ndmin=1).
                                            astype(datatype[1]))
            fset = set_array

    else:
        if reference:
            def set_reference(self, value):
                if value is not None:
                    if not isinstance(value, datatype[0]):
                        msg = "{:s} has to be of type: {}"
                        msg = msg.format(name, datatype[0])
                        raise ValueError(msg)
                    rid = str(getattr(value, '_resource_id')).encode('ascii')
                else:
                    rid = None
                self.__dict__[attr_name] = rid
            fset = set_reference

        elif datatype[0] == datetime.datetime:
            def set_datetime(self, value):
                value = (spectroscopy.util
                         .parse_iso_8601(value)
                         .isoformat())
                self.__dict__[attr_name] = np.array(value.encode('ascii'))
            fset = set_datetime

            def get_datetime(self):
                if self.__dict__[attr_name] is None:
                    return None
                dt = self.__dict__[attr_name]
                return dt.astype('datetime64[s]')
            fget = get_datetime

        elif datatype[0] == np.str_:
            def set_string(self, value):
                self.__dict__[attr_name] = value.encode('ascii')
            fset = set_string

            def get_string(self):
                if self.__dict__[attr_name] is None:
                    return None
                return self.__dict__[attr_name].decode('ascii')

            fget = get_string

    return property(fget=fget, fset=fset)


def _buffer_class_factory(class_name, class_properties=[],
                          class_references=[]):
    """
    Class factory for buffer classes. These contain staged data, that
    can then be written to the HDF5 file.
    """
    cls_attrs = {}
    _properties = []
    # Assign class properties
    for item in class_properties:
        cls_attrs[item[0]] = _buffer_property_factory(item[0], item[1])
        _properties.append(item[0])
    cls_attrs['_properties'] = _properties

    # Assign references to other elements in the datamodel
    _references = []
    for item in class_references:
        cls_attrs[item[0]] = _buffer_property_factory(item[0], item[1],
                                                      reference=True)
        _references.append(item[0])
    cls_attrs['_references'] = _references

    def __init__(self, **kwargs):
        # Set all property values to None or the kwarg value.
        for key in self._properties:
            value = kwargs.pop(key, None)
            setattr(self, key, value)

        for key in self._references:
            value = kwargs.pop(key, None)
            setattr(self, key, value)

        if len(list(kwargs.keys())) > 0:
            msg = "The following names are not a "
            msg += "property or reference of class {:s}: "
            msg += ",".join(list(kwargs.keys()))
            raise AttributeError(msg.format(type(self).__name__))

    def __setattr__(self, key, value):
        prop = getattr(self.__class__, key, None)
        if isinstance(prop, property):
            if value is None:
                attr_name = '_'+key
                self.__dict__[attr_name] = None
            else:
                prop.fset(self, value)
        else:
            raise AttributeError(
                    "%s is not an attribute or reference of class %s" %
                    (key, self.__class__.__name__))

    def __str__(self):
        return class_name.strip('_')

    cls_attrs['__init__'] = __init__
    cls_attrs['__setattr__'] = __setattr__
    cls_attrs['__str__'] = __str__

    return type(class_name, (object,), cls_attrs)


def _base_property_factory(name, datatype, reference=False):
    """
    Generate properties for a base class based on the datatype.
    """

    def getter(self):
        try:
            return self._root._v_attrs[name]
        except KeyError:
            return None
    fget = getter

    if datatype[0] == np.ndarray:
        if reference:
            def get_reference_array(self):
                try:
                    value = self._root._v_attrs[name]
                except KeyError:
                    return None
                _t = []
                for val in value:
                    _t.append((ResourceIdentifier(val.decode('ascii')).
                               get_referred_object()))
                return _t
            fget = get_reference_array

        elif datatype[1] == datetime.datetime:
            # if the array contains datetime we need to convert
            # it into ascii byte strings as pytables can't handle datetime
            # objects
            def get_datetime_array(self):
                try:
                    dt = getattr(self._root, name)[:]
                    return dt.astype('datetime64[ms]')
                except tables.exceptions.NoSuchNodeError:
                    return None
            fget = get_datetime_array

        elif datatype[1] == np.str_:
            # strings are encoded into ascii byte strings
            # as this is how pytables stores strings
            # internally
            def get_string_array(self):
                try:
                    return RetVal(getattr(self._root, name))
                except tables.exceptions.NoSuchNodeError:
                    return None
            fget = get_string_array
        else:
            def get_array(self):
                try:
                    return RetVal(getattr(self._root, name))
                except tables.exceptions.NoSuchNodeError:
                    return None
            fget = get_array

    else:
        if reference:
            def get_reference(self):
                try:
                    value = self._root._v_attrs[name]
                except KeyError:
                    return None
                return (ResourceIdentifier(value.decode('ascii')).
                        get_referred_object())
            fget = get_reference

        elif datatype[0] == datetime.datetime:
            def get_datetime(self):
                try:
                    dt = self._root._v_attrs[name]
                except KeyError:
                    return None
                return dt.astype('datetime64[ms]')
            fget = get_datetime

        elif datatype[0] == np.str_:
            def get_string(self):
                try:
                    val = self._root._v_attrs[name]
                except KeyError:
                    return None
                return val.decode('ascii')
            fget = get_string
        elif name == 'tags':
            def get_tags(self):
                return self._tags
            fget = get_tags

    return property(fget=fget)


def _base_class_factory(class_name, class_type='base', class_properties=[],
                        class_references=[]):
    """
    Class factory for base classes. These are thin wrappers for the
    underlying HDF5 file with methods to write and retrieve data. Only
    buffer class instances can be written to file and no data can be
    changed once written. If a base class is extendable, it can be appended
    to.
    """
    cls_attrs = {}
    _properties = {}
    # Define class properties
    for item in class_properties:
        cls_attrs[item[0]] = _base_property_factory(item[0], item[1])
        _properties[item[0]] = item[1]
    cls_attrs['_properties'] = _properties

    # Define references to other elements in the datamodel
    _references = {}
    for item in class_references:
        cls_attrs[item[0]] = _base_property_factory(item[0], item[1],
                                                    reference=True)
        _references[item[0]] = item[1]
    cls_attrs['_references'] = _references

    def __init__(self, h5node, data_buffer=None, pedantic=False):
        # Set the parent HDF5 group after type checking
        if type(h5node) is not tables.group.Group:
            raise Exception("%s and %s are incompatible types." %
                            (type(h5node), tables.group.Group))
        self.__dict__['_root'] = h5node
        self.__dict__['_tags'] = H5Set(h5node)
        # Every time a new object is created it gets a new resource ID
        ri = ResourceIdentifier(oid=h5node._v_name, referred_object=self)
        self.__dict__['_resource_id'] = ri
        if not hasattr(h5node._v_attrs, 'creation_time'):
            self.__dict__['creation_time'] = \
                    datetime.datetime.utcnow().isoformat()
            h5node._v_attrs.creation_time = self.creation_time
        else:
            self.__dict__['creation_time'] = h5node._v_attrs.creation_time

        if data_buffer is not None:
            f = h5node._v_file
            s = hashlib.sha224()
            for key, prop_type in self._properties.items():
                private_key = '_'+key
                val = getattr(data_buffer, private_key)
                if val is None:
                    continue
                if key == 'tags':
                    for _v in val:
                        self._tags.add(_v)
                    continue
                tohash = '{}'.format(val)
                s.update(tohash.encode('ascii'))
                if prop_type[0] == np.ndarray:
                    try:
                        shape = list(val.shape)
                        shape[0] = 0
                        at = tables.Atom.from_dtype(val.dtype)
                        vl = f.create_earray(h5node, key,
                                             atom=at,
                                             shape=tuple(shape))
                    except Exception as e:
                        print(val.dtype.type)
                        raise e
                    vl.append(val)
                else:
                    h5node._v_attrs[key] = val
            for key in self._references.keys():
                private_key = '_'+key
                val = getattr(data_buffer, private_key)
                h5node._v_attrs[key] = val
            # Add a hash column to be able to avoid adding the same
            # entries more than once
            h = s.digest()
            ea = f.root.hash
            if pedantic:
                for i in range(ea.nrows):
                    if (h == ea[i]):
                        msg = "You can't add the same dataset "
                        msg += "more than once if 'pedantic=True'."
                        raise ValueError(msg)
            ea.append(np.array([h], dtype='S28'))

    def __str__(self):
        return class_name.strip('_')

    def __setattr__(self, name, value):
        msg = '{} is read only.'
        raise AttributeError(msg.format(self.__class__.__name__))

    def __repr__(self):
        msg = ''
        msg += "ID: {:s}\n".format(self._root._v_name)
        for key, datatype in list(self._properties.items()):
            if key == 'tags':
                continue
            prop = getattr(self.__class__, key, None)
            if isinstance(prop, property):
                val = prop.fget(self)
                if val is not None:
                    if datatype[0] == np.ndarray:
                        msg += "{0:s}: {1:}\n".format(key, val.shape)
                    else:
                        msg += "{0:s}: {1:}\n".format(key, val)
        msg += ("Created at: {:s}\n"
                .format(self._root._v_attrs.creation_time))
        return msg

    cls_attrs['__init__'] = __init__
    cls_attrs['__str__'] = __str__
    cls_attrs['__setattr__'] = __setattr__
    cls_attrs['__repr__'] = __repr__
    cls_attrs['__str__'] = __str__

    if class_type == 'extendable':
        def append(self, databuffer, pedantic=False):
            s = hashlib.sha224()
            for key, prop_type in self._properties.items():
                private_key = '_'+key
                val = getattr(databuffer, private_key)
                if val is None:
                    continue
                tohash = '{}'.format(val)
                s.update(tohash.encode('ascii'))
                if prop_type[0] != np.ndarray:
                    continue
                vl = getattr(self._root, key)
                vl.append(val)
            h = s.digest()
            f = self._root._v_file
            ea = f.root.hash
            if pedantic:
                for i in range(ea.nrows):
                    if (h == ea[i]):
                        msg = "You can't add the same dataset "
                        msg += "more than once if 'pedantic=True'."
                        raise ValueError(msg)
            ea.append(np.array([h], dtype='S28'))
            self.__dict__['modification_time'] = \
                datetime.datetime.utcnow().isoformat()
            self._root._v_attrs.modification_time = self.modification_time

        def __repr__(self):
            msg = ''
            msg += "ID: {:s}\n".format(self._root._v_name)
            for key, datatype in list(self._properties.items()):
                if key == 'tags':
                    continue
                prop = getattr(self.__class__, key, None)
                if isinstance(prop, property):
                    val = prop.fget(self)
                    if val is not None:
                        if datatype[0] == np.ndarray:
                            msg += "{0:s}: {1:}\n".format(key, val.shape)
                        else:
                            msg += "{0:s}: {1:}\n".format(key, val)
            ctime = self._root._v_attrs.creation_time
            msg += "Last modified at: {:s}\n".format(ctime)
            mtime = getattr(self._root._v_attrs, 'modification_time',
                            ctime)
            msg += "Created at: {:s}\n".format(mtime)
            return msg

        cls_attrs['append'] = append
        cls_attrs['__repr__'] = __repr__

    return type(class_name, (object,), cls_attrs)
