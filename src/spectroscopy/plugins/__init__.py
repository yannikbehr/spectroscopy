import os
import warnings
import numpy as np


class RamNode(object): pass

class DatasetPluginBase(object):
    """
    Default plugin to keep a Dataset instance in memory.
    """

    def __init__(self):
        self._root = None

    def open(self, filename, format=None):
        raise Exception('Open is undefined for the RAM backend')

    def close(self, filename):
        raise Exception('Close is undefined for the RAM backend')

    def flush(self):
        pass

    def new(self, filename=None):
        self._root = RamNode()

    def get_item(self, path):
        branches = path.split('/')
        parent = self.getattrec(self._root, branches)
        leaf = getattr(parent, branches[0])
        if isinstance(leaf, RamNode):
            raise Exception('Only end nodes can be retrieved.')
        return leaf

    def set_item(self, path, value):
        branches = path.split('/')
        parent = self.getattrec(self._root, branches)
        leaf = getattr(parent, branches[0])
        if isinstance(leaf, RamNode):
            raise Exception('Only end nodes can be set.')
        setattr(parent, branches[0], value)

    def create_item(self, path, value):
        branches = path.split('/')
        current_branch = self._root
        while len(branches) > 1:
            branch = branches.pop(0)
            if not hasattr(current_branch, branch):
                setattr(current_branch, branch, RamNode())
            current_branch = getattr(current_branch, branch)
        setattr(current_branch, branches.pop(0), value)

    def shape(self, path):
        return self.get_item(path).shape

    def append(self, path, value):
        _val = np.array(value)
        if len(self.shape(path)) != len(_val.shape):
            _val = _val.reshape([1] + list(_val.shape))
        self.set_item(path, np.append(self.get_item(path), _val, axis=0))

    def delete_branch(self, path):
        branches = path.split('/')
        parent = self.getattrec(self._root, branches)
        delattr(parent, branches[0])

    @staticmethod
    def get_format():
        return 'ram'

    def getattrec(self, c, namelist):
        if len(namelist) == 1:
            return c
        return self.getattrec(getattr(c, namelist.pop(0)), namelist)

def load_all_plugins():
    """
    Loads all installed spectroscopy dataset plug-ins. Plugins that cannot be loaded
    will be skipped and a warning message issued.
    """
    # import all the plugins from the plugins directory
    plugins_directory = __path__[0]

    cur_dir = os.getcwd()
    os.chdir(plugins_directory)

    # only attempt to import python files and directories
    plugin_list = []
    for f in [f for f in os.listdir(os.path.curdir)]:
        # exclude any hidden files
        if f.startswith('.'):
            continue

        # include any python modules
        if f.endswith('.py'):
            plugin_list.append(f)
            continue

        # include directories only if they are Python packages (look
        # for __init__.py file)
        if os.path.isdir(f):
            if '__init__.py' in os.listdir(f):
                plugin_list.append(f)

    for plugin in plugin_list:
        try:
            __import__("spectroscopy.plugins." + plugin.rstrip(".py"),
                       fromlist=["spectroscopy.plugins"], globals=globals(),
                       locals=locals())
        except Exception, e:
            # skip over any plugins that we cannot import
            warnings.warn('Failed to import plug-in \'%s\'. \n\nregister() '
                          'raised the exception: \'%s\'.' % (plugin, e.args[0]))

    # return to the old working dir
    os.chdir(cur_dir)

    registered_plugins = {DatasetPluginBase.get_format():DatasetPluginBase}

    for c in DatasetPluginBase.__subclasses__():
        assert c.get_format() not in registered_plugins

        registered_plugins[c.get_format()] = c

    return registered_plugins


def get_registered_plugins():

    if hasattr(get_registered_plugins, 'registered_plugins'):
        return get_registered_plugins.registered_plugins
    else:
        get_registered_plugins.registered_plugins = load_all_plugins()
        return get_registered_plugins.registered_plugins

