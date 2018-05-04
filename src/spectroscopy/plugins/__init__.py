import os
import warnings


class DatasetPluginBaseException(Exception):
    pass


class DatasetPluginBase(object):
    """
    Default plugin to keep a Dataset instance in memory.
    """

    def read(self, dataset, filename, **kargs):
        raise Exception("'read' is undefined")

    def write(self, dataset, filename, **kargs):
        raise Exception("'write' is undefined")

    def close(self, filename):
        raise Exception("'close' is undefined")

    @staticmethod
    def get_format():
        return 'base'


def load_all_plugins():
    """
    Loads all installed spectroscopy dataset plug-ins.
    Plugins that cannot be loaded will be skipped and
    a warning message issued.
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
        except Exception as e:
            # skip over any plugins that we cannot import
            msg = "Failed to import plug-in {}\n"
            msg += "register() raised the exception: {}"
            warnings.warn(msg.format(plugin, e.args[0]))

    # return to the old working dir
    os.chdir(cur_dir)

    registered_plugins = {}

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
