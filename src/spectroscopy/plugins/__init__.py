import os
import warnings

registered_plugins = {}

class DatasetPluginBase(object):
    def get_item(self, path):
        pass

    def set_item(self, path, value):
        pass

    def create_item(self, path, value):
        pass

    def shape(self, path):
        pass

    def append(self, path, value):
        pass

    def get_type(self, path):
        pass

    def delete_branch(self, path):
        pass

    @staticmethod
    def get_format():
        return 'ram'


def load_all_plugins():
    """
    Loads all installed spectroscopy dataset plug-ins. Plugins that cannot be loaded
    will be skipped and a warning message issued.
    """
    # import all the plugins from the plugins directory
    plugins_directory = __path__[0]
    
    cur_dir = os.getcwd()
    os.chdir(plugins_directory)
    
    #only attempt to import python files and directories   
    plugin_list = []
    for f in [f for f in os.listdir(os.path.curdir)]:
        #exclude any hidden files
        if f.startswith('.'):
            continue
        
        #include any python modules
        if f.endswith('.py'):
            plugin_list.append(f)
            continue
        
        #include directories only if they are Python packages (look 
        #for __init__.py file)
        if os.path.isdir(f):
            if '__init__.py' in os.listdir(f):
                plugin_list.append(f)
    
    for plugin in plugin_list:
        try:
            __import__("spectroscopy.plugins." + plugin.rstrip(".py"),
                       fromlist=["spectroscopy.plugins"], globals=globals(),
                       locals=locals())
        except Exception, e:
            #skip over any plugins that we cannot import
            warnings.warn('Failed to import plug-in \'%s\'. \n\nregister() '
                          'raised the exception: \'%s\'.' % (plugin, e.args[0]))
        
    #return to the old working dir
    os.chdir(cur_dir)
    
    registered_plugins = {}
    
    for c in DatasetPluginBase.__subclasses__:
        assert c.get_format() not in registered_plugins
        
        registered_plugins[c.get_format()] = c
    
    return registered_plugins


def get_registered_plugins():

    if hasattr(get_registered_plugins, 'registered_plugins'):
        return get_registered_plugins.registered_plugins
    else:
        load_all_plugins()
        get_registered_plugins.registered_plugins = registered_plugins
        return get_registered_plugins.registered_plugins
    
    