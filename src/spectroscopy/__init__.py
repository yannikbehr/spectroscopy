

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
    


def get_registered_plugins():

    if hasattr(get_registered_plugins, 'registered_plugins'):
        return get_registered_plugins.registered_plugins
    else:
        load_all_plugins()
        get_registered_plugins.registered_plugins = registered_plugins
        return get_registered_plugins.registered_plugins