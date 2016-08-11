from spectroscopy.plugins import get_registered_plugins
import numpy as np


class Dataset(object):

    def __init__(self, plugin):
        self._root = plugin
        self._root.create_item('spectra/blub/counts', np.zeros((1, 2048)))

    @staticmethod
    def new(format, filename=None):
        plugins = get_registered_plugins()
        if format not in plugins:
            raise Exception('Format %s is not supported.' % format)
        _p = plugins[format]()
        _p.new(filename)
        return Dataset(_p)

    def __getitem__(self, path):
        return self._root.get_item(path)

    def __setitem__(self, path, value):
        return self._root.set_item(path, value)



    def open(self, filename, format):
        pass
