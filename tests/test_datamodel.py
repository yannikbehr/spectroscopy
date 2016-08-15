import unittest

import numpy as np

from spectroscopy import dataset

class DatamodelTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_ram_plugin(self):
        d = dataset.Dataset.new('ram')
        d._root.create_item('spectra/someid/counts', np.zeros((1, 2048)))
        self.assertTrue(d['spectra/someid/counts'].shape == (1, 2048))
        self.assertTrue(np.alltrue(d['spectra/someid/counts'] < 1))
        d['spectra/someid/counts'] = np.ones((1, 2048))
        self.assertFalse(np.alltrue(d['spectra/someid/counts'] < 1))


def suite():
    return unittest.makeSuite(DatamodelTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
