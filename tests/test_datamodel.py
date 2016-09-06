import unittest

import numpy as np

from spectroscopy.dataset import Dataset, Spectra

class DatamodelTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_ram_plugin(self):
        d = Dataset.new('ram')
        p = d.plugin
        p.create_item('spectra/someid/counts', np.zeros((1, 2048)))
        self.assertTrue(d['spectra/someid/counts'].shape == (1, 2048))
        self.assertTrue(np.alltrue(d['spectra/someid/counts'] < 1))
        d['spectra/someid/counts'] = np.ones((1, 2048))
        self.assertFalse(np.alltrue(d['spectra/someid/counts'] < 1))

    def test_spectra(self):
        d = Dataset.new('ram')
        s = Spectra(d.plugin, counts=np.zeros((1, 2048)))
        self.assertTrue(np.alltrue(s.counts < 1))
        s.angle = np.array([45.0])
        self.assertTrue(s.angle[0] == 45.0)
        d.spectra.append(s.resource_id.id)
        print d.spectra

def suite():
    return unittest.makeSuite(DatamodelTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
