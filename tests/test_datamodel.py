import unittest

import numpy as np

from spectroscopy.dataset import Dataset, _RawData


class DatamodelTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_ram_plugin(self):
        d = Dataset.new('ram')
        p = d.plugin
        p.create_item('rawdata/someid/d_var', np.zeros((1, 2048)))
        self.assertTrue(d['rawdata/someid/d_var'].shape == (1, 2048))
        self.assertTrue(np.alltrue(d['rawdata/someid/d_var'] < 1))
        d['rawdata/someid/d_var'] = np.ones((1, 2048))
        self.assertFalse(np.alltrue(d['rawdata/someid/d_var'] < 1))

    def test_spectra(self):
        d = Dataset.new('ram')
        r = d.new_raw_data(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        self.assertTrue(np.alltrue(r.d_var < 1))
        r.angle = np.array([45.0])
        self.assertTrue(r.angle[0] == 45.0)

    def test_sum(self):
        d1 = Dataset.new('ram')
        r = d1.new_raw_data(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                            datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        d1.raw_data.append(r)
        d2 = Dataset.new('ram')
        d2.raw_data.append(s)
        d3 = d1 + d2
        self.assertEqual(len(d3.raw_data), 2)
        self.assertTrue(d3 != d2)
        self.assertTrue(d3 != d1)
        self.assertEqual(d3.raw_data[0], d3.raw_data[1])
        self.assertEqual(d3.spectra[0].counts.shape, (1, 2048))
        with self.assertRaises(TypeError):
            d4 = d1 + s
        d5 = Dataset.new('ram')
        d5 += d1
        self.assertEqual(d5.spectra[0], d1.spectra[0])

    def test_forbidden(self):
        d = Dataset.new('ram')
        d.new_target(resource_id=5)
        t = d.new_target(position=(1, 2, 3))
        t.resource_id = 5
        with self.assertRaises(AttributeError):
            t.position[0] = 5.

    def test_append(self):
        d1 = Dataset.new('ram')
        d1.register_tag('measurement')
        t = d1.new_target(tag='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        i = d1.new_instrument(tag='MD01', sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        rdt = d1.new_rawdata_type(tag='measurement',
                                  name='1st round measurements',
                                  acquisition='stationary')
        r = d1.new_raw_data(target=t, instrument=i, dtype=rdt,
                            d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                            datetime=np.array(np.datetime64('2017-01-10T15:23:00')))

    def test_dtbuffer(self):
        d = Dataset('deleteme.h5', 'w')
        tb = TargetBuffer(tag='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new_target(tb)
        ib = InstrumentBuffer(tag='MD01', sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new_instrument(ib)
        rdtb = RawDataTypeBuffer(tag='measurement',
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new_raw_data_type(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, dtype=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        r = d.new_raw_data(rb)


def suite():
    return unittest.makeSuite(DatamodelTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
