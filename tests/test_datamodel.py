import tempfile
import unittest


import numpy as np

from spectroscopy.dataset import (Dataset, RawDataBuffer, TargetBuffer,
                                  InstrumentBuffer, RawDataTypeBuffer,
                                  ConcentrationBuffer, GasFlowBuffer,
                                  FluxBuffer, PreferredFluxBuffer,
                                  _Target)


class DatamodelTestCase(unittest.TestCase):

    def setUp(self):
        pass

    def test_DataElementBase(self):
        d = Dataset(tempfile.mktemp(),'w')
        tb = TargetBuffer(target_id='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new(tb)
        np.testing.assert_almost_equal(np.squeeze(t.position[:]),np.array([177.2, -37.5, 50]),1)
        self.assertEqual(t.target_id,'WI001')
        with self.assertRaises(AttributeError):
            t.position = (177.2, -37.5, 50)
        with self.assertRaises(AttributeError):
            t.target_id = 'WI002'
        tid = t.target_id
        tid = 'WI002'
        self.assertEqual(t.target_id,'WI001')

    def test_raw_data(self):
        d = Dataset(tempfile.mktemp(), 'w')
        rb = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime='2017-01-10T15:23:00')
        r = d.new(rb)
        self.assertTrue(np.alltrue(r.d_var[:] < 1))

    def test_ResourceIdentifiers(self):
        d = Dataset(tempfile.mktemp(),'w')
        tb = TargetBuffer(target_id='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new(tb)
        rb = RawDataBuffer(target=t,d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime='2017-01-10T15:23:00')
        r = d.new(rb)
        self.assertEqual(r.target.target_id,'WI001')


    def test_sum(self):
        d1 = Dataset(tempfile.mktemp(), 'w')
        rb = RawDataBuffer()
        r = d1.new_raw_data(rb)
        d2 = Dataset(tempfile.mktemp(), 'w')
        d2.new_raw_data(rb)
        with self.assertRaises(AttributeError):
            d3 = d1 + d2
        d3 = Dataset(tempfile.mktemp(), 'w')
        d3 += d1
        d3 += d2
        self.assertEqual(len(d3.raw_data), 2)
        self.assertNotEqual(d3.raw_data[0], d1.raw_data[0])
        tmp = {}
        tmp.update(d1._rids)
        tmp.update(d2._rids)
        self.assertTrue(tmp == d3._rids)
        self.assertTrue(d3._tags == d1._tags + d2._tags)
        with self.assertRaises(TypeError):
            d4 = d1 + rb
        d1 += d2
        with self.assertRaises(ValueError):
            d1 += d1

    def test_forbidden(self):
        d = Dataset(tempfile.mktemp(), 'w')
        with self.assertRaises(AttributeError):
            tb = TargetBuffer(resource_id=5.)
        tb = TargetBuffer()
        with self.assertRaises(AttributeError):
            tb.blub = 5.
        t = d.new_target(tb)
        with self.assertRaises(AttributeError):
            t.position = (1, 1, 1)
        with self.assertRaises(AttributeError):
            d.raw_data = []
        d.raw_data[0] = 5
        self.assertNotEqual(d.raw_data[0], 5)
        rb = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        r = d.new_raw_data(rb)
        with self.assertRaises(AttributeError):
            r.d_var[0] = 1
        with self.assertRaises(AttributeError):
            r.d_var[0:2] = 1
        with self.assertRaises(AttributeError):
            r.d_var = np.ones((1, 2048))

        out = [i for i in r.d_var.iterrows()]
        self.assertEqual(np.zeros(2048), np.array(out))

    def test_pedantic(self):
        d = Dataset(tempfile.mktemp(), 'w', pedantic=True)
        rb = RawDataBuffer()
        with self.assertRaises(ValueError):
            d.new_raw_data(rb)
        d.register_tags(['WI001'])
        tb = TargetBuffer(tag='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        d.new_target(tb)
        with self.assertRaises(ValueError):
            d.new_target(tb)

    def test_append(self):
        d = Dataset(tempfile.mktemp(), 'w')
        d.register_tags(['WI001', 'MD01', 'measurement'])
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
        rb = RawDataBuffer(target=t, instrument=i, datatype=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        r = d.new_raw_data(rb)
        rb1 = RawDataBuffer(target=t, instrument=i, datatype=rdt,
                            d_var=np.ones((1, 2048)), ind_var=np.arange(2048),
                            datetime=np.array(np.datetime64('2017-01-10T15:23:01')))
        r.append(rb1)
        with self.assertRaises(ValueError):
            r.append(rb1)
        with self.assertRaises(AttributeError):
            t.append(tb)
        tb1 = TargetBuffer(tag='WI002', name='Donald Duck',
                           position=(177.1, -37.4, 50),
                           position_error=(0.2, 0.2, 20),
                           description='Donald Duck vent in January 2010')
        t1 = d.new_target(tb1)
        rb2 = RawDataBuffer(target=t1, instrument=i, datatype=rdt,
                            d_var=np.ones((1, 2048)), ind_var=np.arange(2048),
                            datetime=np.array(np.datetime64('2017-01-10T15:23:02')))
        with self.assertRaises(ValueError):
            rb.append(rb2)

    def test_tagging(self):
        d = Dataset(tempfile.mktemp(), 'w')
        d.register_tags(['measurement'])
        with self.assertRaises(ValueError):
            d.register_tags(['measurement'])
        tb = TargetBuffer(tags=['WI001', 'Eruption16'],
                          name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new_target(tb)
        t.tags.append('SomethingElse')
        t.tags.remove('WI001')
        d.remove_tag('Eruption16')
        self.assertEqual(t.tags, ['SomethingElse'])

    def test_dtbuffer(self):
        d = Dataset(tempfile.mktemp(), 'w')
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new_target(tb)
        ib = InstrumentBuffer(tags=['MD01'], sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new_instrument(ib)
        rdtb = RawDataTypeBuffer(tags=['measurement'],
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new_raw_data_type(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, datatype=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        r = d.new_raw_data(rb)
        self.assertTrue(r.target == t)
        self.assertTrue(r.instrument == i)
        self.assertTrue(r.dtype == rdt)

        rb1 = RawDataBuffer()
        rb1.d_var = np.zeros((1, 2048))
        rb1.ind_var = np.arange(2048),
        rb1.datetime = np.array(np.datetime64('2017-01-10T15:23:00'))
        rb1.target = t
        rb1.instrument = i
        rb1.datatype = rdt
        r1 = d.new_raw_data(rb1)

    def test_times(self):
        d = Dataset(tempfile.mktemp(), 'w')
        rb = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=np.array(np.datetime64('2017-01-10T15:23:00')))
        r = d.new_raw_data(rb)
        rb1 = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                            datetime=np.array(np.datetime64('2017-01-10T15:23:01')))
        self.assertEqual(r.creation_time, r.modification_time)
        ct = r.creation_time
        r.append(rb1)
        self.assertGreater(r.modification_time, r.creation_time)
        self.assertEqual(r.creation_time, ct)


def suite():
    return unittest.makeSuite(DatamodelTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
