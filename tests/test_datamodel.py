import tempfile
import unittest
import warnings

import numpy as np

from spectroscopy.datamodel import (RawDataBuffer, TargetBuffer,
                                  InstrumentBuffer, RawDataTypeBuffer,
                                  ConcentrationBuffer, GasFlowBuffer,
                                  FluxBuffer, PreferredFluxBuffer,
                                  _Target, MethodBuffer)
from spectroscopy.dataset import Dataset

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
        self.assertEqual(t.target_id[0],'WI001')
        with self.assertRaises(AttributeError):
            t.position = (177.2, -37.5, 50)
        with self.assertRaises(AttributeError):
            t.target_id = 'WI002'
        tid = t.target_id
        tid = 'WI002'
        self.assertEqual(t.target_id[0],'WI001')

    def test_typechecking(self):
        """
        Test the type checking and conversion functionality.
        """
        with self.assertRaises(ValueError):
            tb1 = TargetBuffer(target_id='WI001', name='White Island main vent',
                              position=('a', -37.5, 50))
        d = Dataset(tempfile.mktemp(),'w')
        tb2 = TargetBuffer(target_id='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new(tb2)
        with self.assertRaises(ValueError):
            rb = RawDataBuffer(instrument=t,d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                               datetime=['2017-01-10T15:23:00'])

    def test_RawData(self):
        d = Dataset(tempfile.mktemp(), 'w')
        times = [str(np.datetime64('2017-01-10T15:23:00') + np.timedelta64(i*1,'s')) for i in range(10)]
        rb = RawDataBuffer(d_var=np.zeros((10, 2048)),
                           ind_var=np.arange(2048),
                           datetime=times, inc_angle=np.arange(10,110,10))
        r = d.new(rb)
        self.assertEqual(r.d_var.shape, (10, 2048))
        self.assertTrue(np.alltrue(r.d_var[0] < 1))
        self.assertEqual(r.datetime[0],'2017-01-10T15:23:00')

    def test_PreferredFlux(self):
        d = Dataset(tempfile.mktemp(), 'w')
        pfb = PreferredFluxBuffer(datetime=['2017-01-10T15:23:00','2017-01-11T15:23:00'])
        pf = d.new(pfb)
        np.testing.assert_array_equal(pf.datetime[:],['2017-01-10T15:23:00', '2017-01-11T15:23:00'])

    def test_ResourceIdentifiers(self):
        d = Dataset(tempfile.mktemp(),'w')
        tb = TargetBuffer(target_id='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new(tb)
        rb = RawDataBuffer(target=t,d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        self.assertEqual(r.target.target_id[:],'WI001')


    def test_sum(self):
        d1 = Dataset(tempfile.mktemp(), 'w')
        tb = TargetBuffer(target_id='WI001', name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d1.new(tb)
        rb = RawDataBuffer(target=t,d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d1.new(rb)
        d2 = Dataset(tempfile.mktemp(), 'w')
        tb2 = TargetBuffer(target_id='WI002', name='White Island main vent',
                           position=(177.2, -37.5, 50),
                           position_error=(0.2, 0.2, 20),
                           description='Main vent in January 2017')
        t2 = d2.new(tb2)
        rb2 = RawDataBuffer(target=t2,d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        d2.new(rb2)
        with self.assertRaises(AttributeError):
            d3 = d1 + d2
        d3 = Dataset(tempfile.mktemp(), 'w')
        d3 += d1
        d3 += d2
        self.assertEqual(len(d3.elements['RawData']), 2)
        rc3 = d3.elements['RawData'][0]
        rc2 = d2.elements['RawData'][0]
        rc4 = d3.elements['RawData'][1]
        rc1 = d1.elements['RawData'][0]
        # Check that the references are not the same anymore...
        self.assertNotEqual(getattr(rc3._root.data.cols,'target')[0],
                            getattr(rc1._root.data.cols,'target')[0])
        # ...but that the copied elements contain the same information
        self.assertEqual(rc3.target.target_id[:],rc1.target.target_id[:])
        self.assertEqual(rc4.target.target_id[:],rc2.target.target_id[:])
        
        # Now check that this is also working for arrays of references
        mb1 = MethodBuffer(name='Method1')
        mb2 = MethodBuffer(name='Method2')
        d4 = Dataset(tempfile.mktemp(), 'w')
        m1 = d4.new(mb1)
        m2 = d4.new(mb2)
        gfb= GasFlowBuffer(methods=[m1,m2])
        gf = d4.new(gfb)
        d3 += d4
        gf2 = d3.elements['GasFlow'][0]
        self.assertNotEqual(getattr(gf2._root.data.cols,'methods')[0][0],
                            getattr(gf._root.data.cols,'methods')[0][0])
        self.assertEqual(gf2.methods[0].name[:],gf.methods[0].name[:])
        self.assertEqual(gf2.methods[1].name[:],gf.methods[1].name[:])
         ## ToDo: not sure what the _rids feature was there for 
        #tmp = {}
        #tmp.update(d1._rids)
        #tmp.update(d2._rids)
        #self.assertTrue(tmp == d3._rids)
        #self.assertTrue(d3._tags == d1._tags + d2._tags)
        with self.assertRaises(AttributeError):
            d4 = d1 + rb
        # ToDo: also not sure what behaviour we expected from
        # the following line
        # d1 += d2
        with self.assertRaises(ValueError):
            d1 += d1

    def test_forbidden(self):
        d = Dataset(tempfile.mktemp(), 'w')
        with self.assertRaises(AttributeError):
            tb = TargetBuffer(blub=10)
        with self.assertRaises(AttributeError):
            tb = TargetBuffer(resource_id=5.)
        tb = TargetBuffer()
        with self.assertRaises(AttributeError):
            tb.blub = 5.
        t = d.new(tb)
        with self.assertRaises(AttributeError):
            t.position = (1, 1, 1)
        rb = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        with self.assertRaises(AttributeError):
            r.d_var[0] = 1
        with self.assertRaises(AttributeError):
            r.d_var[0:2] = 1
        with self.assertRaises(AttributeError):
            r.d_var = np.ones((1, 2048))
        with self.assertRaises(AttributeError):
            r.blub

        np.testing.assert_array_equal(np.zeros(2048), np.array(r.d_var[0][0]))

    def test_pedantic(self):
        d = Dataset(tempfile.mktemp(), 'w')
        rb = RawDataBuffer()
        with self.assertRaises(ValueError):
            d.new(rb, pedantic=True)
        d.register_tags(['WI001'])
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        d.new(tb)
        with self.assertRaises(ValueError):
            d.new(tb, pedantic=True)

    def test_append(self):
        d = Dataset(tempfile.mktemp(), 'w')
        d.register_tags(['WI001', 'MD01', 'measurement'])
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        t = d.new(tb)
        ib = InstrumentBuffer(tags=['MD01'], sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new(ib)
        rdtb = RawDataTypeBuffer(tags=['measurement'],
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, type=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        rb1 = RawDataBuffer(target=t, instrument=i, type=rdt,
                            d_var=np.ones((1, 2048)), ind_var=np.arange(2048),
                            datetime=['2017-01-10T15:23:01'])
        r.append(rb1)
        self.assertEqual(len(r.ind_var[:]),4096)
        self.assertEqual(np.array(r.ind_var[:]).size,4096)
        self.assertTrue(np.alltrue(np.array(r.d_var[:]) < 2))
        np.testing.assert_array_equal(np.array(r.datetime[:]).flatten(),['2017-01-10T15:23:00','2017-01-10T15:23:01'])
        with self.assertRaises(ValueError):
            r.append(rb1)
        with self.assertRaises(AttributeError):
            t.append(tb)
        d.register_tags(['WI002'])
        tb1 = TargetBuffer(tags=['WI002'], name='Donald Duck',
                           position=(177.1, -37.4, 50),
                           position_error=(0.2, 0.2, 20),
                           description='Donald Duck vent in January 2010')
        t1 = d.new(tb1)
        rb2 = RawDataBuffer(target=t1, instrument=i, type=rdt,
                            d_var=np.ones((1, 2048)), ind_var=np.arange(2048),
                            datetime=['2017-01-10T15:23:02'])
        with self.assertRaises(AttributeError):
            rb.append(rb2)

    def test_read(self):
        """
        Test reading of HDF5 files.
        """
        fn = tempfile.mktemp() 
        d = Dataset(fn, 'w')
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        d.register_tags(['WI001','MD01','measurement'])
        t = d.new(tb)
        ib = InstrumentBuffer(tags=['MD01'], sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new(ib)
        rdtb = RawDataTypeBuffer(tags=['measurement'],
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, type=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        d.close()

        d1 = Dataset.open(fn)
        r1 = d1.elements['RawData'][0]
        self.assertEqual(r1.target.name[:][0],'White Island main vent')
        self.assertEqual(list(r1.instrument.tags)[0],'MD01')

    def test_tagging(self):
        """
        Test the tagging of data elements.
        """
        d = Dataset(tempfile.mktemp(), 'w')
        d.register_tags(['measurement'])
        with self.assertRaises(ValueError):
            d.register_tags(['measurement'])

        tb = TargetBuffer(tags=['WI001', 'Eruption16'])
        with self.assertRaises(ValueError):
            t = d.new(tb)

        d.register_tags(['WI001','Eruption16'])
        t = d.new(tb)
        d.register_tags(['SomethingElse'])
        t.tags.append('SomethingElse')
        t.tags.remove('WI001')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            d.remove_tags(['Eruption16','blub'])
        self.assertEqual(list(t.tags), ['SomethingElse'])
        
        # Ensure the same tag is only added once
        t.tags.append('SomethingElse')
        self.assertEqual(list(t.tags), ['SomethingElse'])
        self.assertEqual(len(d._f.root.tags._v_children['SomethingElse'][:]), 1)


    def test_dtbuffer(self):
        """
        Testing the behaviour of buffer elements.
        """
        d = Dataset(tempfile.mktemp(), 'w')
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        with self.assertRaises(ValueError):
            t = d.new(tb)
        d.register_tags(['WI001','MD01','measurement'])
        t = d.new(tb)
        ib = InstrumentBuffer(tags=['MD01'], sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new(ib)
        rdtb = RawDataTypeBuffer(tags=['measurement'],
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, type=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        self.assertTrue(r.target == t)
        self.assertTrue(r.instrument == i)
        self.assertTrue(r.type == rdt)

        rb1 = RawDataBuffer()
        rb1.d_var = np.zeros((1, 2048))
        rb1.ind_var = np.arange(2048),
        rb1.datetime = ['2017-01-10T15:23:00']
        rb1.target = t
        rb1.instrument = i
        rb1.type = rdt
        r1 = d.new(rb1)

    def test_times(self):
        d = Dataset(tempfile.mktemp(), 'w')
        rb = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        rb1 = RawDataBuffer(d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                            datetime=['2017-01-10T15:23:01'])
        self.assertEqual(r.creation_time, r.modification_time)
        ct = r.creation_time
        r.append(rb1)
        self.assertGreater(r.modification_time, r.creation_time)
        self.assertEqual(r.creation_time, ct)

    @unittest.skip("Skipping")
    def test_select(self):
        d = Dataset(tempfile.mktemp(), 'w')
        tb = TargetBuffer(tags=['WI001'], name='White Island main vent',
                          position=(177.2, -37.5, 50),
                          position_error=(0.2, 0.2, 20),
                          description='Main vent in January 2017')
        d.register_tags(['WI001','MD01','measurement'])
        t = d.new(tb)
        ib = InstrumentBuffer(tags=['MD01'], sensor_id='F00975',
                              location='West rim',
                              no_bits=16, type='DOAS',
                              description='GeoNet permanent instrument')
        i = d.new(ib)
        rdtb = RawDataTypeBuffer(tags=['measurement'],
                                 name='1st round measurements',
                                 acquisition='stationary')
        rdt = d.new(rdtb)
        rb = RawDataBuffer(target=t, instrument=i, type=rdt,
                           d_var=np.zeros((1, 2048)), ind_var=np.arange(2048),
                           datetime=['2017-01-10T15:23:00'])
        r = d.new(rb)
        
        e = d.select("tags == 'MD01'")
        self.assertEqual(e['Target'][0], t)
        self.assertEqual(e['Instrument'][0], i)

        e = d.select("type.acquisition == 'stationary'", etype='RawData')
        self.assertEqual(e['RawData'][0], r)


def suite():
    return unittest.makeSuite(DatamodelTestCase, 'test')

if __name__ == '__main__':
    unittest.main(defaultTest='suite')
