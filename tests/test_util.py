import unittest

import numpy as np

from spectroscopy.util import split_by_scan, _array_multi_sort


class UtilTestCase(unittest.TestCase):
    """
    Test plugin to read FlySpec data.
    """

    def test_split_by_scan(self):
        angles = np.array([30, 35, 40, 35, 30, 35, 40])
        result = [np.array([30, 35, 40]), np.array([30, 35]),
                  np.array([35, 40])]
        for i, a in enumerate(split_by_scan(angles)):
            np.testing.assert_array_equal(a[0], result[i])

        result1 = [np.array([1, 2, 3]), np.array([5, 4]), np.array([6, 7])]
        for i, a in enumerate(split_by_scan(angles,
                                            np.array([1, 2, 3, 4, 5, 6, 7]))):
            np.testing.assert_array_equal(a[1], result1[i])

        angles1 = np.array([30, 30, 35, 40, 35, 30, 35, 40, 40])
        result2 = [np.array([30, 30, 35, 40]), np.array([30, 35]),
                   np.array([35, 40, 40])]
        for i, a in enumerate(split_by_scan(angles1)):
            np.testing.assert_array_equal(a[0], result2[i])

        angles2 = np.array([30, 35, 40, 45, 30, 35, 40, 45])
        result3 = [np.array([30, 35, 40, 45]),
                   np.array([30, 35, 40, 45])]
        for i, a in enumerate(split_by_scan(angles2)):
            np.testing.assert_array_equal(a[0], result3[i])

        angles3 = np.array([30., 35., 40., 40., 45., 30., 35., 40., 45.])
        result4 = [np.array([30, 35, 40, 40, 45]),
                   np.array([30, 35, 40, 45])]
        for i, a in enumerate(split_by_scan(angles3)):
            np.testing.assert_array_equal(a[0], result4[i])

        angles4 = np.array([30, 35, 40, 40, 40, 45, 30, 35, 40, 45])
        with self.assertRaises(ValueError):
            [a for a in split_by_scan(angles4)]

        angles5 = np.array([174.750, 174.750, 174.420, 174.090, 173.750,
                            173.420, 173.080, 172.750, 172.420, 172.080,
                            171.750, 171.750, 171.410, 171.080, 170.740])
        result5 = [angles5[::-1]]
        for i, a in enumerate(split_by_scan(angles5)):
            np.testing.assert_array_equal(a[0], result5[i])

    def test_array_multi_sort(self):
        x1 = np.array([4., 5., 1., 2.])
        x2 = np.array([10., 11., 12., 13.])
        result = (np.array([1.,  2.,  4.,  5.]),
                  np.array([12.,  13.,  10.,  11.]))
        out = _array_multi_sort(*tuple([x1, x2]))
        np.testing.assert_array_equal(out[0], result[0])
        np.testing.assert_array_equal(out[1], result[1])


def suite():
    return unittest.makeSuite(UtilTestCase, 'test')


if __name__ == '__main__':
    unittest.main(defaultTest='suite')
