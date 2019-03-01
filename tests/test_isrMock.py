#
# LSST Data Management System
# Copyright 2008-2017 AURA/LSST.
#
# This product includes software developed by the
# LSST Project (http://www.lsst.org/).
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the LSST License Statement and
# the GNU General Public License along with this program.  If not,
# see <https://www.lsstcorp.org/LegalNotices/>.
#

import unittest
import numpy as np

import lsst.utils.tests

import lsst.afw.image as afwImage
import lsst.ip.isr.isrMock as isrMock


class IsrMockCases(lsst.utils.tests.TestCase):
    r"""Test the generation of IsrMock data.
    """
    def setUp(self):
        self.inputExp = isrMock.TrimmedRawMock().run()
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        try:
            del self.inputExp
            del self.mi
        except Exception:
            pass

    def test_simple(self):
        r"""Chain raw and calibration mock data.

        This test should confirm the raw data is generated as expected.
        """
        fringe = isrMock.FringeMock().run()

        initialMean = np.median(self.mi.getImage().getArray()[:])
        initialStd = np.std(self.mi.getImage().getArray()[:])

        self.mi.getImage().getArray()[:] = (self.mi.getImage().getArray()[:] -
                                            fringe.getMaskedImage().getImage().getArray()[:])
        newMean = np.median(self.mi.getImage().getArray()[:])
        newStd = np.std(self.mi.getImage().getArray()[:])

        self.assertAlmostEqual(newMean, initialMean, -2)
        #        self.assertLess(newStd, initialStd)

        initialMean = newMean
        initialStd = newStd

        flat = isrMock.FlatMock().run()

        self.mi.getImage().getArray()[:] = (self.mi.getImage().getArray()[:] -
                                            flat.getMaskedImage().getImage().getArray()[:])
        newMean = np.median(self.mi.getImage().getArray()[:])
        newStd = np.std(self.mi.getImage().getArray()[:])

        self.assertAlmostEqual(newMean, initialMean, -2)
        # self.assertLess(newStd, initialStd)

        dark = isrMock.DarkMock().run()

        initialMean = newMean
        initialStd = newStd

        flat = isrMock.FlatMock().run()

        self.mi.getImage().getArray()[:] = (self.mi.getImage().getArray()[:] -
                                            dark.getMaskedImage().getImage().getArray()[:])
        newMean = np.median(self.mi.getImage().getArray()[:])
        newStd = np.std(self.mi.getImage().getArray()[:])

        self.assertLess(newMean, initialMean)

        bias = isrMock.BiasMock().run()

        initialMean = newMean
        initialStd = newStd

        flat = isrMock.FlatMock().run()

        self.mi.getImage().getArray()[:] = (self.mi.getImage().getArray()[:] -
                                            bias.getMaskedImage().getImage().getArray()[:])
        newMean = np.median(self.mi.getImage().getArray()[:])
        newStd = np.std(self.mi.getImage().getArray()[:])

        self.assertLess(newMean, initialMean)

    def test_untrimmedSimple(self):
        r"""Confirm untrimmed data classes are generated consistently.
        """
        exposure = isrMock.RawMock().run()
        fringe = isrMock.UntrimmedFringeMock().run()

        initialStd = np.std(exposure.getMaskedImage().getImage().getArray()[:])

        diff = (exposure.getMaskedImage().getImage().getArray()[:] -
                fringe.getMaskedImage().getImage().getArray()[:])

        newStd = np.std(diff[:])

        #        self.assertLess(newStd, initialStd)

    def test_productTypes(self):
        r"""Test non-image data is returned as the expected type.
        """
        assert isinstance(isrMock.BfKernelMock().run(), np.ndarray)
        assert isinstance(isrMock.CrosstalkCoeffMock().run(), np.ndarray)

        assert len(isrMock.DefectMock().run()) > 0
        assert isinstance(isrMock.DefectMock().run()[0], lsst.meas.algorithms.Defect)

        assert isinstance(isrMock.TransmissionMock().run(), afwImage.TransmissionCurve)

    def test_edgeCases(self):
        r"""Test that improperly specified configurations do not return data.
        """
        config = isrMock.IsrMockConfig()
        assert isrMock.IsrMock(config=config).run() is None

        config.doGenerateData = True
        assert isrMock.IsrMock(config=config).run() is None


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
