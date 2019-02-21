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

# import lsst.meas.algorithms as measAlg

# import lsst.afw.geom as afwGeom
# import lsst.afw.image as afwImage
# from lsst.afw.image.testUtils import assertImagesAlmostEqual
import lsst.utils.tests
import lsst.ip.isr.straylight as straylight
import lsst.ip.isr.vignette as vignette
import lsst.ip.isr.masking as masking
import lsst.ip.isr.linearize as linearize

import lsst.ip.isr.isrMock as isrMock


class IsrMiscCases(lsst.utils.tests.TestCase):

    def setUp(self):
        self.inputExp = isrMock.TrimmedRawMock().mock()
        self.mi = self.inputExp.getMaskedImage()

    def tearDown(self):
        pass

    def test_straylight(self):
        task = straylight.StrayLightTask()
        task.run(self.inputExp)
        assert True

    def test_vignette(self):
        config = vignette.VignetteConfig()
        config.radius = 125.0
        config.xCenter = 100.0
        config.yCenter = 100.0

        for doWrite in (False, True):
            with self.subTest(doWrite=doWrite):
                config.doWriteVignettePolygon = doWrite
                task = vignette.VignetteTask(config=config)
                result = task.run(self.inputExp)

                if doWrite:
                    assert result
                else:
                    assert result is None

    def test_masking(self):
        task = masking.MaskingTask()
        result = task.run(self.inputExp)
        assert result is None

    def test_linearize(self):
        for linearityTypeName in ('LookupTable', 'Squared', 'Unknown'):
            with self.subTest(linearityTypeName=linearityTypeName):
                result = linearize.getLinearityTypeByName(linearityTypeName)
                if linearityTypeName == 'Unknown':
                    assert result is None
                else:
                    assert result is not None


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
