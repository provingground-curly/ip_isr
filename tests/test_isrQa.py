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

# import lsst.meas.algorithms as measAlg

# import lsst.afw.geom as afwGeom
# import lsst.afw.image as afwImage
# from lsst.afw.image.testUtils import assertImagesAlmostEqual
import lsst.utils.tests
import lsst.ip.isr.isrQa as isrQa
# import lsst.pex.exceptions as pexExcept
import isr_mocks as mock


class IsrQaCases(lsst.utils.tests.TestCase):

    def setUp(self):
        self.inputExp = mock.TrimmedRawMock().mock()
        self.mi = self.inputExp.getMaskedImage()
        self.config = isrQa.IsrQaConfig()

    def tearDown(self):
        pass

    def test_makeThumbnail(self):
        thumb = isrQa.makeThumbnail(self.inputExp, self.config)
        assert np.nonzero(thumb)

        self.config.thumbnailSatBorder = 0
        thumb = isrQa.makeThumbnail(self.inputExp, self.config)

        assert np.nonzero(thumb)

    def test_writeThumbnail(self):
        dataRef = mock.DataRefMock()
        thumb = isrQa.makeThumbnail(self.inputExp, self.config)

        isrQa.writeThumbnail(dataRef, thumb, "thumbnail")


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
