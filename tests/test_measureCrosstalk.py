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

# import os
import unittest
# import tempfile
import numpy as np

import lsst.utils.tests
from lsst.ip.isr.measureCrosstalk import (MeasureCrosstalkTask, MeasureCrosstalkConfig)
import lsst.ip.isr.isrMock as isrMock
outputName = None  # specify a name (as a string) to save the output repository


class MeasureCrosstalkTaskCases(lsst.utils.tests.TestCase):

    def testMeasureCrosstalkTask_single(self):
        r"""Measure crosstalk from a sequence of image.
        """
        config = isrMock.IsrMockConfig()
        config.rngSeed = 12345
        config.doCrosstalk = True

        mct = MeasureCrosstalkTask()
        fullResult = []

        for idx in range(0, 20):
            config.rngSeed = 12345 + idx
            exposure = isrMock.CalibratedRawMock(config=config).run()
            exposure.writeFits("../czwMCT.fits")
            result = mct.run(exposure, dataId=None)
            fullResult.append(result)

        coeff, coeffSig, coeffNum = mct.reduce(fullResult)

        # Needed because measureCrosstalk cannot find coefficients equal to 0.0
        coeff = np.nan_to_num(coeff)
        coeffSig = np.nan_to_num(coeffSig)

        expectation = isrMock.CrosstalkCoeffMock().run()
        coeffErr = abs(coeff - expectation) <= coeffSig
        assert np.all(coeffErr)

    def testMeasureCrosstalkTask_trimmed(self):
        r"""Measure crosstalk from a sequence of image.
        """
        config = isrMock.IsrMockConfig()
        config.isTrimmed = False
        config.rngSeed = 12345
        config.doCrosstalk = True

        mct = MeasureCrosstalkTask()
        fullResult = []

        for idx in range(0, 20):
            config.rngSeed = 12345 + idx
            exposure = isrMock.CalibratedRawMock(config=config).run()
            result = mct.run(exposure, dataId=None)
            fullResult.append(result)

        coeff, coeffSig, coeffNum = mct.reduce(fullResult)

        # Needed because measureCrosstalk cannot find coefficients equal to 0.0
        coeff = np.nan_to_num(coeff)
        coeffSig = np.nan_to_num(coeffSig)

        expectation = isrMock.CrosstalkCoeffMock().run()
        coeffErr = abs(coeff - expectation) <= coeffSig
        assert np.all(coeffErr)


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
