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

import lsst.utils.tests
from lsst.ip.isr.measureCrosstalk import (MeasureCrosstalkTask)
import isr_mocks as mock
outputName = None  # specify a name (as a string) to save the output repository


class MeasureCrosstalkTaskCases(lsst.utils.tests.TestCase):

    def testMeasureCrosstalkTask_single(self):
        #        outPath = tempfile.mkdtemp() if outputName is None
        # else "{}-MeasureCrosstalkTask".format(outputName)
        #        dataId = dict(visit=1202, ccd=56)
        #        dataIdStrList = ["%s=%s" % (key, val) for key, val in dataId.items()]
        #        fullResult = MeasureCrosstalkTask.parseAndRun(
        #            args=[inputDir, "--output", outPath, "--clobber-config", "--doraise", "--id"] +
        # dataIdStrList,
        #            doReturnResults=True
        #        )
        exposure = mock.TrimmedRawMock().mock()
        mct = MeasureCrosstalkTask()
        fullResult = mct.run(exposure, dataId=None)
        assert fullResult


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
