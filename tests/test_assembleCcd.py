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

import os
import unittest
# import tempfile

import lsst.utils.tests
from lsst.ip.isr.assembleCcdTask import (AssembleCcdConfig, AssembleCcdTask)

obsTDSDir = lsst.utils.getPackageDir("testdata_subaru")
inputDir = os.path.join(obsTDSDir, "DATA")

outputName = None  # specify a name (as a string) to save the output repository


class AssembleCcdCases(lsst.utils.tests.TestCase):

    def testAssembleCcdTask_single(self):
        butler = lsst.daf.persistence.Butler(root=inputDir)

        dataId = dict(visit=1202, ccd=56)
        inputExp = butler.get('raw', dataId=dataId)

        self.config = AssembleCcdConfig()
        self.task = AssembleCcdTask()

        assembleOutput = self.task.assembleCcd(inputExp)

        assert assembleOutput

    def testAssembleCcdTask_noTrim(self):
        butler = lsst.daf.persistence.Butler(root=inputDir)

        dataId = dict(visit=1202, ccd=56)
        inputExp = butler.get('raw', dataId=dataId)

        self.config = AssembleCcdConfig(doTrim=False, keysToRemove=['T_OSMN11', 'T_OSMX11'])

        self.task = AssembleCcdTask()

        assembleOutput = self.task.assembleCcd(inputExp)

        assert assembleOutput


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
