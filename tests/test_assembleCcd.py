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

import lsst.utils.tests
from lsst.ip.isr.assembleCcdTask import (AssembleCcdConfig, AssembleCcdTask)
import isr_mocks as mock


class AssembleCcdCases(lsst.utils.tests.TestCase):

    def testAssembleCcdTask_single(self):
        inputExp = mock.RawMock().mock()

        for trim in (True, False):
            with self.subTest(trim=trim):
                self.config = AssembleCcdConfig(doTrim=trim)
                self.task = AssembleCcdTask()
                assembleOutput = self.task.assembleCcd(inputExp)

        assert assembleOutput

    def testAssembleCcdTask_dict(self):
        inputExpDict = mock.RawDictMock().mock()

        for trim in (True, False):
            with self.subTest(trim=trim):
                self.config = AssembleCcdConfig(doTrim=trim)
                self.task = AssembleCcdTask()
                assembleOutput = self.task.assembleCcd(inputExpDict)

        assert assembleOutput


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
