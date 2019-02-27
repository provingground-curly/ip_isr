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
import lsst.ip.isr.isrMock as isrMock


class AssembleCcdCases(lsst.utils.tests.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        try:
            del self.config
            del self.task
        except AttributeError:
            pass

    def testAssembleCcdTask_single(self):
        inputExp = isrMock.RawMock().run()
        outputExp = isrMock.TrimmedRawMock().run()
        for doTrim in (True, False):
            with self.subTest(doTrim=doTrim):
                self.config = AssembleCcdConfig(doTrim=doTrim, keysToRemove=['SHEEP', 'MONKEYS', 'ZSHEEP'])
                self.task = AssembleCcdTask(config=self.config)
                assembleOutput = self.task.assembleCcd(inputExp)
                if doTrim is True:
                    self.assertEqual(assembleOutput.getBBox(), outputExp.getBBox())

    def testAssembleCcdTask_dict(self):
        inputExpDict = isrMock.RawDictMock().run()
        outputExp = isrMock.TrimmedRawMock().run()
        for doTrim in (True, False):
            with self.subTest(doTrim=doTrim):
                self.config = AssembleCcdConfig(doTrim=doTrim)
                self.task = AssembleCcdTask(config=self.config)
                assembleOutput = self.task.assembleCcd(inputExpDict)
                assert assembleOutput
                if doTrim is True:
                    self.assertEqual(assembleOutput.getBBox(), outputExp.getBBox())

    def testAssembleCcdTask_fail(self):
        inputExp = None
        with self.assertRaises(TypeError):
            self.config = AssembleCcdConfig(doTrim=False)
            self.task = AssembleCcdTask(config=self.config)
            assembleOutput = self.task.assembleCcd(inputExp)
            assert assembleOutput


class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
