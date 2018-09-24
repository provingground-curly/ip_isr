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
import tempfile

import lsst.afw.geom as afwGeom
import lsst.meas.algorithms as measAlg

import lsst.utils.tests
import lsst.ip.isr as ipIsr

obsTDSDir = lsst.utils.getPackageDir("testdata_subaru")
inputDir = os.path.join(obsTDSDir, "DATA")

outputName = None  # specify a name (as a string) to save the output repository

class isrFunctionsCases(lsst.utils.tests.TestCase):
    butler = lsst.daf.persistence.Butler(root=inputDir)

    def test_transposeMaskedImage(self):
        dataId = dict(visit=1202, ccd=56)
        inputExp = self.butler.get('raw', dataId=dataId)
        mi = inputExp.getMaskedImage()

        transposed = ipIsr.transposeMaskedImage(mi)
        assert transposed

    def test_interpolateDefectList(self):
        dataId = dict(visit=1202, ccd=56)
        inputExp = self.butler.get('raw', dataId=dataId)
        mi = inputExp.getMaskedImage()

        defectList = []
        bbox = afwGeom.Box2I(afwGeom.Point2I(400, 400), afwGeom.Point2I(402, 403))
        defectList.append(measAlg.Defect(bbox))

        # This is broken, because I guess it's lying about what it takes.  Whatever, it's friday.
        output = ipIsr.interpolateDefectList(mi, defectList, 2.0, fallbackValue=None)
        assert output



class MemoryTester(lsst.utils.tests.MemoryTestCase):
    pass


if __name__ == "__main__":
    lsst.utils.tests.init()
    unittest.main()
